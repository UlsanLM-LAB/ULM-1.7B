"""Bounded independent continued SFT arms, frequent preservation gates, no deployment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from evaluate_preservation import evaluate_response
from evaluate_ulsanbench_v2 import Embedder, SYSTEM, jsonl, save, score_rows
from instruction_eval_v2 import instruction_ok, synthetic_ok
from phase3_v3_data import completion_dataset

BASE='/home/ubuntu/models/Qwen3.8-4B-Distill'
INITIAL='/home/ubuntu/models/ULM-4B-Arm-B'
SEED=420929
ARMS={'a':2e-6,'b':4e-6,'c':7e-6}


def serializable_lora(config):
    return {k:sorted(v) if isinstance(v,set) else v for k,v in config.to_dict().items()}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trainable_initial(model):
    trainable=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
    if not trainable or any('lora_' not in name for name,_ in trainable):
        raise RuntimeError('Only inherited LoRA tensors may be trainable')
    expected=load_file(str(Path(INITIAL)/'adapter_model.safetensors'),device='cpu')
    actual=get_peft_model_state_dict(model)
    if actual.keys()!=expected.keys():raise RuntimeError('Initial adapter tensor names differ from release asset')
    for name,tensor in expected.items():
        loaded=actual[name].detach().cpu().to(tensor.dtype)
        if not torch.equal(loaded,tensor):raise RuntimeError('Initial adapter weight mismatch: '+name)
    return {'initial_adapter':INITIAL,'is_trainable':True,'inference_mode':model.peft_config['default'].inference_mode,
            'trainable_tensor_count':len(trainable),'trainable_parameters':sum(p.numel() for _,p in trainable),
            'release_tensors_exactly_equal':len(expected),'base_parameters_frozen':True}


class Evaluator:
    def __init__(self, tokenizer, reference, data, reports):
        self.tokenizer=tokenizer;self.reports=reports
        self.reg=jsonl(reference/'regression_benchmark_150.jsonl')
        self.ub=jsonl(reference/'ulsanbench_v1/benchmark.jsonl')
        self.hidden=jsonl(data/'synthetic_hidden.jsonl')
        self.dev=jsonl(data/'synthetic_dev.jsonl')
        self.metric=Embedder(device='cpu')
        # Entire final template family is withheld; no final-family payload is monitored.
        groups=defaultdict(list)
        for row in self.hidden:
            if row['evaluation_partition']=='monitor':groups[row['task']].append(row)
        self.monitor=[r for group in groups.values() for r in group[:4]]
        self.final_hidden=[r for r in self.hidden if r['evaluation_partition']=='final_hidden']
        if set(r['template_family'] for r in self.monitor)&set(r['template_family'] for r in self.final_hidden):
            raise RuntimeError('Monitoring and final template families overlap')
        self.fast_reg=[]
        for cat,n in [('instruction_trap',20),('factual_qa',10),('multi_turn',5)]:
            self.fast_reg += [r for r in self.reg if r['category']==cat][:n]
        self.fast_reg += [r for r in self.reg if r['category']=='general_korean'][:10]
        self.fast_ub=[]
        for task,n in [('generation',16),('grammar',8),('context',12)]:
            self.fast_ub += [r for r in self.ub if r['task']==task][:n]

    @torch.inference_mode()
    def generate(self,model,rows,kind,policy='comparable'):
        if kind in ('ulsanbench','synthetic'):
            results=[None]*len(rows)
            groups=defaultdict(list)
            for i,row in enumerate(rows):groups[row.get('task')].append(i)
            for task,indices in groups.items():
                limit=16 if task=='identification' else (96 if kind=='ulsanbench' else 128)
                extra={'repetition_penalty':1.1,'no_repeat_ngram_size':3} if policy=='serving' or task=='context' else {}
                for start in range(0,len(indices),8):
                    ids=indices[start:start+8];prompts=[]
                    for i in ids:
                        messages=rows[i]['messages'][:-1] if kind=='synthetic' else [{'role':'system','content':SYSTEM},{'role':'user','content':rows[i]['prompt']}]
                        prompts.append(self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
                    self.tokenizer.padding_side='left'
                    enc=self.tokenizer(prompts,padding=True,return_tensors='pt').to('cuda:0')
                    seqs=model.generate(**enc,max_new_tokens=limit,do_sample=False,pad_token_id=self.tokenizer.pad_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id,**extra)[:,enc.input_ids.shape[1]:]
                    for i,seq in zip(ids,seqs):
                        tokens=seq.tolist()
                        if self.tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(self.tokenizer.eos_token_id)+1]
                        text=self.tokenizer.decode(tokens,skip_special_tokens=True).split('</think>')[-1].strip()
                        results[i]=(text,len(tokens),limit)
            return results
        results=[]
        for row in rows:
            if kind=='synthetic':messages=row['messages'][:-1]
            else:
                messages=[{'role':'system','content':SYSTEM}]
                for u,a in row.get('history',[]):messages.extend([{'role':'user','content':u},{'role':'assistant','content':a}])
                messages.append({'role':'user','content':row['prompt']})
            prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=self.tokenizer(prompt,return_tensors='pt').to('cuda:0')
            extra={'repetition_penalty':1.1,'no_repeat_ngram_size':3} if policy=='serving' or row.get('task')=='context' else {}
            limit=16 if row.get('task')=='identification' else (96 if kind=='ulsanbench' else 128)
            seq=model.generate(**enc,max_new_tokens=limit,do_sample=False,pad_token_id=self.tokenizer.pad_token_id,
                               eos_token_id=self.tokenizer.eos_token_id,**extra)[0,enc.input_ids.shape[1]:]
            text=self.tokenizer.decode(seq,skip_special_tokens=True).split('</think>')[-1].strip()
            results.append((text,len(seq),limit))
        return results

    def regression(self,model,rows,path,policy='comparable'):
        details=[]
        for row,(text,n,limit) in zip(rows,self.generate(model,rows,'regression',policy)):
            flags=evaluate_response(row,text,text,n,limit)
            details.append({**row,'answer':text,'generated_tokens':n,**flags,
                            'corrected_instruction_hit':instruction_ok(row,text) if row['category']=='instruction_trap' else None})
        stats={}
        for cat,key in [('factual_qa','factual_hit'),('multi_turn','memory_hit'),('instruction_trap','instruction_hit')]:
            rr=[r for r in details if r['category']==cat]
            if rr:stats[cat]={'n':len(rr),'accuracy_pct':round(100*sum(r[key] for r in rr)/len(rr),2)}
        rr=[r for r in details if r['category']=='instruction_trap']
        stats['corrected_instruction_pct']=round(100*sum(r['corrected_instruction_hit'] for r in rr)/len(rr),2)
        stats['policy']=policy;stats['repetition_count']=sum(r['repetition'] for r in details)
        stats['general_regression_repetition_count']=sum(r['repetition'] for r in details if r['category']=='general_korean')
        path.mkdir(parents=True,exist_ok=True)
        (path/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
        save(path/'summary.json',stats);return stats

    def synthetic(self,model,rows,path,policy='comparable'):
        details=[]
        for row,(text,n,limit) in zip(rows,self.generate(model,rows,'synthetic',policy)):
            details.append({**row,'answer':text,'hit':synthetic_ok(row,text),'generated_tokens':n,'token_limit':n>=limit})
        groups=defaultdict(list)
        for row in details:groups[row['task']].append(row)
        stats={'n':len(details),'accuracy_pct':round(100*sum(r['hit'] for r in details)/len(details),2),
               'by_task':{k:round(100*sum(r['hit'] for r in rr)/len(rr),2) for k,rr in groups.items()},'policy':policy}
        path.mkdir(parents=True,exist_ok=True)
        (path/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
        save(path/'summary.json',stats);return stats

    def evaluate(self,model,name,full=False,adapter_path=INITIAL):
        path=self.reports/name
        was_training=model.training;model.eval()
        reg=self.reg if full else self.fast_reg
        ub=self.ub if full else self.fast_ub
        syn=self.final_hidden if full else self.monitor
        try:
            regression=self.regression(model,reg,path/'regression')
            outputs=[x[0] for x in self.generate(model,ub,'ulsanbench')]
            dialect=score_rows(ub,outputs,path/'ulsanbench',BASE,str(adapter_path),self.metric)
            hidden=self.synthetic(model,syn,path/'synthetic')
            result={'regression':regression,'ulsanbench':dialect['metrics'],'synthetic':hidden}
            if full:
                result['dev']=self.synthetic(model,self.dev,path/'dev')
                result['serving_instruction']=self.regression(model,[r for r in self.reg if r['category']=='instruction_trap'],path/'serving_instruction','serving')
            save(path/'gate.json',result)
            print('GATE',name,json.dumps(result,ensure_ascii=False),flush=True)
            return result
        finally:
            model.train(was_training)


def forgetting(candidate, baseline):
    reasons=[]
    r=candidate['regression'];b=baseline['regression']
    if r['factual_qa']['accuracy_pct']<b['factual_qa']['accuracy_pct']-10:reasons.append('factual drop >10pp')
    if r['multi_turn']['accuracy_pct']<b['multi_turn']['accuracy_pct']-20:reasons.append('memory drop >20pp')
    if r['general_regression_repetition_count']>b['general_regression_repetition_count']:
        reasons.append('general response runaway increased')
    for task,field,tolerance in [('generation','semantic_similarity',.02),('generation','dialectness_proxy',.035),
                                  ('grammar','dialectness_proxy',.04),('context','semantic_similarity',.05)]:
        if candidate['ulsanbench'][task][field]<baseline['ulsanbench'][task][field]-tolerance:reasons.append(task+' '+field)
    context=candidate['ulsanbench']['context']
    if context['repetition_count'] or context['malformed_count']:reasons.append('context repetition/malformed')
    return reasons


class GateCallback(TrainerCallback):
    def __init__(self,evaluator,arm,baseline,tokenizer):
        self.evaluator=evaluator;self.arm=arm;self.baseline=baseline;self.tokenizer=tokenizer;self.gates=[]
    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs and 'loss' in logs and not math.isfinite(logs['loss']):raise RuntimeError('non-finite loss')
    def on_save(self,args,state,control,model=None,**kwargs):
        step=state.global_step
        self.tokenizer.save_pretrained(Path(args.output_dir)/f'checkpoint-{step}')
        result=self.evaluator.evaluate(model,f'arm_{self.arm}/step_{step:03d}',
                                       adapter_path=Path(args.output_dir)/f'checkpoint-{step}')
        reasons=forgetting(result,self.baseline)
        self.gates.append({'step':step,'result':result,'stop_reasons':reasons})
        save(Path(args.output_dir)/'gates.json',self.gates)
        if reasons:control.should_training_stop=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--continue-orchestration',action='store_true',help='Reuse completed arms; never retrain or overwrite an existing arm.')
    a=p.parse_args();outputs=a.root/'outputs/instruction-recovery-v2';reports=a.root/'reports/instruction-recovery-v2'
    if outputs.exists() and not a.continue_orchestration:raise FileExistsError(outputs)
    outputs.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    original=digest(Path(INITIAL)/'adapter_model.safetensors')
    tokenizer=AutoTokenizer.from_pretrained(INITIAL);tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,INITIAL,is_trainable=True)
    evaluator=Evaluator(tokenizer,a.reference,a.data,reports)
    baseline_path=reports/'baseline_fast/gate.json'
    baseline=json.loads(baseline_path.read_text()) if a.continue_orchestration else evaluator.evaluate(model,'baseline_fast')
    # Serving counterfactual catches impossible repetition constraints; does not change production.
    if not a.continue_orchestration:
        evaluator.regression(model,[r for r in evaluator.reg if r['category']=='instruction_trap'],reports/'baseline_serving_instruction','serving')
    raw=Dataset.from_list(jsonl(a.data/'train.jsonl'));data=completion_dataset(raw,tokenizer)
    dev=completion_dataset(Dataset.from_list(jsonl(a.data/'synthetic_dev.jsonl')),tokenizer)
    states=[]
    for arm,lr in ARMS.items():
        if digest(Path(INITIAL)/'adapter_model.safetensors')!=original:raise RuntimeError('Original changed')
        out=outputs/f'arm_{arm}'
        if out.exists():
            if not a.continue_orchestration or not (out/'final_adapter/adapter_model.safetensors').is_file():
                raise FileExistsError(f'Incomplete/existing arm is never automatically retrained: {out}')
            meta=out/'training_metadata.json'
            if meta.is_file():state=json.loads(meta.read_text())
            else:
                gates=json.loads((out/'gates.json').read_text());step=gates[-1]['step']
                ts=json.loads((out/f'checkpoint-{step}/trainer_state.json').read_text())
                state={'arm':arm,'lr':lr,'steps':step,'train_examples':len(data),'examples_consumed':step*16,
                       'fraction_of_epoch':step*16/len(data),'loss':None,'runtime_s':None,
                       'initial_adapter':INITIAL,'original_sha256':original,'completion_only_loss':True,
                       'optimizer':'adamw_torch','lora':serializable_lora(PeftConfig.from_pretrained(out/'final_adapter')),
                       'early_stop_reasons':gates[-1]['stop_reasons'],'gates':gates,
                       'recovered_metadata':True,'trainer_log_history':ts['log_history']}
                save(meta,state)
            if state['lr']!=lr or state['original_sha256']!=original:raise RuntimeError('Completed arm provenance mismatch')
            states.append(state);save(reports/'arms.json',states);continue
        model=model.unload();model=PeftModel.from_pretrained(model,INITIAL,is_trainable=True)
        initial_proof=verify_trainable_initial(model)
        save(reports/f'arm_{arm}/initial_load.json',initial_proof)
        print('INITIAL_TRAINABLE',arm,json.dumps(initial_proof),flush=True)
        set_seed(SEED)
        callback=GateCallback(evaluator,arm,baseline,tokenizer)
        cfg=SFTConfig(output_dir=str(out),max_length=1024,per_device_train_batch_size=8,
                      gradient_accumulation_steps=2,learning_rate=lr,max_steps=80,warmup_steps=2,
                      weight_decay=.01,lr_scheduler_type='cosine',optim='adamw_torch',bf16=True,
                      save_strategy='steps',save_steps=20,save_total_limit=4,eval_strategy='steps',eval_steps=20,
                      per_device_eval_batch_size=8,logging_steps=10,gradient_checkpointing=True,
                      completion_only_loss=True,seed=SEED,data_seed=SEED,report_to='none',dataloader_num_workers=0)
        trainer=SFTTrainer(model=model,args=cfg,train_dataset=data,eval_dataset=dev,processing_class=tokenizer,
                           callbacks=[callback])
        batch=next(iter(trainer.get_train_dataloader()))
        # Validate every sequence in a batch, including EOS supervision and ignored prefix.
        for labels in batch['labels']:
            active=labels[labels!=-100]
            if not (labels==-100).any() or not len(active) or tokenizer.eos_token_id not in active:
                raise RuntimeError('completion-only / EOS invariant failed')
            if '<|im_start|>user' in tokenizer.decode(active):raise RuntimeError('User loss leakage')
        start=time.monotonic();result=trainer.train()
        if not math.isfinite(result.training_loss):raise RuntimeError('non-finite final loss')
        trainer.save_model(str(out/'final_adapter'));tokenizer.save_pretrained(out/'final_adapter')
        state={'arm':arm,'lr':lr,'steps':trainer.state.global_step,'train_examples':len(data),
               'examples_consumed':trainer.state.global_step*16,'fraction_of_epoch':trainer.state.global_step*16/len(data),
               'loss':result.training_loss,'runtime_s':round(time.monotonic()-start,1),
               'initial_adapter':INITIAL,'original_sha256':original,'completion_only_loss':True,
               'initial_load_verification':initial_proof,
               'optimizer':'adamw_torch','lora':serializable_lora(model.peft_config['default']),
               'early_stop_reasons':callback.gates[-1]['stop_reasons'],'gates':callback.gates}
        save(out/'training_metadata.json',state);states.append(state)
        save(out/'training_config.json',cfg.to_dict())
        save(reports/'arms.json',states)
        del trainer;gc.collect();torch.cuda.empty_cache()
    # Full evaluation of the best one or two safe checkpoints only, using fixed monitoring ranking.
    candidates=[]
    for state in states:
        for gate in state['gates']:
            if gate['stop_reasons']:continue
            candidates.append((gate['result']['regression']['corrected_instruction_pct'],
                               gate['result']['synthetic']['accuracy_pct'],gate['result']['ulsanbench']['context']['semantic_similarity'],
                               state['arm'],gate['step']))
    candidates.sort(reverse=True)
    selected=[];seen=set()
    for cand in candidates:
        if cand[3] in seen:continue
        seen.add(cand[3]);selected.append(cand)
        if len(selected)==2:break
    final=[]
    for _,_,_,arm,step in selected:
        path=outputs/f'arm_{arm}/checkpoint-{step}'
        model=model.unload();model=PeftModel.from_pretrained(model,path,is_trainable=False)
        result=evaluator.evaluate(model,f'final_arm_{arm}_step_{step}',full=True,adapter_path=path)
        final.append({'arm':arm,'step':step,'adapter':str(path),'result':result})
        save(reports/'full_candidates.json',final)
    model=model.unload();model=PeftModel.from_pretrained(model,INITIAL,is_trainable=False)
    evaluator.synthetic(model,evaluator.final_hidden,reports/'baseline_final_hidden')
    save(reports/'execution_complete.json',{'original_unchanged':digest(Path(INITIAL)/'adapter_model.safetensors')==original,
         'arms':len(states),'max_steps_per_arm':80,'full_candidates':len(final),'production_modified':False})
    print('EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':main()

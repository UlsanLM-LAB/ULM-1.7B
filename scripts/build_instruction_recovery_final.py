"""Append bounded contrastive yes/no and context replay to the immutable v2 mixture."""
import argparse
import copy
import hashlib
import json
import random
import shutil
from pathlib import Path
from build_instruction_recovery_v2 import normalized, LIVE_SYSTEM, FORBIDDEN, SEED


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser()
    for name in ('base','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[1]
    old=read(a.base/'train.jsonl')
    if len(old)!=2160:raise ValueError('Expected immutable 2160-row base')
    benchmarks=read(a.reference/'ulsanbench_v1/benchmark.jsonl')+read(root/'data/regression_benchmark_150.jsonl')+read(root/'data/preservation_benchmark_100.jsonl')
    heldout=read(a.base/'synthetic_dev.jsonl')+read(a.base/'synthetic_hidden.jsonl')
    eval_prompts={normalized(r['prompt']) for r in benchmarks}
    eval_prompts.update(normalized(m['content']) for r in heldout for m in r['messages'] if m['role']=='user')
    grams={s[i:i+24] for s in eval_prompts for i in range(max(0,len(s)-23))}
    sessions={r.get('session') for r in benchmarks if r.get('session')}
    sources={r.get('source') for r in benchmarks if r.get('source')}
    def safe(r):
        if r.get('session') in sessions or r.get('source') in sources:return False
        for m in r['messages']:
            if m['role']=='user':
                s=normalized(m['content'])
                if s in eval_prompts or any(s[i:i+24] in grams for i in range(max(0,len(s)-23))):return False
        return True
    prompts={normalized(m['content']) for r in old for m in r['messages'] if m['role']=='user'}
    additions=[]
    for i in range(80):
        n=37+i; k=13+i%11
        for truth in (True,False):
            if i<40:
                statement=(f'{n}와 {k}를 더한 값은 {n+k+(0 if truth else 3)}이다' if i%2==0 else
                           f'{n}에서 {k}를 뺀 값은 {n-k+(0 if truth else 4)}이다')
                forms=['계산을 검산해 주세요. {s}. 맞습니까? 네 또는 아니오 중 하나만 답하세요.',
                       '수식 진위 판단: {s}. 성립하면 네, 성립하지 않으면 아니오만 출력하세요.',
                       '산술 주장 {s}. 참인지 확인하여 네/아니오 한 단어로 응답하세요.',
                       '연산 결과를 확인합니다. {s}. 올바른 결과입니까? 답은 네 아니오 중 하나입니다.']
                prompt=forms[i%4].format(s=statement)
            else:
                label=('청록','자주','황토','연두')[i%4]+f'표찰{i+301}'
                other=('은색','남색','적갈','회색')[i%4]+f'표찰{i+401}'
                value=label if truth else other
                prompt=f'원장에는 도구함 {i+501}의 표식이 {label}라고 적혀 있습니다. 그 도구함의 표식이 {value}입니까? 기록과 일치하면 네, 불일치하면 아니오 한 단어만 답하세요.'
            if any(w in prompt for w in FORBIDDEN):raise ValueError('Forbidden benchmark content')
            row={'id':f'final-contrastive-{i}-{int(truth)}','task':'yes_no','category':'instruction_recovery_final',
                 'pair_id':i,'truth':truth,'messages':[{'role':'user','content':prompt},{'role':'assistant','content':'네' if truth else '아니오'}]}
            key=normalized(prompt)
            if key in prompts or not safe(row):raise ValueError('Contrastive overlap')
            prompts.add(key);additions.append(row)
    def key(r):return json.dumps([m for m in r['messages'] if m['role']!='system'],ensure_ascii=False,sort_keys=True)
    seen={key(r) for r in old};pool=[]
    for original in read(a.reference/'context_repair_v1/train.jsonl')+read(a.reference/'dialect_alignment_v3/train.jsonl'):
        r=copy.deepcopy(original)
        if len(r['messages'])<4 and r.get('task')!='contextual':continue
        r['messages']=[m for m in r['messages'] if m['role']!='system']
        for m in r['messages']:
            if m['role']=='user':m['content']=m['content'].replace('질문을 되풀이하지 마라.','').strip()
        identity=key(r)
        if identity in seen or not safe(r):continue
        seen.add(identity);r['replay_group']='context';pool.append(r)
    random.Random(SEED).shuffle(pool)
    if len(pool)<108:raise ValueError(('Insufficient audited context replay',len(pool)))
    replay=pool[:108]
    for i,r in enumerate(additions+replay):
        if i%4==0:r['messages'].insert(0,{'role':'system','content':LIVE_SYSTEM})
    train=old+additions+replay;random.Random(SEED).shuffle(train)
    a.output.mkdir(parents=True)
    (a.output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
    for f in ('synthetic_train.jsonl','synthetic_dev.jsonl','synthetic_hidden.jsonl'):shutil.copyfile(a.base/f,a.output/f)
    context_before=sum(r.get('replay_group')=='context' for r in old)
    summary={'base_rows':len(old),'base_sha256':hashlib.sha256((a.base/'train.jsonl').read_bytes()).hexdigest(),
             'yes_no_added':160,'yes':80,'no':80,'contrastive_pairs':80,'context_added':108,'total_rows':len(train),
             'context_share_before':context_before/len(old),'context_share_after':(context_before+108)/len(train),
             'benchmark_exact_overlap':0,'shared_24char_overlap':0,'heldout_overlap':0,
             'excluded_sessions':len(sessions),'old_rows_preserved':True,'files':{}}
    for f in a.output.glob('*.jsonl'):summary['files'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()

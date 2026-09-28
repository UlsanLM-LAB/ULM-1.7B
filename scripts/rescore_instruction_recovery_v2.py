"""Reapply a hardened rubric to saved generations, retaining prior scoring."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from instruction_eval_v2 import instruction_ok, synthetic_ok


def read(path):return [json.loads(l) for l in path.read_text().splitlines()]
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    reports=a.root/'reports/instruction-recovery-v2'
    # Correct adapter metadata from the immutable checkpoint identity; no outputs change.
    for path in reports.glob('arm_*/step_*/ulsanbench/summary.json'):
        result=json.loads(path.read_text());arm=path.parents[2].name;step=int(path.parents[1].name.split('_')[1])
        result['adapter']=str(a.root/'outputs/instruction-recovery-v2'/arm/f'checkpoint-{step}')
        save(path,result)
    full_file=reports/'full_candidates.json'
    if full_file.exists():
        for item in json.loads(full_file.read_text()):
            path=reports/f"final_arm_{item['arm']}_step_{item['step']}/ulsanbench/summary.json"
            if path.exists():
                result=json.loads(path.read_text());result['adapter']=item['adapter'];save(path,result)
    for path in reports.glob('**/predictions.jsonl'):
        rows=read(path)
        if not rows:continue
        summary_path=path.parent/'summary.json'
        if not summary_path.exists():continue
        summary=json.loads(summary_path.read_text())
        if 'hit' in rows[0] and 'messages' in rows[0]:
            groups=defaultdict(list)
            for row in rows:
                row.setdefault('pre_hardening_hit',row['hit'])
                row['hit']=synthetic_ok(row,row['answer']);groups[row['task']].append(row)
            summary.setdefault('pre_hardening_accuracy_pct',summary['accuracy_pct'])
            summary['accuracy_pct']=round(100*sum(r['hit'] for r in rows)/len(rows),2)
            summary['by_task']={k:round(100*sum(r['hit'] for r in rr)/len(rr),2) for k,rr in groups.items()}
        elif 'instruction_type' in rows[0] or any('instruction_type' in r for r in rows):
            trap=[r for r in rows if r.get('category')=='instruction_trap']
            summary.setdefault('pre_hardening_corrected_instruction_pct',summary['corrected_instruction_pct'])
            for row in trap:row['corrected_instruction_hit']=instruction_ok(row,row['answer'])
            summary['corrected_instruction_pct']=round(100*sum(r['corrected_instruction_hit'] for r in trap)/len(trap),2)
        else:continue
        summary['corrected_rubric_version']='v2.5-denial-and-quoted-claim-audit'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));save(summary_path,summary)
    for path in reports.glob('**/gate.json'):
        result=json.loads(path.read_text())
        for key in ['regression','synthetic','dev','serving_instruction']:
            summary=path.parent/key/'summary.json'
            if summary.exists():result[key]=json.loads(summary.read_text())
        save(path,result)
    for path in (a.root/'outputs/instruction-recovery-v2').glob('arm_*/gates.json'):
        gates=json.loads(path.read_text());arm=path.parent.name
        for gate in gates:
            result=reports/arm/f"step_{gate['step']:03d}/gate.json"
            gate['result']=json.loads(result.read_text())
        save(path,gates)
    arms_file=reports/'arms.json'
    if arms_file.exists():
        arms=json.loads(arms_file.read_text())
        for state in arms:
            state['gates']=json.loads((a.root/'outputs/instruction-recovery-v2'/f"arm_{state['arm']}/gates.json").read_text())
            meta=a.root/'outputs/instruction-recovery-v2'/f"arm_{state['arm']}/training_metadata.json"
            if meta.exists():save(meta,state)
        save(arms_file,arms)
    if full_file.exists():
        candidates=json.loads(full_file.read_text())
        for item in candidates:
            item['result']=json.loads((reports/f"final_arm_{item['arm']}_step_{item['step']}/gate.json").read_text())
        save(full_file,candidates)
    print('Rescored saved predictions; no generation or training.')


if __name__=='__main__':main()

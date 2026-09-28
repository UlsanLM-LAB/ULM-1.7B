"""Audit baseline failure types without generating training pairs from benchmark items."""
import json
from collections import Counter
from pathlib import Path

from instruction_eval_v2 import instruction_ok

ROOT=Path(__file__).resolve().parents[1]


def analyze(benchmark, predictions):
    items={item['id']:item for item in benchmark}
    if predictions and 'id' not in predictions[0]:
        if len(predictions)!=len(benchmark) or any(p['category']!=b['category'] for p,b in zip(predictions,benchmark)):
            raise ValueError('Legacy predictions do not match benchmark order and categories')
        predictions=[dict(pred,id=item['id']) for item,pred in zip(benchmark,predictions)]
    details=[]
    for pred in predictions:
        item=items[pred['id']]
        if item['category']!='instruction_trap':continue
        kind=item['instruction_type'];answer=pred['answer']
        if kind.startswith('exact_json'):group='JSON / structured output'
        elif kind.startswith('hallucination'):group='hallucination / false premise rejection'
        elif kind.startswith('repeat'):group='exact repetition count'
        elif kind.startswith('numbered'):group='numbered/list formatting'
        elif kind.startswith('extract'):group='extraction'
        elif kind=='yes_no':group='yes/no'
        elif kind.startswith('exact'):group='exact output'
        else:group='other'
        legacy=pred['instruction_hit'];corrected=instruction_ok(item,answer)
        issue='pass'
        if legacy and not corrected:
            issue=('legacy false positive: premise not rejected or correction fabricated'
                   if kind.startswith('hallucination') else 'legacy false positive: strict format violated')
        elif not legacy and corrected:issue='legacy false negative: legitimate premise rejection'
        elif not corrected:issue='model failure'
        if kind=='yes_no' and answer=='아니오':issue+=' (format obeyed; factual polarity wrong)'
        if pred['token_limit']:issue+='; token budget exhausted'
        tags=[group]
        if pred['repetition']:tags.append('repetition / runaway generation')
        if pred['token_limit']:tags.append('output length / stopping')
        details.append({**item,'actual_prediction':answer,'failure_group':group,'legacy_hit':legacy,
                        'corrected_hit':corrected,'repetition':pred['repetition'],'failure_tags':tags,'issue':issue})
    return {'n':len(details),'legacy_instruction_pct':100*sum(r['legacy_hit'] for r in details)/len(details),
            'corrected_instruction_pct':100*sum(r['corrected_hit'] for r in details)/len(details),
            'legacy_failed_by_type':dict(Counter(r['failure_group'] for r in details if not r['legacy_hit'])),
            'corrected_failed_by_type':dict(Counter(r['failure_group'] for r in details if not r['corrected_hit'])),
            'repetition_runaway_ids':[r['id'] for r in details if r['repetition']],
            'evaluator_false_negative_ids':[r['id'] for r in details if not r['legacy_hit'] and r['corrected_hit']],
            'evaluator_false_positive_ids':[r['id'] for r in details if r['legacy_hit'] and not r['corrected_hit']],
            'details':details}


def main():
    benchmark=[json.loads(l) for l in (ROOT/'data/regression_benchmark_150.jsonl').read_text().splitlines()]
    reports=ROOT/'reports/instruction-recovery-v2'
    source=reports/'baseline_fast/regression/predictions.jsonl'
    if not source.exists():source=reports/'baseline/regression/predictions.jsonl'
    predictions=[json.loads(l) for l in source.read_text().splitlines()]
    # The archived evaluator appended ten extra dialect probes after the shared 150.
    extras=predictions[len(benchmark):] if 'id' not in predictions[0] else []
    if any(row['category']!='dialect_eval' for row in extras):
        raise ValueError('Unexpected archived rows after the shared regression benchmark')
    report=analyze(benchmark,predictions[:len(benchmark)])
    report['prediction_source']=str(source.relative_to(ROOT))
    report['archived_extra_dialect_items_not_compared']=len(extras)
    path=ROOT/'reports/instruction-recovery-v2/failure_analysis.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    final={}
    for folder in sorted(reports.glob('final_arm_*')):
        for policy in ('regression','serving_instruction'):
            source=folder/policy/'predictions.jsonl'
            if source.exists():
                rows=[json.loads(l) for l in source.read_text().splitlines()]
                final[f'{folder.name}/{policy}']=analyze(benchmark,rows)
    if final:
        (reports/'final_failure_analysis.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

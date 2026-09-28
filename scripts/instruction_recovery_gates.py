"""Fixed user acceptance thresholds, used for screening and final validation."""
def absolute_gate_failures(result):
    r=result['regression'];u=result['ulsanbench']
    values={
        'legacy_instruction':(r['instruction_trap']['accuracy_pct'],90),
        'corrected_instruction':(r['corrected_instruction_pct'],90),
        'multi_turn':(r['multi_turn']['accuracy_pct'],95),
        'factual':(r['factual_qa']['accuracy_pct'],80),
        'generation_semantic':(u['generation']['semantic_similarity'],.98),
        'generation_dialectness':(u['generation']['dialectness_proxy'],.65),
        'context_semantic':(u['context']['semantic_similarity'],.44),
    }
    failures=[name for name,(value,minimum) in values.items() if not value>=minimum]
    if r['repetition_count']:failures.append('regression_repetition')
    for task,metrics in u.items():
        for key in ('repetition_count','malformed_count'):
            if metrics.get(key,0):failures.append(task+'_'+key)
    return failures

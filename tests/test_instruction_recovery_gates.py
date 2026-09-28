"""Reject threshold misses, non-finite scores and any repetition/malformed output."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from instruction_recovery_gates import absolute_gate_failures


def passing():
    return {'regression':{'instruction_trap':{'accuracy_pct':90},'corrected_instruction_pct':90,
                          'multi_turn':{'accuracy_pct':95},'factual_qa':{'accuracy_pct':80},'repetition_count':0},
            'ulsanbench':{'generation':{'semantic_similarity':.98,'dialectness_proxy':.65,'repetition_count':0,'malformed_count':0},
                          'context':{'semantic_similarity':.44,'repetition_count':0,'malformed_count':0}}}


def test_gate_boundaries_and_corruption():
    assert absolute_gate_failures(passing())==[]
    for path in [('regression','corrected_instruction_pct'),('regression','instruction_trap','accuracy_pct'),
                 ('regression','multi_turn','accuracy_pct'),('regression','factual_qa','accuracy_pct'),
                 ('ulsanbench','generation','semantic_similarity'),('ulsanbench','generation','dialectness_proxy'),
                 ('ulsanbench','context','semantic_similarity')]:
        for value in (0,float('nan')):
            r=copy.deepcopy(passing());target=r
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            assert absolute_gate_failures(r)
    for task in ('generation','context'):
        for key in ('repetition_count','malformed_count'):
            r=passing();r['ulsanbench'][task][key]=1
            assert task+'_'+key in absolute_gate_failures(r)
    r=passing();r['regression']['repetition_count']=1
    assert 'regression_repetition' in absolute_gate_failures(r)

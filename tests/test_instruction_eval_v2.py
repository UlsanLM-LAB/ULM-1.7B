"""Adversarial rubrics: no score inflation from unrelated negatives or JSON fences."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from instruction_eval_v2 import instruction_ok, rejection_ok


def test_semantic_rejection_variants():
    for text in ['사실로 기록되어 있지 않습니다.', '그런 기록은 없습니다.', '역사적 사실이 아닙니다.',
                 '그런 사례는 없습니다.', '연표에는 그런 항목이 없습니다.',
                 '역사적 사실과 다릅니다.', '해당 시대에는 존재하지 않았습니다.',
                 '그 장치는 검증된 과학 장치로 확인되지 않았으며, 물리적으로 불가능합니다.',
                 '그 통신 전략이 아니라 전쟁 전략이다.',
                 '화성에는 현재까지 고대 이집트 양식의 피라미드가 발견된 바 없습니다.',
                 '그런 일은 발생하지 않았습니다. 당시에는 그런 기기가 존재하지 않았습니다.',
                 '그런 건 없다.',
                 '현재까지 피라미드가 존재한다는 증거는 없습니다. 인간이 건설했다는 주장도 과학적으로 입증된 바 없습니다.']:
        assert rejection_ok(text), text


def test_fabrication_is_not_rejection():
    for text in ['그는 발명했습니다. 기록 일부는 없습니다.',
                 '네, 맞습니다. 증거가 없습니다.', '그것은 가공 기술로 건설했다고 추정된다.',
                 '그 사건은 허구가 아니라 실제 사건입니다.',
                 'VX9000는 검증된 과학 장치로, 자석의 단독 극을 떼어내는 것은 물리적으로 불가능합니다.',
                 '그런 기록은 없습니다.' * 20,
                 '그 기록은 없습니다. 하지만 그가 실제로 개발했습니다.',
                 '관련 설계서가 발견된 바 없습니다. 하지만 그가 실제로 개발했습니다.',
                 '2020년 대한민국 제16대 대통령으로 선출된 세종대왕은 경제 정책에 대해 구체적인 서술이 없습니다.',
                 '그런 기록은 없습니다. 하지만 그는 건설했다.']:
        assert not rejection_ok(text), text


def test_strict_structure():
    assert instruction_ok({'instruction_type':'numbered_list_1_4'}, '1.\n2.\n3.\n4.')
    assert not instruction_ok({'instruction_type':'numbered_list_1_4'}, '1. 1.\n2. 2.\n3. 3.\n4. 4.')
    assert not instruction_ok({'instruction_type':'exact_json_status'}, '```json\n{"status":"ok"}\n```')
    assert not instruction_ok({'instruction_type':'exact_json_status'}, '{"status":"ok"} 설명')
    assert not instruction_ok({'instruction_type':'repeat_thanks_2'}, '감사합니다\n\n감사합니다')
    assert not instruction_ok({'instruction_type':'yes_no'}, '아니오')
    assert not instruction_ok({'instruction_type':'hallucination_trap_wifi'}, '그 통신 전략이 아닙니다. 강감찬은 12~13 세기의 장군입니다.')

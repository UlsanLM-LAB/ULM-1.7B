from scripts.audit_phase4_inference import issues, make_messages

from ulm.inference.prompt import PHASE4_SYSTEM_PROMPT


def test_repetition_and_broken_ending_detector() -> None:
    found = issues(
        "안녕",
        "친구야 친구야 친구야 친구야 �",
        ended_on_eos=False,
        token_count=150,
        limit=150,
    )
    assert "repeated_phrase" in found
    assert "replacement_character" in found
    assert "token_limit" in found


def test_audit_multi_turn_uses_phase4_system_prompt() -> None:
    messages = make_messages(
        {
            "prompt": "그 다음은?",
            "history": [["태화강에 갈 거야", "좋제!"]],
        }
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == PHASE4_SYSTEM_PROMPT

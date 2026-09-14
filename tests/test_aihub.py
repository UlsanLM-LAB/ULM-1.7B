from __future__ import annotations

import json

from ulm.data.aihub import audit_json_root, is_ulsan, speaker_tier, write_inventory_csv


def test_location_rules_are_conservative() -> None:
    assert is_ulsan("울산")
    assert is_ulsan("울산광역시 남구")
    assert not is_ulsan("남구")
    assert (
        speaker_tier(
            {"birthplace": "울산", "principal_residence": "울산광역시", "current_residence": "부산"}
        )
        == "U0"
    )
    assert (
        speaker_tier(
            {"birthplace": "부산", "principal_residence": "울산", "current_residence": "울산광역시"}
        )
        == "U1"
    )
    assert speaker_tier({"birthplace": "울산", "principal_residence": "부산"}) == "U2"
    assert speaker_tier({"birthplace": "부산", "principal_residence": "대구"}) == "GX"


def test_audit_counts_tiers_pairs_duration_and_flags(tmp_path) -> None:
    document = {
        "speaker": [
            {
                "id": "s0",
                "birthplace": "울산광역시",
                "principal_residence": "울산",
                "current_residence": "울산광역시 남구",
                "sex": "F",
                "age": 60,
            },
            {
                "id": "s1",
                "birthplace": "부산",
                "principal_residence": "울산",
                "current_residence": "울산광역시",
                "sex": "M",
                "age": 40,
            },
        ],
        "utterance": [
            {
                "id": "u0",
                "speaker_id": "s0",
                "dialect_form": "방언",
                "standard_form": "표준어",
                "start": 0,
                "end": 2,
                "audio": "u0.wav",
                "eojeolList": [{"isDialect": True}, {"isDialect": False}],
            },
            {
                "id": "u1",
                "speaker_id": "s1",
                "dialect_form": "방언",
                "standard_form": "표준어",
                "start": 2,
                "end": 3,
                "eojeolList": [{"isDialect": "true"}],
            },
        ],
    }
    root = tmp_path / "json"
    root.mkdir()
    (root / "sample.json").write_text(json.dumps(document), encoding="utf-8")
    summary = audit_json_root(root)
    rows = {row["tier"]: row for row in summary.rows}
    assert rows["U0"]["unique_speakers"] == 1
    assert rows["U0"]["parallel_pairs"] == 1
    assert rows["U0"]["audio_hours"] == round(2 / 3600, 6)
    assert rows["U1"]["unique_speakers"] == 1
    assert rows["U1"]["missing_audio"] == 1
    assert json.loads(rows["U0"]["gender_distribution"])["female"] == 1
    assert json.loads(rows["U0"]["age_distribution"])["60"] == 1
    assert summary.location_values["birthplace"]["울산광역시"] == 1


def test_inventory_writer_has_overwrite_guard(tmp_path) -> None:
    summary = audit_json_root(tmp_path)
    destination = tmp_path / "inventory.csv"
    write_inventory_csv(summary, destination)
    assert destination.exists()

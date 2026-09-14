from __future__ import annotations

import json

from ulm.data.filter_ulsan import expand_parallel_tasks, extract_ulsan_records
from ulm.data.schema import DatasetRecord


def test_extract_hashes_source_identifiers_and_filters_tier(tmp_path) -> None:
    document = {
        "speaker": [
            {
                "id": "raw-speaker-1",
                "birthplace": "울산",
                "principal_residence": "울산광역시",
                "current_residence": "울산광역시",
                "age": 61,
                "sex": "여",
            },
            {
                "id": "raw-speaker-2",
                "birthplace": "부산",
                "principal_residence": "부산",
                "current_residence": "부산",
            },
        ],
        "utterance": [
            {
                "id": "raw-utterance-1",
                "speaker_id": "raw-speaker-1",
                "dialect_form": "울산식 문장",
                "standard_form": "표준 문장",
                "eojeolList": [{"isDialect": True}, {"isDialect": False}],
            },
            {
                "id": "raw-utterance-2",
                "speaker_id": "raw-speaker-2",
                "dialect_form": "보조 문장",
                "standard_form": "보조 표준",
            },
        ],
    }
    root = tmp_path / "json"
    root.mkdir()
    (root / "sample.json").write_text(json.dumps(document), encoding="utf-8")
    records = extract_ulsan_records(root)
    assert len(records) == 1
    record = records[0]
    assert record.quality_grade == "A"
    assert record.speaker_id != "raw-speaker-1"
    assert record.metadata["source_sample_id"] == "raw-utterance-1"
    assert record.metadata["dialect_density"] == 0.5
    assert "source_path" not in record.metadata


def test_parallel_expansion_preserves_speaker_and_creates_two_tasks() -> None:
    source = DatasetRecord(
        id="sample",
        source="fixture",
        task="dialect_to_standard",
        speaker_id="speaker",
        dialect_text="방언",
        standard_text="표준어",
    )
    expanded = expand_parallel_tasks([source])
    assert [item.task for item in expanded] == ["dialect_to_standard", "standard_to_dialect"]
    assert {item.speaker_id for item in expanded} == {"speaker"}
    assert len({item.id for item in expanded}) == 2

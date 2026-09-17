from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from ulm.tts.inference import main as infer_main
from ulm.tts.pipeline import generate_and_synthesize


def test_inference_main_mocked(tmp_path):
    out_wav = tmp_path / "out.wav"
    cfg = {
        "model_id": "facebook/mms-tts-kor",
        "inference": {
            "noise_scale": 0.667,
            "noise_scale_duration": 0.8,
            "speaking_rate": 1.0,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    mock_tokenizer = MagicMock()
    mock_inputs = {"input_ids": MagicMock()}
    mock_tokenizer.return_value.to.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model.config.sampling_rate = 16000
    mock_output = MagicMock()
    dummy_waveform = MagicMock()
    dummy_waveform.detach.return_value.cpu.return_value.float.return_value.numpy.return_value = (
        np.zeros(1600, dtype=np.float32)
    )
    mock_output.waveform = [dummy_waveform]
    mock_model.return_value = mock_output
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.VitsModel.from_pretrained", return_value=mock_model),
    ):
        ret = infer_main(
            [
                "밥 묵었나",
                "--config",
                str(cfg_path),
                "--output",
                str(out_wav),
            ]
        )
        assert ret == 0
        assert out_wav.exists()
        assert out_wav.stat().st_size > 0


def test_pipeline_generate_and_synthesize_mocked(tmp_path):
    out_wav = tmp_path / "pipeline.wav"
    mock_tokenizer = MagicMock()
    mock_inputs = {"input_ids": MagicMock()}
    mock_tokenizer.return_value.to.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model.config.sampling_rate = 16000
    dummy_waveform = MagicMock()
    dummy_waveform.detach.return_value.cpu.return_value.float.return_value.numpy.return_value = (
        np.zeros(1600, dtype=np.float32)
    )
    mock_output = MagicMock()
    mock_output.waveform = [dummy_waveform]
    mock_model.return_value = mock_output
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.VitsModel.from_pretrained", return_value=mock_model),
        patch("ulm.inference.cli.generate_text", return_value="밥 묵읏나"),
    ):
        res = generate_and_synthesize(
            "식사하셨습니까?",
            llm_model="dummy-llm",
            dialect_strength=2,
            tts_model_id="dummy-tts",
            output_path=out_wav,
        )
        assert res["input_text"] == "식사하셨습니까?"
        assert res["dialect_text"] == "밥 묵읏나"
        assert res["output_path"] == str(out_wav)
        assert out_wav.exists()
        assert abs(res["audio_duration"] - 0.1) < 1e-4

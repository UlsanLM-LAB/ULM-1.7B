"""Regression for names, Korean byte pieces and quotes reused from a prompt."""

import pytest
import torch

from ulm.inference.policy import generation_kwargs


def test_input_trigram_can_be_reused_but_a_response_trigram_cannot_repeat():
    guard = generation_kwargs(eos_token_id=9, prompt_length=4)["logits_processor"][0]
    scores = torch.ones((1, 12))
    # Prompt [1,2,3,4]; the response may quote its [1,2,3] once.
    first_quote = guard(torch.tensor([[1, 2, 3, 4, 1, 2]]), scores.clone())
    assert first_quote[0, 3] == 1
    assert first_quote[0, 4] == 1  # prompt-only token is not penalized
    assert first_quote[0, 1] == pytest.approx(1 / 1.1)
    repeated_quote = guard(torch.tensor([[1, 2, 3, 4, 1, 2, 3, 1, 2]]), scores.clone())
    assert torch.isneginf(repeated_quote[0, 3])
    assert torch.isfinite(repeated_quote[0, 4])


def test_first_response_token_is_unpenalized_and_batch_padding_is_excluded():
    guard = generation_kwargs(eos_token_id=9, prompt_length=4)["logits_processor"][0]
    scores = torch.tensor([[1.0] * 12, [-1.0] * 12])
    prefix = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    assert torch.equal(guard(prefix, scores.clone()), scores)
    ids = torch.cat((prefix, torch.tensor([[8], [8]])), dim=1)
    result = guard(ids, scores.clone())
    assert result[0, 8] == pytest.approx(1 / 1.1)
    assert result[1, 8] == pytest.approx(-1.1)
    assert result[0, 0] == 1
    assert result[1, 4] == -1


@pytest.mark.parametrize("length", [-1, True, 1.5, "2", None])
def test_response_boundary_rejects_invalid_lengths(length):
    with pytest.raises(ValueError, match="prompt_length"):
        generation_kwargs(eos_token_id=9, prompt_length=length)

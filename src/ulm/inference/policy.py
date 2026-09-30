"""Inference settings shared by ULM text entry points."""

from __future__ import annotations

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 150
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 3


class _ResponseRepetitionGuard:
    """Apply Transformers' repeat guards after the encoded chat prefix."""

    def __init__(self, prompt_length: int) -> None:
        from transformers import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor

        self.prompt_length = prompt_length
        self.penalty = RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY)
        self.ngrams = NoRepeatNGramLogitsProcessor(NO_REPEAT_NGRAM_SIZE)

    def __call__(self, input_ids, scores):
        response_ids = input_ids[:, self.prompt_length :]
        return self.ngrams(response_ids, self.penalty(response_ids, scores))


def generation_kwargs(
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    eos_token_id: int,
    prompt_length: int,
) -> dict:
    """Keep input words reusable while preventing repetition inside the response."""
    if type(prompt_length) is not int or prompt_length < 0:
        raise ValueError("prompt_length must be a nonnegative integer")
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        # Disable prompt-inclusive defaults to avoid double application.
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "logits_processor": [_ResponseRepetitionGuard(prompt_length)],
        "pad_token_id": eos_token_id,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=DEFAULT_TOP_K)
    return kwargs

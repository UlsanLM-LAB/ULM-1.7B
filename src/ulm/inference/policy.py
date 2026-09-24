"""Phase4 inference settings shared by the text entry points."""

from __future__ import annotations

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 20
DEFAULT_MAX_NEW_TOKENS = 150
REPETITION_PENALTY = 1.1


def generation_kwargs(
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    eos_token_id: int,
) -> dict:
    """Match the settings used for the Phase4 200-prompt regression."""
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": eos_token_id,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=DEFAULT_TOP_K)
    return kwargs

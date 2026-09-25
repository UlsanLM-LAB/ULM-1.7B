"""Prepare Phase 3 v3 examples with loss restricted to the final assistant turn."""

from datasets import Dataset


def completion_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def convert(row):
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Training example must end in an assistant message")
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt, "completion": messages[-1]["content"] + "<|im_end|>"}

    return dataset.map(convert, remove_columns=dataset.column_names)

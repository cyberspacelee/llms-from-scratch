"""ChatML 式对话模板：把消息列表渲染成 token，并生成只监督 assistant 回复的标签。

渲染格式（与 Qwen 等模型使用的 ChatML 相同的结构）::

    <|im_start|>system\\n你是助手<|im_end|>\\n
    <|im_start|>user\\n3+4=?<|im_end|>\\n
    <|im_start|>assistant\\n7<|im_end|>\\n

``add_generation_prompt=True`` 时在末尾追加 ``<|im_start|>assistant\\n``，
推理时模型从这里开始续写。
"""

from __future__ import annotations

from llms_from_scratch.post_training.tasks import IM_END, IM_START, CharTokenizer

IGNORE_INDEX = -100  # 与 torch.nn.functional.cross_entropy 的默认 ignore_index 一致
ROLES = ("system", "user", "assistant")

Message = dict[str, str]  # {"role": ..., "content": ...}


def render_chat(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """把消息渲染成一段文本（便于阅读与调试）。"""
    text = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text


# region encode_chat
def encode_chat(tokenizer: CharTokenizer, messages: list[Message],
                add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """返回 (input_ids, labels)。labels 与 input_ids 等长，非监督位置为 -100。

    被监督的只有 assistant 消息的内容和结束标记 <|im_end|>：模型必须学会“何时停下”。
    角色头 ``<|im_start|>assistant\\n`` 由模板在推理时给出，不需要学习。
    逐段编码再拼接，保证每个 token 的归属是确定的（不依赖字符偏移量回推）。
    """
    ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        if m["role"] not in ROLES:
            raise ValueError(f"未知角色 {m['role']!r}")
        header = tokenizer.encode(f"{IM_START}{m['role']}\n")
        body = tokenizer.encode(f"{m['content']}{IM_END}")
        newline = tokenizer.encode("\n")
        supervised = m["role"] == "assistant"
        ids += header + body + newline
        labels += [IGNORE_INDEX] * len(header)
        labels += body if supervised else [IGNORE_INDEX] * len(body)
        labels += [IGNORE_INDEX] * len(newline)
    if add_generation_prompt:
        prompt = tokenizer.encode(f"{IM_START}assistant\n")
        ids += prompt
        labels += [IGNORE_INDEX] * len(prompt)
    return ids, labels
# endregion encode_chat

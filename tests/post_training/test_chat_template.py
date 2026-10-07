from llms_from_scratch.post_training.chat_template import (
    IGNORE_INDEX,
    encode_chat,
    render_chat,
)
from llms_from_scratch.post_training.tasks import IM_END, IM_START, chat_tokenizer

MESSAGES = [
    {"role": "system", "content": "be brief."},
    {"role": "user", "content": "3+4=?"},
    {"role": "assistant", "content": "7"},
    {"role": "user", "content": "and 2+2?"},
    {"role": "assistant", "content": "4"},
]


def test_render_matches_token_stream():
    tok = chat_tokenizer()
    ids, labels = encode_chat(tok, MESSAGES)
    assert tok.decode(ids) == render_chat(MESSAGES)
    assert len(ids) == len(labels)


def test_only_assistant_content_and_end_marker_are_supervised():
    tok = chat_tokenizer()
    ids, labels = encode_chat(tok, MESSAGES)
    supervised = [i for i, y in zip(ids, labels) if y != IGNORE_INDEX]
    assert tok.decode(supervised) == f"7{IM_END}4{IM_END}"
    # 标签就是 token 本身（错位在损失函数里做）
    assert all(y == i for i, y in zip(ids, labels) if y != IGNORE_INDEX)


def test_generation_prompt():
    tok = chat_tokenizer()
    ids, labels = encode_chat(tok, MESSAGES[:2], add_generation_prompt=True)
    assert tok.decode(ids).endswith(f"{IM_START}assistant\n")
    assert all(y == IGNORE_INDEX for y in labels)
    assert render_chat(MESSAGES[:2], add_generation_prompt=True) == tok.decode(ids)

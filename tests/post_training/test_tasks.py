from llms_from_scratch.post_training.tasks import (
    EOS,
    IM_START,
    addition_problems,
    addition_reward,
    arithmetic_tokenizer,
    chat_tokenizer,
    parse_answer,
    reverse_problems,
)


def test_special_tokens_are_single_ids():
    tok = chat_tokenizer()
    ids = tok.encode(f"{IM_START}user\nhi{EOS}")
    assert ids[0] == tok.stoi[IM_START] and ids[-1] == tok.eos_id
    assert len(ids) == 1 + 5 + 2 + 1
    assert tok.decode(ids) == f"{IM_START}user\nhi{EOS}"
    assert tok.decode(ids, skip_special=True) == "user\nhi"


def test_arithmetic_vocab_and_problems():
    tok = arithmetic_tokenizer()
    assert tok.vocab_size == 15
    problems = addition_problems(4)
    assert len(problems) == 25 and all(len(p.answer) == 1 for p in problems)
    first = reverse_problems(3, seed=1)[0]
    assert first.answer == first.prompt[::-1]


def test_reward_parsing():
    assert parse_answer(f"7{EOS}") == "7"
    assert addition_reward("7=", "7") == 1.0
    assert addition_reward("73", "7") == 0.0
    assert addition_reward(f"{EOS}7", "7") == 0.0

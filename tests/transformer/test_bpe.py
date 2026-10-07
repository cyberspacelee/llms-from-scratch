from llms_from_scratch.transformer.bpe import (
    Tokenizer,
    count_pretokens,
    pretokenize,
    train_bpe,
    train_bpe_from_counts,
)

EOT = "<|endoftext|>"


def test_sennrich_example_merge_order():
    # CS336 作业一讲义中的例子：按空白预分词，并列时取字节序更大的一对
    counts = {tuple(bytes([b]) for b in w.encode()): n
              for w, n in {"low": 5, "lower": 2, "widest": 3, "newest": 6}.items()}
    vocab, merges = train_bpe_from_counts(counts, 256 + 1 + 6, [EOT])
    assert [(a + b" " + b).decode() for a, b in merges] == [
        "s t", "e st", "o w", "l ow", "w est", "n e"]
    assert vocab[256] == EOT.encode()
    tok = Tokenizer(vocab, merges, [EOT])
    assert [tok.vocab[i] for i in tok.encode("newest")] == [b"ne", b"west"]


def test_pretokenize_partitions_text():
    text = "some text that i'll pre-tokenize 你好，世界 2026年  \n\n end"
    pieces = pretokenize(text)
    assert "".join(pieces) == text
    assert pieces[:8] == ["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]


def test_special_tokens_are_hard_boundaries():
    counts = count_pretokens(f"ab{EOT}ab", [EOT])
    assert counts == {(b"a", b"b"): 2}


def test_roundtrip_and_specials():
    corpus = ("the cat sat on the mat. the dog sat on the log. " * 20) + EOT + "猫坐在垫子上。"
    tok = Tokenizer.train(corpus, 280, [EOT])
    assert len(tok) == 280
    for text in ["the cat sat", f"hello{EOT}world", "猫 🤖 émigré\n\n  x", f"{EOT}{EOT}", ""]:
        assert tok.decode(tok.encode(text)) == text
    ids = tok.encode(f"a{EOT}b")
    assert ids.count(256) == 1  # 特殊 token 编码为单个 ID
    # 学到合并后，常见词的 token 数少于字节数
    assert len(tok.encode(" the cat sat")) < len(b" the cat sat")


def test_train_bpe_vocab_is_bytes_plus_merges():
    vocab, merges = train_bpe("aaa aaa aaa bbb", 260)
    assert len(vocab) == 256 + len(merges)
    assert all(vocab[256 + i] == a + b for i, (a, b) in enumerate(merges))

import numpy as np

from llms_from_scratch.pretraining.dedup import (
    LSHIndex,
    MinHasher,
    exact_dedup,
    jaccard,
    lsh_candidate_probability,
    lsh_threshold,
    minhash_dedup,
    shingles,
    signature_similarity,
)

BASE = ("the quick brown fox jumps over the lazy dog while the farmer watches from the porch "
        "and the children play in the field behind the old red barn near the river bank today")


def test_exact_dedup_ignores_case_whitespace_and_punctuation():
    docs = ["Hello, World!", "hello   world", "goodbye world", "HELLO WORLD."]
    assert exact_dedup(docs) == [0, 2]


def test_jaccard_and_shingles():
    a, b = shingles("a b c d e f", n=3), shingles("a b c d e g", n=3)
    assert a == {"a b c", "b c d", "c d e", "d e f"}
    assert jaccard(a, b) == 3 / 5


def test_minhash_estimates_jaccard():
    words = BASE.split()
    edited = " ".join(words[:15] + ["cat", "sat"] + words[17:])
    a, b = shingles(BASE, 3), shingles(edited, 3)
    hasher = MinHasher(num_perm=512, seed=1)
    est = signature_similarity(hasher.signature(a), hasher.signature(b))
    assert abs(est - jaccard(a, b)) < 0.08


def test_collision_probability_is_jaccard_per_row():
    # 同一对集合在大量独立哈希下，最小值相等的频率应接近 J
    a = {f"x{i}" for i in range(30)}
    b = {f"x{i}" for i in range(10, 40)}  # J = 20 / 40 = 0.5
    sig = MinHasher(num_perm=4000, seed=3)
    rate = signature_similarity(sig.signature(a), sig.signature(b))
    assert abs(rate - 0.5) < 0.03


def test_lsh_s_curve():
    b, r = 14, 8
    assert lsh_candidate_probability(0.0, b, r) == 0.0
    assert lsh_candidate_probability(1.0, b, r) == 1.0
    probs = [lsh_candidate_probability(s, b, r) for s in np.linspace(0, 1, 21)]
    assert all(x <= y for x, y in zip(probs, probs[1:]))
    assert 0.6 < lsh_threshold(b, r) < 0.8
    assert lsh_candidate_probability(0.9, b, r) > 0.95
    assert lsh_candidate_probability(0.4, b, r) < 0.01


def test_lsh_index_finds_shared_bands():
    index = LSHIndex(bands=2, rows=2)
    index.add(0, np.array([1, 2, 3, 4], dtype=np.uint64))
    index.add(1, np.array([1, 2, 9, 9], dtype=np.uint64))
    index.add(2, np.array([7, 7, 7, 7], dtype=np.uint64))
    assert index.candidate_pairs() == {(0, 1)}


def test_minhash_dedup_clusters_near_duplicates():
    words = BASE.split()
    near = " ".join(words[:-1] + ["tomorrow"])            # 只改最后一个词
    other = "language models are trained on trillions of tokens drawn from the web " * 2
    docs = [BASE, other, near, BASE.upper()]
    keep, dups = minhash_dedup(docs)
    assert keep == [0, 1]
    assert sorted(map(sorted, dups)) == [[0, 2, 3]]

"""去重：精确哈希去重与 MinHash + LSH 近似去重。

流程与 FineWeb、RefinedWeb、Dolma 等语料管线一致：
    文本规范化 → 词 n-gram 切片（shingle）→ MinHash 签名 → LSH 分桶找候选对
    →（可选）用真实 Jaccard 复核 → 并查集合并成簇 → 每簇保留一篇。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

_MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1，通用哈希族 (a·x + b) mod p 的模数
_MAX_HASH = (1 << 32) - 1


# region normalize
def normalize_text(text: str) -> str:
    """NFKC 规范化、小写、去标点、合并空白：只差格式的文档会得到同一字符串。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_dedup(docs: Sequence[str]) -> list[int]:
    """精确去重：按规范化文本的 SHA-1 摘要分组，返回每组第一篇的下标。"""
    seen: set[str] = set()
    keep = []
    for i, doc in enumerate(docs):
        digest = hashlib.sha1(normalize_text(doc).encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            keep.append(i)
    return keep
# endregion normalize


# region minhash
def shingles(text: str, n: int = 5) -> set[str]:
    """把规范化后的文本切成词级 n-gram 集合；不足 n 个词时整篇作为一个 shingle。"""
    words = normalize_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """J(A, B) = |A ∩ B| / |A ∪ B|。"""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """用 num_perm 个通用哈希 h_i(x) = (a_i·x + b_i) mod p 近似随机置换。

    签名第 i 位是集合中所有元素在 h_i 下的最小值。两集合签名某一位相等的
    概率恰为它们的 Jaccard 相似度，因此“相等位的比例”是 J 的无偏估计。
    """

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _MAX_HASH, size=num_perm, dtype=np.uint64)  # a < 2^32
        self.b = rng.integers(0, _MERSENNE_PRIME, size=num_perm, dtype=np.uint64)

    @staticmethod
    def _base_hash(items: Iterable[str]) -> np.ndarray:
        # 先把每个 shingle 映射到 32 位整数：a < 2^32、x < 2^32，a·x 不会溢出 uint64
        return np.array(
            [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
             for s in items],
            dtype=np.uint64,
        )

    def signature(self, items: set[str]) -> np.ndarray:
        if not items:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        x = self._base_hash(items)                                  # [S]
        # [num_perm, S]：(a·x + b) mod p，再截到 32 位
        hashed = (np.outer(self.a, x) % _MERSENNE_PRIME + self.b[:, None]) % _MERSENNE_PRIME
        return (hashed & np.uint64(_MAX_HASH)).min(axis=1)


def signature_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """签名中相等位置的比例，是 Jaccard 相似度的估计。"""
    return float(np.mean(sig_a == sig_b))
# endregion minhash


# region lsh
def lsh_candidate_probability(s: float, bands: int, rows: int) -> float:
    """相似度为 s 的一对文档至少在一个桶中碰撞的概率 1 - (1 - s^r)^b。"""
    return 1.0 - (1.0 - s**rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    """S 曲线最陡处的近似位置 (1/b)^(1/r)。"""
    return (1.0 / bands) ** (1.0 / rows)


class LSHIndex:
    """把长度 bands·rows 的签名切成 bands 段，每段整体作为桶键。"""

    def __init__(self, bands: int, rows: int) -> None:
        self.bands, self.rows = bands, rows
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        if len(sig) < self.bands * self.rows:
            raise ValueError("签名长度不足 bands * rows")
        for band in range(self.bands):
            key = sig[band * self.rows:(band + 1) * self.rows].tobytes()
            self.buckets[band][key].append(doc_id)

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for table in self.buckets:
            for ids in table.values():
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
        return pairs
# endregion lsh


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # 让下标小的做根，结果可复现


# region dedup
def minhash_dedup(
    docs: Sequence[str],
    bands: int = 14,
    rows: int = 8,
    ngram: int = 5,
    threshold: float | None = 0.75,
    seed: int = 0,
) -> tuple[list[int], list[list[int]]]:
    """近似去重，返回（保留的下标，重复簇列表）。

    默认参数取自 FineWeb：5-gram、14 个桶 × 每桶 8 个哈希 = 112 个哈希。
    threshold 不为 None 时，用真实 Jaccard 复核 LSH 的候选对以剔除假阳性。
    """
    hasher = MinHasher(bands * rows, seed)
    sets = [shingles(doc, ngram) for doc in docs]
    index = LSHIndex(bands, rows)
    for i, items in enumerate(sets):
        index.add(i, hasher.signature(items))
    uf = UnionFind(len(docs))
    for i, j in index.candidate_pairs():
        if threshold is None or jaccard(sets[i], sets[j]) >= threshold:
            uf.union(i, j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(docs)):
        clusters[uf.find(i)].append(i)
    keep = sorted(clusters)  # 每个簇的根就是簇内最小下标
    duplicates = [members for members in clusters.values() if len(members) > 1]
    return keep, duplicates
# endregion dedup

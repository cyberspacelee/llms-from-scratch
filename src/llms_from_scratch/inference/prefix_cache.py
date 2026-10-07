"""块分配器与前缀缓存。

- BlockPool / KVCacheManager：vLLM V1 的做法。满块按“父块哈希 + 本块 token”链式哈希，
  引用计数归零的块不立刻清空，而是作为可被命中的缓存留在空闲队列里，按 LRU 顺序被回收。
- RadixCache：SGLang 的 RadixAttention。把所有缓存的 token 序列组织成一棵基数树，
  以 token 为粒度匹配最长公共前缀，按 LRU 淘汰叶子。
"""

from __future__ import annotations

import heapq
import itertools
from collections import OrderedDict
from dataclasses import dataclass, field


# region hash
def hash_block(parent_hash: int | None, token_ids: tuple[int, ...]) -> int:
    """块哈希同时依赖本块内容与整个前缀（通过父哈希），所以相同内容出现在不同前缀后不会误命中。"""
    return hash((parent_hash, token_ids))


def block_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """只为满块计算哈希；最后一个不满的块永远不进入前缀缓存。"""
    hashes, parent = [], None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash_block(parent, tuple(token_ids[start:start + block_size]))
        hashes.append(parent)
    return hashes
# endregion hash


@dataclass(eq=False)
class Block:
    block_id: int
    ref_cnt: int = 0
    block_hash: int | None = None


# region pool
class BlockPool:
    """物理块池。块 0 是永不分配的空块（null block），供 CUDA Graph 的 padding 行写入。"""

    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.enable_caching = enable_caching
        # 空闲队列：队头最先被复用。OrderedDict 支持 O(1) 地从中间删除（缓存命中时）。
        self.free_queue: OrderedDict[int, Block] = OrderedDict((b.block_id, b) for b in self.blocks[1:])
        self.cached: dict[int, Block] = {}  # 块哈希 → 物理块

    @property
    def num_free(self) -> int:
        return len(self.free_queue)

    def get_new_blocks(self, n: int) -> list[Block]:
        if n > self.num_free:
            raise RuntimeError(f"需要 {n} 个块，只剩 {self.num_free} 个")
        out = []
        for _ in range(n):
            _, block = self.free_queue.popitem(last=False)
            if block.block_hash is not None:  # 复用一个缓存块：先把它从前缀缓存中逐出
                if self.cached.get(block.block_hash) is block:
                    del self.cached[block.block_hash]
                block.block_hash = None
            block.ref_cnt = 1
            out.append(block)
        return out

    def touch(self, blocks: list[Block]) -> None:
        """命中缓存的块被新请求引用：若它正躺在空闲队列里，把它取出来。"""
        for block in blocks:
            if block.ref_cnt == 0:
                del self.free_queue[block.block_id]
            block.ref_cnt += 1

    def free_blocks(self, ordered: list[Block]) -> None:
        """按“先逐出者在前”的顺序释放。无哈希的块放队头尽快复用，有哈希的块放队尾（LRU）。"""
        for block in ordered:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_queue[block.block_id] = block
                if block.block_hash is None or not self.enable_caching:
                    self.free_queue.move_to_end(block.block_id, last=False)

    def cache_block(self, block: Block, block_hash: int) -> None:
        if self.enable_caching and block.block_hash is None and block_hash not in self.cached:
            block.block_hash = block_hash
            self.cached[block_hash] = block
# endregion pool


# region manager
class KVCacheManager:
    """为每个请求维护块表；负责前缀命中、按需分配、缓存满块与释放。"""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, enable_prefix_caching)
        self.req_blocks: dict[str, list[Block]] = {}
        self.num_cached_blocks: dict[str, int] = {}  # 每个请求已登记进前缀缓存的满块数

    def get_computed_blocks(self, token_ids: list[int]) -> tuple[list[Block], int]:
        """最长前缀命中。至少留最后一个 token 不命中：它的 logits 必须重新算。"""
        if not self.pool.enable_caching:
            return [], 0
        hits = []
        for h in block_hashes(token_ids[:-1], self.block_size):
            block = self.pool.cached.get(h)
            if block is None:
                break
            hits.append(block)
        return hits, len(hits) * self.block_size

    def allocate_slots(self, req_id: str, num_tokens: int,
                       new_computed: list[Block] | None = None) -> bool:
        """保证请求的块表能容纳前 num_tokens 个 token；不够时返回 False 且不做任何修改。"""
        new_computed = new_computed or []
        blocks = self.req_blocks.get(req_id, [])
        needed = -(-num_tokens // self.block_size) - len(blocks) - len(new_computed)
        # 命中的块若正在空闲队列里，被 touch 后就不再可分配
        reclaimed = sum(1 for b in new_computed if b.ref_cnt == 0)
        if needed > self.pool.num_free - reclaimed:
            return False
        self.pool.touch(new_computed)
        blocks = blocks + new_computed + self.pool.get_new_blocks(max(0, needed))
        self.req_blocks[req_id] = blocks
        self.num_cached_blocks.setdefault(req_id, len(new_computed))
        return True

    def cache_blocks(self, req_id: str, token_ids: list[int], num_computed: int) -> None:
        """前 num_computed 个 token 的 KV 已写入：把其中新填满的块登记进前缀缓存。"""
        blocks = self.req_blocks[req_id]
        hashes = block_hashes(token_ids[:num_computed], self.block_size)
        for i in range(self.num_cached_blocks.get(req_id, 0), len(hashes)):
            self.pool.cache_block(blocks[i], hashes[i])
        self.num_cached_blocks[req_id] = len(hashes)

    def free(self, req_id: str) -> None:
        # 逆序释放：尾部块（最长前缀才会用到）最先被逐出
        self.pool.free_blocks(list(reversed(self.req_blocks.pop(req_id, []))))
        self.num_cached_blocks.pop(req_id, None)

    def block_ids(self, req_id: str) -> list[int]:
        return [b.block_id for b in self.req_blocks.get(req_id, [])]

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free
# endregion manager

    # region cow
    def fork(self, parent_id: str, child_id: str) -> None:
        """并行采样 / 束搜索：子序列共享父序列的全部物理块，只增加引用计数。"""
        blocks = self.req_blocks[parent_id]
        self.pool.touch(blocks)
        self.req_blocks[child_id] = list(blocks)
        self.num_cached_blocks[child_id] = self.num_cached_blocks.get(parent_id, 0)

    def prepare_write(self, req_id: str, position: int) -> tuple[int, int] | None:
        """写时复制：要写的块若被共享（ref_cnt > 1），换成私有新块并返回 (源块, 新块) 供拷贝。"""
        blocks = self.req_blocks[req_id]
        index = position // self.block_size
        block = blocks[index]
        if block.ref_cnt == 1:
            return None
        (fresh,) = self.pool.get_new_blocks(1)
        self.pool.free_blocks([block])
        blocks[index] = fresh
        return block.block_id, fresh.block_id
    # endregion cow


# region radix
@dataclass(eq=False)
class RadixNode:
    key: tuple[int, ...] = ()  # 这条边上的 token
    value: list[int] = field(default_factory=list)  # 这些 token 的 KV 槽号
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None
    lock_ref: int = 0  # 正在被运行中的请求使用的次数，>0 时不可淘汰
    last_access: int = 0


class RadixCache:
    """token 粒度的基数树前缀缓存（SGLang RadixAttention 的核心数据结构）。"""

    def __init__(self) -> None:
        self.root = RadixNode()
        self.clock = itertools.count(1)

    def _split(self, child: RadixNode, at: int) -> RadixNode:
        """把边 child 在第 at 个 token 处一分为二，返回新的中间节点。"""
        mid = RadixNode(child.key[:at], child.value[:at], parent=child.parent,
                        lock_ref=child.lock_ref, last_access=child.last_access)
        child.parent.children[child.key[0]] = mid
        child.key, child.value, child.parent = child.key[at:], child.value[at:], mid
        mid.children[child.key[0]] = child
        return mid

    def match_prefix(self, tokens: list[int]) -> tuple[list[int], RadixNode]:
        """返回最长已缓存前缀的槽号，以及该前缀结束处的节点（可用于加锁）。"""
        node, i, slots, now = self.root, 0, [], next(self.clock)
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            common = _common_prefix(child.key, tokens[i:])
            if common < len(child.key):
                child = self._split(child, common)
            child.last_access = now
            slots += child.value
            node, i = child, i + common
        return slots, node

    def insert(self, tokens: list[int], slots: list[int]) -> int:
        """插入一条序列及其槽号，返回其中已经在树里的前缀长度（这部分槽号由调用者释放）。"""
        matched, node = self.match_prefix(tokens)
        n = len(matched)
        if n < len(tokens):
            leaf = RadixNode(tuple(tokens[n:]), list(slots[n:]), parent=node,
                             last_access=next(self.clock))
            node.children[tokens[n]] = leaf
        return n

    def lock(self, node: RadixNode, delta: int = 1) -> None:
        while node is not self.root:
            node.lock_ref += delta
            node = node.parent

    def evict(self, num_tokens: int) -> list[int]:
        """从最久未用的未加锁叶子开始淘汰，直到释放至少 num_tokens 个槽。"""
        leaves = [(n.last_access, id(n), n) for n in self._nodes() if not n.children
                  and n.lock_ref == 0]
        heapq.heapify(leaves)
        freed: list[int] = []
        while leaves and len(freed) < num_tokens:
            _, _, leaf = heapq.heappop(leaves)
            freed += leaf.value
            parent = leaf.parent
            del parent.children[leaf.key[0]]
            if parent is not self.root and not parent.children and parent.lock_ref == 0:
                heapq.heappush(leaves, (parent.last_access, id(parent), parent))
        return freed

    def _nodes(self):
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def total_tokens(self) -> int:
        return sum(len(n.key) for n in self._nodes())


def _common_prefix(a: tuple[int, ...], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n
# endregion radix

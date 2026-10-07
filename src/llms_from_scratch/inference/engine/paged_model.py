"""读取分页 KV Cache 的 GPT 前向：复用 transformer/model.py 的全部权重与模块，只重写注意力的数据通路。

输入是“打平”的一批 token（不同请求、prefill 与 decode 混在一起，形状 [N]），
每个 token 自带：位置、KV 写入槽、可见长度、所属请求的块表。于是 prefill、chunked prefill、
decode 和前缀缓存命中都是同一条代码路径——这正是 vLLM V1 统一调度所依赖的执行模型。
"""

from __future__ import annotations

import torch

from llms_from_scratch.inference.paged_kv import allocate_cache, paged_attention, write_kv
from llms_from_scratch.transformer.model import GPT, CausalSelfAttention, apply_rope


class PagedGPT:
    def __init__(self, model: GPT, num_blocks: int, block_size: int) -> None:
        cfg = model.config
        self.model, self.block_size = model, block_size
        dtype, device = model.embed.weight.dtype, model.embed.weight.device
        caches = [allocate_cache(num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim, dtype, device)
                  for _ in range(cfg.n_layers)]
        self.k_caches = [k for k, _ in caches]
        self.v_caches = [v for _, v in caches]

    # region forward
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                seq_lens: torch.Tensor, block_tables: torch.Tensor) -> torch.Tensor:
        """input_ids/positions/slots/seq_lens: [N]；block_tables: [N, max_blocks]。返回隐状态 [N, d]。"""
        m = self.model
        x = m.embed(input_ids)
        freqs = m.freqs[positions]  # 每个 token 用自己的绝对位置做 RoPE
        for layer, block in enumerate(m.blocks):
            h = block.attn_norm(x)
            x = x + self._attention(block.attn, h, freqs, layer, slots, seq_lens, block_tables)
            x = x + block.ffn(block.ffn_norm(x))
        return m.norm(x)

    def _attention(self, attn: CausalSelfAttention, h: torch.Tensor, freqs: torch.Tensor,
                   layer: int, slots, seq_lens, block_tables) -> torch.Tensor:
        n = h.shape[0]
        q = attn.q_proj(h).view(n, attn.n_heads, attn.head_dim)
        k = attn.k_proj(h).view(n, attn.n_kv_heads, attn.head_dim)
        v = attn.v_proj(h).view(n, attn.n_kv_heads, attn.head_dim)
        # apply_rope 要求位置维在倒数第二维：[heads, N, head_dim]
        q = apply_rope(q.transpose(0, 1), freqs).transpose(0, 1)
        k = apply_rope(k.transpose(0, 1), freqs).transpose(0, 1)
        k_cache, v_cache = self.k_caches[layer], self.v_caches[layer]
        write_kv(k_cache, v_cache, slots, k, v)  # 先写入本步的 K/V……
        out = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)  # ……再按块表读
        return attn.o_proj(out.reshape(n, -1))
    # endregion forward

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)

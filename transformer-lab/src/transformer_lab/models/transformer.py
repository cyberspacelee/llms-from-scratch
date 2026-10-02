"""Encoder-only、Decoder-only 和 Encoder–Decoder 的组装及状态管理。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..attention.position import sinusoidal
from ..cache import LayerCache, ModelCache
from ..config import AttentionConfig, BlockConfig, ModelConfig
from ..layers import Routing, make_norm
from .blocks import TransformerBlock


@dataclass(frozen=True)
class EncoderMemory:
    """Encoder 输出 hidden[B,S,D] 与 valid[B,S]，可供多个 decode 会话复用。

    layers 只用于 causal Encoder 的流式追加；双向 Encoder 的新输入会改变
    旧 hidden，必须重新编码。hidden memory 与各 Decoder 层的 Cross KV
    是两种不同状态，后者由每层的 K/V 权重从此 memory 生成。
    """

    hidden: torch.Tensor
    valid: torch.Tensor
    layers: tuple[LayerCache, ...] | None = None  # causal encoder streaming only
    auxiliary_loss: torch.Tensor | None = None
    routing: Routing = ()


@dataclass(frozen=True)
class ModelOutput:
    """logits[B,T,V]、hidden[B,T,D]，以及可选 cache 和训练所需 MoE 统计。"""

    logits: torch.Tensor
    hidden: torch.Tensor
    cache: ModelCache | None
    auxiliary_loss: torch.Tensor
    routing: Routing = ()


class Transformer(nn.Module):
    """用同一 Block 构造三类模型；模型层负责位置、mask、memory 和会话状态。

    Encoder-only 使用双向 self-attention；Decoder-only 使用因果 self-attention；
    Encoder–Decoder 先 encode source，再由 Decoder 通过 cross-attention 读取。
    配置允许输入/输出使用不同 Block、Attention 和层数，默认共享 token embedding。
    绑定 embedding/head 时参数计数只计一次；head 投影仍需约 2*B*T*D*V FLOPs。
    layer_blocks 支持 local/global、RoPE/NoPE、递归/softmax 的逐层交替。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        configs = config.layer_blocks or (config.block,) * config.layers
        cross = config.cross_attention or config.block.attention
        if config.architecture == "encoder_decoder" and cross.kind in {
            "linear",
            "delta",
            "gated_delta",
        }:
            cross = AttentionConfig()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.dim,
                    c,
                    cross if config.architecture == "encoder_decoder" else None,
                    config.cross_causal,
                )
                for c in configs
            ]
        )
        self.norm = (
            make_norm(configs[-1].norm, config.dim)
            if configs[-1].norm_order == "pre"
            else nn.Identity()
        )
        if config.architecture == "encoder_decoder":
            ec = config.encoder_block or config.block
            self.encoder_blocks = nn.ModuleList(
                [TransformerBlock(config.dim, ec) for _ in range(config.encoder_layers)]
            )
            self.encoder_norm = (
                make_norm(ec.norm, config.dim) if ec.norm_order == "pre" else nn.Identity()
            )
        else:
            self.encoder_blocks = nn.ModuleList()
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.mtp_depth:
            from ..training.mtp import MultiTokenPrediction

            self.mtp = MultiTokenPrediction(config.dim, config.block, config.mtp_depth)
        else:
            self.mtp = None
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def validate_ids(self, ids: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if (
            ids.ndim != 2
            or ids.dtype != torch.long
            or ids.numel() == 0
            or ids.device != self.embedding.weight.device
        ):
            raise ValueError("IDs must be nonempty long [B,T] on model device")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("token outside vocabulary")
        if valid is None:
            return torch.ones_like(ids, dtype=torch.bool)
        if valid.dtype != torch.bool or valid.shape != ids.shape or valid.device != ids.device:
            raise ValueError("valid must be boolean [B,T] on input device")
        return valid

    def embed(self, ids: torch.Tensor, offset: int, block: BlockConfig) -> torch.Tensor:
        if offset + ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds max_length")
        x = self.embedding(ids)
        if block.attention.position.kind == "sinusoidal":
            positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            x = math.sqrt(self.config.dim) * x + sinusoidal(positions, self.config.dim, dtype).to(
                x.dtype
            )
        return nn.functional.dropout(x, block.dropout, self.training)

    def encode(
        self,
        ids: torch.Tensor,
        valid: torch.Tensor | None = None,
        past: EncoderMemory | None = None,
        use_cache: bool = False,
    ) -> EncoderMemory:
        """source_ids[B,S] -> memory[B,S,D]；past 只适用于 causal Encoder。

        双向编码通常一次计算再复用；因果编码可分块追加，并拼接旧 hidden。
        编码缓存负责省去 Encoder 的重复计算，Cross cache 负责省去 Decoder
        中各层的重复 source 投影；两者的节省范围不同。
        """
        if self.config.architecture != "encoder_decoder":
            raise ValueError("encode() belongs to encoder-decoder models")
        valid = self.validate_ids(ids, valid)
        ec = self.config.encoder_block or self.config.block
        if past is not None and (not self.config.encoder_causal or past.layers is None):
            raise ValueError("bidirectional encoders cannot append cached hidden states")
        offset = 0 if past is None else past.hidden.shape[1]
        if past is not None:
            if past.hidden.shape[0] != ids.shape[0] or len(past.layers) != len(self.encoder_blocks):
                raise ValueError("encoder cache batch/layer count mismatch")
            valid = torch.cat((past.valid, valid), 1)
        x = self.embed(ids, offset, ec)
        layers, routing, auxiliary = [], (), x.new_zeros(())
        for i, block in enumerate(self.encoder_blocks):
            x, layer, aux, stats = block(
                x,
                self.config.encoder_causal,
                valid,
                None if past is None else past.layers[i],
                use_cache and self.config.encoder_causal,
                offset,
            )
            layers.append(layer)
            auxiliary, routing = auxiliary + aux, routing + stats
        x = self.encoder_norm(x)
        if past is not None:
            x = torch.cat((past.hidden, x), 1)
        return EncoderMemory(
            x,
            valid,
            tuple(layers) if use_cache and self.config.encoder_causal else None,
            auxiliary,
            routing,
        )

    def forward(
        self,
        ids: torch.Tensor,
        *,
        source_ids: torch.Tensor | None = None,
        source_valid: torch.Tensor | None = None,
        memory: EncoderMemory | None = None,
        valid: torch.Tensor | None = None,
        cache: ModelCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """训练传完整 ids[B,T]；prefill 传前缀并 use_cache=True；decode 传新 chunk。

        seq2seq 首次调用必须提供 source_ids 或 memory，二者互斥。
        已缓存会话拥有固定 source，后续仅传 cache，不允许换 source。
        use_cache=False 不返回状态，不影响 autograd；teacher forcing 的右移
        和标签对齐由 training.losses 负责，模型不会自行移动输入或标签。
        """
        c = self.config
        current_valid = self.validate_ids(ids, valid)
        causal = c.architecture != "encoder" or c.encoder_causal
        if not causal and (use_cache or cache is not None):
            raise ValueError("bidirectional encoder forwards cannot append KV cache")
        if cache is not None:
            if (
                not use_cache
                or len(cache.layers) != len(self.blocks)
                or cache.valid.shape[0] != ids.shape[0]
                or cache.valid.dtype != torch.bool
                or cache.valid.device != ids.device
            ):
                raise ValueError("invalid model cache or use_cache=False")
            if any(layer.self_attention.length != cache.length for layer in cache.layers):
                raise ValueError("all layer positions must agree")
            if source_ids is not None or memory is not None or source_valid is not None:
                raise ValueError(
                    "cached session owns encoder memory; start a new session to change source"
                )
        # 同一绝对 offset 传到所有层；valid 由本次 [B,T] 扩展成完整历史 [B,S]。
        offset = 0 if cache is None else cache.length
        full_valid = current_valid if cache is None else torch.cat((cache.valid, current_valid), 1)
        auxiliary, routing = self.embedding.weight.new_zeros(()), ()
        if c.architecture == "encoder_decoder":
            if cache is not None:
                hidden_memory, memory_valid = cache.memory, cache.memory_valid
            else:
                if source_ids is not None and memory is not None:
                    raise ValueError("provide source_ids or prepared memory, not both")
                if memory is None:
                    if source_ids is None:
                        raise ValueError("encoder-decoder requires source_ids or memory")
                    memory = self.encode(source_ids, source_valid)
                elif source_valid is not None:
                    raise ValueError("prepared memory already contains source validity")
                hidden_memory, memory_valid = memory.hidden, memory.valid
                if memory.auxiliary_loss is not None:
                    auxiliary, routing = auxiliary + memory.auxiliary_loss, routing + memory.routing
            if (
                hidden_memory is None
                or hidden_memory.ndim != 3
                or hidden_memory.shape[0] != ids.shape[0]
                or hidden_memory.shape[-1] != c.dim
                or hidden_memory.device != ids.device
                or hidden_memory.dtype != self.embedding.weight.dtype
            ):
                raise ValueError("encoder memory is incompatible")
            if (
                memory_valid is None
                or memory_valid.shape != hidden_memory.shape[:2]
                or memory_valid.dtype != torch.bool
                or memory_valid.device != ids.device
            ):
                raise ValueError("invalid encoder validity mask")
        else:
            if source_ids is not None or memory is not None or source_valid is not None:
                raise ValueError("source memory is only used by encoder-decoder")
            hidden_memory = memory_valid = None
        x = self.embed(ids, offset, c.block)
        layers = []
        for i, block in enumerate(self.blocks):
            x, layer, aux, stats = block(
                x,
                causal,
                full_valid,
                None if cache is None else cache.layers[i],
                use_cache,
                offset,
                hidden_memory,
                memory_valid,
            )
            layers.append(layer)
            auxiliary, routing = auxiliary + aux, routing + stats
        x = self.norm(x)
        result_cache = (
            ModelCache(tuple(layers), full_valid, hidden_memory, memory_valid)
            if use_cache
            else None
        )
        return ModelOutput(self.head(x), x, result_cache, auxiliary, routing)

    @torch.no_grad()
    def update_router_bias(self, routing: Routing) -> None:
        for expert, counts in routing:
            expert.update_balance(counts)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        source_ids: torch.Tensor | None = None,
        memory: EncoderMemory | None = None,
        source_valid: torch.Tensor | None = None,
        eos_id: int | None = None,
        cached: bool = True,
    ) -> torch.Tensor:
        """等长无 padding prompt 的 greedy 自回归生成，返回 prompt+新 token。

        cached=True: 一次 prefill，随后每次只输入最新 token；
        cached=False: 每次重算完整前缀，用作数值对照。seq2seq memory 只编码一次。
        函数暂时进入 eval/no_grad，再恢复原 training 标志；EOS 后保持 EOS。
        """
        if self.config.architecture == "encoder":
            raise ValueError("generation requires a decoder")
        self.validate_ids(prompt, None)
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 0
            or prompt.shape[1] + max_new_tokens > self.config.max_length
        ):
            raise ValueError("generation length exceeds context")
        if eos_id is not None and (
            type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size
        ):
            raise ValueError("invalid EOS")
        was_training = self.training
        self.eval()
        result, cache = prompt.clone(), None
        finished = torch.zeros(prompt.shape[0], device=prompt.device, dtype=torch.bool)
        try:
            if self.config.architecture == "encoder_decoder" and memory is None:
                if source_ids is None:
                    raise ValueError("generation requires encoder source")
                memory = self.encode(source_ids, source_valid)
            for _ in range(max_new_tokens):
                current = result[:, -1:] if cached and cache is not None else result
                output = self(
                    current, memory=memory if cache is None else None, cache=cache, use_cache=cached
                )
                cache = output.cache
                token = output.logits[:, -1].argmax(-1)
                if eos_id is not None:
                    token = torch.where(finished, eos_id, token)
                    finished |= token == eos_id
                result = torch.cat((result, token[:, None]), 1)
                if finished.all():
                    break
        finally:
            self.train(was_training)
        return result

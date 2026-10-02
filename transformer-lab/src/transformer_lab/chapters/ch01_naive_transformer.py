"""01 · 2017：朴素 Encoder–Decoder Transformer，直接写出完整计算。

阅读顺序：NaiveAttention -> NaiveTransformer.forward -> demo。
这里固定结构，不引入配置、路由或缓存；Attention 数学原语复用核心实现。
运行：uv run transformer-lab chapter --chapter 01
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..attention.position import sinusoidal
from ..attention.softmax import scaled_dot_product_attention
from ..config import AttentionConfig, BlockConfig, ModelConfig, PositionConfig
from ..experiments.runners import consistency
from ..models import Transformer
from ..training.losses import teacher_forcing, token_loss


class NaiveAttention(nn.Module):
    """Q 来自待更新序列；K/V 来自被读取序列。x[B,S_q,D]与source[B,S_kv,D] -> output[B,S_q,D]。"""

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: model dimension D，必须能被 num_heads 整除。
            num_heads: query head 数 H_q，必须为正整数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if type(num_heads) is not int or num_heads < 1 or dim < 1 or dim % num_heads:
            raise ValueError("d_model must be divisible by positive num_heads")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, source: torch.Tensor, causal: bool = False) -> torch.Tensor:
        # 投影 [B,S,D] -> 拆头 [B,S,H_q,D_h] -> 转置 [B,H_q,S,D_h]。
        # self-attention: source=x；cross-attention: source=Encoder 的输出。
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            x: float [B,S_q,D]。
            source: float [B,S_kv,D]。
            causal: 是否限制 key 绝对位置不超过 query。

        Returns:
            float [B,S_q,D]，不持有缓存；causal只用于从位置0开始的序列。
        """
        batch_size, query_len, d_model = x.shape
        key_len = source.shape[1]
        q = self.q(x).reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k(source)
            .reshape(batch_size, key_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(source)
            .reshape(batch_size, key_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        visible = None
        if causal:
            # 第 t 行只能读取 0..t。True=可见，不是 PyTorch 某些 API 的屏蔽约定。
            visible = torch.ones(
                x.shape[1], source.shape[1], device=x.device, dtype=torch.bool
            ).tril()
        # 原语内只有 Q@K^T / sqrt(D_h)、masked softmax 和 P@V，没有高级 Attention 封装。
        context = scaled_dot_product_attention(q, k, v, visible)  # [B,H_q,S_q,D_h]
        context = context.transpose(1, 2).reshape(batch_size, query_len, d_model)  # [B,S_q,D]
        return self.output(context)  # [B,S_q,D]


class NaiveTransformer(nn.Module):
    """最小可训练 seq2seq：一个 Encoder、一个带 Cross-Attention 的 Decoder。

    source_ids [B,S_kv]、decoder_ids [B,S_q] -> logits [B,S_q,V]。
    入门约束：两侧共享词表及 embedding，无 padding、无 dropout、无权重绑定。
    dim/hidden_dim/vocab_size 只控制尺寸，forward 没有现代架构开关。
    """

    def __init__(
        self, vocab_size: int = 32, dim: int = 32, hidden_dim: int = 64, num_heads: int = 4
    ) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            vocab_size: 词表大小 V，正整数。
            dim: model dimension D，必须能被 num_heads 整除。
            hidden_dim: FFN intermediate size D_ff。
            num_heads: query head 数 H_q，必须为正整数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if min(vocab_size, dim, hidden_dim) < 1:
            raise ValueError("vocabulary and dimensions must be positive")
        self.dim, self.num_heads = dim, num_heads
        self.embedding = nn.Embedding(vocab_size, dim)
        self.encoder_attention = NaiveAttention(dim, num_heads)
        self.decoder_attention = NaiveAttention(dim, num_heads)
        self.cross_attention = NaiveAttention(dim, num_heads)
        self.encoder_ff = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.decoder_ff = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        # Encoder 两条残差分支；Decoder 三条：self / cross / FFN。
        self.encoder_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.decoder_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """经典输入 sqrt(D)*Embedding + Sinusoidal PE，形状 [B,L_kv,D]。

        Args:
            ids: 非空 long [B,S]，与模型同设备且 ID 在词表内。

        Returns:
            float [B,S,D]，查表后加 Sinusoidal PE[S,D]。
        """
        if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("token ids must be nonempty long [B,L_kv]")
        if ids.device != self.embedding.weight.device:
            raise ValueError("tokens and model must use the same device")
        if ids.min() < 0 or ids.max() >= self.embedding.num_embeddings:
            raise ValueError("token outside vocabulary")
        x = self.embedding(ids) * math.sqrt(self.dim)
        positions = torch.arange(ids.shape[1], device=ids.device)
        # 与核心实现相同：低精度统计升 fp32，double 模型保留 fp64。
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        return x + sinusoidal(positions, self.dim, dtype).to(x.dtype)

    def forward(self, source_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        """依次执行双向 Encoder、因果 Decoder、自输出到输入的 Cross-Attention。

        Args:
            source_ids: 可选 long [B,S_kv] Encoder token IDs。
            decoder_ids: long [B,S_q]，右移后的 Decoder token IDs。

        Returns:
            float logits[B,S_q,V]；双向 Encoder memory[B,S_kv,D]。
        """
        memory, x = self.embed(source_ids), self.embed(decoder_ids)
        if source_ids.shape[0] != decoder_ids.shape[0]:
            raise ValueError("source and decoder batch sizes must match")
        # Post-Norm: LN(x + F(x))。Encoder 不遮挡未来输入。
        memory = self.encoder_norms[0](memory + self.encoder_attention(memory, memory))
        memory = self.encoder_norms[1](memory + self.encoder_ff(memory))
        # Decoder self-attention 防止读取未来目标；Cross 可以读取完整 source。
        x = self.decoder_norms[0](x + self.decoder_attention(x, x, causal=True))
        x = self.decoder_norms[1](x + self.cross_attention(x, memory))
        x = self.decoder_norms[2](x + self.decoder_ff(x))
        return self.head(x)


def demo(device: torch.device) -> dict[str, object]:
    """只做一次前向、反向和参数更新，检查可训练性；不测性能或语言能力。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 单步训练的输入/输出 shape、loss、参数数目和检查结果。
    """
    model = NaiveTransformer().to(device)
    source = torch.tensor([[1, 2, 3], [3, 2, 1]], device=device)
    targets = torch.tensor([[4, 5, 6, 7], [6, 5, 4, 7]], device=device)  # 7 当作 EOS
    inputs, labels, _ = teacher_forcing(targets, bos_id=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(source, inputs)
    loss = token_loss(logits, labels)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = model.embedding.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.embedding.weight)
    # 改动未来 decoder 输入，早先位置必须保持不变。
    model.eval()
    with torch.no_grad():
        expected = model(source, inputs)
        changed = inputs.clone()
        changed[:, 2:] = 8
        torch.testing.assert_close(expected[:, :2], model(source, changed)[:, :2])
    return {
        "model": "naive encoder-decoder",
        "device": str(device),
        "source_shape": list(source.shape),
        "decoder_input": inputs.tolist(),
        "logits_shape": list(logits.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss_before_one_update": loss.item(),
        "checks": ["finite gradients", "optimizer update", "decoder causality"],
    }


def classic_config(heads: int = 4, layers: int = 1) -> ModelConfig:
    """与朴素模型完全同构的配置；heads 增加时总投影宽度仍保持 D=32。

    Args:
        heads: query head 数 H_q，必须整除本例 D=32。
        layers: Transformer Block 数，正整数。

    Returns:
        ModelConfig: 经典 seq2seq，D=32、D_ff=64、H_q*D_h=32。
    """
    if type(heads) is not int or heads < 1 or 32 % heads:
        raise ValueError("heads must be a positive divisor of 32 for this lesson")
    return ModelConfig(
        architecture="encoder_decoder",
        vocab_size=32,
        dim=32,
        layers=layers,
        encoder_layers=layers,
        tie_embeddings=False,
        block=BlockConfig(
            attention=AttentionConfig(
                kind="mha",
                heads=heads,
                head_dim=32 // heads,
                position=PositionConfig(kind="sinusoidal"),
            ),
            norm="layer",
            norm_order="post",
            activation="relu",
            ff_dim=64,
        ),
    )


def to_core(model: NaiveTransformer) -> Transformer:
    """将固定 朴素模型 的权重放入统一模型，验证重组接口并未改动计算。

    教学桥梁仅适用于默认尺寸、可选head数、单层朴素模型；后续升级创建各自模型，
    不声称结构不同的模型能加载相同 checkpoint。

    Args:
        model: 默认尺寸 NaiveTransformer，可为不同合法 head 数。

    Returns:
        具有相同参数/结构的核心 Transformer，沿用 device/dtype/training。
    """
    if (
        model.dim != 32
        or model.embedding.num_embeddings != 32
        or model.encoder_ff[0].out_features != 64
    ):
        raise ValueError("teaching bridge expects the default 朴素模型 dimensions")
    result = Transformer(classic_config(heads=model.num_heads)).to(
        device=model.embedding.weight.device, dtype=model.embedding.weight.dtype
    )
    result.embedding.load_state_dict(model.embedding.state_dict())
    result.head.load_state_dict(model.head.state_dict())
    encoder, decoder = result.encoder_blocks[0], result.blocks[0]
    encoder.attention.load_state_dict(model.encoder_attention.state_dict())
    decoder.attention.load_state_dict(model.decoder_attention.state_dict())
    decoder.cross.load_state_dict(model.cross_attention.state_dict())
    for block, ff, norms in (
        (encoder, model.encoder_ff, model.encoder_norms),
        (decoder, model.decoder_ff, model.decoder_norms),
    ):
        block.ff.up.load_state_dict(ff[0].state_dict())
        block.ff.down.load_state_dict(ff[2].state_dict())
        block.norms.load_state_dict(norms.state_dict())
    result.train(model.training)
    return result


def run(device: torch.device) -> dict[str, object]:
    """输入执行设备；返回单步训练、同权重输出和多头多层检查的报告。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    model = NaiveTransformer().to(device).double().eval()
    core = to_core(model)
    source = torch.tensor([[1, 2, 3]], device=device)  # [B=1,S_kv=3]
    decoder = torch.tensor([[0, 4, 5, 6]], device=device)  # [B=1,S_q=4]
    left = model(source, decoder)  # [1,4,V=32]
    right = core(decoder, source_ids=source).logits
    torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    left.square().mean().backward()
    right.square().mean().backward()
    torch.testing.assert_close(model.embedding.weight.grad, core.embedding.weight.grad)
    return {
        "training": demo(device),
        "source_shape": list(source.shape),
        "decoder_shape": list(decoder.shape),
        "logits_shape": list(left.shape),
        "naive_core_error": (left - right).abs().max().item(),
        "multihead_multilayer": consistency(classic_config(heads=4, layers=2), device),
    }

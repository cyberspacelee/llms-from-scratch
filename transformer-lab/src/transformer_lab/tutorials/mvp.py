"""第 00 步：一层、单头、Post-Norm 的经典 Encoder–Decoder Transformer。

阅读顺序：SingleHeadAttention -> MinimalTransformer.forward -> demo。
这里固定结构，不引入配置、路由或缓存；Attention 数学原语复用核心实现。
运行：uv run python -m transformer_lab.tutorials.mvp --device cpu
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn

from ..attention.position import sinusoidal
from ..attention.softmax import scaled_dot_product_attention
from ..training.losses import teacher_forcing, token_loss


class SingleHeadAttention(nn.Module):
    """Q 来自待更新序列；K/V 来自被读取序列。输入输出都是 [B,T,D]。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, source: torch.Tensor, causal: bool = False) -> torch.Tensor:
        # 只有一个 head，unsqueeze 把 [B,T,D] 变为 [B,1,T,D]。
        # self-attention: source=x；cross-attention: source=Encoder 的输出。
        q, k, v = self.q(x).unsqueeze(1), self.k(source).unsqueeze(1), self.v(source).unsqueeze(1)
        visible = None
        if causal:
            # 第 t 行只能读取 0..t。True=可见，不是 PyTorch 某些 API 的屏蔽约定。
            visible = torch.ones(
                x.shape[1], source.shape[1], device=x.device, dtype=torch.bool
            ).tril()
        # 原语内只有 Q@K^T / sqrt(D)、masked softmax 和 P@V，没有高级 Attention 封装。
        context = scaled_dot_product_attention(q, k, v, visible).squeeze(1)
        return self.output(context)


class MinimalTransformer(nn.Module):
    """最小可训练 seq2seq：一个 Encoder、一个带 Cross-Attention 的 Decoder。

    source_ids [B,S]、decoder_ids [B,T] -> logits [B,T,V]。
    入门约束：两侧共享词表及 embedding，无 padding、无 dropout、无权重绑定。
    dim/hidden_dim/vocab_size 只控制尺寸，forward 没有现代架构开关。
    """

    def __init__(self, vocab_size: int = 32, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(vocab_size, dim, hidden_dim) < 1:
            raise ValueError("vocabulary and dimensions must be positive")
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.encoder_attention = SingleHeadAttention(dim)
        self.decoder_attention = SingleHeadAttention(dim)
        self.cross_attention = SingleHeadAttention(dim)
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
        """经典输入 sqrt(D)*Embedding + Sinusoidal PE，形状 [B,L,D]。"""
        if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("token ids must be nonempty long [B,L]")
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
        """依次执行双向 Encoder、因果 Decoder、自输出到输入的 Cross-Attention。"""
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
    """只做一次前向、反向和参数更新，检查可训练性；不测性能或语言能力。"""
    model = MinimalTransformer().to(device)
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
        "step": "00: minimal encoder-decoder",
        "device": str(device),
        "source_shape": list(source.shape),
        "decoder_input": inputs.tolist(),
        "logits_shape": list(logits.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss_before_one_update": loss.item(),
        "checks": ["finite gradients", "optimizer update", "decoder causality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    print(json.dumps(demo(torch.device(args.device)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

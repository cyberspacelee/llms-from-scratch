"""视觉语言模型的输入端：ViT patch 嵌入、patch 合并与投影器、图文拼接与 M-RoPE 位置编号。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# region patch
def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, H, W] → [B, (H/p)(W/p), C·p·p]：按行优先把图像切成不重叠的 p×p 小块并展平。"""
    B, C, H, W = images.shape
    x = images.unfold(2, patch, patch).unfold(3, patch, patch)  # [B, C, H/p, W/p, p, p]
    return x.permute(0, 2, 3, 1, 4, 5).reshape(B, (H // patch) * (W // patch), C * patch * patch)


class PatchEmbed(nn.Module):
    """ViT 的 patch 嵌入：步长等于核大小的卷积，等价于“切块 + 共享的线性层”。"""

    def __init__(self, in_channels: int, patch: int, d: int) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_channels, d, kernel_size=patch, stride=patch)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)  # [B, N_patches, d]
# endregion


# region projector
class MLPProjector(nn.Module):
    """LLaVA-1.5 的两层 MLP 投影器：把视觉特征映射到语言模型的嵌入空间。"""

    def __init__(self, d_vision: int, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_vision, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PatchMerger(nn.Module):
    """Qwen2-VL 式合并：相邻 2×2 个 patch 拼接后经 MLP 投影，视觉 token 数降为 1/4。"""

    def __init__(self, d_vision: int, d_model: int, merge: int = 2) -> None:
        super().__init__()
        self.merge = merge
        self.mlp = MLPProjector(d_vision * merge * merge, d_model)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        B, _, d = x.shape
        m = self.merge
        x = x.view(B, grid_h // m, m, grid_w // m, m, d).permute(0, 1, 3, 2, 4, 5)
        return self.mlp(x.reshape(B, (grid_h // m) * (grid_w // m), m * m * d))
# endregion


def splice_image_tokens(text_emb: torch.Tensor, is_image: torch.Tensor,
                        image_tokens: torch.Tensor) -> torch.Tensor:
    """把视觉 token 按顺序填入文本序列中占位符所在的位置。

    text_emb[L, d] 为整条序列的文本嵌入（占位符位置的嵌入会被替换），
    is_image[L] 标记占位符，image_tokens[N_img, d] 的行数必须等于占位符个数。
    """
    out = text_emb.clone()
    out[is_image] = image_tokens
    return out


# region mrope
def mrope_position_ids(segments: list[tuple]) -> torch.Tensor:
    """Qwen2-VL 的 M-RoPE 位置编号，返回 [3, L]，三行依次为 (时间, 高, 宽)。

    segments 依次为 ("text", n) 或 ("image", h, w)（h, w 为合并后的网格）。
    文本 token 的三个分量相同，等价于一维 RoPE；图像 token 的时间分量固定为起点，
    高、宽分量为起点 + 行号 / 列号。下一段从当前最大编号 + 1 开始。
    """
    ids, start = [], 0
    for seg in segments:
        if seg[0] == "text":
            pos = torch.arange(start, start + seg[1])
            ids.append(torch.stack([pos, pos, pos]))
        else:
            _, h, w = seg
            rows = torch.arange(h).repeat_interleave(w)
            cols = torch.arange(w).repeat(h)
            ids.append(torch.stack([torch.full((h * w,), start), start + rows, start + cols]))
        start = int(ids[-1].max()) + 1
    return torch.cat(ids, dim=1)


def mrope_frequencies(pos_ids: torch.Tensor, head_dim: int, sections: tuple[int, int, int],
                      theta: float = 10000.0) -> torch.Tensor:
    """把 head_dim/2 个旋转频率分成三段，分别用时间、高、宽编号转动。

    返回 [L, head_dim/2] 的复数因子，可直接交给 ``transformer.model.apply_rope``。
    """
    assert sum(sections) == head_dim // 2
    inv = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    which = torch.repeat_interleave(torch.arange(3), torch.tensor(sections))  # 每个频率用哪一行编号
    angles = pos_ids[which].t().float() * inv  # [L, head_dim/2]
    return torch.polar(torch.ones_like(angles), angles)
# endregion

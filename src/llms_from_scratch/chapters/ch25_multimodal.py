"""25 · 2026：DeepSeek VL 与 Qwen VL/Omni 的模态特征 → 语言 token 接口。

输入已编码的视觉特征，展示 projector/scatter/causal backbone；不下载视觉/音频模型。
"""

from __future__ import annotations

import torch
from torch import nn

from ..models import Transformer
from .ch20_qwen_hybrid import config


class MultimodalTransformer(nn.Module):
    """文本 token 与模态特征在同一个 causal hidden 序列中融合。"""

    def __init__(self, feature_dim: int = 12) -> None:
        """输入模态 feature width D_in；构造四层 hybrid语言骨干和 projector，返回 None。

        Args:
            feature_dim: 模态特征宽度 D_in，正整数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError("feature dimension must be positive")
        self.backbone = Transformer(config())
        self.projector = nn.Linear(feature_dim, self.backbone.config.dim, bias=False)

    def forward(
        self, ids: torch.Tensor, features: torch.Tensor, image_slots: torch.Tensor
    ) -> torch.Tensor:
        """输入 ids[B,T]、features[B,N_image,D_in]、bool image_slots[B,T]。

        每行恰好 N_image 个 slot；projector输出替换这些 token 的 embedding。
        返回 logits[B,T,V]；序列位置使用文本 RoPE，未实现视觉 2D/multimodal RoPE。

        Args:
            ids: 非空 long [B,T]，与模型同设备且 ID 在词表内。
            features: float [B,N_image,D_in]，已由模态 encoder 编码的特征。
            image_slots: bool [B,T]，每行恰好 N_image 个 True。

        Returns:
            float logits[B,T,V]。
        """
        valid = self.backbone.validate_ids(ids, None)
        if (
            image_slots.shape != ids.shape
            or image_slots.dtype != torch.bool
            or image_slots.device != ids.device
            or features.ndim != 3
            or features.shape[0] != ids.shape[0]
            or features.shape[-1] != self.projector.in_features
            or features.device != ids.device
            or features.dtype != self.projector.weight.dtype
            or not torch.isfinite(features).all()
            or not (image_slots.sum(1) == features.shape[1]).all()
        ):
            raise ValueError("features and image slots must align per batch")
        if ids.shape[1] > self.backbone.config.max_length:
            raise ValueError("sequence exceeds context")
        x = self.backbone.embedding(ids)  # [B,T,D]
        visual = self.projector(features)  # [B,N_image,D_in] -> [B,N_image,D]
        x = x.clone()
        x[image_slots] = visual.reshape(-1, x.shape[-1])  # [B*N_image,D] -> selected token rows
        return self.backbone(ids, inputs_embeds=x, valid=valid).logits


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 projector/fusion shape、视觉特征梯度和因果隔离检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    model = MultimodalTransformer().to(device).double()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    slots = torch.tensor([[False, False, True, True, False, False]], device=device)
    features = torch.randn(1, 2, 12, device=device, dtype=torch.float64, requires_grad=True)
    logits = model(ids, features, slots)
    changed = features.detach() + 100
    torch.testing.assert_close(logits[:, :2], model(ids, changed, slots)[:, :2])
    logits.square().mean().backward()
    assert (
        features.grad is not None
        and torch.isfinite(features.grad).all()
        and features.grad.abs().sum() > 0
    )
    return {
        "ids_shape": list(ids.shape),
        "features_shape": list(features.shape),
        "projected_shape": [1, 2, model.backbone.config.dim],
        "fused_hidden_shape": [1, 6, model.backbone.config.dim],
        "logits_shape": list(logits.shape),
        "checks": ["visual projector gradients", "no future image leakage"],
        "note": "VL token interface reference; native vision encoder, mRoPE and Omni Thinker/Talker/audio codecs are covered in docs",
    }

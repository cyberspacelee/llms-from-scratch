"""18 · 2023 推理支线：Greedy Draft/Target 接受、修正、Bonus Token。"""

from __future__ import annotations

import torch

from ..inference import greedy_speculative_generate
from ..models import Transformer
from .ch11_gqa import config


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回同权重/不同权重草稿的接受数及 target-greedy 等价检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    target = Transformer(config()).to(device).eval()
    prompt = torch.tensor([[1, 2]], device=device)  # [B=1,T_prompt=2]
    report = {}
    for same_weights in (True, False):
        draft = Transformer(config()).to(device).eval()
        if same_weights:
            draft.load_state_dict(target.state_dict())
        speculative, stats = greedy_speculative_generate(target, draft, prompt, 5, draft_length=2)
        assert torch.equal(speculative, target.generate(prompt, 5))  # [1,T_prompt+5]
        report["same_weights" if same_weights else "different_weights"] = {
            "prompt_shape": list(prompt.shape),
            "output_shape": list(speculative.shape),
            "stats": stats,
            "tokens": speculative.tolist(),
        }
    return report

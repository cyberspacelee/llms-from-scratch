"""学习率调度：线性预热 + 余弦、WSD（Warmup–Stable–Decay）与多阶梯调度。

所有函数输入从 0 开始的步数 step，返回该步应使用的学习率。
"""

from __future__ import annotations

import math


# region cosine
def warmup_cosine(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """线性预热到 max_lr，再按半个余弦周期在 total 步时降到 min_lr，之后保持 min_lr。"""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
# endregion cosine


# region wsd
def wsd(step: int, max_lr: float, min_lr: float, warmup: int, total: int,
        decay_frac: float = 0.1, shape: str = "linear") -> float:
    """Warmup–Stable–Decay：预热后保持常数，只在最后 decay_frac 的步数里衰减。

    稳定段的任意一个检查点都可以“分叉”出一段短衰减得到一个完成训练的模型，
    因此一次长训练就能产出多个训练长度的结果（MiniCPM 用它做缩放实验）。
    shape 取 "linear"、"cosine" 或 "sqrt"（1 - √x，衰减初期下降更快）。
    """
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step < decay_start:
        return max_lr
    if step >= total:
        return min_lr
    x = (step - decay_start) / max(1, total - decay_start)   # 衰减段内的进度 ∈ [0, 1)
    if shape == "linear":
        w = 1 - x
    elif shape == "cosine":
        w = 0.5 * (1 + math.cos(math.pi * x))
    elif shape == "sqrt":
        w = 1 - math.sqrt(x)
    else:
        raise ValueError(f"未知的衰减形状 {shape}")
    return min_lr + (max_lr - min_lr) * w
# endregion wsd


def multi_step(step: int, max_lr: float, warmup: int, total: int,
               milestones: tuple[float, ...] = (0.8, 0.9),
               factors: tuple[float, ...] = (0.316, 0.1)) -> float:
    """DeepSeek LLM 的多阶梯调度：在 80% 与 90% 的训练进度处分别降到峰值的 31.6% 与 10%。"""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    lr = max_lr
    for milestone, factor in zip(milestones, factors, strict=True):
        if step >= milestone * total:
            lr = max_lr * factor
    return lr


def set_lr(optimizer, lr: float) -> None:
    """把同一学习率写入优化器（或 HybridOptimizer）的全部参数组。"""
    for group in optimizer.param_groups:
        group["lr"] = lr

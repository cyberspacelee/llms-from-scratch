"""Bradley–Terry 奖励模型：GPT 主干 + 标量头，读取最后一个有效 token 的隐藏状态。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.post_training.common import gpt_hidden
from llms_from_scratch.transformer.model import GPT


# region reward_model
class RewardModel(nn.Module):
    """r_φ(x, y)：把 prompt+response 送进 Transformer，在最后一个非填充 token 上输出一个标量。

    InstructGPT 的奖励模型就是去掉 unembedding 层、换成标量输出头的 SFT 模型。
    因果注意力下只有最后一个 token 看到了完整的回复，所以在那里读出分数。
    """

    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """input_ids[B, T]（右侧填充），lengths[B] 为有效长度 → 分数 [B]。"""
        hidden = gpt_hidden(self.backbone, input_ids)  # [B, T, d]
        last = hidden[torch.arange(len(lengths)), lengths - 1]  # [B, d]
        return self.head(last).squeeze(-1)


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """−log σ(r_w − r_l) 的批平均。logsigmoid 在差值很大或很小时都数值稳定。"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
# endregion reward_model


def preference_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    """奖励模型把偏好对排对的比例。"""
    return (r_chosen > r_rejected).float().mean().item()


def bradley_terry_prob(r_a: torch.Tensor, r_b: torch.Tensor) -> torch.Tensor:
    """BT 模型下 a 优于 b 的概率 σ(r_a − r_b)。"""
    return torch.sigmoid(r_a - r_b)

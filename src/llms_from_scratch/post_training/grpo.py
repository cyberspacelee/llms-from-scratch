"""GRPO：组内相对优势 + PPO 式裁剪 + k3 KL 估计，以及一个能在 CPU 上跑的完整训练循环。"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llms_from_scratch.post_training.common import response_logprobs, sample_responses
from llms_from_scratch.post_training.tasks import CharTokenizer, Problem
from llms_from_scratch.transformer.model import GPT


# region advantages
def group_advantages(rewards: torch.Tensor, normalize_std: bool = True,
                     eps: float = 1e-6) -> torch.Tensor:
    """rewards[P, G]：P 个问题、每题 G 条回复 → 组内相对优势 [P, G]。

    Â_i = (r_i − mean(r)) / (std(r) + ε)。组内奖励全相同时优势全为 0，这组不提供梯度。
    normalize_std=False 即 Dr. GRPO 的做法：只减均值，不除标准差。
    """
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize_std:
        return centered
    return centered / (rewards.std(dim=1, keepdim=True) + eps)
# endregion advantages


# region k3
def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """逐 token 的 KL(π_θ ‖ π_ref) 估计：ρ − log ρ − 1，ρ = π_ref / π_θ。

    在 y ~ π_θ 下 E[ρ] = 1，所以期望恰为 E[−log ρ] = KL(π_θ ‖ π_ref)（无偏）；
    且 x − log x − 1 ≥ 0，每个样本的估计都非负，方差比 −log ρ 小。
    """
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1
# endregion k3


# region grpo_loss
def grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: torch.Tensor,
              advantages: torch.Tensor, mask: torch.Tensor, clip_eps: float = 0.2,
              beta: float = 0.04) -> torch.Tensor:
    """DeepSeekMath 中的 GRPO 目标（取负号作为损失）。

    logp/old_logp/ref_logp/mask: [N, T]，advantages: [N]（每条回复一个标量，广播到所有 token）。
    每条回复先在自己的 token 上平均（1/|o_i|），再在 N 条回复间平均。
    """
    adv = advantages[:, None]
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    per_token = -(surrogate - beta * k3_kl(logp, ref_logp))
    mask = mask.to(per_token.dtype)
    per_seq = (per_token * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_seq.mean()
# endregion grpo_loss


@dataclass
class GRPOConfig:
    steps: int = 30
    group_size: int = 8
    max_new_tokens: int = 2
    lr: float = 3e-3
    inner_updates: int = 2  # 每批 rollout 上的梯度步数 μ；μ > 1 时裁剪才起作用
    clip_eps: float = 0.2
    beta: float = 0.0
    temperature: float = 1.0
    seed: int = 0


@torch.no_grad()
def greedy_accuracy(model: GPT, tokenizer: CharTokenizer, problems: list[Problem],
                    reward_fn: Callable[[str, str], float], max_new_tokens: int = 2) -> float:
    prompts = torch.tensor([tokenizer.encode(p.prompt) for p in problems])
    model.eval()
    out = model.generate(prompts, max_new_tokens, temperature=0)
    model.train()
    texts = [tokenizer.decode(row[prompts.shape[1]:].tolist()) for row in out]
    return sum(reward_fn(t, p.answer) for t, p in zip(texts, problems, strict=True)) / len(problems)


# region train_loop
def train_grpo(model: GPT, tokenizer: CharTokenizer, problems: list[Problem],
               reward_fn: Callable[[str, str], float], cfg: GRPOConfig) -> list[dict]:
    """在 problems 上跑 cfg.steps 轮 GRPO。所有问题的提示必须等长（玩具任务满足）。

    每一轮：① 每个问题采样 G 条回复；② 用规则奖励打分；③ 组内归一化得到优势；
    ④ 冻结 old_logp（行为策略）与 ref_logp（参考策略），在同一批数据上做 μ 次梯度更新。
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    prompts = torch.tensor([tokenizer.encode(p.prompt) for p in problems])
    batch = prompts.repeat_interleave(cfg.group_size, dim=0)  # [P·G, L]，同题的 G 条相邻
    answers = [p.answer for p in problems for _ in range(cfg.group_size)]
    logs = []
    for step in range(cfg.steps):
        # ① 采样
        roll = sample_responses(model, batch, cfg.max_new_tokens, tokenizer.eos_id,
                                cfg.temperature, gen)
        # ② 奖励：把回复解码成文本，交给可验证的规则
        texts = [tokenizer.decode(r) for r in roll.responses]
        rewards = torch.tensor([reward_fn(t, a) for t, a in zip(texts, answers, strict=True)])
        # ③ 组内相对优势
        adv = group_advantages(rewards.view(len(problems), cfg.group_size)).view(-1)
        # ④ 冻结行为策略与参考策略的对数概率
        with torch.no_grad():
            old_logp = response_logprobs(model, roll.sequences)
            ref_logp = response_logprobs(reference, roll.sequences)
        for _ in range(cfg.inner_updates):
            logp = response_logprobs(model, roll.sequences)
            loss = grpo_loss(logp, old_logp, ref_logp, adv, roll.response_mask,
                             cfg.clip_eps, cfg.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        logs.append({"step": step, "reward": rewards.mean().item(), "loss": loss.item()})
    return logs
# endregion train_loop


@torch.no_grad()
def answer_token_prob(model: GPT, tokenizer: CharTokenizer, problems: list[Problem]) -> float:
    """第一个回复 token 就是正确答案（单字符）的平均概率：比采样奖励平滑，适合做测试指标。"""
    prompts = torch.tensor([tokenizer.encode(p.prompt) for p in problems])
    targets = torch.tensor([tokenizer.encode(p.answer)[0] for p in problems])
    probs = model(prompts)[:, -1].softmax(-1)
    return probs.gather(1, targets[:, None]).mean().item()


def demo(steps: int = 40) -> None:
    """从随机初始化的极小 GPT 出发，只用 0/1 奖励学会 a+b（a, b ≤ 4）。"""
    from llms_from_scratch.post_training.common import tiny_gpt
    from llms_from_scratch.post_training.tasks import (
        addition_problems,
        addition_reward,
        arithmetic_tokenizer,
    )

    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    problems = addition_problems(4)
    model = tiny_gpt(tok.vocab_size, seed=0, context_length=16)
    before = greedy_accuracy(model, tok, problems, addition_reward)
    logs = train_grpo(model, tok, problems, addition_reward, GRPOConfig(steps=steps))
    after = greedy_accuracy(model, tok, problems, addition_reward)
    print("sampled reward:", " ".join(f"{row['reward']:.2f}" for row in logs))
    print(f"greedy accuracy {before:.2f} -> {after:.2f}")


if __name__ == "__main__":
    demo()

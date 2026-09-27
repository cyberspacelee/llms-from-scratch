"""Auditable answer rewards, group advantages, and finite-policy gradients."""
import re
import torch


def answer_reward(response, correct):
    match = re.fullmatch(r"<answer>\s*(-?\d+)\s*</answer>", response.strip())
    return float(match is not None and int(match.group(1)) == correct)


def advantages(rewards):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("a group must contain at least two scalar rewards")
    return (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)


def verify():
    responses = ["<answer>5</answer>", "<answer>4</answer>", "5", "<answer>5</answer>"]
    rewards = torch.tensor([answer_reward(r, 5) for r in responses], dtype=torch.float64)
    assert rewards.tolist() == [1., 0., 0., 1.]
    adv = advantages(rewards)
    torch.testing.assert_close(adv, torch.tensor([1., -1., -1., 1.], dtype=rewards.dtype),
                               atol=3e-8, rtol=0)
    assert torch.equal(advantages(torch.ones(4, dtype=rewards.dtype)), torch.zeros(4, dtype=rewards.dtype))
    logits = torch.tensor([.1, .2, -.1, .0], dtype=rewards.dtype, requires_grad=True)
    probability = logits.softmax(0)
    objective = (probability * rewards).sum()
    objective.backward()
    expected = probability.detach() * (rewards - objective.detach())
    torch.testing.assert_close(logits.grad, expected)
    samples = torch.tensor([0, 1, 2, 3])
    old_log_probability = logits.detach().log_softmax(0)[samples]
    ratio = (logits.log_softmax(0)[samples] - old_log_probability).exp()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
    assert abs(surrogate.item()) < 1e-8
    surrogate_gradient = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
    assert surrogate_gradient[0] > 0 and surrogate_gradient[1] < 0
    reference = torch.tensor([.4, .1, .1, .4], dtype=rewards.dtype)
    log_policy = logits.log_softmax(0)
    log_ratio = reference.log() - log_policy
    sampled_kl = log_ratio.exp() - log_ratio - 1
    exact_kl = (probability * (log_policy - reference.log())).sum()
    torch.testing.assert_close((probability * sampled_kl).sum(), exact_kl)
    assert sampled_kl.min() >= 0 and exact_kl >= 0
    assert not answer_reward("<answer>5</answer> extra", 5)
    print("RLVR: rewards [1,0,0,1]; group advantages and exact policy gradient verified")


if __name__ == "__main__":
    verify()

"""Auditable answer rewards, group advantages, and finite-policy gradients."""
import re
import copy
from pathlib import Path
import sys
import torch


def answer_reward(response, correct):
    match = re.fullmatch(r"<answer>\s*(-?\d+)\s*</answer>", response.strip())
    return float(match is not None and int(match.group(1)) == correct)


def advantages(rewards):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("a group must contain at least two scalar rewards")
    return (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)


def gae(rewards, values, gamma=1., lam=.95):
    """One terminal episode; values[-1] is the terminal bootstrap, required to be zero."""
    if rewards.ndim != 1 or values.shape != (rewards.numel() + 1,) or rewards.numel() == 0:
        raise ValueError("need rewards[T] and values[T+1]")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1 or values[-1].item() != 0:
        raise ValueError("gamma/lambda in [0,1] and zero terminal bootstrap required")
    result, running = torch.empty_like(rewards), rewards.new_tensor(0.)
    for t in reversed(range(rewards.numel())):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        result[t] = running
    return result


def verify_rollout():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
    from modern_decoder import ModernDecoder

    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = ModernDecoder(16, width=16, ff_width=32, heads=4, kv_heads=2,
                          head_width=4, max_length=8).double()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004, weight_decay=0.)
    # Prompt [BOS,a,b]; response token 8+a+b. Train/test pairs are disjoint.
    train_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    held_pairs = [(0, 2), (2, 0), (1, 2), (2, 1)]
    prompts = torch.tensor([[7, a, b] for a, b in train_pairs])
    correct = torch.tensor([8 + a + b for a, b in train_pairs])

    @torch.no_grad()
    def exact_reward(pairs):
        inputs = torch.tensor([[7, a, b] for a, b in pairs])
        labels = torch.tensor([8 + a + b for a, b in pairs])
        return model(inputs)[:, -1].softmax(-1).gather(1, labels[:, None]).mean().item()

    initial = exact_reward(held_pairs)
    initial_train = exact_reward(train_pairs)
    initial_weights = copy.deepcopy(model.state_dict())
    logs = []
    for step in range(24):
        with torch.no_grad():
            old_logits = model(prompts)[:, -1]
            responses = torch.multinomial(old_logits.softmax(-1), 24, replacement=True)
            rewards = (responses == correct[:, None]).double()
            adv = torch.stack([advantages(group) for group in rewards]).detach()
            old_logp = old_logits.log_softmax(-1).gather(1, responses).detach()
            ref_logp = reference(prompts)[:, -1].log_softmax(-1)
        for _ in range(2):
            log_policy = model(prompts)[:, -1].log_softmax(-1)
            sampled_logp = log_policy.gather(1, responses)
            ratio = (sampled_logp - old_logp).exp()
            surrogate = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
            exact_kl = (log_policy.exp() * (log_policy - ref_logp)).sum(-1).mean()
            loss = -surrogate + .02 * exact_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        assert not old_logp.requires_grad and not adv.requires_grad
        logs.append(dict(step=step, sampled_reward=rewards.mean().item(), kl=exact_kl.item()))
    assert any(not torch.equal(initial_weights[name], value) for name, value in model.state_dict().items())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, initial_weights[name], rtol=0, atol=0)
    assert all(row["kl"] >= -1e-12 for row in logs)
    final = exact_reward(held_pairs)
    print(f"RLVR LM: training exact reward {initial_train:.6f} -> {exact_reward(train_pairs):.6f}")
    print(f"RLVR LM: 24 rollouts x 4 prompts x 24 responses; held-out exact reward {initial:.6f} -> {final:.6f}")
    print("PASS: real modern LM rollout/reward/two clipped updates; frozen old log-probs/advantages/reference; finite gradients")
    # Reward-model pairwise ranking on frozen, explicitly given response features.
    scorer = torch.nn.Linear(3, 1, bias=False).double()
    chosen, rejected = torch.tensor([[1., 0., 0.]]).double(), torch.tensor([[0., 1., 0.]]).double()
    ranking_loss = torch.nn.functional.softplus(-(scorer(chosen) - scorer(rejected))).mean()
    ranking_loss.backward()
    assert scorer.weight.grad[0, 0] < 0 < scorer.weight.grad[0, 1]
    torch.testing.assert_close(gae(torch.tensor([0., 0., 1.]).double(),
                                   torch.tensor([.2, .3, .4, 0.]).double(), lam=1.),
                               torch.tensor([.8, .7, .6]).double())
    print("PASS: preference reward-model gradient direction; terminal GAE returns [0.8,0.7,0.6]")


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
    verify_rollout()

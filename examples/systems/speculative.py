"""Enumerate exact proposal acceptance and residual compensation."""

import numpy as np


def compensation(p, q):
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("matching nonnegative distributions required")
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("probabilities must sum to one")
    accepted = np.minimum(p, q)
    rejected = np.maximum(p - q, 0)
    mass = rejected.sum()
    residual = rejected / mass if mass > 0 else np.zeros_like(p)
    return accepted, residual, mass


def commit(prefix, draft, accepted_count, replacement):
    if not 0 <= accepted_count <= len(draft):
        raise ValueError("invalid accepted count")
    # Output IDs include the replacement; its KV is computed next iteration.
    return prefix + draft[:accepted_count] + [replacement], len(prefix) + accepted_count


def speculative_step(history, target, proposal, draft_length, rng):
    """Executable categorical sampler; callbacks return conditional distributions."""
    if draft_length < 1:
        raise ValueError("positive draft length required")
    draft, proposal_distributions = [], []
    for _ in range(draft_length):
        q = np.asarray(proposal(history + draft), dtype=np.float64)
        if np.any(q < 0) or not np.isclose(q.sum(), 1):
            raise ValueError("invalid draft distribution")
        draft.append(int(rng.choice(len(q), p=q)))
        proposal_distributions.append(q)
    # A GPU target obtains these distributions from one causal block forward.
    targets = [
        np.asarray(target(history + draft[:i]), dtype=np.float64) for i in range(draft_length + 1)
    ]
    for i, candidate in enumerate(draft):
        p, q = targets[i], proposal_distributions[i]
        _, residual, _ = compensation(p, q)
        if rng.random() >= min(1.0, p[candidate] / q[candidate]):
            replacement = int(rng.choice(len(residual), p=residual))
            return draft[:i] + [replacement], i
    bonus = targets[-1]
    if np.any(bonus < 0) or not np.isclose(bonus.sum(), 1):
        raise ValueError("invalid bonus distribution")
    return draft + [int(rng.choice(len(bonus), p=bonus))], len(draft)


def verify():
    for p, q in [
        ([0.5, 0.3, 0.2], [0.6, 0.1, 0.3]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.5, 0.5], [0.5, 0.5]),
    ]:
        p, q = np.array(p), np.array(q)
        accepted, residual, rejected_mass = compensation(p, q)
        assert np.allclose(accepted + rejected_mass * residual, p)
        assert np.isclose(accepted.sum() + rejected_mass, 1)
    ids, cached = commit([1, 2], [0, 2], 1, 1)
    assert ids == [1, 2, 0, 1] and cached == 3
    ids, cached = commit([1, 2], [0, 2], 2, 2)
    assert ids == [1, 2, 0, 2, 2] and cached == 4

    class Draws:
        def __init__(self, choices, uniforms):
            self.choices, self.uniforms = iter(choices), iter(uniforms)

        def choice(self, count, p):
            result = next(self.choices)
            assert 0 <= result < count and p[result] > 0
            return result

        def random(self):
            return next(self.uniforms)

    def target(history):
        return np.array([0.5, 0.3, 0.2])

    def proposal(history):
        return np.array([0.6, 0.1, 0.3])

    for choices, uniforms, expected, accepted_count in [
        ([0, 2, 1], [0.99], [1], 0),
        ([0, 2, 1], [0.0, 0.99], [0, 1], 1),
        ([0, 2, 2], [0.0, 0.0], [0, 2, 2], 2),
    ]:
        result, count = speculative_step([1], target, proposal, 2, Draws(choices, uniforms))
        assert result == expected and count == accepted_count
    histories = []

    def conditional_target(history):
        histories.append(list(history))
        return np.array([0.5, 0.3, 0.2])

    speculative_step([1], conditional_target, proposal, 2, Draws([0, 2, 2], [0.0, 0.0]))
    assert histories == [[1], [1, 0], [1, 0, 2]]
    # A category absent from q is still reachable through rejection compensation.
    result, count = speculative_step(
        [1, 2],
        lambda history: np.array([0.0, 1.0, 0.0]),
        lambda history: np.array([1.0, 0.0, 0.0]),
        1,
        Draws([0, 1], [0.5]),
    )
    assert result == [1] and count == 0

    def conditional_p(history):
        return np.array([0.5, 0.3, 0.2]) if len(history) == 2 else np.array([0.1, 0.8, 0.1])

    def conditional_q(history):
        return np.array([0.6, 0.1, 0.3]) if len(history) == 2 else np.array([0.2, 0.2, 0.6])

    result, count = speculative_step(
        [1, 2],
        conditional_p,
        conditional_q,
        2,
        Draws([0, 2, 1], [0.0, 0.9]),
    )
    assert result == [0, 1] and count == 1
    a, k = 0.8, 4
    assert np.isclose(sum(a**i for i in range(k + 1)), 3.3616)
    print(
        "speculation: exact marginal, zero support, conditional history, rejection and bonus verified"
    )


if __name__ == "__main__":
    verify()

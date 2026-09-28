"""M1 抽样保证与 M2 预测损失的数值核对。"""
from itertools import product

import numpy as np


def log_softmax(z):
    """沿最后一个类别维计算稳定 log-softmax。"""
    shifted = z - np.max(z, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main():
    rng = np.random.default_rng(42)
    values = np.array([1, 5], dtype=np.float64)
    probabilities = np.array([0.75, 0.25], dtype=np.float64)
    expectation = probabilities @ values
    variance = probabilities @ (values - expectation) ** 2
    assert np.isclose(expectation, 2)
    assert np.isclose(variance, 3)
    records = np.array([[1., 1.], [1., 0.], [1., 0.], [5., 1.]])
    assert np.isclose(records[:3, 1].mean(), 1 / 3)
    assert np.isclose(records[:, 1].mean(), 1 / 2)
    assert np.isclose((records[:3, 1].sum() / records[:, 1].sum()), 1 / 2)
    assert np.allclose(records.mean(axis=0), [2, .5])
    assert np.allclose(records.var(axis=0, ddof=0), [3, .25])
    assert np.isclose(records[:, 0].var(ddof=1), 4)

    all_means = np.array([records[list(indices), 0].mean()
                          for indices in product(range(4), repeat=4)])
    exact = np.mean(np.abs(all_means - expectation) < 1)
    assert np.isclose(exact, 27 / 64)
    assert exact >= 1 - variance / 4
    assert np.isclose(1 - variance / 16, 13 / 16)
    assert 1 - variance / 60 >= .95
    assert 1 - variance / 59 < .95

    reweighted = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    new_mean = reweighted @ records[:, 0]
    new_var = reweighted @ (records[:, 0] - new_mean) ** 2
    assert np.isclose(new_mean, 3)
    assert np.isclose(new_var, 4)
    assert np.isclose(1 - new_var / 16, .75)
    assert 1 - new_var / 40 >= .9
    assert 1 - new_var / 39 < .9
    for batch_size in [1, 4, 16, 64]:
        means = rng.choice(records[:, 0], size=(20000, batch_size)).mean(axis=1)
        measured = means.std(ddof=0)
        theoretical = np.sqrt(3 / batch_size)
        assert np.isclose(measured, theoretical, rtol=0.035)
        print(f"B={batch_size:2d}: std(mean)={measured:.6f}, theory={theoretical:.6f}")
    print(f"B=4: P(|mean-2|<1)={exact:.6f}, Chebyshev lower bound=0.250000")

    # 中点积分核对标准正态面积，以及区间 [-1, 1] 的概率。
    edges = np.linspace(-8, 8, 160001, dtype=np.float64)
    midpoints = (edges[1:] + edges[:-1]) / 2
    density = np.exp(-midpoints ** 2 / 2) / np.sqrt(2 * np.pi)
    widths = np.diff(edges)
    assert np.isclose(np.sum(density * widths), 1, atol=1e-10)
    central = np.abs(midpoints) < 1
    assert np.isclose(np.sum(density[central] * widths[central]), 0.682689492, atol=1e-8)

    labels = np.array([1, 1, 0, 1], dtype=np.float64)
    bernoulli_p = 0.8
    bernoulli_values = np.array([0, 1], dtype=np.float64)
    bernoulli_prob = np.array([1 - bernoulli_p, bernoulli_p], dtype=np.float64)
    assert np.isclose(bernoulli_prob @ bernoulli_values, bernoulli_p)
    assert np.isclose(bernoulli_prob @ (bernoulli_values - bernoulli_p) ** 2, 0.16)
    selected = np.array([0.5, 0.75, 0.9], dtype=np.float64)
    selected_nll = -(labels[:, None] * np.log(selected)
                     + (1 - labels[:, None]) * np.log1p(-selected)).mean(axis=0)
    assert np.allclose(selected_nll, [0.69314718, 0.56233514, 0.65466666])
    mle = labels.mean()
    candidates = np.linspace(0.01, 0.99, 99)
    nll = -(labels[:, None] * np.log(candidates)
            + (1 - labels[:, None]) * np.log1p(-candidates)).mean(axis=0)
    assert np.isclose(candidates[nll.argmin()], mle)
    logits = np.array([-1000, 0, 1000], dtype=np.float64)
    binary_nll = np.maximum(logits, 0) - logits + np.log1p(np.exp(-np.abs(logits)))
    assert np.allclose(binary_nll, [1000, np.log(2), 0])

    z = np.array([[0, np.log(2), np.log(3)], [1000, -1000, 0]], dtype=np.float64)
    target = np.array([2, 1])
    log_p = log_softmax(z)
    loss = -log_p[np.arange(len(target)), target].mean()
    assert np.allclose(np.exp(log_p).sum(axis=1), 1)
    assert np.allclose(log_softmax(z + 1234), log_p)
    assert np.isclose(loss, (np.log(2) + 2000) / 2)
    q = np.array([0.5, 0.5], dtype=np.float64)
    p = np.array([0.75, 0.25], dtype=np.float64)
    entropy = -np.sum(q * np.log(q))
    cross_entropy = -np.sum(q * np.log(p))
    kl = np.sum(q * (np.log(q) - np.log(p)))
    assert np.isclose(cross_entropy, entropy + kl)
    assert kl >= 0
    print(f"H={entropy:.6f}, CE={cross_entropy:.6f}, KL={kl:.6f}")
    print("概率与分布全部验证通过。")


if __name__ == "__main__":
    main()

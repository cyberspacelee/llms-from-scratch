"""数学 01/02 的独立数值验证：uv run --with numpy python foundations/probability.py。"""
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

    u = np.array([1, 2, 3, 4], dtype=np.float64)
    assert np.isclose(u.mean(), 2.5)
    assert np.isclose(u.var(ddof=0), 1.25)
    assert np.isclose(u.var(ddof=1), 5 / 3)
    # 样本中心化平方和的分解：无偏方差推导中的确定性恒等式。
    population_mean = 2.0
    assert np.isclose(np.sum((u - population_mean) ** 2),
                      np.sum((u - u.mean()) ** 2)
                      + len(u) * (u.mean() - population_mean) ** 2)
    X = np.column_stack([u, 2 * u + 1])
    centered = X - X.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / len(X)
    assert np.allclose(cov, [[1.25, 2.5], [2.5, 5]])
    var = X.var(axis=0, keepdims=True, ddof=0)
    normalized = centered / np.sqrt(var + 1e-8)
    assert np.allclose(normalized.mean(axis=0), 0)
    assert np.allclose(normalized.var(axis=0), (var / (var + 1e-8))[0])
    dependent_u = np.array([-1, 0, 1], dtype=np.float64)
    dependent_v = dependent_u ** 2
    assert np.isclose(np.mean(dependent_u * dependent_v)
                      - dependent_u.mean() * dependent_v.mean(), 0)
    for batch_size in [1, 4, 16, 64]:
        means = rng.normal(size=(20000, batch_size)).mean(axis=1)
        measured = means.std(ddof=0)
        theoretical = 1 / np.sqrt(batch_size)
        assert np.isclose(measured, theoretical, rtol=0.025)
        print(f"B={batch_size:2d}: std(mean)={measured:.6f}, theory={theoretical:.6f}")

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

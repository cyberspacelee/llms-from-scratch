"""M1–M5：事件、贝叶斯、抽样、信息量与预测损失的数值核对。"""
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
    positive = records[:, 1].astype(bool)
    source_d = np.arange(4) < 3
    masses = np.full(4, 1 / 4)
    p_a = masses[positive].sum()
    p_d = masses[source_d].sum()
    joint_a_d = masses[positive & source_d].sum()
    assert np.isclose(p_a, 1 / 2) and np.isclose(p_d, 3 / 4)
    assert np.isclose(joint_a_d, 1 / 4)
    assert np.isclose(joint_a_d / p_d, 1 / 3)
    assert np.isclose(joint_a_d / p_a, 1 / 2)
    assert not np.isclose(joint_a_d, p_a * p_d)
    first_two = np.arange(4) < 2
    assert np.isclose(masses[positive & first_two].sum(), p_a * masses[first_two].sum())
    assert masses[source_d & ~source_d].sum() == 0 < p_d * (1 - p_d)
    changed_positive = positive.copy()
    changed_positive[2] = True
    changed_joint = masses[changed_positive & source_d].sum()
    assert np.isclose(changed_joint / p_d, 2 / 3)
    assert not np.isclose(changed_joint, masses[changed_positive].sum() * p_d)
    unequal = np.array([1 / 8, 1 / 8, 1 / 4, 1 / 2])
    assert np.isclose(unequal[positive & source_d].sum() / unequal[source_d].sum(), 1 / 4)

    # 行为来源 D/E，列为正类/非正类；改变先验同时核对全概率与后验。
    conditional = np.array([[1 / 3, 2 / 3], [1., 0.]])
    for prior_d, marginal_positive, posterior_d in [
        (3 / 4, 1 / 2, 1 / 2), (1 / 2, 2 / 3, 1 / 4),
        (3 / 5, 3 / 5, 1 / 3), (2 / 5, 11 / 15, 2 / 11),
    ]:
        joint = conditional * np.array([prior_d, 1 - prior_d])[:, None]
        assert np.isclose(joint.sum(), 1)
        assert np.isclose(joint[:, 0].sum(), marginal_positive)
        posterior = joint[:, 0] / joint[:, 0].sum()
        assert np.isclose(posterior[0], posterior_d)
        assert np.isclose(posterior.sum(), 1)
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
    assert np.isclose(cross_entropy, .8369882167858358)
    assert np.isclose(kl, .14384103622589045)
    reverse_kl = np.sum(p * (np.log(p) - np.log(q)))
    assert np.isclose(reverse_kl, .13081203594113697)
    assert not np.isclose(kl, reverse_kl)
    assert np.isclose(-np.log(p)[1 - records[:, 1].astype(int)].mean(), cross_entropy)
    source_entropy = -np.sum(probabilities * np.log(probabilities))
    entropy_d = -np.sum(conditional[0] * np.log(conditional[0]))
    conditional_entropy = 3 / 4 * entropy_d
    joint = conditional * probabilities[:, None]
    positive_joint = joint[joint > 0]
    joint_entropy = -np.sum(positive_joint * np.log(positive_joint))
    assert np.isclose(conditional_entropy, .4773856262211096)
    assert np.isclose(joint_entropy, source_entropy + conditional_entropy)
    assert conditional_entropy <= entropy
    rare_d_label = np.array([5 / 6, 1 / 6])
    rare_d_entropy = -np.sum(rare_d_label * np.log(rare_d_label))
    assert entropy_d > rare_d_entropy >= entropy_d / 4
    assert np.isclose(-np.log(.75) - np.log(1 / 3), -np.log(.25))
    assert np.isclose(entropy / np.log(2), 1)
    print(f"H={entropy:.6f}, CE={cross_entropy:.6f}, KL={kl:.6f}")
    print(f"H(U|V)={conditional_entropy:.6f}, reverse KL={reverse_kl:.6f}")
    print("事件、贝叶斯、抽样、信息量与预测损失全部验证通过。")


if __name__ == "__main__":
    main()

"""用真实数值曲线生成数学 01/02 的 SVG。"""
import numpy as np
from plot_style import plt, configure, save, GREEN, ORANGE, BLUE, OUTPUT


def main():
    configure()
    rng = np.random.default_rng(42)
    batch_sizes = np.array([1, 4, 16, 64])
    measured = [rng.normal(size=(20000, int(b))).mean(axis=1).std()
                for b in batch_sizes]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    grid = np.linspace(1, 64, 400)
    ax.plot(grid, 1 / np.sqrt(grid), color=GREEN, label=r"理论值 $1/\sqrt{B}$")
    ax.scatter(batch_sizes, measured, color=ORANGE, zorder=3, label="20,000 个独立小批量的测量值")
    ax.set(xlabel="批大小 B", ylabel="样本均值的标准差",
           title="独立采样：批大小乘 4，均值标准差减半", xlim=(0, 66), ylim=(0, 1.08))
    ax.legend()
    ax.grid(alpha=0.15)
    save(fig, "probability-batch.svg")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    x = np.linspace(-6, 6, 1400)
    for (mu, sigma), color in zip([(0, 1), (0, 2), (2, 1)], (GREEN, ORANGE, BLUE)):
        density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        ax.plot(x, density, color=color, label=rf"$\mu={mu},\ \sigma={sigma}$")
    central = np.linspace(-1, 1, 300)
    ax.fill_between(central, np.exp(-central ** 2 / 2) / np.sqrt(2 * np.pi),
                    alpha=0.15, color=GREEN, label=r"$P(-1\leq U\leq1)\approx0.683$")
    ax.set(xlabel="随机变量的取值 u", ylabel="概率密度 p(u)", title="正态分布：均值移动中心，标准差改变宽度")
    ax.legend()
    ax.grid(alpha=0.15)
    save(fig, "probability-normal.svg")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    p = np.linspace(0.001, 0.999, 1000)
    axes[0].plot(p, -np.log(p), color=GREEN)
    axes[0].set(xlabel="真实类别的预测概率 p", ylabel="负对数似然（nat）", title=r"越不相信真实类别，$-\log p$ 越大")
    axes[1].plot(p, -p * np.log(p) - (1 - p) * np.log1p(-p), color=BLUE)
    axes[1].scatter([0.5], [np.log(2)], color=ORANGE, zorder=3)
    axes[1].set(xlabel="伯努利分布的概率 p", ylabel="熵（nat）", title="两种结果等可能时，熵最大", ylim=(0, 0.76))
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, "probability-information.svg")
    print(f"已生成 3 张 SVG：{OUTPUT}")


if __name__ == "__main__":
    main()

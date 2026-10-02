"""Generate the counted float32 address-sector heatmap used in G2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.math.plot_style import configure

configure()
fig, axes = plt.subplots(4, 1, figsize=(9, 7), constrained_layout=True)
for ax, (stride, offset) in zip(axes, [(1, 0), (1, 1), (2, 0), (8, 0)]):
    words = offset + np.arange(32) * stride
    sectors = words // 8
    count = len(np.unique(sectors))
    ax.imshow(sectors[None, :], aspect="auto", cmap="cividis", vmin=0, vmax=31)
    for lane, sector in enumerate(sectors):
        ax.text(
            lane,
            0,
            str(sector),
            ha="center",
            va="center",
            fontsize=8,
            color="black" if sector >= 20 else "white",
        )
    ax.set_yticks([])
    ax.set_xticks(range(0, 32, 4))
    ax.set_ylabel(f"步长 {stride}\n偏移 {offset}", rotation=0, labelpad=35, va="center")
    ax.set_title(f"覆盖 {count} 个 32-byte 段；128 字节请求 / {count * 32} 字节段覆盖", fontsize=11)
axes[-1].set_xlabel("lane 编号（格内数字为地址段号）")
path = Path(__file__).resolve().parents[3] / "site/src/assets/figures/gpu/memory-addresses.png"
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=160)
plt.close(fig)
print(path)

"""Generate the actual G3 5x7 @ 7x6 matrix image; run with NumPy/Matplotlib."""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "math"))
from plot_style import configure, INK


def main():
    configure()
    rng = np.random.default_rng(7)
    a = rng.integers(-2, 3, size=(5, 7))
    b = rng.integers(-2, 3, size=(7, 6))
    c = a @ b
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), layout="constrained")
    for ax, matrix, title, tile in zip(axes, (a, b, c), ("A · 5 × 7", "B · 7 × 6", "C = AB · 5 × 6"), ((0, 0, 4, 4), (0, 0, 4, 4), (0, 0, 4, 4))):
        limit = max(1, np.abs(matrix).max())
        ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        ax.set_title(title)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xlabel("列索引")
        ax.set_ylabel("行索引")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        color="white" if abs(matrix[row, col]) > limit * .55 else INK, fontsize=10)
        x, y, width, height = tile
        ax.add_patch(Rectangle((x - .5, y - .5), width, height, fill=False, edgecolor="#087f70", linewidth=2.5))
    output = Path(__file__).resolve().parents[3] / "site/src/assets/figures/gpu/matrix-tiles.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

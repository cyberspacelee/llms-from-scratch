"""数学函数图的公共样式：中文标签、固定配色与可复现 SVG。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parents[2] / "site/src/assets/figures/math"
GREEN, ORANGE, BLUE = "#087f70", "#c16b36", "#3176b5"
INK, MUTED, PAPER = "#263a34", "#68726d", "#fffefa"


def configure():
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                  "Microsoft YaHei", "Heiti SC", "Arial Unicode MS")
    family = next((name for name in candidates if name in available), None)
    if family is None:
        raise RuntimeError("重建中文图像需要中文字体，例如 Noto Sans CJK SC。已生成 SVG 自带字形，阅读无需安装字体。")
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "figure.facecolor": PAPER,
        "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "legend.frameon": False, "legend.fontsize": 10,
        "lines.linewidth": 2, "axes.unicode_minus": False,
        "svg.fonttype": "path", "svg.hashsalt": "dl-math",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = OUTPUT / name
    fig.savefig(path, format="svg", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)

"""CPU reference calculations; optional --plot rebuilds the actual-value heatmap."""
import argparse
from pathlib import Path
import numpy as np


def softmax(x):
    if x.ndim == 0 or x.shape[-1] == 0 or not np.isfinite(x).all():
        raise ValueError('Softmax needs a nonempty last axis and finite values')
    shifted = x - x.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def plot():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'math'))
    from plot_style import configure, plt, GREEN, BLUE, PAPER, INK
    from matplotlib.colors import LinearSegmentedColormap
    configure()
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    w = np.array([[1, 0, -1], [2, 1, 0]], dtype=np.float64)
    arrays = [x, w, x @ w.T]
    labels = ['X：样本 × 特征', 'W：输出 × 特征', 'Z = X @ W.T']
    fig, axes = plt.subplots(3, 1, figsize=(5, 7), layout='constrained')
    cmap = LinearSegmentedColormap.from_list('framework-array', [BLUE, PAPER, GREEN])
    for ax, values, label in zip(axes, arrays, labels):
        ax.imshow(values, cmap=cmap, vmin=-13, vmax=13, aspect='equal')
        ax.set_title(label)
        ax.set_xticks(range(values.shape[1]))
        ax.set_yticks(range(values.shape[0]))
        for (row, col), value in np.ndenumerate(values):
            ax.text(col, row, f'{value:g}', ha='center', va='center', color=INK)
    target = Path(__file__).resolve().parents[2] / 'site/src/assets/figures/frameworks/numpy-matmul.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(f'Generated {target.name} from actual NumPy arrays')


def main():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    np.testing.assert_array_equal(x + [10, 20, 30], [[11, 22, 33], [14, 25, 36]])
    np.testing.assert_array_equal(x + [[100], [200]], [[101, 102, 103], [204, 205, 206]])
    try:
        x + np.array([100, 200])
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible trailing axes accepted')
    np.testing.assert_allclose(x.mean(axis=0), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(x.mean(axis=1, keepdims=True), [[2], [5]])
    np.testing.assert_allclose(x - x.mean(axis=1, keepdims=True), [[-1, 0, 1], [-1, 0, 1]])
    np.testing.assert_allclose(x.var(axis=1, ddof=0), [2/3, 2/3])
    np.testing.assert_allclose(x.var(axis=1, ddof=1), [1, 1])
    assert x.sum(axis=(0, 1)) == 21
    w = np.array([[1, 0, -1], [2, 1, 0]], dtype=np.float64)
    z = x @ w.T
    expected = np.array([[sum(x[b, k] * w[m, k] for k in range(3))
                          for m in range(2)] for b in range(2)])
    np.testing.assert_array_equal(z, expected)
    np.testing.assert_array_equal(z, [[-2, 4], [-2, 13]])
    np.testing.assert_array_equal(z, np.einsum('bd,md->bm', x, w))
    np.testing.assert_array_equal(x * w, [[1, 0, -3], [8, 5, 0]])
    np.testing.assert_array_equal(np.where(x > 3, x, -1), [[-1, -1, -1], [4, 5, 6]])
    np.testing.assert_array_equal(np.clip(x, 2, 4), [[2, 2, 3], [4, 4, 4]])
    indices = x.argmax(axis=1, keepdims=True)
    assert indices.dtype.kind in 'iu'
    np.testing.assert_array_equal(np.take_along_axis(x, indices, axis=1), [[3], [6]])
    assert np.concatenate([x, x], axis=0).shape == (4, 3)
    assert np.stack([x, x], axis=0).shape == (2, 2, 3)
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(2, 3))
    np.testing.assert_array_equal(sample, np.random.default_rng(7).normal(size=(2, 3)))
    logits = np.array([[1000, 1001, 1002], [-1000, -1000, -1000]], dtype=np.float64)
    probabilities = softmax(logits)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])
    np.testing.assert_allclose(probabilities[1], [1/3] * 3)
    np.testing.assert_allclose(probabilities, softmax(logits + 12345))
    for invalid in (np.array(1.0), np.empty((2, 0)), np.array([[-np.inf, -np.inf]])):
        try:
            softmax(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid softmax contract accepted')
    print(f'NumPy {np.__version__}: broadcasting, reductions, contraction, selection and softmax passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main()
    if args.plot:
        plot()

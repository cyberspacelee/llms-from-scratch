"""Spectral decomposition, truncated SVD, conditioning and floating-point checks."""

import numpy as np


def verify():
    close = np.testing.assert_allclose
    rotation = np.array([[.8, -.6], [.6, .8]])
    matrix = rotation @ np.diag([1., 9.]) @ rotation.T
    values, vectors = np.linalg.eigh(matrix)
    close(values, [1., 9.])
    close(vectors.T @ vectors, np.eye(2), atol=1e-14)
    close(vectors @ np.diag(values) @ vectors.T, matrix)
    for eta, stable in [(.1, True), (.25, False)]:
        state = vectors[:, -1].copy()
        initial = np.linalg.norm(state)
        for _ in range(20):
            state -= eta * (matrix @ state)
        assert (np.linalg.norm(state) < initial) == stable
    rectangular = np.array([[3., 0., 0.], [0., 1., 0.]])
    u, singular, vt = np.linalg.svd(rectangular, full_matrices=False)
    close((u * singular) @ vt, rectangular)
    rank_one = (u[:, :1] * singular[:1]) @ vt[:1]
    close(np.linalg.norm(rectangular - rank_one, "fro") ** 2, (singular[1:] ** 2).sum())
    assert np.linalg.matrix_rank(rank_one) == 1
    ill = np.diag([1., 1e-4])
    rhs = np.array([1., 0.])
    delta = np.array([0., 1e-6])
    solution = np.linalg.solve(ill, rhs)
    perturbed = np.linalg.solve(ill, rhs + delta)
    relative_input = np.linalg.norm(delta) / np.linalg.norm(rhs)
    relative_output = np.linalg.norm(perturbed - solution) / np.linalg.norm(solution)
    close(np.linalg.cond(ill, 2), 10000.)
    close(relative_output, .01)
    assert relative_output <= np.linalg.cond(ill, 2) * relative_input * (1 + 1e-12)
    a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.)
    assert (a + b) + c == 1 and a + (b + c) == 0
    logits = np.array([1000., 999.])
    log_partition = np.logaddexp.reduce(logits)
    close(np.exp(logits - log_partition).sum(), 1.)
    print("PASS: eigh reconstruction, curvature step bound, SVD rank-one residual")
    print("PASS: condition number 10000 amplifies RHS error to 1%; float32 nonassociativity; stable logsumexp")


if __name__ == "__main__":
    verify()

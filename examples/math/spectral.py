"""Spectral decomposition, truncated SVD, conditioning and floating-point checks."""

import numpy as np


def verify():
    close = np.testing.assert_allclose
    # region eigen
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    matrix = rotation @ np.diag([1.0, 9.0]) @ rotation.T
    values, vectors = np.linalg.eigh(matrix)
    close(values, [1.0, 9.0])
    close(vectors.T @ vectors, np.eye(2), atol=1e-14)
    close(vectors @ np.diag(values) @ vectors.T, matrix)
    x = np.array([1.0, 0.0])
    y = rotation.T @ x
    close(rotation @ (np.array([1.0, 9.0]) * y), matrix @ x)
    close(matrix @ x, [3.88, -3.84])
    for eta, stable in [(0.1, True), (0.25, False)]:
        state = vectors[:, -1].copy()
        initial = np.linalg.norm(state)
        for _ in range(20):
            state -= eta * (matrix @ state)
        assert (np.linalg.norm(state) < initial) == stable
    # endregion eigen
    # region svd
    # M12 rotates input directions; output directions are the coordinate basis.
    weight = np.diag([1.0, 3.0]) @ rotation.T
    u_w, s_w, vt_w = np.linalg.svd(weight, full_matrices=False)
    close(s_w, [3.0, 1.0])
    close((u_w * s_w) @ vt_w, weight)
    truncated = (u_w[:, :1] * s_w[:1]) @ vt_w[:1]
    close(np.linalg.norm(weight - truncated, "fro"), 1.0)
    close(np.linalg.norm((weight - truncated) @ rotation[:, 0]), 1.0)
    close((weight - truncated) @ rotation[:, 1], np.zeros(2), atol=1e-14)
    close(np.linalg.cond(weight, 2), 3.0)
    rhs_w = np.array([0.0, 1.0])
    perturb_w = np.array([1e-3, 0.0])
    solution_w = np.linalg.solve(weight, rhs_w)
    close(solution_w, rotation[:, 1] / 3)
    relative_w = np.linalg.norm(np.linalg.solve(weight, rhs_w + perturb_w) - solution_w)
    close(relative_w / np.linalg.norm(solution_w), 3e-3)
    # endregion svd
    rectangular = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    u, singular, vt = np.linalg.svd(rectangular, full_matrices=False)
    close((u * singular) @ vt, rectangular)
    rank_one = (u[:, :1] * singular[:1]) @ vt[:1]
    close(np.linalg.norm(rectangular - rank_one, "fro") ** 2, (singular[1:] ** 2).sum())
    assert np.linalg.matrix_rank(rank_one) == 1
    ill = np.diag([1.0, 1e-4])
    rhs = np.array([1.0, 0.0])
    delta = np.array([0.0, 1e-6])
    solution = np.linalg.solve(ill, rhs)
    perturbed = np.linalg.solve(ill, rhs + delta)
    relative_input = np.linalg.norm(delta) / np.linalg.norm(rhs)
    relative_output = np.linalg.norm(perturbed - solution) / np.linalg.norm(solution)
    close(np.linalg.cond(ill, 2), 10000.0)
    close(relative_output, 0.01)
    assert relative_output <= np.linalg.cond(ill, 2) * relative_input * (1 + 1e-12)
    a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
    assert (a + b) + c == 1 and a + (b + c) == 0
    logits = np.array([1000.0, 999.0])
    log_partition = np.logaddexp.reduce(logits)
    close(np.exp(logits - log_partition).sum(), 1.0)
    print("PASS: eigh reconstruction, curvature step bound, SVD rank-one residual")
    print(
        "PASS: condition number 10000 amplifies RHS error to 1%; float32 nonassociativity; stable logsumexp"
    )


if __name__ == "__main__":
    verify()

"""Four-input recurrence: serial, expansion, vector state, and affine scan."""

import torch


def compose(left, right):
    """Apply the left affine update before the right one."""
    a_left, b_left = left
    a_right, b_right = right
    return a_right @ a_left, a_right @ b_left + b_right


def verify():
    dtype = torch.float64
    inputs = torch.tensor([1.0, 2.0, -1.0, 3.0], dtype=dtype)

    state = torch.tensor(0.0, dtype=dtype)
    fixed = []
    for value in inputs:
        state = 0.5 * state + value
        fixed.append(state)
    fixed = torch.stack(fixed)
    torch.testing.assert_close(fixed, torch.tensor([1.0, 2.5, 0.25, 3.125], dtype=dtype))
    expanded = torch.stack(
        [sum(0.5 ** (t - j) * inputs[j] for j in range(t + 1)) for t in range(len(inputs))]
    )
    torch.testing.assert_close(fixed, expanded)

    a = torch.diag(torch.tensor([0.5, 1.0], dtype=dtype))
    b = torch.ones(2, dtype=dtype)
    vector_state = torch.zeros(2, dtype=dtype)
    for value in inputs:
        vector_state = a @ vector_state + b * value
    torch.testing.assert_close(vector_state, torch.tensor([3.125, 5.0], dtype=dtype))
    torch.testing.assert_close(torch.tensor([1.0, 0.0], dtype=dtype) @ vector_state, fixed[-1])
    torch.testing.assert_close(torch.tensor([0.0, 1.0], dtype=dtype) @ vector_state, inputs.sum())

    updates = [(0.5, float(value)) if value >= 0 else (1.0, 0.0) for value in inputs]
    selected = torch.tensor(0.0, dtype=dtype)
    combined = (torch.eye(1, dtype=dtype), torch.zeros(1, dtype=dtype))
    states = []
    for retain, write in updates:
        selected = retain * selected + write
        combined = compose(
            combined,
            (
                torch.tensor([[retain]], dtype=dtype),
                torch.tensor([write], dtype=dtype),
            ),
        )
        states.append(selected)
        torch.testing.assert_close(selected, combined[1][0])
    torch.testing.assert_close(
        torch.stack(states), torch.tensor([1.0, 2.5, 2.5, 4.25], dtype=dtype)
    )
    torch.testing.assert_close(combined[0], torch.tensor([[0.125]], dtype=dtype))
    assert (
        compose(
            (torch.tensor([[0.5]], dtype=dtype), torch.tensor([1.0], dtype=dtype)),
            (torch.tensor([[0.5]], dtype=dtype), torch.tensor([2.0], dtype=dtype)),
        )[1].item()
        != compose(
            (torch.tensor([[0.5]], dtype=dtype), torch.tensor([2.0], dtype=dtype)),
            (torch.tensor([[0.5]], dtype=dtype), torch.tensor([1.0], dtype=dtype)),
        )[1].item()
    )
    print("fixed: 1, 2.5, 0.25, 3.125; selected: 1, 2.5, 2.5, 4.25")


if __name__ == "__main__":
    verify()

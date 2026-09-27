"""Serial recurrence, explicit expansion, and associative affine composition."""
import torch


def compose(left, right):
    """Apply left affine map first, then right."""
    a_left, b_left = left
    a_right, b_right = right
    return a_right @ a_left, a_right @ b_left + b_right


def verify():
    dtype = torch.float64
    inputs = torch.tensor([1., 2., -1., 3.], dtype=dtype)
    state = torch.tensor(0., dtype=dtype)
    outputs = []
    for value in inputs:
        state = .5 * state + value
        outputs.append(state.clone())
    expected = torch.tensor([1., 2.5, .25, 3.125], dtype=dtype)
    torch.testing.assert_close(torch.stack(outputs), expected)
    expanded = torch.stack([sum(.5 ** (t - j) * inputs[j] for j in range(t + 1))
                            for t in range(len(inputs))])
    torch.testing.assert_close(expanded, expected)
    torch.manual_seed(13)
    maps = [(torch.eye(2, dtype=dtype) * (.2 + .15 * i), torch.randn(2, dtype=dtype))
            for i in range(4)]
    serial = torch.zeros(2, dtype=dtype)
    combined = (torch.eye(2, dtype=dtype), serial.clone())
    for a, b in maps:
        serial = a @ serial + b
        combined = compose(combined, (a, b))
    torch.testing.assert_close(serial, combined[1])
    lhs = compose(compose(maps[0], maps[1]), maps[2])
    rhs = compose(maps[0], compose(maps[1], maps[2]))
    for a, b in zip(lhs, rhs):
        torch.testing.assert_close(a, b)
    print("SSM: serial = expansion = composed affine maps; scalar final state 3.125")


if __name__ == "__main__":
    verify()

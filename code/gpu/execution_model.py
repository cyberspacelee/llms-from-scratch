"""CPU checks for the G1 ownership example; this does not emulate GPU timing."""

from collections import Counter


def owners(n: int, threads: int) -> Counter[int]:
    blocks = (n + threads - 1) // threads
    return Counter(
        block * threads + thread
        for block in range(blocks)
        for thread in range(threads)
        if block * threads + thread < n
    )


def stride_owners(n: int, threads: int, blocks: int) -> Counter[int]:
    stride = threads * blocks
    return Counter(i for start in range(stride) for i in range(start, n, stride))


def demo() -> None:
    for n in (0, 1, 100, 130):
        for threads in (16, 64):
            assert owners(n, threads) == Counter({i: 1 for i in range(n)})

    assert divmod(99, 64) == (1, 35)
    assert stride_owners(100, 16, 2) == Counter({i: 1 for i in range(100)})
    assert list(range(3, 100, 32)) == [3, 35, 67, 99]
    assert {thread % 32 for thread in range(32, 64) if 64 + thread < 100} == set(range(4))
    assert divmod(129, 64) == (2, 1)

    rows, cols, tx, ty = 5, 7, 4, 2
    positions = []
    for by in range((rows + ty - 1) // ty):
        for bx in range((cols + tx - 1) // tx):
            for y in range(ty):
                for x in range(tx):
                    row, col = by * ty + y, bx * tx + x
                    if row < rows and col < cols:
                        positions.append(row * cols + col)
    assert Counter(positions) == Counter({i: 1 for i in range(rows * cols)})
    assert (0 * cols + 7) < rows * cols  # A linear-only guard admits invalid column 7.
    print("execution_model: unique coverage, tail warp and 2D bounds passed")


if __name__ == "__main__":
    demo()

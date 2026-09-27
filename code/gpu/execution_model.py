"""CPU checks of CUDA ownership and NVIDIA 32-lane numbering; no GPU timing."""

from collections import Counter


def coverage(n, threads, blocks=None):
    assert n >= 0 and threads > 0
    blocks = (n + threads - 1) // threads if blocks is None else blocks
    assert blocks >= 0 and (n == 0 or blocks > 0)
    stride = threads * blocks
    return Counter(i for start in range(stride) for i in range(start, n, stride))


def demo():
    for n in (0, 1, 31, 32, 33, 100, 1003):
        for threads in (16, 32, 64, 128):
            expected = Counter({i: 1 for i in range(n)})
            assert coverage(n, threads) == expected
            assert coverage(n, threads, 2) == expected
    assert 2 * 64 - 100 == 28
    bx, by, bz = 8, 8, 2
    linear = [x + bx * (y + by * z)
              for z in range(bz) for y in range(by) for x in range(bx)]
    assert linear == list(range(bx * by * bz))
    assert (32 % bx, (32 // bx) % by, 32 // (bx * by)) == (0, 4, 0)
    assert (32 // 32, 32 % 32) == (1, 0)
    rows, cols, tx, ty = 5, 7, 4, 2
    addresses = []
    for block_y in range((rows + ty - 1) // ty):
        for block_x in range((cols + tx - 1) // tx):
            for y in range(ty):
                for x in range(tx):
                    row, col = block_y * ty + y, block_x * tx + x
                    if row < rows and col < cols:
                        addresses.append(row * cols + col)
    assert Counter(addresses) == Counter({i: 1 for i in range(rows * cols)})
    for active in range(1, 33):
        for threshold in range(33):
            a = {lane for lane in range(active) if lane < threshold}
            b = {lane for lane in range(active) if lane >= threshold}
            assert not a & b and a | b == set(range(active))
    print("execution_model: unique coverage, 2D/3D indexing and branch masks passed")


if __name__ == "__main__":
    demo()

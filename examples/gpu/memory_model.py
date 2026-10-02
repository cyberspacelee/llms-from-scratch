"""CPU checks for float32 sector coverage, shared-memory banks and reductions."""

import math
from collections import defaultdict


def sectors(stride=1, offset=0, lanes=32):
    """Aligned base; offset and stride are measured in float32 elements."""
    assert stride >= 0 and offset >= 0 and 0 <= lanes <= 32
    return {(4 * (offset + lane * stride)) // 32 for lane in range(lanes)}


def bank_requests(words):
    """Distinct 32-bit read addresses per bank; equal addresses broadcast."""
    banks = defaultdict(set)
    for word in words:
        banks[word % 32].add(word)
    return max(map(len, banks.values()), default=0)


def block_sum(values, block_size=256):
    assert block_size > 0 and block_size & (block_size - 1) == 0
    partials = []
    for start in range(0, len(values), block_size):
        shared = list(values[start : start + block_size])
        shared += [0.0] * (block_size - len(shared))
        step = block_size // 2
        while step:
            previous = shared.copy()
            for lane in range(step):
                shared[lane] = previous[lane] + previous[lane + step]
            step //= 2
        partials.append(shared[0])
    return math.fsum(partials)


def transpose_indices(rows, cols):
    """Mirror the CUDA kernel's 32x8 thread layout and four load/store rounds."""
    output = [None] * (rows * cols)
    writes = [0] * (rows * cols)
    for by in range((rows + 31) // 32):
        for bx in range((cols + 31) // 32):
            tile = [[0] * 33 for _ in range(32)]
            for ty in range(8):
                for tx in range(32):
                    for j in range(0, 32, 8):
                        row, col = by * 32 + ty + j, bx * 32 + tx
                        if row < rows and col < cols:
                            tile[ty + j][tx] = row * cols + col
            for ty in range(8):
                for tx in range(32):
                    for j in range(0, 32, 8):
                        out_row, out_col = bx * 32 + ty + j, by * 32 + tx
                        if out_row < cols and out_col < rows:
                            index = out_row * rows + out_col
                            output[index] = tile[tx][ty + j]
                            writes[index] += 1
    assert all(count == 1 for count in writes)
    assert output == [row * cols + col for col in range(cols) for row in range(rows)]
    return output


def main():
    assert [len(sectors(s, o)) for s, o in [(1, 0), (1, 1), (2, 0), (8, 0)]] == [4, 5, 8, 32]
    assert sectors(lanes=0) == set()
    assert bank_requests([lane * 32 for lane in range(32)]) == 32
    assert bank_requests([lane * 33 for lane in range(32)]) == 1
    assert bank_requests([7] * 32) == 1
    assert bank_requests([0, 0, 32, 32]) == 2
    for rows, cols in [(1, 1), (32, 32), (35, 67)]:
        transpose_indices(rows, cols)
    assert transpose_indices(35, 67)[66 * 35 + 34] == 34 * 67 + 66
    for length in [0, 1, 31, 256, 257, 1003]:
        values = [float((i % 17) - 8) / 8 for i in range(length)]
        assert block_sum(values) == math.fsum(values)
    print("memory_model: sectors, banks, transpose and reductions OK")


if __name__ == "__main__":
    main()

"""CPU checks for float32 sector coverage, shared-memory banks and reductions."""
from collections import defaultdict
import math


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
        shared = list(values[start:start + block_size])
        shared += [0.0] * (block_size - len(shared))
        step = block_size // 2
        while step:
            previous = shared.copy()
            for lane in range(step):
                shared[lane] = previous[lane] + previous[lane + step]
            step //= 2
        partials.append(shared[0])
    return math.fsum(partials)


def main():
    assert [len(sectors(s, o)) for s, o in [(1, 0), (1, 1), (2, 0), (8, 0)]] == [4, 5, 8, 32]
    assert sectors(lanes=0) == set()
    assert bank_requests([lane * 32 for lane in range(32)]) == 32
    assert bank_requests([lane * 33 for lane in range(32)]) == 1
    assert bank_requests([7] * 32) == 1
    assert bank_requests([0, 0, 32, 32]) == 2
    for length in [0, 1, 31, 256, 257, 1003]:
        values = [float((i % 17) - 8) / 8 for i in range(length)]
        assert block_sum(values) == math.fsum(values)
    print("memory_model: sectors 4/5/8/32, banks 32/1/broadcast, reductions OK")


if __name__ == "__main__":
    main()

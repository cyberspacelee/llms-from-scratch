"""CPU-only checks of teaching resource ceilings, stream dependencies and byte accounting."""

from math import ceil, isfinite


def resident_blocks(
    threads,
    registers,
    shared_kib,
    *,
    warps=64,
    register_capacity=65536,
    shared_capacity_kib=96,
    block_capacity=32,
):
    values = (
        threads,
        registers,
        shared_kib,
        warps,
        register_capacity,
        shared_capacity_kib,
        block_capacity,
    )
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Resource counts must be nonnegative integers")
    if min(threads, registers, warps, register_capacity, shared_capacity_kib, block_capacity) < 1:
        raise ValueError("Threads, registers and capacities must be positive")
    # ponytail: no allocation rounding; query the CUDA occupancy API for a real device.
    warp_count = ceil(threads / 32)
    bounds = (
        warps // warp_count,
        register_capacity // (threads * registers),
        shared_capacity_kib // shared_kib if shared_kib else block_capacity,
        block_capacity,
    )
    blocks = min(bounds)
    return blocks, blocks * warp_count / warps


def stream_schedule(producer, independent, consumer, *, two_streams=True, wait_event=True):
    if any(not isfinite(value) or value <= 0 for value in (producer, independent, consumer)):
        raise ValueError("Durations must be finite and positive")
    # ponytail: ideal concurrent resources; GPU traces determine actual overlap.
    independent_start = 0 if two_streams else producer
    consumer_start = max(
        independent_start + independent, producer if two_streams and wait_event else 0
    )
    finish = max(producer, consumer_start + consumer)
    return consumer_start, finish, not two_streams or wait_event


def effective_bandwidth(read_bytes, written_bytes, seconds):
    if min(read_bytes, written_bytes) < 0 or not isfinite(seconds) or seconds <= 0:
        raise ValueError("Byte counts must be nonnegative and time must be finite and positive")
    return (read_bytes + written_bytes) / seconds / 1e9


def main():
    # Two float32 tiles: shared bytes = 2 * tile**2 * 4.
    assert resident_blocks(16 * 16, 32, 2) == (8, 1.0)
    assert resident_blocks(32 * 32, 32, 8) == (2, 1.0)
    assert resident_blocks(256, 32, 16) == (6, 0.75)
    assert resident_blocks(256, 64, 16) == (4, 0.5)
    assert resident_blocks(256, 32, 32) == (3, 0.375)
    assert resident_blocks(256, 32, 0) == (8, 1.0)
    assert resident_blocks(256, 32, 128) == (0, 0.0)
    assert resident_blocks(33, 32, 0)[0] == 32
    assert stream_schedule(5, 3, 2, two_streams=False) == (8, 10, True)
    assert stream_schedule(5, 3, 2) == (5, 7, True)
    assert stream_schedule(5, 3, 2, wait_event=False) == (3, 5, False)
    assert stream_schedule(3, 5, 2, wait_event=False) == (5, 7, False)
    assert effective_bandwidth(8_000_000, 4_000_000, 0.0001) == 120.0
    for invalid in (0, float("nan")):
        try:
            effective_bandwidth(0, 0, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid measurement accepted")
    print("Tile 16: 256 threads, shared 2 KiB, 8 blocks, 64/64 warps, occupancy 100%")
    print("Tile 32: 1024 threads, shared 8 KiB, 2 blocks, 64/64 warps, occupancy 100%")
    print("Additional shared-memory case: 16 KiB, 6 blocks, occupancy 75%")
    print("Ideal schedules: serial 10; event dependency 7; missing dependency is unsafe")
    print("Resource/dependency/byte checks passed; no GPU timing was measured.")


if __name__ == "__main__":
    main()

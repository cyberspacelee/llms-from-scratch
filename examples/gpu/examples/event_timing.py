"""Optional CUDA event measurement of PyTorch vector addition; requires a real NVIDIA GPU."""

import argparse
import json
import statistics
from time import perf_counter


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=positive_int, default=1_000_003)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--warmup", type=positive_int, default=10)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install CUDA-enabled PyTorch before running this GPU example.") from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; CPU timing cannot substitute for this experiment.")

    torch.manual_seed(7)
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    x = torch.randn(args.elements, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    reference = x.cpu() + y.cpu()
    for _ in range(args.warmup):
        torch.add(x, y, out=out)
    torch.cuda.synchronize(device)

    stream = torch.cuda.current_stream(device)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.repeats)
    ]
    # Initialize the underlying CUDA events before measuring repeated submissions.
    for start, stop in events:
        start.record(stream)
        stop.record(stream)
    events[-1][1].synchronize()
    wall_start = perf_counter()
    for start, stop in events:
        start.record(stream)
        torch.add(x, y, out=out)
        stop.record(stream)
    events[-1][1].synchronize()
    submission_and_wait_ms = (perf_counter() - wall_start) * 1000
    times = [start.elapsed_time(stop) for start, stop in events]
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-6, atol=1e-6)
    median_ms = statistics.median(times)
    if median_ms <= 0:
        raise RuntimeError("Event resolution was insufficient; increase --elements.")
    logical_bytes = 3 * args.elements * x.element_size()
    print(
        json.dumps(
            {
                "gpu": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sm_count": properties.multi_processor_count,
                "pytorch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "elements": args.elements,
                "dtype": str(x.dtype),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "event_median_ms": median_ms,
                "event_min_ms": min(times),
                "event_max_ms": max(times),
                "submission_and_wait_batch_ms": submission_and_wait_ms,
                "logical_bytes_per_add": logical_bytes,
                "effective_GB_per_s": logical_bytes / (median_ms / 1000) / 1e9,
                "scope": "preallocated repeated add, no H2D/D2H; cache reuse possible; not measured HBM traffic",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

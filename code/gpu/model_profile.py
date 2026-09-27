"""Profile the shared modern LM; CPU correctness by default, optional CUDA timing."""

import argparse
import copy
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from modern_decoder import ModernDecoder


def profile_model(device="cpu", trace=None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires CUDA-enabled PyTorch and an NVIDIA GPU")
    torch.set_num_threads(1)
    torch.manual_seed(61)
    dtype = torch.float64 if device == "cpu" else torch.float32
    model = ModernDecoder(64, width=32, ff_width=64, max_length=64).to(device=device, dtype=dtype).eval()
    ids = torch.randint(0, 64, (2, 32), device=device)
    outputs, gradients = [], []
    for backend in ("manual", "sdpa"):
        probe = copy.deepcopy(model)
        output = probe(ids, backend=backend)
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append([p.grad.detach() for p in probe.parameters()])
    tolerance = 1e-10 if device == "cpu" else 2e-5
    torch.testing.assert_close(outputs[0], outputs[1], atol=tolerance, rtol=tolerance)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        for backend in ("manual", "sdpa"):
            for _ in range(3):
                model(ids, backend=backend)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            for backend in ("manual", "sdpa"):
                with torch.profiler.record_function("modern_lm_" + backend):
                    model(ids, backend=backend)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
        if trace:
            profiler.export_chrome_trace(str(trace))
        if device == "cuda":
            for backend in ("manual", "sdpa"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    model(ids, backend=backend)
                end.record()
                end.synchronize()
                print(f"{backend}: CUDA event mean={start.elapsed_time(end)/30:.6f} ms; "
                      f"peak incremental bytes={torch.cuda.max_memory_allocated()-baseline}")
    print(f"PASS: {device} shared LM manual/SDPA logits and parameter gradients agree; timing is local to this shape")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    profile_model(args.device, args.trace)

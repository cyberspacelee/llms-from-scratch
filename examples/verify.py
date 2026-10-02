"""Run every CPU curriculum experiment from one environment (GPU kernels are optional)."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    modules = [
        f"examples.math.{name}"
        for name in (
            "probability",
            "linear_algebra",
            "calculus",
            "backprop",
            "neural_network",
            "spectral",
        )
    ]
    for track in (
        "frameworks",
        "principles",
        "training",
        "post_training",
        "gpu",
        "systems",
        "advanced",
    ):
        modules.extend(
            f"examples.{track}.{path.stem}"
            for path in sorted((root / track).glob("*.py"))
            if path.stem != "__init__"
        )
    failures = []
    for module in modules:
        print(f"VERIFY {module}", flush=True)
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode:
            failures.append(module)
    if failures:
        raise SystemExit("Failed: " + ", ".join(failures))
    print(
        f"PASS: {len(modules)} CPU experiments; CUDA/Triton kernels require separate hardware checks"
    )


if __name__ == "__main__":
    main()

"""Client event metrics with explicit single-token and multi-token policies."""
import numpy as np


def percentile_nearest_rank(values, probability):
    if not values or not 0 <= probability <= 1 or not np.isfinite(values).all():
        raise ValueError("finite samples and probability in [0,1] required")
    ordered = sorted(values)
    return ordered[max(0, int(np.ceil(probability * len(ordered))) - 1)]


def request_metrics(arrival, events):
    if not events or any(count <= 0 for _, count in events):
        raise ValueError("nonempty positive-token events required")
    times = np.array([time for time, _ in events], dtype=np.float64)
    if times[0] < arrival or np.any(np.diff(times) < 0):
        raise ValueError("events must follow arrival monotonically")
    count = sum(count for _, count in events)
    return {"ttft": times[0] - arrival, "itl": np.diff(times),
            "tpot": (times[-1] - times[0]) / (count - 1) if count > 1 else None,
            "latency": times[-1] - arrival, "tokens": count}


def verify():
    a = request_metrics(0., [(1., 1), (1.2, 1), (1.5, 1)])
    assert a["ttft"] == 1 and a["tpot"] == .25
    assert np.allclose(a["itl"], [.2, .3])
    b = request_metrics(.1, [(.6, 1)])
    assert b["tpot"] is None and len(b["itl"]) == 0
    c = request_metrics(.2, [(.8, 1), (1.1, 3)])
    assert np.isclose(c["tpot"], .1) and np.allclose(c["itl"], [.3])
    assert sum(r["tokens"] for r in (a, b, c)) / 1.5 == 8 / 1.5
    assert sum(r["tokens"] for r in (a, b, c) if r["ttft"] <= .7) / 1.5 == 5 / 1.5
    ttfts = [r["ttft"] for r in (a, b, c)]
    assert np.isclose(percentile_nearest_rank(ttfts, .5), .6)
    assert percentile_nearest_rank(ttfts, .95) == 1
    print("serving: TTFT, event ITL, TPOT, one-token policy and goodput verified")


if __name__ == "__main__":
    verify()

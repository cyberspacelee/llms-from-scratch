"""Recalculate the client-visible metrics in S10 from one request trace."""

from math import ceil, isfinite


TRACE = [
    # id, prompt tokens, client arrival, queue exit, [(stream time, tokens)], terminal time, status
    ("A", 4, 0.0, 0.1, [(1.0, 1), (1.2, 1), (1.5, 1)], 1.5, "completed"),
    ("B", 12, 0.1, 0.4, [(0.6, 1)], 0.6, "completed"),
    ("C", 2, 0.2, 0.5, [(0.8, 1), (1.1, 3)], 1.1, "completed"),
    ("D", 6, 0.3, None, [], 0.9, "cancelled"),
]


def nearest_rank(values, probability):
    if not values or not 0 <= probability <= 1 or not all(map(isfinite, values)):
        raise ValueError("finite samples and probability in [0,1] required")
    return sorted(values)[max(0, ceil(probability * len(values)) - 1)]


def measure(request):
    name, prompt, arrival, queue_exit, events, terminal, status = request
    times = [time for time, _ in events]
    if (prompt <= 0 or not all(map(isfinite, [arrival, terminal]))
            or terminal < arrival or status not in {"completed", "cancelled"}
            or (queue_exit is not None and (not isfinite(queue_exit) or not arrival <= queue_exit <= terminal))
            or any(count <= 0 or not isfinite(time) for time, count in events)
            or times != sorted(times) or (times and (times[0] < arrival or times[-1] > terminal))
            or (status == "completed" and (not events or times[-1] != terminal))):
        raise ValueError(f"invalid request trace: {name}")
    count = sum(count for _, count in events)
    return {
        "name": name, "status": status, "queue": None if queue_exit is None else queue_exit - arrival,
        "ttft": None if not events else times[0] - arrival,
        "itl": [right - left for left, right in zip(times, times[1:])],
        "tpot": (times[-1] - times[0]) / (count - 1) if count > 1 else None,
        "latency": terminal - arrival, "tokens": count,
    }


def verify():
    rows = [measure(request) for request in TRACE]
    a, b, c, d = rows
    assert [round(row["queue"], 2) for row in rows[:3]] == [.1, .3, .3]
    assert [round(row["ttft"], 2) for row in rows[:3]] == [1.0, .5, .6]
    assert [round(row["latency"], 2) for row in rows] == [1.5, .5, .9, .6]
    assert [row["tokens"] for row in rows] == [3, 1, 4, 0]
    assert [round(gap, 2) for row in rows for gap in row["itl"]] == [.2, .3, .3]
    assert [round(row["tpot"], 2) if row["tpot"] is not None else None for row in rows] == [.25, None, .1, None]
    assert round((a["tpot"] + c["tpot"]) / 2, 3) == .175
    assert round(((1.5 - 1.0) + (1.1 - .8)) / ((3 - 1) + (4 - 1)), 2) == .16
    assert round(nearest_rank([row["ttft"] for row in rows[:3]], .5), 2) == .6
    assert nearest_rank([row["ttft"] for row in rows[:3]], .95) == 1
    assert sum(row["tokens"] for row in rows) / 1.5 == 8 / 1.5
    good = [row for row in rows if row["status"] == "completed" and row["ttft"] <= .7
            and (row["tpot"] is None or row["tpot"] <= .2) and row["latency"] <= 1]
    assert [row["name"] for row in good] == ["B", "C"]
    assert sum(row["tokens"] for row in good) / 1.5 == 5 / 1.5
    assert d["ttft"] is None and d["queue"] is None and round(d["latency"], 2) == .6
    interrupted = measure(("E", 3, .4, .5, [(.7, 1)], 1.0, "cancelled"))
    assert interrupted["tokens"] == 1 and interrupted["latency"] == .6
    for invalid in [TRACE[0][:4] + ([(.9, 0)], 1.5, "completed"),
                    TRACE[0][:4] + ([(1.2, 1), (1., 1)], 1.5, "completed")]:
        try:
            measure(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid event trace accepted")
    print("S10: request trace, cancellation, percentiles and SLO goodput verified")


if __name__ == "__main__":
    verify()

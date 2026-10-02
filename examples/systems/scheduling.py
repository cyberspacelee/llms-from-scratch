"""CPU reference for one iteration of prefill-first continuous batching."""

from dataclasses import dataclass, field


@dataclass
class Request:
    name: str
    prompt: tuple[str, ...]
    answers: tuple[str, ...]
    cached: int = 0
    output: list[str] = field(default_factory=list)
    state: str = "waiting"

    @property
    def ids(self):
        return self.prompt + tuple(self.output)

    def run(self, count):
        assert self.state != "finished" and 0 < count <= len(self.ids) - self.cached
        positions = tuple(range(self.cached, self.cached + count))
        inputs = self.ids[self.cached : self.cached + count]
        self.cached += count  # KV writes finish before sampling in this synchronous model.
        sampled = None
        if self.cached == len(self.ids):
            sampled = self.answers[len(self.output)]
            self.output.append(sampled)
            self.state = "finished" if len(self.output) == len(self.answers) else "running"
        return inputs, positions, sampled


class Scheduler:
    def __init__(self, budget=4, max_seqs=2):
        if budget <= 0 or max_seqs <= 0:
            raise ValueError("budgets must be positive")
        self.budget = budget
        self.max_seqs = max_seqs
        self.waiting = []
        self.running = []

    def add(self, request):
        assert request.state == "waiting"
        self.waiting.append(request)

    def step(self):
        chosen = []
        remaining = self.budget
        if self.waiting:
            for request in list(self.waiting):
                if not remaining or len(chosen) == self.max_seqs:
                    break
                need = len(request.ids) - request.cached
                if chosen and need > remaining:
                    break  # This fixed policy only chunks the first waiting request.
                count = min(need, remaining)
                chosen.append((request, count))
                remaining -= count
            mode = "prefill"
        else:
            chosen = [(request, 1) for request in self.running[: min(self.max_seqs, self.budget)]]
            mode = "decode"

        records = []
        for request, count in chosen:
            inputs, positions, sampled = request.run(count)
            if request.state != "waiting" and request in self.waiting:
                self.waiting.remove(request)
                if request.state == "running":
                    self.running.append(request)
            if request.state == "finished" and request in self.running:
                self.running.remove(request)
            records.append(
                (
                    request.name,
                    inputs,
                    positions,
                    request.cached,
                    tuple(request.output),
                    sampled,
                    request.state,
                )
            )
        return mode, records

    def preempt(self, request):
        assert request in self.running
        self.running.remove(request)
        request.cached = 0
        request.state = "waiting"
        self.waiting.insert(0, request)

    def cancel(self, request):
        assert request.state != "finished"
        queue = self.waiting if request.state == "waiting" else self.running
        queue.remove(request)
        request.cached = 0
        request.state = "finished"


def verify():
    scheduler = Scheduler()
    a = Request("A", ("A", "B", "C"), ("D", "E"))
    b = Request("B", ("u", "v", "w", "x", "y"), ("F", "H"))
    c = Request("C", ("m", "n"), ("G",))
    scheduler.add(a)
    scheduler.add(b)
    first = scheduler.step()
    assert first == ("prefill", [("A", ("A", "B", "C"), (0, 1, 2), 3, ("D",), "D", "running")])
    assert scheduler.step() == (
        "prefill",
        [("B", ("u", "v", "w", "x"), (0, 1, 2, 3), 4, (), None, "waiting")],
    )
    scheduler.add(c)
    third = scheduler.step()
    assert third == (
        "prefill",
        [
            ("B", ("y",), (4,), 5, ("F",), "F", "running"),
            ("C", ("m", "n"), (0, 1), 2, ("G",), "G", "finished"),
        ],
    )
    fourth = scheduler.step()
    assert fourth == (
        "decode",
        [
            ("A", ("D",), (3,), 4, ("D", "E"), "E", "finished"),
            ("B", ("F",), (5,), 6, ("F", "H"), "H", "finished"),
        ],
    )
    assert scheduler.step() == ("decode", [])

    replay = Scheduler(budget=2)
    r = Request("R", ("a", "b", "c"), ("d", "e"))
    replay.add(r)
    assert replay.step()[1][0][-2] is None  # Incomplete prompt cannot sample.
    replay.step()
    assert r.output == ["d"] and r.cached == 3
    replay.preempt(r)
    assert replay.step()[1][0][-2] is None
    assert replay.step()[1][0][-2] == "e"  # Only d's forward supplies a new next-token result.
    assert r.output == ["d", "e"] and r.cached == 4
    canceled = Request("X", ("q",), ("r", "s"))
    replay.add(canceled)
    replay.cancel(canceled)
    assert canceled.state == "finished" and canceled.cached == 0
    assert replay.step() == ("decode", [])
    narrow = Scheduler(budget=1, max_seqs=2)
    narrow.add(Request("Y", ("y",), ("z", "q")))
    narrow.add(Request("Z", ("z",), ("y", "q")))
    narrow.step()
    narrow.step()
    assert len(narrow.step()[1]) == 1
    print("scheduling: positions, chunking, admission, preemption and cancel verified")


if __name__ == "__main__":
    verify()

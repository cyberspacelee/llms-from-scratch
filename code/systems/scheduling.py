"""Reference request lifecycle; not the nano-vLLM scheduling policy."""
from dataclasses import dataclass


@dataclass
class Request:
    prompt: int
    maximum: int
    cached: int = 0
    generated: int = 0

    @property
    def done(self):
        return self.generated == self.maximum

    def step(self, budget):
        if budget <= 0 or self.prompt <= 0 or self.maximum <= 0:
            raise ValueError("positive lengths and budget required")
        if self.done:
            return 0
        existing = self.prompt + self.generated
        count = min(budget, existing - self.cached)
        self.cached += count
        if self.cached == existing:
            self.generated += 1
        return count

    def preempt(self):
        self.cached = 0


def verify():
    request = Request(5, 3)
    records = []
    while not request.done:
        used = request.step(2)
        records.append((used, request.cached, request.generated))
    assert records == [(2, 2, 0), (2, 4, 0), (1, 5, 1), (1, 6, 2), (1, 7, 3)]
    assert request.step(2) == 0
    request = Request(3, 3)
    request.step(3)
    assert request.generated == 1
    request.preempt()
    request.step(2)
    assert request.generated == 1  # Recompute is not another generated token.
    request.step(2)
    assert request.cached == 4 and request.generated == 2
    print("scheduler: chunk boundaries, no premature sampling, finish and preemption verified")


if __name__ == "__main__":
    verify()

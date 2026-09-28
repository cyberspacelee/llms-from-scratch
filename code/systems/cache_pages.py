"""Block ownership, prefix sharing and copy-on-write on a CPU reference."""
import numpy as np


class Pages:
    def __init__(self, count=8, block_size=4):
        if count <= 0 or block_size <= 0:
            raise ValueError("positive capacity required")
        self.values = np.zeros((count, block_size), dtype=np.float64)
        self.refs = np.zeros(count, dtype=np.int64)
        self.block_size = block_size

    def allocate(self):
        free = np.flatnonzero(self.refs == 0)
        if not len(free):
            raise MemoryError("no free block")
        block = int(free[0])
        self.values[block] = 0
        self.refs[block] = 1
        return block

    def share(self, table):
        if any(self.refs[block] <= 0 for block in table):
            raise ValueError("sharing a free block")
        for block in table:
            self.refs[block] += 1
        return list(table)

    def append(self, table, length, value):
        if length < 0 or len(table) != (length + self.block_size - 1) // self.block_size:
            raise ValueError("table does not match current length")
        logical, offset = divmod(length, self.block_size)
        if logical == len(table):
            table.append(self.allocate())
        block = table[logical]
        if self.refs[block] > 1:
            copy = self.allocate()
            self.values[copy, :offset] = self.values[block, :offset]
            self.refs[block] -= 1
            table[logical] = block = copy
        self.values[block, offset] = value
        return self.slot(table, length)

    def slot(self, table, position):
        return table[position // self.block_size] * self.block_size + position % self.block_size

    def read(self, table, length):
        return np.array([self.values[table[i // self.block_size], i % self.block_size]
                         for i in range(length)])

    def release(self, table):
        if any(self.refs[block] <= 0 for block in table):
            raise ValueError("double release")
        for block in table:
            self.refs[block] -= 1
        table.clear()


def verify():
    pages, a = Pages(), []
    for i in range(6):
        assert pages.append(a, i, i + 1) == i
    assert a == [0, 1]
    assert pages.refs[:3].tolist() == [1, 1, 0]
    assert pages.slot(a, 5) == 5
    b = pages.share(a)
    assert b == [0, 1] and pages.refs[:3].tolist() == [2, 2, 0]
    assert pages.append(a, 6, 7) == 10
    assert a == [0, 2] and b == [0, 1]
    assert pages.refs[:3].tolist() == [2, 1, 1]
    assert pages.append(b, 6, 70) == 6
    assert pages.refs[:3].tolist() == [2, 1, 1]
    assert np.array_equal(pages.read(a, 7), [1, 2, 3, 4, 5, 6, 7])
    assert np.array_equal(pages.read(b, 7), [1, 2, 3, 4, 5, 6, 70])
    pages.release(a)
    assert pages.refs[:3].tolist() == [1, 1, 0]
    assert np.array_equal(pages.read(b, 7), [1, 2, 3, 4, 5, 6, 70])
    pages.release(b)
    assert pages.refs[:3].tolist() == [0, 0, 0]
    assert pages.allocate() == 0
    print("pages: slots 5/10/6, shared reads, copy-on-write and release verified")


if __name__ == "__main__":
    verify()

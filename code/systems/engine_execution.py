"""Check one three-request prefill and decode against runner metadata rules."""

from itertools import accumulate


BLOCK_SIZE = 4


def slots(positions, block_table):
    return [block_table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in positions]


def prefill(requests):
    ids, positions, slot_mapping = [], [], []
    q_lengths, k_lengths = [], []
    for cached, new_ids, block_table in requests:
        current_positions = list(range(cached, cached + len(new_ids)))
        ids.extend(new_ids)
        positions.extend(current_positions)
        slot_mapping.extend(slots(current_positions, block_table))
        q_lengths.append(len(new_ids))
        k_lengths.append(cached + len(new_ids))
    cu_q = [0, *accumulate(q_lengths)]
    cu_k = [0, *accumulate(k_lengths)]
    return ids, positions, slot_mapping, cu_q, cu_k


def decode(requests, graph_size):
    assert 0 < len(requests) <= graph_size
    ids, positions, slot_mapping, context_lens, block_tables = [], [], [], [], []
    width = max(len(table) for _, _, table in requests)
    for token_id, position, table in requests:
        ids.append(token_id)
        positions.append(position)
        slot_mapping.extend(slots([position], table))
        context_lens.append(position + 1)
        block_tables.append(table + [-1] * (width - len(table)))
    # The real graph keeps statically addressed buffers; this list models their active rows.
    padded_slots = slot_mapping + [-1] * (graph_size - len(requests))
    padded_lens = context_lens + [0] * (graph_size - len(requests))
    return ids, positions, slot_mapping, context_lens, block_tables, padded_slots, padded_lens


def verify():
    first = prefill([
        (0, [11, 12, 13], [2]),
        (4, [25, 26], [5, 7]),
        (0, [31, 32, 33, 34], [9]),
    ])
    assert first == (
        [11, 12, 13, 25, 26, 31, 32, 33, 34],
        [0, 1, 2, 4, 5, 0, 1, 2, 3],
        [8, 9, 10, 28, 29, 36, 37, 38, 39],
        [0, 3, 5, 9],
        [0, 3, 9, 13],
    )
    assert [end - 1 for end in first[3][1:]] == [2, 4, 8]

    second = decode([(14, 3, [2]), (27, 6, [5, 7]), (35, 4, [9, 10])], 4)
    assert second == (
        [14, 27, 35], [3, 6, 4], [11, 30, 40], [4, 7, 5],
        [[2, -1], [5, 7], [9, 10]], [11, 30, 40, -1], [4, 7, 5, 0],
    )
    assert set(first[2]).isdisjoint(second[2])
    assert all(slot < 0 or length > 0 for slot, length in zip(second[5], second[6]))

    changed = prefill([(0, [11, 12, 13], [2]), (5, [25, 26], [5, 7]),
                       (0, [31, 32, 33, 34], [9])])
    assert changed[1][3:5] == [5, 6]
    assert changed[2][3:5] == [29, 30]
    assert changed[4] == [0, 3, 10, 14]
    print("engine execution: packed boundaries, KV slots, decode metadata and padding verified")


if __name__ == "__main__":
    verify()

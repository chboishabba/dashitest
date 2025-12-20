"""
svo_traversal_bench.py
----------------------
Tiny ternary sparse-voxel-octree (SVO) traversal benchmark:
- Nodes have 8 children with ternary state {0 empty, 1 partial, 2 full} (no specials).
- Baseline: Python list-of-lists traversal.
- SWAR: packed children states in one uint64 (lanes 0..7), branchless traversal.
Measures traversal time and validates counts match.
"""

import time
import numpy as np
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

LANE_SHIFTS = np.array([5*i for i in range(8)], dtype=np.uint64)


def pack_children(states8: np.ndarray) -> np.uint64:
    """Pack 8 child states (0..2) into a uint64 using 5-bit lanes."""
    return np.sum(states8.astype(np.uint64) << LANE_SHIFTS, dtype=np.uint64)


def generate_tree(depth=4, seed=0):
    """
    Generate a full 8-ary tree of given depth with random ternary states per child.
    Returns baseline structure (list of child arrays) and packed words with child index mapping.
    """
    rng = np.random.default_rng(seed)
    total_nodes = (8**depth - 1) // 7
    children = []
    packed = np.empty(total_nodes, dtype=np.uint64)
    # store child start index for each node in flat array of children (full tree, no sparsity)
    child_start = np.empty(total_nodes, dtype=np.int64)
    idx = 0
    nodes_per_level = 1
    offset = 1  # next node index
    for level in range(depth):
        for n in range(nodes_per_level):
            node_idx = idx
            child_start[node_idx] = offset if level < depth - 1 else -1
            states = rng.integers(0, 3, size=8, dtype=np.uint8)
            children.append(states)
            packed[node_idx] = pack_children(states)
            if level < depth - 1:
                offset += 8
            idx += 1
        nodes_per_level *= 8
    return children, packed, child_start


def traverse_baseline(children, child_start, depth):
    """DFS traversal counting non-empty leaves (state > 0)."""
    stack = [(0, 0)]
    count = 0
    while stack:
        node_idx, level = stack.pop()
        states = children[node_idx]
        if level == depth - 1:
            count += int((states > 0).sum())
        else:
            base = child_start[node_idx]
            for i, s in enumerate(states):
                if s > 0:
                    child_idx = base + i
                    stack.append((child_idx, level + 1))
    return count


def traverse_swar(packed, child_start, depth):
    """Branch-reduced traversal using packed child states."""
    stack = [(0, 0)]
    count = 0
    while stack:
        node_idx, level = stack.pop()
        word = packed[node_idx]
        if level == depth - 1:
            # count lanes >0
            for lane in range(8):
                v = (word >> np.uint64(5*lane)) & np.uint64(0x1F)
                count += int(v > 0)
        else:
            base = child_start[node_idx]
            for lane in range(8):
                v = (word >> np.uint64(5*lane)) & np.uint64(0x1F)
                if v > 0:
                    stack.append((base + lane, level + 1))
    return count


if HAVE_NUMBA:
    @nb.njit
    def traverse_swar_nb(packed, child_start, depth):
        count = 0
        max_nodes = packed.shape[0]
        stack_nodes = np.empty(max_nodes, dtype=np.int64)
        stack_levels = np.empty(max_nodes, dtype=np.int64)
        sp = 0
        stack_nodes[sp] = 0
        stack_levels[sp] = 0
        sp += 1
        while sp > 0:
            sp -= 1
            node_idx = stack_nodes[sp]
            level = stack_levels[sp]
            word = packed[node_idx]
            if level == depth - 1:
                for lane in range(8):
                    v = (word >> np.uint64(5*lane)) & np.uint64(0x1F)
                    if v > 0:
                        count += 1
            else:
                base = child_start[node_idx]
                for lane in range(8):
                    v = (word >> np.uint64(5*lane)) & np.uint64(0x1F)
                    if v > 0:
                        stack_nodes[sp] = base + lane
                        stack_levels[sp] = level + 1
                        sp += 1
        return count


def bench(fn, *args, iters=10):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    depth = 4
    children, packed, child_start = generate_tree(depth=depth, seed=123)

    c_base = traverse_baseline(children, child_start, depth)
    c_swar = traverse_swar(packed, child_start, depth)
    if c_base != c_swar:
        raise AssertionError(f"Count mismatch: baseline={c_base} swar={c_swar}")

    t_base = bench(traverse_baseline, children, child_start, depth, iters=50)
    t_swar = bench(traverse_swar, packed, child_start, depth, iters=200)

    print("Ternary SVO traversal (8-ary, depth=4, random states 0/1/2):")
    print(f"Nodes={(8**depth - 1)//7}: baseline {t_base*1e6:8.2f} µs/traversal   SWAR {t_swar*1e6:8.2f} µs/traversal   speedup x{t_base/t_swar:5.2f}")

    if HAVE_NUMBA:
        # warmup compile
        traverse_swar_nb(packed, child_start, depth)
        t_nb = bench(traverse_swar_nb, packed, child_start, depth, iters=500)
        c_nb = traverse_swar_nb(packed, child_start, depth)
        if c_nb != c_base:
            raise AssertionError("Numba SVO traversal mismatch")
        print(f"Numba SWAR: {t_nb*1e6:8.2f} µs/traversal   speedup vs baseline x{t_base/t_nb:5.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GP Tree Performance Testbed.

Benchmarks 25 GP tree configurations across 5 groups to stress-test
the rsimulator GP evaluation engine.

Usage:
    uv run python bench_gp_trees.py

Groups:
    A — Size sweep     : left-chain, all-add, 3→501 nodes (stack depth always 2)
    F — Size sweep     : balanced full tree, depths 1–7, 3→255 nodes (stack depth ≤ 8)
    B — Shape          : ~31-node trees, different shapes, stack depth 2–8 (max)
                         right_chain/alternating are capped+padded to avoid depth > 8
    C — Operator cost  : ~15-node trees, one operator type each
    D — Terminal cost  : ~15-node trees, one terminal type each
    E — Role sensitivity: same 31-node balanced-d4 tree in routing/sequencing/reject roles

All groups vary only the routing tree except Group E.
Fixed trivial trees: seq = -time_until_due, reject = const(0.0).
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from dvrptw.instance import load_vrpr_csv
from dvrptw import RustSimulator
from rsimulator import (
    gp_strategy,
    flat_gp_const,
    flat_gp_travel_time,
    flat_gp_time_until_due,
    flat_gp_demand,
    flat_gp_current_load,
    flat_gp_remaining_capacity,
    flat_gp_window_earliest,
    FlatGpTree,
)

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------
CSV_PATH = "h1000C1_10_1.csv"
TRUCK_SPEED = 1.0
TRUCK_CAPACITY = 1300.0
NUM_TRUCKS = 10
N_TRIALS = 10

# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

TreeFn = Callable[[], FlatGpTree]


def left_chain(depth: int, terminal_fn: TreeFn, op: str = "+") -> FlatGpTree:
    """Left-associative chain: ((t op t) op t) op t ...
    Node count: 1 + 2 * depth."""
    t = terminal_fn()
    for _ in range(depth):
        t2 = terminal_fn()
        if op == "-":
            t = t - t2
        elif op == "*":
            t = t * t2
        elif op == "/":
            t = t / t2
        else:
            t = t + t2
    return t


def right_chain(depth: int, terminal_fn: TreeFn, op: str = "+") -> FlatGpTree:
    """Right-associative chain: t op (t op (t op t))
    Node count: 1 + 2 * depth."""
    t = terminal_fn()
    for _ in range(depth):
        t2 = terminal_fn()
        if op == "-":
            t = t2 - t
        elif op == "*":
            t = t2 * t
        elif op == "/":
            t = t2 / t
        else:
            t = t2 + t
    return t


def balanced_tree(depth: int, terminal_fn: TreeFn, op: str = "+") -> FlatGpTree:
    """Balanced binary tree.
    depth=0: 1 node; depth=d: 2^(d+1)-1 nodes (depth=4 → 31 nodes)."""
    if depth == 0:
        return terminal_fn()
    left = balanced_tree(depth - 1, terminal_fn, op)
    right = balanced_tree(depth - 1, terminal_fn, op)
    if op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        return left / right
    return left + right


def alternating_tree(depth: int, terminal_fn: TreeFn) -> FlatGpTree:
    """Alternates left/right attachment at each step — medium stack depth ~depth/2.
    Node count: 1 + 2 * depth.

    Stack depth grows by 1 on every odd step:
      depth=12 → stack depth 8 (MAX safe)
      depth=13+ → stack depth 9+ (OVERFLOW)
    """
    t = terminal_fn()
    for i in range(depth):
        t2 = terminal_fn()
        if i % 2 == 0:
            t = t + t2  # left-heavy: does not increase stack depth
        else:
            t = t2 + t  # right-heavy: pushes existing tree deeper
    return t


def right_capped_padded(
    right_depth: int, left_pad: int, terminal_fn: TreeFn, op: str = "+"
) -> FlatGpTree:
    """Right-chain of `right_depth` steps (stack depth = right_depth+1), then
    left-chain-extend by `left_pad` steps (keeps stack depth constant).

    Node count: (1 + 2*right_depth) + 2*left_pad
    Example: right_capped_padded(7, 8, ...) → 15+16 = 31 nodes, stack depth 8.
    """
    t = right_chain(right_depth, terminal_fn, op)
    for _ in range(left_pad):
        t2 = terminal_fn()
        if op == "-":
            t = t - t2
        elif op == "*":
            t = t * t2
        elif op == "/":
            t = t / t2
        else:
            t = t + t2
    return t


def alternating_padded(
    alt_depth: int, left_pad: int, terminal_fn: TreeFn
) -> FlatGpTree:
    """alternating_tree(alt_depth) then left-chain-extend by left_pad steps.

    Node count: (1 + 2*alt_depth) + 2*left_pad
    Example: alternating_padded(12, 3, ...) → 25+6 = 31 nodes, stack depth 8.
    """
    t = alternating_tree(alt_depth, terminal_fn)
    for _ in range(left_pad):
        t = t + terminal_fn()
    return t


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark(
    instance,
    routing_fn: TreeFn,
    seq_fn: TreeFn,
    rej_fn: TreeFn,
    n_trials: int = N_TRIALS,
) -> tuple[int, float, float, float]:
    """Run n_trials simulations; return (routing_bytes, min_ms, mean_ms, max_ms).

    Tree objects are rebuilt each trial (cheap) so gp_strategy sees fresh
    objects. Timing starts at gp_strategy() call (includes Rust JIT + sim).
    """
    routing_bytes = len(routing_fn().ops)
    times_ms: list[float] = []
    for _ in range(n_trials):
        routing = routing_fn()
        seq = seq_fn()
        rej = rej_fn()
        t0 = time.perf_counter()
        strategy = gp_strategy(routing, seq, rej)
        sim = RustSimulator(instance, strategy)
        sim.run()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return routing_bytes, min(times_ms), sum(times_ms) / len(times_ms), max(times_ms)


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

# Each config: (display_name, routing_fn, seq_fn, rej_fn)
Config = tuple[str, TreeFn, TreeFn, TreeFn]


def make_configs() -> list[Config]:
    def trivial_seq():
        return -flat_gp_time_until_due()

    def trivial_rej():
        return flat_gp_const(0.0)

    def trivial_routing():
        return flat_gp_remaining_capacity()

    configs: list[Config] = []

    # ---- Group A: Size sweep (left-chain, all-add, travel_time; stack depth always 2) ----
    # All three trees (routing, sequencing, reject) use the same left-chain size.
    for label, depth in [
        ("A1 (d=1,   3n)", 1),
        ("A2 (d=4,   9n)", 4),
        ("A3 (d=8,  17n)", 8),
        ("A4 (d=16, 33n)", 16),
        ("A5 (d=32, 65n)", 32),
        ("A6 (d=64,129n)", 64),
        ("A7 (d=128,257n)", 128),
        ("A8 (d=250,501n)", 250),
    ]:
        d = depth

        def fn(d=d):
            return left_chain(d, flat_gp_travel_time)

        configs.append((label, fn, fn, fn))

    # ---- Group F: Full (balanced) tree size sweep, depths 1-7 (stack depth ≤ 8) ----
    # balanced(d) → 2^(d+1)-1 nodes, stack depth d+1
    # depth=7 → 255 nodes, stack depth 8 (MAX safe); depth=8 would overflow
    # All three trees use the same balanced depth.
    for label, depth in [
        ("F1 (d=1,  3n)", 1),
        ("F2 (d=2,  7n)", 2),
        ("F3 (d=3, 15n)", 3),
        ("F4 (d=4, 31n)", 4),
        ("F5 (d=5, 63n)", 5),
        ("F6 (d=6,127n)", 6),
        ("F7 (d=7,255n)", 7),
    ]:
        d = depth

        def fn(d=d):
            return balanced_tree(d, flat_gp_travel_time)

        configs.append((label, fn, fn, fn))

    # ---- Group B: Shape comparison (~31 nodes, all stack depth ≤ 8) ----
    # B1: pure left-chain — stack depth 2 (min)
    configs.append(
        (
            "B1 (left-chain)",
            lambda: left_chain(15, flat_gp_travel_time),
            trivial_seq,
            trivial_rej,
        )
    )
    # B2: right-core (depth 7, stack 8) + left padding → 31 nodes, stack depth 8 (MAX)
    #     right_chain(15) would need stack 16 — exceeds limit
    configs.append(
        (
            "B2 (right-core+pad)",
            lambda: right_capped_padded(7, 8, flat_gp_travel_time),
            trivial_seq,
            trivial_rej,
        )
    )
    # B3: balanced depth-4 — stack depth 5, canonical
    configs.append(
        (
            "B3 (balanced-d4)",
            lambda: balanced_tree(4, flat_gp_travel_time),
            trivial_seq,
            trivial_rej,
        )
    )
    # B4: alternating(12)+pad → 31 nodes, stack depth 8 (MAX)
    #     alternating(15) reaches stack depth 9 — exceeds limit
    configs.append(
        (
            "B4 (alt+pad)",
            lambda: alternating_padded(12, 3, flat_gp_travel_time),
            trivial_seq,
            trivial_rej,
        )
    )

    # ---- Group C: Operator cost (~15 nodes, left-chain, travel_time) ----
    for label, op in [
        ("C1 (add)", "+"),
        ("C2 (sub)", "-"),
        ("C3 (mul)", "*"),
        ("C4 (div)", "/"),
    ]:
        o = op
        configs.append(
            (
                label,
                lambda o=o: left_chain(7, flat_gp_travel_time, o),
                trivial_seq,
                trivial_rej,
            )
        )

    # ---- Group D: Terminal cost (~15 nodes, left-chain, all-add) ----
    terminal_cases: list[tuple[str, TreeFn]] = [
        ("D1 (const)", lambda: flat_gp_const(1.0)),
        ("D2 (demand)", flat_gp_demand),
        ("D3 (win_earliest)", flat_gp_window_earliest),
        ("D4 (cur_load)", flat_gp_current_load),
        ("D5 (travel_time)", flat_gp_travel_time),
    ]
    for label, term_fn in terminal_cases:
        tf = term_fn
        configs.append(
            (label, lambda tf=tf: left_chain(7, tf), trivial_seq, trivial_rej)
        )

    # ---- Group E: Role sensitivity (31-node balanced-d4 in different roles) ----
    def complex_fn():
        return balanced_tree(4, flat_gp_travel_time)

    configs.append(("E1 (routing)", complex_fn, trivial_seq, trivial_rej))
    configs.append(("E2 (sequencing)", trivial_routing, complex_fn, trivial_rej))
    configs.append(("E3 (reject)", trivial_routing, trivial_seq, complex_fn))
    configs.append(("E4 (all)", complex_fn, complex_fn, complex_fn))

    return configs


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

HDR = f"  {'Config':<22} {'bytes/tree':>10} {'min_ms':>8} {'mean_ms':>9} {'max_ms':>8}"
SEP = "  " + "-" * 60
ROW = "  {:<22} {:>10} {:>8.1f} {:>9.1f} {:>8.1f}"


def print_group(title: str, prefix: str, configs: list[Config], instance) -> None:
    print(title)
    if prefix in ("A", "F"):
        print("  (all three trees — routing, sequencing, reject — use the same size)")
    if prefix == "E":
        print("  (bytes/tree = routing tree; complex tree = 31 bytes)")
    print(HDR)
    print(SEP)
    for name, routing_fn, seq_fn, rej_fn in configs:
        if not name.startswith(prefix):
            continue
        routing_bytes, mn, mean, mx = benchmark(instance, routing_fn, seq_fn, rej_fn)
        print(ROW.format(name, routing_bytes, mn, mean, mx))
        sys.stdout.flush()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading instance from {CSV_PATH} ...")
    instance = load_vrpr_csv(CSV_PATH, TRUCK_SPEED, TRUCK_CAPACITY, NUM_TRUCKS)
    customers = [r for r in instance.requests if not r.is_depot]
    n_customers = len(customers)
    n_static = sum(1 for r in customers if r.release_time == 0.0)
    print(
        f"  Customers : {n_customers}  ({n_static} static, {n_customers - n_static} dynamic)"
    )
    print(
        f"  Vehicles  : {len(instance.vehicles)} × capacity {instance.vehicles[0].capacity}"
    )
    print(f"  Trials    : {N_TRIALS} per config")
    print()

    configs = make_configs()

    groups = [
        ("Group A — Size sweep: left-chain (stack depth 2, up to 501 nodes)", "A"),
        (
            "Group F — Size sweep: balanced full tree (stack depth ≤ 8, up to 255 nodes)",
            "F",
        ),
        ("Group B — Shape comparison (~31 nodes, all travel_time+add)", "B"),
        ("Group C — Operator cost (~15 nodes, left-chain, travel_time)", "C"),
        ("Group D — Terminal cost (~15 nodes, left-chain, all-add)", "D"),
        ("Group E — Role sensitivity (31-node balanced, travel_time+add)", "E"),
    ]

    t_total = time.perf_counter()
    for title, prefix in groups:
        print_group(title, prefix, configs, instance)

    print(f"Total wall time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()

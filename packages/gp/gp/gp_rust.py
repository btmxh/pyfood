"""GP driver that evaluates individuals by running the Rust simulator.

Each individual is a triple of `rsimulator.FlatGpTree` objects:
    (routing_tree, sequencing_tree, reject_tree)

Fitness is computed by constructing a native GP strategy via
`rsimulator.gp_strategy(routing, sequencing, reject)`, passing that to the
Rust-backed simulator (`dvrptw.simulator.rust.RustSimulator`) and running a
full simulation. The simulator returns the two objectives (travel cost,
rejections); an `Evaluator` is used to collapse them to a scalar for the GA.

This module relies on the Rust extension for the heavy work (strategy
execution and simulation).  The GA operators (tournament selection,
crossover/mutation of FlatGpTree byte arrays) are implemented in Python and
operate on the compact opcode bytes exposed by `FlatGpTree.ops`.
"""

from __future__ import annotations

import random
import math
from typing import Callable

import rsimulator as rs
from dvrptw import RustSimulator, DVRPTWInstance, Evaluator, StarNormEvaluator

from .evolve import make_full, make_grow


def subtree_start(ops: bytes, end: int) -> int:
    """Find the start index of the subtree whose last token index is `end`.

    Mirrors `FlatTree::subtree_start` scanning algorithm.
    """
    needed = 1
    i = end
    while i >= 0:
        b = ops[i]
        if (b & 0xC0) != 0xC0:  # leaf
            needed -= 1
        else:
            needed += 1
        if needed == 0:
            return i
        i -= 1
    return 0


def random_subtree_range(ops: bytes) -> tuple[int, int]:
    if not ops:
        return (0, -1)
    end = random.randrange(len(ops))
    start = subtree_start(ops, end)
    return (start, end)


def all_subtree_ranges(ops: bytes) -> list[tuple[int, int]]:
    """Return a list of all (start, end) subtree ranges for `ops`.

    Each subtree is identified by the index of its last token `end` and the
    corresponding start found via `subtree_start`.
    """
    ranges: list[tuple[int, int]] = []
    for end in range(len(ops)):
        start = subtree_start(ops, end)
        ranges.append((start, end))
    return ranges


def splice(ops: bytes, a_start: int, a_end: int, replacement: bytes) -> bytes:
    return ops[:a_start] + replacement + ops[a_end + 1 :]


Individual = tuple[rs.FlatGpTree, rs.FlatGpTree, rs.FlatGpTree]


def make_individual(depth: int) -> Individual:
    """Create one individual: (routing, sequencing, reject)."""

    # Use the same initializer for each role
    def mk(d: int) -> rs.FlatGpTree:
        return make_grow(d) if random.random() < 0.5 else make_full(d)

    return (mk(depth), mk(depth), mk(depth))


def init_population(pop_size: int, max_depth: int) -> list[Individual]:
    return [make_individual(1 + (i % max_depth)) for i in range(pop_size)]


def subtree_crossover(
    a: rs.FlatGpTree,
    b: rs.FlatGpTree,
    max_nodes: int,
    make_tree_fn: Callable[[int], rs.FlatGpTree],
) -> rs.FlatGpTree | None:
    """Crossover a subtree from `b` into `a` while respecting `max_nodes`.

    If no valid replacement subtree fits the budget, returns `None` to signal
    the caller that this crossover attempt failed.
    """
    a_ops: bytes = a.ops
    b_ops: bytes = b.ops

    a_s, a_e = random_subtree_range(a_ops)
    # context size = nodes remaining in `a` after removing the selected subtree
    context_size = len(a_ops) - (a_e - a_s + 1)
    budget = max_nodes - context_size
    if budget < 1:
        return None

    # find all b-subtrees that fit into the budget
    candidates = [r for r in all_subtree_ranges(b_ops) if (r[1] - r[0] + 1) <= budget]
    if candidates:
        b_s, b_e = random.choice(candidates)
        replacement = b_ops[b_s : b_e + 1]
    else:
        # no subtree of b fits; create a small fresh tree that fits
        # derive the maximum full-tree depth that fits in `budget`
        max_d = max(0, int(math.floor(math.log2(budget + 1))) - 1)
        replacement = make_tree_fn(max_d).to_bytes()

    child_ops = splice(a_ops, a_s, a_e, replacement)
    return rs.FlatGpTree.from_bytes(child_ops)


def crossover_individual(
    parent_a: Individual,
    parent_b: Individual,
    max_nodes: int,
    make_tree_fn: Callable[[int], rs.FlatGpTree],
) -> Individual | None:
    # crossover per-role; return None if any role fails to produce a child
    r = subtree_crossover(parent_a[0], parent_b[0], max_nodes, make_tree_fn)
    if r is None:
        return None
    s = subtree_crossover(parent_a[1], parent_b[1], max_nodes, make_tree_fn)
    if s is None:
        return None
    rej = subtree_crossover(parent_a[2], parent_b[2], max_nodes, make_tree_fn)
    if rej is None:
        return None
    return (r, s, rej)


def mutate_individual(
    ind: Individual,
    max_subtree_depth: int,
    make_tree_fn: Callable[[int], rs.FlatGpTree],
    max_nodes: int,
) -> Individual:
    # pick a role to mutate
    role = random.randrange(3)
    trees = list(ind)
    # reuse engine mutation logic: replace a random subtree with a new small tree
    ops = trees[role].ops
    s, e = random_subtree_range(ops)
    # compute budget after removing selected subtree
    context_size = len(ops) - (e - s + 1)
    budget = max_nodes - context_size
    if budget < 1:
        return ind

    # derive a depth that fits in the budget, and respect requested max_subtree_depth
    max_d = max(0, int(math.floor(math.log2(budget + 1))) - 1)
    chosen_d = min(max_subtree_depth, max_d)
    replacement = make_tree_fn(chosen_d).to_bytes()
    new_ops = splice(ops, s, e, replacement)
    trees[role] = rs.FlatGpTree.from_bytes(new_ops)
    return tuple(trees)  # type: ignore[return-value]


def evaluate_individual(
    individual: Individual,
    instance: DVRPTWInstance,
) -> tuple[float, int]:
    """Run the Rust simulator using the GP native strategy and return raw objectives.

    Returns (total_travel_cost, rejected_count).
    """
    routing, sequencing, reject = individual
    native = rs.gp_strategy(routing, sequencing, reject)
    sim = RustSimulator(instance, native)
    try:
        result = sim.run()
        return (result.metrics.total_travel_cost, result.metrics.rejected)
    except Exception:
        # The Rust simulator may raise on infeasible dispatches or other
        # runtime errors. Treat exceptions as very bad fitness: high travel
        # cost and all requests rejected. This lets the GA continue safely.
        n_customers = len([r for r in instance.requests if not r.is_depot])
        penalty_cost = float(1e9)
        return (penalty_cost, n_customers)


def run_gp_rust(
    instance: DVRPTWInstance,
    pop_size: int = 50,
    max_depth: int = 4,
    generations: int = 30,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    elitism: int = 1,
    evaluator: Evaluator | None = None,
    make_tree_fn: Callable[[int], rs.FlatGpTree] | None = None,
    max_nodes: int | None = None,
) -> tuple[Individual, tuple[float, int], list[dict]]:
    """Evolve GP trees where fitness is a DVRPTW simulation result.

    Returns the best individual and its raw objectives (travel_cost, rejected).
    The `evaluator` collapses the two objectives to a scalar for selection; if
    omitted the StarNormEvaluator with equal weights is used.
    """
    if evaluator is None:
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
    assert evaluator is not None

    if make_tree_fn is None:

        def _mk(d: int) -> rs.FlatGpTree:
            return make_grow(d) if random.random() < 0.5 else make_full(d)

        make_tree_fn = _mk
    assert make_tree_fn is not None

    pop = init_population(pop_size, max_depth)

    # derive max_nodes if not provided: size of a full binary tree at max_depth
    if max_nodes is None:
        max_nodes_val = 2 ** (max_depth + 1) - 1
    else:
        max_nodes_val = int(max_nodes)

    # simple sanity check and ensure concrete int for downstream calls
    if max_nodes_val < 1:
        raise ValueError("max_nodes must be >= 1")
    max_nodes_int: int = int(max_nodes_val)

    best_ind: Individual = pop[0]
    best_scalar = float("inf")
    best_obj: tuple[float, int] = (float("inf"), 10**9)

    history: list[dict] = []

    for gen in range(generations):
        # evaluate population sequentially (could be parallelised later)
        raw_objs: list[tuple[float, int]] = [
            evaluate_individual(ind, instance) for ind in pop
        ]
        scalars: list[float] = [evaluator.scalar(o[0], o[1]) for o in raw_objs]

        # update best
        for ind, s, o in zip(pop, scalars, raw_objs):
            if s < best_scalar:
                best_scalar = s
                best_ind = ind
                best_obj = o

        # create next generation
        ranked = sorted(range(len(pop)), key=lambda i: scalars[i])
        new_pop: list[Individual] = [pop[i] for i in ranked[:elitism]]

        # record generation statistics
        mean_scalar = sum(scalars) / len(scalars)
        mean_cost = sum(o[0] for o in raw_objs) / len(raw_objs)
        mean_rej = sum(o[1] for o in raw_objs) / len(raw_objs)
        gen_best_idx = ranked[0]
        gen_best = {
            "gen": gen,
            "best_scalar": scalars[gen_best_idx],
            "best_cost": raw_objs[gen_best_idx][0],
            "best_rej": raw_objs[gen_best_idx][1],
            "mean_scalar": mean_scalar,
            "mean_cost": mean_cost,
            "mean_rej": mean_rej,
        }
        history.append(gen_best)
        # print concise generation progress info
        try:
            print(
                f"Gen {gen}: best_scalar={gen_best['best_scalar']:.6g} "
                f"best_cost={gen_best['best_cost']:.6g} best_rej={gen_best['best_rej']} "
                f"mean_scalar={gen_best['mean_scalar']:.6g}"
            )
        except Exception:
            # protect GA loop from any unforeseen formatting errors
            print("Gen", gen, "(progress)")

        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                # attempt crossover, retrying with new parents up to N times
                max_parent_retries = 5
                child = None
                attempts = 0
                a = None
                b = None
                while attempts < max_parent_retries and child is None:
                    a = tournament_select_by_scalar(pop, scalars, tournament_k)
                    b = tournament_select_by_scalar(pop, scalars, tournament_k)
                    child = crossover_individual(a, b, max_nodes_int, make_tree_fn)
                    attempts += 1
                if child is None:
                    # fallback: clone the last selected parent `a` (ensure a is set)
                    if a is None:
                        a = tournament_select_by_scalar(pop, scalars, tournament_k)
                    child = a
            else:
                child = tournament_select_by_scalar(pop, scalars, tournament_k)

            if random.random() < mutation_rate:
                child = mutate_individual(child, max_depth, make_tree_fn, max_nodes_int)

            new_pop.append(child)

        pop = new_pop

    return best_ind, best_obj, history


def tournament_select_by_scalar(
    pop: list[Individual], scalars: list[float], k: int
) -> Individual:
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if scalars[i] < scalars[best]:
            best = i
    return pop[best]

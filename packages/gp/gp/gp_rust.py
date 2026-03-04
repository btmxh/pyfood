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
from typing import Callable, List, Sequence, Tuple

import rsimulator as rs
from dvrptw.simulator import RustSimulator
from dvrptw.evaluator import Evaluator, StarNormEvaluator
from dvrptw.instance import DVRPTWInstance

from .evolve import make_full, make_grow
from .engine import splice, random_subtree_range


Individual = Tuple[rs.FlatGpTree, rs.FlatGpTree, rs.FlatGpTree]


def make_individual(depth: int) -> Individual:
    """Create one individual: (routing, sequencing, reject)."""

    # Use the same initializer for each role
    def mk(d: int) -> rs.FlatGpTree:
        return make_grow(d) if random.random() < 0.5 else make_full(d)

    return (mk(depth), mk(depth), mk(depth))


def init_population(pop_size: int, max_depth: int) -> List[Individual]:
    return [make_individual(1 + (i % max_depth)) for i in range(pop_size)]


def subtree_crossover(a: rs.FlatGpTree, b: rs.FlatGpTree) -> rs.FlatGpTree:
    a_ops: bytes = a.ops
    b_ops: bytes = b.ops
    a_s, a_e = random_subtree_range(a_ops)
    b_s, b_e = random_subtree_range(b_ops)
    child_ops = splice(a_ops, a_s, a_e, b_ops[b_s : b_e + 1])
    return rs.FlatGpTree.from_bytes(child_ops)


def crossover_individual(parent_a: Individual, parent_b: Individual) -> Individual:
    # crossover per-role
    r = subtree_crossover(parent_a[0], parent_b[0])
    s = subtree_crossover(parent_a[1], parent_b[1])
    rej = subtree_crossover(parent_a[2], parent_b[2])
    return (r, s, rej)


def mutate_individual(
    ind: Individual,
    max_subtree_depth: int,
    make_tree_fn: Callable[[int], rs.FlatGpTree],
) -> Individual:
    # pick a role to mutate
    role = random.randrange(3)
    trees = list(ind)
    # reuse engine mutation logic: replace a random subtree with a new small tree
    ops = trees[role].ops
    s, e = random_subtree_range(ops)
    replacement = make_tree_fn(random.randint(0, max_subtree_depth)).to_bytes()
    new_ops = splice(ops, s, e, replacement)
    trees[role] = rs.FlatGpTree.from_bytes(new_ops)
    return tuple(trees)  # type: ignore[return-value]


def evaluate_individual(
    individual: Individual,
    instance: DVRPTWInstance,
) -> Tuple[float, int]:
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
) -> Tuple[Individual, Tuple[float, int]]:
    """Evolve GP trees where fitness is a DVRPTW simulation result.

    Returns the best individual and its raw objectives (travel_cost, rejected).
    The `evaluator` collapses the two objectives to a scalar for selection; if
    omitted the StarNormEvaluator with equal weights is used.
    """
    if evaluator is None:
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)

    if make_tree_fn is None:

        def _mk(d: int) -> rs.FlatGpTree:
            return make_grow(d) if random.random() < 0.5 else make_full(d)

        make_tree_fn = _mk

    pop = init_population(pop_size, max_depth)

    best_ind: Individual = pop[0]
    best_scalar = float("inf")
    best_obj: Tuple[float, int] = (float("inf"), 10**9)

    history: List[dict] = []

    for gen in range(generations):
        # evaluate population sequentially (could be parallelised later)
        raw_objs: List[Tuple[float, int]] = [
            evaluate_individual(ind, instance) for ind in pop
        ]
        scalars: List[float] = [evaluator.scalar(o[0], o[1]) for o in raw_objs]

        # update best
        for ind, s, o in zip(pop, scalars, raw_objs):
            if s < best_scalar:
                best_scalar = s
                best_ind = ind
                best_obj = o

        # create next generation
        ranked = sorted(range(len(pop)), key=lambda i: scalars[i])
        new_pop: List[Individual] = [pop[i] for i in ranked[:elitism]]

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

        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                a = tournament_select_by_scalar(pop, scalars, tournament_k)
                b = tournament_select_by_scalar(pop, scalars, tournament_k)
                child = crossover_individual(a, b)
            else:
                child = tournament_select_by_scalar(pop, scalars, tournament_k)

            if random.random() < mutation_rate:
                child = mutate_individual(child, max_depth, make_tree_fn)

            new_pop.append(child)

        pop = new_pop

    return best_ind, best_obj, history


def tournament_select_by_scalar(
    pop: Sequence[Individual], scalars: Sequence[float], k: int
) -> Individual:
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if scalars[i] < scalars[best]:
            best = i
    return pop[best]

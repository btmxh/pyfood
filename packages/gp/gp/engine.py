"""Simple, self-contained GP engine (pure Python) using FlatGpTree bytes.

This engine interprets the compact postfix opcode representation exported by
`rsimulator.FlatGpTree.ops` and implements a toy GP loop (tournament
selection, subtree crossover, subtree mutation, elitism). It intentionally
keeps evaluation in Python so the package is self-contained and easy to run
without calling into Rust for fitness evaluation.

Notes:
- Token encoding mirrors `packages/rsimulator/src/strategies/gp_tree.rs`:
  * MSB 0 => const (7-bit payload: [S:MMM:EEE])
  * bits 7..6 = 0b10 => terminal id (0..=63)
  * bits 7..6 = 0b11 => op id (0=add,1=sub,2=mul,3=div)
- The engine uses Python float arithmetic; protected division returns 1.0
  when divisor is zero (same semantics as Rust implementation).
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import rsimulator as rs

OpId = int


# --- low-level token helpers -------------------------------------------------


def token_is_const(b: int) -> bool:
    return (b & 0x80) == 0


def token_is_terminal(b: int) -> bool:
    return (b & 0xC0) == 0x80


def token_is_op(b: int) -> bool:
    return (b & 0xC0) == 0xC0


def decode_const7(payload: int) -> float:
    """Decode 7-bit const payload to float (same as Rust `decode_const7`)."""
    if payload == 0:
        return 0.0
    sign = -1.0 if (payload & 0x40) != 0 else 1.0
    mmm = (payload >> 3) & 0x07
    eee = payload & 0x07
    val = sign * (1.0 + mmm / 8.0) * (2.0 ** (eee - 3))
    return float(val)


def apply_op(op_id: OpId, left: float, right: float) -> float:
    if op_id == 0:
        return left + right
    if op_id == 1:
        return left - right
    if op_id == 2:
        return left * right
    if op_id == 3:
        # protected div
        return 1.0 if right == 0.0 else (left / right)
    return 0.0


# --- tree evaluation (postfix ops) ------------------------------------------


def eval_scalar_from_slice(ops: bytes, terminals: Sequence[float]) -> float:
    """Evaluate tree encoded by `ops` (bytes) for a single terminal vector.

    `terminals[id]` should provide the value for terminal id `id`.
    """
    stack: List[float] = []
    for b in ops:
        if token_is_const(b):
            stack.append(decode_const7(b & 0x7F))
        elif token_is_terminal(b):
            idx = b & 0x3F
            v = terminals[idx] if idx < len(terminals) else 0.0
            stack.append(float(v))
        else:
            right = stack.pop() if stack else 0.0
            left = stack.pop() if stack else 0.0
            stack.append(apply_op(b & 0x3F, left, right))
    return stack.pop() if stack else 0.0


def eval_batch(ops: bytes, terminal_matrix: Sequence[Sequence[float]]) -> List[float]:
    """Evaluate `ops` for a batch of terminals (list of terminal vectors).

    Returns a list of floats, one per input row.
    """
    out: List[float] = []
    for terminals in terminal_matrix:
        out.append(eval_scalar_from_slice(ops, terminals))
    return out


# --- subtree utilities (postfix) --------------------------------------------


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


def subtree_range(ops: bytes, end: int) -> Tuple[int, int]:
    return (subtree_start(ops, end), end)


def random_subtree_range(ops: bytes) -> Tuple[int, int]:
    if not ops:
        return (0, -1)
    end = random.randrange(len(ops))
    start = subtree_start(ops, end)
    return (start, end)


# --- genetic operators ------------------------------------------------------


def splice(ops: bytes, a_start: int, a_end: int, replacement: bytes) -> bytes:
    return ops[:a_start] + replacement + ops[a_end + 1 :]


def crossover(parent_a: rs.FlatGpTree, parent_b: rs.FlatGpTree) -> rs.FlatGpTree:
    a_ops: bytes = parent_a.ops
    b_ops: bytes = parent_b.ops
    a_s, a_e = random_subtree_range(a_ops)
    b_s, b_e = random_subtree_range(b_ops)
    child_ops = splice(a_ops, a_s, a_e, b_ops[b_s : b_e + 1])
    return rs.FlatGpTree.from_bytes(child_ops)


def mutate(tree: rs.FlatGpTree, max_subtree_depth: int, make_tree_fn) -> rs.FlatGpTree:
    """Replace a random subtree with a newly generated random subtree.

    `make_tree_fn(depth)` should return an `rs.FlatGpTree` instance.
    """
    ops = tree.ops
    s, e = random_subtree_range(ops)
    # create a small random replacement
    replacement = make_tree_fn(random.randint(0, max_subtree_depth)).to_bytes()
    out = splice(ops, s, e, replacement)
    return rs.FlatGpTree.from_bytes(out)


# --- selection / population utilities ---------------------------------------


def tournament_select(
    pop: Sequence[rs.FlatGpTree], fitness: Sequence[float], k: int
) -> rs.FlatGpTree:
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return pop[best]


# --- high level GP loop ----------------------------------------------------


def evaluate_population(
    pop: Sequence[rs.FlatGpTree], terminal_matrix: Sequence[Sequence[Sequence[float]]]
) -> List[float]:
    """Evaluate each tree on a list of evaluation cases.

    `terminal_matrix` is a sequence of cases; each case is a sequence of terminal
    values (length should be >= largest terminal id used). Fitness is the sum
    of scores across cases (higher is better).
    """
    fitness: List[float] = []
    for tree in pop:
        ops = tree.ops
        total = 0.0
        for case in terminal_matrix:
            total += eval_scalar_from_slice(ops, case)
        fitness.append(total)
    return fitness


def run_gp(
    evaluate_cases: Sequence[Sequence[float]],
    pop_size: int = 100,
    max_depth: int = 4,
    generations: int = 20,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.05,
    elitism: int = 1,
    make_tree_fn=None,
) -> Tuple[rs.FlatGpTree, float]:
    """Run a simple generational GP and return the best tree and its fitness.

    - `evaluate_cases`: sequence of terminal vectors (one per case)
    - `make_tree_fn(depth)` constructs a random tree (defaults to grow/full mix)
    """
    if make_tree_fn is None:
        from .evolve import make_grow, make_full

        def _make(d: int):
            return make_grow(d) if random.random() < 0.5 else make_full(d)

        make_tree_fn = _make

    # initialize population
    pop: List[rs.FlatGpTree] = [
        make_tree_fn(1 + (i % max_depth)) for i in range(pop_size)
    ]

    best_tree = pop[0]
    best_fit = float("-inf")

    for gen in range(generations):
        fitness = evaluate_population(pop, evaluate_cases)
        # record best
        for t, f in zip(pop, fitness):
            if f > best_fit:
                best_fit = f
                best_tree = t

        new_pop: List[rs.FlatGpTree] = []
        # elitism
        ranked = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
        for i in range(min(elitism, pop_size)):
            new_pop.append(pop[ranked[i]])

        # generate rest
        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                a = tournament_select(pop, fitness, tournament_k)
                b = tournament_select(pop, fitness, tournament_k)
                child = crossover(a, b)
            else:
                child = tournament_select(pop, fitness, tournament_k)

            if random.random() < mutation_rate:
                child = mutate(child, max_depth, make_tree_fn)

            new_pop.append(child)

        pop = new_pop

    _ = best_tree
    # final evaluation
    fitness = evaluate_population(pop, evaluate_cases)
    best_idx = int(max(range(len(pop)), key=lambda i: fitness[i]))
    return pop[best_idx], fitness[best_idx]

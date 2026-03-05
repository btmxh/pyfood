import random
from typing import Callable, List

import rsimulator as rs

# binary operator factories
BINOPS: List[Callable[[rs.FlatGpTree, rs.FlatGpTree], rs.FlatGpTree]] = [
    rs.flat_gp_add,
    rs.flat_gp_sub,
    rs.flat_gp_mul,
    rs.flat_gp_div,
]

# terminal factories (call to create a leaf)
TERMINALS: List[Callable[[], rs.FlatGpTree]] = [
    rs.flat_gp_travel_time,
    rs.flat_gp_window_earliest,
    rs.flat_gp_window_latest,
    rs.flat_gp_time_until_due,
    rs.flat_gp_demand,
    rs.flat_gp_current_load,
    rs.flat_gp_remaining_capacity,
    rs.flat_gp_release_time,
]


def random_terminal_or_const() -> rs.FlatGpTree:
    if random.random() < 0.7:
        return random.choice(TERMINALS)()
    else:
        return rs.flat_gp_const(random.uniform(-5.0, 5.0))


def make_full(depth: int) -> rs.FlatGpTree:
    if depth <= 0:
        return random_terminal_or_const()
    left = make_full(depth - 1)
    right = make_full(depth - 1)
    op = random.choice(BINOPS)
    return op(left, right)


def make_grow(depth: int) -> rs.FlatGpTree:
    if depth <= 0:
        return random_terminal_or_const()
    if random.random() < 0.5:
        return random_terminal_or_const()
    left = make_grow(depth - 1)
    right = make_grow(depth - 1)
    op = random.choice(BINOPS)
    return op(left, right)


def ramped_half_and_half(pop_size: int, max_depth: int) -> List[rs.FlatGpTree]:
    pop: List[rs.FlatGpTree] = []
    for i in range(pop_size):
        depth = 1 + (i % max_depth)
        if random.random() < 0.5:
            pop.append(make_full(depth))
        else:
            pop.append(make_grow(depth))
    return pop

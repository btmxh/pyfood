"""Toy GP implementation using rsimulator FlatGpTree."""

from .evolve import (
    ramped_half_and_half,
    random_terminal_or_const,
    make_full,
    make_grow,
)
from .engine import run_gp, evaluate_population, crossover, mutate
from .gp_rust import (
    run_gp_rust,
    evaluate_individual,
    make_individual,
    init_population,
)

__all__ = [
    "ramped_half_and_half",
    "random_terminal_or_const",
    "make_full",
    "make_grow",
    "run_gp",
    "evaluate_population",
    "crossover",
    "mutate",
    "run_gp_rust",
    "evaluate_individual",
    "make_individual",
    "init_population",
]

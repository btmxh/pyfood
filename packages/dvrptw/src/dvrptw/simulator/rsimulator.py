"""Typed public wrapper around the external `rsimulator` extension.

This module mirrors the runtime symbols exported by the compiled
``rsimulator`` extension but provides local, project-level typing for
use throughout the ``dvrptw`` package and for downstream consumers.

At runtime the module delegates directly to the compiled extension so
behaviour is identical to importing ``rsimulator``. When type checking
the module exposes richer types so tools can reason about the API.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Expose typed signatures for the typechecker without importing the
    # external extension stubs.
    from .state import SchedulerAction

    class NativeStrategyWrapper:  # pragma: no cover - typing only
        """Opaque marker for a native (Rust) strategy wrapper."""

    class NativeCallbackWrapper:  # pragma: no cover - typing only
        def __call__(
            self, time: float, action: SchedulerAction, auto: bool
        ) -> None: ...

    class FlatGpTree:  # pragma: no cover - typing only
        ...

    class Simulator:  # pragma: no cover - typing only
        def __init__(
            self, instance: Any, strategy: Any, action_callback: Any = None
        ) -> None: ...
        def run(self) -> dict[str, Any]: ...

    def greedy_strategy() -> NativeStrategyWrapper: ...
    def composable_strategy(router: Any, scheduler: Any) -> NativeStrategyWrapper: ...
    def batch_composable_strategy(
        router: Any, scheduler: Any, slot_size: float
    ) -> NativeStrategyWrapper: ...

    # GP helpers
    def gp_strategy(
        routing: FlatGpTree, sequencing: FlatGpTree, reject: FlatGpTree
    ) -> NativeStrategyWrapper: ...
    def flat_gp_const(value: float) -> FlatGpTree: ...
    def flat_gp_add(a: FlatGpTree, b: FlatGpTree) -> FlatGpTree: ...
    def flat_gp_sub(a: FlatGpTree, b: FlatGpTree) -> FlatGpTree: ...
    def flat_gp_mul(a: FlatGpTree, b: FlatGpTree) -> FlatGpTree: ...
    def flat_gp_div(a: FlatGpTree, b: FlatGpTree) -> FlatGpTree: ...
    def flat_gp_travel_time() -> FlatGpTree: ...
    def flat_gp_window_earliest() -> FlatGpTree: ...
    def flat_gp_window_latest() -> FlatGpTree: ...
    def flat_gp_time_until_due() -> FlatGpTree: ...
    def flat_gp_demand() -> FlatGpTree: ...
    def flat_gp_current_load() -> FlatGpTree: ...
    def flat_gp_remaining_capacity() -> FlatGpTree: ...
    def flat_gp_release_time() -> FlatGpTree: ...

else:
    # Runtime: delegate to the compiled extension for identical behaviour.
    import rsimulator as _rsim

    NativeStrategyWrapper = _rsim.NativeStrategyWrapper
    NativeCallbackWrapper = _rsim.NativeCallbackWrapper
    FlatGpTree = _rsim.FlatGpTree
    Simulator = _rsim.Simulator

    greedy_strategy = _rsim.greedy_strategy
    composable_strategy = _rsim.composable_strategy
    batch_composable_strategy = _rsim.batch_composable_strategy

    # GP helpers
    gp_strategy = _rsim.gp_strategy
    flat_gp_const = _rsim.flat_gp_const
    flat_gp_add = _rsim.flat_gp_add
    flat_gp_sub = _rsim.flat_gp_sub
    flat_gp_mul = _rsim.flat_gp_mul
    flat_gp_div = _rsim.flat_gp_div
    flat_gp_travel_time = _rsim.flat_gp_travel_time
    flat_gp_window_earliest = _rsim.flat_gp_window_earliest
    flat_gp_window_latest = _rsim.flat_gp_window_latest
    flat_gp_time_until_due = _rsim.flat_gp_time_until_due
    flat_gp_demand = _rsim.flat_gp_demand
    flat_gp_current_load = _rsim.flat_gp_current_load
    flat_gp_remaining_capacity = _rsim.flat_gp_remaining_capacity
    flat_gp_release_time = _rsim.flat_gp_release_time

__all__ = [
    "NativeStrategyWrapper",
    "NativeCallbackWrapper",
    "FlatGpTree",
    "Simulator",
    "greedy_strategy",
    "composable_strategy",
    "batch_composable_strategy",
    # GP helpers
    "gp_strategy",
    "flat_gp_const",
    "flat_gp_add",
    "flat_gp_sub",
    "flat_gp_mul",
    "flat_gp_div",
    "flat_gp_travel_time",
    "flat_gp_window_earliest",
    "flat_gp_window_latest",
    "flat_gp_time_until_due",
    "flat_gp_demand",
    "flat_gp_current_load",
    "flat_gp_remaining_capacity",
    "flat_gp_release_time",
]

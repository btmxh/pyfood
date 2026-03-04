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

    class NativeDispatchStrategy:
        """Opaque marker for a native (Rust) dispatching strategy.

        Note: At runtime the compiled extension names this type
        `NativeDispatchStrategy`. The aliasing in the non-TYPE_CHECKING
        branch ensures compatibility with both the Python-facing name and
        the typed marker used during static analysis.
        """

    class NativeEventCallback:
        """Opaque marker for a native (Rust) event callback."""

    def python_dispatching_strategy(strategy: Any) -> NativeDispatchStrategy: ...
    def python_event_callback(callback: Any) -> NativeEventCallback: ...

    class FlatGpTree:  # pragma: no cover - typing only
        ...

    class Simulator:  # pragma: no cover - typing only
        def __init__(
            self, instance: Any, strategy: Any, action_callback: Any = None
        ) -> None: ...
        def run(self) -> dict[str, Any]: ...

    # GP helpers
    def gp_strategy(
        routing: FlatGpTree, sequencing: FlatGpTree, reject: FlatGpTree
    ) -> NativeDispatchStrategy: ...
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

    # Some builds/export variants name the pyclass wrappers differently
    # (e.g. `NativeStrategyWrapper` / `NativeCallbackWrapper`). Accept both
    # to remain robust across local build variations.
    NativeDispatchStrategy = _rsim.NativeDispatchStrategy
    NativeEventCallback = _rsim.NativeEventCallback
    FlatGpTree = _rsim.FlatGpTree
    Simulator = _rsim.Simulator

    # Runtime function exported from the extension is named
    # `python_dispatch_strategy` (no "ing"). Provide the typed alias
    # `python_dispatching_strategy` for compatibility with the Python-side
    # code that expects that name.
    # Backwards/forwards-compat: compiled extension historically exported
    # `python_dispatch_strategy` and `python_event_callback` (no "ing").
    # Accept either name to be robust across builds.
    python_dispatching_strategy = getattr(
        _rsim,
        "python_dispatching_strategy",
        getattr(_rsim, "python_dispatch_strategy", None),
    )
    python_event_callback = getattr(
        _rsim, "python_event_callback", getattr(_rsim, "python_event_callback", None)
    )
    if python_dispatching_strategy is None or python_event_callback is None:
        raise AttributeError(
            "rsimulator extension missing python dispatch/callback helpers"
        )

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
    "NativeDispatchStrategy",
    "NativeEventCallback",
    "FlatGpTree",
    "Simulator",
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

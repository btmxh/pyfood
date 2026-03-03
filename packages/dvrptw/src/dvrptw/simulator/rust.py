"""Rust-backed simulation engine and its Python-side adapters."""

from typing import Callable, cast

from .rsimulator import (  # noqa: F401
    Simulator as _RustSimulator,
    NativeStrategyWrapper,
    NativeCallbackWrapper,
    greedy_strategy,
)

from .events import RejectEvent, SchedulerAction
from ..instance import DVRPTWInstance, Request
from .base import Simulator
from ..solution import Solution
from .state import (
    DispatchingStrategy,
    SimulationMetrics,
    SimulationResult,
    SimulationState,
    VehicleState,
)


class RustSimulator(Simulator):
    """Rust-backed DVRPTW simulation engine via the ``rsimulator`` extension.

    Accepts both Python ``DispatchingStrategy`` objects and
    ``NativeStrategyWrapper`` instances returned by factory functions such as
    :func:`rsimulator.greedy_strategy`.  When a native wrapper is passed the
    simulation hot path runs entirely in Rust with no GIL acquisition.
    """

    def __init__(
        self,
        instance: DVRPTWInstance,
        strategy: "DispatchingStrategy | NativeStrategyWrapper",
        action_callback: Callable[[float, SchedulerAction, bool], None] | None = None,
    ):
        super().__init__(instance, strategy, action_callback)

        # If the strategy is already a NativeStrategyWrapper, pass it directly —
        # the Rust Simulator.__new__ will extract the Box<dyn RustStrategy> and
        # run the entire hot path without touching the GIL.
        #
        # We use a type-name check rather than isinstance() as a defensive
        # measure in case the type object identity is ever broken by unusual
        # import paths.  The name is stable and sufficient for this purpose.
        #
        # Otherwise wrap it in _RustStrategyAdapter so it receives a proper
        # SimulationState object (with attribute access) rather than the raw dict
        # that the Rust-side PyStrategyAdapter produces.
        if type(strategy).__name__ == "NativeStrategyWrapper":
            effective_strategy = strategy
        else:
            # `strategy` here is known to be a Python DispatchingStrategy
            # (the NativeStrategyWrapper case is handled above).  Narrow the
            # type for static checkers with an explicit cast so linters
            # understand we are passing a DispatchingStrategy instance.
            from typing import cast

            effective_strategy = _RustStrategyAdapter(
                cast(DispatchingStrategy, strategy), instance
            )

        # Same logic for the callback: NativeCallbackWrapper passes through;
        # a Python callable is wrapped so it receives typed action objects.
        if action_callback is None:
            effective_callback = None
        elif isinstance(action_callback, NativeCallbackWrapper):
            effective_callback = action_callback
        else:
            effective_callback = _RustCallbackAdapter(action_callback)

        self._rust = _RustSimulator(instance, effective_strategy, effective_callback)

    def run(self) -> SimulationResult:
        raw = self._rust.run()

        solution = Solution(
            routes=raw["solution"]["routes"],
            service_times=raw["solution"]["service_times"],
        )
        metrics = SimulationMetrics(
            total_travel_cost=raw["metrics"]["total_travel_cost"],
            rejected=raw["metrics"]["rejected"],
        )

        # Reconstruct served/rejected from the solution so callers can inspect them.
        served: set[int] = set()
        for route in solution.routes:
            served.update(route)
        all_request_ids = {r.id for r in self.instance.requests if not r.is_depot}
        self.served_requests = served
        self.rejected_requests = all_request_ids - served

        return SimulationResult(solution=solution, metrics=metrics)


class _RustStrategyAdapter:
    """Wraps a Python DispatchingStrategy so it receives a SimulationState object.

    The Rust-side ``PyStrategyAdapter`` calls ``strategy.next_events(state_dict)``
    where ``state_dict`` is a plain Python dict.  This adapter converts that dict
    into the proper ``SimulationState`` dataclass (with ``VehicleState`` objects)
    before forwarding to the wrapped strategy, keeping user strategies portable
    across both backends.

    ``released_requests`` in the dict now contains ``{id: id}`` sentinel entries
    (the Rust snapshot holds only IDs).  We use the keys to look up the full
    ``Request`` objects from the Python instance, as before.
    """

    def __init__(self, strategy: DispatchingStrategy, instance: DVRPTWInstance):
        self._strategy = strategy
        self._instance = instance

    def next_events(self, state_dict: dict) -> list[SchedulerAction]:
        vehicles = [
            VehicleState(
                vehicle_id=v["vehicle_id"],
                position=v["position"],
                current_load=v["current_load"],
                available_at=v["available_at"],
                route=list(v["route"]),
                service_times=list(v["service_times"]),
            )
            for v in state_dict["vehicles"]
        ]

        # released_requests dict keys are request IDs; values are sentinel ints
        # (the snapshot only carries IDs).  Look up the full Python Request objects.
        released: dict[int, Request] = {}
        for req_id_raw in state_dict["released_requests"]:
            req_id = int(req_id_raw)
            try:
                released[req_id] = self._instance.get_request(req_id)
            except (KeyError, ValueError):
                pass

        state = SimulationState(
            time=state_dict["time"],
            pending_requests=set(state_dict["pending_requests"]),
            served_requests=set(state_dict["served_requests"]),
            rejected_requests=set(state_dict["rejected_requests"]),
            vehicles=vehicles,
            released_requests=released,
        )
        return self._strategy.next_events(state)


class _RustCallbackAdapter:
    """Wraps a Python callback so it receives typed action objects.

    The Rust simulator now fires callbacks with proper Python dataclass instances
    (``DispatchEvent``, ``WaitEvent``, ``RejectEvent``) via ``sim_action_to_py``.
    This adapter is kept for defensive compatibility; the dict-action branch is
    no longer exercised by the current Rust implementation.
    """

    def __init__(
        self, callback: Callable[[float, SchedulerAction, bool], None]
    ) -> None:
        self._callback = callback

    def __call__(self, time: float, action: object, auto: bool) -> None:
        converted: SchedulerAction
        # Defensive: convert plain dicts in case of a downlevel rsimulator build.
        if isinstance(action, dict):
            if action.get("type") == "reject":  # type: ignore[call-overload]
                converted = RejectEvent(request_id=cast(int, action["request_id"]))  # type: ignore[call-overload]
            else:
                converted = action  # type: ignore[assignment]
        else:
            converted = action  # type: ignore[assignment]
        self._callback(time, converted, auto)

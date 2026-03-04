"""Rust-backed simulation engine and its Python-side adapters."""

from typing import Callable, cast, override

from .rsimulator import (
    Simulator as _RustSimulator,
    NativeDispatchStrategy,
    NativeEventCallback,
    python_dispatching_strategy,
    python_event_callback,
)

from .events import DispatchEvent, WaitEvent, RejectEvent, SchedulerAction
from ..instance import DVRPTWInstance, Request, TimeWindow
from .base import Simulator, PythonDispatchStrategy, PythonEventCallback
from ..solution import Solution
from .state import (
    SimulationMetrics,
    SimulationResult,
    VehicleSnapshot,
    SimulationSnapshot,
    InstanceView,
    VehicleSpec,
)


class RustSimulator(Simulator):
    """Rust-backed DVRPTW simulation engine via the ``rsimulator`` extension.

    Accepts both Python ``DispatchStrategy`` objects and
    ``NativeDispatchStrategy`` instances returned by factory functions such as
    :func:`rsimulator.greedy_strategy`.  When a native wrapper is passed the
    simulation hot path runs entirely in Rust with no GIL acquisition.
    """

    def __init__(
        self,
        instance: DVRPTWInstance,
        strategy: NativeDispatchStrategy,
        action_callback: NativeEventCallback | None = None,
    ):
        super().__init__(instance, strategy, action_callback)
        self._rust = _RustSimulator(instance, strategy, action_callback)

    @classmethod
    @override
    def wrap_strategy(cls, strategy: PythonDispatchStrategy) -> NativeDispatchStrategy:
        return python_dispatching_strategy(NativeStrategyAdapter(strategy))

    @classmethod
    @override
    def wrap_callback(cls, callback: PythonEventCallback) -> NativeEventCallback:
        return python_event_callback(
            NativeCallbackAdapter(callback or (lambda *_: None))
        )

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


class NativeStrategyAdapter:
    """Wraps a Python DispatchStrategy so it receives a SimulationState object.

    The Rust-side ``PyStrategyAdapter`` calls ``strategy.next_events(state_dict)``
    where ``state_dict`` is a plain Python dict.  This adapter converts that dict
    into the proper ``SimulationState`` dataclass (with ``VehicleState`` objects)
    before forwarding to the wrapped strategy, keeping user strategies portable
    across both backends.

    ``released_requests`` in the dict now contains ``{id: id}`` sentinel entries
    (the Rust snapshot holds only IDs).  We use the keys to look up the full
    ``Request`` objects from the Python instance, as before.
    """

    def __init__(self, strategy: PythonDispatchStrategy):
        self._strategy = strategy

    def next_events(
        self, state_dict: dict, instance_view_dict: dict
    ) -> list[SchedulerAction]:
        vehicles = [
            VehicleSnapshot(
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

        state = SimulationSnapshot(
            time=state_dict["time"],
            pending=set(state_dict["pending_requests"]),
            served=set(state_dict["served_requests"]),
            rejected=set(state_dict["rejected_requests"]),
            vehicles=vehicles,
        )

        released_requests: dict[int, Request] = {}
        for id, req_dict in instance_view_dict["released_requests"].items():
            req = Request(
                id=req_dict["id"],
                position=tuple(req_dict["position"]),
                demand=req_dict["demand"],
                time_window=TimeWindow(
                    req_dict["time_window"]["earliest"],
                    req_dict["time_window"]["latest"],
                ),
                service_time=req_dict["service_time"],
                release_time=req_dict.get("release_time", 0.0),
                is_depot=req_dict.get("is_depot", False),
            )
            released_requests[id] = req

        instance_view = InstanceView(
            released_requests=released_requests,
            vehicles=[
                VehicleSpec(
                    id=v["id"],
                    capacity=v["capacity"],
                    speed=v["speed"],
                )
                for v in instance_view_dict["vehicles"]
            ],
            depot_id=instance_view_dict["depot_id"],
        )
        return self._strategy.next_events(state, instance_view)


class NativeCallbackAdapter:
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

    def __call__(self, time: float, action: dict[str, object], auto: bool) -> None:
        converted: SchedulerAction
        match action["type"]:
            case "dispatch":
                converted = DispatchEvent(
                    vehicle_id=cast(int, action["vehicle_id"]),
                    destination_node=cast(int, action["destination_node"]),
                )
            case "wait":
                converted = WaitEvent(until_time=cast(float, action["until_time"]))
            case "reject":
                converted = RejectEvent(request_id=cast(int, action["request_id"]))
        self._callback(time, converted, auto)

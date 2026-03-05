"""Python-side adapters and protocols for composable Rust strategies.

Usage pattern (mirrors the dispatch strategy wrapper in ``RustSimulator``):

.. code-block:: python

    from dvrptw.strategies import (
        python_routing_strategy,
        python_scheduling_strategy,
        composable_strategy,
    )

    class MyRouter:
        def route(
            self,
            request_id: int,
            vehicles: list[VehicleSnapshot],
            view: StrategyView,
        ) -> int | None:
            return vehicles[0].vehicle_id  # assign to first vehicle

    class MyScheduler:
        def schedule(
            self,
            vehicle: VehicleSnapshot,
            queue: list[int],
            view: StrategyView,
        ) -> int:
            return min(queue)  # FIFO

    strategy = composable_strategy(
        python_routing_strategy(MyRouter()),
        python_scheduling_strategy(MyScheduler()),
    )

Alternatively use ``python_composable_strategy`` as a convenience wrapper:

.. code-block:: python

    strategy = python_composable_strategy(MyRouter(), MyScheduler())
"""

from typing import Protocol

import rsimulator as rs

from dvrptw.simulator.rsimulator import (
    NativeBatchRoutingStrategy,
    NativeDispatchStrategy,
    NativeRoutingStrategy,
    NativeSchedulingStrategy,
    python_batch_routing_strategy as _rs_python_batch_routing_strategy,
    python_routing_strategy as _rs_python_routing_strategy,
    python_scheduling_strategy as _rs_python_scheduling_strategy,
)
from dvrptw.simulator.state import StrategyView, VehicleSnapshot, VehicleSpec


# ---------------------------------------------------------------------------
# Typed protocols for Python routing / scheduling sub-strategies
# ---------------------------------------------------------------------------


class PythonRoutingStrategy(Protocol):
    """Protocol for Python routers used with :func:`python_routing_strategy`.

    Receives typed dataclasses instead of raw dicts.
    """

    def route(
        self,
        request_id: int,
        vehicles: list[VehicleSnapshot],
        view: StrategyView,
    ) -> int | None:
        """Assign ``request_id`` to a vehicle or reject it.

        Returns the vehicle ID to assign to, or ``None`` to reject.
        """
        ...


class PythonSchedulingStrategy(Protocol):
    """Protocol for Python schedulers used with :func:`python_scheduling_strategy`.

    Receives typed dataclasses instead of raw dicts.
    """

    def schedule(
        self,
        vehicle: VehicleSnapshot,
        queue: list[int],
        view: StrategyView,
    ) -> int:
        """Choose the next request to dispatch for ``vehicle``.

        Must return one of the IDs present in ``queue``.
        """
        ...


class PythonBatchRoutingStrategy(Protocol):
    """Protocol for Python batch routers used with :func:`python_batch_routing_strategy`.

    Receives typed dataclasses instead of raw dicts.
    """

    def route_batch(
        self,
        requests: list[int],
        vehicles: list[VehicleSnapshot],
        view: StrategyView,
    ) -> list[tuple[int, int | None]]:
        """Assign each request to a vehicle or reject it.

        Must return exactly one ``(request_id, vehicle_id | None)`` tuple per
        entry in ``requests``.  ``None`` vehicle_id rejects the request.
        """
        ...


# ---------------------------------------------------------------------------
# Layer 2 adapters: dicts → typed Python dataclasses
# ---------------------------------------------------------------------------


class NativeRoutingAdapter:
    """Bridges the dict-based Rust protocol to a typed :class:`PythonRoutingStrategy`.

    This is Layer 2 in the two-layer wrapper chain:

    * Layer 1 (Rust → dicts): ``PyRoutingAdapter`` inside the compiled extension.
    * Layer 2 (dicts → typed): this class, which converts before calling user code.

    Normally you do not instantiate this directly — use :func:`python_routing_strategy`.
    """

    def __init__(self, router: PythonRoutingStrategy) -> None:
        self._router = router

    def route(
        self,
        request_id: int,
        vehicles: list[dict],
        instance_view: dict,
    ) -> int | None:
        typed_vehicles = [
            VehicleSnapshot(
                vehicle_id=v["vehicle_id"],
                position=v["position"],
                current_load=v["current_load"],
                available_at=v["available_at"],
                route=list(v["route"]),
                service_times=list(v["service_times"]),
            )
            for v in vehicles
        ]
        typed_view = StrategyView(
            depot_id=instance_view["depot_id"],
            vehicle_specs=[
                VehicleSpec(id=s["id"], capacity=s["capacity"], speed=s["speed"])
                for s in instance_view["vehicle_specs"]
            ],
        )
        return self._router.route(request_id, typed_vehicles, typed_view)


class NativeSchedulingAdapter:
    """Bridges the dict-based Rust protocol to a typed :class:`PythonSchedulingStrategy`.

    This is Layer 2 in the two-layer wrapper chain for schedulers.

    Normally you do not instantiate this directly — use :func:`python_scheduling_strategy`.
    """

    def __init__(self, scheduler: PythonSchedulingStrategy) -> None:
        self._scheduler = scheduler

    def schedule(
        self,
        vehicle: dict,
        queue: list[int],
        instance_view: dict,
    ) -> int:
        typed_vehicle = VehicleSnapshot(
            vehicle_id=vehicle["vehicle_id"],
            position=vehicle["position"],
            current_load=vehicle["current_load"],
            available_at=vehicle["available_at"],
            route=list(vehicle["route"]),
            service_times=list(vehicle["service_times"]),
        )
        typed_view = StrategyView(
            depot_id=instance_view["depot_id"],
            vehicle_specs=[
                VehicleSpec(id=s["id"], capacity=s["capacity"], speed=s["speed"])
                for s in instance_view["vehicle_specs"]
            ],
        )
        return self._scheduler.schedule(typed_vehicle, queue, typed_view)


class NativeBatchRoutingAdapter:
    """Bridges the dict-based Rust protocol to a typed :class:`PythonBatchRoutingStrategy`.

    This is Layer 2 in the two-layer wrapper chain for batch routers.

    Normally you do not instantiate this directly — use :func:`python_batch_routing_strategy`.
    """

    def __init__(self, router: PythonBatchRoutingStrategy) -> None:
        self._router = router

    def route_batch(
        self,
        requests: list[int],
        vehicles: list[dict],
        instance_view: dict,
    ) -> list[tuple[int, int | None]]:
        typed_vehicles = [
            VehicleSnapshot(
                vehicle_id=v["vehicle_id"],
                position=v["position"],
                current_load=v["current_load"],
                available_at=v["available_at"],
                route=list(v["route"]),
                service_times=list(v["service_times"]),
            )
            for v in vehicles
        ]
        typed_view = StrategyView(
            depot_id=instance_view["depot_id"],
            vehicle_specs=[
                VehicleSpec(id=s["id"], capacity=s["capacity"], speed=s["speed"])
                for s in instance_view["vehicle_specs"]
            ],
        )
        return self._router.route_batch(requests, typed_vehicles, typed_view)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def python_routing_strategy(router: PythonRoutingStrategy) -> NativeRoutingStrategy:
    """Wrap a typed Python router as a :class:`NativeRoutingStrategy`.

    Composes both adapter layers:

    * :class:`NativeRoutingAdapter` (Layer 2) converts dicts → typed dataclasses.
    * ``python_routing_strategy`` in the Rust extension (Layer 1) boxes the
      adapter as a ``Box<dyn RoutingStrategy>``.

    The returned :class:`NativeRoutingStrategy` can be passed to
    :func:`composable_strategy`.
    """
    return _rs_python_routing_strategy(NativeRoutingAdapter(router))


def python_scheduling_strategy(
    scheduler: PythonSchedulingStrategy,
) -> NativeSchedulingStrategy:
    """Wrap a typed Python scheduler as a :class:`NativeSchedulingStrategy`.

    Composes both adapter layers analogously to :func:`python_routing_strategy`.
    """
    return _rs_python_scheduling_strategy(NativeSchedulingAdapter(scheduler))


def python_batch_routing_strategy(
    router: PythonBatchRoutingStrategy,
) -> NativeBatchRoutingStrategy:
    """Wrap a typed Python batch router as a :class:`NativeBatchRoutingStrategy`.

    Composes both adapter layers:

    * :class:`NativeBatchRoutingAdapter` (Layer 2) converts dicts → typed dataclasses.
    * ``python_batch_routing_strategy`` in the Rust extension (Layer 1) boxes the
      adapter as a ``Box<dyn BatchRoutingStrategy>``.

    The returned :class:`NativeBatchRoutingStrategy` can be passed to
    :func:`batch_composable_strategy`.
    """
    return _rs_python_batch_routing_strategy(NativeBatchRoutingAdapter(router))


def greedy_strategy() -> NativeDispatchStrategy:
    return rs.greedy_strategy()


def composable_strategy(
    router: NativeRoutingStrategy, scheduler: NativeSchedulingStrategy
) -> NativeDispatchStrategy:
    return rs.composable_strategy(router, scheduler)


def python_composable_strategy(
    router: PythonRoutingStrategy,
    scheduler: PythonSchedulingStrategy,
) -> NativeDispatchStrategy:
    """Convenience wrapper: wraps typed Python router + scheduler and builds a composable strategy.

    Equivalent to::

        composable_strategy(
            python_routing_strategy(router),
            python_scheduling_strategy(scheduler),
        )
    """
    return composable_strategy(
        python_routing_strategy(router),
        python_scheduling_strategy(scheduler),
    )


def batch_composable_strategy(
    router: NativeBatchRoutingStrategy,
    scheduler: NativeSchedulingStrategy,
    slot_size: float,
) -> NativeDispatchStrategy:
    return rs.batch_composable_strategy(router, scheduler, slot_size)


def python_batch_composable_strategy(
    router: PythonBatchRoutingStrategy,
    scheduler: PythonSchedulingStrategy,
    slot_size: float,
) -> NativeDispatchStrategy:
    """Convenience wrapper: wraps typed Python batch router + scheduler and builds a batch composable strategy.

    Equivalent to::

        batch_composable_strategy(
            python_batch_routing_strategy(router),
            python_scheduling_strategy(scheduler),
            slot_size=slot_size,
        )
    """
    return batch_composable_strategy(
        python_batch_routing_strategy(router),
        python_scheduling_strategy(scheduler),
        slot_size,
    )

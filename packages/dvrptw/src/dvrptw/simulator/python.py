"""Pure-Python DVRPTW simulation backend."""

import heapq
from typing import Callable

from .events import DispatchEvent, RejectEvent, SchedulerAction, WaitEvent
from ..instance import DVRPTWInstance
from ..solution import Solution
from .state import (
    DispatchingStrategy,
    SimulationMetrics,
    SimulationResult,
    SimulationState,
    VehicleState,
)
from .base import Simulator


class PythonSimulator(Simulator):
    """Pure-Python DVRPTW simulation engine."""

    strategy: DispatchingStrategy  # narrowed from base — PythonSimulator always needs a Python strategy

    def __init__(
        self,
        instance: DVRPTWInstance,
        strategy: DispatchingStrategy,
        action_callback: Callable[[float, SchedulerAction, bool], None] | None = None,
    ):
        super().__init__(instance, strategy, action_callback)

        self.time = 0.0
        self.vehicles = self._init_vehicles()
        self.pending_requests = self._init_pending_requests()

        self._event_queue: list[tuple[float, str, int]] = []
        self._schedule_request_arrivals()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_vehicles(self) -> list[VehicleState]:
        depot_id = self.instance.depot_ids[0]
        return [
            VehicleState(
                vehicle_id=v.id,
                position=depot_id,
                current_load=0.0,
                available_at=0.0,
            )
            for v in self.instance.vehicles
        ]

    def _init_pending_requests(self) -> set[int]:
        return {r.id for r in self.instance.requests if not r.is_depot}

    def _schedule_request_arrivals(self) -> None:
        for req in self.instance.requests:
            if not req.is_depot:
                heapq.heappush(self._event_queue, (req.release_time, "arrival", req.id))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        while self._event_queue or any(
            v.available_at > self.time for v in self.vehicles
        ):
            self._process_next_event()
            state = self._create_state()
            for action in self.strategy.next_events(state):
                self._execute_action(action)
        return self._finalize_result()

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_next_event(self) -> None:
        vehicle_completions = [
            (v.available_at, v.vehicle_id)
            for v in self.vehicles
            if v.available_at > self.time
        ]

        if not self._event_queue and not vehicle_completions:
            return

        next_time = float("inf")
        if self._event_queue:
            next_time = min(next_time, self._event_queue[0][0])
        if vehicle_completions:
            next_time = min(next_time, min(t for t, _ in vehicle_completions))
        if next_time == float("inf"):
            return

        self.time = next_time

        while self._event_queue and self._event_queue[0][0] == self.time:
            _event_time, event_type, _event_id = heapq.heappop(self._event_queue)
            if event_type == "arrival":
                pass  # request already in pending_requests

        self._auto_reject_closed_requests()

    def _auto_reject_closed_requests(self) -> None:
        to_reject = [
            req_id
            for req_id in self.pending_requests
            if self.time > self.instance.get_request(req_id).time_window.latest
        ]
        for req_id in to_reject:
            self.pending_requests.remove(req_id)
            self.rejected_requests.add(req_id)
            if self.action_callback:
                self.action_callback(self.time, RejectEvent(request_id=req_id), True)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def _create_state(self) -> SimulationState:
        return SimulationState(
            time=self.time,
            pending_requests=self.pending_requests.copy(),
            served_requests=self.served_requests.copy(),
            rejected_requests=self.rejected_requests.copy(),
            vehicles=[self._copy_vehicle_state(v) for v in self.vehicles],
            released_requests={
                req.id: req
                for req in self.instance.requests
                if not req.is_depot and req.release_time <= self.time
            },
        )

    def _copy_vehicle_state(self, v: VehicleState) -> VehicleState:
        return VehicleState(
            vehicle_id=v.vehicle_id,
            position=v.position,
            current_load=v.current_load,
            available_at=v.available_at,
            route=v.route.copy(),
            service_times=v.service_times.copy(),
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: SchedulerAction) -> None:
        if self.action_callback:
            self.action_callback(self.time, action, False)

        if isinstance(action, DispatchEvent):
            self._execute_dispatch(action)
        elif isinstance(action, WaitEvent):
            self._execute_wait(action)
        elif isinstance(action, RejectEvent):
            self._execute_reject(action)

    def _execute_dispatch(self, event: DispatchEvent) -> None:
        vehicle = self._get_vehicle(event.vehicle_id)
        destination = self.instance.get_request(event.destination_node)

        if vehicle.available_at > self.time:
            raise ValueError(
                f"Vehicle {event.vehicle_id} is not idle at time {self.time}"
            )

        from_req = self.instance.get_request(vehicle.position)
        distance = from_req.distance_to(destination)
        from_vehicle = self.instance.vehicles[event.vehicle_id]
        travel_time = from_vehicle.travel_time(distance)
        arrival_time = self.time + travel_time

        tw = destination.time_window
        service_start = max(arrival_time, tw.earliest)

        if service_start > tw.latest:
            raise ValueError(
                f"Cannot serve request {destination.id} at vehicle {event.vehicle_id}: "
                f"service_start={service_start} > latest={tw.latest}"
            )

        vehicle.position = destination.id
        vehicle.route.append(destination.id)
        vehicle.service_times.append(service_start)
        vehicle.available_at = service_start + destination.service_time

        if not destination.is_depot:
            vehicle.current_load += destination.demand
            if vehicle.current_load > from_vehicle.capacity:
                raise ValueError(
                    f"Dispatch {event.vehicle_id} → {destination.id}: "
                    f"exceeds capacity {from_vehicle.capacity}"
                )
            if destination.id in self.pending_requests:
                self.pending_requests.remove(destination.id)
                self.served_requests.add(destination.id)
        else:
            vehicle.current_load = 0.0

    def _execute_wait(self, event: WaitEvent) -> None:
        if event.until_time <= self.time:
            raise ValueError(
                f"WaitEvent until_time={event.until_time} must be > current time={self.time}"
            )
        heapq.heappush(self._event_queue, (event.until_time, "wake", -1))

    def _execute_reject(self, event: RejectEvent) -> None:
        if event.request_id not in self.pending_requests:
            return

        for v in self.vehicles:
            if event.request_id in v.route:
                raise ValueError(
                    f"Cannot reject request {event.request_id}: "
                    f"it is being served by vehicle {v.vehicle_id}"
                )

        self.pending_requests.remove(event.request_id)
        self.rejected_requests.add(event.request_id)

    def _get_vehicle(self, vehicle_id: int) -> VehicleState:
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        raise ValueError(f"Vehicle {vehicle_id} not found")

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _finalize_result(self) -> SimulationResult:
        routes = [v.route for v in self.vehicles]
        service_times = [v.service_times for v in self.vehicles]
        solution = Solution(routes=routes, service_times=service_times)
        metrics = self._compute_metrics(solution)
        return SimulationResult(solution=solution, metrics=metrics)

    def _compute_metrics(self, solution: Solution) -> SimulationMetrics:
        total_cost = 0.0
        depot_id = self.instance.depot_ids[0]

        for route in solution.routes:
            if not route:
                continue
            nodes = (
                [self.instance.get_request(depot_id)]
                + [self.instance.get_request(r) for r in route]
                + [self.instance.get_request(depot_id)]
            )
            for a, b in zip(nodes, nodes[1:]):
                total_cost += a.distance_to(b)

        return SimulationMetrics(
            total_travel_cost=total_cost,
            rejected=len(self.rejected_requests),
        )

"""DVRPTW simulation engine."""

from dataclasses import dataclass, field
from typing import Callable, Protocol
import heapq

from .instance import DVRPTWInstance, Request
from .solution import Solution


# ============================================================================
# Events/Actions returned by strategy
# ============================================================================


@dataclass
class DispatchEvent:
    """Instructs simulator to send a vehicle to a destination node."""

    vehicle_id: int
    destination_node: int


@dataclass
class WaitEvent:
    """Instructs simulator to advance time until a specific moment."""

    until_time: float


@dataclass
class RejectEvent:
    """Instructs simulator to reject a pending request."""

    request_id: int


SchedulerAction = DispatchEvent | WaitEvent | RejectEvent


# ============================================================================
# Simulation State
# ============================================================================


@dataclass
class VehicleState:
    """Current state of a single vehicle."""

    vehicle_id: int
    position: int  # current node (request ID or depot ID 0)
    current_load: float  # total demand currently carried
    available_at: float  # time when vehicle becomes idle
    route: list[int] = field(
        default_factory=list
    )  # planned sequence of stops (excluding depot)
    service_times: list[float] = field(default_factory=list)  # service start times


@dataclass
class SimulationState:
    """Current state of the simulation (strategy only sees released requests)."""

    time: float
    pending_requests: set[int]  # request IDs not yet rejected/completed
    served_requests: set[int]  # request IDs that have been served
    rejected_requests: set[int]  # request IDs that have been rejected
    vehicles: list[VehicleState]
    released_requests: dict[
        int, Request
    ]  # Only released (not depot, release_time <= time) requests


class DispatchingStrategy(Protocol):
    """Protocol for dispatching strategy implementations."""

    def next_events(self, state: SimulationState) -> list[SchedulerAction]:
        """
        Called when simulator needs next actions from strategy.

        Returns a list of DispatchEvent, WaitEvent, or RejectEvent.
        """
        ...


# ============================================================================
# Simulation Metrics
# ============================================================================


@dataclass
class SimulationMetrics:
    """Performance metrics from simulation."""

    total_travel_cost: float
    accepted: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for serialization."""
        return {
            "total_travel_cost": self.total_travel_cost,
            "accepted": self.accepted,
        }


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    solution: Solution
    metrics: SimulationMetrics


# ============================================================================
# Simulator Engine
# ============================================================================


class Simulator:
    """DVRPTW Simulation Engine."""

    def __init__(
        self,
        instance: DVRPTWInstance,
        strategy: DispatchingStrategy,
        action_callback: Callable[[float, SchedulerAction, bool], None] | None = None,
    ):
        """
        Initialize simulator.

        Args:
            instance: DVRPTW problem instance
            strategy: Dispatching strategy implementation
            action_callback: Optional callback invoked for each action (time, action)
        """
        self.instance = instance
        self.strategy = strategy
        self.action_callback = action_callback

        # Validate instance before simulation
        self.instance.validate()

        # Simulation state
        self.time = 0.0
        self.vehicles = self._init_vehicles()
        self.pending_requests = self._init_pending_requests()
        self.served_requests: set[int] = set()
        self.rejected_requests: set[int] = set()

        # Event queue for automatic triggers (request arrivals)
        self._event_queue: list[tuple[float, str, int]] = []
        self._schedule_request_arrivals()

    def _init_vehicles(self) -> list[VehicleState]:
        """Initialize vehicle states."""
        vehicles = []
        depot_id = self.instance.depot_ids[0]
        for v in self.instance.vehicles:
            vehicles.append(
                VehicleState(
                    vehicle_id=v.id,
                    position=depot_id,
                    current_load=0.0,
                    available_at=0.0,
                )
            )
        return vehicles

    def _init_pending_requests(self) -> set[int]:
        """Initialize pending requests (exclude depot)."""
        return {r.id for r in self.instance.requests if not r.is_depot}

    def _schedule_request_arrivals(self) -> None:
        """Schedule all request arrival events."""
        for req in self.instance.requests:
            if not req.is_depot:
                heapq.heappush(self._event_queue, (req.release_time, "arrival", req.id))

    def run(self) -> SimulationResult:
        """Execute the simulation and return results."""
        while self._event_queue or any(
            v.available_at > self.time for v in self.vehicles
        ):
            # Process next event(s)
            self._process_next_event()

            # Ask strategy for actions
            state = self._create_state()
            actions = self.strategy.next_events(state)

            # Execute actions
            for action in actions:
                self._execute_action(action)

        # Compute final solution and metrics
        return self._finalize_result()

    def _process_next_event(self) -> None:
        """Process the next event from the event queue or vehicle completions."""
        # Check for vehicle completion events
        vehicle_completions = [
            (v.available_at, v.vehicle_id)
            for v in self.vehicles
            if v.available_at > self.time
        ]

        if not self._event_queue and not vehicle_completions:
            return

        # Determine next event time
        next_time = float("inf")

        if self._event_queue:
            next_time = min(next_time, self._event_queue[0][0])

        if vehicle_completions:
            next_time = min(next_time, min(t for t, _ in vehicle_completions))

        if next_time == float("inf"):
            return

        self.time = next_time

        # Process all events at this time
        while self._event_queue and self._event_queue[0][0] == self.time:
            event_time, event_type, event_id = heapq.heappop(self._event_queue)
            if event_type == "arrival":
                # Request arrived - it becomes pending (will be seen by strategy)
                if event_id in self.pending_requests:
                    pass  # Already in pending

        # Auto-reject requests with closed time windows
        self._auto_reject_closed_requests()

    def _auto_reject_closed_requests(self) -> None:
        """Auto-reject any requests whose time window has closed."""
        to_reject = []
        for req_id in self.pending_requests:
            req = self.instance.get_request(req_id)
            if self.time > req.time_window.latest:
                to_reject.append(req_id)

        for req_id in to_reject:
            self.pending_requests.remove(req_id)
            self.rejected_requests.add(req_id)
            # If action_callback exists, call with auto=True since these are simulator-driven
            if self.action_callback:
                self.action_callback(self.time, RejectEvent(request_id=req_id), True)

    def _create_state(self) -> SimulationState:
        """Create current simulation state for strategy."""
        return SimulationState(
            time=self.time,
            pending_requests=self.pending_requests.copy(),
            served_requests=self.served_requests.copy(),
            rejected_requests=self.rejected_requests.copy(),
            vehicles=[self._copy_vehicle_state(v) for v in self.vehicles],
            released_requests={
                req.id: req
                for req in self.instance.requests
                if (not req.is_depot and req.release_time <= self.time)
            },
        )

    def _copy_vehicle_state(self, v: VehicleState) -> VehicleState:
        """Create a copy of vehicle state."""
        return VehicleState(
            vehicle_id=v.vehicle_id,
            position=v.position,
            current_load=v.current_load,
            available_at=v.available_at,
            route=v.route.copy(),
            service_times=v.service_times.copy(),
        )

    def _execute_action(self, action: SchedulerAction) -> None:
        """Execute an action from the strategy."""
        if self.action_callback:
            # Always pass auto=False here, this is strategy actions (will update for auto actions)
            self.action_callback(self.time, action, False)

        if isinstance(action, DispatchEvent):
            self._execute_dispatch(action)
        elif isinstance(action, WaitEvent):
            self._execute_wait(action)
        elif isinstance(action, RejectEvent):
            self._execute_reject(action)

    def _execute_dispatch(self, event: DispatchEvent) -> None:
        """Execute a dispatch event."""
        vehicle = self._get_vehicle(event.vehicle_id)
        destination = self.instance.get_request(event.destination_node)

        # Validate: vehicle must be idle
        if vehicle.available_at > self.time:
            raise ValueError(
                f"Vehicle {event.vehicle_id} is not idle at time {self.time}"
            )

        # Calculate travel time from current position
        from_req = self.instance.get_request(vehicle.position)
        distance = from_req.distance_to(destination)
        from_vehicle = self.instance.vehicles[event.vehicle_id]
        travel_time = from_vehicle.travel_time(distance)
        arrival_time = self.time + travel_time

        # Determine service start time (respecting time window)
        tw = destination.time_window
        service_start = max(arrival_time, tw.earliest)

        # Validate time window
        if service_start > tw.latest:
            raise ValueError(
                f"Cannot serve request {destination.id} at vehicle {event.vehicle_id}: "
                f"service_start={service_start} > latest={tw.latest}"
            )

        # Update vehicle state
        vehicle.position = destination.id
        vehicle.route.append(destination.id)
        vehicle.service_times.append(service_start)
        vehicle.available_at = service_start + destination.service_time

        # Update load
        if not destination.is_depot:
            vehicle.current_load += destination.demand

            # Check capacity
            if vehicle.current_load > from_vehicle.capacity:
                raise ValueError(
                    f"Dispatch {event.vehicle_id} → {destination.id}: "
                    f"exceeds capacity {from_vehicle.capacity}"
                )

            # Mark as served
            if destination.id in self.pending_requests:
                self.pending_requests.remove(destination.id)
                self.served_requests.add(destination.id)
        else:
            # Returned to depot, reset load
            vehicle.current_load = 0.0

    def _execute_wait(self, event: WaitEvent) -> None:
        """Execute a wait event."""
        if event.until_time <= self.time:
            raise ValueError(
                f"WaitEvent until_time={event.until_time} must be > current time={self.time}"
            )
        # Schedule a wake-up event
        heapq.heappush(self._event_queue, (event.until_time, "wake", -1))

    def _execute_reject(self, event: RejectEvent) -> None:
        """Execute a reject event."""
        if event.request_id not in self.pending_requests:
            return  # Already rejected or served

        # Check that request is not being served
        for v in self.vehicles:
            if event.request_id in v.route:
                raise ValueError(
                    f"Cannot reject request {event.request_id}: "
                    f"it is being served by vehicle {v.vehicle_id}"
                )

        self.pending_requests.remove(event.request_id)
        self.rejected_requests.add(event.request_id)

    def _get_vehicle(self, vehicle_id: int) -> VehicleState:
        """Get vehicle by ID."""
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        raise ValueError(f"Vehicle {vehicle_id} not found")

    def _finalize_result(self) -> SimulationResult:
        """Compute final solution and metrics."""
        # Build solution from vehicle routes
        routes = [v.route for v in self.vehicles]
        service_times = [v.service_times for v in self.vehicles]
        solution = Solution(routes=routes, service_times=service_times)

        # Compute metrics
        metrics = self._compute_metrics(solution)

        return SimulationResult(solution=solution, metrics=metrics)

    def _compute_metrics(self, solution: Solution) -> SimulationMetrics:
        """Compute simulation metrics."""
        # Total travel cost
        total_cost = 0.0
        depot_id = self.instance.depot_ids[0]

        for route in solution.routes:
            if not route:
                continue

            # depot -> first node
            from_node = self.instance.get_request(depot_id)
            to_node = self.instance.get_request(route[0])
            total_cost += from_node.distance_to(to_node)

            # intermediate edges
            for i in range(len(route) - 1):
                from_node = self.instance.get_request(route[i])
                to_node = self.instance.get_request(route[i + 1])
                total_cost += from_node.distance_to(to_node)

            # last node -> depot
            from_node = self.instance.get_request(route[-1])
            to_node = self.instance.get_request(depot_id)
            total_cost += from_node.distance_to(to_node)

        num_accepted = len(self.served_requests)

        return SimulationMetrics(
            total_travel_cost=total_cost,
            accepted=num_accepted,
        )

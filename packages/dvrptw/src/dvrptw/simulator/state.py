"""Simulation state types, metrics, results, and the strategy protocol."""

from dataclasses import dataclass, field
from typing import Protocol

from .events import SchedulerAction
from ..instance import Request
from ..solution import Solution


@dataclass
class VehicleState:
    """Current state of a single vehicle."""

    vehicle_id: int
    position: int  # current node (request ID or depot ID 0)
    current_load: float  # total demand currently carried
    available_at: float  # time when vehicle becomes idle
    route: list[int] = field(default_factory=list)  # stops visited (excluding depot)
    service_times: list[float] = field(default_factory=list)  # service start times


@dataclass
class SimulationState:
    """Current state of the simulation (strategy only sees released requests)."""

    time: float
    pending_requests: set[int]  # request IDs not yet rejected/completed
    served_requests: set[int]  # request IDs that have been served
    rejected_requests: set[int]  # request IDs that have been rejected
    vehicles: list[VehicleState]
    released_requests: dict[int, Request]  # release_time <= time, non-depot


class DispatchingStrategy(Protocol):
    """Protocol for dispatching strategy implementations."""

    def next_events(self, state: SimulationState) -> list[SchedulerAction]:
        """Called when simulator needs next actions from strategy.

        Returns a list of DispatchEvent, WaitEvent, or RejectEvent.
        """
        ...


@dataclass
class SimulationMetrics:
    """Performance metrics from simulation."""

    total_travel_cost: float
    rejected: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for serialization."""
        return {
            "total_travel_cost": self.total_travel_cost,
            "rejected": self.rejected,
        }


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    solution: Solution
    metrics: SimulationMetrics

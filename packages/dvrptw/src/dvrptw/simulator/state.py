"""Simulation state types, metrics, results, and the strategy protocol."""

from dataclasses import dataclass, field

from ..instance import Request
from ..solution import Solution


@dataclass
class VehicleSnapshot:
    """Current state of a single vehicle."""

    vehicle_id: int
    position: int  # current node (request ID or depot ID 0)
    current_load: float  # total demand currently carried
    available_at: float  # time when vehicle becomes idle
    route: list[int] = field(default_factory=list)  # stops visited (excluding depot)
    service_times: list[float] = field(default_factory=list)  # service start times


@dataclass
class SimulationSnapshot:
    time: float
    pending: set[int]
    served: set[int]
    rejected: set[int]
    vehicles: list[VehicleSnapshot]


@dataclass
class VehicleSpec:
    """Static information about a vehicle from the instance."""

    id: int
    capacity: float
    speed: float


@dataclass
class InstanceView:
    released_requests: dict[int, Request]
    vehicles: list[VehicleSpec]
    depot_id: int


@dataclass
class StrategyView:
    """Instance view passed to composable sub-strategies (routing/scheduling).

    Narrower than :class:`InstanceView` — only carries static instance data
    (depot identity and vehicle specs), not the full released-request map.
    """

    depot_id: int
    vehicle_specs: list[VehicleSpec]


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

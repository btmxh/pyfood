"""Simulator package for DVRPTW instance models.

This module lives under `packages/simulator/src/simulator` so the package is
installed/importable as `simulator` (PEP 517/518 layout).
"""

from .instance import (
    Time,
    Coord,
    ID,
    TimeWindow,
    Request,
    Vehicle,
    DVRPTWInstance,
    euclidean,
)
from .solution import Solution
from .simulator import (
    Simulator,
    SimulationState,
    SimulationResult,
    SimulationMetrics,
    VehicleState,
    DispatchEvent,
    WaitEvent,
    RejectEvent,
    SchedulerAction,
    DispatchingStrategy,
)

__all__ = [
    "Time",
    "Coord",
    "ID",
    "TimeWindow",
    "Request",
    "Vehicle",
    "DVRPTWInstance",
    "euclidean",
    "Solution",
    "Simulator",
    "SimulationState",
    "SimulationResult",
    "SimulationMetrics",
    "VehicleState",
    "DispatchEvent",
    "WaitEvent",
    "RejectEvent",
    "SchedulerAction",
    "DispatchingStrategy",
]

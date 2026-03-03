"""DVRPTW — problem instance models, solution representation, and simulation engine.

Importable as ``dvrptw`` (PEP 517/518 src layout under packages/dvrptw/src/dvrptw).
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
from .evaluator import (
    Evaluator,
    WeightedSumEvaluator,
    LinearNormEvaluator,
    StarNormEvaluator,
)
from .strategies import ILPStrategy
from .simulator import (
    DispatchEvent,
    WaitEvent,
    RejectEvent,
    SchedulerAction,
    VehicleState,
    SimulationState,
    DispatchingStrategy,
    SimulationMetrics,
    SimulationResult,
    Simulator,
    PythonSimulator,
    RustSimulator,
)
from .simulator import rsimulator

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
    "Evaluator",
    "WeightedSumEvaluator",
    "LinearNormEvaluator",
    "StarNormEvaluator",
    "Simulator",
    "PythonSimulator",
    "RustSimulator",
    "SimulationState",
    "SimulationResult",
    "SimulationMetrics",
    "VehicleState",
    "DispatchEvent",
    "WaitEvent",
    "RejectEvent",
    "SchedulerAction",
    "DispatchingStrategy",
    "ILPStrategy",
    "rsimulator",
]

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
from .strategies import (
    ILPStrategy,
    greedy_strategy,
    composable_strategy,
    python_composable_strategy,
    python_routing_strategy,
    python_scheduling_strategy,
    batch_composable_strategy,
)
from .simulator import (
    DispatchEvent,
    WaitEvent,
    RejectEvent,
    SchedulerAction,
    VehicleSnapshot,
    SimulationSnapshot,
    SimulationMetrics,
    SimulationResult,
    Simulator,
    PythonSimulator,
    RustSimulator,
    PythonDispatchStrategy,
    python_dispatch_strategy,
    python_event_callback,
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
    "SimulationSnapshot",
    "SimulationResult",
    "SimulationMetrics",
    "VehicleState",
    "VehicleSnapshot",
    "DispatchEvent",
    "WaitEvent",
    "RejectEvent",
    "SchedulerAction",
    "DispatchStrategy",
    "ILPStrategy",
    "PythonDispatchStrategy",
    "rsimulator",
    "composable_strategy",
    "python_composable_strategy",
    "python_routing_strategy",
    "python_scheduling_strategy",
    "batch_composable_strategy",
    "greedy_strategy",
    "python_dispatch_strategy",
    "python_event_callback",
]

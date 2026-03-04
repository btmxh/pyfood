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
    NativeBatchRouter,
    NativeRouter,
    NativeSequencer,
    greedy_strategy,
    composable_strategy,
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
    "NativeBatchRouter",
    "NativeRouter",
    "NativeSequencer",
    "composable_strategy",
    "batch_composable_strategy",
    "greedy_strategy",
]

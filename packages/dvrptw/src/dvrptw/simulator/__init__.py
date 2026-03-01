"""DVRPTW simulation engine — events, state, and simulator backends."""

from .events import DispatchEvent, RejectEvent, SchedulerAction, WaitEvent
from .state import (
    DispatchingStrategy,
    SimulationMetrics,
    SimulationResult,
    SimulationState,
    VehicleState,
)
from .base import Simulator
from .python import PythonSimulator
from .rust import (
    RustSimulator,
    NativeStrategyWrapper,
    NativeCallbackWrapper,
    greedy_strategy,
)  # noqa: F401

__all__ = [
    "DispatchEvent",
    "WaitEvent",
    "RejectEvent",
    "SchedulerAction",
    "VehicleState",
    "SimulationState",
    "DispatchingStrategy",
    "SimulationMetrics",
    "SimulationResult",
    "Simulator",
    "PythonSimulator",
    "RustSimulator",
    "NativeStrategyWrapper",
    "NativeCallbackWrapper",
    "greedy_strategy",
]

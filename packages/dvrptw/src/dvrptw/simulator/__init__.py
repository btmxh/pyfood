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
from .rust import RustSimulator

# Native strategy/callback wrappers — only available when rsimulator is built.
try:
    from .rust import NativeStrategyWrapper, NativeCallbackWrapper, greedy_strategy  # noqa: F401

    _NATIVE_EXPORTS = [
        "NativeStrategyWrapper",
        "NativeCallbackWrapper",
        "greedy_strategy",
    ]
except ImportError:
    _NATIVE_EXPORTS = []

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
] + _NATIVE_EXPORTS

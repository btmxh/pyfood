"""DVRPTW simulation engine — events, state, and simulator backends."""

from .events import DispatchEvent, RejectEvent, SchedulerAction, WaitEvent
from .state import (
    SimulationMetrics,
    SimulationResult,
    SimulationSnapshot,
    VehicleSnapshot,
    VehicleSpec,
    InstanceView,
)
from .base import Simulator, PythonDispatchStrategy, PythonEventCallback
from .python import PythonSimulator
from .rust import (
    RustSimulator,
    NativeDispatchStrategy,
    NativeEventCallback,
)
from . import rsimulator

__all__ = [
    "DispatchEvent",
    "WaitEvent",
    "RejectEvent",
    "SchedulerAction",
    "VehicleState",
    "SimulationState",
    "PythonDispatchStrategy",
    "PythonEventCallback",
    "SimulationMetrics",
    "SimulationResult",
    "SimulationSnapshot",
    "VehicleSnapshot",
    "VehicleSpec",
    "InstanceView",
    "Simulator",
    "PythonSimulator",
    "RustSimulator",
    "NativeDispatchStrategy",
    "NativeEventCallback",
    "greedy_strategy",
    "rsimulator",
]

# Backwards-compatible aliases: older code/tests expect `SimulationState`
# and `VehicleState` names. Provide simple aliases to the newer
# `SimulationSnapshot`/`VehicleSnapshot` dataclasses.
VehicleState = VehicleSnapshot
SimulationState = SimulationSnapshot

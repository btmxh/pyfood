"""DVRPTW simulation engine — events, state, and simulator backends."""

from .events import DispatchEvent, RejectEvent, SchedulerAction, WaitEvent
from .state import (
    SimulationMetrics,
    SimulationResult,
    SimulationSnapshot,
    VehicleSnapshot,
    VehicleSpec,
    InstanceView,
    StrategyView,
)
from .base import Simulator, PythonDispatchStrategy, PythonEventCallback
from .python import PythonSimulator
from .rust import (
    RustSimulator,
    NativeDispatchStrategy,
    NativeEventCallback,
    python_dispatch_strategy,
    python_event_callback,
)
from .rsimulator import (
    NativeRoutingStrategy,
    NativeSchedulingStrategy,
    python_routing_strategy,
    python_scheduling_strategy,
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
    "StrategyView",
    "NativeRoutingStrategy",
    "NativeSchedulingStrategy",
    "python_routing_strategy",
    "python_scheduling_strategy",
    "Simulator",
    "PythonSimulator",
    "RustSimulator",
    "NativeDispatchStrategy",
    "NativeEventCallback",
    "greedy_strategy",
    "rsimulator",
    "python_dispatch_strategy",
    "python_event_callback",
]

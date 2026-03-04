"""Abstract Simulator base class."""

from typing import Callable, Protocol
from abc import ABC, abstractmethod
from ..instance import DVRPTWInstance
from .state import SimulationResult, SimulationSnapshot, InstanceView
from .events import SchedulerAction

PythonEventCallback = Callable[[float, SchedulerAction, bool], None] | None


class PythonDispatchStrategy(Protocol):
    """Protocol for dispatching strategy implementations."""

    def next_events(
        self, state: SimulationSnapshot, view: InstanceView
    ) -> list[SchedulerAction]:
        """Called when simulator needs next actions from strategy.

        Returns a list of DispatchEvent, WaitEvent, or RejectEvent.
        """
        ...


class Simulator[StrategyType, CallbackType](ABC):
    """Abstract DVRPTW Simulation Engine.

    Subclasses implement ``run()`` with a concrete backend.
    After ``run()`` returns, ``served_requests`` and ``rejected_requests``
    are always populated regardless of backend.
    """

    def __init__(
        self,
        instance: DVRPTWInstance,
        dispatch_strategy: StrategyType,
        event_callback: CallbackType | None = None,
    ):
        instance.validate()
        self.instance = instance

    @abstractmethod
    def run(self) -> SimulationResult:
        """Execute the simulation and return results."""
        ...

    @classmethod
    @abstractmethod
    def wrap_strategy(cls, strategy: PythonDispatchStrategy) -> StrategyType: ...

    @classmethod
    @abstractmethod
    def wrap_callback(cls, callback: PythonEventCallback) -> CallbackType: ...

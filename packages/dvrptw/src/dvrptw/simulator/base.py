"""Abstract Simulator base class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from .events import SchedulerAction
from ..instance import DVRPTWInstance
from .state import DispatchingStrategy, SimulationResult

if TYPE_CHECKING:
    from rsimulator import NativeStrategyWrapper as _NativeStrategyWrapperT


class Simulator(ABC):
    """Abstract DVRPTW Simulation Engine.

    Subclasses implement ``run()`` with a concrete backend.
    After ``run()`` returns, ``served_requests`` and ``rejected_requests``
    are always populated regardless of backend.
    """

    def __init__(
        self,
        instance: DVRPTWInstance,
        strategy: "DispatchingStrategy | _NativeStrategyWrapperT",
        action_callback: Callable[[float, SchedulerAction, bool], None] | None = None,
    ):
        instance.validate()
        self.instance = instance
        self.strategy = strategy
        self.action_callback = action_callback

        self.served_requests: set[int] = set()
        self.rejected_requests: set[int] = set()

    @abstractmethod
    def run(self) -> SimulationResult:
        """Execute the simulation and return results."""
        ...

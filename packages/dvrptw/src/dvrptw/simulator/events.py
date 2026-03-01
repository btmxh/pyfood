"""Scheduler actions returned by dispatching strategies."""

from dataclasses import dataclass


@dataclass
class DispatchEvent:
    """Instructs simulator to send a vehicle to a destination node."""

    vehicle_id: int
    destination_node: int


@dataclass
class WaitEvent:
    """Instructs simulator to advance time until a specific moment."""

    until_time: float


@dataclass
class RejectEvent:
    """Instructs simulator to reject a pending request."""

    request_id: int


SchedulerAction = DispatchEvent | WaitEvent | RejectEvent

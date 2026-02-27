# Simulator–Strategy Interface and Event API

## Overview

This document describes the interface between the simulation engine and dispatching strategy (“policy”) for the DVRPTW simulation research platform.
The design is maximally flexible, capturing a wide spectrum of online and time-driven dispatch methods, with full transparency and replayability.

## Core Principles

- **All vehicle route intent is managed by the user-provided strategy.**
- **The simulator never plans ahead or auto-assigns unserved requests.**
- **At each event, the simulator provides the up-to-date state; the strategy returns a list of explicit actions (“events”) to schedule next.**
- **All consequential logic (actual vehicle movement, time progression, and request acceptance/rejection) is reflected by these actions.**

---

## Strategy → Simulator Actions/Events

Each time the simulator requests input (after an observable event, or when awoken by a wait), the strategy returns a **list of actions**. Supported types:

### 1. DispatchEvent

> Instructs the simulator to send a specific vehicle to a specific node for service, as soon as possible.

**Fields**:
- `vehicle_id`: The ID of the vehicle to dispatch.
- `destination_node`: The node to visit (e.g., request node or depot, as per instance spec).

**NOTE**: This is only allowed when the vehicle indicated by `vehicle_id` is
idle. To queue dispatches, the user must use a combination of WaitEvent and
DispatchEvent.

---

### 2. WaitEvent

> Instructs the simulator to “sleep” or advance time until a particular future time, at which point the strategy will be “woken” for another decision opportunity.

**Fields**:
- `until_time`: Absolute simulation time to wake the strategy. MUST be greater
  than the current timestamp.

**Use cases**:
- Regular/planned batch dispatch intervals
- Cost-based or context-driven time batching
- Backoff/“do nothing” mode

---

### 3. RejectEvent

> Explicitly rejects a known, still-pending request.

**Fields**:
- `request_id`: The request to reject. This MUST NOT be a request that is being
  served by any vehicle.

**Notes**:
- Once rejected, a request is no longer eligible for assignment.
- Requests with closed time windows are automatically rejected.

---

## Strategy Interface

The user strategy must implement:

```
class DispatchingStrategy(Protocol):
    def next_events(self, state: SimulationState) -> list[SchedulerAction]:
        """
        Called at each event where the strategy is eligible to act:
        - after any real-world event triggers (request arrival, vehicle finishes, etc)
        - after WaitEvent awakes at the requested time

        Receives the current SimulationState (includes sim time, pending requests, vehicle status, past/committed journeys, etc.)
        Returns a flat list of events:
            - any number of DispatchEvent, WaitEvent, RejectEvent objects,
            - in any sensible order (simulator will validate feasibility & constraints).
        """
```
- The current plan (vehicle → intended next stops) is managed entirely by the strategy; the simulator only acts on events returned.

---

## Simulation Output

**SimulationResult structure:**

```
@dataclass
class SimulationResult:
    # provide actions_log via a callback-based API, do NOT store it on RAM or
    # anything, as this is EXTREMELY memory-intensive!
    # actions_log: list[SchedulerAction]   # All actions returned by the strategy, chronologically
    solution: Solution                   # Derived from executed actions for convenience
    # create a dedicated class for Metrics, do NOT use dict like in this example
    metrics: dict                        # e.g., total travel cost, accept/reject counts, weighted objectives, etc.
```
- The actions_log is sufficient to reconstruct the simulation step-by-step, for debugging, visualization, or ML applications.

---

## Example Workflow

1. At simulation start and after every relevant event, simulator asks strategy for `next_events(state)`.
2. Strategy can return a list of SchedulerAction, of which elements can either be:
     - `DispatchEvent`s (e.g. “vehicle 0 to customer 5”).
     - `WaitEvent`(s) to pause until a (or potentially many) future time.
     - `RejectEvent`s for requests no longer desired.
3. Simulator executes events in order (though order does not matter much in our API), advances time as required, and records each action (without implicitly storing everything in memory!).
4. At the end of simulation, all results and the full decision trace are returned in `SimulationResult`.

---

## Example Action/Event

```
DispatchEvent(
    vehicle_id=0,
    destination_node=5,
    request_id=5,
    time=23.5,
    details={"strategy":"greedy_batch"}
)
```
```
WaitEvent(
    until_time=30.0
)
```
```
RejectEvent(
    request_id=9,
    reason="Demand exceeds all vehicles"
)
```

---

## Reconstructing Solutions

The driver/wrapper can reconstruct the full solution (routes, service times, accept/reject sets) from the actions_log and the original instance data.

---

## Benefits

- Supports purely online, deferred (“just-in-time”) and batch scheduling.
- Allows explicit, auditable request rejection and explanation.
- Policy is free to “sleep”/wake, yielding only as often as actually required.
- All consequential actions are logged for fairness, reproducibility, and research use.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DVRPTW (Dynamic Vehicle Routing Problem with Time Windows) solver research project. The problem involves routing vehicles to serve customer requests while respecting:
- Vehicle capacity constraints
- Time windows (earliest/latest service start times)
- Dynamic request arrivals (requests unknown until they appear during simulation)
- Service time requirements at each stop
- Dual objectives: minimize travel cost and maximize accepted requests

See `docs/STATEMENT.md` for the complete problem formulation.

## Project Structure

This is a monorepo using uv workspace with two main packages:

- **`packages/dvrptw/`**: Pure Python package — DVRPTW domain library
  - Uses PEP 517/518 src layout: importable as `dvrptw`
  - Main modules:
    - `instance.py`: `DVRPTWInstance`, `Request`, `Vehicle`, `TimeWindow`, `load_vrpr_csv()`
    - `solution.py`: `Solution` dataclass
    - `evaluator.py`: `Evaluator` protocol and implementations (`WeightedSumEvaluator`, `StarNormEvaluator`, `LinearNormEvaluator`)
    - `simulator/` subpackage:
      - `base.py`: Abstract `Simulator` class, `PythonDispatchStrategy` protocol, `PythonEventCallback`
      - `state.py`: `SimulationSnapshot`, `VehicleSnapshot`, `InstanceView`, `StrategyView`, `VehicleSpec`, `SimulationResult`, `SimulationMetrics`
      - `events.py`: `DispatchEvent`, `WaitEvent`, `RejectEvent`, `SchedulerAction`
      - `python.py`: `PythonSimulator` (pure Python backend)
      - `rust.py`: `RustSimulator` (Rust-backed backend via rsimulator)
      - `rsimulator.py`: Plain re-export of rsimulator extension symbols
    - `strategies/` subpackage:
      - `ilp.py`: `ILPStrategy` dispatching strategy
      - `rust.py`: Protocols, Layer-2 adapters, and factory functions for composable Rust strategies

- **`packages/rsimulator/`**: Rust performance module using PyO3/maturin
  - Python bindings to Rust code for performance-critical simulation
  - Implements:
    - `Simulator` backend (same interface as Python simulator)
    - Native strategy types: `NativeDispatchStrategy`, `NativeRoutingStrategy`, `NativeSchedulingStrategy`, `NativeBatchRoutingStrategy`, `NativeEventCallback`
    - Native strategies: `greedy_strategy()`, `composable_strategy()`, `batch_composable_strategy()`
    - Python-wrapping functions: `python_dispatch_strategy()`, `python_routing_strategy()`, `python_scheduling_strategy()`, `python_batch_routing_strategy()`, `python_event_callback()`
    - GP (Genetic Programming) API: `gp_strategy()` and expression tree builders (`flat_gp_*` functions)
  - Source files:
    - `src/strategies/composable.rs`: `ComposableStrategy`, `PyRoutingAdapter`, `PySchedulingAdapter`
    - `src/strategies/batch.rs`: `BatchComposableStrategy`, `PyBatchRoutingAdapter`
    - `src/strategies/gp_tree.rs`: GP tree evaluation engine
    - `src/strategies/gp_strategy.rs`: GP strategy implementation
  - Type stubs: `rsimulator.pyi` (canonical type source for the compiled extension)

The root `pyproject.toml` defines the workspace and main project dependencies.

## Development Commands

### Setup
```bash
# Enter Nix environment (sets up shell with uv, cargo, etc.)
direnv allow  # or nix develop

# Install all packages and dev dependencies
uv sync

# Build the Rust extension (required once, and after any Rust changes)
uv run maturin develop --manifest-path packages/rsimulator/Cargo.toml
```

### Testing
```bash
uv run pytest

# Run tests for specific package
uv run pytest packages/dvrptw/tests/
uv run pytest packages/rsimulator/

# Run single test file
uv run pytest packages/dvrptw/tests/test_simulator_instance.py

# Run specific test
uv run pytest packages/dvrptw/tests/test_simulator_instance.py::TestSimulatorInstance::test_load_vrpr_csv_basic
```

### Building Rust Extension
```bash
uv run maturin develop --manifest-path packages/rsimulator/Cargo.toml
```

### Type Checking
```bash
uv run ty check
```

### Linting and Formatting
```bash
# Format Python code
uv run ruff format

# Lint Python code
uv run ruff check

# Format Rust code
cargo fmt --manifest-path packages/rsimulator/Cargo.toml

# Lint Rust code
cargo clippy --manifest-path packages/rsimulator/Cargo.toml
```

## Coding Guidelines

### Type Hints
Use built-in collection types for type hints (Python 3.9+):
- Use `list[int]` instead of `List[int]`
- Use `dict[str, int]` instead of `Dict[str, int]`
- Use `tuple[float, float]` instead of `Tuple[float, float]`

Do not import `List`, `Dict`, `Tuple` from the `typing` module unless using older Python-specific features.

## Architecture Notes

### Solution Representation (`dvrptw.Solution`)

The solution structure is intentionally minimal to provide a uniform API for algorithms:

```python
@dataclass
class Solution:
    routes: list[list[int]]           # routes[k] = node sequence for vehicle k
    service_times: list[list[float]]  # service_times[k][i] = service start time at node
```

**Design rationale:**
- Routes exclude depot (implicitly start/end at depot node 0)
- Service times capture strategic waiting decisions in dynamic scenarios
- Everything else (arrival times, wait times, costs, feasibility) should be computed by evaluator functions taking `(solution, instance)`
- Algorithms output simple lists; no need to maintain complex internal state

### Instance Model (`dvrptw.DVRPTWInstance`)

Core dataclass representing a DVRPTW problem instance:
- `requests: list[Request]`: List of all nodes (depot + customers), where depot has `is_depot=True`
- `vehicles: list[Vehicle]`: Fleet specifications with capacity and speed
- `depot_id: ID`: ID of depot nodes (typically just `0`)
- `planning_horizon: Time | None`: Optional time limit

Each `Request` contains:
- `id`, `position: Coord`, `demand`, `time_window: TimeWindow`, `service_time`, `release_time`, `is_depot`

Each `Vehicle` contains:
- `id`, `capacity`, `start_depot`, `end_depot`, `max_time`, `speed`

Loading and serialization:
- `load_vrpr_csv(csv_path, truck_speed, truck_capacity, num_trucks)`: Load instances from CSV format
- `to_json()` / `from_json()`: JSON serialization
- `pairwise_distance_matrix()`: Compute all pairwise Euclidean distances
- `get_request(req_id)`: Lookup request by ID
- `validate()`: Validate instance constraints

### Objective Evaluators (`dvrptw.evaluator`)

Evaluators collapse the two raw DVRPTW objectives (minimize travel cost, minimize rejections) into a single scalar for optimization:

**Protocol:**
```python
class Evaluator(Protocol):
    def scalar(obj1: float, obj2: float) -> float:
        """obj1 = total travel cost, obj2 = rejected request count"""
        ...

    def ilp_coefficients(star_cost: float, n_customers: float) -> tuple[float, float]:
        """Return (coeff1, coeff2) for ILP objective coefficients"""
        ...
```

**Implementations:**
- `WeightedSumEvaluator(w1, w2)`: Simple weighted sum (raw units, no normalization)
- `StarNormEvaluator(w1, w2)`: Normalizes obj1 by star cost (Σ 2·dist(depot, customer)), obj2 by n_customers
- `LinearNormEvaluator(w1, w2, lo1, hi1, lo2, hi2)`: Caller-supplied explicit bounds for linear normalization

### Simulation Engine (`dvrptw.Simulator`)

The simulator implements the DVRPTW simulation loop with a pluggable dispatching strategy:

**Core API:**
```python
simulator = PythonSimulator(instance, strategy, action_callback=None)
result = simulator.run()  # Returns SimulationResult
```

**How it works:**
1. Strategy implements `PythonDispatchStrategy.next_events(state: SimulationSnapshot, view: InstanceView) -> list[SchedulerAction]`
2. Simulator calls strategy at each event (request arrivals, vehicle completions, wake-ups)
3. Strategy returns actions: `DispatchEvent`, `WaitEvent`, or `RejectEvent`
4. Simulator validates constraints and advances time accordingly
5. Final result includes routes, service times, and metrics

**Key constraints enforced:**
- Vehicle capacity limits (checked on dispatch)
- Time window feasibility (service start must be in [earliest, latest])
- Auto-rejection of requests with closed time windows

**Backends:**
- `PythonSimulator` — pure Python, always available
- `RustSimulator` — delegates to `rsimulator` extension; requires `maturin develop`
  - Also supports native Rust strategies via `NativeDispatchStrategy` (from rsimulator module)
  - Example: `rsimulator.greedy_strategy()` returns a native strategy that runs entirely in Rust

### Dispatching Strategies

Strategies implement the `PythonDispatchStrategy` protocol (defined in `simulator/base.py`):
```python
class PythonDispatchStrategy(Protocol):
    def next_events(self, state: SimulationSnapshot, view: InstanceView) -> list[SchedulerAction]:
        """Called when the simulator needs the next batch of actions."""
```

**`SimulationSnapshot`** carries mutable runtime state (time, pending/served/rejected sets, vehicle snapshots).
**`InstanceView`** carries the static instance data visible to the strategy: `released_requests`, `vehicles` (specs), `depot_id`.

**Built-in strategies:**
- `ILPStrategy(instance, evaluator=None)`: Solves full VRPTW as a Mixed-Integer Linear Program using PuLP, then replays the solution during simulation
  - Formulation includes vehicle flow, capacity, time window, and service constraints
  - Supports objective weights through pluggable evaluators
  - Respects dynamic release times: dispatches according to precomputed plan when possible, auto-rejects infeasible requests

**Native Rust strategies:**
- Available from `rsimulator` module (e.g., `greedy_strategy()`)
- Run entirely in Rust with no GIL contention
- Wrapped by `RustSimulator` automatically

### Composable Strategies (`dvrptw.strategies.rust`)

Composable strategies split dispatching into independent **routing** (request → vehicle assignment) and **scheduling** (vehicle queue ordering) sub-strategies. The architecture uses a two-layer adapter chain:

- **Layer 1 (Rust → dicts)**: Built into the extension. `PyRoutingAdapter` / `PySchedulingAdapter` / `PyBatchRoutingAdapter` call Python objects with raw `dict` arguments.
- **Layer 2 (dicts → typed)**: `NativeRoutingAdapter` / `NativeSchedulingAdapter` / `NativeBatchRoutingAdapter` in `strategies/rust.py` convert dicts into `VehicleSnapshot` and `StrategyView` before calling user code.

**Typed protocols** for user implementations:
```python
class PythonRoutingStrategy(Protocol):
    def route(self, request_id: int, vehicles: list[VehicleSnapshot], view: StrategyView) -> int | None: ...

class PythonSchedulingStrategy(Protocol):
    def schedule(self, vehicle: VehicleSnapshot, queue: list[int], view: StrategyView) -> int: ...

class PythonBatchRoutingStrategy(Protocol):
    def route_batch(self, requests: list[int], vehicles: list[VehicleSnapshot], view: StrategyView) -> list[tuple[int, int | None]]: ...
```

**`StrategyView`** is narrower than `InstanceView` — carries only `depot_id` and `vehicle_specs` (static instance data, no released-request map).

**Factory functions** (all in `dvrptw.strategies`):

| Function | Description |
|---|---|
| `python_routing_strategy(router)` | Wrap typed router → `NativeRoutingStrategy` |
| `python_scheduling_strategy(scheduler)` | Wrap typed scheduler → `NativeSchedulingStrategy` |
| `python_batch_routing_strategy(router)` | Wrap typed batch router → `NativeBatchRoutingStrategy` |
| `composable_strategy(router, scheduler)` | Combine native router + scheduler |
| `batch_composable_strategy(router, scheduler, slot_size)` | Slot-based batch routing variant |
| `python_composable_strategy(router, scheduler)` | Convenience: wraps + composes in one call |
| `python_batch_composable_strategy(router, scheduler, slot_size)` | Convenience batch variant |

**Example:**
```python
from dvrptw.strategies import python_composable_strategy

class MyRouter:
    def route(self, request_id: int, vehicles: list[VehicleSnapshot], view: StrategyView) -> int | None:
        return vehicles[0].vehicle_id  # assign to first vehicle

class MyScheduler:
    def schedule(self, vehicle: VehicleSnapshot, queue: list[int], view: StrategyView) -> int:
        return min(queue)  # FIFO

strategy = python_composable_strategy(MyRouter(), MyScheduler())
simulator = RustSimulator(instance, strategy)
result = simulator.run()
```

### Genetic Programming (GP) Strategies (`rsimulator.gp_*`)

GP strategies allow building custom routing/sequencing/rejection logic from expression trees:

**Core API:**
```python
from dvrptw import rsimulator

# Build expression trees from features and operators
travel_time_tree = rsimulator.flat_gp_travel_time()
capacity_tree = rsimulator.flat_gp_remaining_capacity()
score_tree = rsimulator.flat_gp_sub(travel_time_tree, capacity_tree)

# Create a strategy that uses three trees:
# - routing: scores vehicles (returns selected vehicle ID)
# - sequencing: scores pending requests (for dispatch order)
# - reject: scores for auto-rejection (reject if <= 0)
strategy = rsimulator.gp_strategy(routing_tree, sequencing_tree, reject_tree)
simulator = RustSimulator(instance, strategy)
result = simulator.run()
```

**Tree Construction:**
- Leaf nodes (features): `flat_gp_travel_time()`, `flat_gp_demand()`, `flat_gp_current_load()`, `flat_gp_remaining_capacity()`, `flat_gp_window_earliest()`, `flat_gp_window_latest()`, `flat_gp_time_until_due()`, `flat_gp_release_time()`
- Numeric literals: `flat_gp_const(value: float)`
- Binary operators: `flat_gp_add()`, `flat_gp_sub()`, `flat_gp_mul()`, `flat_gp_div()`

**How it works:**
1. **Routing tree**: Evaluates for each vehicle in the fleet; vehicle with highest score is selected to receive the next dispatch
2. **Sequencing tree**: Evaluates for each released but unscheduled request; requests are dispatched in order of descending score
3. **Reject tree**: Evaluates the selected request-vehicle pair; if reject_score > routing_score, the request is auto-rejected
4. All evaluation is JIT-compiled to SIMD f32 bytecode in Rust

### Workspace Dependencies

The `dvrptw` package depends on `rsimulator` (Rust extension). When making changes:
1. Build `rsimulator` first if modifying Rust code
2. The workspace ensures consistent dependency resolution via `uv.lock`

## Testing Data

Test datasets are located in `packages/dvrptw/tests/data/` (local copies to avoid external dependencies). The standard format is CSV files from the VRPR benchmark set.

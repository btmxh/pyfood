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

- **`packages/dvrptw/`**: Pure Python package — DVRPTW domain library (instance models, solution representation, simulation engine)
  - Contains `DVRPTWInstance`, `Solution`, `PythonSimulator`, `RustSimulator`, and related types
  - Uses PEP 517/518 src layout: importable as `dvrptw`
  - Source split across: `instance.py`, `solution.py`, `events.py`, `state.py`, `python_simulator.py`, `rust_simulator.py`

- **`packages/rsimulator/`**: Rust performance module using PyO3/maturin
  - Python bindings to Rust code for performance-critical simulation
  - Used internally by `dvrptw.RustSimulator`

The root `pyproject.toml` defines the workspace and main project dependencies.

## Development Commands

### Setup
```bash
# Nix-based environment (preferred)
direnv allow  # or nix develop

# Alternative: use uv directly
uv sync
```

### Testing
```bash
# Run all tests
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
# Build rsimulator (maturin-based)
cd packages/rsimulator
maturin develop  # or maturin build

# The Python package depends on this being built
```

### Linting and Formatting
```bash
# Format Python code
ruff format

# Lint Python code
ruff check

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

Contains:
- Depot and customer requests (positions, demands, time windows, service times)
- Vehicle fleet (capacity, speed)
- Dynamic release times for requests
- Objective weight parameter

Key methods:
- `load_vrpr_csv()`: Load instances from CSV format
- `to_json()`/`from_json()`: Serialization
- `pairwise_distance()`: Precomputed distance matrix

### Simulation Engine (`dvrptw.Simulator`)

The simulator implements the DVRPTW simulation loop with a pluggable dispatching strategy:

**Core API:**
```python
simulator = PythonSimulator(instance, strategy, action_callback=None)
result = simulator.run()  # Returns SimulationResult
```

**How it works:**
1. Strategy implements `DispatchingStrategy.next_events(state: SimulationState) -> list[SchedulerAction]`
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

### Workspace Dependencies

The `dvrptw` package depends on `rsimulator` (Rust extension). When making changes:
1. Build `rsimulator` first if modifying Rust code
2. The workspace ensures consistent dependency resolution via `uv.lock`

## Testing Data

Test datasets are located in `packages/dvrptw/tests/data/` (local copies to avoid external dependencies). The standard format is CSV files from the VRPR benchmark set.

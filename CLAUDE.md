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

- **`packages/simulator/`**: Pure Python package for DVRPTW instance models and solution representation
  - Contains `DVRPTWInstance` (problem instances) and `Solution` (algorithm output)
  - Uses PEP 517/518 src layout: importable as `simulator`

- **`packages/rsimulator/`**: Rust performance module using PyO3/maturin
  - Python bindings to Rust code for performance-critical operations
  - Currently contains placeholder code; intended for simulation engine

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
uv run pytest packages/simulator/tests/
uv run pytest packages/rsimulator/

# Run single test file
uv run pytest packages/simulator/tests/test_simulator_instance.py

# Run specific test
uv run pytest packages/simulator/tests/test_simulator_instance.py::TestSimulatorInstance::test_load_vrpr_csv_basic
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

### Solution Representation (`simulator.Solution`)

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

### Instance Model (`simulator.DVRPTWInstance`)

Contains:
- Depot and customer requests (positions, demands, time windows, service times)
- Vehicle fleet (capacity, speed)
- Dynamic release times for requests
- Objective weight parameter

Key methods:
- `load_vrpr_csv()`: Load instances from CSV format
- `to_json()`/`from_json()`: Serialization
- `pairwise_distance()`: Precomputed distance matrix

### Workspace Dependencies

The `simulator` package depends on `rsimulator` (Rust extension). When making changes:
1. Build `rsimulator` first if modifying Rust code
2. The workspace ensures consistent dependency resolution via `uv.lock`

## Testing Data

Test datasets are located in `packages/simulator/tests/data/` (local copies to avoid external dependencies). The standard format is CSV files from the VRPR benchmark set.

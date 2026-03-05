"""GP3 dispatch on the 1000-node h1000C1_10_1 instance.

Usage:
    uv run python run_gp3_1000.py

Runs two weight-agnostic native Rust strategies on the full 1000-customer
instance and reports metrics:

  GreedyRust  – baseline: dispatch in ascending request-ID order
  GP3         – routing tree: -(travel_time + current_load)
                sequencing tree: -time_until_due  (serve most-urgent first)
                reject tree:     routing_score - 1  (reject if routing_score < 1)

Both run entirely inside Rust via RustSimulator — no Python overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from dvrptw.instance import load_vrpr_csv
from dvrptw import RustSimulator
from rsimulator import (
    greedy_strategy,
    gp_strategy,
    flat_gp_const,
    flat_gp_travel_time,
    flat_gp_time_until_due,
    flat_gp_current_load,
)

# ---------------------------------------------------------------------------
# Instance parameters — adjust to match the benchmark that generated the CSV
# ---------------------------------------------------------------------------
CSV_PATH = "h1000C1_10_1.csv"
TRUCK_SPEED = 1.0
TRUCK_CAPACITY = 1300.0
NUM_TRUCKS = 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    strategy_name: str
    served: int
    rejected: int
    total_requests: int
    travel_cost: float
    wall_time_s: float

    @property
    def service_rate(self) -> float:
        return 100.0 * self.served / self.total_requests if self.total_requests else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading instance from {CSV_PATH} ...")
    instance = load_vrpr_csv(CSV_PATH, TRUCK_SPEED, TRUCK_CAPACITY, NUM_TRUCKS)

    customers = [r for r in instance.requests if not r.is_depot]
    n_customers = len(customers)
    n_static = sum(1 for r in customers if r.release_time == 0.0)
    n_dynamic = n_customers - n_static

    print(f"  Customers  : {n_customers}  ({n_static} static, {n_dynamic} dynamic)")
    print(
        f"  Vehicles   : {len(instance.vehicles)} × capacity {instance.vehicles[0].capacity}"
    )
    print(f"  Speed      : {instance.vehicles[0].speed}")
    horizon = instance.planning_horizon
    print(f"  Horizon    : {horizon}")
    print()

    results: list[RunResult] = []

    # --- GreedyRust baseline ---
    print("Running GreedyRust baseline ...")
    t0 = time.perf_counter()
    sim = RustSimulator(instance, greedy_strategy())
    res = sim.run()
    elapsed = time.perf_counter() - t0
    served = n_customers - res.metrics.rejected
    results.append(
        RunResult(
            strategy_name="GreedyRust",
            served=served,
            rejected=res.metrics.rejected,
            total_requests=n_customers,
            travel_cost=res.metrics.total_travel_cost,
            wall_time_s=elapsed,
        )
    )
    print(
        f"  served={served}/{n_customers} ({100 * served / n_customers:.1f}%)"
        f"  cost={res.metrics.total_travel_cost:.2f}"
        f"  time={elapsed:.3f}s"
    )
    print()

    # --- GP3 ---
    # routing tree    : -(travel_time + current_load)  → prefer fast, light vehicles
    # sequencing tree : -time_until_due                → serve most-urgent requests first
    # reject tree     : routing_score - 1              → reject when routing_score < 1
    print("Building GP3 expression trees ...")
    routing = -(flat_gp_travel_time() + flat_gp_current_load())
    sequencing = -flat_gp_time_until_due()
    reject = routing - flat_gp_const(1.0)
    strategy = gp_strategy(routing, sequencing, reject)

    print("Running GP3 ...")
    t0 = time.perf_counter()
    try:
        sim = RustSimulator(instance, strategy)
        res = sim.run()
        elapsed = time.perf_counter() - t0
        served = n_customers - res.metrics.rejected
        results.append(
            RunResult(
                strategy_name="GP3",
                served=served,
                rejected=res.metrics.rejected,
                total_requests=n_customers,
                travel_cost=res.metrics.total_travel_cost,
                wall_time_s=elapsed,
            )
        )
        print(
            f"  served={served}/{n_customers} ({100 * served / n_customers:.1f}%)"
            f"  cost={res.metrics.total_travel_cost:.2f}"
            f"  time={elapsed:.3f}s"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  GP3 failed after {elapsed:.3f}s: {exc}")
        results.append(
            RunResult(
                strategy_name="GP3",
                served=0,
                rejected=n_customers,
                total_requests=n_customers,
                travel_cost=float("inf"),
                wall_time_s=elapsed,
            )
        )

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)
    print(
        f"{'Strategy':<14} {'Served':>10} {'Rejected':>9} {'Svc%':>6}"
        f" {'Cost':>12} {'Time(s)':>8}"
    )
    print("-" * 62)
    for r in results:
        cost_str = f"{r.travel_cost:.2f}" if r.travel_cost != float("inf") else "inf"
        print(
            f"{r.strategy_name:<14} {r.served:>6}/{r.total_requests:<4}"
            f" {r.rejected:>9} {r.service_rate:>5.1f}%"
            f" {cost_str:>12} {r.wall_time_s:>8.3f}"
        )
    print()

    # Delta vs greedy
    greedy = next((r for r in results if r.strategy_name == "GreedyRust"), None)
    gp3 = next((r for r in results if r.strategy_name == "GP3"), None)
    if greedy and gp3 and gp3.travel_cost != float("inf"):
        delta_rej = gp3.rejected - greedy.rejected
        delta_cost = gp3.travel_cost - greedy.travel_cost
        sign_rej = "+" if delta_rej >= 0 else ""
        sign_cost = "+" if delta_cost >= 0 else ""
        ratio = (
            gp3.travel_cost / greedy.travel_cost
            if greedy.travel_cost > 0
            else float("nan")
        )
        print("GP3 vs GreedyRust:")
        print(f"  Δrejected : {sign_rej}{delta_rej}")
        print(f"  Δcost     : {sign_cost}{delta_cost:.2f}")
        print(f"  cost ratio: {ratio:.3f}x")


if __name__ == "__main__":
    main()

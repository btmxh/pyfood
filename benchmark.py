"""Benchmark: ILP baseline vs. dynamic nearest-neighbor strategy on a trimmed DVRPTW instance.

Usage:
    uv run python benchmark.py

The script:
1. Loads the 100-customer CSV instance.
2. Trims it to 15 customers: 8 static (release=0) + 7 with meaningful
   release times spread across the horizon, preserving the dynamic character.
3. Defines a proper online dynamic strategy: Nearest Feasible Neighbor (NFN).
   - At each event it inspects all currently released & pending requests.
   - For each idle vehicle it greedily picks the nearest request that is
     time-window feasible (vehicle can arrive before tw.latest) and does not
     violate capacity.
   - If no request is immediately feasible, it waits until the next vehicle
     becomes free or the next release arrives.
4. Runs both strategies for objective weights w ∈ {0.2, 0.4, 0.6, 0.8}.
5. Prints a comparison table.

Notes
-----
- ILP is a *non-causal* upper-bound baseline: it sees all release times
  ahead of time and solves the full offline VRPTW.
- NFN is fully *causal*: at each decision point it only sees requests whose
  release_time <= current simulation time.
- objective_weight w controls: min w·cost - (1-w)·served
  (ILP uses it directly; NFN ignores it — it always maximises service first,
   then breaks ties by distance, so we run it once and report the same
   service-count across weights; cost varies because routes are fixed.)
  To make NFN weight-aware we run two NFN variants:
    - NFN-serve: always prioritise accepting requests (greedy on service)
    - NFN-cost:  only dispatch if the detour is "worth it" relative to w

  Actually, to keep it honest and interesting, we implement a single NFN
  that scores candidates by a weighted combination:
      score = w * normalised_distance + (1-w) * (1 - urgency)
  where urgency = time_remaining / tw_width.  This makes the strategy
  weight-aware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from dvrptw.instance import DVRPTWInstance, Request, load_vrpr_csv
from dvrptw.simulator.events import (
    DispatchEvent,
    WaitEvent,
    SchedulerAction,
)
from dvrptw.simulator.state import SimulationState
from dvrptw import PythonSimulator, RustSimulator, ILPStrategy, StarNormEvaluator
from rsimulator import greedy_strategy


# ---------------------------------------------------------------------------
# 1. Instance construction
# ---------------------------------------------------------------------------

CSV_PATH = "packages/dvrptw/tests/data/h100rc101.csv"
TRUCK_SPEED = 1.0
TRUCK_CAPACITY = 200.0
NUM_TRUCKS = 5


def build_trimmed_instance(n_static: int = 8, n_dynamic: int = 7) -> DVRPTWInstance:
    """Load the full 100-customer instance and trim to n_static + n_dynamic customers.

    Static customers have release_time = 0 (known from the start).
    Dynamic customers have release_time > 10 (arrive meaningfully late).
    """
    full = load_vrpr_csv(CSV_PATH, TRUCK_SPEED, TRUCK_CAPACITY, NUM_TRUCKS)
    depot = next(r for r in full.requests if r.is_depot)
    customers = [r for r in full.requests if not r.is_depot]

    static_pool = sorted(
        [c for c in customers if c.release_time == 0.0],
        key=lambda c: c.id,
    )
    dynamic_pool = sorted(
        [c for c in customers if c.release_time >= 10.0],
        key=lambda c: c.release_time,
    )

    chosen = static_pool[:n_static] + dynamic_pool[:n_dynamic]
    # Re-assign IDs 1..N to keep them compact (depot stays 0)
    new_requests: list[Request] = []
    id_map: dict[int, int] = {depot.id: 0}
    for new_id, r in enumerate(chosen, start=1):
        id_map[r.id] = new_id
        new_requests.append(
            Request(
                id=new_id,
                position=r.position,
                demand=r.demand,
                time_window=r.time_window,
                service_time=r.service_time,
                release_time=r.release_time,
            )
        )

    return DVRPTWInstance(
        id="trimmed_rc101",
        requests=[depot] + new_requests,
        vehicles=full.vehicles,
        planning_horizon=full.planning_horizon,
        depot_ids=[depot.id],
    )


# ---------------------------------------------------------------------------
# 2. Dynamic strategy: Weight-Aware Nearest Feasible Neighbor (NFN)
# ---------------------------------------------------------------------------


class NearestFeasibleNeighborStrategy:
    """Online causal dynamic strategy.

    At each event:
    - Only considers requests that have been *released* (release_time <= now).
    - For each idle vehicle picks the candidate with the best weighted score:
        score = w * dist_norm + (1-w) * (1 - urgency)
      where urgency = time_left / tw_width  (closer to deadline = more urgent).
    - Respects capacity; skips requests that would exceed vehicle load.
    - If no dispatch is possible, waits until next vehicle idle or next release.

    Parameters
    ----------
    instance:
        Full instance (used for distance and release time lookup).
    objective_weight:
        Same w as used by ILP so we can compare at matching weights.
        w=0  → maximise service (prefer any feasible request, nearest wins ties)
        w=1  → minimise cost (strongly prefer close requests)
    """

    def __init__(self, instance: DVRPTWInstance, objective_weight: float = 0.5) -> None:
        self._instance = instance
        self._w = objective_weight
        self._req_by_id: dict[int, Request] = {r.id: r for r in instance.requests}
        # Precompute max distance for normalisation
        dist_vals = [
            a.distance_to(b)
            for a in instance.requests
            for b in instance.requests
            if a.id != b.id
        ]
        self._max_dist: float = max(dist_vals) if dist_vals else 1.0

    def next_events(self, state: SimulationState) -> list[SchedulerAction]:
        actions: list[SchedulerAction] = []
        now = state.time

        if not state.pending_requests:
            return []

        # Released and still pending requests at this moment
        available: set[int] = (
            set(state.released_requests.keys()) & state.pending_requests
        )

        idle_vehicles = [v for v in state.vehicles if v.available_at <= now]
        assigned_this_round: set[int] = set()

        for vehicle in idle_vehicles:
            current_req = self._req_by_id.get(vehicle.position)
            current_pos: tuple[float, float] = (
                current_req.position if current_req else (0.0, 0.0)
            )

            best_id: int | None = None
            best_score: float = math.inf

            for rid in available:
                if rid in assigned_this_round:
                    continue
                req = self._req_by_id[rid]

                # Capacity check
                if (
                    vehicle.current_load + req.demand
                    > self._instance.vehicles[vehicle.vehicle_id].capacity
                ):
                    continue

                dist = math.hypot(
                    req.position[0] - current_pos[0],
                    req.position[1] - current_pos[1],
                )
                arrival = now + dist / TRUCK_SPEED
                # Time-window feasibility: can we arrive before window closes?
                if arrival > req.time_window.latest:
                    continue

                # Score: lower is better
                dist_norm = dist / self._max_dist
                tw_width = max(req.time_window.latest - req.time_window.earliest, 1.0)
                time_left = req.time_window.latest - now
                urgency = 1.0 - min(
                    time_left / tw_width, 1.0
                )  # high urgency = window closing

                # w weights cost (distance) vs (1-w) weights service (urgency bonus)
                score = self._w * dist_norm - (1.0 - self._w) * urgency

                if score < best_score:
                    best_score = score
                    best_id = rid

            if best_id is not None:
                actions.append(
                    DispatchEvent(
                        vehicle_id=vehicle.vehicle_id, destination_node=best_id
                    )
                )
                assigned_this_round.add(best_id)
                available.discard(best_id)

        if not actions:
            # Compute next wake-up time
            candidates: list[float] = []
            # Next vehicle becomes idle
            busy_times = [
                v.available_at for v in state.vehicles if v.available_at > now
            ]
            if busy_times:
                candidates.append(min(busy_times))
            # Next request release
            for rid in state.pending_requests - set(state.released_requests.keys()):
                req = self._req_by_id.get(rid)
                if req and req.release_time > now:
                    candidates.append(req.release_time)
            if candidates:
                actions.append(WaitEvent(until_time=min(candidates)))

        return actions


# ---------------------------------------------------------------------------
# 3. Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    strategy_name: str
    weight: float
    rejected: int
    total_requests: int
    travel_cost: float
    wall_time_s: float


def run_benchmark() -> list[RunResult]:
    instance = build_trimmed_instance(n_static=8, n_dynamic=7)
    n_customers = len([r for r in instance.requests if not r.is_depot])
    weights = [0.2, 0.4, 0.6, 0.8]

    results: list[RunResult] = []

    print(f"Instance: {instance.id}")
    print(f"Customers: {n_customers} (8 static, 7 dynamic)")
    print(
        f"Vehicles:  {len(instance.vehicles)} × capacity {instance.vehicles[0].capacity}"
    )
    print(f"Speed:     {instance.vehicles[0].speed}")
    print()

    # Print instance summary
    customers = [r for r in instance.requests if not r.is_depot]
    print(
        f"{'ID':>4}  {'demand':>6}  {'tw_open':>7}  {'tw_close':>8}  {'release':>7}  type"
    )
    print("-" * 50)
    for c in customers:
        kind = "static" if c.release_time == 0 else f"dyn@{c.release_time:.1f}"
        print(
            f"{c.id:>4}  {c.demand:>6.1f}  {c.time_window.earliest:>7.1f}  {c.time_window.latest:>8.1f}  {c.release_time:>7.1f}  {kind}"
        )
    print()

    # --- ILP ---
    print("=" * 60)
    print("Running ILP (non-causal, full foreknowledge)...")
    print("=" * 60)
    for w in weights:
        t0 = time.perf_counter()
        evaluator = StarNormEvaluator.from_instance(w1=w, w2=1.0 - w, instance=instance)
        strategy = ILPStrategy(
            instance, evaluator=evaluator, time_limit_s=120.0, mip_gap=0.005
        )
        sim = PythonSimulator(instance, strategy)
        result = sim.run()
        elapsed = time.perf_counter() - t0
        results.append(
            RunResult(
                strategy_name="ILP",
                weight=w,
                rejected=result.metrics.rejected,
                total_requests=n_customers,
                travel_cost=result.metrics.total_travel_cost,
                wall_time_s=elapsed,
            )
        )
        print(
            f"  w={w:.1f}  rejected={result.metrics.rejected}/{n_customers}"
            f"  cost={result.metrics.total_travel_cost:.2f}  time={elapsed:.1f}s"
        )

    print()

    # --- NFN ---
    print("=" * 60)
    print("Running NFN (causal, online dynamic strategy)...")
    print("=" * 60)
    for w in weights:
        t0 = time.perf_counter()
        strategy = NearestFeasibleNeighborStrategy(instance, objective_weight=w)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()
        elapsed = time.perf_counter() - t0
        results.append(
            RunResult(
                strategy_name="NFN",
                weight=w,
                rejected=result.metrics.rejected,
                total_requests=n_customers,
                travel_cost=result.metrics.total_travel_cost,
                wall_time_s=elapsed,
            )
        )
        print(
            f"  w={w:.1f}  rejected={result.metrics.rejected}/{n_customers}"
            f"  cost={result.metrics.total_travel_cost:.2f}  time={elapsed:.3f}s"
        )

    print()

    # --- GreedyRust ---
    # Weight-agnostic: dispatches by ascending request-ID order; runs entirely
    # in Rust with no GIL overhead.  Run once and broadcast to all weights.
    print("=" * 60)
    print("Running GreedyRust (causal, native Rust strategy)...")
    print("=" * 60)
    t0 = time.perf_counter()
    strategy = greedy_strategy()
    sim = RustSimulator(instance, strategy)
    result = sim.run()
    elapsed = time.perf_counter() - t0
    print(
        f"  (weight-agnostic)  rejected={result.metrics.rejected}/{n_customers}"
        f"  cost={result.metrics.total_travel_cost:.2f}  time={elapsed:.3f}s"
    )
    for w in weights:
        results.append(
            RunResult(
                strategy_name="GreedyRust",
                weight=w,
                rejected=result.metrics.rejected,
                total_requests=n_customers,
                travel_cost=result.metrics.total_travel_cost,
                wall_time_s=elapsed,
            )
        )

    return results


def print_comparison_table(results: list[RunResult]) -> None:
    weights = sorted({r.weight for r in results})
    strategies = ["ILP", "NFN"]
    if any(r.strategy_name == "GreedyRust" for r in results):
        strategies.append("GreedyRust")

    print()
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    # Header
    print(f"{'':>6}", end="")
    for s in strategies:
        print(f"  {'── ' + s + ' ──':^26}", end="")
    print()
    print(f"{'weight':>6}", end="")
    for _ in strategies:
        print(f"  {'rejected':>8}  {'cost':>10}  {'time(s)':>6}", end="")
    print()
    print("-" * 70)

    for w in weights:
        print(f"  {w:.1f} ", end="")
        for s in strategies:
            row = next(
                (r for r in results if r.strategy_name == s and r.weight == w), None
            )
            if row:
                pct = 100.0 * row.rejected / row.total_requests
                print(
                    f"  {row.rejected:>3}/{row.total_requests} ({pct:4.0f}%)"
                    f"  {row.travel_cost:>10.2f}  {row.wall_time_s:>6.2f}",
                    end="",
                )
            else:
                print(f"  {'N/A':>26}", end="")
        print()

    print()
    print("Gap analysis (online strategies vs ILP):")
    print(
        f"  {'weight':>6}  {'strategy':>10}  {'Δrejected':>10}  {'Δcost':>10}  {'cost ratio':>10}"
    )
    print("  " + "-" * 54)
    for w in weights:
        ilp = next(
            (r for r in results if r.strategy_name == "ILP" and r.weight == w), None
        )
        for sname in [s for s in strategies if s != "ILP"]:
            row = next(
                (r for r in results if r.strategy_name == sname and r.weight == w), None
            )
            if ilp and row:
                delta_rej = row.rejected - ilp.rejected
                delta_cost = row.travel_cost - ilp.travel_cost
                sign_rej = "+" if delta_rej >= 0 else ""
                sign_cost = "+" if delta_cost >= 0 else ""
                ratio_str = (
                    f"{row.travel_cost / ilp.travel_cost:>10.3f}x"
                    if ilp.travel_cost > 0
                    else "       N/A"
                )
                print(
                    f"  {w:>6.1f}  {sname:>10}  {sign_rej}{delta_rej:>9}  {sign_cost}{delta_cost:>9.2f}  {ratio_str}"
                )


if __name__ == "__main__":
    results = run_benchmark()
    print_comparison_table(results)

"""Baseline ILP strategy for DVRPTW.

Formulates the full VRPTW as a Mixed-Integer Linear Program and solves it
ahead of time using the complete instance.  During simulation the strategy
replays the pre-computed plan, dispatching vehicles exactly as the ILP
decided while respecting dynamic release times.

Formulation
-----------
Sets
    N  = {0} ∪ C        depot (0) plus all customer nodes
    K  = {0, …, K-1}    vehicles
    A  = {(i,j) | i,j ∈ N, i≠j}  directed arcs

Decision variables
    x[i,j,k] ∈ {0,1}   vehicle k uses arc (i→j)
    t[i,k]   ≥ 0        service-start time of vehicle k at node i
    y[i]     ∈ {0,1}    customer i is served (1) or rejected (0)

Objective (minimise travel cost + rejections)
    min  w1 · Σ_{i,j,k} dist(i,j)·x[i,j,k]  +  w2 · Σ_i (1 - y[i])

    Both terms are non-negative.  Equivalently, since Σ(1-y[i]) = n - Σ y[i]
    and n is a constant, the ILP minimises:
        w1 · travel_cost  -  w2 · Σ_i y[i]   (plus constant w2·n)

Constraints
    (1)  Flow conservation at every visited node i ∈ N, vehicle k
    (2)  Each customer served at most once across all vehicles
    (3)  Served customer linked to its serving vehicle:
             Σ_k Σ_j x[i,j,k] = y[i]          (flow-out equals service flag)
    (4)  Vehicle capacity per vehicle
    (5)  Time propagation (big-M):
             t[j,k] ≥ t[i,k] + s(i) + travel(i,j) - M·(1 - x[i,j,k])
    (6)  Time-window bounds for every node × vehicle pair
    (7)  Depot departure / return: every vehicle leaves and returns to depot

Dynamic replay
--------------
The ILP is solved once during __init__ using all customer data.  During the
simulation the strategy simply issues DispatchEvent actions following the
pre-computed routes, skipping requests whose time windows have expired or
whose demand would exceed remaining capacity.  Unserved requests are
explicitly rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import os
import shutil

import pulp

from ..instance import DVRPTWInstance, Request
from ..evaluator import Evaluator, StarNormEvaluator
from ..simulator import (
    DispatchEvent,
    RejectEvent,
    WaitEvent,
    SchedulerAction,
    SimulationSnapshot,
    InstanceView,
    PythonDispatchStrategy,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal plan representation
# ---------------------------------------------------------------------------


@dataclass
class _VehiclePlan:
    """Pre-computed route and service times for one vehicle."""

    vehicle_id: int
    stops: list[int]  # customer node IDs in visit order
    service_starts: list[float]  # planned service-start time per stop
    next_stop_idx: int = 0  # pointer to the next stop yet to be dispatched

    @property
    def done(self) -> bool:
        return self.next_stop_idx >= len(self.stops)

    def peek(self) -> tuple[int, float] | None:
        """Return (node_id, planned_service_start) for the next stop, or None."""
        if self.done:
            return None
        i = self.next_stop_idx
        return self.stops[i], self.service_starts[i]

    def advance(self) -> None:
        self.next_stop_idx += 1


# ---------------------------------------------------------------------------
# ILP solver
# ---------------------------------------------------------------------------


def _solve_vrptw(
    instance: DVRPTWInstance,
    evaluator: Evaluator,
    time_limit_s: float,
    mip_gap: float,
    msg: bool,
    cbc_path: str | None = None,
) -> list[_VehiclePlan]:
    """Build and solve the VRPTW ILP; return one plan per vehicle."""

    depot = next(r for r in instance.requests if r.is_depot)
    customers: list[Request] = [r for r in instance.requests if not r.is_depot]

    # Node index list: depot always at position 0
    nodes: list[Request] = [depot] + customers
    node_ids: list[int] = [r.id for r in nodes]
    # idx: dict[int, int] = {r.id: i for i, r in enumerate(nodes)}  # id → list index
    n = len(nodes)
    K = len(instance.vehicles)

    # Precompute Euclidean distances and travel times
    speed = instance.vehicles[0].speed if instance.vehicles else 1.0
    dist: dict[tuple[int, int], float] = {}  # (list-index i, list-index j)
    travel: dict[tuple[int, int], float] = {}
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            d = a.distance_to(b)
            dist[(i, j)] = d
            travel[(i, j)] = d / speed

    capacity = instance.vehicles[0].capacity  # assume homogeneous fleet

    # Big-M for time propagation: upper bound on any service start
    horizon = max(r.time_window.latest for r in nodes) + max(
        r.service_time for r in nodes
    )
    M = horizon + max(travel.values(), default=0.0) + 1.0

    # -----------------------------------------------------------------------
    # Problem definition
    # -----------------------------------------------------------------------
    prob = pulp.LpProblem("VRPTW", pulp.LpMinimize)

    # x[i,j,k] = 1 iff vehicle k travels from node-index i to node-index j
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j, k) for i in range(n) for j in range(n) for k in range(K) if i != j],
        cat=pulp.const.LpBinary,
    )

    # t[i,k] = service-start time of vehicle k at node-index i
    t = pulp.LpVariable.dicts(
        "t",
        [(i, k) for i in range(n) for k in range(K)],
        lowBound=0.0,
        cat=pulp.const.LpContinuous,
    )

    # y[c] = 1 iff customer at node-index c (c >= 1) is served
    y = pulp.LpVariable.dicts(
        "y",
        list(range(1, n)),  # customers only (index 0 = depot)
        cat=pulp.const.LpBinary,
    )

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    travel_cost = pulp.lpSum(
        dist[(i, j)] * x[(i, j, k)]
        for i in range(n)
        for j in range(n)
        for k in range(K)
        if i != j
    )
    served_count = pulp.lpSum(y[c] for c in range(1, n))

    # Derive ILP coefficients from the evaluator.
    # The evaluator's scalar is:  w1·cost_norm + w2·rejected_norm
    # Since rejected = n - served, the coefficient on the served variable is -w2·(norm factor).
    # star_cost is passed as context even if the evaluator doesn't use it.
    depot_idx = 0
    star_cost = sum(2.0 * dist[(depot_idx, c)] for c in range(1, n)) or 1.0
    max_customers = float(len(customers)) or 1.0

    c1, c2 = evaluator.ilp_coefficients(star_cost, max_customers)
    # c1 > 0 penalises travel; c2 < 0 rewards service (= penalises rejection)
    prob += (c1 * travel_cost + c2 * served_count, "weighted_objective")

    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    # (C1) Depot flow: every vehicle leaves and returns to depot exactly once
    #      (allow vehicles to stay unused by setting both to 0)
    for k in range(K):
        # out-flow from depot ≤ 1
        prob += (
            pulp.lpSum(x[(0, j, k)] for j in range(1, n)) <= 1,
            f"depot_out_k{k}",
        )
        # in-flow to depot ≤ 1
        prob += (
            pulp.lpSum(x[(i, 0, k)] for i in range(1, n)) <= 1,
            f"depot_in_k{k}",
        )
        # flow balance: out == in
        prob += (
            pulp.lpSum(x[(0, j, k)] for j in range(1, n))
            == pulp.lpSum(x[(i, 0, k)] for i in range(1, n)),
            f"depot_balance_k{k}",
        )

    # (C2) Flow conservation at customer nodes
    for c in range(1, n):
        for k in range(K):
            prob += (
                pulp.lpSum(x[(c, j, k)] for j in range(n) if j != c)
                == pulp.lpSum(x[(i, c, k)] for i in range(n) if i != c),
                f"flow_conserv_c{c}_k{k}",
            )

    # (C3) Each customer is served by at most one vehicle
    #      (linking y[c] to the flow)
    for c in range(1, n):
        prob += (
            pulp.lpSum(x[(i, c, k)] for i in range(n) for k in range(K) if i != c)
            == y[c],
            f"serve_c{c}",
        )

    # (C4) Vehicle capacity
    for k in range(K):
        prob += (
            pulp.lpSum(
                nodes[c].demand * pulp.lpSum(x[(i, c, k)] for i in range(n) if i != c)
                for c in range(1, n)
            )
            <= capacity,
            f"capacity_k{k}",
        )

    # (C5) Time propagation (big-M linearisation of arc-conditional constraint)
    for i in range(n):
        for j in range(1, n):  # depot has no upper service-time constraint as dest
            if i == j:
                continue
            for k in range(K):
                prob += (
                    t[(j, k)]
                    >= t[(i, k)]
                    + nodes[i].service_time
                    + travel[(i, j)]
                    - M * (1 - x[(i, j, k)]),
                    f"time_prop_i{i}_j{j}_k{k}",
                )

    # (C6) Time-window bounds
    for i in range(n):
        tw = nodes[i].time_window
        for k in range(K):
            # Lower bound always applies (vehicle must not start before opening
            # or before the request's release_time).  The original formulation
            # did not consider dynamic release times, which made ILP plans
            # unreplayable during the online simulation: a service scheduled
            # before a request's release would be delayed at replay time and
            # could miss the time-window.  Enforce t >= release_time here so
            # planned service starts are causally feasible during replay.
            release_lo = getattr(nodes[i], "release_time", 0.0)
            lo = max(tw.earliest, release_lo)
            prob += (t[(i, k)] >= lo, f"tw_lo_i{i}_k{k}")
            # Upper bound: if not visited by this vehicle, t is unconstrained
            # but we bound it anyway (it won't affect objective since x=0)
            prob += (t[(i, k)] <= tw.latest, f"tw_hi_i{i}_k{k}")

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    # Determine CBC executable path. Preference order:
    # 1) explicit cbc_path argument
    # 2) DVRPTW_CBC_PATH environment variable
    # 3) fall back to default behavior (let PuLP find it)
    # Resolve CBC path with preference: explicit arg -> env var -> PATH 'cbc'
    resolved_path: str | None = None
    if cbc_path:
        resolved_path = cbc_path if os.path.isabs(cbc_path) else shutil.which(cbc_path)
    if not resolved_path:
        env_path = os.environ.get("DVRPTW_CBC_PATH")
        if env_path:
            resolved_path = (
                env_path if os.path.isabs(env_path) else shutil.which(env_path)
            )
    if not resolved_path:
        resolved_path = shutil.which("cbc")

    if resolved_path:
        # Use generic COIN_CMD when an explicit path is provided; PULP_CBC_CMD
        # forbids setting a custom path in some pulp versions.
        solver = pulp.COIN_CMD(
            path=resolved_path,
            timeLimit=time_limit_s,
            gapRel=mip_gap,
            msg=1 if msg else 0,
        )
    else:
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit_s,
            gapRel=mip_gap,
            msg=1 if msg else 0,
        )
    prob.solve(solver)
    log.info(
        "ILP solved: status=%s objective=%.4f",
        pulp.LpStatus[prob.status],
        pulp.value(prob.objective) or 0.0,
    )

    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        # "Not Solved" can occur on time-limit with a feasible incumbent
        pass  # proceed with whatever solution is available

    # -----------------------------------------------------------------------
    # Extract solution
    # -----------------------------------------------------------------------
    plans: list[_VehiclePlan] = []

    for k in range(K):
        # Build adjacency: next_node[i] = j  for arcs with x[i,j,k] ~ 1
        next_node: dict[int, int] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = pulp.value(x.get((i, j, k), 0)) or 0.0
                if val > 0.5:
                    next_node[i] = j

        # Trace route starting from depot (index 0)
        stops: list[int] = []
        service_starts: list[float] = []
        current = 0
        visited: set[int] = {0}
        while current in next_node:
            nxt = next_node[current]
            if nxt == 0 or nxt in visited:
                break
            visited.add(nxt)
            stops.append(node_ids[nxt])
            svc_t = pulp.value(t.get((nxt, k), 0)) or 0.0
            service_starts.append(max(svc_t, nodes[nxt].time_window.earliest))
            current = nxt

        plans.append(
            _VehiclePlan(
                vehicle_id=instance.vehicles[k].id,
                stops=stops,
                service_starts=service_starts,
            )
        )

    return plans


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ILPStrategy(PythonDispatchStrategy):
    """VRPTW ILP strategy that solves ahead-of-time and replays during simulation.

    Since an exact ILP requires the full instance to be known up front, this
    strategy is constructed directly from a ``DVRPTWInstance`` and solves the
    problem once during ``__init__``.  The simulation then replays the plan.

    Parameters
    ----------
    instance:
        Complete DVRPTW instance.  All requests (including those with non-zero
        ``release_time``) are used to build the ILP — this is the key
        limitation that makes this strategy a *non-causal baseline*: it has
        full foreknowledge of future requests.
    evaluator:
        Objective evaluator that collapses the two raw objectives (travel cost
        and service count) into a single minimisation scalar.  Defaults to
        ``StarNormEvaluator`` with equal weights (w1=0.5, w2=0.5), which
        normalises both objectives by instance-derived bounds.
    time_limit_s:
        CBC solver wall-clock time limit in seconds.  Defaults to 60.
    mip_gap:
        Relative MIP optimality gap at which to stop early.  Defaults to 0.01
        (1 %).
    msg:
        If True, print CBC solver output.  Defaults to False.
    """

    def __init__(
        self,
        instance: DVRPTWInstance,
        *,
        evaluator: Evaluator | None = None,
        time_limit_s: float = 60.0,
        mip_gap: float = 0.01,
        msg: bool = False,
        cbc_path: str | None = None,
    ) -> None:
        self._instance = instance
        if evaluator is None:
            evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        # Resolve CBC path: prefer explicit arg, then env var, then PATH lookup
        if cbc_path is None:
            cbc_path = os.environ.get("DVRPTW_CBC_PATH") or shutil.which("cbc")

        self._plans = _solve_vrptw(
            instance,
            evaluator=evaluator,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
            msg=msg,
            cbc_path=cbc_path,
        )
        # Map vehicle_id -> plan for fast lookup
        self._plan_by_vehicle: dict[int, _VehiclePlan] = {
            p.vehicle_id: p for p in self._plans
        }
        # Track which requests have been rejected by the strategy already
        self._rejected: set[int] = set()
        # Track which requests have been dispatched
        self._dispatched: set[int] = set()

    # ------------------------------------------------------------------
    # DispatchStrategy protocol
    # ------------------------------------------------------------------

    def next_events(
        self, state: SimulationSnapshot, view: InstanceView
    ) -> list[SchedulerAction]:
        actions: list[SchedulerAction] = []
        instance_view = view

        # Reject any pending request that the ILP plan does not serve
        planned_requests: set[int] = {
            node for plan in self._plans for node in plan.stops
        }
        for req_id in state.pending:
            if req_id not in planned_requests and req_id not in self._rejected:
                self._rejected.add(req_id)
                actions.append(RejectEvent(request_id=req_id))

        # Dispatch idle vehicles according to the pre-computed plan
        idle_vehicles = {
            v.vehicle_id: v for v in state.vehicles if v.available_at <= state.time
        }

        earliest_next_event: float | None = None

        for vid, vehicle in idle_vehicles.items():
            plan = self._plan_by_vehicle.get(vid)
            if plan is None or plan.done:
                continue

            node_id, planned_start = plan.peek()  # type: ignore[misc]

            # Check if the request has been released yet
            if node_id not in instance_view.released_requests:
                # Not yet released — we need to wait until it is
                try:
                    release_t = self._instance.get_request(node_id).release_time
                    earliest_next_event = (
                        release_t
                        if earliest_next_event is None
                        else min(earliest_next_event, release_t)
                    )
                except KeyError:
                    pass
                continue

            # Check if the request is still pending (not already served/rejected)
            if node_id not in state.pending:
                # Already handled (served by another vehicle or rejected);
                # skip this stop and try the next one
                plan.advance()
                # Recurse — try the next stop in the same event loop iteration
                # by checking again (safe because we just advanced the pointer)
                peek = plan.peek()
                if peek is None:
                    continue
                node_id, planned_start = peek
                if node_id not in state.pending:
                    continue  # give up for this cycle; will retry next event

            # Compute travel time from vehicle current position to the node so
            # we can decide whether to dispatch now or wait until the planned
            # departure that yields the ILP's planned service start.
            try:
                from_req = self._instance.get_request(vehicle.position)
            except KeyError:
                # Missing position info: dispatch immediately and hope for best
                actions.append(DispatchEvent(vehicle_id=vid, destination_node=node_id))
                self._dispatched.add(node_id)
                plan.advance()
                continue

            to_req = self._instance.get_request(node_id)
            dist = from_req.distance_to(to_req)
            speed = self._instance.vehicles[0].speed if self._instance.vehicles else 1.0
            travel_time = dist / speed if speed > 0.0 else float("inf")

            # planned_start is the ILP's service-start time for this stop.
            # To achieve it we should depart at planned_departure = planned_start - travel_time.
            planned_departure = planned_start - travel_time

            # If the planned departure is in the future, request a wait until
            # that time (the simulator will wake us then).  Otherwise dispatch now.
            if planned_departure > state.time:
                earliest_next_event = (
                    planned_departure
                    if earliest_next_event is None
                    else min(earliest_next_event, planned_departure)
                )
                continue

            # Before dispatching, validate feasibility under the current
            # runtime snapshot: capacity and time-window checks. If the
            # dispatch would be infeasible now, reject the request (the ILP
            # planned it assuming ideal timings which may not hold during
            # replay) and advance the plan pointer.
            dest_req = self._instance.get_request(node_id)
            # capacity check (depots always allowed)
            veh_spec = next(
                (vs for vs in self._instance.vehicles if vs.id == vid), None
            )
            cap = veh_spec.capacity if veh_spec is not None else float("inf")
            if (not dest_req.is_depot) and (
                vehicle.current_load + dest_req.demand > cap
            ):
                # cannot serve due to capacity → reject and skip
                self._rejected.add(node_id)
                actions.append(RejectEvent(request_id=node_id))
                plan.advance()
                continue

            # compute arrival/service start if we depart now
            arrival = state.time + travel_time
            service_start = max(arrival, dest_req.time_window.earliest)
            if service_start > dest_req.time_window.latest:
                # cannot meet time window → reject and skip
                self._rejected.add(node_id)
                actions.append(RejectEvent(request_id=node_id))
                plan.advance()
                continue

            # dispatch now
            actions.append(DispatchEvent(vehicle_id=vid, destination_node=node_id))
            self._dispatched.add(node_id)
            plan.advance()

        # If no actions and there are pending requests, emit a WaitEvent so the
        # simulator wakes us up at the next relevant time.  Only consider
        # candidate times strictly in the future (>
        # state.time) to avoid producing invalid WaitEvents.
        if not actions:
            candidates: list[float] = []
            if earliest_next_event is not None and earliest_next_event > state.time:
                candidates.append(earliest_next_event)
            # Wake up when the next vehicle becomes idle
            busy_times = [
                v.available_at for v in state.vehicles if v.available_at > state.time
            ]
            if busy_times:
                candidates.append(min(busy_times))
            if candidates:
                until = min(candidates)
                if until > state.time:
                    actions.append(WaitEvent(until_time=until))

        return actions

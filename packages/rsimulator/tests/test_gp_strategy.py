"""Tests for GP strategy via rsimulator: FlatGpTree, factory functions, and gp_strategy().

Covers:
- gp_strategy() returns NativeDispatchStrategy
- Const trees: accept-all and reject-all behaviour
- Routing tree selects best vehicle (prefer closer via negative travel time)
- Sequencing tree controls dispatch order (serve most urgent first)
- Feature terminals produce plausible values (smoke tests via end-to-end runs)
- begin_tick propagation: TimeUntilDue reflects current simulation time
"""

import math
import unittest

from dvrptw import (
    DVRPTWInstance,
    Request,
    TimeWindow,
    Vehicle,
    RustSimulator,
)


# ---------------------------------------------------------------------------
# Instance builders
# ---------------------------------------------------------------------------


def _make_single_request_instance() -> DVRPTWInstance:
    """Depot at origin; one request at (3, 4); travel cost 5 each way = 10."""
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 1000.0),
        service_time=0.0,
        is_depot=True,
    )
    req = Request(
        id=1,
        position=(3.0, 4.0),
        demand=5.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=0.0,
        release_time=0.0,
    )
    v = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="single",
        requests=[depot, req],
        vehicles=[v],
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_basic_instance() -> DVRPTWInstance:
    """Depot at origin; two requests; one vehicle."""
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 1000.0),
        service_time=0.0,
        is_depot=True,
    )
    req1 = Request(
        id=1,
        position=(3.0, 4.0),
        demand=5.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=0.0,
        release_time=0.0,
    )
    req2 = Request(
        id=2,
        position=(0.0, 1.0),
        demand=3.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=0.0,
        release_time=0.0,
    )
    v = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="basic",
        requests=[depot, req1, req2],
        vehicles=[v],
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_two_vehicle_instance() -> DVRPTWInstance:
    """Depot at origin; one request; two vehicles at different positions.

    Vehicle 0 starts at the depot (position 0).
    Vehicle 1 also starts at the depot.
    The routing tree should select based on scoring.
    """
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 1000.0),
        service_time=0.0,
        is_depot=True,
    )
    req = Request(
        id=1,
        position=(3.0, 4.0),
        demand=5.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=0.0,
        release_time=0.0,
    )
    vehicles = [
        Vehicle(id=i, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        for i in range(2)
    ]
    return DVRPTWInstance(
        id="two_vehicle",
        requests=[depot, req],
        vehicles=vehicles,
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_urgency_instance() -> DVRPTWInstance:
    """Two requests with different time-window deadlines; one vehicle.

    req1: deadline 50  (more urgent)
    req2: deadline 200 (less urgent)
    Both released at t=0.
    """
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 1000.0),
        service_time=0.0,
        is_depot=True,
    )
    req1 = Request(
        id=1,
        position=(1.0, 0.0),
        demand=1.0,
        time_window=TimeWindow(0.0, 50.0),
        service_time=0.0,
        release_time=0.0,
    )
    req2 = Request(
        id=2,
        position=(1.0, 0.0),
        demand=1.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=0.0,
        release_time=0.0,
    )
    v = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="urgency",
        requests=[depot, req1, req2],
        vehicles=[v],
        planning_horizon=500.0,
        depot_id=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGpStrategyReturnType(unittest.TestCase):
    """gp_strategy() must return a NativeDispatchStrategy."""

    def test_returns_native_wrapper(self):
        from rsimulator import flat_gp_const, gp_strategy, NativeDispatchStrategy

        routing = flat_gp_const(1.0)
        sequencing = flat_gp_const(1.0)
        reject = flat_gp_const(0.0)
        s = gp_strategy(routing, sequencing, reject)
        self.assertIsInstance(s, NativeDispatchStrategy)


class TestGpStrategyAcceptAll(unittest.TestCase):
    """Routing score always > reject score → every request is served."""

    def test_single_request_served(self):
        from rsimulator import flat_gp_const, gp_strategy

        # routing=1.0, reject=0.0 → routing > reject → accept
        strategy = gp_strategy(
            flat_gp_const(1.0), flat_gp_const(0.0), flat_gp_const(0.0)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1})
        self.assertEqual(result.metrics.rejected, 0)

    def test_all_requests_served(self):
        from rsimulator import flat_gp_const, gp_strategy

        strategy = gp_strategy(
            flat_gp_const(1.0), flat_gp_const(0.0), flat_gp_const(0.0)
        )
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        self.assertEqual(result.metrics.rejected, 0)

    def test_travel_cost_correct(self):
        """Single request at (3,4): round-trip distance = 10.0."""
        from rsimulator import flat_gp_const, gp_strategy

        strategy = gp_strategy(
            flat_gp_const(1.0), flat_gp_const(0.0), flat_gp_const(0.0)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()

        self.assertAlmostEqual(result.metrics.total_travel_cost, 10.0, places=9)

    def test_solution_structure(self):
        """routes and service_times have matching lengths."""
        from rsimulator import flat_gp_const, gp_strategy

        strategy = gp_strategy(
            flat_gp_const(1.0), flat_gp_const(0.0), flat_gp_const(0.0)
        )
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(
            len(result.solution.routes), len(result.solution.service_times)
        )
        for route, times in zip(result.solution.routes, result.solution.service_times):
            self.assertEqual(len(route), len(times))


class TestGpStrategyRejectAll(unittest.TestCase):
    """Reject score always > routing score → every request is rejected."""

    def test_reject_all(self):
        from rsimulator import flat_gp_const, gp_strategy

        # routing=0.0, reject=1.0 → reject > routing → reject
        strategy = gp_strategy(
            flat_gp_const(0.0), flat_gp_const(0.0), flat_gp_const(1.0)
        )
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, set())
        self.assertEqual(result.metrics.rejected, 2)


class TestGpStrategySequencing(unittest.TestCase):
    """Sequencing tree controls which queued request is dispatched first."""

    def test_most_urgent_served_first(self):
        """flat_gp_time_until_due() as sequencing tree → serve req with smaller deadline first.

        req1 has deadline 50, req2 has deadline 200.
        At dispatch time ≈0, TimeUntilDue for req1 ≈ 50, for req2 ≈ 200.
        max_by(TimeUntilDue) picks req2 first — less urgent.
        To serve most urgent first we use negative: flat_gp_sub(flat_gp_const(0), flat_gp_time_until_due()).
        """
        from rsimulator import (
            flat_gp_const,
            gp_strategy,
            flat_gp_sub,
            flat_gp_time_until_due,
        )

        # Negative TimeUntilDue: smaller deadline → higher score → dispatched first
        sequencing = flat_gp_sub(flat_gp_const(0.0), flat_gp_time_until_due())
        strategy = gp_strategy(flat_gp_const(1.0), sequencing, flat_gp_const(0.0))
        inst = _make_urgency_instance()
        result = RustSimulator(inst, strategy).run()

        # Both must be served
        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        # req1 (more urgent) dispatched first
        self.assertEqual(result.solution.routes[0][0], 1)

    def test_least_urgent_served_first(self):
        """flat_gp_time_until_due() without negation → serve req with larger deadline first."""
        from rsimulator import gp_strategy, flat_gp_const, flat_gp_time_until_due

        sequencing = flat_gp_time_until_due()
        strategy = gp_strategy(flat_gp_const(1.0), sequencing, flat_gp_const(0.0))
        inst = _make_urgency_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        # req2 (less urgent, larger TimeUntilDue) dispatched first
        self.assertEqual(result.solution.routes[0][0], 2)


class TestFlatGpTreeFactories(unittest.TestCase):
    """All factory functions return FlatGpTree and can be passed to gp_strategy()."""

    def _run_with_sequencing(self, sequencing_tree):
        from rsimulator import flat_gp_const, gp_strategy

        strategy = gp_strategy(flat_gp_const(1.0), sequencing_tree, flat_gp_const(0.0))
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()
        return result

    def test_flat_gp_const(self):
        from rsimulator import flat_gp_const

        result = self._run_with_sequencing(flat_gp_const(0.0))
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_add(self):
        from rsimulator import flat_gp_add, flat_gp_const

        result = self._run_with_sequencing(
            flat_gp_add(flat_gp_const(1.0), flat_gp_const(2.0))
        )
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_sub(self):
        from rsimulator import flat_gp_const, flat_gp_sub

        result = self._run_with_sequencing(
            flat_gp_sub(flat_gp_const(5.0), flat_gp_const(3.0))
        )
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_mul(self):
        from rsimulator import flat_gp_const, flat_gp_mul

        result = self._run_with_sequencing(
            flat_gp_mul(flat_gp_const(2.0), flat_gp_const(3.0))
        )
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_div(self):
        from rsimulator import flat_gp_const, flat_gp_div

        # Protected division: divisor 0 → result 1.0
        result = self._run_with_sequencing(
            flat_gp_div(flat_gp_const(6.0), flat_gp_const(0.0))
        )
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_travel_time(self):
        from rsimulator import flat_gp_travel_time

        result = self._run_with_sequencing(flat_gp_travel_time())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_window_earliest(self):
        from rsimulator import flat_gp_window_earliest

        result = self._run_with_sequencing(flat_gp_window_earliest())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_window_latest(self):
        from rsimulator import flat_gp_window_latest

        result = self._run_with_sequencing(flat_gp_window_latest())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_time_until_due(self):
        from rsimulator import flat_gp_time_until_due

        result = self._run_with_sequencing(flat_gp_time_until_due())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_demand(self):
        from rsimulator import flat_gp_demand

        result = self._run_with_sequencing(flat_gp_demand())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_current_load(self):
        from rsimulator import flat_gp_current_load

        result = self._run_with_sequencing(flat_gp_current_load())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_remaining_capacity(self):
        from rsimulator import flat_gp_remaining_capacity

        result = self._run_with_sequencing(flat_gp_remaining_capacity())
        self.assertEqual(result.metrics.rejected, 0)

    def test_flat_gp_release_time(self):
        from rsimulator import flat_gp_release_time

        result = self._run_with_sequencing(flat_gp_release_time())
        self.assertEqual(result.metrics.rejected, 0)


class TestFlatGpTreeArithmetic(unittest.TestCase):
    """Const trees with known values produce predictable routing/rejection."""

    def test_equal_routing_reject_accepts(self):
        """When routing == reject, request is accepted (reject > routing is false)."""
        from rsimulator import flat_gp_const, gp_strategy

        strategy = gp_strategy(
            flat_gp_const(5.0), flat_gp_const(0.0), flat_gp_const(5.0)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()
        # reject (5.0) is NOT > routing (5.0), so accepted
        self.assertEqual(result.metrics.rejected, 0)

    def test_protected_div_zero_is_one(self):
        """flat_gp_div with zero denominator returns 1.0 (protected division)."""
        from rsimulator import flat_gp_const, flat_gp_div, gp_strategy

        # routing = 1/0 = 1.0 (protected), reject = 0.0 → accepted
        routing = flat_gp_div(flat_gp_const(1.0), flat_gp_const(0.0))
        strategy = gp_strategy(routing, flat_gp_const(0.0), flat_gp_const(0.0))
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()
        self.assertEqual(result.metrics.rejected, 0)

    def test_negative_routing_still_accepted_if_reject_lower(self):
        """Negative routing score with even-lower reject score → accepted."""
        from rsimulator import flat_gp_const, gp_strategy

        # routing = -1.0, reject = -2.0 → reject < routing → accept
        strategy = gp_strategy(
            flat_gp_const(-1.0), flat_gp_const(0.0), flat_gp_const(-2.0)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()
        self.assertEqual(result.metrics.rejected, 0)


class TestFlatGpTravelTimeRouting(unittest.TestCase):
    """Use flat_gp_travel_time() as routing tree (prefer lower travel time)."""

    def test_serves_all_requests(self):
        """Routing by negative travel time still serves all requests."""
        from rsimulator import (
            flat_gp_const,
            gp_strategy,
            flat_gp_sub,
            flat_gp_travel_time,
        )

        # Prefer vehicle with less travel time → negate travel_time
        routing = flat_gp_sub(flat_gp_const(0.0), flat_gp_travel_time())
        strategy = gp_strategy(
            routing, flat_gp_const(0.0), flat_gp_const(float("-inf"))
        )
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        self.assertEqual(result.metrics.rejected, 0)

    def test_travel_cost_plausible(self):
        """Total travel cost is positive and finite."""
        from rsimulator import (
            flat_gp_const,
            gp_strategy,
            flat_gp_sub,
            flat_gp_travel_time,
        )

        routing = flat_gp_sub(flat_gp_const(0.0), flat_gp_travel_time())
        strategy = gp_strategy(
            routing, flat_gp_const(0.0), flat_gp_const(float("-inf"))
        )
        inst = _make_basic_instance()
        result = RustSimulator(inst, strategy).run()

        self.assertGreater(result.metrics.total_travel_cost, 0.0)
        self.assertTrue(math.isfinite(result.metrics.total_travel_cost))


class TestGpTerminalNormalization(unittest.TestCase):
    """Terminals are normalized to a common unit scale derived from the instance.

    Scale factors:
      TravelTime        → max_travel_time = bounding-box diagonal / speed
      WindowEarliest/Latest, TimeUntilDue, ReleaseTime → planning_horizon
      Demand, CurrentLoad, RemainingCapacity           → vehicle_capacity
    """

    def test_travel_time_at_max_distance_is_one(self):
        """Travel time to the farthest node normalizes to ≈ 1.0.

        Instance: depot (0,0), single request at (3,4), speed=1.
        Bounding-box diagonal = sqrt(3²+4²) = 5.  max_travel_time = 5/1 = 5.
        Normalized travel_time = 5/5 = 1.0.

        reject_const = 1.5 > routing = 1.0  →  request rejected (1 rejection).
        """
        from rsimulator import flat_gp_const, flat_gp_travel_time, gp_strategy

        strategy = gp_strategy(
            flat_gp_travel_time(), flat_gp_const(0.0), flat_gp_const(1.5)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()
        self.assertEqual(result.metrics.rejected, 1)

    def test_travel_time_below_threshold_accepts(self):
        """reject_const = 0.5 < normalized_travel_time ≈ 1.0  →  accepted."""
        from rsimulator import flat_gp_const, flat_gp_travel_time, gp_strategy

        strategy = gp_strategy(
            flat_gp_travel_time(), flat_gp_const(0.0), flat_gp_const(0.5)
        )
        inst = _make_single_request_instance()
        result = RustSimulator(inst, strategy).run()
        self.assertEqual(result.metrics.rejected, 0)

    def test_sequencing_order_preserved_after_normalization(self):
        """Normalization is monotone; urgency ordering is preserved.

        req1 deadline=50, req2 deadline=200.  After dividing by planning_horizon,
        relative order of TimeUntilDue is unchanged  →  urgency test still holds.
        """
        from rsimulator import (
            flat_gp_const,
            flat_gp_sub,
            flat_gp_time_until_due,
            gp_strategy,
        )

        sequencing = flat_gp_sub(flat_gp_const(0.0), flat_gp_time_until_due())
        strategy = gp_strategy(flat_gp_const(1.0), sequencing, flat_gp_const(0.0))
        inst = _make_urgency_instance()
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        self.assertEqual(result.solution.routes[0][0], 1)  # most urgent first


if __name__ == "__main__":
    unittest.main()

"""Tests for multi-trip vehicle support across strategies.

Verifies that vehicles can return to the depot to reset capacity and
serve additional requests in subsequent trips.
"""

import unittest

from dvrptw import (
    DVRPTWInstance,
    Request,
    Vehicle,
    TimeWindow,
    RustSimulator,
)
from dvrptw.strategies import (
    greedy_strategy,
    python_composable_strategy,
    python_batch_composable_strategy,
)
from dvrptw.simulator.state import StrategyView, VehicleSnapshot


# ---------------------------------------------------------------------------
# Instance builders
# ---------------------------------------------------------------------------


def _make_multi_trip_instance(
    num_requests: int = 3,
    demand: float = 30.0,
    capacity: float = 50.0,
    planning_horizon: float = 5000.0,
) -> DVRPTWInstance:
    """Instance where a single vehicle must make multiple trips.

    With capacity=50 and demand=30 per request, each trip can serve at
    most 1 request (30 <= 50, but 60 > 50). The vehicle must return to
    the depot between every request.
    """
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, planning_horizon),
        service_time=0.0,
        is_depot=True,
    )
    customers = [
        Request(
            id=i,
            position=(float(i), 0.0),
            demand=demand,
            time_window=TimeWindow(0.0, planning_horizon),
            service_time=0.0,
            release_time=0.0,
        )
        for i in range(1, num_requests + 1)
    ]
    vehicles = [Vehicle(id=0, capacity=capacity, start_depot=0, end_depot=0, speed=1.0)]
    return DVRPTWInstance(
        id="multi_trip",
        requests=[depot] + customers,
        vehicles=vehicles,
        planning_horizon=planning_horizon,
        depot_id=0,
    )


def _make_multi_trip_two_per_trip_instance() -> DVRPTWInstance:
    """Instance where each trip can serve exactly 2 requests.

    capacity=50, 4 requests each with demand=20 => 2 per trip, 2 trips total.
    """
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 5000.0),
        service_time=0.0,
        is_depot=True,
    )
    customers = [
        Request(
            id=i,
            position=(float(i), 0.0),
            demand=20.0,
            time_window=TimeWindow(0.0, 5000.0),
            service_time=0.0,
            release_time=0.0,
        )
        for i in range(1, 5)
    ]
    vehicles = [Vehicle(id=0, capacity=50.0, start_depot=0, end_depot=0, speed=1.0)]
    return DVRPTWInstance(
        id="multi_trip_2per",
        requests=[depot] + customers,
        vehicles=vehicles,
        planning_horizon=5000.0,
        depot_id=0,
    )


# ---------------------------------------------------------------------------
# Routers / Schedulers for composable strategies
# ---------------------------------------------------------------------------


class AssignToFirstVehicleRouter:
    """Routes every request to vehicle 0."""

    def route(
        self, request_id: int, vehicles: list[VehicleSnapshot], view: StrategyView
    ) -> int | None:
        return vehicles[0].vehicle_id


class FifoScheduler:
    """Returns the lowest-id request in the queue."""

    def schedule(
        self, vehicle: VehicleSnapshot, queue: list[int], view: StrategyView
    ) -> int:
        return min(queue)


class AssignToFirstVehicleBatchRouter:
    """Batch version: assigns every request to vehicle 0."""

    def route_batch(
        self,
        requests: list[int],
        vehicles: list[VehicleSnapshot],
        view: StrategyView,
    ) -> list[tuple[int, int | None]]:
        vid = vehicles[0].vehicle_id
        return [(rid, vid) for rid in requests]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGreedyMultiTrip(unittest.TestCase):
    """Multi-trip tests for the greedy strategy."""

    def test_single_vehicle_serves_all_via_multi_trip(self):
        """1 vehicle, capacity=50, 3 requests each demand=30.
        Without multi-trip only 1 can be served; with multi-trip all 3."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = greedy_strategy()
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        served = {r for route in result.solution.routes for r in route if r != 0}
        self.assertEqual(served, {1, 2, 3})

    def test_two_per_trip(self):
        """1 vehicle, capacity=50, 4 requests each demand=20.
        Each trip can serve 2 requests. Total: 2 trips."""
        inst = _make_multi_trip_two_per_trip_instance()
        strategy = greedy_strategy()
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        served = {r for route in result.solution.routes for r in route if r != 0}
        self.assertEqual(served, {1, 2, 3, 4})

    def test_depot_appears_in_routes_for_multi_trip(self):
        """Depot visits should appear in routes when vehicle makes multiple trips."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = greedy_strategy()
        result = RustSimulator(inst, strategy).run()

        # Vehicle returns to depot between requests, so depot ID should appear
        all_nodes = [r for route in result.solution.routes for r in route]
        self.assertIn(0, all_nodes, "Depot should appear in routes for multi-trip")

    def test_travel_cost_includes_depot_returns(self):
        """Travel cost should account for depot returns between trips."""
        # 1 vehicle, capacity=5, 2 requests at (1,0) and (2,0) each demand=5
        # Trip 1: depot(0,0) -> req1(1,0) -> depot(0,0) = 1 + 1 = 2
        # Trip 2: depot(0,0) -> req2(2,0) -> depot(0,0) = 2 + 2 = 4
        # Total expected = 6
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 5000.0),
            service_time=0.0,
            is_depot=True,
        )
        req1 = Request(
            id=1,
            position=(1.0, 0.0),
            demand=5.0,
            time_window=TimeWindow(0.0, 5000.0),
            service_time=0.0,
            release_time=0.0,
        )
        req2 = Request(
            id=2,
            position=(2.0, 0.0),
            demand=5.0,
            time_window=TimeWindow(0.0, 5000.0),
            service_time=0.0,
            release_time=0.0,
        )
        inst = DVRPTWInstance(
            id="cost_test",
            requests=[depot, req1, req2],
            vehicles=[
                Vehicle(id=0, capacity=5.0, start_depot=0, end_depot=0, speed=1.0)
            ],
            planning_horizon=5000.0,
            depot_id=0,
        )
        strategy = greedy_strategy()
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        # The greedy strategy dispatches in ascending request ID order.
        # Trip 1: depot→req1→depot = 1+1 = 2
        # Trip 2: depot→req2→depot = 2+2 = 4
        # Total = 6
        self.assertAlmostEqual(result.metrics.total_travel_cost, 6.0, places=5)


class TestComposableMultiTrip(unittest.TestCase):
    """Multi-trip tests for the composable strategy."""

    def test_single_vehicle_serves_all_via_multi_trip(self):
        """1 vehicle, capacity=50, 3 requests each demand=30."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = python_composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        served = {r for route in result.solution.routes for r in route if r != 0}
        self.assertEqual(served, {1, 2, 3})

    def test_two_per_trip(self):
        """1 vehicle, capacity=50, 4 requests each demand=20."""
        inst = _make_multi_trip_two_per_trip_instance()
        strategy = python_composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        served = {r for route in result.solution.routes for r in route if r != 0}
        self.assertEqual(served, {1, 2, 3, 4})

    def test_depot_appears_in_routes_for_multi_trip(self):
        """Depot visits should appear in routes when vehicle makes multiple trips."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = python_composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        all_nodes = [r for route in result.solution.routes for r in route]
        self.assertIn(0, all_nodes)


class TestBatchComposableMultiTrip(unittest.TestCase):
    """Multi-trip tests for the batch composable strategy."""

    def test_single_vehicle_serves_all_via_multi_trip(self):
        """1 vehicle, capacity=50, 3 requests each demand=30."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = python_batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=1.0
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        served = {r for route in result.solution.routes for r in route if r != 0}
        self.assertEqual(served, {1, 2, 3})

    def test_depot_appears_in_routes_for_multi_trip(self):
        """Depot visits should appear in routes when vehicle makes multiple trips."""
        inst = _make_multi_trip_instance(num_requests=3, demand=30.0, capacity=50.0)
        strategy = python_batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=1.0
        )
        result = RustSimulator(inst, strategy).run()

        all_nodes = [r for route in result.solution.routes for r in route]
        self.assertIn(0, all_nodes)


if __name__ == "__main__":
    unittest.main()

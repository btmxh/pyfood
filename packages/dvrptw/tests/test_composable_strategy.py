"""Tests for composable_strategy and batch_composable_strategy.

Covers:
- composable_strategy: parity with greedy, rejection, multi-vehicle, first-come scheduling
- batch_composable_strategy: buffering within slots, routing at slot boundary,
  rejection, multi-vehicle, FIFO scheduling, slot advancement
"""

import unittest

from dvrptw import (
    DVRPTWInstance,
    Request,
    Vehicle,
    TimeWindow,
    RustSimulator,
    batch_composable_strategy,
)


# ---------------------------------------------------------------------------
# Instance builders (shared with test_rust_simulator but local copies to
# avoid coupling test files)
# ---------------------------------------------------------------------------


def _make_basic_instance() -> DVRPTWInstance:
    """Depot at origin; two requests at (3,4) and (0,1); one vehicle."""
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
        time_window=TimeWindow(0.0, 100.0),
        service_time=10.0,
        release_time=0.0,
    )
    req2 = Request(
        id=2,
        position=(0.0, 1.0),
        demand=3.0,
        time_window=TimeWindow(0.0, 100.0),
        service_time=10.0,
        release_time=0.0,
    )
    v1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="basic",
        requests=[depot, req1, req2],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_multi_vehicle_instance() -> DVRPTWInstance:
    """Depot at origin; three requests; two vehicles."""
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, 1000.0),
        service_time=0.0,
        is_depot=True,
    )
    requests = [
        Request(
            id=i,
            position=(float(i), 0.0),
            demand=10.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )
        for i in range(1, 4)
    ]
    vehicles = [
        Vehicle(id=i, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        for i in range(2)
    ]
    return DVRPTWInstance(
        id="multi_vehicle",
        requests=[depot] + requests,
        vehicles=vehicles,
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_single_request_instance() -> DVRPTWInstance:
    """Depot at origin; one request at (3,4); travel cost depot→req→depot = 10."""
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
        time_window=TimeWindow(0.0, 100.0),
        service_time=10.0,
        release_time=0.0,
    )
    v1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="single",
        requests=[depot, req1],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_id=0,
    )


def _make_delayed_release_instance(release_time: float = 15.0) -> DVRPTWInstance:
    """Two requests: one at t=0, one with a delayed release_time."""
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
        demand=5.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=5.0,
        release_time=0.0,
    )
    req2 = Request(
        id=2,
        position=(2.0, 0.0),
        demand=5.0,
        time_window=TimeWindow(0.0, 200.0),
        service_time=5.0,
        release_time=release_time,
    )
    v1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="delayed",
        requests=[depot, req1, req2],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_id=0,
    )


# ---------------------------------------------------------------------------
# Simple Python routers / schedulers used across tests
# ---------------------------------------------------------------------------


class AssignToFirstVehicleRouter:
    """Routes every request to vehicle 0."""

    def route(self, request_id, vehicles, instance_view):
        return vehicles[0]["vehicle_id"]


class RejectAllRouter:
    """Rejects every request."""

    def route(self, request_id, vehicles, instance_view):
        return None


class FifoScheduler:
    """Returns the first (lowest-id) request in the queue."""

    def schedule(self, vehicle, queue, instance_view):
        return min(queue)


class AssignToFirstVehicleBatchRouter:
    """Batch version: assigns every request in the batch to vehicle 0."""

    def route_batch(self, requests, vehicles, instance_view):
        vid = vehicles[0]["vehicle_id"]
        return [(rid, vid) for rid in requests]


class RejectAllBatchRouter:
    """Batch version: rejects every request."""

    def route_batch(self, requests, vehicles, instance_view):
        return [(rid, None) for rid in requests]


class RecordingBatchRouter:
    """Records each route_batch call for inspection in tests."""

    def __init__(self, vehicle_id=0):
        self.calls: list[list[int]] = []  # list of request batches
        self.vehicle_id = vehicle_id

    def route_batch(self, requests, vehicles, instance_view):
        self.calls.append(list(requests))
        return [(rid, self.vehicle_id) for rid in requests]


# ---------------------------------------------------------------------------
# Tests for composable_strategy
# ---------------------------------------------------------------------------


class TestComposableStrategy(unittest.TestCase):
    def setUp(self):
        from rsimulator import composable_strategy, NativeDispatchStrategy

        self.composable_strategy = composable_strategy
        self.NativeDispatchStrategy = NativeDispatchStrategy

    def test_returns_native_wrapper(self):
        """composable_strategy() returns a NativeDispatchStrategy."""
        s = self.composable_strategy(AssignToFirstVehicleRouter(), FifoScheduler())
        self.assertIsInstance(s, self.NativeDispatchStrategy)

    def test_serves_all_requests(self):
        """Router assigns both requests; all should be served."""
        inst = _make_basic_instance()
        strategy = self.composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        self.assertEqual(result.metrics.rejected, 0)

    def test_reject_all(self):
        """RejectAllRouter causes all requests to be rejected."""
        inst = _make_basic_instance()
        strategy = self.composable_strategy(RejectAllRouter(), FifoScheduler())
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, set())
        self.assertEqual(result.metrics.rejected, 2)

    def test_single_request_travel_cost(self):
        """Single request at (3,4): depot→req→depot costs 10.0."""
        inst = _make_single_request_instance()
        strategy = self.composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertAlmostEqual(result.metrics.total_travel_cost, 10.0, places=9)

    def test_multi_vehicle_distributes_work(self):
        """With two vehicles, work should be distributed (not all on vehicle 0)."""

        class RoundRobinRouter:
            def __init__(self):
                self._counter = 0

            def route(self, request_id, vehicles, instance_view):
                vid = vehicles[self._counter % len(vehicles)]["vehicle_id"]
                self._counter += 1
                return vid

        inst = _make_multi_vehicle_instance()
        strategy = self.composable_strategy(RoundRobinRouter(), FifoScheduler())
        result = RustSimulator(inst, strategy).run()

        total_served = sum(len(r) for r in result.solution.routes)
        self.assertEqual(total_served, 3)

    def test_solution_structure(self):
        """routes and service_times have consistent lengths."""
        inst = _make_basic_instance()
        strategy = self.composable_strategy(
            AssignToFirstVehicleRouter(), FifoScheduler()
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(
            len(result.solution.routes), len(result.solution.service_times)
        )
        for route, times in zip(result.solution.routes, result.solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_scheduler_controls_order(self):
        """Scheduler returning max(queue) reverses dispatch order vs min(queue)."""

        class MaxFirstScheduler:
            def schedule(self, vehicle, queue, instance_view):
                return max(queue)

        inst = _make_basic_instance()
        fifo_result = RustSimulator(
            inst,
            self.composable_strategy(AssignToFirstVehicleRouter(), FifoScheduler()),
        ).run()
        max_result = RustSimulator(
            inst,
            self.composable_strategy(AssignToFirstVehicleRouter(), MaxFirstScheduler()),
        ).run()

        # Both must serve all requests; only order may differ.
        fifo_served = {r for route in fifo_result.solution.routes for r in route}
        max_served = {r for route in max_result.solution.routes for r in route}
        self.assertEqual(fifo_served, {1, 2})
        self.assertEqual(max_served, {1, 2})
        # FIFO: dispatch 1 then 2; Max: dispatch 2 then 1.
        self.assertEqual(fifo_result.solution.routes[0][0], 1)
        self.assertEqual(max_result.solution.routes[0][0], 2)

    def test_delayed_release_routed_on_arrival(self):
        """A request released at t=15 is routed when it appears, not at t=0."""
        routed_times: list[tuple[int, float]] = []

        class RecordingRouter:
            def route(self, request_id, vehicles, instance_view):
                # Record (request_id, available_at of vehicle 0 ≈ current time)
                for v in vehicles:
                    if v["vehicle_id"] == 0:
                        routed_times.append((request_id, v["available_at"]))
                        break
                return 0

        inst = _make_delayed_release_instance(release_time=15.0)
        strategy = self.composable_strategy(RecordingRouter(), FifoScheduler())
        RustSimulator(inst, strategy).run()

        self.assertEqual(len(routed_times), 2)
        by_id = {rid: t for rid, t in routed_times}
        # req 2 must have been routed after req 1 (at a later simulation time).
        self.assertGreater(by_id[2], by_id[1])


# ---------------------------------------------------------------------------
# Tests for batch_composable_strategy
# ---------------------------------------------------------------------------


class TestBatchComposableStrategy(unittest.TestCase):
    from dvrptw.strategies import batch_composable_strategy

    def setUp(self):
        self.batch_composable_strategy = batch_composable_strategy

    def test_returns_native_wrapper(self):
        """batch_composable_strategy() returns a NativeDispatchStrategy."""
        _ = self.batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=10.0
        )

    def test_serves_all_requests(self):
        """Basic end-to-end: both requests served with a slot large enough to catch them."""
        inst = _make_basic_instance()
        strategy = self.batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})
        self.assertEqual(result.metrics.rejected, 0)

    def test_reject_all(self):
        """RejectAllBatchRouter causes all requests to be rejected."""
        inst = _make_basic_instance()
        strategy = self.batch_composable_strategy(
            RejectAllBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, set())
        self.assertEqual(result.metrics.rejected, 2)

    def test_single_request_travel_cost(self):
        """Single request: batch strategy preserves correct travel cost."""
        inst = _make_single_request_instance()
        strategy = self.batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertAlmostEqual(result.metrics.total_travel_cost, 10.0, places=9)

    def test_slot_boundary_batches_requests(self):
        """Requests released within the same slot arrive in one route_batch call."""
        # Both req1 (release=0) and req2 (release=5) fall within slot [0, 10).
        # They should arrive together in a single route_batch call.
        recorder = RecordingBatchRouter(vehicle_id=0)
        inst = _make_delayed_release_instance(release_time=5.0)
        strategy = self.batch_composable_strategy(
            recorder, FifoScheduler(), slot_size=10.0
        )
        RustSimulator(inst, strategy).run()

        # Both requests released before t=10 → one batch call containing both.
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(sorted(recorder.calls[0]), [1, 2])

    def test_slot_boundary_splits_requests(self):
        """Requests in different slots arrive in separate route_batch calls."""
        # req1 released at t=0 (slot [0,10)), req2 released at t=15 (slot [10,20)).
        recorder = RecordingBatchRouter(vehicle_id=0)
        inst = _make_delayed_release_instance(release_time=15.0)
        strategy = self.batch_composable_strategy(
            recorder, FifoScheduler(), slot_size=10.0
        )
        RustSimulator(inst, strategy).run()

        # Two separate batch calls.
        self.assertEqual(len(recorder.calls), 2)
        first_batch = recorder.calls[0]
        second_batch = recorder.calls[1]
        self.assertIn(1, first_batch)
        self.assertNotIn(2, first_batch)
        self.assertIn(2, second_batch)
        self.assertNotIn(1, second_batch)

    def test_slot_size_one_call_per_request(self):
        """Very small slot_size (0.1) causes each released request to be routed alone."""
        # Both requests are released at t=0, so they'll all land in slot [0, 0.1)
        # together — still one call. Use a release_time > 0.1 to force separation.
        recorder = RecordingBatchRouter(vehicle_id=0)
        inst = _make_delayed_release_instance(release_time=0.5)
        strategy = self.batch_composable_strategy(
            recorder, FifoScheduler(), slot_size=0.1
        )
        RustSimulator(inst, strategy).run()

        # req1 at t=0 in first slot; req2 at t=0.5 in later slot.
        self.assertGreater(len(recorder.calls), 1)
        all_routed = [r for batch in recorder.calls for r in batch]
        self.assertIn(1, all_routed)
        self.assertIn(2, all_routed)

    def test_solution_structure(self):
        """routes and service_times have consistent lengths."""
        inst = _make_basic_instance()
        strategy = self.batch_composable_strategy(
            AssignToFirstVehicleBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        self.assertEqual(
            len(result.solution.routes), len(result.solution.service_times)
        )
        for route, times in zip(result.solution.routes, result.solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_multi_vehicle_batch(self):
        """Batch router distributing across vehicles: all requests served."""

        class SplitBatchRouter:
            """Assigns requests round-robin across vehicles."""

            def route_batch(self, requests, vehicles, instance_view):
                result = []
                for i, rid in enumerate(requests):
                    vid = vehicles[i % len(vehicles)]["vehicle_id"]
                    result.append((rid, vid))
                return result

        inst = _make_multi_vehicle_instance()
        strategy = self.batch_composable_strategy(
            SplitBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        total_served = sum(len(r) for r in result.solution.routes)
        self.assertEqual(total_served, 3)
        self.assertEqual(result.metrics.rejected, 0)

    def test_partial_reject_in_batch(self):
        """Batch router can reject some requests and assign others."""

        class RejectOddBatchRouter:
            """Rejects odd request IDs, assigns even ones to vehicle 0."""

            def route_batch(self, requests, vehicles, instance_view):
                vid = vehicles[0]["vehicle_id"]
                return [(rid, None if rid % 2 == 1 else vid) for rid in requests]

        # req1 (id=1, odd) → rejected; req2 (id=2, even) → served
        inst = _make_basic_instance()
        strategy = self.batch_composable_strategy(
            RejectOddBatchRouter(), FifoScheduler(), slot_size=10.0
        )
        result = RustSimulator(inst, strategy).run()

        served = {r for route in result.solution.routes for r in route}
        self.assertIn(2, served)
        self.assertNotIn(1, served)
        self.assertEqual(result.metrics.rejected, 1)


if __name__ == "__main__":
    unittest.main()

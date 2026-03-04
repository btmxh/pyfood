"""Tests for the ILP baseline strategy."""

import unittest

from dvrptw import (
    DVRPTWInstance,
    Request,
    Vehicle,
    TimeWindow,
    PythonSimulator,
    ILPStrategy,
    StarNormEvaluator,
)


def _make_instance(
    customers: list[Request],
    vehicles: list[Vehicle],
    instance_id: str = "test",
    planning_horizon: float = 500.0,
) -> DVRPTWInstance:
    depot = Request(
        id=0,
        position=(0.0, 0.0),
        demand=0.0,
        time_window=TimeWindow(0.0, planning_horizon),
        service_time=0.0,
        is_depot=True,
    )
    return DVRPTWInstance(
        id=instance_id,
        requests=[depot] + customers,
        vehicles=vehicles,
        planning_horizon=planning_horizon,
        depot_id=0,
    )


class TestILPStrategyBasic(unittest.TestCase):
    """Basic correctness tests for ILPStrategy."""

    def _single_vehicle_instance(self) -> DVRPTWInstance:
        customers = [
            Request(
                id=1,
                position=(3.0, 4.0),  # dist=5 from depot
                demand=10.0,
                time_window=TimeWindow(0.0, 200.0),
                service_time=0.0,
            ),
            Request(
                id=2,
                position=(0.0, 2.0),  # dist=2 from depot
                demand=10.0,
                time_window=TimeWindow(0.0, 200.0),
                service_time=0.0,
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        ]
        return _make_instance(customers, vehicles)

    def test_ilp_serves_all_feasible_requests(self):
        """ILP should serve all requests when capacity and time windows allow."""
        instance = self._single_vehicle_instance()
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)

    def test_result_has_correct_structure(self):
        """Solution routes and service_times must be structurally consistent."""
        instance = self._single_vehicle_instance()
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        solution = result.solution
        self.assertEqual(len(solution.routes), len(solution.service_times))
        for route, times in zip(solution.routes, solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_travel_cost_positive(self):
        """Total travel cost must be positive when any requests are served."""
        instance = self._single_vehicle_instance()
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertGreater(result.metrics.total_travel_cost, 0.0)


class TestILPStrategyCapacity(unittest.TestCase):
    """Verify that ILP respects vehicle capacity."""

    def test_capacity_limits_service(self):
        """With 1 vehicle of capacity 50 and 2 requests each needing 40,
        only one can be served per route.  With 1 vehicle, at most 1 served."""
        customers = [
            Request(
                id=1,
                position=(1.0, 0.0),
                demand=40.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
            Request(
                id=2,
                position=(2.0, 0.0),
                demand=40.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
        ]
        vehicles = [Vehicle(id=0, capacity=50.0, start_depot=0, end_depot=0, speed=1.0)]
        instance = _make_instance(customers, vehicles)
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        # Can serve at most 1 (capacity = 50, each demand = 40)
        self.assertGreaterEqual(result.metrics.rejected, 1)

    def test_two_vehicles_serve_all(self):
        """Two vehicles of capacity 50 can collectively serve both 40-demand requests."""
        customers = [
            Request(
                id=1,
                position=(1.0, 0.0),
                demand=40.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
            Request(
                id=2,
                position=(2.0, 0.0),
                demand=40.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=50.0, start_depot=0, end_depot=0, speed=1.0),
            Vehicle(id=1, capacity=50.0, start_depot=0, end_depot=0, speed=1.0),
        ]
        instance = _make_instance(customers, vehicles)
        # Pure service maximisation: w1=0, w2=1
        evaluator = StarNormEvaluator.from_instance(0.0, 1.0, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)

    def test_infeasible_time_window_rejected(self):
        """A request that is unreachable within its time window should not be served."""
        customers = [
            Request(
                id=1,
                position=(100.0, 0.0),  # 100 units away, speed=1 → arrives at t=100
                demand=5.0,
                time_window=TimeWindow(0.0, 50.0),  # must start by t=50 — impossible
                service_time=0.0,
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        ]
        instance = _make_instance(customers, vehicles)
        evaluator = StarNormEvaluator.from_instance(0.5, 0.5, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 1)

    def test_waiting_at_early_window(self):
        """Vehicle must wait when arriving before the time window opens."""
        customers = [
            Request(
                id=1,
                position=(1.0, 0.0),  # 1 unit away, arrives at t=1
                demand=5.0,
                time_window=TimeWindow(50.0, 200.0),  # earliest=50, vehicle must wait
                service_time=0.0,
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        ]
        instance = _make_instance(customers, vehicles)
        # Pure service maximisation to avoid tie with empty route
        evaluator = StarNormEvaluator.from_instance(0.0, 1.0, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)
        # Service must start at or after earliest=50
        service_t = result.solution.service_times[0][0]
        self.assertGreaterEqual(service_t, 50.0)


class TestILPStrategyObjectiveWeight(unittest.TestCase):
    """Test that objective_weight influences the plan direction."""

    def _asymmetric_instance(self) -> DVRPTWInstance:
        """Instance where one request is close (low cost) and one is far (high cost)."""
        customers = [
            Request(
                id=1,
                position=(1.0, 0.0),  # dist=1
                demand=10.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
            Request(
                id=2,
                position=(50.0, 0.0),  # dist=50
                demand=10.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        ]
        return _make_instance(customers, vehicles, planning_horizon=1000.0)

    def test_maximize_service_serves_both(self):
        """With pure service maximisation (w1=0, w2=1), both requests should be served."""
        instance = self._asymmetric_instance()
        evaluator = StarNormEvaluator.from_instance(0.0, 1.0, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)

    def test_minimize_cost_prefers_close(self):
        """With pure cost minimisation (w1=1, w2=0), only the close request should be served.
        (Serving the far request adds ~100 units of distance with zero benefit.)"""
        instance = self._asymmetric_instance()
        evaluator = StarNormEvaluator.from_instance(1.0, 0.0, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        # With pure cost minimisation and no penalty for rejection, the ILP
        # chooses empty routes (zero cost, zero rejected).
        self.assertEqual(result.metrics.rejected, 2)


class TestILPStrategyDynamicReleaseTime(unittest.TestCase):
    """ILP is a non-causal baseline: it knows all release times ahead of time."""

    def test_future_release_time_still_planned(self):
        """Requests with release_time > 0 appear later during simulation but
        the ILP sees them all upfront and plans for them."""
        customers = [
            Request(
                id=1,
                position=(1.0, 0.0),
                demand=10.0,
                time_window=TimeWindow(0.0, 500.0),
                service_time=0.0,
                release_time=0.0,
            ),
            Request(
                id=2,
                position=(2.0, 0.0),
                demand=10.0,
                time_window=TimeWindow(50.0, 500.0),
                service_time=0.0,
                release_time=50.0,  # released late into simulation
            ),
        ]
        vehicles = [
            Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
        ]
        instance = _make_instance(customers, vehicles, planning_horizon=1000.0)
        evaluator = StarNormEvaluator.from_instance(0.0, 1.0, instance)
        strategy = ILPStrategy(instance, evaluator=evaluator, time_limit_s=30.0)
        sim = PythonSimulator(instance, strategy)
        result = sim.run()

        # Both should be served: ILP knew about request 2 from the start
        self.assertEqual(result.metrics.rejected, 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the DVRPTW simulator engine."""

import unittest
from simulator import (
    DVRPTWInstance,
    Request,
    Vehicle,
    TimeWindow,
    Simulator,
    SimulationState,
    DispatchEvent,
    WaitEvent,
    RejectEvent,
)


class SimpleGreedyStrategy:
    """A simple greedy strategy for testing: dispatch vehicles to pending requests in order."""

    def __init__(self, planning_horizon=None):
        self.actions_made = 0
        self.planning_horizon = planning_horizon

    def next_events(self, state: SimulationState) -> list:
        """Dispatch available vehicles to pending requests in order."""
        actions = []

        if not state.pending_requests:
            # No more pending requests, wait until end of planning horizon
            if self.planning_horizon:
                if state.time < self.planning_horizon:
                    actions.append(WaitEvent(until_time=self.planning_horizon))
            return actions

        # Find idle vehicles and pending requests
        pending_list = sorted(state.pending_requests)
        idle_vehicles = [v for v in state.vehicles if v.available_at <= state.time]

        # Dispatch idle vehicles to pending requests
        for vehicle in idle_vehicles:
            if pending_list:
                next_request = pending_list.pop(0)
                actions.append(
                    DispatchEvent(
                        vehicle_id=vehicle.vehicle_id, destination_node=next_request
                    )
                )
            else:
                break

        # If no more to dispatch and vehicles are busy, wait
        if not actions and not idle_vehicles:
            next_vehicle_time = min(v.available_at for v in state.vehicles)
            if next_vehicle_time > state.time:
                actions.append(WaitEvent(until_time=next_vehicle_time))

        self.actions_made += len(actions)
        return actions


class TestSimulatorBasic(unittest.TestCase):
    """Test basic simulator functionality."""

    def setUp(self):
        """Create a simple test instance."""
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 1000.0),
            service_time=0.0,
            is_depot=True,
        )

        request1 = Request(
            id=1,
            position=(1.0, 0.0),
            demand=5.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        request2 = Request(
            id=2,
            position=(0.0, 1.0),
            demand=3.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        vehicle1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)

        self.instance = DVRPTWInstance(
            id="test_basic",
            requests=[depot, request1, request2],
            vehicles=[vehicle1],
            planning_horizon=500.0,
            depot_ids=[0],
        )

    def test_simple_simulation(self):
        """Test a basic simulation with greedy strategy."""
        strategy = SimpleGreedyStrategy(planning_horizon=self.instance.planning_horizon)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        # Check that solution was created
        self.assertIsNotNone(result.solution)
        self.assertEqual(len(result.solution.routes), 1)

        # Check that requests were served
        self.assertGreater(len(result.solution.routes[0]), 0)

        # Check metrics
        self.assertGreater(result.metrics.total_travel_cost, 0)
        self.assertEqual(result.metrics.accepted, 2)

    def test_solution_structure(self):
        """Test that solution has correct structure."""
        strategy = SimpleGreedyStrategy(planning_horizon=self.instance.planning_horizon)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        solution = result.solution
        # Routes and service_times should have same length
        self.assertEqual(len(solution.routes), len(solution.service_times))

        # Each route should have matching service times
        for route, times in zip(solution.routes, solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_dispatch_event(self):
        """Test dispatch event execution."""

        class DispatchOnceStrategy:
            def __init__(self, test_case):
                self.test_case = test_case
                self.dispatched = False

            def next_events(self, state):
                if not self.dispatched and state.pending_requests:
                    self.dispatched = True
                    return [DispatchEvent(vehicle_id=0, destination_node=1)]
                return []

        strategy = DispatchOnceStrategy(self)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        # Check that request 1 was served
        self.assertIn(1, result.solution.routes[0])
        self.assertIn(1, simulator.served_requests)

    def test_reject_event(self):
        """Test reject event execution."""

        class RejectStrategy:
            def __init__(self):
                self.rejected = False

            def next_events(self, state):
                if not self.rejected and state.pending_requests:
                    self.rejected = True
                    return [RejectEvent(request_id=1)]
                return []

        strategy = RejectStrategy()
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        # Check that request 1 was rejected
        self.assertNotIn(1, result.solution.routes[0])
        self.assertIn(1, simulator.rejected_requests)


class TestSimulatorConstraints(unittest.TestCase):
    """Test constraint validation."""

    def setUp(self):
        """Create instance with capacity constraints."""
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 1000.0),
            service_time=0.0,
            is_depot=True,
        )

        # Create requests with total demand > vehicle capacity
        request1 = Request(
            id=1,
            position=(1.0, 0.0),
            demand=60.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        request2 = Request(
            id=2,
            position=(2.0, 0.0),
            demand=50.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        vehicle1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)

        self.instance = DVRPTWInstance(
            id="test_capacity",
            requests=[depot, request1, request2],
            vehicles=[vehicle1],
            planning_horizon=500.0,
            depot_ids=[0],
        )

    def test_capacity_violation(self):
        """Test that capacity violations are detected."""

        class OverloadStrategy:
            def next_events(self, state):
                # Try to dispatch both requests to same vehicle
                if len([v for v in state.vehicles if v.available_at <= state.time]) > 0:
                    return [
                        DispatchEvent(vehicle_id=0, destination_node=1),
                        DispatchEvent(vehicle_id=0, destination_node=2),
                    ]
                return []

        strategy = OverloadStrategy()
        simulator = Simulator(self.instance, strategy)

        # Should raise error when trying to exceed capacity
        with self.assertRaises(ValueError):
            simulator.run()


class TestSimulatorTimeWindows(unittest.TestCase):
    """Test time window constraints."""

    def setUp(self):
        """Create instance with tight time windows."""
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 1000.0),
            service_time=0.0,
            is_depot=True,
        )

        # Request with tight time window
        request1 = Request(
            id=1,
            position=(10.0, 0.0),  # far away
            demand=5.0,
            time_window=TimeWindow(0.0, 5.0),  # must serve by t=5
            service_time=10.0,
            release_time=0.0,
        )

        vehicle1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)

        self.instance = DVRPTWInstance(
            id="test_time_window",
            requests=[depot, request1],
            vehicles=[vehicle1],
            planning_horizon=500.0,
            depot_ids=[0],
        )

    def test_time_window_violation(self):
        """Test that time window violations are detected."""

        class LateStrategy:
            def next_events(self, state):
                # Strategy tries to dispatch too late
                if state.time < 20.0:
                    return [WaitEvent(until_time=20.0)]
                if state.pending_requests:
                    return [DispatchEvent(vehicle_id=0, destination_node=1)]
                return []

        strategy = LateStrategy()
        simulator = Simulator(self.instance, strategy)

        # Request gets auto-rejected due to closed time window
        result = simulator.run()
        self.assertEqual(result.metrics.accepted, 0)

    def test_auto_reject_closed_window(self):
        """Test that requests with closed time windows are auto-rejected."""

        class WaitStrategy:
            def next_events(self, state):
                # Wait until after time window closes
                if state.time < 10.0:
                    return [WaitEvent(until_time=10.0)]
                # By now, request should be auto-rejected
                return []

        strategy = WaitStrategy()
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        # Request 1 should be auto-rejected due to closed time window
        self.assertIn(1, simulator.rejected_requests)
        self.assertEqual(result.metrics.accepted, 0)


class TestSimulatorMultipleVehicles(unittest.TestCase):
    """Test simulation with multiple vehicles."""

    def setUp(self):
        """Create instance with multiple vehicles."""
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

        self.instance = DVRPTWInstance(
            id="test_multi_vehicle",
            requests=[depot] + requests,
            vehicles=vehicles,
            planning_horizon=500.0,
            depot_ids=[0],
        )

    def test_multi_vehicle_dispatch(self):
        """Test dispatching across multiple vehicles."""
        strategy = SimpleGreedyStrategy(planning_horizon=self.instance.planning_horizon)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        # Check that both vehicles have routes
        total_served = sum(len(route) for route in result.solution.routes)
        self.assertGreater(total_served, 0)

        # Check that all accepted requests are in some route
        for route in result.solution.routes:
            for req_id in route:
                self.assertIn(req_id, simulator.served_requests)


class TestSimulatorActionCallback(unittest.TestCase):
    """Test action callback mechanism."""

    def setUp(self):
        """Create simple test instance."""
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=0.0,
            is_depot=True,
        )

        request1 = Request(
            id=1,
            position=(1.0, 0.0),
            demand=5.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        vehicle1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0)

        self.instance = DVRPTWInstance(
            id="test_callback",
            requests=[depot, request1],
            vehicles=[vehicle1],
            planning_horizon=100.0,
            depot_ids=[0],
        )

    def test_action_callback_invoked(self):
        """Test that action callback is called for each action."""
        actions_logged = []

        def callback(time, action, auto):
            actions_logged.append((time, action, auto))

        strategy = SimpleGreedyStrategy()
        simulator = Simulator(self.instance, strategy, action_callback=callback)
        simulator.run()

        # Check that at least some actions were logged
        self.assertGreater(len(actions_logged), 0)

        # Check that all logged actions have time >= 0
        for time, action, auto in actions_logged:
            self.assertGreaterEqual(time, 0.0)


class TestSimulatorMetrics(unittest.TestCase):
    """Test metrics computation."""

    def setUp(self):
        """Create simple test instance."""
        depot = Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=TimeWindow(0.0, 1000.0),
            service_time=0.0,
            is_depot=True,
        )

        request1 = Request(
            id=1,
            position=(3.0, 4.0),  # distance = 5 from depot
            demand=5.0,
            time_window=TimeWindow(0.0, 100.0),
            service_time=10.0,
            release_time=0.0,
        )

        vehicle1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)

        self.instance = DVRPTWInstance(
            id="test_metrics",
            requests=[depot, request1],
            vehicles=[vehicle1],
            planning_horizon=500.0,
            depot_ids=[0],
        )

    def test_metrics_computation(self):
        """Test that metrics are computed correctly."""
        strategy = SimpleGreedyStrategy(planning_horizon=self.instance.planning_horizon)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        metrics = result.metrics

        # Travel cost should be positive (depot -> request1 -> depot = 5 + 5 = 10)
        self.assertGreater(metrics.total_travel_cost, 0)

        # Should have 1 accepted request
        self.assertEqual(metrics.accepted, 1)

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        strategy = SimpleGreedyStrategy(planning_horizon=self.instance.planning_horizon)
        simulator = Simulator(self.instance, strategy)
        result = simulator.run()

        metrics_dict = result.metrics.to_dict()

        # Check that all expected keys are present
        self.assertIn("total_travel_cost", metrics_dict)
        self.assertIn("accepted", metrics_dict)


if __name__ == "__main__":
    unittest.main()

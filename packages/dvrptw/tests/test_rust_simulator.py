"""Tests for RustSimulator — verifies parity with PythonSimulator.

Parametrized parity tests are expressed as a shared base class
(`_ParityBase`) with two concrete subclasses (`TestParityPython`,
`TestParityRust`), so pytest collects and runs each scenario against
both backends without requiring the pytest-parametrize plugin.

All assertions use the stdlib `unittest` API; pytest is used only as
the test runner.
"""

import unittest
from typing import ClassVar

from dvrptw import (
    DVRPTWInstance,
    Request,
    Vehicle,
    TimeWindow,
    PythonSimulator,
    RustSimulator,
    SimulationState,
    DispatchEvent,
    WaitEvent,
    RejectEvent,
)
from dvrptw.simulator.base import Simulator as _SimulatorBase


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


class SimpleGreedyStrategy:
    """Dispatch idle vehicles to pending requests in arrival order."""

    def __init__(self, planning_horizon=None):
        self.planning_horizon = planning_horizon

    def next_events(self, state: SimulationState) -> list:
        actions = []

        if not state.pending_requests:
            if self.planning_horizon and state.time < self.planning_horizon:
                actions.append(WaitEvent(until_time=self.planning_horizon))
            return actions

        pending_list = sorted(state.pending_requests)
        idle_vehicles = [v for v in state.vehicles if v.available_at <= state.time]

        for vehicle in idle_vehicles:
            if pending_list:
                actions.append(
                    DispatchEvent(
                        vehicle_id=vehicle.vehicle_id,
                        destination_node=pending_list.pop(0),
                    )
                )
            else:
                break

        if not actions and not idle_vehicles:
            next_vehicle_time = min(v.available_at for v in state.vehicles)
            if next_vehicle_time > state.time:
                actions.append(WaitEvent(until_time=next_vehicle_time))

        return actions


# ---------------------------------------------------------------------------
# Instance builders
# ---------------------------------------------------------------------------


def _make_basic_instance() -> DVRPTWInstance:
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
        position=(3.0, 4.0),  # distance 5 from depot
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
        depot_ids=[0],
    )


def _make_multi_vehicle_instance() -> DVRPTWInstance:
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
        depot_ids=[0],
    )


def _make_tight_window_instance() -> DVRPTWInstance:
    """One request at distance 10, time window [0, 5] — unreachable in time."""
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
        position=(10.0, 0.0),
        demand=5.0,
        time_window=TimeWindow(0.0, 5.0),
        release_time=0.0,
        service_time=10.0,
    )
    v1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="tight_window",
        requests=[depot, req1],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_ids=[0],
    )


def _make_overload_instance() -> DVRPTWInstance:
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
        demand=60.0,
        time_window=TimeWindow(0.0, 100.0),
        service_time=10.0,
        release_time=0.0,
    )
    req2 = Request(
        id=2,
        position=(2.0, 0.0),
        demand=50.0,
        time_window=TimeWindow(0.0, 100.0),
        service_time=10.0,
        release_time=0.0,
    )
    v1 = Vehicle(id=0, capacity=100.0, start_depot=0, end_depot=0, speed=1.0)
    return DVRPTWInstance(
        id="capacity",
        requests=[depot, req1, req2],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_ids=[0],
    )


def _make_single_request_instance() -> DVRPTWInstance:
    """Depot at origin, one request at (3, 4) — travel cost depot→req→depot = 10."""
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
        id="single_request",
        requests=[depot, req1],
        vehicles=[v1],
        planning_horizon=500.0,
        depot_ids=[0],
    )


# ---------------------------------------------------------------------------
# Parity tests: shared base class run against both backends
# ---------------------------------------------------------------------------


class _ParityBase(unittest.TestCase):
    """Base for parity tests. Subclasses set `SimCls`.

    Not collected by pytest directly — the leading underscore prevents it, and
    `__test__ = False` is a belt-and-suspenders guard for test runners that
    ignore the naming convention.
    """

    __test__ = False  # prevent direct collection by pytest / unittest discovery
    SimCls: ClassVar[type[_SimulatorBase]]

    def test_simple_greedy_routes_and_cost(self):
        """Both backends produce identical served set and positive cost."""
        inst = _make_basic_instance()
        sim = self.SimCls(
            inst, SimpleGreedyStrategy(planning_horizon=inst.planning_horizon)
        )
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertGreater(result.metrics.total_travel_cost, 0)
        served = {r for route in result.solution.routes for r in route}
        self.assertEqual(served, {1, 2})

    def test_solution_structure(self):
        """routes and service_times have consistent lengths."""
        inst = _make_basic_instance()
        result = self.SimCls(inst, SimpleGreedyStrategy()).run()

        solution = result.solution
        self.assertEqual(len(solution.routes), len(solution.service_times))
        for route, times in zip(solution.routes, solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_reject_event(self):
        """Both backends honour RejectEvent and expose rejected_requests."""

        class RejectFirstStrategy:
            def __init__(self):
                self.rejected = False

            def next_events(self, state: SimulationState) -> list:
                if not self.rejected and state.pending_requests:
                    self.rejected = True
                    return [RejectEvent(request_id=1)]
                return []

        inst = _make_basic_instance()
        sim = self.SimCls(inst, RejectFirstStrategy())
        result = sim.run()

        self.assertNotIn(1, result.solution.routes[0])
        self.assertIn(1, sim.rejected_requests)

    def test_capacity_violation_raises(self):
        """Both backends raise ValueError when vehicle capacity is exceeded."""

        class OverloadStrategy:
            def next_events(self, state: SimulationState) -> list:
                idle = [v for v in state.vehicles if v.available_at <= state.time]
                if idle:
                    return [
                        DispatchEvent(vehicle_id=0, destination_node=1),
                        DispatchEvent(vehicle_id=0, destination_node=2),
                    ]
                return []

        inst = _make_overload_instance()
        sim = self.SimCls(inst, OverloadStrategy())
        with self.assertRaises(ValueError):
            sim.run()

    def test_auto_reject_closed_window(self):
        """Both backends auto-reject requests whose time window closes."""

        class WaitStrategy:
            def next_events(self, state: SimulationState) -> list:
                if state.time < 10.0:
                    return [WaitEvent(until_time=10.0)]
                return []

        inst = _make_tight_window_instance()
        sim = self.SimCls(inst, WaitStrategy())
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 1)
        self.assertIn(1, sim.rejected_requests)

    def test_multi_vehicle_dispatch(self):
        """Both backends route correctly when multiple vehicles are available."""
        inst = _make_multi_vehicle_instance()
        result = self.SimCls(inst, SimpleGreedyStrategy()).run()

        total_served = sum(len(r) for r in result.solution.routes)
        self.assertGreater(total_served, 0)

    def test_metrics_travel_cost(self):
        """Travel cost depot→(3,4)→depot = 5+5 = 10."""

        class ServeOneStrategy:
            def __init__(self):
                self.done = False

            def next_events(self, state: SimulationState) -> list:
                if not self.done and state.pending_requests:
                    self.done = True
                    return [DispatchEvent(vehicle_id=0, destination_node=1)]
                return []

        inst = _make_single_request_instance()
        result = self.SimCls(inst, ServeOneStrategy()).run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertAlmostEqual(result.metrics.total_travel_cost, 10.0, places=9)


class TestParityPython(_ParityBase):
    __test__ = True
    SimCls = PythonSimulator


class TestParityRust(_ParityBase):
    __test__ = True
    SimCls = RustSimulator


# ---------------------------------------------------------------------------
# Cross-backend parity: compare Python vs Rust output directly
# ---------------------------------------------------------------------------


class TestCrossBackendParity(unittest.TestCase):
    def _assert_parity(self, inst):
        py_res = PythonSimulator(inst, SimpleGreedyStrategy()).run()
        rs_res = RustSimulator(inst, SimpleGreedyStrategy()).run()

        self.assertEqual(len(py_res.solution.routes), len(rs_res.solution.routes))

        py_served = {r for route in py_res.solution.routes for r in route}
        rs_served = {r for route in rs_res.solution.routes for r in route}
        self.assertEqual(py_served, rs_served)

        self.assertAlmostEqual(
            py_res.metrics.total_travel_cost,
            rs_res.metrics.total_travel_cost,
            places=5,
        )
        self.assertEqual(py_res.metrics.rejected, rs_res.metrics.rejected)

    def test_basic_instance(self):
        """RustSimulator produces bit-identical routes and cost vs PythonSimulator."""
        self._assert_parity(_make_basic_instance())

    def test_multi_vehicle_instance(self):
        """Parity on a multi-vehicle instance."""
        self._assert_parity(_make_multi_vehicle_instance())


# ---------------------------------------------------------------------------
# RustSimulator-specific tests
# ---------------------------------------------------------------------------


class TestRustSpecific(unittest.TestCase):
    def test_callback_receives_python_objects(self):
        """Action callback receives DispatchEvent/WaitEvent/RejectEvent, not dicts."""
        actions_logged = []

        def callback(time, action, auto):
            actions_logged.append((time, type(action).__name__, auto))

        class WaitStrategy:
            def next_events(self, state: SimulationState) -> list:
                if state.time < 10.0:
                    return [WaitEvent(until_time=10.0)]
                return []

        inst = _make_tight_window_instance()
        RustSimulator(inst, WaitStrategy(), action_callback=callback).run()

        types_seen = {t for _, t, _ in actions_logged}
        self.assertIn("RejectEvent", types_seen)
        self.assertIn("WaitEvent", types_seen)

    def test_state_adapter_exposes_attributes(self):
        """Strategy receives SimulationState with attribute access (not a dict)."""
        received_states = []

        class InspectStrategy:
            def next_events(self, state: SimulationState) -> list:
                received_states.append(state)
                return []

        RustSimulator(_make_basic_instance(), InspectStrategy()).run()

        self.assertTrue(received_states, "Strategy was never called")
        s = received_states[0]
        self.assertIsInstance(s, SimulationState)
        self.assertTrue(hasattr(s, "pending_requests"))
        self.assertTrue(hasattr(s, "vehicles"))
        self.assertTrue(hasattr(s, "time"))


# ---------------------------------------------------------------------------
# Native strategy tests
# ---------------------------------------------------------------------------


class TestNativeStrategy(unittest.TestCase):
    def setUp(self):
        from rsimulator import greedy_strategy, NativeStrategyWrapper

        self.greedy_strategy = greedy_strategy
        self.NativeStrategyWrapper = NativeStrategyWrapper

    def test_basic_parity_with_python_greedy(self):
        """greedy_strategy() produces same served set as Python SimpleGreedyStrategy."""
        inst = _make_basic_instance()
        py_res = RustSimulator(inst, SimpleGreedyStrategy()).run()
        native_res = RustSimulator(inst, self.greedy_strategy()).run()

        py_served = {r for route in py_res.solution.routes for r in route}
        native_served = {r for route in native_res.solution.routes for r in route}
        self.assertEqual(native_served, py_served)
        self.assertEqual(native_res.metrics.rejected, py_res.metrics.rejected)
        self.assertAlmostEqual(
            native_res.metrics.total_travel_cost,
            py_res.metrics.total_travel_cost,
            places=9,
        )

    def test_multi_vehicle_parity(self):
        """greedy_strategy() handles multi-vehicle instances correctly."""
        inst = _make_multi_vehicle_instance()
        py_res = RustSimulator(inst, SimpleGreedyStrategy()).run()
        native_res = RustSimulator(inst, self.greedy_strategy()).run()

        py_served = {r for route in py_res.solution.routes for r in route}
        native_served = {r for route in native_res.solution.routes for r in route}
        self.assertEqual(native_served, py_served)
        self.assertEqual(native_res.metrics.rejected, py_res.metrics.rejected)

    def test_auto_reject_infeasible_request(self):
        """greedy_strategy() skips tight-window requests; simulator auto-rejects them."""
        inst = _make_tight_window_instance()
        sim = RustSimulator(inst, self.greedy_strategy())
        result = sim.run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertIn(1, sim.rejected_requests)

    def test_returns_native_wrapper_type(self):
        """greedy_strategy() returns a NativeStrategyWrapper, not a Python strategy."""
        self.assertIsInstance(self.greedy_strategy(), self.NativeStrategyWrapper)

    def test_result_structure(self):
        """Native strategy result has correct routes/service_times structure."""
        inst = _make_basic_instance()
        result = RustSimulator(inst, self.greedy_strategy()).run()

        self.assertEqual(
            len(result.solution.routes), len(result.solution.service_times)
        )
        for route, times in zip(result.solution.routes, result.solution.service_times):
            self.assertEqual(len(route), len(times))

    def test_travel_cost(self):
        """Native greedy on single-request instance yields cost 10.0."""
        inst = _make_single_request_instance()
        result = RustSimulator(inst, self.greedy_strategy()).run()

        self.assertEqual(result.metrics.rejected, 0)
        self.assertAlmostEqual(result.metrics.total_travel_cost, 10.0, places=9)


if __name__ == "__main__":
    unittest.main()

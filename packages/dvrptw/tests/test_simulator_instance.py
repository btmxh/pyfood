import unittest
from pathlib import Path
import dvrptw.instance as inst_mod


class TestSimulatorInstance(unittest.TestCase):
    def test_load_vrpr_csv_basic(self):
        # prefer local copy in the package tests data to avoid depending on .temp_third_party
        csv = Path("packages/dvrptw/tests/data/h100rc101.csv")
        self.assertTrue(csv.exists(), f"test data missing: {csv}")
        inst = inst_mod.load_vrpr_csv(
            str(csv), truck_speed=1.0, truck_capacity=100.0, num_trucks=5
        )
        self.assertEqual(inst.id, csv.stem)
        # depot + 100 requests => 101
        self.assertEqual(len(inst.requests), 101)
        self.assertEqual(len(inst.vehicles), 5)
        depot = inst.requests[0]
        self.assertTrue(depot.is_depot)

    def test_to_json_roundtrip_and_pairwise_distance(self):
        # small instance
        depot = inst_mod.Request(
            id=0,
            position=(0.0, 0.0),
            demand=0.0,
            time_window=inst_mod.TimeWindow(0.0, 100.0),
            service_time=0.0,
            is_depot=True,
        )
        r1 = inst_mod.Request(
            id=1,
            position=(3.0, 4.0),
            demand=1.0,
            time_window=inst_mod.TimeWindow(10.0, 50.0),
            service_time=5.0,
        )
        v = inst_mod.Vehicle(id=0, capacity=10.0, start_depot=0, speed=1.0)
        inst = inst_mod.DVRPTWInstance(
            id="t1", requests=[depot, r1], vehicles=[v], depot_ids=[0]
        )
        j = inst.to_json()
        inst2 = inst_mod.DVRPTWInstance.from_json(j)
        self.assertEqual(inst2.id, "t1")
        self.assertEqual(len(inst2.requests), 2)
        # distance between (0,0) and (3,4) is 5.0
        mat = inst.pairwise_distance_matrix()
        self.assertEqual(mat[(0, 1)], 5.0)

    def test_validate_timewindow_and_missing_depot_errors(self):
        # invalid time window
        tw = inst_mod.TimeWindow(10.0, 5.0)
        with self.assertRaises(ValueError):
            tw.validate()

        depot = inst_mod.Request(
            id=0,
            position=(0, 0),
            demand=0.0,
            time_window=inst_mod.TimeWindow(0, 10),
            service_time=0.0,
            is_depot=True,
        )
        inst = inst_mod.DVRPTWInstance(
            id="bad", requests=[depot], vehicles=[], depot_ids=[]
        )
        with self.assertRaises(ValueError):
            inst.validate()


if __name__ == "__main__":
    unittest.main()

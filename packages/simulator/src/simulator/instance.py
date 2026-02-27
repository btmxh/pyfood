"""DVRPTW instance dataclasses and helpers.

Captured from docs/STATEMENT.md: depot, requests, vehicles, time windows,
dynamic release times, service times, and objective weight.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import math
import json

from pathlib import Path

Time = float
Coord = tuple[float, float]
ID = int


def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class TimeWindow:
    earliest: Time
    latest: Time

    def validate(self) -> None:
        if self.earliest > self.latest:
            raise ValueError(
                f"TimeWindow invalid: earliest ({self.earliest}) > latest ({self.latest})"
            )


@dataclass
class Request:
    id: ID
    position: Coord
    demand: float
    time_window: TimeWindow
    service_time: Time
    release_time: Time = 0.0
    is_depot: bool = False

    def distance_to(self, other: "Request") -> float:
        return euclidean(self.position, other.position)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["time_window"] = {
            "earliest": self.time_window.earliest,
            "latest": self.time_window.latest,
        }
        d["position"] = list(self.position)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Request":
        tw = TimeWindow(d["time_window"]["earliest"], d["time_window"]["latest"])
        return Request(
            id=d["id"],
            position=tuple(d["position"]),
            demand=d["demand"],
            time_window=tw,
            service_time=d["service_time"],
            release_time=d.get("release_time", 0.0),
            is_depot=d.get("is_depot", False),
        )


@dataclass
class Vehicle:
    id: ID
    capacity: float
    start_depot: ID
    end_depot: ID | None = None
    max_time: Time | None = None
    speed: float = 1.0

    def travel_time(self, distance: float) -> float:
        return distance / self.speed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Vehicle":
        return Vehicle(**d)


@dataclass
class DVRPTWInstance:
    id: str
    requests: list[Request] = field(default_factory=list)
    vehicles: list[Vehicle] = field(default_factory=list)
    weight_obj1: float = 0.5
    seed: int | None = None
    planning_horizon: Time | None = None
    depot_ids: list[ID] = field(default_factory=list)

    def get_request(self, req_id: ID) -> Request:
        for r in self.requests:
            if r.id == req_id:
                return r
        raise KeyError(f"Request id {req_id} not found")

    def validate(self) -> None:
        if not (0 <= self.weight_obj1 <= 1):
            raise ValueError("weight_obj1 must be in [0,1]")
        for r in self.requests:
            r.time_window.validate()
        if not self.depot_ids:
            raise ValueError("At least one depot id must be provided in depot_ids")
        depot_set = {r.id for r in self.requests if r.is_depot}
        for di in self.depot_ids:
            if di not in depot_set:
                raise ValueError(
                    f"depot id {di} not present or not marked as depot in requests"
                )

    def pairwise_distance_matrix(self) -> dict[tuple[ID, ID], float]:
        matrix: dict[tuple[ID, ID], float] = {}
        for a in self.requests:
            for b in self.requests:
                matrix[(a.id, b.id)] = euclidean(a.position, b.position)
        return matrix

    def to_json(self) -> str:
        obj = {
            "id": self.id,
            "requests": [r.to_dict() for r in self.requests],
            "vehicles": [v.to_dict() for v in self.vehicles],
            "weight_obj1": self.weight_obj1,
            "seed": self.seed,
            "planning_horizon": self.planning_horizon,
            "depot_ids": self.depot_ids,
        }
        return json.dumps(obj, indent=2)

    @staticmethod
    def from_json(s: str) -> "DVRPTWInstance":
        o = json.loads(s)
        reqs = [Request.from_dict(rd) for rd in o["requests"]]
        vehs = [Vehicle.from_dict(vd) for vd in o["vehicles"]]
        inst = DVRPTWInstance(
            id=o.get("id", ""),
            requests=reqs,
            vehicles=vehs,
            weight_obj1=o.get("weight_obj1", 0.5),
            seed=o.get("seed"),
            planning_horizon=o.get("planning_horizon"),
            depot_ids=o.get("depot_ids", []),
        )
        return inst


def load_vrpr_csv(
    csv: str, truck_speed: float, truck_capacity: float, num_trucks: int
) -> DVRPTWInstance:
    """Load dataset files from the vrpr repository CSV format.

    This mirrors the loader in the third-party repo (src/sim/problem.rs):
    - skips the CSV header
    - columns: x,y,demand,open,close,...,time (time at index 7)
    - service_time is set to 10.0 for all requests
    - the first entry is considered the depot and removed from the requests list
    """
    path = Path(csv)
    requests: list[Request] = []
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv}")
    with path.open("r", encoding="utf-8") as fh:
        # skip header
        lines = fh.readlines()
    if len(lines) <= 1:
        raise ValueError("CSV appears empty or missing header")
    for idx, line in enumerate(lines[1:]):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # parse floats; be tolerant of extra/missing columns
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(0.0)
        # ensure we have enough columns
        # follow rust loader: x=0,y=1,demand=2,open=3,close=4,time=7
        x = vals[0] if len(vals) > 0 else 0.0
        y = vals[1] if len(vals) > 1 else 0.0
        demand = vals[2] if len(vals) > 2 else 0.0
        open_t = vals[3] if len(vals) > 3 else 0.0
        close_t = vals[4] if len(vals) > 4 else 0.0
        service_time = 10.0
        time_field = vals[7] if len(vals) > 7 else 0.0
        req = Request(
            id=idx,
            position=(x, y),
            demand=demand,
            time_window=TimeWindow(open_t, close_t),
            service_time=service_time,
            release_time=time_field,
            is_depot=False,
        )
        requests.append(req)

    if not requests:
        raise ValueError("No requests parsed from CSV")
    depot = requests.pop(0)
    depot.is_depot = True

    # create vehicles
    vehicles: list[Vehicle] = []
    for vid in range(num_trucks):
        v = Vehicle(
            id=vid,
            capacity=truck_capacity,
            start_depot=depot.id,
            end_depot=depot.id,
            speed=truck_speed,
        )
        vehicles.append(v)

    inst = DVRPTWInstance(
        id=path.stem,
        requests=[depot] + requests,
        vehicles=vehicles,
        weight_obj1=0.5,
        seed=None,
        planning_horizon=None,
        depot_ids=[depot.id],
    )
    return inst

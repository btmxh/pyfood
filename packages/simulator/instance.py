"""DVRPTW instance dataclasses and helpers.

Captured from docs/STATEMENT.md: depot, requests, vehicles, time windows,
dynamic release times, service times, and objective weight.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import math
import json

Time = float
Coord = Tuple[float, float]
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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["time_window"] = {
            "earliest": self.time_window.earliest,
            "latest": self.time_window.latest,
        }
        d["position"] = list(self.position)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Request":
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
    end_depot: Optional[ID] = None
    max_time: Optional[Time] = None
    speed: float = 1.0

    def travel_time(self, distance: float) -> float:
        return distance / self.speed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Vehicle":
        return Vehicle(**d)


@dataclass
class DVRPTWInstance:
    id: str
    requests: List[Request] = field(default_factory=list)
    vehicles: List[Vehicle] = field(default_factory=list)
    weight_obj1: float = 0.5
    seed: Optional[int] = None
    planning_horizon: Optional[Time] = None
    depot_ids: List[ID] = field(default_factory=list)

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

    def pairwise_distance_matrix(self) -> Dict[Tuple[ID, ID], float]:
        matrix: Dict[Tuple[ID, ID], float] = {}
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

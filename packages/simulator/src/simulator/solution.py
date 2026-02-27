"""Solution representation for DVRPTW."""

from dataclasses import dataclass


@dataclass
class Solution:
    """
    Represents a DVRPTW solution with routes and timing.

    routes[k][i]: the i-th node visited by vehicle k (excluding depot)
    service_times[k][i]: when service starts at that node

    Example:
        Solution(
            routes=[[1, 3, 5], [2, 4], []],
            service_times=[[10.0, 25.5, 40.0], [15.0, 30.0], []]
        )
        - Vehicle 0 visits nodes 1 → 3 → 5, starting service at times 10.0, 25.5, 40.0
        - Vehicle 1 visits nodes 2 → 4, starting service at times 15.0, 30.0
        - Vehicle 2 is unused
    """

    routes: list[list[int]]
    service_times: list[list[float]]

    def __post_init__(self):
        """Validate that routes and service_times have matching dimensions."""
        if len(self.routes) != len(self.service_times):
            raise ValueError(
                f"routes and service_times must have same length: "
                f"got {len(self.routes)} routes but {len(self.service_times)} service_times"
            )

        for k, (route, times) in enumerate(zip(self.routes, self.service_times)):
            if len(route) != len(times):
                raise ValueError(
                    f"Route {k} has {len(route)} nodes but {len(times)} service times"
                )

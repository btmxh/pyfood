from __future__ import annotations

from typing import Dict, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from dvrptw.instance import DVRPTWInstance


def solve_static_vrp(
    instance: DVRPTWInstance,
    num_vehicles: int,
    time_limit_s: float = 30.0,
    scale: int = 1000,
    reject_penalty: int | None = None,
) -> Tuple[str, float, float, Dict]:
    """Solve the static VRPTW instance using OR-Tools Routing solver.

    Returns (status, travel_cost, rejected, solution_dict).
    - travel_cost: sum of all edge costs traversed (obj1)
    - rejected: number of customers not served (obj2 — lower is better)
    - solution_dict: mapping with vehicle routes
    """
    reqs = instance.requests
    n = len(reqs)

    pair = instance.pairwise_distance_matrix()
    ids = [r.id for r in reqs]
    dist_mat = [
        [int(round(pair[(ids[i], ids[j])] * scale)) for j in range(n)] for i in range(n)
    ]

    # find depot index
    depot_idx = next(i for i, r in enumerate(reqs) if r.id == instance.depot_id)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_mat[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # capacity dimension
    demands = [int(round(r.demand)) for r in reqs]
    vehicle_capacities = [int(round(v.capacity)) for v in instance.vehicles]

    def demand_callback(index):
        node = manager.IndexToNode(index)
        return demands[node]

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index, 0, vehicle_capacities, True, "Capacity"
    )

    # allow optional visits via disjunctions with a penalty for rejection
    # penalty is in the same units as distance (scaled)
    if reject_penalty is None:
        # default: large penalty to discourage rejection but allow feasibility
        # choose based on max distance
        max_d = max(max(row) for row in dist_mat) if dist_mat else 0
        reject_penalty = int(max_d * 1000 + 1_000_000)
    for node_idx, r in enumerate(reqs):
        if r.is_depot:
            continue
        index = manager.NodeToIndex(node_idx)
        routing.AddDisjunction([index], reject_penalty)

    # time dimension (travel time = distance / speed). we scale consistently
    service = [int(round(r.service_time * scale)) for r in reqs]
    tw_earliest = [int(round(r.time_window.earliest * scale)) for r in reqs]
    tw_latest = [int(round(r.time_window.latest * scale)) for r in reqs]

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        speed = instance.vehicles[0].speed if instance.vehicles else 1.0
        travel = int(round(pair[(ids[from_node], ids[to_node])] * scale / speed))
        return travel + service[from_node]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    horizon = int(max(tw_latest) + max(service) + 1000)
    routing.AddDimension(time_cb_idx, 1000000, horizon, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    # set time window constraints per node
    for node_idx, r in enumerate(reqs):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetRange(
            tw_earliest[node_idx], tw_latest[node_idx]
        )

    # search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = {}
        total_dist = 0
        visited_ids: set[int] = set()
        for k in range(num_vehicles):
            index = routing.Start(k)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if not reqs[node].is_depot:
                    route.append(reqs[node].id)
                    visited_ids.add(reqs[node].id)
                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)
                total_dist += dist_mat[node][next_node]
                index = next_index
            if route:
                routes[k] = route

        all_customer_ids = {r.id for r in reqs if not r.is_depot}
        rejected = len(all_customer_ids - visited_ids)

        return "FEASIBLE", total_dist / scale, rejected, {"routes": routes}
    else:
        return "UNKNOWN", float("inf"), float("nan"), {}

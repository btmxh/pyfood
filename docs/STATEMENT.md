## Problem Statement: DVRPTW

The Dynamic Vehicle Routing Problem with Time Windows (DVRPTW) is defined as a directed graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$.

### 1. Physical and Fleet Constraints
* **Depot ($v_{0}$):** The designated starting and ending point for all $K$ vehicles.
* **Traversal Cost ($c(e)$):** Calculated as the Euclidean distance between two nodes: $c(e)=|p(v)-p(w)|_{2}$.
* **Vehicle Capacity ($c_{k}$):** Every vehicle has a maximum capacity limit that must satisfy $c_{k} \ge \sum_{v\in R_{k}}d(v)$, where $d(v)$ is request demand.
* **Operational Assumptions:** Vehicles move at a constant speed in a straight line between nodes.

### 2. Temporal and Dynamic Elements
* **Time Windows ($t(v)$):** Defined as the interval $[b(v), e(v)]$, where $b(v)$ is the earliest and $e(v)$ is the latest service start time.
* **Waiting Time:** If a vehicle arrives before $b(v)$, it must wait until the window opens.
* **Dynamic Arrivals:** Requests are unknown to the solver until their specific arrival time within the simulation.
* **Service Time:** Each stop requires a specific duration for service or unloading.

### 3. Optimization Objectives
The problem aims to satisfy two conflicting objectives, which are combined into a weighted sum fitness function $f = w \cdot \overline{obj_1} + (1-w) \cdot \overline{obj_2}$:

1.  **Minimize Total Travel Cost ($obj_1$):** The sum of the costs of all edges traversed by the vehicle fleet: $\sum_{k=1}^{K} c(\pi_{k})$.
2.  **Maximize Accepted Requests ($obj_2$):** Scheduling the maximum number of customers while respecting capacity and time window constraints.

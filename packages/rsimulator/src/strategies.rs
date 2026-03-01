/// strategies.rs — Native Rust dispatching strategy implementations.
///
/// # Contents
/// - [`GreedyRustStrategy`] — greedy dispatch in ascending request-ID order
/// - [`greedy_strategy`]    — `#[pyfunction]` factory returning a wrapped strategy
use pyo3::prelude::*;

use crate::instance::{InstanceView, Request, RustStrategy};
use crate::types::{
    NativeStrategyWrapper, RequestId, SimAction, SimulationSnapshot, VehicleSnapshot,
};

// ---------------------------------------------------------------------------
// GreedyRustStrategy — example native strategy
// ---------------------------------------------------------------------------

/// Greedy native strategy: dispatch idle vehicles to pending requests in
/// ascending request-ID order.  Semantically equivalent to the Python
/// `SimpleGreedyStrategy` used in the test suite.
///
/// Feasibility check: before dispatching vehicle `v` to request `r`, verify
/// that `max(available_at + travel_time, earliest) <= latest`.  Infeasible
/// requests are skipped (the simulator will auto-reject them once their time
/// window closes).
///
/// Request data is fetched on-demand via [`InstanceView::get`] using IDs that
/// arrive through the snapshot — the strategy never enumerates the full
/// request table.
struct GreedyRustStrategy {
    /// Depot request, cached during [`initialize`] for position lookups.
    depot: Option<Request>,
    /// Vehicle speed (all vehicles share one speed in this model).
    speed: f64,
}

impl RustStrategy for GreedyRustStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        self.depot = view.get(view.depot_id()).cloned();
        self.speed = view
            .vehicle_specs()
            .first()
            .map(|vs| vs.speed)
            .unwrap_or(1.0);
    }

    fn next_events(
        &mut self,
        state: &SimulationSnapshot,
        view: &InstanceView<'_>,
    ) -> Vec<SimAction> {
        if state.pending.is_empty() {
            return vec![];
        }

        let mut pending_sorted: Vec<i64> = state.pending.iter().map(|r| r.0).collect();
        pending_sorted.sort_unstable();

        let mut idle_vehicles: Vec<&VehicleSnapshot> = state
            .vehicles
            .iter()
            .filter(|v| v.available_at <= state.time)
            .collect();
        // Stable order by vehicle_id for determinism
        idle_vehicles.sort_unstable_by_key(|v| v.vehicle_id);

        let mut actions = Vec::new();
        // Remaining feasible pending requests (in sorted order)
        let mut remaining: std::collections::VecDeque<i64> = pending_sorted.into_iter().collect();

        'vehicles: for vehicle in idle_vehicles {
            // Resolve the vehicle's current position request for distance calcs.
            // IDs in vehicle.position come from the snapshot — legitimate access.
            let vehicle_pos_req: Option<&Request> =
                if vehicle.position == self.depot.as_ref().map(|d| d.id).unwrap_or(RequestId(-1)) {
                    self.depot.as_ref()
                } else {
                    view.get(vehicle.position)
                };

            let n = remaining.len();
            for i in 0..n {
                let dest_id = RequestId(remaining[i]);
                // ID comes from state.pending — legitimately revealed by the simulator.
                let dest_req = match view.get(dest_id) {
                    Some(r) => r,
                    None => continue,
                };

                // Compute earliest possible service start
                let travel_time = match vehicle_pos_req {
                    Some(pos) => {
                        let dist = pos.distance_to(dest_req);
                        if self.speed > 0.0 {
                            dist / self.speed
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                };
                let arrival = vehicle.available_at + travel_time;
                let service_start = f64::max(arrival, dest_req.time_window.earliest);

                if service_start <= dest_req.time_window.latest {
                    // Feasible — dispatch and remove from remaining
                    remaining.remove(i);
                    actions.push(SimAction::Dispatch {
                        vehicle_id: vehicle.vehicle_id,
                        dest: dest_id,
                    });
                    continue 'vehicles;
                }
                // Infeasible — try next request for this vehicle
            }
            // No feasible request found for this vehicle; leave it idle.
        }

        // If nothing was dispatched and vehicles are busy, schedule a wake-up
        if actions.is_empty() {
            let next_vehicle_time = state
                .vehicles
                .iter()
                .map(|v| v.available_at)
                .filter(|&t| t > state.time)
                .fold(f64::INFINITY, f64::min);
            if next_vehicle_time.is_finite() {
                actions.push(SimAction::Wait {
                    until: next_vehicle_time,
                });
            }
        }

        actions
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Return a [`NativeStrategyWrapper`] containing the built-in greedy strategy.
///
/// The greedy strategy dispatches idle vehicles to pending requests in
/// ascending request-ID order — equivalent to the Python `SimpleGreedyStrategy`
/// used in the test suite.  Infeasible requests (where the vehicle cannot arrive
/// before the time window closes) are skipped; the simulator auto-rejects them
/// when their window expires.
#[pyfunction]
pub fn greedy_strategy() -> NativeStrategyWrapper {
    NativeStrategyWrapper {
        inner: Some(Box::new(GreedyRustStrategy {
            depot: None,
            speed: 1.0,
        })),
    }
}

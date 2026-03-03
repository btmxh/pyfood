/// gp_strategy.rs — GP-tree-driven routing and scheduling strategies.
///
/// Implements [`RoutingStrategy`] and [`SchedulingStrategy`] using GP expression
/// trees, combined via [`ComposableStrategy`].  Exposed to Python via [`gp_strategy`].
use std::collections::HashMap;

use pyo3::prelude::*;

use crate::instance::InstanceView;
use crate::strategies::{ComposableStrategy, RoutingStrategy, SchedulingStrategy};
use crate::types::{NativeStrategyWrapper, RequestId, VehicleSnapshot};

use super::gp_tree::{EvalContext, FlatGpTree, FlatTree};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of terminal types in GP trees (0=TravelTime, 1=WindowEarliest, 2=WindowLatest,
/// 3=TimeUntilDue, 4=Demand, 5=CurrentLoad, 6=RemainingCapacity, 7=ReleaseTime)
const NUM_TERMINALS: usize = 8;

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

fn make_ctx<'a>(
    request: &'a crate::instance::Request,
    vehicle: &'a VehicleSnapshot,
    view: &'a InstanceView<'_>,
    current_time: f32,
    speed: f32,
    vehicle_capacity: f32,
) -> EvalContext<'a> {
    let (vx, vy) = view
        .get(vehicle.position)
        .map(|r| (r.x, r.y))
        .unwrap_or((0.0, 0.0));
    EvalContext {
        request,
        vehicle,
        current_time,
        speed,
        vehicle_capacity,
        vehicle_pos_x: vx,
        vehicle_pos_y: vy,
    }
}

/// Extract speed and capacity from the first vehicle spec in the view.
fn view_vehicle_params(view: &InstanceView<'_>) -> (f32, f32) {
    view.vehicle_specs()
        .first()
        .map(|vs| (vs.speed, vs.capacity))
        .unwrap_or((1.0, f32::INFINITY))
}

// ---------------------------------------------------------------------------
// GpRoutingStrategy
// ---------------------------------------------------------------------------

pub struct GpRoutingStrategy {
    routing_tree: FlatTree,
    reject_tree: FlatTree,
}

impl RoutingStrategy for GpRoutingStrategy {
    fn route(
        &mut self,
        rid: RequestId,
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
        time: f32,
    ) -> Option<i64> {
        if vehicles.is_empty() {
            return None;
        }

        let request = view.get(rid)?;
        let (speed, capacity) = view_vehicle_params(view);

        // Build terminal matrix for batch evaluation (NUM_TERMINALS × n vehicles)
        let mut terminal_matrix: Vec<Vec<f32>> = vec![Vec::new(); NUM_TERMINALS];
        for vehicle in vehicles {
            let ctx = make_ctx(request, vehicle, view, time, speed, capacity);
            for term_id in 0..NUM_TERMINALS {
                terminal_matrix[term_id].push(ctx.terminal_value(term_id as u8));
            }
        }

        // Batch evaluate routing and reject trees
        let routing_scores = self.routing_tree.eval_batch(&terminal_matrix);
        let reject_scores = self.reject_tree.eval_batch(&terminal_matrix);

        // Find best vehicle by routing score
        let mut best_idx = 0;
        let mut best_routing_score = routing_scores[0];
        for (i, &score) in routing_scores.iter().enumerate().skip(1) {
            if score > best_routing_score {
                best_routing_score = score;
                best_idx = i;
            }
        }

        let best_reject_score = reject_scores[best_idx];
        if best_reject_score > best_routing_score {
            None
        } else {
            Some(vehicles[best_idx].vehicle_id)
        }
    }
}

// ---------------------------------------------------------------------------
// GpSchedulingStrategy
// ---------------------------------------------------------------------------

pub struct GpSchedulingStrategy {
    sequencing_tree: FlatTree,
}

impl SchedulingStrategy for GpSchedulingStrategy {
    fn schedule(
        &mut self,
        vehicle: &VehicleSnapshot,
        queue: &[RequestId],
        view: &InstanceView<'_>,
        time: f32,
    ) -> RequestId {
        if queue.is_empty() {
            return RequestId(0);
        }

        let (speed, capacity) = view_vehicle_params(view);

        // Build terminal matrix for batch evaluation (NUM_TERMINALS × queue length)
        let mut terminal_matrix: Vec<Vec<f32>> = vec![Vec::new(); NUM_TERMINALS];
        let mut valid_requests = Vec::new();

        for &rid in queue {
            if let Some(request) = view.get(rid) {
                let ctx = make_ctx(request, vehicle, view, time, speed, capacity);
                for term_id in 0..NUM_TERMINALS {
                    terminal_matrix[term_id].push(ctx.terminal_value(term_id as u8));
                }
                valid_requests.push(rid);
            }
        }

        if valid_requests.is_empty() {
            return queue[0];
        }

        // Batch evaluate sequencing tree
        let scores = self.sequencing_tree.eval_batch(&terminal_matrix);

        // Find index with highest score
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        valid_requests.get(best_idx).copied().unwrap_or(queue[0])
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Return a [`NativeStrategyWrapper`] driven by three GP expression trees.
///
/// - `routing_tree`    — scores `(request, vehicle)` pairs; highest wins.
/// - `sequencing_tree` — scores queued requests for an idle vehicle; highest wins.
/// - `reject_tree`     — if its score exceeds the routing score, request is rejected.
#[pyfunction]
pub fn gp_strategy(
    routing_tree: FlatGpTree,
    sequencing_tree: FlatGpTree,
    reject_tree: FlatGpTree,
) -> NativeStrategyWrapper {
    use std::collections::HashSet;

    NativeStrategyWrapper {
        inner: Some(Box::new(ComposableStrategy {
            router: Box::new(GpRoutingStrategy {
                routing_tree: routing_tree.inner,
                reject_tree: reject_tree.inner,
            }),
            scheduler: Box::new(GpSchedulingStrategy {
                sequencing_tree: sequencing_tree.inner,
            }),
            queues: HashMap::new(),
            routed: HashSet::new(),
        })),
    }
}

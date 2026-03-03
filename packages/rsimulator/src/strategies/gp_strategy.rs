/// gp_strategy.rs — GP-tree-driven routing and scheduling strategies.
///
/// Implements [`RoutingStrategy`] and [`SchedulingStrategy`] using GP expression
/// trees, combined via [`ComposableStrategy`].  Exposed to Python via [`gp_strategy`].
use std::collections::HashMap;

use pyo3::prelude::*;

use crate::instance::{InstanceView, Request};
use crate::strategies::{ComposableStrategy, RoutingStrategy, SchedulingStrategy};
use crate::types::{NativeStrategyWrapper, RequestId, VehicleSnapshot};

use super::gp_tree::{EvalContext, FlatGpTree, FlatTree};

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

fn make_ctx<'a>(
    request: &'a Request,
    vehicle: &'a VehicleSnapshot,
    requests: &HashMap<RequestId, Request>,
    current_time: f32,
    speed: f32,
    vehicle_capacity: f32,
) -> EvalContext<'a> {
    let (vx, vy) = requests
        .get(&vehicle.position)
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

// ---------------------------------------------------------------------------
// GpRoutingStrategy
// ---------------------------------------------------------------------------

pub struct GpRoutingStrategy {
    routing_tree: FlatTree,
    reject_tree: FlatTree,
    requests: HashMap<RequestId, Request>,
    speed: f32,
    capacity: f32,
    current_time: f32,
}

impl RoutingStrategy for GpRoutingStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        if let Some(vs) = view.vehicle_specs().first() {
            self.speed = vs.speed;
            self.capacity = vs.capacity;
        }
        if let Some(depot) = view.get(view.depot_id()) {
            self.requests.insert(view.depot_id(), depot.clone());
        }
    }

    fn begin_tick(&mut self, time: f32) {
        self.current_time = time;
    }

    fn route(
        &mut self,
        rid: RequestId,
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
    ) -> Option<i64> {
        if !self.requests.contains_key(&rid) {
            if let Some(req) = view.get(rid) {
                self.requests.insert(rid, req.clone());
            }
        }
        for v in vehicles {
            if !self.requests.contains_key(&v.position) {
                if let Some(req) = view.get(v.position) {
                    self.requests.insert(v.position, req.clone());
                }
            }
        }

        let request = self.requests.get(&rid)?;

        let mut best_vehicle_id: Option<i64> = None;
        let mut best_routing_score = f32::NEG_INFINITY;
        let mut best_reject_score = f32::NEG_INFINITY;

        for vehicle in vehicles {
            let ctx = make_ctx(
                request,
                vehicle,
                &self.requests,
                self.current_time,
                self.speed,
                self.capacity,
            );
            let routing_score = self.routing_tree.eval_scalar(&ctx);
            let reject_score = self.reject_tree.eval_scalar(&ctx);

            if routing_score > best_routing_score {
                best_routing_score = routing_score;
                best_reject_score = reject_score;
                best_vehicle_id = Some(vehicle.vehicle_id);
            }
        }

        match best_vehicle_id {
            None => None,
            Some(vid) => {
                if best_reject_score > best_routing_score {
                    None
                } else {
                    Some(vid)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpSchedulingStrategy
// ---------------------------------------------------------------------------

pub struct GpSchedulingStrategy {
    sequencing_tree: FlatTree,
    requests: HashMap<RequestId, Request>,
    speed: f32,
    capacity: f32,
    current_time: f32,
}

impl SchedulingStrategy for GpSchedulingStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        if let Some(vs) = view.vehicle_specs().first() {
            self.speed = vs.speed;
            self.capacity = vs.capacity;
        }
        if let Some(depot) = view.get(view.depot_id()) {
            self.requests.insert(view.depot_id(), depot.clone());
        }
    }

    fn begin_tick(&mut self, time: f32) {
        self.current_time = time;
    }

    fn schedule(
        &mut self,
        vehicle: &VehicleSnapshot,
        queue: &[RequestId],
        view: &InstanceView<'_>,
    ) -> RequestId {
        if !self.requests.contains_key(&vehicle.position) {
            if let Some(req) = view.get(vehicle.position) {
                self.requests.insert(vehicle.position, req.clone());
            }
        }
        for &rid in queue {
            if !self.requests.contains_key(&rid) {
                if let Some(req) = view.get(rid) {
                    self.requests.insert(rid, req.clone());
                }
            }
        }

        queue
            .iter()
            .copied()
            .filter_map(|rid| {
                let request = self.requests.get(&rid)?;
                let ctx = make_ctx(
                    request,
                    vehicle,
                    &self.requests,
                    self.current_time,
                    self.speed,
                    self.capacity,
                );
                Some((rid, self.sequencing_tree.eval_scalar(&ctx)))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(rid, _)| rid)
            .unwrap_or(queue[0])
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
                requests: HashMap::new(),
                speed: 1.0_f32,
                capacity: f32::INFINITY,
                current_time: 0.0_f32,
            }),
            scheduler: Box::new(GpSchedulingStrategy {
                sequencing_tree: sequencing_tree.inner,
                requests: HashMap::new(),
                speed: 1.0_f32,
                capacity: f32::INFINITY,
                current_time: 0.0_f32,
            }),
            queues: HashMap::new(),
            routed: HashSet::new(),
        })),
    }
}

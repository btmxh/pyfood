/// gp_strategy.rs — GP-tree-driven routing and scheduling strategies.
///
/// Implements [`RoutingStrategy`], [`SchedulingStrategy`], and
/// [`BatchRoutingStrategy`] using GP expression trees, combined via
/// [`ComposableStrategy`] / [`BatchComposableStrategy`].
///
/// # Role contexts
///
/// Each of the three tree roles has its own [`RoleContext`] implementor:
///
/// | Role               | Context type      | Used by                             |
/// |--------------------|-------------------|-------------------------------------|
/// | Vehicle assignment | [`RoutingCtx`]    | [`GpRoutingStrategy`], [`GpBatchRoutingStrategy`] |
/// | Queue ordering     | [`SequencingCtx`] | [`GpSchedulingStrategy`]            |
/// | Request rejection  | [`RejectCtx`]     | [`GpRoutingStrategy`], [`GpBatchRoutingStrategy`] |
///
/// All three currently expose the same 8 terminals (IDs 0–7) but are defined
/// independently so their terminal sets can diverge without coupling.
use crate::hashmap::{Map as HashMap, Set as HashSet};

use pyo3::prelude::*;

use crate::instance::InstanceView;
use crate::strategies::{
    BatchRoutingStrategy, ComposableStrategy, RoutingStrategy, SchedulingStrategy,
};
use crate::types::{NativeDispatchStrategy, RequestId, VehicleSnapshot};

use super::gp_tree::{EvalContext, FlatGpTree, RoleContext};
use crate::instance::InstanceLimits;

// ---------------------------------------------------------------------------
// Role-specific context types
// ---------------------------------------------------------------------------

/// Evaluation context for the vehicle-assignment (routing) tree.
pub struct RoutingCtx<'a>(EvalContext<'a>);
/// Evaluation context for the queue-ordering (sequencing) tree.
pub struct SequencingCtx<'a>(EvalContext<'a>);
/// Evaluation context for the request-rejection tree.
pub struct RejectCtx<'a>(EvalContext<'a>);

// Terminal IDs 0–7 are shared across all three roles:
//   0=TravelTime, 1=WindowEarliest, 2=WindowLatest, 3=TimeUntilDue,
//   4=Demand, 5=CurrentLoad, 6=RemainingCapacity, 7=ReleaseTime

impl RoleContext for RoutingCtx<'_> {
    const NUM_TERMINALS: usize = 8;
    #[inline(always)]
    fn eval_terminal(&self, id: u8) -> f32 {
        self.0.terminal_value(id)
    }
}

impl RoleContext for SequencingCtx<'_> {
    const NUM_TERMINALS: usize = 8;
    #[inline(always)]
    fn eval_terminal(&self, id: u8) -> f32 {
        self.0.terminal_value(id)
    }
}

impl RoleContext for RejectCtx<'_> {
    const NUM_TERMINALS: usize = 8;
    #[inline(always)]
    fn eval_terminal(&self, id: u8) -> f32 {
        self.0.terminal_value(id)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_ctx<'a>(
    request: &'a crate::instance::Request,
    vehicle: &'a VehicleSnapshot,
    view: &'a InstanceView<'_>,
    current_time: f32,
    speed: f32,
    vehicle_capacity: f32,
    bounds: &'a InstanceLimits,
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
        bounds: Some(bounds),
    }
}

fn view_vehicle_params(view: &InstanceView<'_>) -> (f32, f32) {
    view.vehicle_specs()
        .first()
        .map(|vs| (vs.speed, vs.capacity))
        .unwrap_or((1.0, f32::INFINITY))
}

/// Route a single request using the routing and reject trees.
///
/// Shared by [`GpRoutingStrategy`] and [`GpBatchRoutingStrategy`] to avoid
/// code duplication.
fn route_one(
    routing_tree: &FlatGpTree,
    reject_tree: &FlatGpTree,
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

    // Bounds are pre-computed once per instance and stored in InstanceView.
    let bounds = &view.limits;

    let t_mkctx = std::time::Instant::now();
    let routing_ctxs: Vec<RoutingCtx<'_>> = vehicles
        .iter()
        .map(|v| RoutingCtx(make_ctx(request, v, view, time, speed, capacity, bounds)))
        .collect();
    crate::bench::TIME_MAKE_CTX_NS.fetch_add(
        crate::bench::elapsed_ns(t_mkctx),
        std::sync::atomic::Ordering::Relaxed,
    );

    let t_evbat = std::time::Instant::now();
    let routing_scores = routing_tree.eval_batch_for(&routing_ctxs);
    crate::bench::TIME_EVAL_BATCH_ROUTING_NS.fetch_add(
        crate::bench::elapsed_ns(t_evbat),
        std::sync::atomic::Ordering::Relaxed,
    );

    let mut best_idx = 0;
    let mut best_routing_score = routing_scores[0];
    let t_best = std::time::Instant::now();
    for (i, &score) in routing_scores.iter().enumerate().skip(1) {
        if score > best_routing_score {
            best_routing_score = score;
            best_idx = i;
        }
    }
    crate::bench::TIME_ROUTE_ONE_BESTSCAN_NS.fetch_add(
        crate::bench::elapsed_ns(t_best),
        std::sync::atomic::Ordering::Relaxed,
    );

    let t_mkr = std::time::Instant::now();
    let reject_ctx = RejectCtx(make_ctx(
        request,
        &vehicles[best_idx],
        view,
        time,
        speed,
        capacity,
        bounds,
    ));
    crate::bench::TIME_ROUTE_ONE_MAKE_REJECT_NS.fetch_add(
        crate::bench::elapsed_ns(t_mkr),
        std::sync::atomic::Ordering::Relaxed,
    );
    let t_rej = std::time::Instant::now();
    let reject_score = reject_tree.eval_ctx(&reject_ctx);
    crate::bench::TIME_REJECT_EVAL_NS.fetch_add(
        crate::bench::elapsed_ns(t_rej),
        std::sync::atomic::Ordering::Relaxed,
    );

    crate::bench::ROUTE_ONE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if reject_score > best_routing_score {
        None
    } else {
        Some(vehicles[best_idx].vehicle_id)
    }
}

// ---------------------------------------------------------------------------
// GpRoutingStrategy
// ---------------------------------------------------------------------------

pub struct GpRoutingStrategy {
    routing_tree: FlatGpTree,
    reject_tree: FlatGpTree,
}

impl RoutingStrategy for GpRoutingStrategy {
    fn route(
        &mut self,
        rid: RequestId,
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
        time: f32,
    ) -> Option<i64> {
        route_one(
            &self.routing_tree,
            &self.reject_tree,
            rid,
            vehicles,
            view,
            time,
        )
    }
}

// ---------------------------------------------------------------------------
// GpBatchRoutingStrategy
// ---------------------------------------------------------------------------

/// Naive [`BatchRoutingStrategy`] that applies the per-request GP routing logic
/// sequentially to each request in the batch.
///
/// Because [`BatchRoutingStrategy::route_batch`] does not receive a simulation
/// time, time-dependent terminals (`TimeUntilDue = latest − t`) are evaluated
/// at `t = 0.0`.
pub struct GpBatchRoutingStrategy {
    routing_tree: FlatGpTree,
    reject_tree: FlatGpTree,
}

impl BatchRoutingStrategy for GpBatchRoutingStrategy {
    fn route_batch(
        &mut self,
        requests: &[RequestId],
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
    ) -> Vec<(RequestId, Option<i64>)> {
        requests
            .iter()
            .map(|&rid| {
                let assignment = route_one(
                    &self.routing_tree,
                    &self.reject_tree,
                    rid,
                    vehicles,
                    view,
                    0.0,
                );
                (rid, assignment)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GpSchedulingStrategy
// ---------------------------------------------------------------------------

pub struct GpSchedulingStrategy {
    sequencing_tree: FlatGpTree,
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

        // Bounds are pre-computed once per instance and stored in InstanceView.
        let bounds = &view.limits;

        let mut valid_requests = Vec::new();
        let mut ctxs: Vec<SequencingCtx<'_>> = Vec::new();

        for &rid in queue {
            if let Some(request) = view.get(rid) {
                ctxs.push(SequencingCtx(make_ctx(
                    request, vehicle, view, time, speed, capacity, bounds,
                )));
                valid_requests.push(rid);
            }
        }

        if valid_requests.is_empty() {
            return queue[0];
        }

        let scores = self.sequencing_tree.eval_batch_for(&ctxs);

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
// Factory functions
// ---------------------------------------------------------------------------

/// Return a [`NativeDispatchStrategy`] driven by three GP expression trees
/// using per-request (non-batched) routing.
///
/// - `routing_tree`    — scores `(request, vehicle)` pairs; highest wins.
/// - `sequencing_tree` — scores queued requests for an idle vehicle; highest wins.
/// - `reject_tree`     — if its score exceeds the routing score, request is rejected.
#[pyfunction]
pub fn gp_strategy(
    routing_tree: FlatGpTree,
    sequencing_tree: FlatGpTree,
    reject_tree: FlatGpTree,
) -> NativeDispatchStrategy {
    NativeDispatchStrategy {
        inner: Some(Box::new(ComposableStrategy::new(
            Box::new(GpRoutingStrategy {
                routing_tree,
                reject_tree,
            }),
            Box::new(GpSchedulingStrategy { sequencing_tree }),
        ))),
    }
}

/// Return a [`NativeDispatchStrategy`] using slot-based batch routing with GP trees.
///
/// Requests released within each `slot_size`-wide time window are collected and
/// then routed all at once by applying the GP routing logic sequentially to each
/// request (naive batch: no time is available during routing, so time-dependent
/// terminals evaluate at `t = 0.0`).
///
/// - `routing_tree`    — scores `(request, vehicle)` pairs; highest wins.
/// - `sequencing_tree` — scores queued requests for an idle vehicle; highest wins.
/// - `reject_tree`     — if its score exceeds the routing score, request is rejected.
/// - `slot_size`       — width of each routing time slot.
#[pyfunction]
pub fn gp_batch_strategy(
    routing_tree: FlatGpTree,
    sequencing_tree: FlatGpTree,
    reject_tree: FlatGpTree,
    slot_size: f64,
) -> NativeDispatchStrategy {
    use crate::strategies::batch::BatchComposableStrategy;

    let slot_size = slot_size as f32;
    NativeDispatchStrategy {
        inner: Some(Box::new(BatchComposableStrategy {
            router: Box::new(GpBatchRoutingStrategy {
                routing_tree,
                reject_tree,
            }),
            scheduler: Box::new(GpSchedulingStrategy { sequencing_tree }),
            slot_size,
            next_slot_end: slot_size,
            buffer: Vec::new(),
            seen: HashSet::default(),
            queues: HashMap::default(),
        })),
    }
}

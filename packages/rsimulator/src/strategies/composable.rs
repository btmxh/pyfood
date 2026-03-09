use crate::hashmap::{Map as HashMap, Set as HashSet};
/// composable.rs — ComposableStrategy: per-request router + scheduler template.
///
/// # Contents
/// - [`PyRoutingAdapter`]    — wraps a Python router as `Box<dyn RoutingStrategy>`
/// - [`PySchedulingAdapter`] — wraps a Python scheduler as `Box<dyn SchedulingStrategy>`
/// - [`ComposableStrategy`]  — `RustStrategy` impl combining router + scheduler
/// - [`composable_strategy`] — `#[pyfunction]` factory
use std::collections::VecDeque;

use pyo3::prelude::*;

use super::{NativeRoutingStrategy, NativeSchedulingStrategy, RoutingStrategy, SchedulingStrategy};
use crate::instance::{DispatchStrategy, InstanceView};
use crate::types::{
    NativeDispatchStrategy, RequestId, SimAction, SimulationSnapshot, VehicleSnapshot,
};

use super::{build_py_instance_view, build_py_vehicle, build_py_vehicles};

// ---------------------------------------------------------------------------
// PyRoutingAdapter — wraps Py<PyAny> as Box<dyn RoutingStrategy>
// ---------------------------------------------------------------------------

/// Adapts a Python object to [`RoutingStrategy`].
///
/// Expected Python protocol:
/// ```python
/// def route(
///     self,
///     request_id: int,
///     vehicles: list[dict],
///     instance_view: dict,
/// ) -> int | None:
///     ...  # return vehicle_id to assign, or None to reject
/// ```
///
/// `instance_view` has keys `"depot_id"` and `"vehicle_specs"`.
pub struct PyRoutingAdapter {
    pub py_router: Py<PyAny>,
    pub py_view: Option<Py<PyAny>>,
}

impl RoutingStrategy for PyRoutingAdapter {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        Python::attach(|py| {
            self.py_view =
                Some(build_py_instance_view(py, view).expect("failed to build instance view dict"));
        });
    }

    fn route(
        &mut self,
        request: RequestId,
        vehicles: &[VehicleSnapshot],
        _view: &InstanceView<'_>,
        _time: f32,
    ) -> Option<i64> {
        Python::attach(|py| {
            let py_vehicles =
                build_py_vehicles(py, vehicles).expect("failed to build vehicles list");
            // Build instance view on-demand if initialize() wasn't called.
            if self.py_view.is_none() {
                let built =
                    build_py_instance_view(py, _view).expect("failed to build instance view dict");
                self.py_view = Some(built);
            }
            let py_view = self.py_view.as_ref().unwrap().clone_ref(py);
            let result = self
                .py_router
                .call_method1(py, "route", (request.0, py_vehicles, py_view))
                .expect("Python router.route() raised an exception");
            if result.is_none(py) {
                None
            } else {
                Some(
                    result
                        .extract::<i64>(py)
                        .expect("router.route() must return int or None"),
                )
            }
        })
    }
}

// ---------------------------------------------------------------------------
// PySchedulingAdapter — wraps Py<PyAny> as Box<dyn SchedulingStrategy>
// ---------------------------------------------------------------------------

/// Adapts a Python object to [`SchedulingStrategy`].
///
/// Expected Python protocol:
/// ```python
/// def schedule(
///     self,
///     vehicle: dict,
///     queue: list[int],
///     instance_view: dict,
/// ) -> int:
///     ...  # return a request_id from queue
/// ```
pub struct PySchedulingAdapter {
    pub py_scheduler: Py<PyAny>,
    pub py_view: Option<Py<PyAny>>,
}

impl SchedulingStrategy for PySchedulingAdapter {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        Python::attach(|py| {
            self.py_view =
                Some(build_py_instance_view(py, view).expect("failed to build instance view dict"));
        });
    }

    fn schedule(
        &mut self,
        vehicle: &VehicleSnapshot,
        queue: &[RequestId],
        _view: &InstanceView<'_>,
        _time: f32,
    ) -> RequestId {
        Python::attach(|py| {
            let py_vehicle = build_py_vehicle(py, vehicle).expect("failed to build vehicle dict");
            let py_queue: Vec<i64> = queue.iter().map(|r| r.0).collect();
            // Build instance view on-demand if initialize() wasn't called.
            if self.py_view.is_none() {
                let built =
                    build_py_instance_view(py, _view).expect("failed to build instance view dict");
                self.py_view = Some(built);
            }
            let py_view = self.py_view.as_ref().unwrap().clone_ref(py);
            let result = self
                .py_scheduler
                .call_method1(py, "schedule", (py_vehicle, py_queue, py_view))
                .expect("Python scheduler.schedule() raised an exception");
            let rid = result
                .extract::<i64>(py)
                .expect("scheduler.schedule() must return an int request_id");
            let request_id = RequestId(rid);
            debug_assert!(
                queue.contains(&request_id),
                "scheduler returned {rid} which is not in the queue"
            );
            request_id
        })
    }
}

// ---------------------------------------------------------------------------
// ComposableStrategy
// ---------------------------------------------------------------------------

/// A [`RustStrategy`] template that separates routing and scheduling concerns.
///
/// On each simulation tick:
///
/// 1. **Routing** — for each newly-released request (not yet routed), call
///    `router.route(request, vehicles, view)` once.  `Some(vehicle_id)` pushes
///    the request onto that vehicle's queue; `None` emits a [`SimAction::Reject`].
///
/// 2. **Scheduling** — for each idle vehicle with a non-empty queue, call
///    `scheduler.schedule(vehicle, queue, view)` → emit [`SimAction::Dispatch`]
///    for the returned request and remove it from the queue.
///
/// 3. **Wait** — if no actions result but work remains, emit [`SimAction::Wait`]
///    until the earlier of the next vehicle free time or next request release.
pub struct ComposableStrategy {
    pub router: Box<dyn RoutingStrategy>,
    pub scheduler: Box<dyn SchedulingStrategy>,
    /// Per-vehicle pending queue (assigned but not yet dispatched).
    pub queues: HashMap<i64, VecDeque<RequestId>>,
    /// Requests already passed to the router (de-dup guard).
    pub routed: HashSet<RequestId>,
    /// Per-vehicle total demand of requests currently queued (assigned but not dispatched).
    /// Densely stored: index = vehicle_id_to_index[vehicle_id]
    pub queued_loads: Vec<f32>,
    /// Map vehicle_id -> index into `queued_loads`.
    pub vehicle_id_to_index: HashMap<i64, usize>,
    last_time: f32,
    next_release_idx: usize,
}

impl ComposableStrategy {
    pub fn new(router: Box<dyn RoutingStrategy>, scheduler: Box<dyn SchedulingStrategy>) -> Self {
        Self {
            router,
            scheduler,
            queues: HashMap::default(),
            routed: HashSet::default(),
            queued_loads: Vec::new(),
            vehicle_id_to_index: HashMap::default(),
            last_time: 0.0,
            next_release_idx: 0,
        }
    }
}

impl DispatchStrategy for ComposableStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        self.router.initialize(view);
        self.scheduler.initialize(view);
        self.queued_loads.clear();
        self.vehicle_id_to_index.clear();
        for (i, vs) in view.vehicle_specs().iter().enumerate() {
            self.queues.entry(vs.id).or_default();
            self.vehicle_id_to_index.insert(vs.id, i);
            self.queued_loads.push(0.0);
        }
        self.last_time = f32::NEG_INFINITY;
        self.next_release_idx = 0;
    }

    fn next_events(
        &mut self,
        state: &SimulationSnapshot<'_>,
        view: &InstanceView<'_>,
    ) -> Vec<SimAction> {
        // Defensive: ensure sub-strategies have been initialized. Some
        // call-sites may not invoke `initialize` on the boxed trait object
        // before the first `next_events` call; calling it here is idempotent
        // and cheap for Python adapters (rebuilds their cached `py_view`).
        self.router.initialize(view);
        self.scheduler.initialize(view);
        let mut actions: Vec<SimAction> = Vec::new();

        // Phase 1: route newly-released requests (exactly once per request).
        let t_p1_scan = std::time::Instant::now();
        let start = self.next_release_idx;
        while self.next_release_idx < view.ascending_release.len()
            && view.ascending_release[self.next_release_idx].0 <= state.time
        {
            self.next_release_idx += 1;
        }
        let end = self.next_release_idx;

        let new_requests = &view.ascending_release[start..end];
        crate::bench::TIME_PHASE1_SCAN_NS.fetch_add(
            crate::bench::elapsed_ns(t_p1_scan),
            std::sync::atomic::Ordering::Relaxed,
        );

        // Build augmented vehicles once per tick by applying per-vehicle queued
        // load totals (kept in `self.queued_loads`). As we assign new requests
        // during this loop, update both `self.queued_loads` and the
        // corresponding `aug_vehicles` entry so subsequent routing sees the
        // incremental assignments without rescanning queues.
        let mut aug_vehicles = state.vehicles.to_vec();
        for v in aug_vehicles.iter_mut() {
            if let Some(idx) = self.vehicle_id_to_index.get(&v.vehicle_id) {
                if let Some(qload) = self.queued_loads.get(*idx) {
                    v.current_load += *qload;
                }
            }
        }

        let t_p1_routing = std::time::Instant::now();
        for (_, rid) in new_requests {
            let rid = *rid;
            self.routed.insert(rid);

            match self.router.route(rid, &aug_vehicles, view, state.time) {
                Some(vehicle_id) => {
                    // push into queue and update queued_loads and aug_vehicles
                    self.queues.entry(vehicle_id).or_default().push_back(rid);
                    if let Some(req) = view.get(rid) {
                        let delta = req.demand;
                        // ensure we have an index for this vehicle
                        let idx = if let Some(idx) = self.vehicle_id_to_index.get(&vehicle_id) {
                            *idx
                        } else {
                            let new_idx = self.queued_loads.len();
                            self.vehicle_id_to_index.insert(vehicle_id, new_idx);
                            self.queued_loads.push(0.0);
                            new_idx
                        };
                        if let Some(entry) = self.queued_loads.get_mut(idx) {
                            *entry += delta;
                        }
                        // find and update matching aug_vehicles entry
                        if let Some(v) =
                            aug_vehicles.iter_mut().find(|v| v.vehicle_id == vehicle_id)
                        {
                            v.current_load += delta;
                        }
                    }
                }
                None => {
                    actions.push(SimAction::Reject { request_id: rid });
                }
            }
        }
        crate::bench::TIME_PHASE1_ROUTING_NS.fetch_add(
            crate::bench::elapsed_ns(t_p1_routing),
            std::sync::atomic::Ordering::Relaxed,
        );

        // Phase 2: dispatch idle vehicles with queued work.
        let mut vehicle_ids: Vec<i64> = state.vehicles.iter().map(|v| v.vehicle_id).collect();
        vehicle_ids.sort_unstable();

        for vid in vehicle_ids {
            let vehicle = match state.vehicles.iter().find(|v| v.vehicle_id == vid) {
                Some(v) => v,
                None => continue,
            };
            if vehicle.available_at > state.time {
                continue;
            }
            // get mutable queue for this vehicle
            let queue = match self.queues.get_mut(&vid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };

            // Phase 2a: filter out infeasible requests from the queue BEFORE
            // calling the scheduler. A request is infeasible if, when dispatched
            // now from the vehicle's current position, its service_start would
            // exceed its time-window latest, or if it would violate capacity.
            let t_p2f = std::time::Instant::now();
            let mut retained_queue: VecDeque<RequestId> = VecDeque::new();
            let mut infeasible: Vec<RequestId> = Vec::new();
            for &rid in queue.iter() {
                if let Some(req) = view.get(rid) {
                    // find vehicle spec for this vehicle id
                    let (speed, capacity) = view
                        .vehicle_specs()
                        .iter()
                        .find(|vs| vs.id == vehicle.vehicle_id)
                        .map(|vs| (vs.speed, vs.capacity))
                        .unwrap_or((1.0_f32, f32::INFINITY));

                    // from position -> request distance
                    if let Some(from_req) = view.get(vehicle.position) {
                        let dx = from_req.x - req.x;
                        let dy = from_req.y - req.y;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let travel_time = if speed == 0.0 {
                            f32::INFINITY
                        } else {
                            dist / speed
                        };
                        let arrival = state.time + travel_time;
                        let service_start = arrival.max(req.time_window.earliest);

                        let capacity_ok = if req.is_depot {
                            true
                        } else {
                            vehicle.current_load + req.demand <= capacity
                        };

                        if service_start > req.time_window.latest || !capacity_ok {
                            infeasible.push(rid);
                        } else {
                            retained_queue.push_back(rid);
                        }
                    } else {
                        // missing from-position info → be defensive and mark infeasible
                        infeasible.push(rid);
                    }
                } else {
                    // request not found in view (shouldn't happen) — mark infeasible
                    infeasible.push(rid);
                }
            }
            crate::bench::TIME_PHASE2_FEASIBILITY_NS.fetch_add(
                crate::bench::elapsed_ns(t_p2f),
                std::sync::atomic::Ordering::Relaxed,
            );

            // Replace the queue with only the retained (feasible) requests
            // Recompute queued_load for this vehicle from retained_queue (one pass)
            let mut new_queue_load: f32 = 0.0;
            for &qrid in retained_queue.iter() {
                if let Some(r) = view.get(qrid) {
                    new_queue_load += r.demand;
                }
            }
            *queue = retained_queue;
            // write back into dense queued_loads vector
            if let Some(idx) = self.vehicle_id_to_index.get(&vid) {
                if let Some(entry) = self.queued_loads.get_mut(*idx) {
                    *entry = new_queue_load;
                }
            } else {
                // mapping missing — append new slot
                let new_idx = self.queued_loads.len();
                self.vehicle_id_to_index.insert(vid, new_idx);
                self.queued_loads.push(new_queue_load);
            }

            // Emit Reject actions for infeasible assignments
            for rid in infeasible.iter() {
                actions.push(SimAction::Reject { request_id: *rid });
            }

            // If nothing feasible remains, continue
            if queue.is_empty() {
                continue;
            }

            // Ask scheduler to pick one request from the remaining feasible queue
            let t_p2s = std::time::Instant::now();
            let queue_slice: Vec<RequestId> = queue.iter().copied().collect();
            let chosen = self
                .scheduler
                .schedule(vehicle, &queue_slice, view, state.time);
            // remove chosen from queue and decrement queued_loads
            if let Some(pos) = queue.iter().position(|r| *r == chosen) {
                // remove by position
                queue.remove(pos);
                if let Some(req) = view.get(chosen) {
                    if let Some(idx) = self.vehicle_id_to_index.get(&vid) {
                        if let Some(entry) = self.queued_loads.get_mut(*idx) {
                            *entry -= req.demand;
                            if *entry < 0.0 {
                                *entry = 0.0; // guard against negative due to rounding
                            }
                        }
                    }
                }
            }
            actions.push(SimAction::Dispatch {
                vehicle_id: vid,
                dest: chosen,
            });
            crate::bench::TIME_PHASE2_SCHEDULE_NS.fetch_add(
                crate::bench::elapsed_ns(t_p2s),
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        self.last_time = state.time;

        actions
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Wrap a Python object as a [`NativeRoutingStrategy`].
///
/// The Python object must implement the dict-based routing protocol:
/// ```python
/// def route(self, request_id: int, vehicles: list[dict], instance_view: dict) -> int | None:
///     ...
/// ```
/// In practice, pass a Python-side adapter (e.g. `NativeRoutingAdapter`) that
/// converts these dicts into typed dataclasses before calling user code.
#[pyfunction]
pub fn python_routing_strategy(py_router: Py<PyAny>) -> NativeRoutingStrategy {
    NativeRoutingStrategy {
        inner: Some(Box::new(PyRoutingAdapter {
            py_router,
            py_view: None,
        })),
    }
}

/// Wrap a Python object as a [`NativeSchedulingStrategy`].
///
/// The Python object must implement the dict-based scheduling protocol:
/// ```python
/// def schedule(self, vehicle: dict, queue: list[int], instance_view: dict) -> int:
///     ...
/// ```
/// In practice, pass a Python-side adapter (e.g. `NativeSchedulingAdapter`) that
/// converts these dicts into typed dataclasses before calling user code.
#[pyfunction]
pub fn python_scheduling_strategy(py_scheduler: Py<PyAny>) -> NativeSchedulingStrategy {
    NativeSchedulingStrategy {
        inner: Some(Box::new(PySchedulingAdapter {
            py_scheduler,
            py_view: None,
        })),
    }
}

/// Return a [`NativeDispatchStrategy`] using the composable strategy template.
///
/// Both `router` and `scheduler` must be [`NativeRoutingStrategy`] /
/// [`NativeSchedulingStrategy`] instances.  Use [`python_routing_strategy`] and
/// [`python_scheduling_strategy`] to wrap Python objects:
///
/// ```python
/// from rsimulator import composable_strategy, python_routing_strategy, python_scheduling_strategy
///
/// strategy = composable_strategy(
///     python_routing_strategy(MyRouter()),
///     python_scheduling_strategy(MyScheduler()),
/// )
/// sim = Simulator(instance, strategy)
/// ```
#[pyfunction]
pub fn composable_strategy(
    router: &mut NativeRoutingStrategy,
    scheduler: &mut NativeSchedulingStrategy,
) -> NativeDispatchStrategy {
    NativeDispatchStrategy {
        inner: Some(Box::new(ComposableStrategy::new(
            router
                .inner
                .take()
                .expect("NativeRoutingStrategy already consumed"),
            scheduler
                .inner
                .take()
                .expect("NativeSchedulingStrategy already consumed"),
        ))),
    }
}

/// composable.rs — ComposableStrategy: per-request router + scheduler template.
///
/// # Contents
/// - [`PyRoutingAdapter`]    — wraps a Python router as `Box<dyn RoutingStrategy>`
/// - [`PySchedulingAdapter`] — wraps a Python scheduler as `Box<dyn SchedulingStrategy>`
/// - [`ComposableStrategy`]  — `RustStrategy` impl combining router + scheduler
/// - [`composable_strategy`] — `#[pyfunction]` factory
use std::collections::{HashMap, HashSet, VecDeque};

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
            eprintln!("PyRoutingAdapter::initialize() called");
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
            eprintln!("PySchedulingAdapter::initialize() called");
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
}

impl DispatchStrategy for ComposableStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        eprintln!("ComposableStrategy::initialize() called");
        self.router.initialize(view);
        self.scheduler.initialize(view);
        for vs in view.vehicle_specs() {
            self.queues.entry(vs.id).or_default();
        }
    }

    fn next_events(
        &mut self,
        state: &SimulationSnapshot,
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
        let mut new_requests: Vec<RequestId> = state
            .released
            .iter()
            .filter(|rid| {
                !self.routed.contains(rid)
                    && !state.served.contains(rid)
                    && !state.rejected.contains(rid)
            })
            .copied()
            .collect();
        new_requests.sort_unstable_by_key(|r| r.0);

        for rid in new_requests {
            self.routed.insert(rid);
            match self.router.route(rid, &state.vehicles, view, state.time) {
                Some(vehicle_id) => {
                    self.queues.entry(vehicle_id).or_default().push_back(rid);
                }
                None => {
                    actions.push(SimAction::Reject { request_id: rid });
                }
            }
        }

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
            let queue = match self.queues.get_mut(&vid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };

            let queue_slice: Vec<RequestId> = queue.iter().copied().collect();
            let chosen = self
                .scheduler
                .schedule(vehicle, &queue_slice, view, state.time);
            queue.retain(|r| *r != chosen);
            actions.push(SimAction::Dispatch {
                vehicle_id: vid,
                dest: chosen,
            });
        }

        // Phase 3: emit Wait if nothing to do but work remains.
        if actions.is_empty() {
            let has_queued = self.queues.values().any(|q| !q.is_empty());
            let has_pending = !state.pending.is_empty();

            if has_queued || has_pending {
                let next_vehicle_free = state
                    .vehicles
                    .iter()
                    .filter(|v| v.available_at > state.time)
                    .map(|v| v.available_at)
                    .fold(f32::INFINITY, f32::min);

                let next_release = state
                    .pending
                    .iter()
                    .filter(|rid| !state.released.contains(rid))
                    .filter_map(|rid| view.get(*rid))
                    .map(|r| r.release_time)
                    .fold(f32::INFINITY, f32::min);

                let until = f32::min(next_vehicle_free, next_release);
                if until.is_finite() && until > state.time {
                    actions.push(SimAction::Wait { until });
                }
            }
        }

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
        inner: Some(Box::new(ComposableStrategy {
            router: router
                .inner
                .take()
                .expect("NativeRoutingStrategy already consumed"),
            scheduler: scheduler
                .inner
                .take()
                .expect("NativeSchedulingStrategy already consumed"),
            queues: HashMap::new(),
            routed: HashSet::new(),
        })),
    }
}

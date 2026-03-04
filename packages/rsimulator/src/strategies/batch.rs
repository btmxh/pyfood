/// batch.rs — BatchComposableStrategy: slot-based batch router + scheduler.
///
/// # Contents
/// - [`BatchRoutingStrategy`]     — trait: route a whole batch of requests at once
/// - [`PyBatchRoutingAdapter`]    — wraps a Python batch router as the trait
/// - [`BatchComposableStrategy`]  — `RustStrategy` impl using batch routing + scheduling
/// - [`batch_composable_strategy`] — `#[pyfunction]` factory
///
/// # How it works
///
/// Time is divided into fixed-size slots (`slot_size`).  On every tick:
///
/// 1. Accumulate all newly-released requests into a pending buffer (no routing yet).
/// 2. If the current slot boundary has been crossed (i.e. the simulation time has
///    reached or passed `next_slot_end`), call `router.route_batch(buffer, vehicles,
///    view)` with the entire buffer at once.  Each returned `(request_id, vehicle_id)`
///    pair pushes the request onto that vehicle's queue; `None` vehicle_id rejects it.
///    Advance `next_slot_end` by `slot_size`.
/// 3. Dispatch idle vehicles with non-empty queues via the scheduler (same as
///    [`super::composable::ComposableStrategy`]).
/// 4. Emit a Wait if nothing to do but work remains, targeting the earlier of the
///    next slot boundary, next vehicle free time, or next pending release.
use std::collections::{HashMap, HashSet, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyList;

use super::{BatchRoutingStrategy, SchedulingStrategy};
use crate::instance::{DispatchStrategy, InstanceView};
use crate::types::{
    NativeDispatchStrategy, RequestId, SimAction, SimulationSnapshot, VehicleSnapshot,
};

use super::{build_py_instance_view, build_py_vehicles};

// ---------------------------------------------------------------------------
// PyBatchRoutingAdapter
// ---------------------------------------------------------------------------

/// Adapts a Python object to [`BatchRoutingStrategy`].
///
/// Expected Python protocol:
/// ```python
/// def route_batch(
///     self,
///     requests: list[int],
///     vehicles: list[dict],
///     instance_view: dict,
/// ) -> list[tuple[int, int | None]]:
///     ...  # return [(request_id, vehicle_id | None), ...]
/// ```
///
/// Every request in `requests` **must** appear exactly once in the returned
/// list.  Missing IDs are treated as rejections in debug builds (assertion)
/// and silently rejected in release builds.
pub struct PyBatchRoutingAdapter {
    pub py_router: Py<PyAny>,
    pub py_view: Option<Py<PyAny>>,
}

impl BatchRoutingStrategy for PyBatchRoutingAdapter {
    fn initialize(&mut self, view: &InstanceView<'_>) {
        Python::attach(|py| {
            self.py_view =
                Some(build_py_instance_view(py, view).expect("failed to build instance view dict"));
        });
    }

    fn route_batch(
        &mut self,
        requests: &[RequestId],
        vehicles: &[VehicleSnapshot],
        _view: &InstanceView<'_>,
    ) -> Vec<(RequestId, Option<i64>)> {
        Python::attach(|py| {
            let py_requests: Vec<i64> = requests.iter().map(|r| r.0).collect();
            let py_vehicles =
                build_py_vehicles(py, vehicles).expect("failed to build vehicles list");
            let py_view = self
                .py_view
                .as_ref()
                .expect("initialize() must be called before route_batch()")
                .clone_ref(py);
            let result = self
                .py_router
                .call_method1(py, "route_batch", (py_requests, py_vehicles, py_view))
                .expect("Python router.route_batch() raised an exception");
            let list = result
                .bind(py)
                .cast::<PyList>()
                .expect("route_batch() must return a list");

            list.iter()
                .map(|item| {
                    // Each item is a tuple (request_id: int, vehicle_id: int | None)
                    let tup = item
                        .cast::<pyo3::types::PyTuple>()
                        .expect("route_batch() items must be (int, int | None) tuples");
                    let rid = RequestId(
                        tup.get_item(0)
                            .expect("tuple index 0")
                            .extract::<i64>()
                            .expect("request_id must be int"),
                    );
                    let vid: Option<i64> = {
                        let v = tup.get_item(1).expect("tuple index 1");
                        if v.is_none() {
                            None
                        } else {
                            Some(v.extract::<i64>().expect("vehicle_id must be int or None"))
                        }
                    };
                    (rid, vid)
                })
                .collect()
        })
    }
}

// ---------------------------------------------------------------------------
// BatchComposableStrategy
// ---------------------------------------------------------------------------

/// A [`RustStrategy`] that uses slot-based batch routing with per-request scheduling.
///
/// Routing decisions are deferred until the end of each fixed-size time slot,
/// at which point all buffered requests are passed to the batch router at once.
/// After assignment, per-vehicle queues are managed identically to
/// [`super::composable::ComposableStrategy`].
pub struct BatchComposableStrategy {
    pub router: Box<dyn BatchRoutingStrategy>,
    pub scheduler: Box<dyn SchedulingStrategy>,
    /// Duration of each routing slot.
    pub slot_size: f32,
    /// Simulation time at which the current slot ends.
    pub next_slot_end: f32,
    /// Requests that have been released but not yet passed to the router.
    pub buffer: Vec<RequestId>,
    /// Requests already seen (released) to avoid double-buffering.
    pub seen: HashSet<RequestId>,
    /// Per-vehicle pending queue (assigned but not yet dispatched).
    pub queues: HashMap<i64, VecDeque<RequestId>>,
}

impl DispatchStrategy for BatchComposableStrategy {
    fn initialize(&mut self, view: &InstanceView<'_>) {
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
        // Defensive init for Python adapters: ensure router/scheduler have a
        // built instance view before being invoked.
        self.router.initialize(view);
        self.scheduler.initialize(view);
        let mut actions: Vec<SimAction> = Vec::new();

        // Phase 1: buffer newly-released, unrouted requests.
        let mut newly_released: Vec<RequestId> = state
            .released
            .iter()
            .filter(|rid| {
                !self.seen.contains(rid)
                    && !state.served.contains(rid)
                    && !state.rejected.contains(rid)
            })
            .copied()
            .collect();
        newly_released.sort_unstable_by_key(|r| r.0);
        for rid in newly_released {
            self.seen.insert(rid);
            self.buffer.push(rid);
        }

        // Phase 2: if slot boundary reached, route the entire buffer.
        if state.time >= self.next_slot_end && !self.buffer.is_empty() {
            // Sort buffer for determinism.
            self.buffer.sort_unstable_by_key(|r| r.0);
            let batch: Vec<RequestId> = self.buffer.drain(..).collect();

            let assignments = self.router.route_batch(&batch, &state.vehicles, view);

            // Validate coverage in debug mode.
            debug_assert_eq!(
                assignments.len(),
                batch.len(),
                "route_batch() must return one assignment per request"
            );

            for (rid, vid_opt) in assignments {
                match vid_opt {
                    Some(vid) => {
                        self.queues.entry(vid).or_default().push_back(rid);
                    }
                    None => {
                        actions.push(SimAction::Reject { request_id: rid });
                    }
                }
            }

            // Advance to next slot boundary.
            while self.next_slot_end <= state.time {
                self.next_slot_end += self.slot_size;
            }
        }

        // Phase 3: dispatch idle vehicles with queued work.
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

        // Phase 4: emit Wait if nothing to do but work remains.
        if actions.is_empty() {
            let has_queued = self.queues.values().any(|q| !q.is_empty());
            let has_buffered = !self.buffer.is_empty();
            let has_pending = !state.pending.is_empty();

            if has_queued || has_buffered || has_pending {
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

                // Also wake at the next slot boundary if buffer is non-empty.
                let next_slot = if has_buffered {
                    self.next_slot_end
                } else {
                    f32::INFINITY
                };

                let until = f32::min(next_slot, f32::min(next_vehicle_free, next_release));
                if until.is_finite() && until > state.time {
                    actions.push(SimAction::Wait { until });
                }
            }
        }

        actions
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Return a [`NativeDispatchStrategy`] using the batch composable strategy template.
///
/// Requests are accumulated within each `slot_size`-wide time window and
/// routed all at once when the window closes.  After assignment they enter
/// per-vehicle queues; `scheduler` picks the dispatch order.
///
/// `router` must implement:
/// ```python
/// def route_batch(
///     self,
///     requests: list[int],
///     vehicles: list[dict],
///     instance_view: dict,
/// ) -> list[tuple[int, int | None]]:
///     ...  # return one (request_id, vehicle_id | None) per input request
/// ```
///
/// `scheduler` must implement:
/// ```python
/// def schedule(
///     self,
///     vehicle: dict,
///     queue: list[int],
///     instance_view: dict,
/// ) -> int:
///     ...  # return a request_id from queue
/// ```
#[pyfunction]
pub fn batch_composable_strategy(
    router: Py<PyAny>,
    scheduler: Py<PyAny>,
    slot_size: f64,
) -> NativeDispatchStrategy {
    let slot_size_f32 = slot_size as f32;
    NativeDispatchStrategy {
        inner: Some(Box::new(BatchComposableStrategy {
            router: Box::new(PyBatchRoutingAdapter {
                py_router: router,
                py_view: None,
            }),
            scheduler: Box::new(crate::strategies::composable::PySchedulingAdapter {
                py_scheduler: scheduler,
                py_view: None,
            }),
            slot_size: slot_size_f32,
            next_slot_end: slot_size_f32,
            buffer: Vec::new(),
            seen: HashSet::new(),
            queues: HashMap::new(),
        })),
    }
}

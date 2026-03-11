use crate::hashmap::{Map as HashMap, Set as HashSet};
/// simulator.rs — The main PyO3-exposed `Simulator` struct and its impl blocks.
///
/// # Architecture
///
/// The hot path is fully Rust-native when a [`NativeDispatchStrategy`] is used:
///
/// ```text
/// Simulator::run()
///   └─ build_snapshot()           ← cheap: integer copies, no PyO3
///   └─ RustStrategy::next_events() ← pure Rust, no GIL
///   └─ execute_action(SimAction)  ← enum match, no PyO3
/// ```
///
/// Python strategies (any object with `next_events(state_dict)`) are supported
/// via [`PyStrategyAdapter`], which acquires the GIL, serialises the snapshot
/// to a Python dict, calls the Python method, and deserialises the returned
/// action objects — exactly the original behaviour, but now concentrated in one
/// place behind the [`RustStrategy`] trait.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BinaryHeap;

use crate::instance::{
    DispatchStrategy, Event, EventCallback, EventKind, InstanceLimits, InstanceView, MinEvent,
};
use crate::py_bridge::{extract_request, extract_vehicle_spec};
use crate::types::VehicleState;
use crate::types::{
    NativeDispatchStrategy, NativeEventCallback, RequestId, SimAction, SimulationSnapshot,
};

// ---------------------------------------------------------------------------
// Simulator — the main PyO3-exposed struct
// ---------------------------------------------------------------------------

#[pyclass]
pub struct Simulator {
    // Static instance data
    requests: HashMap<RequestId, crate::instance::Request>,
    vehicle_specs: Vec<crate::instance::VehicleSpec>,
    depot_id: RequestId,
    /// Pre-computed normalization scale factors — derived once from the static
    /// instance data and passed by copy into every `InstanceView`.
    instance_limits: InstanceLimits,

    // O(1) vehicle lookup by id
    vehicle_index: HashMap<i64, usize>,

    // Mutable simulation state
    time: f32,
    vehicles: Vec<VehicleState>,
    pending_requests: HashSet<RequestId>,
    served_requests: HashSet<RequestId>,
    rejected_requests: HashSet<RequestId>,
    /// Tracks which RequestIds are currently in some vehicle's route.
    /// Updated on dispatch; checked O(1) in execute_reject.
    being_served: HashSet<RequestId>,
    event_queue: BinaryHeap<MinEvent>,

    /// Non-depot requests sorted by ascending `release_time`.
    sorted_releases: Vec<(f32, RequestId)>,
    sorted_tw_latest: Vec<(f32, RequestId)>,
    next_deadline_idx: usize,

    /// Incrementally-maintained set of released request IDs.
    /// Updated in `process_next_event` by advancing `released_cursor` through
    /// `sorted_releases`. This lets snapshot construction borrow it as a reference
    /// instead of scanning all requests every tick.
    released_set: HashSet<RequestId>,
    /// Index into `sorted_releases` — all entries before this have been inserted
    /// into `released_set`.
    released_cursor: usize,

    // Strategy and callback — both behind trait objects.
    // Native implementations run with no GIL; Python ones acquire it inside
    // PyStrategyAdapter / PyCallbackAdapter.
    strategy: Box<dyn DispatchStrategy>,
    action_callback: Option<Box<dyn EventCallback>>,
}

#[pymethods]
impl Simulator {
    /// Create a new Simulator.
    ///
    /// Args:
    ///     instance: DVRPTWInstance Python object
    ///     strategy: either a [`NativeDispatchStrategy`] (zero-overhead, no GIL)
    ///               or any Python object implementing `next_events(state) -> list`
    ///     action_callback: optional callable `(time, action, auto) -> None`
    ///                      or a [`NativeCallbackWrapper`]
    #[new]
    #[pyo3(signature = (instance, strategy, action_callback=None))]
    pub fn new(
        _py: Python,
        instance: &Bound<PyAny>,
        strategy: &Bound<NativeDispatchStrategy>,
        action_callback: Option<&Bound<NativeEventCallback>>,
    ) -> PyResult<Self> {
        // Extract requests
        let py_requests = instance.getattr("requests")?;
        let py_requests_list = py_requests.cast::<PyList>()?;
        let mut requests: HashMap<RequestId, crate::instance::Request> = HashMap::default();
        for item in py_requests_list.iter() {
            let req = extract_request(&item)?;
            requests.insert(req.id, req);
        }

        // Extract vehicle specs
        let py_vehicles = instance.getattr("vehicles")?;
        let py_vehicles_list = py_vehicles.cast::<PyList>()?;
        let mut vehicle_specs: Vec<crate::instance::VehicleSpec> = Vec::new();
        for item in py_vehicles_list.iter() {
            vehicle_specs.push(extract_vehicle_spec(&item)?);
        }

        // Depot id
        let depot_id = RequestId(instance.getattr("depot_id")?.extract::<i64>()?);

        // Initialise vehicle runtime state (all start at depot, idle at t=0)
        let vehicles: Vec<VehicleState> = vehicle_specs
            .iter()
            .map(|vs| VehicleState {
                vehicle_id: vs.id,
                position: depot_id,
                current_load: 0.0,
                available_at: 0.0,
                route: Vec::new(),
                service_times: Vec::new(),
            })
            .collect();

        // O(1) vehicle index
        let vehicle_index: HashMap<i64, usize> = vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| (v.vehicle_id, i))
            .collect();

        // Pending = all non-depot requests
        let pending_requests: HashSet<RequestId> = requests
            .values()
            .filter(|r| !r.is_depot)
            .map(|r| r.id)
            .collect();

        // Pre-sort non-depot requests by release_time for O(log N) released-set
        // construction in build_snapshot (replaces the O(N) HashMap scan).
        let mut sorted_releases: Vec<(f32, RequestId)> = requests
            .values()
            .filter(|r| !r.is_depot)
            .map(|r| (r.release_time, r.id))
            .collect();
        sorted_releases
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut sorted_tw_latest: Vec<(f32, RequestId)> = requests
            .values()
            .filter(|r| !r.is_depot)
            .map(|r| (r.time_window.latest, r.id))
            .collect();
        sorted_tw_latest
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Pre-populate released_set for requests that are already available at t=0.
        let mut released_set: HashSet<RequestId> = HashSet::default();
        let mut released_cursor: usize = 0;
        while released_cursor < sorted_releases.len() && sorted_releases[released_cursor].0 <= 0.0 {
            released_set.insert(sorted_releases[released_cursor].1);
            released_cursor += 1;
        }

        // Schedule arrival events for every non-depot request
        let mut event_queue: BinaryHeap<MinEvent> = BinaryHeap::new();
        for req in requests.values() {
            if !req.is_depot {
                event_queue.push(MinEvent(Event {
                    time: req.release_time,
                    kind: EventKind::Arrival,
                    id: req.id.0,
                }));
            }
        }

        let mut strategy = strategy
            .extract::<PyRefMut<NativeDispatchStrategy>>()?
            .inner
            .take()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "NativeDispatchStrategy has already been consumed",
                )
            })?;

        // Compute normalization scale factors once from the static instance data.
        let instance_limits = InstanceLimits::from_data(&requests, &vehicle_specs);

        // Call the strategy initializer so native strategies (and adapters)
        // can cache instance data before the simulation loop starts.
        // This mirrors the documented DispatchStrategy::initialize hook.
        let init_view = InstanceView {
            requests: &requests,
            vehicles: &vehicle_specs,
            ascending_release: &sorted_releases,
            depot_id,
            limits: instance_limits,
        };
        strategy.initialize(&init_view);

        let action_callback = action_callback
            .map(|cb| cb.extract::<PyRefMut<NativeEventCallback>>())
            .transpose()?
            .map(|mut wrapper| {
                wrapper.inner.take().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "NativeCallbackWrapper has already been consumed",
                    )
                })
            })
            .transpose()?;
        Ok(Simulator {
            requests,
            vehicle_specs,
            depot_id,
            instance_limits,
            vehicle_index,
            time: 0.0_f32,
            vehicles,
            pending_requests,
            served_requests: HashSet::default(),
            rejected_requests: HashSet::default(),
            being_served: HashSet::default(),
            event_queue,
            sorted_releases,
            sorted_tw_latest,
            next_deadline_idx: 0,
            released_set,
            released_cursor,
            strategy,
            action_callback,
        })
    }

    /// Execute the simulation and return a dict:
    ///   {
    ///     "solution": {"routes": [[...], ...], "service_times": [[...], ...]},
    ///     "metrics":  {"total_travel_cost": float, "rejected": int}
    ///   }
    pub fn run(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        crate::bench::reset();
        let run_start = std::time::Instant::now();

        // Ensure the strategy has been initialised with a view that references
        // the simulator's instance data. Some callers construct the native
        // strategy externally and rely on the simulator to invoke
        // `initialize` before the first `next_events` call.
        self.strategy.initialize(&InstanceView {
            requests: &self.requests,
            vehicles: &self.vehicle_specs,
            ascending_release: &self.sorted_releases,
            depot_id: self.depot_id,
            limits: self.instance_limits,
        });

        loop {
            let any_busy = self.vehicles.iter().any(|v| v.available_at > self.time);
            if self.event_queue.is_empty() && !any_busy {
                break;
            }

            // Advance time, drain events, auto-reject expired requests
            let t0 = std::time::Instant::now();
            self.process_next_event()?;
            crate::bench::TIME_PROCESS_EVENT_NS.fetch_add(
                crate::bench::elapsed_ns(t0),
                std::sync::atomic::Ordering::Relaxed,
            );

            // Build snapshot (zero-copy: all fields are borrows into simulator state).
            let snapshot = SimulationSnapshot {
                time: self.time,
                pending: &self.pending_requests,
                served: &self.served_requests,
                rejected: &self.rejected_requests,
                released: &self.released_set,
                vehicles: &self.vehicles,
            };

            // Ask strategy for actions — may or may not acquire GIL depending on
            // whether the strategy is native or Python-backed.
            let t0 = std::time::Instant::now();
            let actions = self.strategy.next_events(
                &snapshot,
                &InstanceView {
                    requests: &self.requests,
                    vehicles: &self.vehicle_specs,
                    ascending_release: &self.sorted_releases,
                    depot_id: self.depot_id,
                    limits: self.instance_limits,
                },
            );
            crate::bench::TIME_STRATEGY_NS.fetch_add(
                crate::bench::elapsed_ns(t0),
                std::sync::atomic::Ordering::Relaxed,
            );

            let t0 = std::time::Instant::now();
            for action in actions {
                self.execute_action(&action)?;
                // Fire callback after each action (auto=false)
                if let Some(cb) = &self.action_callback {
                    cb.on_action(self.time, &action, false);
                }
            }
            crate::bench::TIME_EXECUTE_NS.fetch_add(
                crate::bench::elapsed_ns(t0),
                std::sync::atomic::Ordering::Relaxed,
            );
            crate::bench::TICK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let total_ns = crate::bench::elapsed_ns(run_start);
        crate::bench::print_summary(total_ns);

        self.finalize_result(py)
    }
}

// ---------------------------------------------------------------------------
// Internal simulation logic (not exposed to Python)
// ---------------------------------------------------------------------------

impl Simulator {
    fn process_next_event(&mut self) -> PyResult<()> {
        let t0 = std::time::Instant::now();
        let queue_time = self
            .event_queue
            .peek()
            .map(|e| e.0.time)
            .unwrap_or(f32::INFINITY);

        let vehicle_time = self
            .vehicles
            .iter()
            .filter(|v| v.available_at > self.time)
            .map(|v| v.available_at)
            .fold(f32::INFINITY, f32::min);

        let next_time = queue_time.min(vehicle_time);
        if next_time == f32::INFINITY {
            return Ok(());
        }

        self.time = next_time;
        crate::bench::TIME_PNE_NEXT_TIME_NS.fetch_add(
            crate::bench::elapsed_ns(t0),
            std::sync::atomic::Ordering::Relaxed,
        );

        // Drain all events exactly at this time
        let t0 = std::time::Instant::now();
        while let Some(top) = self.event_queue.peek() {
            if top.0.time <= self.time {
                self.event_queue.pop();
            } else {
                break;
            }
        }
        crate::bench::TIME_PNE_EVENT_DRAIN_NS.fetch_add(
            crate::bench::elapsed_ns(t0),
            std::sync::atomic::Ordering::Relaxed,
        );

        let t0 = std::time::Instant::now();
        self.auto_reject_closed_requests()?;
        crate::bench::TIME_PNE_AUTO_REJECT_NS.fetch_add(
            crate::bench::elapsed_ns(t0),
            std::sync::atomic::Ordering::Relaxed,
        );

        // Advance released cursor to include all newly-released requests.
        let t0 = std::time::Instant::now();
        while self.released_cursor < self.sorted_releases.len()
            && self.sorted_releases[self.released_cursor].0 <= self.time
        {
            let id = self.sorted_releases[self.released_cursor].1;
            self.released_set.insert(id);
            self.released_cursor += 1;
        }
        crate::bench::TIME_PNE_RELEASE_CURSOR_NS.fetch_add(
            crate::bench::elapsed_ns(t0),
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(())
    }

    fn auto_reject_closed_requests(&mut self) -> PyResult<()> {
        // from self.sorted_tw_latest, find all requests with latest <= self.time and remove them
        // from pending_requests using binary search
        while self.next_deadline_idx < self.sorted_tw_latest.len()
            && self.sorted_tw_latest[self.next_deadline_idx].0 <= self.time
        {
            let rid = self.sorted_tw_latest[self.next_deadline_idx].1;
            self.next_deadline_idx += 1;

            if self.pending_requests.remove(&rid) {
                self.rejected_requests.insert(rid);

                if let Some(cb) = &self.action_callback {
                    let action = SimAction::Reject { request_id: rid };
                    cb.on_action(self.time, &action, true);
                }
            }
        }
        Ok(())
    }

    fn execute_action(&mut self, action: &SimAction) -> PyResult<()> {
        match action {
            SimAction::Dispatch { vehicle_id, dest } => {
                self.execute_dispatch(*vehicle_id, *dest)?;
            }
            SimAction::Wait { until } => {
                self.execute_wait(*until)?;
            }
            SimAction::Reject { request_id } => {
                self.execute_reject(*request_id)?;
            }
        }
        Ok(())
    }

    fn execute_dispatch(&mut self, vehicle_id: i64, destination: RequestId) -> PyResult<()> {
        // O(1) lookup via vehicle_index
        let vehicle_idx = self
            .vehicle_index
            .get(&vehicle_id)
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Vehicle {vehicle_id} not found"))
            })?;

        if self.vehicles[vehicle_idx].available_at > self.time {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vehicle {vehicle_id} is not idle at time {}",
                self.time
            )));
        }

        let from_pos = self.vehicles[vehicle_idx].position;
        let distance = {
            let from_req = self.requests.get(&from_pos).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Request {:?} not found", from_pos))
            })?;
            let dest_req = self.requests.get(&destination).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Request {:?} not found",
                    destination
                ))
            })?;
            from_req.distance_to(dest_req)
        };

        // Find the matching VehicleSpec index (specs are in construction order; use vehicle_index)
        let spec_idx = self
            .vehicle_specs
            .iter()
            .position(|vs| vs.id == vehicle_id)
            .unwrap_or(0);

        let travel_time = self.vehicle_specs[spec_idx].travel_time(distance);
        let arrival_time = self.time + travel_time;

        let (tw_earliest, tw_latest, service_time, demand, is_depot, capacity) = {
            let dest_req = self.requests.get(&destination).unwrap();
            (
                dest_req.time_window.earliest,
                dest_req.time_window.latest,
                dest_req.service_time,
                dest_req.demand,
                dest_req.is_depot,
                self.vehicle_specs[spec_idx].capacity,
            )
        };

        let service_start = f32::max(arrival_time, tw_earliest);
        if service_start > tw_latest {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot serve request {:?} with vehicle {vehicle_id}: \
                 service_start={service_start} > latest={tw_latest}",
                destination
            )));
        }

        let v = &mut self.vehicles[vehicle_idx];
        v.position = destination;
        v.route.push(destination);
        v.service_times.push(service_start);
        v.available_at = service_start + service_time;

        if is_depot {
            v.current_load = 0.0;
        } else {
            v.current_load += demand;
            if v.current_load > capacity {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Dispatch {vehicle_id} → {:?}: exceeds capacity {capacity}",
                    destination
                )));
            }
            if self.pending_requests.remove(&destination) {
                self.served_requests.insert(destination);
                self.being_served.insert(destination);
            }
        }

        Ok(())
    }

    fn execute_wait(&mut self, until_time: f32) -> PyResult<()> {
        if until_time <= self.time {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "WaitEvent until_time={until_time} must be > current time={}",
                self.time
            )));
        }
        self.event_queue.push(MinEvent(Event {
            time: until_time,
            kind: EventKind::Wake,
            id: -1,
        }));
        Ok(())
    }

    fn execute_reject(&mut self, request_id: RequestId) -> PyResult<()> {
        if !self.pending_requests.contains(&request_id) {
            return Ok(()); // already gone
        }
        // O(1) check via being_served set
        if self.being_served.contains(&request_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot reject request {:?}: being served by a vehicle",
                request_id
            )));
        }
        self.pending_requests.remove(&request_id);
        self.rejected_requests.insert(request_id);
        Ok(())
    }

    fn finalize_result(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Build solution dict
        let routes = PyList::empty(py);
        let service_times_list = PyList::empty(py);
        for v in &self.vehicles {
            let route_ids: Vec<i64> = v.route.iter().map(|r| r.0).collect();
            routes.append(PyList::new(py, &route_ids)?)?;
            let st: Vec<f64> = v.service_times.iter().map(|&t| t as f64).collect();
            service_times_list.append(PyList::new(py, &st)?)?;
        }

        // Compute total travel cost
        let mut total_cost = 0.0_f32;
        let depot = self.requests.get(&self.depot_id).unwrap();
        for v in &self.vehicles {
            if v.route.is_empty() {
                continue;
            }
            let first = self.requests.get(&v.route[0]).unwrap();
            total_cost += depot.distance_to(first);
            for i in 0..v.route.len() - 1 {
                let a = self.requests.get(&v.route[i]).unwrap();
                let b = self.requests.get(&v.route[i + 1]).unwrap();
                total_cost += a.distance_to(b);
            }
            let last = self.requests.get(v.route.last().unwrap()).unwrap();
            total_cost += last.distance_to(depot);
        }

        let rejected = self.rejected_requests.len() as i64;

        let solution = PyDict::new(py);
        solution.set_item("routes", routes)?;
        solution.set_item("service_times", service_times_list)?;

        let metrics = PyDict::new(py);
        metrics.set_item("total_travel_cost", total_cost as f64)?;
        metrics.set_item("rejected", rejected)?;

        let result = PyDict::new(py);
        result.set_item("solution", solution)?;
        result.set_item("metrics", metrics)?;

        Ok(result.into())
    }
}

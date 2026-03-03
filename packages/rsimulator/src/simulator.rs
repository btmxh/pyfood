/// simulator.rs — The main PyO3-exposed `Simulator` struct and its impl blocks.
///
/// # Architecture
///
/// The hot path is fully Rust-native when a [`NativeStrategyWrapper`] is used:
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
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::instance::{
    Event, EventKind, InstanceView, MinEvent, RustCallback, RustStrategy, VehicleState,
};
use crate::py_bridge::{
    PyCallbackAdapter, PyStrategyAdapter, extract_request, extract_vehicle_spec,
};
use crate::types::{
    NativeCallbackWrapper, NativeStrategyWrapper, RequestId, SimAction, SimulationSnapshot,
    VehicleSnapshot,
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

    // Strategy and callback — both behind trait objects.
    // Native implementations run with no GIL; Python ones acquire it inside
    // PyStrategyAdapter / PyCallbackAdapter.
    strategy: Box<dyn RustStrategy>,
    action_callback: Option<Box<dyn RustCallback>>,
}

#[pymethods]
impl Simulator {
    /// Create a new Simulator.
    ///
    /// Args:
    ///     instance: DVRPTWInstance Python object
    ///     strategy: either a [`NativeStrategyWrapper`] (zero-overhead, no GIL)
    ///               or any Python object implementing `next_events(state) -> list`
    ///     action_callback: optional callable `(time, action, auto) -> None`
    ///                      or a [`NativeCallbackWrapper`]
    #[new]
    #[pyo3(signature = (instance, strategy, action_callback=None))]
    pub fn new(
        _py: Python,
        instance: &Bound<PyAny>,
        strategy: &Bound<PyAny>,
        action_callback: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        // Validate instance first (mirrors Python Simulator.__init__)
        instance.call_method0("validate")?;

        // Extract requests
        let py_requests = instance.getattr("requests")?;
        let py_requests_list = py_requests.cast::<PyList>()?;
        let mut requests: HashMap<RequestId, crate::instance::Request> = HashMap::new();
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
        let py_depot_ids = instance.getattr("depot_ids")?;
        let py_depot_ids_list = py_depot_ids.cast::<PyList>()?;
        let depot_id = RequestId(py_depot_ids_list.get_item(0)?.extract::<i64>()?);

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

        // Resolve strategy: native wrapper (no-GIL) or Python object (GIL adapter)
        let mut strategy_box: Box<dyn RustStrategy> =
            if let Ok(mut wrapper) = strategy.extract::<PyRefMut<NativeStrategyWrapper>>() {
                wrapper.inner.take().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "NativeStrategyWrapper has already been consumed",
                    )
                })?
            } else {
                Box::new(PyStrategyAdapter {
                    py_strategy: strategy.clone().unbind(),
                })
            };

        // Give native strategies access to instance data (no-op for Python adapters).
        strategy_box.initialize(&InstanceView {
            requests: &requests,
            vehicles: &vehicle_specs,
            depot_id,
        });

        // Resolve callback: native wrapper or Python callable
        let callback_box: Option<Box<dyn RustCallback>> = match action_callback {
            None => None,
            Some(cb) => {
                if let Ok(mut wrapper) = cb.extract::<PyRefMut<NativeCallbackWrapper>>() {
                    Some(wrapper.inner.take().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "NativeCallbackWrapper has already been consumed",
                        )
                    })?)
                } else {
                    Some(Box::new(PyCallbackAdapter {
                        py_callback: cb.clone().unbind(),
                    }))
                }
            }
        };

        Ok(Simulator {
            requests,
            vehicle_specs,
            depot_id,
            vehicle_index,
            time: 0.0_f32,
            vehicles,
            pending_requests,
            served_requests: HashSet::new(),
            rejected_requests: HashSet::new(),
            being_served: HashSet::new(),
            event_queue,
            strategy: strategy_box,
            action_callback: callback_box,
        })
    }

    /// Execute the simulation and return a dict:
    ///   {
    ///     "solution": {"routes": [[...], ...], "service_times": [[...], ...]},
    ///     "metrics":  {"total_travel_cost": float, "rejected": int}
    ///   }
    pub fn run(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        loop {
            let any_busy = self.vehicles.iter().any(|v| v.available_at > self.time);
            if self.event_queue.is_empty() && !any_busy {
                break;
            }

            // Advance time, drain events, auto-reject expired requests
            self.process_next_event()?;

            // Build snapshot (cheap — no PyO3 allocations for native strategies)
            let snapshot = self.build_snapshot();

            // Ask strategy for actions — may or may not acquire GIL depending on
            // whether the strategy is native or Python-backed.
            let actions = self.strategy.next_events(
                &snapshot,
                &InstanceView {
                    requests: &self.requests,
                    vehicles: &self.vehicle_specs,
                    depot_id: self.depot_id,
                },
            );

            for action in actions {
                self.execute_action(&action)?;
                // Fire callback after each action (auto=false)
                if let Some(cb) = &self.action_callback {
                    cb.on_action(self.time, &action, false);
                }
            }
        }

        self.finalize_result(py)
    }
}

// ---------------------------------------------------------------------------
// Internal simulation logic (not exposed to Python)
// ---------------------------------------------------------------------------

impl Simulator {
    fn process_next_event(&mut self) -> PyResult<()> {
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

        // Drain all events exactly at this time
        while let Some(top) = self.event_queue.peek() {
            if top.0.time <= self.time {
                self.event_queue.pop();
            } else {
                break;
            }
        }

        self.auto_reject_closed_requests()?;
        Ok(())
    }

    fn auto_reject_closed_requests(&mut self) -> PyResult<()> {
        let to_reject: Vec<RequestId> = self
            .pending_requests
            .iter()
            .filter(|&&rid| {
                self.requests
                    .get(&rid)
                    .map(|r| self.time > r.time_window.latest)
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        for rid in to_reject {
            self.pending_requests.remove(&rid);
            self.rejected_requests.insert(rid);
            if let Some(cb) = &self.action_callback {
                let action = SimAction::Reject { request_id: rid };
                cb.on_action(self.time, &action, true);
            }
        }
        Ok(())
    }

    /// Build a [`SimulationSnapshot`] from current state.
    ///
    /// This is O(vehicles × route_len) with only integer copies — no PyO3
    /// allocations.  When using a native strategy the hot path never touches
    /// the GIL.
    fn build_snapshot(&self) -> SimulationSnapshot {
        let vehicles: Vec<VehicleSnapshot> = self
            .vehicles
            .iter()
            .map(|v| VehicleSnapshot {
                vehicle_id: v.vehicle_id,
                position: v.position,
                current_load: v.current_load,
                available_at: v.available_at,
                route: v.route.clone(),
                service_times: v.service_times.clone(),
            })
            .collect();

        let released: HashSet<RequestId> = self
            .requests
            .values()
            .filter(|r| !r.is_depot && r.release_time <= self.time)
            .map(|r| r.id)
            .collect();

        SimulationSnapshot {
            time: self.time,
            pending: self.pending_requests.clone(),
            served: self.served_requests.clone(),
            rejected: self.rejected_requests.clone(),
            released,
            vehicles,
        }
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

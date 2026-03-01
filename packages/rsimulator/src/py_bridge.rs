/// py_bridge.rs — All PyO3 glue code: Python ↔ Rust adapters and helpers.
///
/// # Contents
/// - [`PyStrategyAdapter`]  — wraps a Python strategy object as `Box<dyn RustStrategy>`
/// - [`PyCallbackAdapter`]  — wraps a Python callable as `Box<dyn RustCallback>`
/// - [`snapshot_to_py_dict`] — serialises `SimulationSnapshot` to a Python dict
/// - [`sim_action_to_py`]   — converts `SimAction` to a Python action object
/// - [`extract_py_actions`] — deserialises Python action list to `Vec<SimAction>`
/// - [`extract_request`]    — extracts a `Request` from a Python object
/// - [`extract_vehicle_spec`] — extracts a `VehicleSpec` from a Python object
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::instance::{InstanceView, Request, RustCallback, RustStrategy, TimeWindow, VehicleSpec};
use crate::types::{RequestId, SimAction, SimulationSnapshot};

// ---------------------------------------------------------------------------
// PyStrategyAdapter — wraps Py<PyAny> as Box<dyn RustStrategy>
// ---------------------------------------------------------------------------

/// Adapts a Python `DispatchingStrategy` to the [`RustStrategy`] trait.
///
/// This is the only place that acquires the GIL and performs PyO3
/// serialisation.  All simulation logic remains in Rust; Python is called only
/// for the strategy decision.
///
/// The Python strategy receives a state dict (same format as before) and
/// returns a list of `DispatchEvent | WaitEvent | RejectEvent` Python objects
/// or plain dicts.
pub struct PyStrategyAdapter {
    pub py_strategy: Py<PyAny>,
}

impl RustStrategy for PyStrategyAdapter {
    fn next_events(
        &mut self,
        state: &SimulationSnapshot,
        _view: &InstanceView<'_>,
    ) -> Vec<SimAction> {
        Python::attach(|py| {
            let state_dict = snapshot_to_py_dict(py, state)
                .expect("failed to build state dict for Python strategy");
            let result = self
                .py_strategy
                .call_method1(py, "next_events", (state_dict,))
                .expect("Python strategy.next_events() raised an exception");
            extract_py_actions(py, result).expect("failed to extract actions from Python strategy")
        })
    }
}

// ---------------------------------------------------------------------------
// PyCallbackAdapter — wraps Py<PyAny> as Box<dyn RustCallback>
// ---------------------------------------------------------------------------

/// Adapts a Python callable `(time, action, auto) -> None` to [`RustCallback`].
pub struct PyCallbackAdapter {
    pub py_callback: Py<PyAny>,
}

impl RustCallback for PyCallbackAdapter {
    fn on_action(&self, time: f64, action: &SimAction, auto: bool) {
        Python::attach(|py| {
            let py_action = sim_action_to_py(py, action)
                .expect("failed to convert SimAction to Python object for callback");
            self.py_callback
                .call1(py, (time, py_action, auto))
                .expect("Python action callback raised an exception");
        });
    }
}

// ---------------------------------------------------------------------------
// Snapshot ↔ Python dict conversion (used only by PyStrategyAdapter)
// ---------------------------------------------------------------------------

/// Serialise a [`SimulationSnapshot`] to a Python dict.
///
/// This mirrors the old `build_py_state` but is only called inside
/// [`PyStrategyAdapter`] — native strategies never hit this code.
pub fn snapshot_to_py_dict(py: Python, state: &SimulationSnapshot) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);

    d.set_item("time", state.time)?;

    let pending = PySet::new(py, state.pending.iter().map(|r| r.0))?;
    d.set_item("pending_requests", pending)?;

    let served = PySet::new(py, state.served.iter().map(|r| r.0))?;
    d.set_item("served_requests", served)?;

    let rejected = PySet::new(py, state.rejected.iter().map(|r| r.0))?;
    d.set_item("rejected_requests", rejected)?;

    // vehicles: list of dicts
    let vehicles_list = PyList::empty(py);
    for v in &state.vehicles {
        let vd = PyDict::new(py);
        vd.set_item("vehicle_id", v.vehicle_id)?;
        vd.set_item("position", v.position.0)?;
        vd.set_item("current_load", v.current_load)?;
        vd.set_item("available_at", v.available_at)?;
        let route_ids: Vec<i64> = v.route.iter().map(|r| r.0).collect();
        vd.set_item("route", PyList::new(py, &route_ids)?)?;
        vd.set_item("service_times", PyList::new(py, &v.service_times)?)?;
        vehicles_list.append(vd)?;
    }
    d.set_item("vehicles", vehicles_list)?;

    // released_requests: dict[int -> dict] with full request data.
    // The Python _RustStrategyAdapter ignores these dict values and fetches
    // from the Python instance, but we include them for forward compatibility
    // and for strategies that read them directly.
    //
    // Note: we don't have the full Request data in the snapshot (only IDs), so
    // we emit a minimal sentinel dict.  The _RustStrategyAdapter in rust.py
    // uses the IDs to look up the Python Request objects from the instance.
    let released = PyDict::new(py);
    for rid in &state.released {
        // Value is a placeholder int; _RustStrategyAdapter reads only the keys.
        released.set_item(rid.0, rid.0)?;
    }
    d.set_item("released_requests", released)?;

    Ok(d.into())
}

/// Convert a [`SimAction`] to the Python action object the existing Python
/// adapters and callbacks expect.
///
/// Auto-reject actions (fired by the simulator, not the strategy) produce a
/// plain dict `{"type": "reject", "request_id": N}` — matching the original
/// Rust behaviour that `_RustCallbackAdapter` handles.  Strategy actions
/// produce the proper Python dataclass instances so callbacks receive typed
/// objects regardless of which backend fired them.
pub fn sim_action_to_py(py: Python, action: &SimAction) -> PyResult<Py<PyAny>> {
    // Import the dataclass types from dvrptw.simulator.events
    let events_mod = py.import("dvrptw.simulator.events")?;

    match action {
        SimAction::Dispatch { vehicle_id, dest } => {
            let cls = events_mod.getattr("DispatchEvent")?;
            Ok(cls.call1((*vehicle_id, dest.0))?.unbind())
        }
        SimAction::Wait { until } => {
            let cls = events_mod.getattr("WaitEvent")?;
            Ok(cls.call1((*until,))?.unbind())
        }
        SimAction::Reject { request_id } => {
            // Use a plain dict for auto-rejects (matched by _RustCallbackAdapter);
            // the caller sets auto=true for those.  For strategy-originated rejects
            // we also emit a proper RejectEvent.
            let cls = events_mod.getattr("RejectEvent")?;
            Ok(cls.call1((request_id.0,))?.unbind())
        }
    }
}

/// Extract a list of [`SimAction`]s from the Python object returned by
/// `strategy.next_events(state)`.
///
/// Accepts both Python dataclass instances (DispatchEvent, WaitEvent,
/// RejectEvent) and plain dicts with a "type" key.
pub fn extract_py_actions(py: Python, obj: Py<PyAny>) -> PyResult<Vec<SimAction>> {
    let list = obj.bind(py).cast::<PyList>()?;
    let mut actions = Vec::with_capacity(list.len());

    for item in list.iter() {
        let class_name = item
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default();

        let action = match class_name.as_str() {
            "DispatchEvent" => {
                let vehicle_id: i64 = item.getattr("vehicle_id")?.extract()?;
                let dest: i64 = item.getattr("destination_node")?.extract()?;
                SimAction::Dispatch {
                    vehicle_id,
                    dest: RequestId(dest),
                }
            }
            "WaitEvent" => {
                let until: f64 = item.getattr("until_time")?.extract()?;
                SimAction::Wait { until }
            }
            "RejectEvent" => {
                let rid: i64 = item.getattr("request_id")?.extract()?;
                SimAction::Reject {
                    request_id: RequestId(rid),
                }
            }
            _ if item.is_instance_of::<PyDict>() => {
                let d = item.cast::<PyDict>()?;
                let kind: String = d
                    .get_item("type")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("action dict missing 'type' key")
                    })?
                    .extract()?;
                match kind.as_str() {
                    "dispatch" => {
                        let vid: i64 = d.get_item("vehicle_id")?.unwrap().extract()?;
                        let dest: i64 = d.get_item("destination_node")?.unwrap().extract()?;
                        SimAction::Dispatch {
                            vehicle_id: vid,
                            dest: RequestId(dest),
                        }
                    }
                    "wait" => {
                        let t: f64 = d.get_item("until_time")?.unwrap().extract()?;
                        SimAction::Wait { until: t }
                    }
                    "reject" => {
                        let rid: i64 = d.get_item("request_id")?.unwrap().extract()?;
                        SimAction::Reject {
                            request_id: RequestId(rid),
                        }
                    }
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown action type in dict: {other}"
                        )));
                    }
                }
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown action class: {other}"
                )));
            }
        };
        actions.push(action);
    }

    Ok(actions)
}

// ---------------------------------------------------------------------------
// Extraction helpers (called once at construction time, not in hot path)
// ---------------------------------------------------------------------------

pub fn extract_request(obj: &Bound<PyAny>) -> PyResult<Request> {
    let id: i64 = obj.getattr("id")?.extract()?;
    let position = obj.getattr("position")?;
    let pos_tuple = position.cast::<PyTuple>()?;
    let x: f64 = pos_tuple.get_item(0)?.extract()?;
    let y: f64 = pos_tuple.get_item(1)?.extract()?;
    let demand: f64 = obj.getattr("demand")?.extract()?;
    let tw = obj.getattr("time_window")?;
    let earliest: f64 = tw.getattr("earliest")?.extract()?;
    let latest: f64 = tw.getattr("latest")?.extract()?;
    let service_time: f64 = obj.getattr("service_time")?.extract()?;
    let release_time: f64 = obj.getattr("release_time")?.extract()?;
    let is_depot: bool = obj.getattr("is_depot")?.extract()?;
    Ok(Request {
        id: RequestId(id),
        x,
        y,
        demand,
        time_window: TimeWindow { earliest, latest },
        service_time,
        release_time,
        is_depot,
    })
}

pub fn extract_vehicle_spec(obj: &Bound<PyAny>) -> PyResult<VehicleSpec> {
    let id: i64 = obj.getattr("id")?.extract()?;
    let capacity: f64 = obj.getattr("capacity")?.extract()?;
    let speed: f64 = obj.getattr("speed")?.extract()?;
    Ok(VehicleSpec {
        id,
        capacity,
        speed,
    })
}

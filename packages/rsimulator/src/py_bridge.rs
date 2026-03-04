use std::collections::HashSet;

/// py_bridge.rs — All PyO3 glue code: Python ↔ Rust adapters and helpers.
///
/// # Contents
/// - [`PyStrategyAdapter`]  — wraps a Python strategy object as `Box<dyn RustStrategy>`
/// - [`PyCallbackAdapter`]  — wraps a Python callable as `Box<dyn RustCallback>`
/// - [`snapshot_to_py_dict`] — serialises `SimulationSnapshot` to a Python dict
/// - [`sim_action_to_py_dict`]   — converts `SimAction` to a Python dict
/// - [`extract_py_actions`] — deserialises Python action list to `Vec<SimAction>`
/// - [`extract_request`]    — extracts a `Request` from a Python object
/// - [`extract_vehicle_spec`] — extracts a `VehicleSpec` from a Python object
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::instance::{
    DispatchStrategy, EventCallback, InstanceView, Request, TimeWindow, VehicleSpec,
};
use crate::types::{RequestId, SimAction, SimulationSnapshot};
use crate::{NativeDispatchStrategy, NativeEventCallback};

// ---------------------------------------------------------------------------
// PyStrategyAdapter — wraps Py<PyAny> as Box<dyn RustStrategy>
// ---------------------------------------------------------------------------

/// Adapts a Python `DispatchStrategy` to the [`RustStrategy`] trait.
///
/// This is the only place that acquires the GIL and performs PyO3
/// serialisation.  All simulation logic remains in Rust; Python is called only
/// for the strategy decision.
///
/// The Python strategy receives a state dict (same format as before) and
/// returns a list of `DispatchEvent | WaitEvent | RejectEvent` Python objects
/// or plain dicts.
pub struct PyDispatchStrategyAdapter {
    pub py_strategy: Py<PyAny>,
}

impl PyDispatchStrategyAdapter {
    pub fn new(py_strategy: impl Into<Py<PyAny>>) -> Self {
        Self {
            py_strategy: py_strategy.into(),
        }
    }
}

impl DispatchStrategy for PyDispatchStrategyAdapter {
    fn next_events(
        &mut self,
        state: &SimulationSnapshot,
        view: &InstanceView<'_>,
    ) -> Vec<SimAction> {
        Python::attach(|py| {
            let state_dict = snapshot_to_py_dict(py, state)
                .expect("failed to build state dict for Python strategy");
            let mut available_ids = HashSet::new();
            available_ids.extend(&state.pending);
            available_ids.extend(&state.served);
            available_ids.extend(&state.rejected);

            let view_dict = instance_view_to_py_dict(py, view, available_ids)
                .expect("failed to build view dict for Python strategy");
            let result = self
                .py_strategy
                .call_method1(py, "next_events", (state_dict, view_dict))
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

impl PyCallbackAdapter {
    pub fn new(py_callback: impl Into<Py<PyAny>>) -> Self {
        Self {
            py_callback: py_callback.into(),
        }
    }
}

impl EventCallback for PyCallbackAdapter {
    fn on_action(&self, time: f32, action: &SimAction, auto: bool) {
        Python::attach(|py| {
            let py_action = sim_action_to_py_dict(py, action)
                .expect("failed to convert SimAction to Python object for callback");
            self.py_callback
                .call1(py, (time as f64, py_action, auto))
                .expect("Python action callback raised an exception");
        });
    }
}

// ---------------------------------------------------------------------------
// InstanceView ↔ Python dict conversion (used only by PyStrategyAdapter)
// ---------------------------------------------------------------------------

pub fn instance_view_to_py_dict(
    py: Python,
    view: &InstanceView,
    available_requests: impl IntoIterator<Item = RequestId>,
) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);

    let requests_list = PyDict::new(py);
    for id in available_requests {
        let r = view
            .get(id)
            .expect("request ID in snapshot not found in instance view");
        let rd = PyDict::new(py);
        rd.set_item("id", r.id.0)?;
        rd.set_item("position", (r.x as f64, r.y as f64))?;
        rd.set_item("demand", r.demand as f64)?;
        let tw = PyDict::new(py);
        tw.set_item("earliest", r.time_window.earliest as f64)?;
        tw.set_item("latest", r.time_window.latest as f64)?;
        rd.set_item("time_window", tw)?;
        rd.set_item("service_time", r.service_time as f64)?;
        rd.set_item("release_time", r.release_time as f64)?;
        rd.set_item("is_depot", r.is_depot)?;

        requests_list.set_item(id.0, rd)?;
    }
    d.set_item("released_requests", requests_list)?;

    let vehicles_list = PyList::empty(py);
    for v in view.vehicle_specs() {
        let vd = PyDict::new(py);
        vd.set_item("id", v.id)?;
        vd.set_item("capacity", v.capacity as f64)?;
        vd.set_item("speed", v.speed as f64)?;
        vehicles_list.append(vd)?;
    }
    d.set_item("vehicles", vehicles_list)?;

    let depot_id = view.depot_id().0;
    d.set_item("depot_id", depot_id)?;

    Ok(d.into())
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

    d.set_item("time", state.time as f64)?;

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
        vd.set_item("current_load", v.current_load as f64)?;
        vd.set_item("available_at", v.available_at as f64)?;
        let route_ids: Vec<i64> = v.route.iter().map(|r| r.0).collect();
        vd.set_item("route", PyList::new(py, &route_ids)?)?;
        let st: Vec<f64> = v.service_times.iter().map(|&t| t as f64).collect();
        vd.set_item("service_times", PyList::new(py, &st)?)?;
        vehicles_list.append(vd)?;
    }
    d.set_item("vehicles", vehicles_list)?;

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
pub fn sim_action_to_py_dict(py: Python, action: &SimAction) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match action {
        SimAction::Dispatch { vehicle_id, dest } => {
            d.set_item("type", "dispatch")?;
            d.set_item("vehicle_id", *vehicle_id)?;
            d.set_item("destination_node", dest.0)?;
        }
        SimAction::Wait { until } => {
            d.set_item("type", "wait")?;
            d.set_item("until_time", *until as f64)?;
        }
        SimAction::Reject { request_id } => {
            d.set_item("type", "reject")?;
            d.set_item("request_id", request_id.0)?;
        }
    };
    Ok(d.into())
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
                SimAction::Wait {
                    until: until as f32,
                }
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
                        SimAction::Wait { until: t as f32 }
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
        x: x as f32,
        y: y as f32,
        demand: demand as f32,
        time_window: TimeWindow {
            earliest: earliest as f32,
            latest: latest as f32,
        },
        service_time: service_time as f32,
        release_time: release_time as f32,
        is_depot,
    })
}

pub fn extract_vehicle_spec(obj: &Bound<PyAny>) -> PyResult<VehicleSpec> {
    let id: i64 = obj.getattr("id")?.extract()?;
    let capacity: f64 = obj.getattr("capacity")?.extract()?;
    let speed: f64 = obj.getattr("speed")?.extract()?;
    Ok(VehicleSpec {
        id,
        capacity: capacity as f32,
        speed: speed as f32,
    })
}

#[pyfunction]
pub fn python_dispatch_strategy(py_strategy: Py<PyAny>) -> NativeDispatchStrategy {
    NativeDispatchStrategy::new(PyDispatchStrategyAdapter::new(py_strategy))
}

#[pyfunction]
pub fn python_event_callback(py_callback: Py<PyAny>) -> NativeEventCallback {
    NativeEventCallback::new(PyCallbackAdapter::new(py_callback))
}

/// strategies/ — Native Rust dispatching strategy implementations.
///
/// # Modules
/// - [`greedy`]     — greedy dispatch in ascending request-ID order
/// - [`composable`] — template combining per-request router + scheduler
/// - [`batch`]      — template combining slot-based batch router + scheduler
///
/// # Sub-traits
/// - [`RoutingStrategy`]      — assign a single released request to a vehicle (or reject)
/// - [`SchedulingStrategy`]   — choose the next request for an idle vehicle
/// - [`BatchRoutingStrategy`] — assign a whole batch of requests at once
///
/// # Shared helpers
/// Python dict builders for [`crate::types::VehicleSnapshot`] and
/// [`crate::instance::InstanceView`] used by all Python-bridged sub-strategies.
pub mod batch;
pub mod composable;
pub mod greedy;

pub use batch::batch_composable_strategy;
pub use composable::composable_strategy;
pub use greedy::greedy_strategy;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::instance::InstanceView;
use crate::types::{RequestId, VehicleSnapshot};

// ---------------------------------------------------------------------------
// RoutingStrategy and SchedulingStrategy — composable strategy sub-traits
// ---------------------------------------------------------------------------

/// Decides which vehicle to assign a released request to, or rejects it.
///
/// Called exactly once per request, on the tick it first appears in
/// `SimulationSnapshot::released`.
pub trait RoutingStrategy: Send + Sync {
    /// Called once before the simulation loop starts.
    /// Override to cache instance data needed for routing decisions.
    fn initialize(&mut self, _view: &InstanceView<'_>) {}

    /// Return `Some(vehicle_id)` to assign `request` to that vehicle's queue,
    /// or `None` to reject it immediately.
    fn route(
        &mut self,
        request: RequestId,
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
    ) -> Option<i64>;
}

/// Decides which queued request a vehicle should serve next.
///
/// Called each time an idle vehicle has a non-empty pending queue.
pub trait SchedulingStrategy: Send + Sync {
    /// Called once before the simulation loop starts.
    /// Override to cache instance data needed for scheduling decisions.
    fn initialize(&mut self, _view: &InstanceView<'_>) {}

    /// Return a `RequestId` from `queue` to dispatch next.
    ///
    /// # Panics
    /// Panics in debug builds if the returned ID is not present in `queue`.
    fn schedule(
        &mut self,
        vehicle: &VehicleSnapshot,
        queue: &[RequestId],
        view: &InstanceView<'_>,
    ) -> RequestId;
}

// ---------------------------------------------------------------------------
// BatchRoutingStrategy — slot-based variant of RoutingStrategy
// ---------------------------------------------------------------------------

/// Decides vehicle assignments for a whole batch of released requests at once.
///
/// Called by [`BatchComposableStrategy`] at the end of each time slot with all
/// requests that have been released during that slot but not yet routed.
pub trait BatchRoutingStrategy: Send + Sync {
    /// Called once before the simulation loop starts.
    fn initialize(&mut self, _view: &InstanceView<'_>) {}

    /// Return one `(RequestId, Option<vehicle_id>)` pair per input request.
    ///
    /// - `Some(vehicle_id)` → assign the request to that vehicle's queue.
    /// - `None`             → reject the request immediately.
    ///
    /// The returned `Vec` must contain exactly one entry for each element of
    /// `requests` (assertion enforced in debug builds by the caller).
    fn route_batch(
        &mut self,
        requests: &[RequestId],
        vehicles: &[VehicleSnapshot],
        view: &InstanceView<'_>,
    ) -> Vec<(RequestId, Option<i64>)>;
}

// ---------------------------------------------------------------------------
// Shared Python dict builders (used by composable and batch adapters)
// ---------------------------------------------------------------------------

/// Build a `dict` representing the instance view for Python sub-strategies.
///
/// Keys: `"depot_id"` (int), `"vehicle_specs"` (list of `{id, capacity, speed}` dicts).
pub(super) fn build_py_instance_view(py: Python, view: &InstanceView<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("depot_id", view.depot_id().0)?;
    let specs = PyList::empty(py);
    for vs in view.vehicle_specs() {
        let sd = PyDict::new(py);
        sd.set_item("id", vs.id)?;
        sd.set_item("capacity", vs.capacity)?;
        sd.set_item("speed", vs.speed)?;
        specs.append(sd)?;
    }
    d.set_item("vehicle_specs", specs)?;
    Ok(d.into())
}

/// Build a Python list of vehicle dicts from a [`VehicleSnapshot`] slice.
pub(super) fn build_py_vehicles(py: Python, vehicles: &[VehicleSnapshot]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for v in vehicles {
        list.append(build_py_vehicle(py, v)?)?;
    }
    Ok(list.into())
}

/// Build a single vehicle dict from a [`VehicleSnapshot`].
pub(super) fn build_py_vehicle(py: Python, v: &VehicleSnapshot) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("vehicle_id", v.vehicle_id)?;
    d.set_item("position", v.position.0)?;
    d.set_item("current_load", v.current_load)?;
    d.set_item("available_at", v.available_at)?;
    let route_ids: Vec<i64> = v.route.iter().map(|r| r.0).collect();
    d.set_item("route", PyList::new(py, &route_ids)?)?;
    d.set_item("service_times", PyList::new(py, &v.service_times)?)?;
    Ok(d.into())
}

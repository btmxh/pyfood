/// types.rs — Fundamental data types with no internal crate dependencies.
///
/// # Contents
/// - [`RequestId`]             — newtype over i64
/// - [`SimAction`]             — zero-allocation action enum
/// - [`VehicleSnapshot`]       — per-vehicle snapshot slice
/// - [`SimulationSnapshot`]    — lightweight state passed to strategies
/// - [`NativeStrategyWrapper`] — `#[pyclass]` box around `Box<dyn RustStrategy>`
/// - [`NativeCallbackWrapper`] — `#[pyclass]` box around `Box<dyn RustCallback>`
///
/// The [`RustStrategy`], [`RustCallback`], and [`InstanceView`] traits/types
/// live in `instance.rs` (they depend on `Request`/`VehicleSpec` defined
/// there, and keeping them together avoids a circular import).
use std::collections::HashSet;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// RequestId newtype
// ---------------------------------------------------------------------------

/// Newtype wrapper around `i64` for request/node identifiers.
///
/// Rust-internal only — not exposed as a `#[pyclass]`. Prevents accidental
/// confusion between vehicle IDs, request IDs, and raw integers in function
/// signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub i64);

// ---------------------------------------------------------------------------
// SimAction enum — zero-allocation replacement for Python action objects
// ---------------------------------------------------------------------------

/// The three actions a strategy can return.
///
/// This enum is used everywhere inside the Rust simulation loop. It is never
/// heap-allocated in the hot path (returned in a `Vec<SimAction>` by value).
/// Python actions are converted to/from this type only inside the adapters.
#[derive(Debug, Clone)]
pub enum SimAction {
    Dispatch { vehicle_id: i64, dest: RequestId },
    Wait { until: f32 },
    Reject { request_id: RequestId },
}

// ---------------------------------------------------------------------------
// Snapshot types — passed by reference to RustStrategy::next_events
// ---------------------------------------------------------------------------

/// Lightweight per-vehicle state slice, part of [`SimulationSnapshot`].
///
/// All fields are plain Rust values — no PyO3 types, no allocation beyond the
/// `Vec`s that mirror the vehicle's existing route storage.
#[derive(Debug, Clone)]
pub struct VehicleSnapshot {
    pub vehicle_id: i64,
    pub position: RequestId,
    pub current_load: f32,
    pub available_at: f32,
    pub route: Vec<RequestId>,
    pub service_times: Vec<f32>,
}

/// A lightweight snapshot of simulation state passed to [`RustStrategy`].
///
/// `released` contains only IDs (not full request data).  A native strategy
/// that needs request details should hold a reference to the instance data it
/// was constructed with.
#[derive(Debug, Clone)]
pub struct SimulationSnapshot {
    pub time: f32,
    pub pending: HashSet<RequestId>,
    pub served: HashSet<RequestId>,
    pub rejected: HashSet<RequestId>,
    /// IDs of all non-depot requests whose `release_time <= time`.
    pub released: HashSet<RequestId>,
    pub vehicles: Vec<VehicleSnapshot>,
}

// ---------------------------------------------------------------------------
// NativeStrategyWrapper — #[pyclass] box around Box<dyn RustStrategy>
// ---------------------------------------------------------------------------

/// A Python-visible wrapper around a native [`RustStrategy`] implementation.
///
/// Construct instances using the factory functions exposed by this module
/// (e.g. [`greedy_strategy`]).  Pass them directly to `Simulator.__init__`
/// instead of a Python `DispatchingStrategy` object to bypass all GIL
/// overhead during simulation.
///
/// ```python
/// from rsimulator import Simulator, greedy_strategy
///
/// sim = Simulator(instance, greedy_strategy())
/// result = sim.run()
/// ```
#[pyclass]
pub struct NativeStrategyWrapper {
    pub inner: Option<Box<dyn crate::instance::RustStrategy>>,
}

/// A Python-visible wrapper around a native [`RustCallback`] implementation.
#[pyclass]
pub struct NativeCallbackWrapper {
    pub inner: Option<Box<dyn crate::instance::RustCallback>>,
}

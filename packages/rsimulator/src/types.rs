/// types.rs — Fundamental data types with no internal crate dependencies.
///
/// # Contents
/// - [`RequestId`]              — newtype over i64
/// - [`SimAction`]              — zero-allocation action enum
/// - [`VehicleState`]           — per-vehicle mutable runtime state (also aliased as VehicleSnapshot)
/// - [`VehicleSnapshot`]        — type alias for VehicleState (used by strategy APIs)
/// - [`SimulationSnapshot`]     — zero-copy borrowed view of simulator state
/// - [`NativeDispatchStrategy`] — `#[pyclass]` box around `Box<dyn DispatchStrategy>`
/// - [`NativeEventCallback`]    — `#[pyclass]` box around `Box<dyn EventCallback>`
use std::collections::HashSet;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// RequestId newtype
// ---------------------------------------------------------------------------

/// Newtype wrapper around `i64` for request/node identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub i64);

// ---------------------------------------------------------------------------
// SimAction enum
// ---------------------------------------------------------------------------

/// The three actions a strategy can return.
#[derive(Debug, Clone)]
pub enum SimAction {
    Dispatch { vehicle_id: i64, dest: RequestId },
    Wait { until: f32 },
    Reject { request_id: RequestId },
}

// ---------------------------------------------------------------------------
// VehicleState / VehicleSnapshot
// ---------------------------------------------------------------------------

/// Mutable per-vehicle runtime state, also used as the snapshot type passed
/// to strategies. Moved here from `instance.rs` so that `SimulationSnapshot`
/// can hold a `&[VehicleState]` without introducing a circular import.
#[derive(Debug, Clone)]
pub struct VehicleState {
    pub vehicle_id: i64,
    pub position: RequestId,
    pub current_load: f32,
    pub available_at: f32,
    pub route: Vec<RequestId>,
    pub service_times: Vec<f32>,
}

/// Type alias kept for backwards compatibility with strategy code that uses
/// `VehicleSnapshot`.
pub type VehicleSnapshot = VehicleState;

// ---------------------------------------------------------------------------
// SimulationSnapshot — zero-copy borrowed view of simulator state
// ---------------------------------------------------------------------------

/// A zero-copy snapshot of simulation state passed by reference to strategies.
///
/// All fields are borrows into the `Simulator`'s own collections, so
/// constructing a snapshot is O(1) — no allocation, no cloning.
///
/// `released` points to an incrementally-maintained `HashSet` inside the
/// `Simulator` that is updated in O(new_releases) per tick via a sorted-Vec
/// cursor (see `simulator.rs`).
#[derive(Debug)]
pub struct SimulationSnapshot<'a> {
    pub time: f32,
    pub pending: &'a HashSet<RequestId>,
    pub served: &'a HashSet<RequestId>,
    pub rejected: &'a HashSet<RequestId>,
    /// IDs of all non-depot requests whose `release_time <= time`.
    pub released: &'a HashSet<RequestId>,
    pub vehicles: &'a [VehicleState],
}

// ---------------------------------------------------------------------------
// NativeDispatchStrategy
// ---------------------------------------------------------------------------

#[pyclass]
pub struct NativeDispatchStrategy {
    pub inner: Option<Box<dyn crate::instance::DispatchStrategy>>,
}

impl NativeDispatchStrategy {
    pub fn new(trait_obj: impl crate::instance::DispatchStrategy + 'static) -> Self {
        Self {
            inner: Some(Box::new(trait_obj)),
        }
    }
}

/// A Python-visible wrapper around a native [`RustCallback`] implementation.
#[pyclass]
pub struct NativeEventCallback {
    pub inner: Option<Box<dyn crate::instance::EventCallback>>,
}

impl NativeEventCallback {
    pub fn new(trait_obj: impl crate::instance::EventCallback + 'static) -> Self {
        Self {
            inner: Some(Box::new(trait_obj)),
        }
    }
}

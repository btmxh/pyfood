/// instance.rs — Domain data types, event queue primitives, and strategy traits.
///
/// # Contents
/// - [`TimeWindow`]      — earliest/latest service window
/// - [`Request`]         — a customer request or depot node
/// - [`VehicleSpec`]     — static vehicle configuration
/// - [`VehicleState`]    — mutable per-vehicle runtime state
/// - [`EventKind`]       — arrival or wake-up event discriminant
/// - [`Event`]           — timed simulation event
/// - [`MinEvent`]        — min-heap wrapper for `BinaryHeap`
/// - [`InstanceView`]    — read-only view of instance data for strategies
/// - [`RustStrategy`]    — trait for native dispatching strategies
/// - [`RustCallback`]    — trait for native action callbacks
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::types::{RequestId, SimAction, SimulationSnapshot};

// ---------------------------------------------------------------------------
// Instance data copied from Python on construction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub earliest: f32,
    pub latest: f32,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub x: f32,
    pub y: f32,
    pub demand: f32,
    pub time_window: TimeWindow,
    pub service_time: f32,
    pub release_time: f32,
    pub is_depot: bool,
}

impl Request {
    pub fn distance_to(&self, other: &Request) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct VehicleSpec {
    pub id: i64,
    pub capacity: f32,
    pub speed: f32,
}

impl VehicleSpec {
    pub fn travel_time(&self, distance: f32) -> f32 {
        distance / self.speed
    }
}

// ---------------------------------------------------------------------------
// Mutable per-vehicle runtime state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VehicleState {
    pub vehicle_id: i64,
    pub position: RequestId,
    pub current_load: f32,
    pub available_at: f32,
    pub route: Vec<RequestId>,
    pub service_times: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Min-heap event queue helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    Arrival,
    Wake,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub time: f32,
    pub kind: EventKind,
    pub id: i64, // request id for Arrival, -1 for Wake
}

/// Newtype wrapper so BinaryHeap becomes a min-heap on event time.
#[derive(Debug, Clone, PartialEq)]
pub struct MinEvent(pub Event);

impl Eq for MinEvent {}

impl PartialOrd for MinEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so smallest time is at the top of the max-heap.
        other
            .0
            .time
            .partial_cmp(&self.0.time)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// InstanceView — read-only view of instance data for strategies
// ---------------------------------------------------------------------------

/// Read-only view of instance data passed to [`RustStrategy::initialize`]
/// and [`RustStrategy::next_events`].
///
/// Requests are exposed only via point lookup (`get`) so that strategies
/// cannot enumerate the full request table and "cheat" by inspecting requests
/// that haven't been released yet.  A strategy discovers request IDs solely
/// through [`SimulationSnapshot::pending`] and [`SimulationSnapshot::vehicles`].
///
/// Vehicle specs and the depot ID are global instance constants and are always
/// available; they don't reveal any undisclosed per-request information.
pub struct InstanceView<'a> {
    pub(crate) requests: &'a HashMap<RequestId, Request>,
    pub(crate) vehicles: &'a [VehicleSpec],
    pub(crate) depot_id: RequestId,
}

impl<'a> InstanceView<'a> {
    /// Return a reference to the request with the given ID, or `None`.
    pub fn get(&self, id: RequestId) -> Option<&Request> {
        self.requests.get(&id)
    }

    /// All vehicle specs for the instance.
    pub fn vehicle_specs(&self) -> &[VehicleSpec] {
        self.vehicles
    }

    /// The depot node ID.
    pub fn depot_id(&self) -> RequestId {
        self.depot_id
    }
}

// ---------------------------------------------------------------------------
// RustStrategy and RustCallback traits
// ---------------------------------------------------------------------------

/// A dispatching strategy implemented entirely in Rust.
///
/// Implementing this trait instead of the Python-side `DispatchingStrategy`
/// protocol eliminates all GIL acquisitions and PyO3 serialisation overhead
/// from the simulation hot path.
///
/// # Example
///
/// ```rust
/// struct AlwaysWait;
/// impl RustStrategy for AlwaysWait {
///     fn next_events(&mut self, state: &SimulationSnapshot, _view: &InstanceView<'_>) -> Vec<SimAction> {
///         vec![]
///     }
/// }
/// ```
pub trait RustStrategy: Send + Sync {
    /// Called once by the simulator after instance data is parsed, before the
    /// simulation loop starts.  Override to cache instance data (e.g. request
    /// time windows, vehicle speed) for use in [`next_events`].
    ///
    /// `view` provides point lookup by [`RequestId`], the full vehicle specs,
    /// and the depot ID.  Strategies cannot enumerate all requests and must
    /// discover them via the snapshot.
    ///
    /// The default implementation is a no-op.
    fn initialize(&mut self, _view: &InstanceView<'_>) {}

    fn next_events(
        &mut self,
        state: &SimulationSnapshot,
        view: &InstanceView<'_>,
    ) -> Vec<SimAction>;
}

/// An action callback implemented entirely in Rust.
///
/// Called once per action (strategy-originated or auto-reject) with the
/// current simulation time, the action, and whether it was automatic.
pub trait RustCallback: Send + Sync {
    fn on_action(&self, time: f32, action: &SimAction, auto: bool);
}

/// instance.rs — Domain data types, event queue primitives, and strategy traits.
///
/// # Contents
/// - [`TimeWindow`]      — earliest/latest service window
/// - [`Request`]         — a customer request or depot node
/// - [`VehicleSpec`]     — static vehicle configuration
/// - [`VehicleState`]    — re-exported from `types` (moved there to break circular import)
/// - [`EventKind`]       — arrival or wake-up event discriminant
/// - [`Event`]           — timed simulation event
/// - [`MinEvent`]        — min-heap wrapper for `BinaryHeap`
/// - [`InstanceView`]    — read-only view of instance data for strategies
/// - [`DispatchStrategy`] — trait for native dispatching strategies
/// - [`EventCallback`]   — trait for native action callbacks
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::types::{RequestId, SimAction, SimulationSnapshot};

// Re-export so existing `use crate::instance::VehicleState` still works.
pub use crate::types::VehicleState;

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
    pub id: i64,
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
        other
            .0
            .time
            .partial_cmp(&self.0.time)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// InstanceView
// ---------------------------------------------------------------------------

pub struct InstanceView<'a> {
    pub(crate) requests: &'a HashMap<RequestId, Request>,
    pub(crate) ascending_release: &'a [(f32, RequestId)],
    pub(crate) vehicles: &'a [VehicleSpec],
    pub(crate) depot_id: RequestId,
}

impl<'a> InstanceView<'a> {
    pub fn get(&self, id: RequestId) -> Option<&Request> {
        self.requests.get(&id)
    }

    pub fn vehicle_specs(&self) -> &[VehicleSpec] {
        self.vehicles
    }

    pub fn depot_id(&self) -> RequestId {
        self.depot_id
    }
}

// ---------------------------------------------------------------------------
// DispatchStrategy and EventCallback traits
// ---------------------------------------------------------------------------

pub trait DispatchStrategy: Send + Sync {
    fn initialize(&mut self, _view: &InstanceView<'_>) {}

    fn next_events(
        &mut self,
        state: &SimulationSnapshot<'_>,
        view: &InstanceView<'_>,
    ) -> Vec<SimAction>;
}

pub trait EventCallback: Send + Sync {
    fn on_action(&self, time: f32, action: &SimAction, auto: bool);
}

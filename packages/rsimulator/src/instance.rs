use crate::hashmap::Map as HashMap;
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
// InstanceLimits
// ---------------------------------------------------------------------------

/// Instance-derived scale factors for normalizing GP terminal values.
///
/// Computed once from static instance data (positions, time windows, vehicle
/// specs) and stored in [`InstanceView`] so every strategy call shares the same
/// normalization without re-scanning the instance.
#[derive(Debug, Clone, Copy)]
pub struct InstanceLimits {
    /// Bounding-box diagonal / speed — characteristic travel-time scale.
    pub max_travel_time: f32,
    /// Maximum `time_window.latest` across all requests — time scale.
    pub planning_horizon: f32,
    /// First vehicle's capacity — load/demand scale.
    pub vehicle_capacity: f32,
}

impl InstanceLimits {
    /// Compute limits in a single O(|requests|) pass over the raw instance data.
    pub fn from_data(
        requests: &HashMap<RequestId, Request>,
        vehicle_specs: &[VehicleSpec],
    ) -> Self {
        let (speed, capacity) = vehicle_specs
            .first()
            .map(|vs| (vs.speed, vs.capacity))
            .unwrap_or((1.0, 1.0));

        let mut xmin = f32::INFINITY;
        let mut xmax = f32::NEG_INFINITY;
        let mut ymin = f32::INFINITY;
        let mut ymax = f32::NEG_INFINITY;
        let mut max_latest = 1.0_f32;
        for req in requests.values() {
            xmin = xmin.min(req.x);
            xmax = xmax.max(req.x);
            ymin = ymin.min(req.y);
            ymax = ymax.max(req.y);
            max_latest = max_latest.max(req.time_window.latest);
        }

        let max_dist = ((xmax - xmin).powi(2) + (ymax - ymin).powi(2)).sqrt();
        let max_travel_time = if speed > 0.0 {
            (max_dist / speed).max(1.0)
        } else {
            1.0
        };
        let vehicle_capacity = if capacity.is_finite() && capacity > 0.0 {
            capacity
        } else {
            1.0
        };

        InstanceLimits {
            max_travel_time,
            planning_horizon: max_latest,
            vehicle_capacity,
        }
    }

    /// Normalize a raw terminal value by its characteristic scale.
    ///
    /// Terminal IDs:
    ///   0 → TravelTime        / max_travel_time
    ///   1 → WindowEarliest    / planning_horizon
    ///   2 → WindowLatest      / planning_horizon
    ///   3 → TimeUntilDue      / planning_horizon
    ///   4 → Demand            / vehicle_capacity
    ///   5 → CurrentLoad       / vehicle_capacity
    ///   6 → RemainingCapacity / vehicle_capacity
    ///   7 → ReleaseTime       / planning_horizon
    #[inline(always)]
    pub fn normalize(&self, terminal_id: u8, raw: f32) -> f32 {
        let scale = match terminal_id {
            0 => self.max_travel_time,
            1 | 2 | 3 | 7 => self.planning_horizon,
            4..=6 => self.vehicle_capacity,
            _ => return raw,
        };
        raw / scale
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
    /// Pre-computed normalization scale factors — computed once per instance
    /// so strategies pay no per-call overhead.
    pub limits: InstanceLimits,
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

/// gp_tree.rs — GP tree representation and evaluation.
///
/// # Contents
/// - [`Terminal`]    — leaf nodes (constants and features)
/// - [`Node`]        — internal operator nodes
/// - [`Tree`]        — recursive GP tree enum (terminal or internal node)
/// - [`EvalContext`] — context passed to `eval` (request + vehicle + instance info)
/// - [`GpTree`]      — `#[pyclass]` wrapper so Python can construct and pass trees
///
/// # Feature terminals
///
/// | Terminal              | Value                                          |
/// |-----------------------|------------------------------------------------|
/// | `Const(v)`            | Literal constant `v`                           |
/// | `TravelTime`          | Travel time from vehicle position to request   |
/// | `WindowEarliest`      | Request time window earliest start             |
/// | `WindowLatest`        | Request time window latest start               |
/// | `TimeUntilDue`        | `latest - current_time` (urgency)              |
/// | `Demand`              | Request demand                                 |
/// | `CurrentLoad`         | Vehicle's current load                         |
/// | `RemainingCapacity`   | `vehicle_capacity - current_load`              |
/// | `ReleaseTime`         | Request release time                           |
use pyo3::prelude::*;

use crate::instance::Request;
use crate::types::VehicleSnapshot;

// ---------------------------------------------------------------------------
// Terminal nodes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Terminal {
    Const(f64),
    TravelTime,
    WindowEarliest,
    WindowLatest,
    TimeUntilDue,
    Demand,
    CurrentLoad,
    RemainingCapacity,
    ReleaseTime,
}

// ---------------------------------------------------------------------------
// Internal (operator) nodes
// ---------------------------------------------------------------------------

/// Division is *protected*: `a / 0` returns `1.0`.
#[derive(Debug, Clone)]
pub enum Node {
    Add,
    Sub,
    Mul,
    Div,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Tree {
    Leaf(Terminal),
    Node {
        op: Node,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

/// Context passed to [`Tree::eval`] — no heap allocation per call.
pub struct EvalContext<'a> {
    pub request: &'a Request,
    pub vehicle: &'a VehicleSnapshot,
    pub current_time: f64,
    pub speed: f64,
    pub vehicle_capacity: f64,
    pub vehicle_pos_x: f64,
    pub vehicle_pos_y: f64,
}

impl Tree {
    pub fn eval(&self, ctx: &EvalContext<'_>) -> f64 {
        match self {
            Tree::Leaf(terminal) => match terminal {
                Terminal::Const(v) => *v,
                Terminal::TravelTime => {
                    let dx = ctx.vehicle_pos_x - ctx.request.x;
                    let dy = ctx.vehicle_pos_y - ctx.request.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if ctx.speed == 0.0 {
                        f64::INFINITY
                    } else {
                        dist / ctx.speed
                    }
                }
                Terminal::WindowEarliest => ctx.request.time_window.earliest,
                Terminal::WindowLatest => ctx.request.time_window.latest,
                Terminal::TimeUntilDue => ctx.request.time_window.latest - ctx.current_time,
                Terminal::Demand => ctx.request.demand,
                Terminal::CurrentLoad => ctx.vehicle.current_load,
                Terminal::RemainingCapacity => ctx.vehicle_capacity - ctx.vehicle.current_load,
                Terminal::ReleaseTime => ctx.request.release_time,
            },
            Tree::Node { op, left, right } => {
                let l = left.eval(ctx);
                let r = right.eval(ctx);
                match op {
                    Node::Add => l + r,
                    Node::Sub => l - r,
                    Node::Mul => l * r,
                    Node::Div => {
                        if r == 0.0 {
                            1.0
                        } else {
                            l / r
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpTree — #[pyclass] wrapper
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct GpTree {
    pub(crate) inner: Tree,
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn gp_const(value: f64) -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::Const(value)),
    }
}

#[pyfunction]
pub fn gp_add(left: GpTree, right: GpTree) -> GpTree {
    GpTree {
        inner: Tree::Node {
            op: Node::Add,
            left: Box::new(left.inner),
            right: Box::new(right.inner),
        },
    }
}

#[pyfunction]
pub fn gp_sub(left: GpTree, right: GpTree) -> GpTree {
    GpTree {
        inner: Tree::Node {
            op: Node::Sub,
            left: Box::new(left.inner),
            right: Box::new(right.inner),
        },
    }
}

#[pyfunction]
pub fn gp_mul(left: GpTree, right: GpTree) -> GpTree {
    GpTree {
        inner: Tree::Node {
            op: Node::Mul,
            left: Box::new(left.inner),
            right: Box::new(right.inner),
        },
    }
}

#[pyfunction]
pub fn gp_div(left: GpTree, right: GpTree) -> GpTree {
    GpTree {
        inner: Tree::Node {
            op: Node::Div,
            left: Box::new(left.inner),
            right: Box::new(right.inner),
        },
    }
}

#[pyfunction]
pub fn gp_travel_time() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::TravelTime),
    }
}

#[pyfunction]
pub fn gp_window_earliest() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::WindowEarliest),
    }
}

#[pyfunction]
pub fn gp_window_latest() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::WindowLatest),
    }
}

#[pyfunction]
pub fn gp_time_until_due() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::TimeUntilDue),
    }
}

#[pyfunction]
pub fn gp_demand() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::Demand),
    }
}

#[pyfunction]
pub fn gp_current_load() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::CurrentLoad),
    }
}

#[pyfunction]
pub fn gp_remaining_capacity() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::RemainingCapacity),
    }
}

#[pyfunction]
pub fn gp_release_time() -> GpTree {
    GpTree {
        inner: Tree::Leaf(Terminal::ReleaseTime),
    }
}

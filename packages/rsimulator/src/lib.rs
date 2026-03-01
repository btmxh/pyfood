/// rsimulator — Rust-backed DVRPTW simulation engine exposed to Python via PyO3.
///
/// # Module layout
///
/// ```text
/// lib.rs          ← this file: mod declarations + #[pymodule] registration
/// types.rs        ← RequestId, SimAction, snapshots, NativeStrategyWrapper/NativeCallbackWrapper
/// instance.rs     ← Request, VehicleSpec, VehicleState, event queue, InstanceView, strategy traits
/// py_bridge.rs    ← PyO3 glue: PyStrategyAdapter, PyCallbackAdapter, dict/action helpers
/// strategies.rs   ← GreedyRustStrategy + greedy_strategy() #[pyfunction]
/// simulator.rs    ← Simulator #[pyclass] + all impl blocks
/// ```
///
/// # Hot path (native strategy)
///
/// ```text
/// Simulator::run()
///   └─ build_snapshot()            ← cheap: integer copies, no PyO3
///   └─ RustStrategy::next_events() ← pure Rust, no GIL
///   └─ execute_action(SimAction)   ← enum match, no PyO3
/// ```
///
/// Python strategies fall back to [`PyStrategyAdapter`] which acquires the GIL
/// only for the strategy call, concentrating all serialisation overhead in one
/// place.
mod instance;
mod py_bridge;
mod simulator;
mod strategies;
mod types;

pub use simulator::Simulator;
pub use strategies::greedy_strategy;
pub use types::{NativeCallbackWrapper, NativeStrategyWrapper};

use pyo3::prelude::*;

#[pymodule]
fn rsimulator(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    m.add_class::<NativeStrategyWrapper>()?;
    m.add_class::<NativeCallbackWrapper>()?;
    m.add_function(wrap_pyfunction!(greedy_strategy, m)?)?;
    Ok(())
}

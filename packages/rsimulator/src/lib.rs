/// rsimulator — Rust-backed DVRPTW simulation engine exposed to Python via PyO3.
///
/// # Module layout
///
/// ```text
/// lib.rs                ← this file: mod declarations + #[pymodule] registration
/// types.rs              ← RequestId, SimAction, snapshots, NativeDispatchStrategy/NativeCallbackWrapper
/// instance.rs           ← Request, VehicleSpec, VehicleState, event queue, InstanceView, strategy traits
/// py_bridge.rs          ← PyO3 glue: PyStrategyAdapter, PyCallbackAdapter, dict/action helpers
/// simulator.rs          ← Simulator #[pyclass] + all impl blocks
/// strategies/
///   mod.rs              ← re-exports, sub-traits, shared Python dict builders
///   greedy.rs           ← GreedyRustStrategy + greedy_strategy()
///   composable.rs       ← ComposableStrategy + per-request adapters + composable_strategy()
///   batch.rs            ← BatchComposableStrategy + batch adapter + batch_composable_strategy()
///   gp_tree.rs          ← GpTree #[pyclass] + Tree/Terminal/Node + factory functions
///   gp_strategy.rs      ← GpRoutingStrategy, GpSchedulingStrategy, GpBatchRoutingStrategy, gp_strategy(), gp_batch_strategy()
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
pub mod instance;
mod py_bridge;
mod simulator;
pub mod strategies;
pub mod types;

pub use simulator::Simulator;
pub use strategies::{
    ComposableStrategy, batch_composable_strategy, composable_strategy, greedy_strategy,
};
pub use strategies::{RoutingStrategy, SchedulingStrategy};
pub use types::{NativeDispatchStrategy, NativeEventCallback};

use pyo3::prelude::*;

#[pymodule]
fn rsimulator(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Simulator>()?;
    m.add_class::<NativeDispatchStrategy>()?;
    m.add_class::<NativeEventCallback>()?;
    // Built-in strategies
    m.add_function(wrap_pyfunction!(greedy_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(composable_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_composable_strategy, m)?)?;
    // GP strategies
    m.add_class::<strategies::gp_tree::FlatGpTree>()?;
    m.add_function(wrap_pyfunction!(strategies::gp_strategy::gp_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_strategy::gp_batch_strategy,
        m
    )?)?;
    // FlatGpTree factories
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_const, m)?)?;
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_add, m)?)?;
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_sub, m)?)?;
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_mul, m)?)?;
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_div, m)?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_travel_time,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_window_earliest,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_window_latest,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_time_until_due,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(strategies::gp_tree::flat_gp_demand, m)?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_current_load,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_remaining_capacity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        strategies::gp_tree::flat_gp_release_time,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_bridge::python_dispatch_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(py_bridge::python_event_callback, m)?)?;
    Ok(())
}

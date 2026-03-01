"""Type stubs for the rsimulator Rust extension."""

from typing import Any

class NativeStrategyWrapper:
    """Opaque wrapper around a native Rust dispatching strategy."""

    ...

class NativeCallbackWrapper:
    """Opaque wrapper around a native Rust action callback."""

    ...

class Simulator:
    """Rust-backed DVRPTW simulator."""

    def __init__(
        self,
        instance: Any,
        strategy: Any,
        action_callback: Any = None,
    ) -> None: ...
    def run(self) -> dict[str, Any]: ...

def greedy_strategy() -> NativeStrategyWrapper: ...
def composable_strategy(router: Any, scheduler: Any) -> NativeStrategyWrapper: ...
def batch_composable_strategy(
    router: Any, scheduler: Any, slot_size: float
) -> NativeStrategyWrapper: ...

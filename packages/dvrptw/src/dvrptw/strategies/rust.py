import rsimulator as rs

from dvrptw.simulator.rsimulator import NativeDispatchStrategy


class NativeBatchRouter:
    """Opaque wrapper around a Rust batch router used for batch composable strategies."""


class NativeRouter:
    """Opaque wrapper around a Rust router used for composable strategies."""


class NativeSequencer:
    """Opaque wrapper around a Rust sequencer used for composable strategies."""


def greedy_strategy() -> NativeDispatchStrategy:
    return rs.greedy_strategy()


def composable_strategy(
    router: NativeRouter, sequencer: NativeSequencer
) -> NativeDispatchStrategy:
    return rs.composable_strategy(router, sequencer)


def batch_composable_strategy(
    router: NativeBatchRouter, sequencer: NativeSequencer, slot_size: float
) -> NativeDispatchStrategy:
    return rs.batch_composable_strategy(router, sequencer, slot_size)

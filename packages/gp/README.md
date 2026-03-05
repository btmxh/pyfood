Toy GP package that uses the high-performance GP FlatGpTree from
the rsimulator extension. This package implements a simple, non-performance
critical GP loop in Python and uses FlatGpTree for tree construction and
serialization.

This is intentionally minimal — performance-critical operators (evaluation,
SIMD batch eval, low-level crossover) live in the Rust extension.

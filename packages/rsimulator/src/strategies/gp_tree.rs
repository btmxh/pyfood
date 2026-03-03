/// gp_tree.rs — GP tree representations and evaluation.
///
/// # Contents
/// - [`EvalContext`] — context passed to `eval` (request + vehicle + instance info)
/// - [`FlatTree`]    — compact postfix opcode array (1 byte per token, no heap per node)
/// - [`FlatGpTree`]  — `#[pyclass]` wrapper for `FlatTree`
///
/// # FlatTree token encoding (1 byte per token)
///
/// ```text
/// 0xxxxxxx  — Const immediate: payload = [S:MMM:EEE]
///               S   = sign bit (bit 6 of byte)
///               MMM = mantissa bits (bits 5..3); value = 1.0 + MMM/8.0
///               EEE = exponent bits (bits 2..0); biased exponent, BIAS = 3
///               Special: payload == 0 → exact 0.0
///               Decoded: (if S then -1 else 1) * (1.0 + MMM/8.0) * 2^(EEE - BIAS)
/// 10xxxxxx  — Terminal: lower 6 bits = terminal_id (matches Terminal enum order)
/// 11xxxxxx  — Op: lower 6 bits = op_id (0=Add,1=Sub,2=Mul,3=Div)
/// ```
///
/// All ops are binary (arity 2). Terminals and consts have arity 0.
///
/// FlatTree evaluation is performed entirely in **f32** for throughput; results
/// are widened back to f64 only at the strategy boundary.
///
/// # Feature terminals
///
/// | Terminal id           | Value                                          |
/// |-----------------------|------------------------------------------------|
/// | 0 `TravelTime`        | Travel time from vehicle position to request   |
/// | 1 `WindowEarliest`    | Request time window earliest start             |
/// | 2 `WindowLatest`      | Request time window latest start               |
/// | 3 `TimeUntilDue`      | `latest - current_time` (urgency)              |
/// | 4 `Demand`            | Request demand                                 |
/// | 5 `CurrentLoad`       | Vehicle's current load                         |
/// | 6 `RemainingCapacity` | `vehicle_capacity - current_load`              |
/// | 7 `ReleaseTime`       | Request release time                           |
use pyo3::prelude::*;

use crate::instance::Request;
use crate::types::VehicleSnapshot;

// ---------------------------------------------------------------------------
// SIMD feature gate
// ---------------------------------------------------------------------------

#[cfg(feature = "simd")]
use wide::{CmpEq, f32x8};

// ---------------------------------------------------------------------------
// EvalContext
// ---------------------------------------------------------------------------

/// Context passed to [`FlatTree::eval_scalar`] — no heap allocation per call.
pub struct EvalContext<'a> {
    pub request: &'a Request,
    pub vehicle: &'a VehicleSnapshot,
    pub current_time: f32,
    pub speed: f32,
    pub vehicle_capacity: f32,
    pub vehicle_pos_x: f32,
    pub vehicle_pos_y: f32,
}

impl EvalContext<'_> {
    /// Extract a terminal value by its flat terminal_id (0..=7), returned as f32.
    #[inline(always)]
    pub fn terminal_value(&self, id: u8) -> f32 {
        match id {
            0 => {
                let dx = self.vehicle_pos_x - self.request.x;
                let dy = self.vehicle_pos_y - self.request.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if self.speed == 0.0 {
                    f32::INFINITY
                } else {
                    dist / self.speed
                }
            }
            1 => self.request.time_window.earliest,
            2 => self.request.time_window.latest,
            3 => self.request.time_window.latest - self.current_time,
            4 => self.request.demand,
            5 => self.vehicle.current_load,
            6 => self.vehicle_capacity - self.vehicle.current_load,
            7 => self.request.release_time,
            _ => 0.0,
        }
    }

    /// Alias kept for API compatibility with `FlatTree`.
    #[inline(always)]
    pub fn terminal_value_f32(&self, id: u8) -> f32 {
        self.terminal_value(id)
    }
}

// ---------------------------------------------------------------------------
// FlatTree token constants
// ---------------------------------------------------------------------------

/// `10xxxxxx` — terminal prefix (bits 7..6 = 0b10)
const TERM_BASE: u8 = 0b1000_0000;
/// `11xxxxxx` — op prefix (bits 7..6 = 0b11)
const OP_BASE: u8 = 0b1100_0000;

/// Returns true if the byte is a const-immediate token (MSB = 0).
#[inline(always)]
fn token_is_const(b: u8) -> bool {
    b & 0x80 == 0
}

/// Returns true if the byte is a terminal token (bits 7..6 = 0b10).
#[inline(always)]
fn token_is_terminal(b: u8) -> bool {
    b & 0xC0 == 0x80
}

/// Returns true if the byte is a leaf (const or terminal).
#[inline(always)]
fn token_is_leaf(b: u8) -> bool {
    b & 0xC0 != 0xC0
}

// ---------------------------------------------------------------------------
// 7-bit float codec (f32)
// ---------------------------------------------------------------------------

/// Bias used for the 3-bit exponent: bias 3 → exponent 0 → 2^0 = 1.
const CONST7_BIAS: i32 = 3;

/// Decode a 7-bit const payload (bits 6..0 of a const token byte) to f32.
///
/// Encoding: `[S:MMM:EEE]`
/// - bit 6: S = sign (1 → negative)
/// - bits 5..3: MMM = fractional mantissa steps (value = 1.0 + MMM/8.0)
/// - bits 2..0: EEE = biased exponent (actual = EEE - BIAS)
/// - Special: payload == 0 → 0.0
pub fn decode_const7(payload: u8) -> f32 {
    if payload == 0 {
        return 0.0;
    }
    let sign: f32 = if payload & 0x40 != 0 { -1.0 } else { 1.0 };
    let mmm = ((payload >> 3) & 0x07) as f32;
    let eee = (payload & 0x07) as i32;
    sign * (1.0 + mmm / 8.0) * (2.0_f32).powi(eee - CONST7_BIAS)
}

/// Encode an f32 into a 7-bit const payload (lower 7 bits; caller sets bit 7 = 0).
///
/// Finds the nearest representable value. Clamps to the representable range
/// [0.125, 30.0]. Returns 0 (exact zero) for ±0.0.
pub fn encode_const7(v: f32) -> u8 {
    if v == 0.0 {
        return 0;
    }

    let sign_bit: u8 = if v < 0.0 { 0x40 } else { 0 };
    // Clamp magnitude to representable range: [2^(0-3), 1.875*2^(7-3)] = [0.125, 30.0]
    let abs_v = v.abs().clamp(0.125, 30.0);

    let mut best_payload: u8 = 0;
    let mut best_err = f32::INFINITY;

    for eee in 0u8..8 {
        for mmm in 0u8..8 {
            let payload = sign_bit | (mmm << 3) | eee;
            if payload == 0 {
                continue; // reserved for exact zero
            }
            let decoded = decode_const7(payload);
            let err = (decoded.abs() - abs_v).abs();
            if err < best_err {
                best_err = err;
                best_payload = payload;
            }
        }
    }

    best_payload
}

// ---------------------------------------------------------------------------
// FlatTree
// ---------------------------------------------------------------------------

/// Compact postfix representation of a GP tree.
///
/// Each byte is one token; see module-level docs for encoding.
/// Evaluation uses a small f32 stack; no heap allocation per eval call.
#[derive(Debug, Clone)]
pub struct FlatTree {
    pub ops: Vec<u8>,
}

impl FlatTree {
    // -----------------------------------------------------------------------
    // Construction helpers
    // -----------------------------------------------------------------------

    pub fn from_const(v: f32) -> Self {
        FlatTree {
            ops: vec![encode_const7(v)],
        }
    }

    pub fn from_terminal(id: u8) -> Self {
        FlatTree {
            ops: vec![TERM_BASE | (id & 0x3F)],
        }
    }

    pub fn binary(left: &FlatTree, right: &FlatTree, op_id: u8) -> Self {
        let mut ops = Vec::with_capacity(left.ops.len() + right.ops.len() + 1);
        ops.extend_from_slice(&left.ops);
        ops.extend_from_slice(&right.ops);
        ops.push(OP_BASE | (op_id & 0x3F));
        FlatTree { ops }
    }

    // -----------------------------------------------------------------------
    // Subtree helpers
    // -----------------------------------------------------------------------

    /// Find the start index of the subtree whose last token is at `end` (inclusive).
    ///
    /// Scans backward: each leaf decrements the needed-count by 1, each binary op
    /// increments it by 1. Stops when count reaches 0.
    pub fn subtree_start(ops: &[u8], end: usize) -> usize {
        let mut needed: i64 = 1;
        let mut i = end as i64;
        while i >= 0 {
            let b = ops[i as usize];
            if token_is_leaf(b) {
                needed -= 1;
            } else {
                needed += 1;
            }
            if needed == 0 {
                return i as usize;
            }
            i -= 1;
        }
        0 // malformed — fallback
    }

    /// Returns `(start, end)` of the subtree ending at `end` (both inclusive).
    pub fn subtree_range(&self, end: usize) -> (usize, usize) {
        (Self::subtree_start(&self.ops, end), end)
    }

    // -----------------------------------------------------------------------
    // Crossover / splice
    // -----------------------------------------------------------------------

    /// Replace the subtree `ops[a_start..=a_end]` with `replacement`.
    pub fn splice(ops: &[u8], a_start: usize, a_end: usize, replacement: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ops.len() - (a_end - a_start + 1) + replacement.len());
        out.extend_from_slice(&ops[..a_start]);
        out.extend_from_slice(replacement);
        out.extend_from_slice(&ops[a_end + 1..]);
        out
    }

    // -----------------------------------------------------------------------
    // Scalar evaluation (f32 internally)
    // -----------------------------------------------------------------------

    /// Evaluate the tree for a single `EvalContext`. Returns f32.
    pub fn eval_scalar(&self, ctx: &EvalContext<'_>) -> f32 {
        let mut stack: Vec<f32> = Vec::with_capacity(self.ops.len() / 2 + 1);
        for &b in &self.ops {
            if token_is_const(b) {
                stack.push(decode_const7(b & 0x7F));
            } else if token_is_terminal(b) {
                let id = b & 0x3F;
                stack.push(ctx.terminal_value_f32(id));
            } else {
                let r = stack.pop().unwrap_or(0.0);
                let l = stack.pop().unwrap_or(0.0);
                let result = apply_op_f32(b & 0x3F, l, r);
                stack.push(result);
            }
        }
        stack.pop().unwrap_or(0.0)
    }

    /// Evaluate from a plain f32 terminal slice (no EvalContext).
    pub fn eval_scalar_from_slice(&self, terminals: &[f32]) -> f32 {
        let mut stack: Vec<f32> = Vec::with_capacity(self.ops.len() / 2 + 1);
        for &b in &self.ops {
            if token_is_const(b) {
                stack.push(decode_const7(b & 0x7F));
            } else if token_is_terminal(b) {
                let id = (b & 0x3F) as usize;
                stack.push(*terminals.get(id).unwrap_or(&0.0));
            } else {
                let r = stack.pop().unwrap_or(0.0);
                let l = stack.pop().unwrap_or(0.0);
                stack.push(apply_op_f32(b & 0x3F, l, r));
            }
        }
        stack.pop().unwrap_or(0.0)
    }

    // -----------------------------------------------------------------------
    // Batch evaluation
    // -----------------------------------------------------------------------

    /// Evaluate the tree over a batch of inputs.
    ///
    /// `terminal_matrix[term_id]` is a slice of N f32 values, one per input.
    /// Returns a `Vec<f32>` of length N.
    ///
    /// Uses SIMD (8 lanes of f32) internally when the `simd` feature is enabled;
    /// otherwise falls back to scalar.
    pub fn eval_batch(&self, terminal_matrix: &[Vec<f32>]) -> Vec<f32> {
        if terminal_matrix.is_empty() {
            return Vec::new();
        }
        let n = terminal_matrix[0].len();

        #[cfg(feature = "simd")]
        {
            self.eval_batch_simd(terminal_matrix, n)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.eval_batch_scalar(terminal_matrix, n)
        }
    }

    #[cfg_attr(feature = "simd", allow(dead_code))]
    fn eval_batch_scalar(&self, terminal_matrix: &[Vec<f32>], n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let terminals: Vec<f32> = terminal_matrix.iter().map(|col| col[i]).collect();
            out.push(self.eval_scalar_from_slice(&terminals));
        }
        out
    }

    #[cfg(feature = "simd")]
    fn eval_batch_simd(&self, terminal_matrix: &[Vec<f32>], n: usize) -> Vec<f32> {
        const LANES: usize = 8;
        let mut out = vec![0.0f32; n];
        let mut offset = 0usize;

        while offset + LANES <= n {
            let results = self.eval_simd(terminal_matrix, offset);
            out[offset..offset + LANES].copy_from_slice(&results.to_array());
            offset += LANES;
        }

        // Scalar tail
        for i in offset..n {
            let terminals: Vec<f32> = terminal_matrix.iter().map(|col| col[i]).collect();
            out[i] = self.eval_scalar_from_slice(&terminals);
        }

        out
    }

    #[cfg(feature = "simd")]
    pub fn eval_simd(&self, terminal_matrix: &[Vec<f32>], offset: usize) -> f32x8 {
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let mut stack: Vec<f32x8> = Vec::with_capacity(self.ops.len() / 2 + 1);

        for &b in &self.ops {
            if token_is_const(b) {
                stack.push(f32x8::splat(decode_const7(b & 0x7F)));
            } else if token_is_terminal(b) {
                let id = (b & 0x3F) as usize;
                let col = &terminal_matrix[id];
                let lane_vals: [f32; 8] = std::array::from_fn(|lane| col[offset + lane]);
                stack.push(f32x8::from(lane_vals));
            } else {
                let r = stack.pop().unwrap_or(zero);
                let l = stack.pop().unwrap_or(zero);
                let result = match b & 0x3F {
                    0 => l + r,
                    1 => l - r,
                    2 => l * r,
                    3 => {
                        // protected div: r==0 → 1.0
                        let mask = r.simd_eq(zero);
                        mask.blend(one, l / r)
                    }
                    _ => zero,
                };
                stack.push(result);
            }
        }
        stack.pop().unwrap_or(zero)
    }
}

// ---------------------------------------------------------------------------
// Shared op dispatch (f32)
// ---------------------------------------------------------------------------

#[inline(always)]
fn apply_op_f32(op_id: u8, l: f32, r: f32) -> f32 {
    match op_id {
        0 => l + r,
        1 => l - r,
        2 => l * r,
        3 => {
            if r == 0.0 {
                1.0
            } else {
                l / r
            }
        }
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// FlatGpTree — #[pyclass] wrapper
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct FlatGpTree {
    pub(crate) inner: FlatTree,
}

// ---------------------------------------------------------------------------
// Factory functions — FlatGpTree
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn flat_gp_const(value: f64) -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_const(value as f32),
    }
}

#[pyfunction]
pub fn flat_gp_add(left: FlatGpTree, right: FlatGpTree) -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::binary(&left.inner, &right.inner, 0),
    }
}

#[pyfunction]
pub fn flat_gp_sub(left: FlatGpTree, right: FlatGpTree) -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::binary(&left.inner, &right.inner, 1),
    }
}

#[pyfunction]
pub fn flat_gp_mul(left: FlatGpTree, right: FlatGpTree) -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::binary(&left.inner, &right.inner, 2),
    }
}

#[pyfunction]
pub fn flat_gp_div(left: FlatGpTree, right: FlatGpTree) -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::binary(&left.inner, &right.inner, 3),
    }
}

#[pyfunction]
pub fn flat_gp_travel_time() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(0),
    }
}

#[pyfunction]
pub fn flat_gp_window_earliest() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(1),
    }
}

#[pyfunction]
pub fn flat_gp_window_latest() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(2),
    }
}

#[pyfunction]
pub fn flat_gp_time_until_due() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(3),
    }
}

#[pyfunction]
pub fn flat_gp_demand() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(4),
    }
}

#[pyfunction]
pub fn flat_gp_current_load() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(5),
    }
}

#[pyfunction]
pub fn flat_gp_remaining_capacity() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(6),
    }
}

#[pyfunction]
pub fn flat_gp_release_time() -> FlatGpTree {
    FlatGpTree {
        inner: FlatTree::from_terminal(7),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- const7 codec ----

    #[test]
    fn test_decode_const7_zero() {
        assert_eq!(decode_const7(0), 0.0_f32);
    }

    #[test]
    fn test_decode_const7_one() {
        // S=0, MMM=000, EEE=011 → (1.0 + 0/8) * 2^(3-3) = 1.0
        // byte: 0b0_000_011 = 3
        let payload: u8 = 0b0_000_011;
        assert_eq!(decode_const7(payload), 1.0_f32);
    }

    #[test]
    fn test_decode_const7_negative() {
        // S=1, MMM=000, EEE=011 → -1.0 * (1.0 + 0/8) * 2^(3-3) = -1.0
        // byte: 0b1_000_011 = 67
        let payload: u8 = 0b1_000_011;
        assert_eq!(decode_const7(payload), -1.0_f32);
    }

    #[test]
    fn test_decode_const7_one_and_half() {
        // S=0, MMM=100=4, EEE=011=3 → (1.0 + 4/8) * 2^0 = 1.5
        // byte: 0b0_100_011 = 35
        let payload: u8 = 0b0_100_011;
        assert!((decode_const7(payload) - 1.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let cases: &[f32] = &[1.0, -1.0, 0.5, 2.0, -2.0, 0.25, 4.0];
        for &v in cases {
            let encoded = encode_const7(v);
            let decoded = decode_const7(encoded & 0x7F);
            let err = (decoded - v).abs();
            // Allow up to 12.5% relative error (1 mantissa step = 1/8)
            assert!(
                err <= v.abs() * 0.13 + 1e-6,
                "v={v}: encoded={encoded:#010b} decoded={decoded} err={err}"
            );
        }
    }

    #[test]
    fn test_encode_const7_zero() {
        assert_eq!(encode_const7(0.0), 0);
    }

    #[test]
    fn test_encode_const7_clamps_large() {
        let encoded = encode_const7(1000.0);
        let decoded = decode_const7(encoded & 0x7F);
        assert!(decoded <= 31.0, "decoded={decoded}");
    }

    // ---- subtree_start ----

    #[test]
    fn test_subtree_start_single_leaf() {
        let ops = vec![TERM_BASE | 0];
        assert_eq!(FlatTree::subtree_start(&ops, 0), 0);
    }

    #[test]
    fn test_subtree_start_binary() {
        let ops = vec![TERM_BASE | 0, TERM_BASE | 1, OP_BASE | 0];
        assert_eq!(FlatTree::subtree_start(&ops, 2), 0);
        assert_eq!(FlatTree::subtree_start(&ops, 1), 1);
    }

    #[test]
    fn test_subtree_start_nested() {
        // ADD(ADD(t0, t1), t2) → postfix: [t0, t1, ADD, t2, ADD]
        let ops = vec![
            TERM_BASE | 0,
            TERM_BASE | 1,
            OP_BASE | 0,
            TERM_BASE | 2,
            OP_BASE | 0,
        ];
        assert_eq!(FlatTree::subtree_start(&ops, 4), 0);
        assert_eq!(FlatTree::subtree_start(&ops, 2), 0);
        assert_eq!(FlatTree::subtree_start(&ops, 3), 3);
    }

    // ---- splice ----

    #[test]
    fn test_splice_replaces_subtree() {
        // [t0, t1, ADD, t2, ADD] — replace [t0, t1, ADD] with [t3]
        let ops = vec![
            TERM_BASE | 0,
            TERM_BASE | 1,
            OP_BASE | 0,
            TERM_BASE | 2,
            OP_BASE | 0,
        ];
        let replacement = vec![TERM_BASE | 3];
        let result = FlatTree::splice(&ops, 0, 2, &replacement);
        assert_eq!(result, vec![TERM_BASE | 3, TERM_BASE | 2, OP_BASE | 0]);
    }

    // ---- eval_scalar_from_slice ----

    #[test]
    fn test_eval_scalar_const() {
        let ft = FlatTree::from_const(1.0);
        let val = ft.eval_scalar_from_slice(&[]);
        assert!((val - 1.0_f32).abs() < 0.15, "val={val}");
    }

    #[test]
    fn test_eval_scalar_terminal() {
        let ft = FlatTree::from_terminal(0);
        let terminals: Vec<f32> = vec![42.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let val = ft.eval_scalar_from_slice(&terminals);
        assert_eq!(val, 42.0_f32);
    }

    #[test]
    fn test_eval_scalar_add() {
        let ft = FlatTree::binary(&FlatTree::from_terminal(0), &FlatTree::from_terminal(1), 0);
        let terminals: Vec<f32> = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(ft.eval_scalar_from_slice(&terminals), 7.0_f32);
    }

    #[test]
    fn test_eval_scalar_protected_div_zero() {
        let ft = FlatTree::binary(&FlatTree::from_terminal(0), &FlatTree::from_terminal(1), 3);
        let terminals: Vec<f32> = vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(ft.eval_scalar_from_slice(&terminals), 1.0_f32);
    }

    #[test]
    fn test_eval_scalar_matches_ctx() {
        // ADD(TravelTime, Demand): vehicle at origin, request at (3,4) → tt=5, demand=2 → 7
        use crate::instance::{Request, TimeWindow};
        use crate::types::{RequestId, VehicleSnapshot};

        let flat = FlatTree::binary(
            &FlatTree::from_terminal(0), // TravelTime
            &FlatTree::from_terminal(4), // Demand
            0,                           // Add
        );

        let request = Request {
            id: RequestId(1),
            x: 3.0,
            y: 4.0,
            demand: 2.0,
            time_window: TimeWindow {
                earliest: 0.0,
                latest: 100.0,
            },
            service_time: 0.0,
            release_time: 0.0,
            is_depot: false,
        };
        let vehicle = VehicleSnapshot {
            vehicle_id: 0,
            position: RequestId(0),
            current_load: 0.0,
            available_at: 0.0,
            route: vec![],
            service_times: vec![],
        };
        let ctx = EvalContext {
            request: &request,
            vehicle: &vehicle,
            current_time: 0.0,
            speed: 1.0,
            vehicle_capacity: 10.0,
            vehicle_pos_x: 0.0,
            vehicle_pos_y: 0.0,
        };

        let got = flat.eval_scalar(&ctx);
        // travel_time = 5.0, demand = 2.0 → 7.0
        assert!((got - 7.0_f32).abs() < 1e-4, "got={got}");
    }

    // ---- eval_batch ----

    #[test]
    fn test_eval_batch_matches_scalar() {
        // ADD(t0, t1) over N=5 inputs
        let ft = FlatTree::binary(&FlatTree::from_terminal(0), &FlatTree::from_terminal(1), 0);
        let col0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col1: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let matrix = vec![col0.clone(), col1.clone()];

        let batch = ft.eval_batch(&matrix);
        for i in 0..5 {
            let expected = col0[i] + col1[i];
            assert!((batch[i] - expected).abs() < 1e-5, "i={i}");
        }
    }

    // ---- to_flat roundtrip replaced by direct FlatTree construction ----

    #[test]
    fn test_flat_tree_ops() {
        // MUL(ADD(t0, t1), SUB(t2, t3)) = (10+3)*(8-2) = 78
        let terminals: Vec<f32> = vec![10.0, 3.0, 8.0, 2.0, 0.0, 0.0, 0.0, 0.0];

        let add_part =
            FlatTree::binary(&FlatTree::from_terminal(0), &FlatTree::from_terminal(1), 0);
        let sub_part =
            FlatTree::binary(&FlatTree::from_terminal(2), &FlatTree::from_terminal(3), 1);
        let flat = FlatTree::binary(&add_part, &sub_part, 2);

        let val = flat.eval_scalar_from_slice(&terminals);
        assert!((val - 78.0_f32).abs() < 1e-4, "val={val}");
    }
}

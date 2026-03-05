// bench.rs — Conditional benchmark counters and helpers.
//
// When the `bench` feature is enabled we expose real AtomicU64 counters and
// helper functions. When disabled we provide lightweight no-op stubs so calls
// compile away with minimal overhead.

#[cfg(feature = "bench")]
mod real {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    // ---------------------------------------------------------------------------
    // Simulator loop
    // ---------------------------------------------------------------------------
    pub static TIME_PROCESS_EVENT_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_STRATEGY_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_EXECUTE_NS: AtomicU64 = AtomicU64::new(0);
    pub static TICK_COUNT: AtomicU64 = AtomicU64::new(0);

    // ---------------------------------------------------------------------------
    // process_next_event internals
    // ---------------------------------------------------------------------------
    pub static TIME_PNE_NEXT_TIME_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PNE_EVENT_DRAIN_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PNE_AUTO_REJECT_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PNE_RELEASE_CURSOR_NS: AtomicU64 = AtomicU64::new(0);

    // ---------------------------------------------------------------------------
    // ComposableStrategy::next_events phases
    // ---------------------------------------------------------------------------
    pub static TIME_PHASE1_SCAN_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PHASE1_ROUTING_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PHASE2_FEASIBILITY_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PHASE2_SCHEDULE_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_PHASE3_NS: AtomicU64 = AtomicU64::new(0);

    // ---------------------------------------------------------------------------
    // route_one internals
    // ---------------------------------------------------------------------------
    pub static TIME_MAKE_CTX_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_EVAL_BATCH_ROUTING_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_REJECT_EVAL_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROUTE_ONE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_ROUTE_ONE_BESTSCAN_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_ROUTE_ONE_MAKE_REJECT_NS: AtomicU64 = AtomicU64::new(0);

    // ---------------------------------------------------------------------------
    // eval_batch_for internals
    // ---------------------------------------------------------------------------
    pub static TIME_MATRIX_BUILD_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_EVAL_BATCH_INNER_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_AUG_CLONE_VEHICLES_NS: AtomicU64 = AtomicU64::new(0);
    pub static TIME_AUG_ACCUM_QUEUE_NS: AtomicU64 = AtomicU64::new(0);

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    pub fn reset() {
        for c in [
            &TIME_PROCESS_EVENT_NS,
            &TIME_STRATEGY_NS,
            &TIME_EXECUTE_NS,
            &TICK_COUNT,
            &TIME_PNE_NEXT_TIME_NS,
            &TIME_PNE_EVENT_DRAIN_NS,
            &TIME_PNE_AUTO_REJECT_NS,
            &TIME_PNE_RELEASE_CURSOR_NS,
            &TIME_PHASE1_SCAN_NS,
            &TIME_PHASE1_ROUTING_NS,
            &TIME_PHASE2_FEASIBILITY_NS,
            &TIME_PHASE2_SCHEDULE_NS,
            &TIME_PHASE3_NS,
            &TIME_MAKE_CTX_NS,
            &TIME_EVAL_BATCH_ROUTING_NS,
            &TIME_REJECT_EVAL_NS,
            &ROUTE_ONE_CALLS,
            &TIME_ROUTE_ONE_BESTSCAN_NS,
            &TIME_ROUTE_ONE_MAKE_REJECT_NS,
            &TIME_AUG_CLONE_VEHICLES_NS,
            &TIME_AUG_ACCUM_QUEUE_NS,
            &TIME_MATRIX_BUILD_NS,
            &TIME_EVAL_BATCH_INNER_NS,
        ] {
            c.store(0, Relaxed);
        }
    }

    #[inline(always)]
    pub fn elapsed_ns(t: std::time::Instant) -> u64 {
        t.elapsed().as_nanos() as u64
    }

    pub fn print_summary(total_ns: u64) {
        let ms = |ns: u64| ns as f64 / 1_000_000.0;
        let pct = |ns: u64| {
            if total_ns > 0 {
                100.0 * ns as f64 / total_ns as f64
            } else {
                0.0
            }
        };

        let ticks = TICK_COUNT.load(Relaxed);
        let t_event = TIME_PROCESS_EVENT_NS.load(Relaxed);
        let t_strat = TIME_STRATEGY_NS.load(Relaxed);
        let t_exec = TIME_EXECUTE_NS.load(Relaxed);

        let t_pne_next = TIME_PNE_NEXT_TIME_NS.load(Relaxed);
        let t_pne_drain = TIME_PNE_EVENT_DRAIN_NS.load(Relaxed);
        let t_pne_reject = TIME_PNE_AUTO_REJECT_NS.load(Relaxed);
        let t_pne_cursor = TIME_PNE_RELEASE_CURSOR_NS.load(Relaxed);

        let t_p1s = TIME_PHASE1_SCAN_NS.load(Relaxed);
        let t_p1r = TIME_PHASE1_ROUTING_NS.load(Relaxed);
        let t_p2f = TIME_PHASE2_FEASIBILITY_NS.load(Relaxed);
        let t_p2s = TIME_PHASE2_SCHEDULE_NS.load(Relaxed);
        let t_p3 = TIME_PHASE3_NS.load(Relaxed);

        let t_mkctx = TIME_MAKE_CTX_NS.load(Relaxed);
        let t_evbat = TIME_EVAL_BATCH_ROUTING_NS.load(Relaxed);
        let t_rejct = TIME_REJECT_EVAL_NS.load(Relaxed);
        let n_calls = ROUTE_ONE_CALLS.load(Relaxed);
        let t_bestscan = TIME_ROUTE_ONE_BESTSCAN_NS.load(Relaxed);
        let t_mkr = TIME_ROUTE_ONE_MAKE_REJECT_NS.load(Relaxed);

        let t_mat = TIME_MATRIX_BUILD_NS.load(Relaxed);
        let t_inner = TIME_EVAL_BATCH_INNER_NS.load(Relaxed);
        let t_aug_clone = TIME_AUG_CLONE_VEHICLES_NS.load(Relaxed);
        let t_aug_acc = TIME_AUG_ACCUM_QUEUE_NS.load(Relaxed);

        let strat_other = t_strat.saturating_sub(t_p1s + t_p1r + t_p2f + t_p2s + t_p3);
        let p1r_other = t_p1r.saturating_sub(t_mkctx + t_evbat + t_rejct + t_bestscan + t_mkr);
        let evb_other = t_evbat.saturating_sub(t_mat + t_inner);
        let pne_other =
            t_event.saturating_sub(t_pne_next + t_pne_drain + t_pne_reject + t_pne_cursor);

        eprintln!();
        eprintln!(
            "╔══ TIMING BREAKDOWN  total={:.2}ms  ticks={} ══╗",
            ms(total_ns),
            ticks
        );
        eprintln!(
            "║ process_next_event     {:6.2}ms  {:5.1}%",
            ms(t_event),
            pct(t_event)
        );
        eprintln!(
            "║   next-time compute    {:6.2}ms  {:5.1}%",
            ms(t_pne_next),
            pct(t_pne_next)
        );
        eprintln!(
            "║   event drain          {:6.2}ms  {:5.1}%",
            ms(t_pne_drain),
            pct(t_pne_drain)
        );
        eprintln!(
            "║   auto_reject          {:6.2}ms  {:5.1}%",
            ms(t_pne_reject),
            pct(t_pne_reject)
        );
        eprintln!(
            "║   release cursor       {:6.2}ms  {:5.1}%",
            ms(t_pne_cursor),
            pct(t_pne_cursor)
        );
        eprintln!(
            "║   other in pne         {:6.2}ms  {:5.1}%",
            ms(pne_other),
            pct(pne_other)
        );
        eprintln!(
            "║ strategy.next_events   {:6.2}ms  {:5.1}%",
            ms(t_strat),
            pct(t_strat)
        );
        eprintln!(
            "║   phase1 scan          {:6.2}ms  {:5.1}%",
            ms(t_p1s),
            pct(t_p1s)
        );
        eprintln!(
            "║   phase1 route_one loop {:5.2}ms  {:5.1}%",
            ms(t_p1r),
            pct(t_p1r)
        );
        eprintln!(
            "║     make_ctx           {:6.2}ms  {:5.1}%",
            ms(t_mkctx),
            pct(t_mkctx)
        );
        eprintln!(
            "║     eval_batch_for     {:6.2}ms  {:5.1}%",
            ms(t_evbat),
            pct(t_evbat)
        );
        eprintln!(
            "║       matrix build     {:6.2}ms  {:5.1}%",
            ms(t_mat),
            pct(t_mat)
        );
        eprintln!(
            "║       eval_batch(SIMD) {:6.2}ms  {:5.1}%",
            ms(t_inner),
            pct(t_inner)
        );
        eprintln!(
            "║       other in evb_for {:6.2}ms  {:5.1}%",
            ms(evb_other),
            pct(evb_other)
        );
        eprintln!(
            "║     reject eval_ctx    {:6.2}ms  {:5.1}%",
            ms(t_rejct),
            pct(t_rejct)
        );
        eprintln!(
            "║     best-index scan    {:6.2}ms  {:5.1}%",
            ms(t_bestscan),
            pct(t_bestscan)
        );
        eprintln!(
            "║     make_reject_ctx    {:6.2}ms  {:5.1}%",
            ms(t_mkr),
            pct(t_mkr)
        );
        eprintln!(
            "║     aug clone vehicles {:6.2}ms  {:5.1}%",
            ms(t_aug_clone),
            pct(t_aug_clone)
        );
        eprintln!(
            "║     aug accumulate     {:6.2}ms  {:5.1}%",
            ms(t_aug_acc),
            pct(t_aug_acc)
        );
        eprintln!(
            "║     other in route_one {:6.2}ms  {:5.1}%   ({} calls)",
            ms(p1r_other),
            pct(p1r_other),
            n_calls
        );
        eprintln!(
            "║   phase2 feasibility   {:6.2}ms  {:5.1}%",
            ms(t_p2f),
            pct(t_p2f)
        );
        eprintln!(
            "║   phase2 schedule      {:6.2}ms  {:5.1}%",
            ms(t_p2s),
            pct(t_p2s)
        );
        eprintln!(
            "║   phase3 wait calc     {:6.2}ms  {:5.1}%",
            ms(t_p3),
            pct(t_p3)
        );
        eprintln!(
            "║   other in strategy    {:6.2}ms  {:5.1}%",
            ms(strat_other),
            pct(strat_other)
        );
        eprintln!(
            "║ execute_actions        {:6.2}ms  {:5.1}%",
            ms(t_exec),
            pct(t_exec)
        );
        eprintln!(
            "║ unaccounted            {:6.2}ms  {:5.1}%",
            ms(total_ns.saturating_sub(t_event + t_strat + t_exec)),
            pct(total_ns.saturating_sub(t_event + t_strat + t_exec))
        );
        eprintln!("╚═══════════════════════════════════════════════════╝");
        eprintln!();
    }
}

#[cfg(not(feature = "bench"))]
mod stub {
    use std::sync::atomic::Ordering;

    // Minimal no-op stubs when bench feature is disabled. These keep the same
    // identifiers so calls remain valid but compile out measurement work.
    pub struct BenchCounter;
    impl BenchCounter {
        pub const fn new() -> Self {
            BenchCounter
        }
        pub fn fetch_add(&self, _v: u64, _o: Ordering) {}
        pub fn load(&self, _o: Ordering) -> u64 {
            0
        }
        pub fn store(&self, _v: u64, _o: Ordering) {}
    }

    pub static TIME_PROCESS_EVENT_NS: BenchCounter = BenchCounter::new();
    pub static TIME_STRATEGY_NS: BenchCounter = BenchCounter::new();
    pub static TIME_EXECUTE_NS: BenchCounter = BenchCounter::new();
    pub static TICK_COUNT: BenchCounter = BenchCounter::new();

    pub static TIME_PNE_NEXT_TIME_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PNE_EVENT_DRAIN_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PNE_AUTO_REJECT_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PNE_RELEASE_CURSOR_NS: BenchCounter = BenchCounter::new();

    pub static TIME_PHASE1_SCAN_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PHASE1_ROUTING_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PHASE2_FEASIBILITY_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PHASE2_SCHEDULE_NS: BenchCounter = BenchCounter::new();
    pub static TIME_PHASE3_NS: BenchCounter = BenchCounter::new();

    pub static TIME_MAKE_CTX_NS: BenchCounter = BenchCounter::new();
    pub static TIME_EVAL_BATCH_ROUTING_NS: BenchCounter = BenchCounter::new();
    pub static TIME_REJECT_EVAL_NS: BenchCounter = BenchCounter::new();
    pub static ROUTE_ONE_CALLS: BenchCounter = BenchCounter::new();
    pub static TIME_ROUTE_ONE_BESTSCAN_NS: BenchCounter = BenchCounter::new();
    pub static TIME_ROUTE_ONE_MAKE_REJECT_NS: BenchCounter = BenchCounter::new();

    pub static TIME_MATRIX_BUILD_NS: BenchCounter = BenchCounter::new();
    pub static TIME_EVAL_BATCH_INNER_NS: BenchCounter = BenchCounter::new();
    pub static TIME_AUG_CLONE_VEHICLES_NS: BenchCounter = BenchCounter::new();
    pub static TIME_AUG_ACCUM_QUEUE_NS: BenchCounter = BenchCounter::new();

    pub fn reset() {}
    pub fn elapsed_ns(_t: std::time::Instant) -> u64 {
        0
    }
    pub fn print_summary(_total_ns: u64) {}
}

// Re-export selected symbols at crate root so call-sites keep using
// `crate::bench::...` unchanged.
#[cfg(feature = "bench")]
pub use real::*;
#[cfg(not(feature = "bench"))]
pub use stub::*;

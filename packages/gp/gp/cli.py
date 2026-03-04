#!/usr/bin/env python3
"""CLI to run the GP (Rust-backed) experiment and save history.

This is a relocated copy of the former `scripts/run_gp_rust.py` so it can be
invoked from the `gp` package directly (for example
`python -m gp.cli`).
"""

from __future__ import annotations

import argparse
import csv

from dvrptw.instance import load_vrpr_csv

from .gp_rust import run_gp_rust


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="packages/dvrptw/tests/data/h100rc101.csv")
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--gen", type=int, default=6)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--out-csv", default="gp_history.csv")
    p.add_argument("--out-png", default="gp_history.png")
    args = p.parse_args()

    print("Loading instance:", args.instance)
    instance = load_vrpr_csv(
        args.instance, truck_speed=1.0, truck_capacity=30.0, num_trucks=2
    )
    print("Instance loaded: requests=", len(instance.requests))

    print(f"Running GP: pop={args.pop} gen={args.gen} depth={args.depth}")
    best, best_obj, history = run_gp_rust(
        instance,
        pop_size=args.pop,
        generations=args.gen,
        max_depth=args.depth,
    )

    print("Best objective (travel_cost, rejected):", best_obj)

    # write CSV
    keys = [
        "gen",
        "best_scalar",
        "best_cost",
        "best_rej",
        "mean_scalar",
        "mean_cost",
        "mean_rej",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k) for k in keys})

    print("Wrote history to", args.out_csv)

    # attempt to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        gens = [r["gen"] for r in history]
        bests = [r["best_scalar"] for r in history]
        mean_cost = [r["mean_cost"] for r in history]
        mean_rej = [r["mean_rej"] for r in history]

        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.plot(gens, bests, label="best_scalar", marker="o")
        ax1.set_xlabel("generation")
        ax1.set_ylabel("scalar (lower better)")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(gens, mean_cost, label="mean_cost", color="C1", linestyle="--")
        ax2.plot(gens, mean_rej, label="mean_rej", color="C2", linestyle=":")
        ax2.set_ylabel("mean cost / mean rejected")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.title("GP progress")
        plt.tight_layout()
        plt.savefig(args.out_png)
        print("Wrote plot to", args.out_png)
    except Exception as e:  # pragma: no cover - optional plotting
        print("matplotlib not available or plotting failed:", e)


if __name__ == "__main__":
    main()

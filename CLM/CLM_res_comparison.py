#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def load_nested_selfsup(path):
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for ca, sub in raw.items():
        for cb, payload in sub.items():
            key = f"{int(ca)}_to_{int(cb)}"
            if not isinstance(payload, dict):
                continue
            times = payload.get("time_pair", [])
            out[key] = list(map(float, times))
    return out


def load_flat_gt(path):
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            out[k] = list(map(float, v))
        elif isinstance(v, dict) and "time_pair" in v:
            out[k] = list(map(float, v["time_pair"]))
    return out


def kde_on_grid(samples, bandwidth, x_grid):
    if len(samples) < 2:
        return None
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(np.asarray(samples).reshape(-1, 1))
    log_d = kde.score_samples(x_grid.reshape(-1, 1))
    return np.exp(log_d)


def build_grid(values_a, values_b, pad=0.05, num=400):
    # Fixed axis from 0 to 200 seconds
    return np.linspace(0, 200, num)  # <<< changed (your preference)


def main():
    p = argparse.ArgumentParser(description="Compare KDEs from two transition JSON files.")
    p.add_argument("json_a", type=str, help="Path to self_supervised_clm_time_transition.json (nested)")
    p.add_argument("json_b", type=str, help="Path to MTC_time_transition_GT.json (flat)")
    p.add_argument("--directions", type=str, nargs="*",
                   help="Specific directions to plot, e.g. 1_to_2 2_to_1. If omitted, auto-detect all present.")
    p.add_argument("--bandwidth", type=float, default=3.0,
                   help="KDE bandwidth in SECONDS.")
    p.add_argument("--title_a", type=str, default="Self-supervised", help="Legend label for JSON A.")
    p.add_argument("--title_b", type=str, default="MC-GT", help="Legend label for JSON B.")
    p.add_argument("--outdir", type=str, default="", help="If provided, save PNGs (one per direction).")
    p.add_argument("--show", action="store_true", help="Show interactive windows.")
    args = p.parse_args()

    A = load_nested_selfsup(Path(args.json_a))
    B = load_flat_gt(Path(args.json_b))

    directions = sorted(set(A.keys()).union(B.keys())) if not args.directions else args.directions

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for d in directions:
        a_vals = np.array(A.get(d, []), dtype=float)
        b_vals = np.array(B.get(d, []), dtype=float)

        if len(a_vals) < 2 and len(b_vals) < 2:
            print(f"[skip] {d}: not enough data in either file.")
            continue

        grid = build_grid(a_vals, b_vals, pad=0.05, num=400)

        dens_a = kde_on_grid(a_vals, bandwidth=args.bandwidth, x_grid=grid) if len(a_vals) >= 2 else None
        dens_b = kde_on_grid(b_vals, bandwidth=args.bandwidth, x_grid=grid) if len(b_vals) >= 2 else None

        plt.figure(figsize=(8, 4.2))
        if dens_a is not None:
            plt.plot(grid, dens_a, label=f"{args.title_a} ({d})", linewidth=2)
            plt.fill_between(grid, dens_a, alpha=0.2)
        else:
            plt.plot([], [], label=f"{args.title_a} ({d}) [insufficient data]")

        if dens_b is not None:
            plt.plot(grid, dens_b, label=f"{args.title_b} ({d})", linewidth=2)
            plt.fill_between(grid, dens_b, alpha=0.2)
        else:
            plt.plot([], [], label=f"{args.title_b} ({d}) [insufficient data]")

        # --- Rug plots at actual sample positions, clipped to [0, 200] ---
        if len(a_vals):
            a_clip = a_vals[(a_vals >= 0) & (a_vals <= 200)]  # <<< changed
            if len(a_clip):
                plt.plot(a_clip, np.full_like(a_clip, -0.001), "|", alpha=0.7)  # <<< changed
        if len(b_vals):
            b_clip = b_vals[(b_vals >= 0) & (b_vals <= 200)]  # <<< changed
            if len(b_clip):
                plt.plot(b_clip, np.full_like(b_clip, -0.002), "|", alpha=0.7)  # <<< changed

        plt.title(f"KDE Comparison: {d}")
        plt.xlabel("Time transition (seconds)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 200)  # <<< changed: lock axis
        plt.tight_layout()

        if outdir:
            out_path = outdir / f"kde_compare_{d}.png"
            plt.savefig(out_path, dpi=200)
            print(f"[saved] {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close()

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


DEFAULT_GRID_DIRECTIONS = [
    "1_to_2",
    "2_to_1",
    "2_to_3",
    "3_to_2",
    "3_to_4",
    "4_to_3",
]


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
    return np.linspace(0, 100, num)  # <<< changed (your preference)


def plot_kde_comparison_on_axis(
    ax,
    *,
    direction,
    a_vals,
    b_vals,
    bandwidth,
    title_a,
    title_b,
    x_min=0,
    x_max=100,
    num=400,
):
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)

    if len(a_vals) < 2 and len(b_vals) < 2:
        ax.set_title(f"{direction} (insufficient data)")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Time transition (seconds)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        return False

    grid = np.linspace(x_min, x_max, num)
    dens_a = kde_on_grid(a_vals, bandwidth=bandwidth, x_grid=grid) if len(a_vals) >= 2 else None
    dens_b = kde_on_grid(b_vals, bandwidth=bandwidth, x_grid=grid) if len(b_vals) >= 2 else None

    if dens_a is not None:
        ax.plot(grid, dens_a, label=f"{title_a}", linewidth=2)
        ax.fill_between(grid, dens_a, alpha=0.2)
    else:
        ax.plot([], [], label=f"{title_a} [insufficient]")

    if dens_b is not None:
        ax.plot(grid, dens_b, label=f"{title_b}", linewidth=2)
        ax.fill_between(grid, dens_b, alpha=0.2)
    else:
        ax.plot([], [], label=f"{title_b} [insufficient]")

    # Rug plots at sample positions, clipped to axis range
    if len(a_vals):
        a_clip = a_vals[(a_vals >= x_min) & (a_vals <= x_max)]
        if len(a_clip):
            ax.plot(a_clip, np.full_like(a_clip, -0.001), "|", alpha=0.7)
    if len(b_vals):
        b_clip = b_vals[(b_vals >= x_min) & (b_vals <= x_max)]
        if len(b_clip):
            ax.plot(b_clip, np.full_like(b_clip, -0.002), "|", alpha=0.7)

    ax.set_title(direction)
    ax.set_xlabel("Time transition (seconds)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    return True


def main():
    p = argparse.ArgumentParser(description="Compare KDEs from two transition JSON files.")
    p.add_argument("json_a", type=str, help="Path to self_supervised_clm_time_transition.json (nested)")
    p.add_argument("json_b", type=str, help="Path to MTC_time_transition_GT.json (flat)")
    p.add_argument("--directions", type=str, nargs="*",
                   help="Specific directions to plot, e.g. 1_to_2 2_to_1. If omitted, auto-detect all present.")
    p.add_argument("--grid", action="store_true",
                   help="Plot multiple directions into one figure with subplots (default 2x3 for 6 directions).")
    p.add_argument("--grid-outname", type=str, default="kde_compare_grid.png",
                   help="Filename for the combined grid figure (used with --grid).")
    p.add_argument("--bandwidth", type=float, default=3.0,
                   help="KDE bandwidth in SECONDS.")
    p.add_argument("--title_a", type=str, default="Self-supervised", help="Legend label for JSON A.")
    p.add_argument("--title_b", type=str, default="MC-GT", help="Legend label for JSON B.")
    p.add_argument("--outdir", type=str, default="", help="If provided, save PNGs (one per direction).")
    p.add_argument("--show", action="store_true", help="Show interactive windows.")
    args = p.parse_args()

    A = load_nested_selfsup(Path(args.json_a))
    B = load_flat_gt(Path(args.json_b))

    if args.directions:
        directions = args.directions
    elif args.grid:
        # In grid mode, default to the 6 expected camera directions.
        directions = DEFAULT_GRID_DIRECTIONS
    else:
        directions = sorted(set(A.keys()).union(B.keys()))

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    if args.grid:
        # Heuristic layout: 2x3 for 6; otherwise, pack into 3 columns.
        n = len(directions)
        ncols = 3 if n >= 3 else max(1, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

        for idx, d in enumerate(directions):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            a_vals = A.get(d, [])
            b_vals = B.get(d, [])
            plotted = plot_kde_comparison_on_axis(
                ax,
                direction=d,
                a_vals=a_vals,
                b_vals=b_vals,
                bandwidth=args.bandwidth,
                title_a=args.title_a,
                title_b=args.title_b,
                x_min=0,
                x_max=100,
                num=400,
            )
            if not plotted:
                print(f"[info] {d}: not enough data in either file.")

        # Hide any unused axes.
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")

        # One shared legend to avoid clutter.
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        fig.suptitle("KDE Comparison (All Directions)")
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))

        if outdir:
            out_path = outdir / args.grid_outname
            fig.savefig(out_path, dpi=200)
            print(f"[saved] {out_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)
    else:
        for d in directions:
            a_vals = A.get(d, [])
            b_vals = B.get(d, [])

            fig, ax = plt.subplots(figsize=(8, 6))
            plotted = plot_kde_comparison_on_axis(
                ax,
                direction=d,
                a_vals=a_vals,
                b_vals=b_vals,
                bandwidth=args.bandwidth,
                title_a=args.title_a,
                title_b=args.title_b,
                x_min=0,
                x_max=100,
                num=400,
            )
            if plotted:
                ax.set_title(f"KDE Comparison: {d}")
            fig.tight_layout()

            if outdir:
                out_path = outdir / f"kde_compare_{d}.png"
                fig.savefig(out_path, dpi=200)
                print(f"[saved] {out_path}")

            if args.show:
                plt.show()
            else:
                plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()

# example usage (single): python CLM_res_comparison.py cam_pair.json MTC_time_transition_GT.json --directions 4_to_3 --outdir .
# example usage (grid):   python CLM_res_comparison.py cam_pair.json MTC_time_transition_GT.json --grid --outdir .
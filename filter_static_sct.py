#!/usr/bin/env python3
"""
Filter static tracklets from a MOT-style detection/track file.

Input format per line (whitespace separated):
<frame_id> <track_id> <x_min> <y_min> <x_max> <y_max>

Example:
1 1 297.8 1115.7 883.5 1558.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrackStats:
    track_id: int
    n: int
    frames_span: int
    max_center_disp: float
    mean_center_step: float
    std_center: float
    max_size_change_ratio: float  # max(|w-w0|/w0, |h-h0|/h0) over time


def read_tracks(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Returns dict: track_id -> list of (frame_id, x1, y1, x2, y2)
    """
    tracks: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                raise ValueError(f"Line {ln}: expected 6 columns, got {len(parts)}: {s}")
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x1, y1, x2, y2 = map(float, parts[2:6])
            tracks.setdefault(track_id, []).append((frame_id, x1, y1, x2, y2))
    # sort each track by frame_id
    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])
    return tracks


def compute_stats(
    tid: int,
    dets: List[Tuple[int, float, float, float, float]],
) -> TrackStats:
    frames = np.array([d[0] for d in dets], dtype=np.int64)
    x1 = np.array([d[1] for d in dets], dtype=np.float64)
    y1 = np.array([d[2] for d in dets], dtype=np.float64)
    x2 = np.array([d[3] for d in dets], dtype=np.float64)
    y2 = np.array([d[4] for d in dets], dtype=np.float64)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = np.clip(x2 - x1, 1e-6, None)
    h = np.clip(y2 - y1, 1e-6, None)

    # Center displacement relative to first detection
    dx0 = cx - cx[0]
    dy0 = cy - cy[0]
    disp0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
    max_center_disp = float(np.max(disp0))

    # Mean per-step motion (between consecutive available frames; gaps allowed)
    if len(cx) >= 2:
        dcx = np.diff(cx)
        dcy = np.diff(cy)
        step = np.sqrt(dcx * dcx + dcy * dcy)
        mean_center_step = float(np.mean(step))
    else:
        mean_center_step = 0.0

    # Overall spatial spread of centers
    std_center = float(np.sqrt(np.var(cx) + np.var(cy)))

    # Size stability (relative to first bbox)
    w0, h0 = w[0], h[0]
    size_change_ratio = np.maximum(np.abs(w - w0) / w0, np.abs(h - h0) / h0)
    max_size_change_ratio = float(np.max(size_change_ratio))

    frames_span = int(frames[-1] - frames[0] + 1)

    return TrackStats(
        track_id=tid,
        n=int(len(dets)),
        frames_span=frames_span,
        max_center_disp=max_center_disp,
        mean_center_step=mean_center_step,
        std_center=std_center,
        max_size_change_ratio=max_size_change_ratio,
    )


def is_static(
    st: TrackStats,
    min_len: int,
    max_disp_px: float,
    max_mean_step_px: float,
    max_std_center_px: float,
    max_size_change_ratio: float,
) -> bool:
    """
    Static tracklet definition (simple, robust):
    - must have at least min_len detections
    - center does not move much: max_center_disp <= max_disp_px
    - and typical step is small: mean_center_step <= max_mean_step_px
    - and center spread small: std_center <= max_std_center_px
    - and bbox size stable: max_size_change_ratio <= max_size_change_ratio
    """
    if st.n < min_len:
        return False
    return (
        st.max_center_disp <= max_disp_px
        and st.mean_center_step <= max_mean_step_px
        and st.std_center <= max_std_center_px
        and st.max_size_change_ratio <= max_size_change_ratio
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, type=Path, help="Input tracking txt")
    ap.add_argument(
        "--out_file",
        type=Path,
        default=None,
        help="Optional output txt with static tracklets removed",
    )
    ap.add_argument("--min_len", type=int, default=10, help="Minimum detections per tracklet")
    ap.add_argument("--max_disp_px", type=float, default=5.0, help="Max center displacement (px)")
    ap.add_argument("--max_mean_step_px", type=float, default=1.0, help="Max mean step (px)")
    ap.add_argument("--max_std_center_px", type=float, default=2.0, help="Max center spread std (px)")
    ap.add_argument(
        "--max_size_change_ratio",
        type=float,
        default=0.15,
        help="Max relative bbox size change (e.g., 0.15 = 15%%)",
    )
    ap.add_argument(
        "--report_topk",
        type=int,
        default=30,
        help="Print details for up to top-k static tracks (sorted by n desc)",
    )
    args = ap.parse_args()

    tracks = read_tracks(args.in_file)

    stats: List[TrackStats] = []
    for tid, dets in tracks.items():
        stats.append(compute_stats(tid, dets))

    static_ids = [
        st.track_id
        for st in stats
        if is_static(
            st,
            min_len=args.min_len,
            max_disp_px=args.max_disp_px,
            max_mean_step_px=args.max_mean_step_px,
            max_std_center_px=args.max_std_center_px,
            max_size_change_ratio=args.max_size_change_ratio,
        )
    ]
    static_ids_set = set(static_ids)

    # Report
    static_stats = [st for st in stats if st.track_id in static_ids_set]
    static_stats.sort(key=lambda s: (-s.n, s.max_center_disp))

    print(f"Total tracks: {len(stats)}")
    print(f"Static tracks: {len(static_stats)}")
    print("Static track IDs:", static_ids)

    if static_stats:
        print("\nTop static tracks (track_id, n, span, max_disp, mean_step, std_center, max_size_change_ratio):")
        for st in static_stats[: args.report_topk]:
            print(
                f"{st.track_id:6d}  n={st.n:4d}  span={st.frames_span:4d}  "
                f"max_disp={st.max_center_disp:7.3f}  mean_step={st.mean_center_step:7.3f}  "
                f"std={st.std_center:7.3f}  sizechg={st.max_size_change_ratio:6.3f}"
            )

    # Optionally write filtered file
    if args.out_file is not None:
        with args.in_file.open("r", encoding="utf-8") as fin, args.out_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    fout.write(line)
                    continue
                parts = s.split()
                if len(parts) < 2:
                    fout.write(line)
                    continue
                tid = int(float(parts[1]))
                if tid in static_ids_set:
                    continue
                fout.write(line)
        print(f"\nWrote filtered file (static removed): {args.out_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding: utf-8

"""
Compute inter-camera time transitions (in frames) from multi-camera GT.
Only adjacent cameras are considered: (1<->2), (2<->3), (3<->4).

GT line format (space-separated):
cam_id track_id frame_id x1 y1 w h xworld yworld
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ADJACENT_PAIRS = [(1, 2), (2, 3), (3, 4)]  # define adjacency once


def read_mcgt(gt_path):
    """
    Read multi-camera tracking ground truth.

    Each line is:
        cam_id track_id frame_id x1 y1 w h xworld yworld

    Returns:
        dict[int, list[dict]]: track_id -> list of {cam_id, frame_num, bbox}
    """
    mcgt_data = defaultdict(list)
    with open(gt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect 9 tokens; tolerate extra tokens by slicing
            if len(parts) < 9:
                continue

            cam_id, track_id, frame_num = map(int, parts[:3])
            x1, y1, width, height = map(float, parts[3:7])
            # xworld, yworld = map(float, parts[7:9])  # not used here

            mcgt_data[track_id].append(
                {
                    "cam_id": cam_id,
                    "frame_num": frame_num,
                    "bbox": [x1, y1, width, height],
                }
            )
    return mcgt_data


def _init_transitions(cam_ids):
    transitions = {}
    for a, b in ADJACENT_PAIRS:
        if a in cam_ids and b in cam_ids:
            transitions[f"{a}_to_{b}"] = []
            transitions[f"{b}_to_{a}"] = []
    return transitions


def get_time_transition_gt(mcgt_data, cam_id_list=(1, 2, 3, 4)):
    """
    For each track, compute time gaps (in frames) between adjacent cameras.
    If the last frame in cam A is earlier than the first frame in cam B,
    we record (min_frame_B - max_frame_A) under 'A_to_B'.

    Overlapping or out-of-order appearances are ignored for that direction.

    Returns:
        dict[str, list[int]]: e.g., {'1_to_2':[...], '2_to_1':[...], ...}
    """
    cam_transition = _init_transitions(set(cam_id_list))

    for track_id, entries in mcgt_data.items():
        # Aggregate per-camera min/max frame for this track
        per_cam = {}
        for e in entries:
            cam = e["cam_id"]
            if cam not in cam_id_list:
                continue
            f = e["frame_num"]
            if cam not in per_cam:
                per_cam[cam] = {"min": f, "max": f}
            else:
                if f < per_cam[cam]["min"]:
                    per_cam[cam]["min"] = f
                if f > per_cam[cam]["max"]:
                    per_cam[cam]["max"] = f

        # For each adjacent pair, try both directions
        for a, b in ADJACENT_PAIRS:
            if a in per_cam and b in per_cam:
                # a -> b
                gap_ab = per_cam[b]["min"] - per_cam[a]["max"]
                if gap_ab >= 0:
                    cam_transition[f"{a}_to_{b}"].append(int(gap_ab)/15)

                # b -> a
                gap_ba = per_cam[a]["min"] - per_cam[b]["max"]
                if gap_ba >= 0:
                    cam_transition[f"{b}_to_{a}"].append(int(gap_ba)/15)

    return cam_transition


def summarize(cam_transition):
    """Create a compact summary (count, mean, median) per direction."""
    import statistics

    summary = {}
    for k, vals in cam_transition.items():
        if len(vals) == 0:
            summary[k] = {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        else:
            summary[k] = {
                "count": len(vals),
                "mean": float(sum(vals) / len(vals)),
                "median": float(statistics.median(vals)),
                "min": int(min(vals)),
                "max": int(max(vals)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute inter-camera time transitions from MC GT.")
    parser.add_argument("gt_path", type=str, help="Path to ground-truth txt file.")
    parser.add_argument("--save_json", action="store_true", help="Flag to save transitions & summary to JSON file.")
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4], help="Cameras to include (default: 1 2 3 4)")
    args = parser.parse_args()

    mcgt_data = read_mcgt(args.gt_path)
    cam_transition = get_time_transition_gt(mcgt_data, cam_id_list=tuple(args.cams))
    summary = summarize(cam_transition)

    print("\n=== Transition Summary (gaps in frames) ===")
    for k in sorted(cam_transition.keys()):
        s = summary[k]
        print(
            f"{k:8s}  count={s['count']:4d}  "
            f"mean={s['mean']:.2f}  median={s['median']:.2f}  "
            f"min={s['min']}  max={s['max']}"
            if s["count"] > 0
            else f"{k:8s}  count=0"
        )

    if args.save_json:
        with open(Path('MTC_time_transition_GT.json'), "w") as f:
            json.dump(cam_transition, f)
        print(f"\nSaved transitions & summary to: {args.save_json}")


if __name__ == "__main__":
    main()

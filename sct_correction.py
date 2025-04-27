#!/usr/bin/env python3
"""
sct_correction.py ────────────────

An interactive CLI tool for post‑processing MOT tracking results.

Features
========
1. **Integrate (merge) two tracklets**  
   *Prompts*: `i`, then `trackid1`, `trackid2`  
   The detections of `trackid2` are appended to `trackid1`; missing frames
   between them (and inside the merged track) are filled by linear
   interpolation (up to `--max-gap` frames). `trackid2` is removed.
2. **Delete a tracklet**  
   *Prompts*: `d`, then `trackid`.
3. **Write results & optionally visualise**  
   *Prompt*: `w` – saves a new TXT file plus an MP4 visualisation showing the
   edited tracks.
4. **Quit without saving**  
   *Prompt*: `q`.

Example
-------
```bash
python interactive_tracklet_editor.py tracking.txt --img_dir /dataset/detection/imagesSB \
       --img_pattern "%06d.jpg" --fps 15 --output_prefix imagesSB_edited
```
This will open an interactive session; when you type `w` the script produces
`imagesSB_edited.txt` + `imagesSB_edited.mp4`.

Dependencies: `opencv‑python` ≥4.8, `numpy`, `tqdm` (for progress bar).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

BBox = List[float]                 # [x1, y1, x2, y2]
Det  = Tuple[int, BBox, int]       # (frame, bbox, cls)
Track = List[Det]                  # list of detections sorted by frame
Tracks = Dict[int, Track]          # track_id -> Track

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def linear_interpolate(b1: BBox, b2: BBox, alpha: float) -> BBox:
    """Linear interpolation between *b1* and *b2* (alpha∈[0,1])."""
    return [b1[i] + alpha * (b2[i] - b1[i]) for i in range(4)]


def interpolate_track(track: Track, max_gap: int = 10) -> Track:
    """Fill small gaps inside *track* by linear bbox interpolation."""
    if len(track) < 2:
        return track
    full: Track = []
    for (f_i, bb_i, cls_i), (f_j, bb_j, _) in zip(track[:-1], track[1:]):
        full.append((f_i, bb_i, cls_i))
        gap = f_j - f_i
        if 1 < gap <= max_gap:  # only interpolate reasonable gaps
            for f in range(f_i + 1, f_j):
                a = (f - f_i) / (f_j - f_i)
                full.append((f, linear_interpolate(bb_i, bb_j, a), cls_i))
    full.append(track[-1])
    return sorted(full, key=lambda x: x[0])


def merge_tracks(tracks: Tracks, id_keep: int, id_merge: int, max_gap: int) -> None:
    """Merge *id_merge* into *id_keep* and interpolate inside the new track."""
    if id_keep not in tracks or id_merge not in tracks:
        print("✗ One of the specified track IDs does not exist.")
        return
    merged = tracks[id_keep] + tracks[id_merge]
    merged = sorted(merged, key=lambda x: x[0])
    merged = interpolate_track(merged, max_gap)
    tracks[id_keep] = merged
    del tracks[id_merge]
    print(f"✓ Track {id_merge} merged into {id_keep} ‑ total detections: {len(merged)}")


def load_tracks(txt_path: str) -> Tracks:
    """Read MOT text file → dict of tracks."""
    tracks: Tracks = {}
    with open(txt_path, "r") as fh:
        for ln in fh:
            if ln.strip() == "":
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            f, tid = int(float(parts[0])), int(float(parts[1]))
            bb = list(map(float, parts[2:6]))
            cls = int(float(parts[6]))
            tracks.setdefault(tid, []).append((f, bb, cls))
    # sort
    for tid in tracks:
        tracks[tid] = sorted(tracks[tid], key=lambda x: x[0])
    return tracks


def save_tracks(tracks: Tracks, out_txt: str) -> None:
    with open(out_txt, "w") as fh:
        for tid, dets in tracks.items():
            for f, bb, cls in dets:
                fh.write(f"{f} {tid} {bb[0]:.2f} {bb[1]:.2f} {bb[2]:.2f} {bb[3]:.2f} {cls}\n")
    print(f"✓ Results saved to {out_txt}")


def render_video(tracks: Tracks, img_dir: str, img_pattern: str, fps: int, out_mp4: str):
    all_frames = sorted({f for trk in tracks.values() for f, _, _ in trk})
    if not all_frames:
        print("✗ No frames found – skipping video.")
        return

    first_img_path = os.path.join(img_dir, img_pattern % all_frames[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"✗ Cannot open first image at {first_img_path}; video skipped.")
        return

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))

    colour_map = {}
    np.random.seed(42)

    print("Rendering video …")
    for f in tqdm(all_frames):
        img_path = os.path.join(img_dir, img_pattern % f)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        # draw
        for tid, dets in tracks.items():
            for fr, bb, cls in dets:
                if fr != f:
                    continue
                colour = colour_map.setdefault(tid, tuple(int(c) for c in np.random.randint(0, 255, 3)))
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, str(tid), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        vw.write(frame)
    vw.release()
    print(f"✓ Video written to {out_mp4}")

# ────────────────────────────────────────────────────────────────────────────────
# Main interactive loop
# ────────────────────────────────────────────────────────────────────────────────

def interactive_session(tracks: Tracks, args):
    print("Loaded", len(tracks), "tracks.")
    while True:
        cmd = input("[i]ntegrate, [d]elete, [w]rite & quit, [q]uit: ").strip().lower()
        if cmd == "i":
            try:
                t1 = int(input("  trackid1 (kept): "))
                t2 = int(input("  trackid2 (merged → trackid1): "))
                merge_tracks(tracks, t1, t2, args.max_gap)
            except ValueError:
                print("✗ Invalid input; IDs must be integers.")
        elif cmd == "d":
            try:
                t = int(input("  trackid to delete: "))
                if tracks.pop(t, None) is not None:
                    print(f"✓ Track {t} deleted.")
                else:
                    print("✗ Track ID not found.")
            except ValueError:
                print("✗ Invalid input; ID must be integer.")
        elif cmd == "w":
            out_txt  = f"{args.output_prefix}.txt"
            out_mp4  = f"{args.output_prefix}.mp4"
            save_tracks(tracks, out_txt)
            render_video(tracks, args.img_dir, args.img_pattern, args.fps, out_mp4)
            break
        elif cmd == "q":
            print("Exiting without saving …")
            break
        else:
            print("✗ Unknown command.")

# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Interactive post‑processing for MOT tracklets.")
    ap.add_argument("tracking_txt", help="Input tracking result file (frame trackID x1 y1 x2 y2 class).")
    # ap.add_argument("--img_dir", required=True, help="Directory containing sequence frames.")
    ap.add_argument("--img_pattern", default="img%06d.jpg", help="Printf‑style pattern for image names (default: %%06d.jpg).")
    # ap.add_argument("--fps", type=int, default=15, help="FPS for output video (default 15).")
    # ap.add_argument("--output_prefix", default="edited", help="Prefix for output files (default 'edited').")
    # ap.add_argument("--max_gap", type=int, default=10, help="Max gap (frames) to interpolate when merging (default 10).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.img_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/imagesSB/img1'
    args.fps = 15
    args.output_prefix = f'corrected_mot_{args.tracking_txt.split(".")[0]}'
    args.max_gap = 20
    if not os.path.isfile(args.tracking_txt):
        sys.exit(f"Tracking file {args.tracking_txt} not found.")
    tracks = load_tracks(args.tracking_txt)
    interactive_session(tracks, args)


if __name__ == "__main__":
    main()

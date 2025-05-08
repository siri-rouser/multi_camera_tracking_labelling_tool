import cv2
import os
import numpy as np
import random

def linear_interpolate(bbox1, bbox2, alpha):
    """
    Linearly interpolate between bbox1 and bbox2.
    bbox1 and bbox2 are lists: [x1, y1, x2, y2]
    alpha is between 0 and 1.
    """
    return [bbox1[i] + alpha * (bbox2[i] - bbox1[i]) for i in range(4)]

def main():
    # Base directories (adjust these paths as needed)
    base_tracking_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    base_img_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/dataset/detection'
    seqs = ['imagesc002', 'imagesc003', 'imagesc004']
    fps = 30  # Video frame rate
    
    for seq in seqs:
        print(f"Processing sequence {seq}...")
        # Path to the tracking result file, e.g. "imagesSB_mot.txt"
        tracking_file = os.path.join(base_tracking_dir, seq, f"{seq}_mot.txt")
        if not os.path.exists(tracking_file):
            print(f"Tracking file {tracking_file} not found. Skipping {seq}.")
            continue

        # Parse the tracking file.
        # Each row: frame_num track_id x1 y1 x2 y2 class
        tracks = {}  # track_id -> list of (frame, [x1, y1, x2, y2], class)
        with open(tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
                cls = int(float(parts[6]))
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append((frame, [x1, y1, x2, y2], cls))

        # Sort each track's detections by frame.
        print(f"Found {len(tracks)} tracks in {seq}.")
        for tid in tracks:
            tracks[tid] = sorted(tracks[tid], key=lambda x: x[0])

        # Interpolate missing frames for each track.
        # Also accumulate interpolated results for saving to file.
        interpolated_results = []  # each element: (frame, track_id, x1, y1, x2, y2, cls)
        for tid, det_list in tracks.items():
            full_list = []
            for i in range(len(det_list) - 1):
                frame_i, bbox_i, cls_i = det_list[i]
                frame_j, bbox_j, cls_j = det_list[i + 1]
                full_list.append((frame_i, bbox_i, cls_i))
                gap = frame_j - frame_i
                if gap > 1 and gap < 5:
                    # For each missing frame, perform linear interpolation.
                    for f in range(frame_i + 1, frame_j):
                        alpha = (f - frame_i) / (frame_j - frame_i)
                        interp_bbox = linear_interpolate(bbox_i, bbox_j, alpha)
                        # print(f"Interpolated frame {f} for track {tid}: {interp_bbox}")
                        full_list.append((f, interp_bbox, cls_i))
            full_list.append(det_list[-1])
            # Ensure the detections for the track are sorted by frame.
            full_list = sorted(full_list, key=lambda x: x[0])
            tracks[tid] = full_list
            # Add to the overall results list.
            for (frm, bbox, cls) in full_list:
                interpolated_results.append((frm, tid, bbox[0], bbox[1], bbox[2], bbox[3], cls))
        
        # inspect fragile tracklets
        # Remove tracks with less than 5 detections
        tracks = {tid: det_list for tid, det_list in tracks.items() if len(det_list) >= 5}
        print(f"Removed tracks with less than 5 detections. Remaining tracks: {len(tracks)}")


        # Save the interpolated results to a text file.
        interpolated_results.sort(key=lambda x: (x[0], x[1]))  # sort by frame then track id
        interpolated_file = os.path.join(base_tracking_dir, seq, f"{seq}_mot_interpolated.txt")
        with open(interpolated_file, 'w') as f:
            for res in interpolated_results:
                # Format: frame track_id x1 y1 x2 y2 class
                f.write(f"{res[0]} {res[1]} {res[2]:.2f} {res[3]:.2f} {res[4]:.2f} {res[5]:.2f} {res[6]}\n")
        print(f"Interpolated tracking results saved to {interpolated_file}")


if __name__ == "__main__":
    main()

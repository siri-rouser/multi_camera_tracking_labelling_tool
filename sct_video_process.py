import cv2
import os
import sys
import numpy as np
import random
from collections import defaultdict

def main(seq):
    # Base directories (adjust paths as necessary)
    base_detect_merge = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    base_detection = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection'
    
    # Interpolated tracking result file (assumed generated previously)
    interp_file = os.path.join(base_detect_merge, seq, f"{seq}_mot_interpolated.txt")
    interp_file = 'corrected_mot_imagesc003_mot_interpolated_final.txt'
    # Directory containing original images (full resolution)
    img_dir = os.path.join(base_detection, seq, 'img1')
    # Output video directory and file
    video_out_dir = os.path.join(base_detect_merge, seq, 'tracking_video')
    os.makedirs(video_out_dir, exist_ok=True)
    video_out_path = os.path.join(video_out_dir, f"{seq}_tracking_interpolated.mp4")
    
    # Build a dictionary mapping frame numbers to detections and record track bounds.
    # Each detection is (track_id, (x1, y1, x2, y2), class)
    frames_dict = defaultdict(list)
    track_bounds = {}  # track_id -> [min_frame, max_frame]
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]
    with open(interp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            cls = int(parts[6])
            frames_dict[frame].append((track_id, (x1, y1, x2, y2), cls))
            if track_id not in track_bounds:
                track_bounds[track_id] = [frame, frame]
            else:
                if frame < track_bounds[track_id][0]:
                    track_bounds[track_id][0] = frame
                if frame > track_bounds[track_id][1]:
                    track_bounds[track_id][1] = frame

    # Determine overall frame range
    all_frames = sorted(frames_dict.keys())
    if not all_frames:
        print("No frames found in the interpolated file.")
        return
    start_frame = all_frames[0]
    end_frame = all_frames[-1]
    
    # Read first image to get frame dimensions.
    first_img_path = os.path.join(img_dir, f"img{start_frame:06d}.jpg")


    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Failed to read first image: {first_img_path}")
        return
    height, width, _ = first_img.shape

    # Set up video writer with 15fps and original image size.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, 15, (width, height))
    
    # Colors for visualization (BGR format)
    color_start = (255, 0, 0)   # Blue: start of a tracklet
    color_end   = (0, 0, 255)   # Red: end of a tracklet
    
    # Process each frame in the overall range.
    for frame_num in range(start_frame, end_frame + 1):
        # Try to load image using 6-digit zero-padded filenames.
        frame_filename = f"img{(frame_num-1):06d}.jpg"
        frame_path = os.path.join(img_dir, frame_filename)
        img = cv2.imread(frame_path)
        
        # Draw detections for the current frame.
        detections = frames_dict.get(frame_num, [])
        for (track_id, bbox, cls) in detections:
            x1, y1, x2, y2 = map(int, bbox)
            start_f, end_f = track_bounds[track_id]
            if frame_num == start_f:
                color = color_start
                label = f"{track_id} start"
            elif frame_num == end_f:
                color = color_end
                label = f"{track_id} end"
            else:
                color = colors[track_id % len(colors)]
                label = f"{track_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        out.write(img)
        print(f"Processing frame {frame_num}", end="\r")
    
    out.release()
    print(f"\nSaved interpolated tracking video for {seq} at {video_out_path}")

if __name__ == "__main__":
    seqs = ['imagesc003']
    for seq in seqs:
        print(f"Processing sequence {seq}...")
        main(seq)

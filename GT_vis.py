import cv2
import os
import numpy as np
import random
from collections import defaultdict

def load_ground_truth(gt_file):
    """
    Loads the ground truth file.
    Each line is expected to have:
      camera_id, obj_id, frame_id, xmin, ymin, width, height, xworld, yworld
    Returns a dictionary: {camera_id: {frame_id: [detection, ...]}}
    where each detection is a tuple: (obj_id, xmin, ymin, width, height)
    """
    gt_data = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(gt_file):
        print(f"Ground truth file {gt_file} not found.")
        return gt_data
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cam = 'imagesNB' if parts[0] == str(1) else 'imagesSB'
            try:
                obj_id = int(parts[1])
                frame_id = int(parts[2])
                xmin = float(parts[3])
                ymin = float(parts[4])
                width = float(parts[5])
                height = float(parts[6])
            except ValueError:
                continue
            gt_data[cam][frame_id].append((obj_id, xmin, ymin, width, height))
    return gt_data

def get_color_for_id(obj_id, color_dict):
    """
    Returns a BGR color for a given object id.
    If not present, assign a random color and store it.
    """
    if obj_id not in color_dict:
        color_dict[obj_id] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    return color_dict[obj_id]

def main():
    # Base paths
    base_detection = '/dataset/detection'
    # Assume images for each camera are stored in: <base_detection>/<cam>/img1/
    cam_list = ['imagesNB', 'imagesSB']
    img_dirs = {cam: os.path.join(base_detection, cam, 'img1') for cam in cam_list}
    
    # Path to the ground truth file (adjust if needed)
    gt_file = 'multi_camera_ground_truth.txt'
    gt_data = load_ground_truth(gt_file)
    
    # Determine overall frame range for each camera (we assume frames are numbered as in the GT file)
    frame_range = {}
    for cam in cam_list:
        frames = list(gt_data[cam].keys())
        if frames:
            frame_range[cam] = (min(frames), max(frames))
        else:
            frame_range[cam] = (1, 1)
    
    # To combine both cameras side-by-side, we take the global range:
    global_min = 0
    global_max = 4499

    
    # Get sample image size from one camera (assume both have same resolution)
    sample_cam = cam_list[0]
    sample_img_path = os.path.join(img_dirs[sample_cam], f"img{global_min:06d}.jpg")
    sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        # Try png if jpg not found
        sample_img_path = os.path.join(img_dirs[sample_cam], f"img{global_min:06d}.png")
        sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        print(f"Failed to load sample image from {img_dirs[sample_cam]}")
        return
    h, w, _ = sample_img.shape
    combined_size = (2 * w, h)
    
    # Output video file
    output_video = os.path.join('/dataset/detect_merge', 'final_ground_truth.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    out = cv2.VideoWriter(output_video, fourcc, fps, combined_size)
    
    # Color dictionary for object IDs
    color_dict = {}
    
    for frame in range(global_min, global_max + 1):
        imgs = {}
        for cam in cam_list:
            # Build file path; assuming image file names are in the format: imgXXXXXX.jpg
            img_path = os.path.join(img_dirs[cam], f"img{frame:06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dirs[cam], f"img{frame:06d}.png")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image from {img_path}")
                return
            # Overlay ground truth boxes if available for this frame.
            if frame in gt_data[cam]:
                for (obj_id, xmin, ymin, width_box, height_box) in gt_data[cam][frame+1]:
                    color = get_color_for_id(obj_id, color_dict)
                    x2 = xmin + width_box
                    y2 = ymin + height_box
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(x2), int(y2)), color, 3)
                    cv2.putText(img, f"ID {obj_id}", (int(xmin), max(int(ymin)-10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            imgs[cam] = img
        
        # Concatenate the two camera views horizontally
        combined_frame = np.hstack((imgs['imagesNB'], imgs['imagesSB']))
        # Optionally, overlay the current frame number on the combined image.
        cv2.putText(combined_frame, f"Frame {frame}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        out.write(combined_frame)
        print(f"Processing frame {frame}", end="\r")
    
    out.release()
    print(f"\nSaved final ground truth video at {output_video}")

if __name__ == "__main__":
    main()

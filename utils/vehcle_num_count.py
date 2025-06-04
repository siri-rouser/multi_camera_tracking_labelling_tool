import os
import sys
from pathlib import Path

def read_vehicle_num(cam_id):
    path = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam_id}/{cam_id}_mot_interpolated_final.txt'
    track_dict = {}
    bbox_num = 0
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts =  line.strip().split()
            bbox_num += 1
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            cls = int(parts[6])
            if track_id not in track_dict:
                track_dict[track_id] = []
            track_dict[track_id].append((frame, [x1, y1, x2, y2], cls))
    print(f"Camera {cam_id} has {len(track_dict)} unique vehicles tracked.")
    print(f"Total bounding boxes in {cam_id}: {bbox_num}")
    return bbox_num, track_dict

if __name__ == "__main__":
    total_bbox_num = 0

    cam_list = ['imagesc001', 'imagesc002', 'imagesc003', 'imagesc004']

    for cam in cam_list:
        bbox_num,_ = read_vehicle_num(cam)
        if bbox_num is not None:
            total_bbox_num += bbox_num
    print(f"Total bounding boxes across all cameras: {total_bbox_num}")
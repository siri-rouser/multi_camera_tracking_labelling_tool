import os
import pandas as pd
import sys
from pathlib import Path

def get_mot_dict(cam_id):
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


def read_excel(cam_id):
    xls_id_dict = []
    file_path = f'../temp_res/Vehicle_statistics_mk3.xlsx'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    df = pd.read_excel(file_path, sheet_name=cam_id)

    for idx, track_id in enumerate(df['Number']):
        if track_id not in xls_id_dict:
            xls_id_dict.append(track_id)

    return xls_id_dict

def get_missing_id(track_dict, xls_id_dict,cam):
    print(f"Checking missing IDs for camera {cam}...")
    missing_ids = []
    for track_id in track_dict.keys():
        if track_id not in xls_id_dict:
            missing_ids.append(track_id)
    if missing_ids:
        print(f"Missing IDs in {cam}: {len(missing_ids)}")
        print(f"Missing IDs: {missing_ids}")



if __name__ == "__main__":
    '''
    Main function to compare the xlsx and SCT ground turth to get missing vehicle IDs from SCT ground truth files.
    '''
    total_bbox_num = 0

    cam_list = ['imagesc001', 'imagesc002', 'imagesc003', 'imagesc004']

    for cam in cam_list:
        bbox_num, track_dict = get_mot_dict(cam)
        xls_id_dict = read_excel(cam)
        get_missing_id(track_dict, xls_id_dict,cam)

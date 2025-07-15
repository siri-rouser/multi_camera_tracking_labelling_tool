import os
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import json

def gen_sct_track_dict(reid_dict):
    base_dir = Path('/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/')
    cam_list = ['imagesc001', 'imagesc002', 'imagesc003', 'imagesc004']

    sct_track_dict = {}
    for cam in cam_list:
        cam_dir = os.path.join(base_dir, cam, f'{cam}_mot_interpolated_final.txt')
        multi_track_id = []
        for track_id, data in reid_dict[cam[-1]].items():
            multi_track_id.append(str(data['ori_id']))
        sct_track_dict[cam] = {}
        if not os.path.exists(cam_dir):
            print(f"File {cam_dir} does not exist.")
            continue
        with open(cam_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    print(f"Skipping line due to insufficient parts: {line.strip()}")
                    continue
                frame, track_id, x1, y1, x2, y2 ,cls = parts[:7]
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                if str(track_id) in multi_track_id:
                    print(f"Track ID {track_id} in camera {cam} is in the multi-track list, skipping.")
                    continue
                if track_id not in sct_track_dict[cam]:
                    sct_track_dict[cam][track_id] = []
                sct_track_dict[cam][track_id].append((int(cam[-1]), int(frame), bbox))
    return sct_track_dict


def read_txt(path):
    """
    Reads a text file and returns a dictionary where each line is split by spaces.
    The first element of each line is used as the key, and the rest are stored in a list.

    :param file_path: Path to the text file.
    :return: Dictionary with keys as the first element of each line and values as lists of remaining elements.
    """
    track_dict = {}
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cam_id, track_id, frame, x1, y1, w, h ,_ ,_ = parts[:9]

            if track_id not in track_dict:
                track_dict[track_id] = []
            track_dict[track_id].append((int(cam_id), int(frame), [float(x1), float(y1), float(x1) + float(w), float(y1) + float(h)]))
 
    return track_dict

def filter_data(track_dict,occlusion_area):
    # NOTE: check label.txt, functionality pendind to add 
    txt_base_dir = Path('/data/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/')
    test_image_dict = {}
    query_image_dict = {}
    train_image_dict = {}

    for track_id, data in track_dict.items():
        if len(data) < 20:
            print(f"Track ID {track_id} has less than 20 frames in total, skipping.")
            continue
        test_image_dict[track_id] = []
        query_image_dict[track_id] = []
        train_image_dict[track_id] = []
        high_quality_image = None
        high_quality_image_hold = None
        flag_t = -1
        flag_q = -1
        flag_train = -1
        track_id_data = {}
        for cam_id, frame, bbox in data:
            if cam_id not in track_id_data:
                track_id_data[cam_id] = {'min_frame': frame, 'max_frame': frame}
            else:
                track_id_data[cam_id]['min_frame'] = min(track_id_data[cam_id]['min_frame'], frame)
                track_id_data[cam_id]['max_frame'] = max(track_id_data[cam_id]['max_frame'], frame)

        for idx,(cam_id, frame, bbox) in enumerate(data):
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area <= 3500:
                continue

            # Check if there is any overlap between bbox and occlusion area
            txt_dir = os.path.join(txt_base_dir, f'imagesc00{cam_id}', 'labels_filtered', f'img{frame-1:06d}.txt')
            detect_gt_dict = []
            with open(txt_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    clss_id, gt_x1, gt_y1, gt_x2, gt_y2, conf = parts[:7]
                    detect_gt_dict.append([float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)])

            def _iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                return iou

            has_high_iou = False
            for gt_bbox in detect_gt_dict:
                if _iou([x1, y1, x2, y2], gt_bbox) > 0.9:
                    has_high_iou = True
                    break
            if not has_high_iou:
                print(f"Track ID {track_id} in camera {cam_id} does not have high IoU with any ground truth, skipping.")
                continue

            occl_key = f'imagesc00{cam_id}'
            if occl_key in occlusion_area:
                occl_x1, occl_y1, occl_x2, occl_y2 = occlusion_area[occl_key]
                # Calculate overlap
                overlap_x1 = max(x1, occl_x1)
                overlap_y1 = max(y1, occl_y1)
                overlap_x2 = min(x2, occl_x2)
                overlap_y2 = min(y2, occl_y2)
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    #print(f"Track ID {track_id} in camera {cam_id} overlaps with occlusion area, skipping.")
                    continue  # There is overlap
            # Skip if this is one of the last 5 frames in the track
            if (track_id_data[cam_id]['max_frame'] - frame < 15) or (frame - track_id_data[cam_id]['min_frame'] < 15):
                #print(f"Track ID {track_id} in camera {cam_id} is in the last 5 frames, skipping.")
                continue
            if area >= 10000 and high_quality_image_hold is None:
                high_quality_image_hold = (cam_id, frame, bbox)
            else:
                if area >= 10000 and high_quality_image is None:
                    high_quality_image = (cam_id, frame, bbox)
            
            # if not query_image_dict[track_id] or cam_id not in [item[0] for item in query_image_dict[track_id]]:
            #     query_image_dict[track_id].append((cam_id, frame, bbox))
            
            flag_q += 1
            flag_t += 1
            flag_train += 1

            if idx == len(data) - 1 and high_quality_image_hold is not None:
                query_image_dict[track_id].append(high_quality_image_hold)
                high_quality_image_hold = None
                continue

            if flag_q % 45 == 0:
                if area < 5000:
                    flag_q = -1
                else:
                    query_image_dict[track_id].append((cam_id, frame, bbox))
                    continue
            
            if flag_t % 5 == 0:
                test_image_dict[track_id].append((cam_id, frame, bbox))
                continue

            if flag_train % 3 == 0:
                train_image_dict[track_id].append((cam_id, frame, bbox))
                continue

    return test_image_dict,query_image_dict,train_image_dict

def train_dict_enrich(train_image_dict, sct_track_dict,occlusion_area):
    """
    Enrich the train_image_dict with additional data from sct_track_dict.
    
    :param train_image_dict: Dictionary containing training images.
    :param sct_track_dict: Dictionary containing additional track data.
    :return: Enriched train_image_dict.
    """
    max_track_id = max(int(track_id) for track_id in train_image_dict.keys()) if train_image_dict else 1000
    txt_base_dir = Path('/data/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/')
    for cam in sct_track_dict:
        dict= sct_track_dict[cam]
        for track_id, data in dict.items():
            if len(data) < 40:
                print(f"Track ID {track_id} has less than 50 frames in total, skipping.")
                continue
            flag = -1
            high_quality_image = None
            track_id_data = {}
            for cam_id, frame, bbox in data:
                if cam_id not in track_id_data:
                    track_id_data[cam_id] = {'min_frame': frame, 'max_frame': frame}
                else:
                    track_id_data[cam_id]['min_frame'] = min(track_id_data[cam_id]['min_frame'], frame)
                    track_id_data[cam_id]['max_frame'] = max(track_id_data[cam_id]['max_frame'], frame)

            for idx, (cam_id, frame, bbox) in enumerate(data):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area <= 3500:
                    continue
                txt_dir = os.path.join(txt_base_dir, f'imagesc00{cam_id}', 'labels_filtered', f'img{frame-1:06d}.txt')
                detect_gt_dict = []
                with open(txt_dir, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        clss_id, gt_x1, gt_y1, gt_x2, gt_y2, conf = parts[:6]
                        detect_gt_dict.append([float(gt_x1), float(gt_y1), float(gt_x2), float(gt_y2)])

                def _iou(boxA, boxB):
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                    return iou
                
                has_high_iou = False
                for gt_bbox in detect_gt_dict:
                    if _iou([x1, y1, x2, y2], gt_bbox) > 0.9:
                        has_high_iou = True
                        break
                if not has_high_iou:
                    print(f"Track ID {track_id} in camera {cam_id} does not have high IoU with any ground truth, skipping.")
                    continue

                occl_key = f'imagesc00{cam_id}'
                if occl_key in occlusion_area:
                    occl_x1, occl_y1, occl_x2, occl_y2 = occlusion_area[occl_key]
                    # Calculate overlap
                    overlap_x1 = max(x1, occl_x1)
                    overlap_y1 = max(y1, occl_y1)
                    overlap_x2 = min(x2, occl_x2)
                    overlap_y2 = min(y2, occl_y2)
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        #print(f"Track ID {track_id} in camera {cam_id} overlaps with occlusion area, skipping.")
                        continue  # There is overlap
                # Skip if this is one of the last 5 frames in the track
                if (track_id_data[cam_id]['max_frame'] - frame < 15) or (frame - track_id_data[cam_id]['min_frame'] < 15):
                    #print(f"Track ID {track_id} in camera {cam_id} is in the last 5 frames, skipping.")
                    continue

                if area >= 10000 and high_quality_image is None:
                    high_quality_image = (cam_id, frame, bbox)
                
                flag+= 1
                if flag == 0:
                    new_track_id = max_track_id + 1
                    max_track_id += 1
                    train_image_dict[max_track_id] = []
                    if high_quality_image is not None:
                        train_image_dict[max_track_id].append(high_quality_image)
                        high_quality_image = None
                    train_image_dict[max_track_id].append((cam_id, frame, bbox))

                elif flag % 5 == 0:
                    if high_quality_image is not None:
                        train_image_dict[max_track_id].append(high_quality_image)
                        high_quality_image = None
                    train_image_dict[max_track_id].append((cam_id, frame, bbox))

    return train_image_dict


def save_data(test_image_dict, query_image_dict, train_image_dict, output_dir):

    base_dir = Path('/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/')
    txt_out_path = os.path.join(output_dir, "RoundaboutHD_Reid_GT.txt")
    train_txt_out_path = os.path.join(output_dir, "RoundaboutHD_Reid_Train.txt")
    img_output_dir = os.path.join(output_dir, "RoundaboutHD_Reid")
    test_image_dir = os.path.join(img_output_dir, "test_images")
    query_image_dir = os.path.join(img_output_dir, "query_images")
    train_image_dir = os.path.join(img_output_dir, "train_images")
    
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(query_image_dir, exist_ok=True)
    os.makedirs(train_image_dir, exist_ok=True)
    print(f"Output directory created at {img_output_dir}")
    print(f"Test images will be saved to {test_image_dir}")
    print(f"Query images will be saved to {query_image_dir}")
    print(f"Train images will be saved to {train_image_dir}")

    print(f'Total tracks in query_image_dict: {len(query_image_dict)}')
    print(f'Total tracks in train_image_dict: {len(train_image_dict)}')
    print(f'Total tracks in test_image_dict: {len(test_image_dict)}')

    print(f"Start data processing. Total tracks in test_image_dict: {len(test_image_dict)}")
    total_images = sum(len(images) for images in test_image_dict.values())
    print(f"Total images in test_image_dict: {total_images}")

    total_query_images = sum(len(images) for images in query_image_dict.values())
    print(f"Total images in query_image_dict: {total_query_images}")

    total_train_images = sum(len(images) for images in train_image_dict.values())
    print(f"Total images in train_image_dict: {total_train_images}")

    for track_id, data in tqdm(test_image_dict.items(), desc="Processing tracks"):
        for cam_id, frame, bbox in data:
            x1, y1, x2, y2 = bbox
            frame = frame -1 
            img_path = os.path.join(base_dir, f'imagesc00{cam_id}', 'img1', f'img{frame:06d}.jpg')
            frame_image = cv2.imread(img_path)
            if frame_image is None:
                print(f"Image not found: {img_path}")
                continue
            crop = frame_image[int(y1):int(y2), int(x1):int(x2)]
            image_filename = f"{int(track_id):04d}_cam{cam_id:02d}_{frame:06d}.jpg"
            save_path = os.path.join(test_image_dir, image_filename)
            cv2.imwrite(save_path, crop)
    print(f"Test images saved to {test_image_dir}")

    print(f"Start data processing. Total tracks in query_image_dict: {len(query_image_dict)}")
    total_images = sum(len(images) for images in query_image_dict.values())
    print(f"Total images in query_image_dict: {total_images}")

    for track_id, data in tqdm(query_image_dict.items(), desc="Processing query images"):
        for cam_id, frame, bbox in data:
            x1, y1, x2, y2 = bbox
            frame = frame - 1
            img_path = os.path.join(base_dir, f'imagesc00{cam_id}', 'img1', f'img{frame:06d}.jpg')
            frame_image = cv2.imread(img_path)
            if frame_image is None:
                print(f"Image not found: {img_path}")
                continue
            crop = frame_image[int(y1):int(y2), int(x1):int(x2)]
            image_filename = f"{int(track_id):04d}_cam{cam_id:02d}_{frame:06d}.jpg"
            save_path = os.path.join(query_image_dir, image_filename)
            cv2.imwrite(save_path, crop)
            with open(txt_out_path, 'a') as f:
                line = f'{image_filename.split(".")[0]} {track_id} {cam_id}\n'
                f.write(line)
    
    print(f"Start data processing. Total tracks in train_image_dict: {len(train_image_dict)}")
    for track_id, data in tqdm(train_image_dict.items(), desc="Processing train images"):
        for cam_id, frame, bbox in data:
            x1, y1, x2, y2 = bbox
            frame = frame - 1
            img_path = os.path.join(base_dir, f'imagesc00{cam_id}', 'img1', f'img{frame:06d}.jpg')
            frame_image = cv2.imread(img_path)
            if frame_image is None:
                print(f"Image not found: {img_path}")
                continue
            crop = frame_image[int(y1):int(y2), int(x1):int(x2)]
            image_filename = f"{int(track_id):04d}_cam{cam_id:02d}_{frame:06d}.jpg"
            save_path = os.path.join(train_image_dir, image_filename)
            cv2.imwrite(save_path, crop)
            with open(train_txt_out_path, 'a') as f:
                line = f'{image_filename.split(".")[0]} {track_id} {cam_id}\n'
                f.write(line)

if __name__ == "__main__":
    '''
    Main function to create image-based datasets for multi-camera vehicle tracking.
    '''
    occlusion_area = {}
    occlusion_area['imagesc001'] = (1542,86,2012,350)
    occlusion_area['imagesc002'] = (1704,398,2102,604)
    occlusion_area['imagesc003'] = (1746,236,1948,442)
    occlusion_area['imagesc004'] = (1722,510,2040,652)

    reid_path = './utils/final_redict.json'
    with open(reid_path, 'r') as f:
        reid_dict = json.load(f)
    ground_turth_dir = Path("Multi_CAM_Ground_Turth.txt")
    output_dir = Path('/home/yuqiang/yl4300/project/MCVT_YQ/datasets/AIC22_Track1_MTMC_Tracking/')

    track_dict = read_txt(ground_turth_dir)
    if track_dict is None:
        print("Ground truth file missing or empty. Exiting.")
        sys.exit(1)
    test_image_dict,query_image_dict,train_image_dict = filter_data(track_dict,occlusion_area)
    sct_track_dict = gen_sct_track_dict(reid_dict)
    for cam in sct_track_dict:
        dict = sct_track_dict[cam]
        print(f"Processing camera {cam} with {len(dict)} tracks.")
    train_image_dict = train_dict_enrich(train_image_dict, sct_track_dict,occlusion_area)
    save_data(test_image_dict, query_image_dict,train_image_dict, output_dir)
import os
import cv2
import sys

x_low_threshold = 0
x_high_threshold = 3840
y_low_threshold = 250
y_high_threshold = 2160

area = [2580,0, 3840, 360]  # x1, y1, x2, y2

cam_id = 'imagesc003'

def is_point_in_bbox(point, bbox):
    x, y = point
    _, x1, y1, x2, y2, _ = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

ori_label_dir = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam_id}/labels_corrected'
label_files =sorted(os.listdir(ori_label_dir))
processed_label_dir = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam_id}/labels_wiped'

points = []

points.append([(330,406), 1, 2]) 
points.append([(319,281), 0.6, 2]) 
points.append([(154,325), 0.8, 2]) 
points.append([(30,370), 0.6, 2])
points.append([(1384, 130), 0.8, 7])  # Example point, adjust as needed
points.append([(2762, 185), 0.5, 2])  # Example point, adjust as needed
points.append([(1576, 108), 0.6, 2])  # Example point, adjust as needed
os.makedirs(processed_label_dir, exist_ok=True)


for label_file in label_files:
    txt_path = os.path.join(ori_label_dir, label_file)
    new_bboxes = []
    with open(txt_path, 'r') as f:
        bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
    for bbox in bboxes:
        cx, cy = (bbox[1] + bbox[3]) / 2, (bbox[2] + bbox[4]) / 2
        if cx < x_low_threshold or cx > x_high_threshold or cy < y_low_threshold or cy > y_high_threshold:
            print(f'Removed bbox out of bounds: {bbox}')
            continue
        
        if  bbox[1] > area[0] and bbox[2] > area[1] and bbox[3] < area[2] and bbox[4] < area[3]:
            print(f'Removed bbox: {bbox}')
            continue

        remove_flag = False
        for point in points:
            # print(f'Checking point {point[0]} with confidence {point[1]} and class {point[2]} against bbox {bbox}')
            if is_point_in_bbox(point[0], bbox) and bbox[5] < point[1] and int(bbox[0]) == int(point[2]):
                print(f'Removed bbox: {bbox}')
                remove_flag = True
                break

        if not remove_flag:
            new_bboxes.append(bbox)

    write_path = os.path.join(processed_label_dir, label_file)
    with open(write_path, 'w') as f:  
        for bbox in new_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')

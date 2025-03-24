import os
import cv2

def is_point_in_bbox(point, bbox):
    x, y = point
    _, x1, y1, x2, y2, _ = bbox
    return x1 <= x <= x2 and y1 <= y <= y2



label_dir = f'/dataset/imagesNB/labels_filtered/'
label_files =sorted(os.listdir(label_dir))

points = []

# points.append([(742,318), 0.45])  #0.4
# points.append([(2092,302), 0.65]) #0.65
# points.append([(16,386), 0.55]) #0.55
# points.append([(2196,312), 0.75]) #0.75
# points.append([(2439,339), 0.6]) #0.6
# points.append([(1434,190), 0.5]) #0.5
# points.append([(2343,321), 0.67]) #0.6

points.append([(2552,618), 1])  #0.4 for NB

for point in points:
    for label_file in label_files:
        txt_path = os.path.join(label_dir, label_file)
        with open(txt_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
        for idx, bbox in enumerate(bboxes):
            if is_point_in_bbox(point[0],bbox) and bbox[5] < point[1]:
                bboxes.pop(idx)
                print(f'Removed bbox: {bbox}')
                break
        with open(txt_path, 'w') as f:  
            for bbox in bboxes:
                f.write(' '.join(map(str, bbox)) + '\n')
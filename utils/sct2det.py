import os 


def read_mot(mot_file):
    det_data = {}
    with open(mot_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])
        cls = int(parts[6])  # Assuming class ID is the second part
        x1, y1, x2, y2 = map(float, parts[2:6])
        conf = 1

        if frame_id not in det_data:
            det_data[frame_id] = []
        det_data[frame_id].append((cls,x1, y1, x2, y2, conf))
    return det_data

if __name__ == "__main__":
    cam_list = ['imagesc001', 'imagesc002', 'imagesc003', 'imagesc004']

    for cam in cam_list:
        print(f'Processing camera: {cam}')
        mot_file = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/{cam}_mot_interpolated_final.txt'
        output_label_dir = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/labels_sct2det'
        os.makedirs(output_label_dir, exist_ok=True)

        det_data = read_mot(mot_file)

        for frame_id, bboxes in det_data.items():
            frame_id = int(frame_id)-1
            output_path = os.path.join(output_label_dir, f'img{frame_id:06d}.txt')
            with open(output_path, 'w') as f:
                for bbox in bboxes:
                    cls, x1, y1, x2, y2, conf = bbox
                    # Assuming class ID is 0 for all detections
                    f.write(f'{cls} {x1} {y1} {x2} {y2} {conf}\n')
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    # print(f'Frame {frame_id:06d}: Detected bbox: ({x1}, {y1}, {x2}, {y2}) with confidence {conf}')
#                     print(f'Frame {frame_id:06d}: Detected bbox: ({x1}, {y1}, {x2}, {y2}) with confidence {conf}')
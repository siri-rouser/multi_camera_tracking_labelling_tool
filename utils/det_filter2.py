import cv2
import os
import sys

# Base directory for detection results
base_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'

seqs = ['imagesc004']

# Processing parameters
conf_threshold = 0.35
area_threshold = 900
fps = 15

for seq in seqs:
    # Setup directories for images, labels, output filtered labels and video
    img_dir = os.path.join('/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection', seq, 'img1')
    label_dir = os.path.join(base_dir, seq, 'labels_wiped')
    output_label_dir = os.path.join(base_dir, seq, 'labels_filtered2')
    output_img_dir = os.path.join(base_dir, seq, 'images_filtered2')

    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Get list of image files (assuming .jpg or .png, adjust as needed)
    img_files = sorted(os.listdir(img_dir))
    if not img_files:
        print(f"No images found in {img_dir} for sequence {seq}.")
        continue

    # Use first image to get dimensions for video writer
    first_img_path = os.path.join(img_dir, img_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Could not read {first_img_path}.")
        continue
    height, width, _ = first_img.shape

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        det_name = img_file[:-4]

        label_file = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        output_label_file = os.path.join(output_label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}. Skipping.")
            continue
        
        filtered_bboxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # Skip invalid lines
                cls_id = str(parts[0])
                x1 = int(float(parts[1]))
                y1 = int(float(parts[2]))
                x2 = int(float(parts[3]))
                y2 = int(float(parts[4]))
                conf = float(parts[5])
                area = (x2 - x1) * (y2 - y1)
                # Filter condition: if both low confidence and small area, skip this bbox
                if conf < conf_threshold or area < area_threshold:
                    print(f"Filtered bbox: {cls_id} {x1} {y1} {x2} {y2} {conf:.2f}")
                    continue
                else:
                    filtered_bboxes.append((cls_id, x1, y1, x2, y2, conf))
            
            # Save filtered bounding boxes to new label file
            with open(output_label_file, 'w') as f:
                for bbox in filtered_bboxes:
                    cls_id, x1, y1, x2, y2, conf = bbox
                    f.write(f"{cls_id} {x1} {y1} {x2} {y2} {conf:.2f}\n")
        else:
            print(f"Label file {label_file} not found. Skipping filtering for this image.")
        
        # Draw filtered bounding boxes on the image for visualization in the video
        for bbox in filtered_bboxes:
            cls_id, x1, y1, x2, y2, conf = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'Class {cls_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imwrite(os.path.join(output_img_dir,img_file), img)

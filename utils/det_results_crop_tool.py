import cv2
import os
import pickle
import argparse

# Directories for original images and filtered labels
# Original images from the detector
base_img_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection'
# Labels from the post process step (filtered labels)
base_label_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Crop detections from images based on filtered labels.")
parser.add_argument('--seqs', nargs='+', required=True, help="List of sequences to process.")
args = parser.parse_args()

seqs = args.seqs

for seq in seqs:
    out_dict = {}
    # Original images directory for this sequence
    img_dir = os.path.join(base_img_dir, seq, 'img1')
    # Filtered label files directory
    label_filtered_dir = os.path.join(base_label_dir, seq, 'labels_sct2det')
    # Output directory for cropped detection images
    output_dir = os.path.join(base_label_dir, seq, 'dets_sct2det')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all label files in the labels_filtered directory
    label_files = sorted(os.listdir(label_filtered_dir))
    for label_file in label_files:
        base_name, ext = os.path.splitext(label_file)
        label_path = os.path.join(label_filtered_dir, label_file)
        
        # Try to find the corresponding image in the original image directory.
        # Adjust possible extensions if necessary.
        possible_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in possible_extensions:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if img_path is None:
            print(f"Image for {base_name} not found in {img_dir}. Skipping.")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}.")
            continue
        
        # Read filtered labels for this image
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        det_num = 1  # to enumerate the detections for naming
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # skip invalid lines
            # Parse the bbox (format: class_id, x1, y1, x2, y2, conf)
            cls_id = int(float(parts[0]))
            x1 = int(float(parts[1]))
            y1 = int(float(parts[2]))
            x2 = int(float(parts[3]))
            y2 = int(float(parts[4]))
            # Confidence is available as parts[5] if needed
            conf = float(parts[5])
            # Optionally, ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Crop the detection region from the image
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                print(f"Warning: empty crop for {base_name} bbox {det_num}. Skipping this bbox.")
                continue
            
            # Save the cropped detection image with a naming pattern like "imageName_001.png"
            det_img_name = f"{base_name}_{det_num:03d}.png"
            out_dict[det_img_name[:-4]] = {
                    'bbox': (x1, y1, x2, y2),
                    'frame': base_name,
                    'id': det_num,
                    'imgname': det_img_name,
                    'class': cls_id,  # placeholder; update if you have class info
                    'conf': conf    # placeholder; update if needed
                }
            det_img_path = os.path.join(output_dir, det_img_name)
            cv2.imwrite(det_img_path, crop_img)
            print(f"Saved cropped image: {det_img_path}")
            det_num += 1

    output_pkl_path = os.path.join(base_label_dir, seq, f'{seq}_sct2dets.pkl')
    with open(output_pkl_path, 'wb') as pkl_file:
        pickle.dump(out_dict, pkl_file)

print("Cropping completed.")

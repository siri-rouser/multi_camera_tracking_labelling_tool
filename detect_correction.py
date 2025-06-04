import cv2
import os
import argparse

def is_point_in_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.

    Args:
        point (tuple): A tuple (x, y) representing the point.
        bbox (list): A list [cls_id, x1, y1, x2, y2, conf] representing the bounding box.

    Returns:
        bool: True if the point is inside the bounding box, False otherwise.
    """
    _, x1, y1, x2, y2, _ = bbox
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2

def check_class_id(point,img_file):
    txt_path = os.path.join(label_dir_corrected, img_file[:-4]  + '.txt')
    print(txt_path)
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
    else:
        bboxes = []
    for idx, bbox in enumerate(bboxes):
        if is_point_in_bbox(point,bbox):
            print(f'Class id: {int(bbox[0])}')
            return int(bbox[0])
    return int(input('Enter class id: '))

# Paths setup
base_img_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection'
parser = argparse.ArgumentParser(description="Multi-camera tracking labeling tool")
parser.add_argument('--seqs', nargs='+', required=True, help="List of sequences to process. Example: --seqs imagesc001")
args = parser.parse_args()

seqs = args.seqs

display_scale = 0.9
display_scale_a = 0.9
delete_display_scale = 0.9

for seq in seqs:
    img_dir = os.path.join(base_img_dir, seq, 'img1')
    label_dir = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{seq}/labels_filtered2/'
    label_dir_corrected = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{seq}/labels_corrected_final/'
    img_dir_corrected = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{seq}/images_corrected_final/'
    img_files = sorted(os.listdir(img_dir))
    os.makedirs(label_dir_corrected, exist_ok=True)
    os.makedirs(img_dir_corrected, exist_ok=True)

    # Set the image file name to start from
    start_img_num = input(f"Enter the starting image filename for {seq} (leave empty to start from beginning): ")
    start_img = 'img' + start_img_num.zfill(6) + '.jpg'
    if start_img in img_files:
        start_idx = img_files.index(start_img)
    else:
        start_idx = 0

    img_files = img_files[start_idx:]

    for IDX,img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(label_dir, img_file[:-4]  + '.txt')
        txt_path_corrected = os.path.join(label_dir_corrected ,img_file[:-4] + '.txt')
        corrected_img_path = os.path.join(img_dir_corrected, img_file)

        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
            if len(bboxes) == 0:
                print(f"No bounding boxes found in {txt_path}")
                continue
        else:
            print(f"Label file {txt_path} does not exist.")
            continue
        
        window_name = f"Review {img_file.split('.')[0]} (Press a:add, d:delete, s:save, q:quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(2560*display_scale), int(1440*display_scale)) # Resize window to fit the screen

        while True:
            img_display = img.copy()
            for idx, (cls_id, x1, y1, x2, y2, conf) in enumerate(bboxes):
                cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_display, f'Class {int(cls_id)} idx {idx}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            cv2.imshow(window_name, img_display)
            key = cv2.waitKey(0)

            if key == ord('a'):
                # Resize the image for ROI selection
                roi_window_name = 'Draw BBox'
                img_scaled = cv2.resize(img, (0, 0), fx=display_scale_a, fy=display_scale_a)
                cv2.namedWindow(roi_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(roi_window_name, int(2560 * display_scale_a), int(1440 * display_scale_a))  # Resize window to fit the screen
                # Let the user select ROI on the scaled image
                roi = None
                key1 = 0 
                while True:
                    roi = cv2.selectROI(roi_window_name, img_scaled, False)
                    key1 = cv2.waitKey(0)
                    if key1 == 27:  # ESC to cancel
                        print("Cancelled bbox adding")
                        cv2.destroyWindow("Draw BBox")
                        break
                    if roi is not None:
                        cv2.destroyWindow(roi_window_name)  # Close the ROI window after selection

                        x, y, w, h = map(int, roi)
                        # Convert the coordinates back to the original image scale
                        x = int(x / display_scale_a)
                        y = int(y / display_scale_a)
                        w = int(w / display_scale_a)
                        h = int(h / display_scale_a)
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        central_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                        class_id = check_class_id(central_point,img_files[(IDX-1)])
                        bboxes.append([class_id, x1, y1, x2, y2, 1.0])
                        print("BBox added.")
                        break
                
                # cv2.destroyWindow(roi_window_name)  # Close the ROI window after selection

                # x, y, w, h = map(int, roi)
                # # Convert the coordinates back to the original image scale
                # x = int(x / display_scale_a)
                # y = int(y / display_scale_a)
                # w = int(w / display_scale_a)
                # h = int(h / display_scale_a)
                # x1, y1, x2, y2 = x, y, x + w, y + h
                # central_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                # class_id = check_class_id(central_point,img_files[(IDX-1)])
                # bboxes.append([class_id, x1, y1, x2, y2, 1.0])
                # print("BBox added.")

            elif key == ord('d'):
                print("Click on a bounding box to delete it (press ESC to cancel).")
                selected_idx = [-1]  # Mutable container to capture selected index
                key2 = 0

                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        # Convert click position back to original scale
                        x_full = int(x / delete_display_scale)
                        y_full = int(y / delete_display_scale)
                        for idx, bbox in enumerate(bboxes):
                            if is_point_in_bbox((x_full, y_full), bbox):
                                selected_idx[0] = idx
                                print(f"Clicked on bbox index: {idx}")

                                break

                # Resize image and draw boxes for deletion window
                temp_display = cv2.resize(img_display.copy(), (0, 0), fx=delete_display_scale, fy=delete_display_scale)
                for idx, (cls_id, x1, y1, x2, y2, _) in enumerate(bboxes):
                    x1_s, y1_s = int(x1 * delete_display_scale), int(y1 * delete_display_scale)
                    x2_s, y2_s = int(x2 * delete_display_scale), int(y2 * delete_display_scale)
                    cv2.rectangle(temp_display, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 255), 2)
                    # cv2.putText(temp_display, f"Class {int(cls_id)}", (x1_s, y1_s - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.namedWindow("Click to Delete", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Click to Delete", int(2560 * delete_display_scale), int(1440 * delete_display_scale))  # Resize window to fit the screen
                cv2.setMouseCallback("Click to Delete", mouse_callback)
                cv2.imshow("Click to Delete", temp_display)

                while True:
                    print(selected_idx[0])
                    if key2 == 27:  # ESC to cancel
                        print("Cancelled deletion.")
                        cv2.destroyWindow("Click to Delete")
                        break
                    if selected_idx[0] != -1:
                        removed_bbox = bboxes.pop(selected_idx[0])
                        print(f"Deleted bbox: {removed_bbox}")
                        break
                    key2 = cv2.waitKey(0)

            elif key == ord('s'):
            # Draw processed bounding boxes on the image
                for cls_id, x1, y1, x2, y2, conf in bboxes:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, f'Class {int(cls_id)}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # # Save the image with processed bounding boxes
                # cv2.imwrite(corrected_img_path, img)

                with open(txt_path_corrected, 'w') as f:
                    for bbox in bboxes:
                        f.write(f"{int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])} {int(bbox[4])} {bbox[5]:2f}\n")
                
                print(f'saved {img_file} to {txt_path_corrected}')  
                cv2.destroyAllWindows()
                break

            elif key == ord('m'):
                try:
                    id1 = int(input('Enter the first index of bbox to merge: '))
                    id2 = int(input('Enter the second index of bbox to merge: '))
                    
                    if id1 < 0 or id1 >= len(bboxes) or id2 < 0 or id2 >= len(bboxes) or id1 == id2:
                        print("Invalid indices. Please enter valid and distinct indices.")
                        continue
                    
                    bbox1 = bboxes[id1]
                    bbox2 = bboxes[id2]
                    
                    x1 = min(bbox1[1], bbox2[1])
                    y1 = min(bbox1[2], bbox2[2])
                    x2 = max(bbox1[3], bbox2[3])
                    y2 = max(bbox1[4], bbox2[4])
                    
                    # Use the class ID of the first bbox or prompt the user for a new class ID
                    class_id = int(input(f"Enter class ID for the merged bbox (default {int(bbox1[0])}): ") or bbox1[0])
                    conf = 1
                    
                    # Remove the bboxes in reverse order to avoid index shifting
                    for idx in sorted([id1, id2], reverse=True):
                        removed_bbox = bboxes.pop(idx)
                        print(f"Removed bbox: {removed_bbox}")
                    
                    bboxes.append([class_id, x1, y1, x2, y2, conf])
                    print(f"Merged bbox: {bboxes[-1]}")
                except ValueError:
                    print("Invalid input. Please enter numeric indices.")

            # Press 'q' during reviewing images to quit the loop
            elif key == ord('q'):
                print(f"Stopped at {img_file}.")
                exit(0)
                    
            else:
                print("Invalid key. Press 'a' to add, 'd' to delete, 's' for next image, or 'q' to quit.")

cv2.destroyAllWindows()

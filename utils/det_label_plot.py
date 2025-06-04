import os
import cv2

def res_plot(cam):
    abs_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/'
    source_path = os.path.join(abs_path, 'detection', cam, 'img1')
    abs_label_path = os.path.join(abs_path, 'detect_merge', cam, 'labels_corrected')
    plot_path = os.path.join(abs_path, 'detect_merge', cam, 'images_corrected')

    img_files = sorted(os.listdir(source_path))

    os.makedirs(plot_path, exist_ok=True)

    for IDX,img_file in enumerate(img_files):
        print(f"Processing {IDX+1}/{len(img_files)}: {img_file}", end='\r')
        img_path = os.path.join(source_path, img_file)
        label_path = os.path.join(abs_label_path, img_file.replace('.jpg', '.txt'))
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_path} could not be read.")
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            cls, x1, y1, x2, y2, conf = line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f'Class {int(cls)}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(plot_path, img_file), img)


if __name__ == "__main__":
    cam_list = ['imagesc001', 'imagesc002']
    for cam in cam_list:
        res_plot(cam)
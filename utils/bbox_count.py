import os

if __name__ == "__main__":
    bbox_count = 0
    base_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge'
    cam_list = ['imagesc001', 'imagesc002', 'imagesc003', 'imagesc004']

    for cam in cam_list:
        path = os.path.join(base_dir, cam, 'labels_filtered')    
        file_list = os.listdir(path)
        for file in file_list:
            open_file = os.path.join(path, file)
            with open(open_file, 'r') as f:
                lines = f.readlines()
                bbox_count += len(lines)


    print(f'Total bbox count: {bbox_count}')
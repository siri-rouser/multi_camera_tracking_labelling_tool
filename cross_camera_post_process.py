import os


if __name__ == "__main__":
    cam_list = ['imagesc001','imagesc002','imagec003', 'imagec004']
    track_dict = {}

    for index in range(len(cam_list)-1):
        with open(f'{cam_list[index]}_{cam_list[index+1]}_ground_truth.txt', 'a') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cam_id = int(parts[0])
                track_id = int(float(parts[1]))
                frame = int(float(parts[2]))
                x1 = float(parts[3])
                y1 = float(parts[4])
                w = float(parts[5])
                h = float(parts[6])
                
                pass

            # NOTE: find bug on reid_dict{}, need to rewrite the programe, use the above to load data.


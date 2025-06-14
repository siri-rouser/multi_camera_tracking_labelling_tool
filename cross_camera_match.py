import os
import sys


def load_data(cam):
    """
    Loads the interpolated final tracklet data for a given camera.
    Assumes each line in the file has:
      frame track_id x1 y1 x2 y2 class
    Returns a dictionary mapping each track_id (int) to a list of detections.
    Each detection is a tuple: (frame, [x1, y1, x2, y2], cls)
    """
    data = {}
    filepath = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam}/{cam}_mot_interpolated_final.txt'
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        data = {}
    else:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
                cls = int(float(parts[6]))
                if track_id not in data:
                    data[track_id] = []
                data[track_id].append((frame, [x1, y1, x2, y2], cls))
    return data


def format_detection_line(cam_num, info, id_index):
    """
    Formats a detection into a line with the following fields:
      camera_id, obj_id, frame_id, xmin, ymin, width, height, xworld, yworld
    Expects 'info' to be a tuple: (frame, [x1, y1, x2, y2], cls).
    Here, xworld and yworld are fixed to -1.
    """
    frame_num = int(info[0])
    bbox = info[1]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    xworld = -1
    yworld = -1
    line = f"{cam_num} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {xworld} {yworld}"
    return line

def get_current_id_index(filename):
    """
    Returns the next object id index by scanning the ground truth file.
    """
    max_id = -1
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    obj_id = int(parts[1])
                    if obj_id > max_id:
                        max_id = obj_id
                except ValueError:
                    continue
    else:
        print(f"File {filename} does not exist. Creating a new file.")
        with open(filename, 'w') as f:
            pass  # Create an empty file
    return max_id + 1


def delete_pair(ground_truth_file):
    """
    Deletes an existing pair from the ground truth file.
    The file is read into pairs (each pair is 2 lines), the user selects a pair to delete,
    and the file is re-written without that pair.
    """
    if not os.path.exists(ground_truth_file):
        print("No ground truth file found.")
        return
    with open(ground_truth_file, 'r') as f:
        lines = f.readlines()
    pairs = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            pairs.append((lines[i].strip(), lines[i+1].strip()))
        else:
            pairs.append((lines[i].strip(), ""))
    if not pairs:
        print("No pairs found.")
        return
    print("Existing pairs:")
    for idx, (line1, line2) in enumerate(pairs):
        print(f"Pair {idx}:")
        print("  ", line1)
        print("  ", line2)
    try:
        del_idx = int(input("Enter the index of the pair to delete: ").strip())
        if del_idx < 0 or del_idx >= len(pairs):
            print("Index out of range.")
            return
    except ValueError:
        print("Invalid input.")
        return
    pairs.pop(del_idx)
    with open(ground_truth_file, 'w') as f:
        for pair in pairs:
            f.write(pair[0] + "\n")
            if pair[1]:
                f.write(pair[1] + "\n")
    print("Pair deleted.")

def main(tracklet_dict,cam_list):
    """
    Interactive loop for creating ground truth pairs.
    The user can add a new pair or delete an existing pair.
    The results are saved to a text file.
    """
    try:
        numeric_id_0 = int(cam_list[0].split('imagesc')[1])
        numeric_id_1 = int(cam_list[1].split('imagesc')[1])
    except ValueError:
        print(f"Invalid camera id format: {cam_list[0]}")
        return
    record_dict = {}
    record_dict[f'{cam_list[0]}'] = {}
    record_dict[f'{cam_list[1]}'] = {}

    ground_truth_file = f"{cam_list[0]}_{cam_list[1]}_ground_truth.txt"

    id_index = get_current_id_index(ground_truth_file)

    while True:
        print("\nEnter command: [a]dd pair, [d]elete pair, [q]uit")
        cmd = input("Command: ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 'a':
            cam_choice = input(f"Enter first camera id (enter {numeric_id_0} for {cam_list[0]}, {numeric_id_1} for {cam_list[1]}): ").strip()
            if cam_choice == str(numeric_id_0):
                cam1 = f'{cam_list[0]}'
            elif cam_choice == str(numeric_id_1):
                cam1 = f'{cam_list[1]}'
            else:
                print("Invalid camera choice.")
                continue

            # The other camera:
            cam2 = f'{cam_list[1]}' if cam1 == f'{cam_list[0]}' else f'{cam_list[0]}'

            try:
                track_id_cam1 = int(input(f'Enter track id for {cam1}: ').strip())
                track_id_cam2 = int(input(f'Enter track id for {cam2}: ').strip())
            except ValueError:
                print("Invalid track id input.")
                continue

            if track_id_cam1 not in tracklet_dict[cam1]:
                print(f"Track id {track_id_cam1} not found in {cam1}.")
                continue
            if track_id_cam2 not in tracklet_dict[cam2]:
                print(f"Track id {track_id_cam2} not found in {cam2}.")
                continue

            # Get information for each camera.
            info_cam1 = sorted(tracklet_dict[cam1][track_id_cam1], key=lambda x: x[0])
            info_cam2 = sorted(tracklet_dict[cam2][track_id_cam2], key=lambda x: x[0])
            
            cam1_num = numeric_id_0 if cam1 == f'{cam_list[0]}'else numeric_id_1
            cam2_num = numeric_id_1 if cam1 == f'{cam_list[0]}'else numeric_id_0

            with open(ground_truth_file, 'a') as f:
                min_length = min(len(info_cam1), len(info_cam2))
                for info1, info2 in zip(info_cam1[:min_length], info_cam2[:min_length]):
                    line1 = format_detection_line(cam1_num, info1, id_index)
                    line2 = format_detection_line(cam2_num, info2, id_index)
                    f.write(f"{line1}\n")
                    f.write(f"{line2}\n")

                # Handle overflow if one tracklet is longer than the other
                if len(info_cam1) > min_length:
                    for info1 in info_cam1[min_length:]:
                        line1 = format_detection_line(cam1_num, info1, id_index)
                        f.write(f"{line1}\n")
                elif len(info_cam2) > min_length:
                    for info2 in info_cam2[min_length:]:
                        line2 = format_detection_line(cam2_num, info2, id_index)
                        f.write(f"{line2}\n")
            print(f"Pair with id {id_index} saved.")
            id_index += 1

        elif cmd == 'd':
            # Delete an existing pair.
            delete_pair(ground_truth_file)

        else:
            print("Unknown command. Please enter 'a', 'd', or 'q'.")

if __name__ == "__main__":
    cam_list = ['imagesc003', 'imagesc002']

    tracklet_dict = {}
    for cam in cam_list:
        tracklet_dict[cam] = load_data(cam)
    print("Loaded tracklet data.")
    print(f'len of {cam_list[0]} is {len(tracklet_dict[f"{cam_list[0]}"])}')
    print(f'len of {cam_list[1]} is {len(tracklet_dict[f"{cam_list[1]}"])}')
    main(tracklet_dict,cam_list)
    print("Finished creating ground truth pairs.")

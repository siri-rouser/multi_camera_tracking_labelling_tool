from collections import defaultdict

def analyze_obj_ids(file_path):
    # Step 1: Map each obj_id to a set of camera_ids
    obj_to_cameras = defaultdict(set)
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                camera_id = parts[0]
                obj_id = parts[1]
                obj_to_cameras[obj_id].add(camera_id)

    # Step 2: Count the number of cameras per obj_id
    cam_count_distribution = defaultdict(int)
    for cameras in obj_to_cameras.values():
        cam_count = len(cameras)
        cam_count_distribution[cam_count] += 1

    # Step 3: Print results
    total_unique_objs = len(obj_to_cameras)
    print(f"Total unique obj_ids: {total_unique_objs}")
    print(f"Observed in 2 cameras: {cam_count_distribution[2]}")
    print(f"Observed in 3 cameras: {cam_count_distribution[3]}")
    print(f"Observed in 4 cameras: {cam_count_distribution[4]}")

    return cam_count_distribution

# Example usage
file_path = '..//Multi_CAM_Ground_Turth.txt' # replace with your actual file path
analyze_obj_ids(file_path)

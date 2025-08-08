import cv2
import os
import numpy as np

def get_color(track_id):
    """
    Generate a color based on the track ID.
    """
    np.random.seed(int(track_id) % 1000)  # Seed for consistent color
    color = np.random.randint(0, 255, 3).tolist()
    return tuple(map(int, color))

def frame_process(frames, cam_lists, track_dict, frame_id):
    """
    Process frames and draw bounding boxes with consistent color per track ID.
    """
    for i, frame in enumerate(frames):
        cam_id = cam_lists[i][-1]  # e.g., '1' from 'imagesc001'
        cv2.putText(frame, f"Cam {cam_id}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), thickness=8)
        if cam_id in track_dict:
            if str(frame_id) in track_dict[cam_id]:
                for track_id, bbox in track_dict[cam_id][str(frame_id)]:
                    x1, y1, x2, y2 = bbox
                    color = get_color(track_id)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return frames


def read_tracks(GT_dir):
# {cam_num} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {xworld} {yworld}
    track_dict = {}
    with open(GT_dir, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            cam_id = parts[0]
            track_id = parts[1]
            frame_id = parts[2]
            x1 = float(parts[3])
            y1 = float(parts[4])
            w = float(parts[5])
            h = float(parts[6])
            if cam_id not in track_dict:
                track_dict[cam_id] = {}
            if frame_id not in track_dict[cam_id]:
                track_dict[cam_id][frame_id] = []
            track_dict[cam_id][frame_id].append((track_id, [x1, y1, x1 + w, y1 + h]))
    return track_dict


if __name__ == "__main__":
    cam_list = ['imagesc001', 'imagesc002', 'imagesc003','imagesc004']
    base_dir = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/AIC22_Track1_MTMC_Tracking/test/carmel/'
    GT_dir = 'Multi_CAM_Ground_Turth.txt'
    track_dict = read_tracks(GT_dir)
    
    video_list = [os.path.join(base_dir, cam[-4:], 'trimmed_video.mp4') for cam in cam_list]
    caps = [cv2.VideoCapture(path) for path in video_list]

    # Get frame size (assume all videos have the same size)
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    # Output size: 2x2 grid
    out_width = frame_width * 2
    out_height = frame_height * 2

    # Output video writer
    out = cv2.VideoWriter(
        "output_grid.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (out_width, out_height)
    )

    frame_id = 1
    while True:
        rets, frames = zip(*[cap.read() for cap in caps])

        # Stop if any video ends
        if not all(rets):
            break

        frame_process(frames, cam_list,track_dict,frame_id)

        # Resize or pad if needed (optional, assuming same size here)
        # Arrange in 2x2 grid
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top_row, bottom_row))

        # Write to output
        out.write(grid)

        # Optional: display for debug
        # cv2.imshow("Multi-Cam View", grid)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_id += 1  # Increment frame_id for each loop
        print(f"\rProcessed frame {frame_id}", end='', flush=True)

    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()


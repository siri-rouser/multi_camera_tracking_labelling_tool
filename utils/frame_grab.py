import cv2

# Set video path and target frame index
cam_name = 'imagesc004'
video_path = f'/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detect_merge/{cam_name}/tracking_video/tracking_video_{cam_name}.mp4'
target_frame = (540+6)*15 # Change this to the frame number you want to save
save_path = f'{cam_name}_{target_frame:06d}.jpg'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Set the position of the video to the target frame
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

# Read the frame
ret, frame = cap.read()

if ret:
    # Save the frame as an image
    cv2.imwrite(save_path, frame)
    print(f"Frame {target_frame} saved as {save_path}")
else:
    print(f"Error: Could not read frame {target_frame}")

cap.release()

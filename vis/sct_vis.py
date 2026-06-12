import argparse
from pathlib import Path

import cv2

from common import draw_detection, find_frame, read_sct_tracks, stable_color


def track_bounds(frames):
    bounds = {}
    for frame_id, detections in frames.items():
        for detection in detections:
            bounds.setdefault(detection.track_id, [frame_id, frame_id])
            bounds[detection.track_id][0] = min(bounds[detection.track_id][0], frame_id)
            bounds[detection.track_id][1] = max(bounds[detection.track_id][1], frame_id)
    return bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize single-camera tracking results.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder containing imgXXXXXX frames.")
    parser.add_argument("--tracks", type=Path, required=True, help="Tracking text file.")
    parser.add_argument("--output", type=Path, default=Path("outputs/sct_vis.mp4"), help="Output video path.")
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=-1,
        help="Image-frame minus track-frame offset. Use -1 when labels are one-indexed and images are zero-indexed.",
    )
    parser.add_argument("--fps", type=float, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = read_sct_tracks(args.tracks)
    if not frames:
        raise ValueError(f"No detections found in {args.tracks}")

    bounds = track_bounds(frames)
    start_frame = min(frames)
    end_frame = max(frames)

    first_image_frame = start_frame + args.frame_offset
    first_frame_path = find_frame(args.image_dir, first_image_frame)
    if first_frame_path is None:
        raise FileNotFoundError(f"No sample frame found in {args.image_dir}")

    first_image = cv2.imread(str(first_frame_path))
    if first_image is None:
        raise ValueError(f"Failed to read sample frame: {first_frame_path}")

    height, width = first_image.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    for track_frame in range(start_frame, end_frame + 1):
        image_frame = track_frame + args.frame_offset
        frame_path = find_frame(args.image_dir, image_frame)
        if frame_path is None:
            raise FileNotFoundError(f"No frame {image_frame} found in {args.image_dir}")

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        for detection in frames.get(track_frame, []):
            first_track_frame, last_track_frame = bounds[detection.track_id]
            color = stable_color(detection.track_id)
            label = str(detection.track_id)

            if track_frame == first_track_frame:
                color = (255, 0, 0)
                label = f"{detection.track_id} start"
            elif track_frame == last_track_frame:
                color = (0, 0, 255)
                label = f"{detection.track_id} end"

            draw_detection(image, detection, label=label, color=color)

        writer.write(image)
        print(f"Processing frame {track_frame}", end="\r")

    writer.release()
    print(f"\nSaved single-camera visualization to {args.output}")


if __name__ == "__main__":
    main()

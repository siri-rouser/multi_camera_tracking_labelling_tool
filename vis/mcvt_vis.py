import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from common import draw_detection, parse_key_value_map, read_mtmc_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-camera tracking labels over video files."
    )
    parser.add_argument("--tracks", type=Path, required=True, help="MTMC track file.")
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Camera/video pair in the form camera_name=/path/to/video.mp4. Repeat for each camera.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/mcvt_grid.mp4"))
    parser.add_argument(
        "--camera-map",
        nargs="*",
        default=[],
        help="Optional raw track camera ID to video camera name mapping, e.g. 1=imagesc001.",
    )
    parser.add_argument("--start-frame", type=int, default=1, help="First one-indexed track frame to render.")
    return parser.parse_args()


def parse_video_args(values: list[str]) -> dict[str, Path]:
    return {
        camera: Path(video_path)
        for camera, video_path in parse_key_value_map(values).items()
    }


def make_grid(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]

    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    height, width = frames[0].shape[:2]
    blank = np.zeros_like(frames[0])

    padded = frames + [blank] * (rows * columns - len(frames))
    row_images = [
        np.hstack(padded[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    return np.vstack(row_images)


def main() -> None:
    args = parse_args()
    videos = parse_video_args(args.video)
    tracks = read_mtmc_tracks(args.tracks, parse_key_value_map(args.camera_map))

    captures = {camera: cv2.VideoCapture(str(path)) for camera, path in videos.items()}
    if any(not capture.isOpened() for capture in captures.values()):
        bad = [camera for camera, capture in captures.items() if not capture.isOpened()]
        raise ValueError(f"Failed to open video(s): {', '.join(bad)}")

    first_capture = next(iter(captures.values()))
    frame_width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_capture.get(cv2.CAP_PROP_FPS) or 15

    sample_grid = make_grid([np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in captures])
    output_height, output_width = sample_grid.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )

    frame_id = args.start_frame
    while True:
        frames = []
        for camera, capture in captures.items():
            ok, frame = capture.read()
            if not ok:
                for cap in captures.values():
                    cap.release()
                writer.release()
                print(f"\nSaved multi-camera video grid to {args.output}")
                return

            cv2.putText(
                frame,
                camera,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                thickness=4,
            )

            for detection in tracks.get(camera, {}).get(frame_id, []):
                draw_detection(frame, detection)

            frames.append(frame)

        writer.write(make_grid(frames))
        frame_id += 1
        print(f"Processed frame {frame_id}", end="\r")


if __name__ == "__main__":
    main()

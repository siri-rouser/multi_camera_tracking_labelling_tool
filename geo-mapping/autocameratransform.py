import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from CameraFit import AutofitConfig, CameraFit
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit camera parameters from matched image pixels and GPS landmarks."
    )
    parser.add_argument("config_path", type=Path, help="Path to an autofit YAML config.")
    parser.add_argument(
        "-l",
        "--lower-angle-x",
        type=float,
        help="Lower bound for view_x_deg grid search.",
    )
    parser.add_argument(
        "-u",
        "--upper-angle-x",
        type=float,
        help="Upper bound for view_x_deg grid search.",
    )
    parser.add_argument(
        "-s",
        "--step-size",
        type=float,
        default=1,
        help="Step size for the view_x_deg grid search.",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=5,
        help="Number of fits to run for each view_x_deg candidate.",
    )
    parser.add_argument(
        "-c",
        "--camera-name",
        type=str,
        default="cam01",
        help="Name used as the output file prefix.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots, top views, and fitted camera JSON are written.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> AutofitConfig:
    with config_path.open("r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)
    return AutofitConfig.model_validate(yaml_config)


def validate_grid_search_args(args: argparse.Namespace) -> None:
    search_args = (args.lower_angle_x, args.upper_angle_x)
    if any(value is not None for value in search_args) and not all(
        value is not None for value in search_args
    ):
        raise ValueError("Both --lower-angle-x and --upper-angle-x are required for grid search.")
    if args.step_size <= 0:
        raise ValueError("--step-size must be greater than zero.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero.")
    if args.lower_angle_x is not None and args.lower_angle_x >= args.upper_angle_x:
        raise ValueError("--lower-angle-x must be less than --upper-angle-x.")


def fit_camera(config: AutofitConfig, args: argparse.Namespace) -> tuple[CameraFit, float | None]:
    if args.lower_angle_x is None:
        camera = CameraFit(fitconfig=config)
        return camera, config.camera_parameters.rectilinear_projection.view_x_deg

    best_camera = None
    best_view_x = None
    best_distance = float("inf")
    batch_mins = []

    view_x_deg_grid = np.arange(args.lower_angle_x, args.upper_angle_x, args.step_size)
    if len(view_x_deg_grid) == 0:
        raise ValueError("No view_x_deg candidates were generated. Check the grid bounds.")

    for view_x_deg in tqdm(view_x_deg_grid, desc="Searching view_x_deg"):
        batch_min = float("inf")

        for _ in range(args.repeats):
            run_config = config.model_copy(deep=True)
            run_config.camera_parameters.rectilinear_projection.view_x_deg = float(view_x_deg)
            camera = CameraFit(fitconfig=run_config)
            distance = camera.get_perf()

            if distance < best_distance:
                best_camera = camera
                best_view_x = float(view_x_deg)
                best_distance = distance

            batch_min = min(batch_min, distance)

        batch_mins.append((float(view_x_deg), round(batch_min, 2)))

    print(f"Grid-search minima: {batch_mins}")
    if best_camera is None:
        raise RuntimeError("Camera fitting did not produce a candidate.")
    return best_camera, best_view_x


def save_outputs(camera: CameraFit, camera_name: str, output_root: Path) -> None:
    output_dir = output_root / camera_name
    output_dir.mkdir(parents=True, exist_ok=True)

    camera.plot_fit_information_image_space(output_dir / f"{camera_name}_info.png")
    camera.plot_trace(output_dir / f"{camera_name}_trace.png")

    undistorted = cv2.cvtColor(camera.get_undistorted_image(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{camera_name}_undistorted.png"), undistorted)

    topview = camera.get_topview()
    if topview is not None:
        topview_bgr = cv2.cvtColor(topview, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{camera_name}_topview.jpg"), topview_bgr)

    camera.save_cam(output_dir / f"{camera_name}_fitted_cam.json")


def main() -> None:
    args = parse_args()
    validate_grid_search_args(args)

    config = load_config(args.config_path)
    best_camera, best_view_x = fit_camera(config, args)

    best_camera.print_parameters()
    print(
        f"Best solution: average landmark error {best_camera.get_perf():.2f} meters "
        f"(view_x_deg={best_view_x})"
    )

    save_outputs(best_camera, args.camera_name, args.output_dir)


if __name__ == "__main__":
    main()

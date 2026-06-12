# Geo-Mapping Camera Fitting Tool

This tool estimates camera parameters from a single image and a set of matched
landmarks. Each landmark connects an image pixel to a GPS coordinate. The fitted
camera can then project image points into world coordinates and generate quick
diagnostic plots.

## What It Does

- Fits unknown camera orientation, elevation, and lens distortion parameters.
- Supports a grid search over horizontal field of view (`view_x_deg`).
- Writes fitting diagnostics, an undistorted image, a top-view image, and a
  fitted camera JSON file.

## Installation

From the repository root:

```bash
poetry install --no-root
```

## Prepare A Config

Copy the example config and edit it for your camera:

```bash
cp geo-mapping/autofit.example.yaml geo-mapping/autofit.my-camera.yaml
```

Update:

- `image_path`: a still frame from the camera.
- `camera_parameters.gps_location`: the camera latitude and longitude.
- `camera_parameters.rectilinear_projection`: at least one of `focallength_mm`,
  `view_x_deg`, or `view_y_deg`.
- `landmarks`: image pixels paired with GPS coordinates.

## Run

Fit one camera:

```bash
poetry run python geo-mapping/autocameratransform.py \
  geo-mapping/autofit.my-camera.yaml \
  --camera-name cam01 \
  --output-dir outputs
```

Run a field-of-view search:

```bash
poetry run python geo-mapping/autocameratransform.py \
  geo-mapping/autofit.my-camera.yaml \
  --lower-angle-x 45 \
  --upper-angle-x 95 \
  --step-size 1 \
  --repeats 5 \
  --camera-name cam01
```

Outputs are written to `outputs/<camera-name>/`.
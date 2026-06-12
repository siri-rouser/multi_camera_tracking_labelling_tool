# Multi-Camera Tracking Labeling Tool

This repository is a toolkit for labeling ground truth data in multi-camera object tracking datasets. It provides a suite of tools for manually correcting the outputs of object detectors and trackers, as well as for associating single-camera tracklets into multi-camera trajectories.

We use this toolkit to label [RoundaboutHD](https://github.com/siri-rouser/RoundaboutHD).

<!-- 
## Update

Several utility scripts have been added to the `./utils` directory to support various tasks related to multi-camera tracking and data processing. Below is a brief description of each script:

1. bbox_count.py  
   Counts the total number of bounding boxes in your detection results.

2. det_filter.py  
   Filters detection results based on custom thresholds for confidence score and bounding box size.

3. det_label_plot.py  
   Visualizes detection results by plotting bounding boxes and labels on the images.

4. det_wipe.py  
   Removes duplicate detections that appear at the same location across multiple frames, based on a confidence threshold.

5. detection_crop_tool.py  
   Crops and saves all detected objects as individual image files (e.g., img000000_001.jpg, img000000_002.jpg, ...).

6. frame_grab.py  
   Extracts and saves a specific frame from a video file.

7. get_missing_vehicle.py  
   Compares the Excel annotation and SCT ground truth to identify missing vehicle IDs in the SCT result.

8. read_reid_dict.py  
   Converts cross-camera vehicle matching information from an Excel file into a formatted ground truth .txt file. The example xlsx file is `Vehicle Tracking Example.xlsx`.

9. reid_subset_creation.py  
   Generates image-based datasets for multi-camera vehicle re-identification (ReID) from tracking results.

10. sct2det.py  
    Converts single-camera tracking (SCT) results into detection label format.

11. vehicle_num_count1.py  
    Counts the number of unique vehicles from multi-camera tracking ground truth files.

12. vehicle_num_count.py  
    Counts the number of unique vehicles from single-camera tracking ground truth files.

13. vehicle_statistics.py  
    Reads vehicle statistics from an Excel file and visualizes the distribution using pie charts.
-->

## Functionality Overview

This toolkit is designed to help you:

- **Manually correct detection and tracking results**: We use the YOLOv12x.pt model for object detection, ResNet101 for feature extraction, and SMILEtrack for tracking.
- **Associate single-camera tracklets into multi-camera trajectories**: We use an Excel file (see `Vehicle Tracking Example.xlsx`) to generate MCVT ground truth files in the following format: `{cam} {id} {frame} {x1} {y1} {width} {height} {xworld} {yworld}`
- **Visualization**: Tools are provided to visualize both single-camera and multi-camera tracking results.

> **Note:** You need to have your own object detector and object tracker before using this toolkit.

## Installation

1. **Install Poetry**  
   If you haven't installed Poetry yet, run:
   ```
   pip install poetry
   ```

2. **Install project dependencies**  
   Navigate to the repository directory and run:
   ```
   poetry install --no-root
   ```

## Main Script Descriptions

- **detect_correction.py**  
  Allows you to manually review and correct detection results. Keybindings:
  | Key | Action |
  |-----|--------|
  | `a` | Add a detection |
  | `d` | Delete a detection |
  | `s` | Save corrections for the current frame |
  | `q` | Quit the program |

- **detection_result_process.py**  
  Saves all detection results locally as text files.  
  > Tip: You can run the full post-processing pipeline more quickly with:
  > ```
  > bash detection_post_process.sh
  > ```

- **sct_correction.py**  
  Correct and edit single-camera tracking results. Keybindings:
  | Key | Action |
  |-----|--------|
  | `i` | Merge two tracklets into one unified trajectory |
  | `d` | Delete an existing tracklet |
  | `w` | Save results to `.txt` and render a `.mp4` visualization |
  | `q` | Quit without saving |

- **tracklet_post_process.py**  
  Interpolates single-camera tracking results (for gaps shorter than 5 frames) to prepare tracklets for multi-camera association.

- **cross_camera_match.py** *(Previous Version)*  
  Facilitates manual association of tracklets across cameras to form the ground truth. Keybindings:
  | Key | Action |
  |-----|--------|
  | `a` | Add an association between tracklets from different cameras |
  | `d` | Delete an existing association |
  | `q` | Save associations and exit |

  Output is saved to `multi_camera_ground_truth.txt` (an example file is included in the repository).

  > **Note:** In the current version, `cross_camera_match.py` has been replaced by `MCVT_data_creation.py`.

- **MCVT_data_creation.py**  
  We found `cross_camera_match.py` to be impractical when dealing with many cameras. Instead, all potential camera pairs are saved to an Excel file, which is then read to generate the MCVT ground truth.

  ```
  python MCVT_data_creation.py /path/to/your/file.xlsx
  ```

## Evaluation

- **eval_label.py**  
  Evaluates results against the multi-camera tracking ground truth.
  ```
  python eval_label.py <ground_truth> <prediction>
  ```

- **eval_sct.py**  
  Evaluates results against the single-camera tracking ground truth.
  ```
  python eval_sct.py <ground_truth> <prediction>
  ```

- **eval_det.py**  
  Evaluates results against the object detection ground truth. Takes a folder path as input; the folder should contain per-frame detection files (e.g., `img000000.txt`, `img000001.txt`, ...).
  ```
  python eval_det.py <ground_truth_path> <prediction_path>
  ```

<!-- ## Example Directory Structure

An example dataset structure is provided below:

```
Dataset/
├── detection/
│   ├── imagesNB/
│   │   └── img1/
│   │       ├── img000000.jpg
│   │       ├── img000001.jpg
│   │       └── img000xxxxxxx.jpg
│   └── imagesSB/
│       └── img1/
│           ├── img000000.jpg
│           ├── img000001.jpg
│           └── img000xxxxxxx.jpg
└── detect_merge/
    ├── imagesNB/
    │   ├── dets/
    │   │   ├── img000000_000.jpg
    │   │   ├── img000000_001.jpg
    │   │   └── img000001_000.jpg
    │   ├── dets_corrected/
    │   │   ├── img000000_000.jpg
    │   │   ├── img000000_001.jpg
    │   │   └── img000001_000.jpg
    │   ├── dets_debug/
    │   │   ├── img000000.jpg
    │   │   ├── img000001.jpg
    │   │   └── img000xxxxxxx.jpg
    │   ├── feature/
    │   │   ├── img000000.json
    │   │   ├── img000001.json
    │   │   └── img000xxxxxxx.json
    │   ├── images_corrected/
    │   │   ├── img000000.jpg
    │   │   ├── img000001.jpg
    │   │   └── img000xxxxxxx.jpg
    │   ├── labels/
    │   │   ├── img000000.txt
    │   │   ├── img000001.txt
    │   │   └── img000xxxxxxx.txt
    │   ├── labels_corrected/
    │   │   ├── img000000.txt
    │   │   ├── img000001.txt
    │   │   └── img000xxxxxxx.txt
    │   ├── labels_filtered/
    │   │   ├── img000000.txt
    │   │   ├── img000001.txt
    │   │   └── img000xxxxxxx.txt
    │   ├── labels_xy/
    │   │   ├── img000000.txt
    │   │   ├── img000001.txt
    │   │   └── img000xxxxxxx.txt
    │   └── tracking_video/
    │       └── imagesNB_tracking_interpolated.mp4
``` -->

## Workflow Example

This workflow outlines the full pipeline for **Multi-Camera Multi-Object (Vehicle) Tracking**:

1. **Object Detection** (e.g., YOLO)
2. **detect_correction.py** — Bounding box refinement
3. **detection_crop_tool.py** — Prepare detections for feature extraction
4. **Feature Extraction** (e.g., ResNet)
5. **Single Camera Tracker** (e.g., DeepSORT)
6. **sct_video_process.py** — Post-process single-camera tracking results (step 1)
7. **sct_correction.py** — Post-process single-camera tracking results (step 2)
8. Manually inspect SCT results and create the Excel annotation file
9. **MCVT_data_creation.py** — Generate multi-camera ground truth from the Excel file
10. **MCVT_vis.py** — Visualization of tracking results

## Other Functions
See /vis and /geo-mapping

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

MIT License

Copyright (c) 2025 Yuqiang Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

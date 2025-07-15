# Multi-Camera Tracking Labeling Tool

This repository contains a toolkit for labeling ground truth data in multi-camera object tracking datasets. It provides a suite of tools for manually correcting the outputs of object detectors and trackers, as well as for associating single-camera tracklets into multi-camera trajectories.

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
   Converts cross-camera vehicle matching information from an Excel file into a formatted ground truth .txt file. The example of xlsx file is `Vehicle Tracking Example.xlsx`

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

## Overview

**Note before use: You need to have your own object-detector and object-tracker before using this toolkit.**

A typical multi-camera tracking pipeline includes:
1. **Object Detection**
2. **Feature Extraction**
3. **Single-Camera Tracking**
4. **Multi-Camera Tracking**

This toolkit is designed to help you:
- **Manually correct detection results**: We use the YOLOv12x.pt model for object detection, ResNet101 for feature extraction, and SMILEtrack for tracking.
- **Associate single-camera tracklets into multi-camera trajectories**: Generate ground truth files with the following format:{cam} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {xworld} {yworld}

where:
- **cam**: Camera ID
- **id_index**: Object ID
- **frame_num**: Frame number
- **x1**: Left x-coordinate of the bounding box
- **y1**: Top y-coordinate of the bounding box
- **width**: Width of the bounding box
- **height**: Height of the bounding box
- **xworld, yworld**: World coordinates

## Installation

1. **Install Poetry**  
 If you haven't installed Poetry, run: pip install poetry

2. **Install Project Dependencies**  
Navigate to the repository directory and run: poetry install --no-root

## Script Descriptions

- **detect_correction.py**  
Allows you to manually review and correct detection results. Use the following keys:
- `a`: Add a detection.
- `d`: Delete a detection.
- `s`: Save the corrections for the current frame.
- `q`: Quit the program.

- **detection_result_process.py**  
Saves all detection results locally as text files.

Note: You can do the below to get the post-processed result in a quicker way.
```
bash detection_post_process.sh
```
---
- **tracklet_post_process.py**  
Interpolates single-camera tracking results (for gaps shorter than 5 frames) to prepare the tracklets for multi-camera association.

---

- **cross_camera_match.py**  
Facilitates manual association of tracklets across different cameras to form the ground truth. Use:
- `a`: Add an association between tracklets from different cameras.
- `d`: Delete an existing association.
- `q`: Save the associations and exit.

The output is saved in `multi_camera_ground_truth.txt` (an example file is provided in the repository).

**Note**: In the updated version: this `cross_camera_match.py` should be replaced by read_reid_dict.py for this current version.

---

- **eval_label.py**  
Evaluates your results against the multi-camera tracking ground truth. Usage:python eval_label.py <ground_truth> <prediction>

- **GT_vis.py**  
Visualizes the ground truth results.

- **sct_video_process.py**  
Visualizes and interpolate the single camera tracking results(for GT).

- **sct_correction.py**  
Correct and edit the single camera tracking results
- `i`: Intergated two tracklets into one unified trajectory
- `d`: Delete an existing tracklet.
- `w`: Write and quite the corrected/edited results into .txt and visualize it to a .mp4 file.
- `q`: quit the program without saving the results.

- **sct_vis.py**  
Used for vis the single-camera tracking results by my own tracker.

- **eval_sct.py** 
Evaluate the results against the single camera tracking ground turth. Usage:python eval_label.py <ground_truth> <prediction>

- **eval_det.py** 
Evaluate the results against the sobject detection ground turth. It takes a folder path as input, under this folder it should contain detection results e.g. img000000.txt, img000001.txt... Usage:python eval_label.py <ground_truth_path> <prediction_path>

## Example Directory Structure

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
```

## Workflow Example

This workflow outlines the pipeline for **Multi-Camera Multi-Object (Vehicle) Tracking**:

1. **Object Detection** (e.g., YOLO)
2. **wipe_point.py** (Noise removal from detections)
3. **detect_correction.py** (Bounding box refinement)
4. **detection_crop_tppl.py** (Prepare detections for feature extraction)
5. **Feature Extraction** (e.g., ResNet)
6. **Single Camera Tracker** (e.g., DeepSORT)
7. **sct_video_process.py** (Post-process single-camera tracking results1)
8. **sct_correction.py** (Post-process single-camera tracking results2)
9. **cross_camera_match.py** (Match objects across multiple cameras)
10. **GT_vis.py** (Visualization of tracking results)


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
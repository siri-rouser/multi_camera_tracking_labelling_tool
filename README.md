# Multi-Camera Tracking Labeling Tool

This repository contains a toolkit for labeling ground truth data in multi-camera object tracking datasets. It provides a suite of tools for manually correcting the outputs of object detectors and trackers, as well as for associating single-camera tracklets into multi-camera trajectories.

**Note before use: You need to have your own object-detector and object-tracker before using this toolkit.**

## Overview

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

- **wipe_point.py**  
Removes duplicate detections (above a confidence threshold) that occur at the same location across multiple frames.

- **detection_correction.py**  
Allows you to manually review and correct detection results. Use the following keys:
- `a`: Add a detection.
- `d`: Delete a detection.
- `s`: Save the corrections for the current frame.
- `q`: Quit the program.

- **detection_result_process.py**  
Saves all detection results locally as text files.

- **detection_crop_tool.py**  
Crops all corrected detections and saves them locally.

- **tracklet_post_process.py**  
Interpolates single-camera tracking results (for gaps shorter than 5 frames) to prepare the tracklets for multi-camera association.

- **cross_camera_match.py**  
Facilitates manual association of tracklets across different cameras to form the ground truth. Use:
- `a`: Add an association between tracklets from different cameras.
- `d`: Delete an existing association.
- `q`: Save the associations and exit.

The output is saved in `multi_camera_ground_truth.txt` (an example file is provided in the repository).

- **eval_label.py**  
Evaluates your results against the ground truth. Usage:python eval_label.py <ground_truth> <prediction>

- **GT_vis.py**  
Visualizes the ground truth results.

- **video_process.py**  
Visualizes the single-camera tracking results.

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






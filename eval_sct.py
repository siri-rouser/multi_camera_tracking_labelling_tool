#!/usr/bin/env python
"""
Evaluate single camera tracking results using motmetrics.

Expected input file format (txt):
    frame, track_id, x1, y1, x2, y2, cls

Usage:
    python eval.py groundtruth.txt testdata.txt
"""

import sys
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm

def load_data(filepath):
    """
    Loads tracking data from a text file.
    
    The file is assumed to have 7 columns:
      frame, track_id, x1, y1, x2, y2, cls
      
    This function computes the bounding box parameters (X, Y, Width, Height)
    where X, Y correspond to the top-left corner (x1, y1) and width and height
    are computed as (x2 - x1) and (y2 - y1) respectively.
    
    Parameters
    ----------
    filepath : str
        Path to the txt file.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with a multi-index of (frame, track_id) and columns
        ['X', 'Y', 'Width', 'Height'].
    """
    try:
        # Read the file; support whitespace or comma as delimiter.
        df = pd.read_csv(filepath, header=None, 
                         names=["frame", "track_id", "x1", "y1", "x2", "y2", "cls"],
                         sep=r'\s+|,', engine='python')
    except Exception as e:
        sys.exit("Error reading {}: {}".format(filepath, e))
    
    # Compute bounding box in (X, Y, Width, Height) format.
    df["X"] = df["x1"]
    df["Y"] = df["y1"]
    df["Width"] = df["x2"] - df["x1"]
    df["Height"] = df["y2"] - df["y1"]
    
    # Optionally, remove duplicate detections (if any) for the same frame and track id.
    df = df.drop_duplicates(subset=["frame", "track_id"], keep='first')
    
    # Set multi-index as (frame, track_id) as expected by motmetrics.
    df = df.set_index(["frame", "track_id"])
    df.index.names = ['FrameId', 'Id']
    
    return df[["X", "Y", "Width", "Height"]]

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single camera tracking results using motmetrics."
    )
    parser.add_argument("groundtruth", help="Path to ground truth txt file")
    parser.add_argument("prediction", help="Path to prediction txt file")
    args = parser.parse_args()
    
    # Load ground truth and prediction data.
    gt = load_data(args.groundtruth)
    pred = load_data(args.prediction)
    print("Ground truth data:", gt)
    print("Prediction data:", pred)

    # Use motmetrics' utility function to compare to ground truth using IOU.
    try:
        acc = mm.utils.compare_to_groundtruth(gt, pred, dist='iou', distfields=None, distth=0.5)
    except Exception as e:
        sys.exit("Error during evaluation: {}".format(e))
    
    # Create the metrics handler and compute tracking metrics.
    mh = mm.metrics.create()
    # Here we use the standard MOTChallenge metrics (which include measures such as MOTA, MOTP, IDF1, etc.)
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="SingleCamera")
    
    # Render the results as a human-readable summary.
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    
if __name__ == '__main__':
    main()

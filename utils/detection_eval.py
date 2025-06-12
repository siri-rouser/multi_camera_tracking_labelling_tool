import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np


def parse_label_file(path: str, with_score: bool = False) -> List[Tuple[int, float, float, float, float, float]]:
    """Read a detection or ground truth label file.

    Returns a list of tuples (cls, x1, y1, x2, y2, score).
    For ground truth, score will be 1.0.
    """
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x1, y1, x2, y2 = map(float, parts[1:5])
            score = float(parts[5]) if with_score and len(parts) > 5 else 1.0
            boxes.append((cls, x1, y1, x2, y2, score))
    return boxes


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter
    return inter / union


def compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute AP using the VOC 2010 method."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_detection(gt_dir: str, pred_dir: str, iou_thresholds: List[float]):
    """Evaluate detection results under given IoU thresholds."""
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".txt")])

    # Load ground truths and predictions
    gts: Dict[str, List[Tuple[int, float, float, float, float, float]]] = {}
    preds: List[Tuple[str, int, float, float, float, float, float]] = []
    classes = set()
    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        gt_boxes = parse_label_file(gt_path)
        gts[fname] = gt_boxes
        for b in gt_boxes:
            classes.add(b[0])

        pred_path = os.path.join(pred_dir, fname)
        pred_boxes = parse_label_file(pred_path, with_score=True)
        for b in pred_boxes:
            preds.append((fname, b[0], b[1], b[2], b[3], b[4], b[5]))
            classes.add(b[0])

    results = {}
    for thr in iou_thresholds:
        ap_per_class = []
        for cls in classes:
            npos = 0
            gt_used = defaultdict(list)
            for fname, boxes in gts.items():
                for idx, b in enumerate(boxes):
                    if b[0] == cls:
                        npos += 1
                        gt_used[fname].append(False)
            if npos == 0:
                continue

            predictions = [p for p in preds if p[1] == cls]
            predictions.sort(key=lambda x: x[6], reverse=True)

            tp = np.zeros(len(predictions))
            fp = np.zeros(len(predictions))
            for i, p in enumerate(predictions):
                fname, _, x1, y1, x2, y2, score = p
                max_iou = 0.0
                max_idx = -1
                boxes = [b for b in gts.get(fname, []) if b[0] == cls]
                for j, b in enumerate(boxes):
                    iou = compute_iou((x1, y1, x2, y2), (b[1], b[2], b[3], b[4]))
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                if max_iou >= thr and not gt_used[fname][max_idx]:
                    tp[i] = 1
                    gt_used[fname][max_idx] = True
                else:
                    fp[i] = 1

            if tp.size == 0:
                ap_per_class.append(0.0)
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / npos
            prec = tp_cum / (tp_cum + fp_cum)
            ap = compute_ap(rec, prec)
            ap_per_class.append(ap)
        results[thr] = np.mean(ap_per_class) if ap_per_class else 0.0
    mAP = np.mean(list(results.values())) if results else 0.0
    results["mAP"] = mAP
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument("gt_dir", help="Ground truth directory")
    parser.add_argument("pred_dir", help="Prediction directory")
    args = parser.parse_args()

    thresholds = [0.5, 0.75, 0.9]
    res = evaluate_detection(args.gt_dir, args.pred_dir, thresholds)
    for t in thresholds:
        print(f"AP@{t}: {res[t]:.4f}")
    print(f"mAP: {res['mAP']:.4f}")
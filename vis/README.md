# Visualization Tools

These scripts create quick MP4 visualizations for tracking labels. They are
intended as lightweight debugging tools rather than a full annotation UI.

## Single-Camera Tracks

```bash
poetry run python vis/sct_vis.py \
  --image-dir /path/to/imagesNB/img1 \
  --tracks imagesNB_mot_interpolated_final.txt \
  --output outputs/sct_vis.mp4
```

## Multi-Camera Videos

```bash
poetry run python vis/mcvt_vis.py \
  --tracks Multi_CAM_Ground_Truth.txt \
  --video imagesc001=/path/to/c001.mp4 \
  --video imagesc002=/path/to/c002.mp4 \
  --video imagesc003=/path/to/c003.mp4 \
  --video imagesc004=/path/to/c004.mp4 \
  --camera-map 1=imagesc001 2=imagesc002 3=imagesc003 4=imagesc004 \
  --output outputs/mcvt_grid.mp4
```

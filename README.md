# Discam

AI motorized camera for Ultimate filming and analysis.

## Data generation

### Fixed camera

Generate training data from footage with a fixed camera that sees the entire field.

Detects motion within a manually marked field region.
Computes bounding box around motion.

File structure:
- A video file somewhere.
- A json file, same name and directory as video file.
  Contains keys `tl`, `tr`, `br`, `bl` in that order.
  Four corners of the field in pixel coordinates (not necessarily rectangular).
- A separate directory where frames and bounding boxes will be saved.

Example:
```
Folder1
|__ video.mp4
|__ video.json

Folder2
|__ 0.bbox.json  # This will be generated
...
```

First, detect motion and compute bounding box every N frames.

```bash
python make_bbox_data.py Folder1/video.mp4 Folder2/
```

Second, interpolate bounding boxes and save frames.
```bash
python make_training_data.py Folder1/video.mp4 Folder2/
```

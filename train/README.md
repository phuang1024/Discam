# Discam model training

## Data generation

Generate training data from footage with a **fixed camera** that sees the entire field.

Detects motion within a manually marked field region.
Computes bounding box around motion.

### Setup

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

Example `bbox.json`:
```json
{
    "tl": [723, 415],
    "tr": [1068, 391],
    "br": [1914, 886],
    "bl": [3, 771]
}
```

### Usage

First, detect motion and compute bounding box every N frames.
Second, interpolate bounding boxes and save frames.

```bash
python make_bbox_data.py Folder1/video.mp4 Folder2/
python make_training_data.py Folder1/video.mp4 Folder2/
```

## Training

### Setup

File structure:
- A directory with subdirs, each one of which is generated data from a video (see above step).
- A separate directory where simulated data and model checkpoints will be saved.

Example:
```
data/
|__ video1/
|   |__ 0.jpg
|   |__ 0.bbox.json
|   |__ ...
|__ video2/
|   |__ ...
|__ ...

results/
|__ ...
```

### Usage

```bash
python train.py --results results/ --data data/ --epochs 100
```

### Results

File structure:
- Under `results/`, a directory is created for each epoch.
- In each epoch, a subdir `data` contains simulated data for that epoch.
- A model checkpoint *may* be saved, depending on the `save_every` argument.
- `results/latest.pt` is saved every epoch.

Using `test.py`:

Make sure to set `--data` to *one* video data directory (not the directory containing all videos).

```bash
python test.py --model results/latest.pt --data data/video1/
```

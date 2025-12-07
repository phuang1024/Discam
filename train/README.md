# Train

## Model

The model is a function.

- Takes in N images, which are N frames of the captured video with some fixed time step.
- Outputs PTZ or edge weighs TODO

## Training

Training data is film of a game with a ground truth bounding box computed at every frame.
I.e. the bbox is where the camera should be looking, ideally.

For each sample, take N frames with a fixed frame step. Critically, each frame is cropped
to it's bbox plus some offset. E.g. shift the bbox right or up, shrink or expand it slightly.

This offset is constant across each of the N frames (although, of course, it will randomly
change with different training samples).

The ground truth edge weights are computed using this offset.

The model's output and the ground truth edge weights are compared.

## Files

Scripts to run:

- `make_data.py`: Generate training data.
- `vis_data.py`: Visualize training data from `make_data.py`.
- `train.py`: Train the model.
- `test.py`: Visualize model inference.

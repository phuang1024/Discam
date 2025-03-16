# Algorithm

This algorithm crops the video to follow the action in the film.

The model is a CNN that takes in a single frame (possibly at lower resolution)
and outputs a heatmap of salient pixels (or downscaled patches, if Pooling exists).

To generate training data, we take a film with a stationary camera. Therefore
we know exactly where the field is for the entire film.

We use YOLO to detect all humans. We draw a bounding box around humans in the field.
This is the label.

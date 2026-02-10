Components
==========

This page describes the various software components in this project.

In general, the functionality and theory are documented here. How to use it and
implementation details are in the code.


Tracking
--------

This module detects, tracks and classifies person movement in a segment of
footage, to determine whether they are an active player. Detection and tracking
is done with YOLO, and we train a neural network to classify trajectories.

A track is a time series of bounding boxes which correspond to a single person's
movement. The length of each track is variable. TODO. Tracks can end even when
the player is fully visible, due to tracker inaccuracies. We may limit the
length of a track. Classifications are per-track. TODO, research performance.

Neural network
^^^^^^^^^^^^^^

Ideas:

Constant FPS, set a max length.

Input is (N, 4). N is number of points in time series. 4 is
``(x, y, vel_x, vel_y)``. Normalize x, y to be 0-1 according to frame width,
and subtract the COM so they are around 0. Velocity should be something more
than just consecutive elements subtract. Maybe moving window average.

For tracks shorter than max length, zero pad the rest, AND pass in a "mask"
which says which points are actual data.

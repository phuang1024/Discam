Components
==========

This page describes the various software components in this project. This is
from a developer's perspective, providing a high level overview of the
functionality. Specifics are documented in code, and the module's README.


Tracking
--------

The purpose of this module is to detect, track and classify players in a
segment of footage. Detection and tracking is done with YOLO, and we train a
neural network to classify trajectories.

A track is a time series of bounding boxes which correspond to a single person's
movement. The length of each track is variable. TODO. Tracks can end even when
the player is fully visible, due to tracker inaccuracies. We may limit the
length of a track. Classifications are per-track. TODO, research performance.

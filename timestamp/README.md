# Timestamp-er

The goal of this module is to automatically create timestamps of points.
I.e. detect when each point starts and save those timestamps, for ease of viewing.

The NN classifies whether each moment is in play.
Querying the NN on clips of the entire game allows us to determine sections
in play.
Apply some filter, and find the beginning of each section for the timestamps.

## NN format

Input:

- Video clip (3D tensor).
- Sampled at some low FPS.
- A few seconds in length.

Output:

- Binary clasification of whether this clip is game in play.

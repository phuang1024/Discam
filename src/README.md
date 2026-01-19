# Usage

Run this program to deploy the PTZ camera.

Simultaneously records and gives PTZ commands.


# Design

The main thread spawns worker threads.

Globals (shared between threads):
- Run flag (bool).
- Frame queue.
- `cv2.VideoCapture` of the camera.

Threads:
- Main thread: Creates globals and spawns threads.
- Reader thread: Reads frames from camera and adds to queue.
- Writer thread: Writes frames from queue to video file.
- PTZ thread: Periodically computes PTZ movements, and sends to camera
  via `VideoCapture.set()`.

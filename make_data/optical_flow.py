"""
Utils relating to optical flow.
"""

from collections import deque

import cv2
import numpy as np

# Number of frames per optical flow chunk.
CHUNK_SIZE = 60
# Points must track for at least this many frames to be considered.
MIN_DURATION = CHUNK_SIZE // 2
# Pixels per frame threshold. Actually, this is min speed.
MIN_VELOCITY = 1
# Players with lower Y coords are farther from the camera, so scale velocity up.
VELOCITY_Y_SCALING = 4


def optical_flow(frames: deque, y_min, y_max):
    """
    Run optical flow on a sequence of frames.
    Points are detected once, at the beginning.

    Returns a set of points with sufficient velocity per frame.
    """
    first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    init_points = cv2.goodFeaturesToTrack(
        first_frame,
        maxCorners=3000,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    # Trajectories of all points. List of (x, y) locations.
    # As some points fail to track, corresponding trajectories will stop growing.
    all_trajs = [[] for _ in range(len(init_points))]
    for i, p in enumerate(init_points):
        # p has an extra dimension.
        all_trajs[i].append(p)

    prev_frame = first_frame
    # A mutable copy of points still tracking.
    curr_trajs = all_trajs
    for i in range(1, len(frames)):
        if len(curr_trajs) == 0:
            break

        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Find new points.
        prev_points = np.array([t[-1] for t in curr_trajs])
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None)
        status = status.flatten() == 1

        # Append good points to their trajectories.
        for j in range(len(curr_trajs)):
            if status[j]:
                curr_trajs[j].append(curr_points[j])

        # Remove lost points.
        curr_trajs = [traj for j, traj in enumerate(curr_trajs) if status[j]]

        prev_frame = curr_frame

    # Points returned from above have a redundant dimension.
    for i in range(len(all_trajs)):
        all_trajs[i] = [p[0] for p in all_trajs[i]]

    #vis_optical_flow(frames, all_trajs)

    # Remove short trajectories.
    all_trajs = [traj for traj in all_trajs if len(traj) >= MIN_DURATION]

    # For each frame, get the points with sufficient velocity.
    speeds = np.zeros(len(all_trajs))
    for i in range(len(all_trajs)):
        for j in range(1, len(all_trajs[i])):
            speeds[i] += np.linalg.norm(all_trajs[i][j] - all_trajs[i][j - 1])
        speeds[i] /= len(all_trajs[i])

        # Scale velocity by Y position.
        y = all_trajs[i][0][1]
        scale = np.interp(y, [y_min, y_max], [VELOCITY_Y_SCALING, 1])
        scale = np.clip(scale, 1, VELOCITY_Y_SCALING)
        speeds[i] *= scale

    good_trajs = speeds >= MIN_VELOCITY
    all_trajs = [traj for i, traj in enumerate(all_trajs) if good_trajs[i]]

    vis_optical_flow(frames, all_trajs)


def vis_optical_flow(frames, trajectories):
    """
    Note: Modifies frames in place.
    """
    colors = np.random.randint(0, 255, (len(trajectories), 3))
    for i, traj in enumerate(trajectories):
        for j, point in enumerate(traj):
            cv2.circle(frames[j], point.astype(int), 3, colors[i].tolist(), -1)

    for i, frame in enumerate(frames):
        cv2.imshow("a", frame)
        cv2.waitKey(10)


def chunked_optical_flow(video_path, bounds):
    """
    Returns a set of points with sufficient velocity per frame.

    If batch size is N, the window is shifted by N//2 each time.
        I.e. 0th batch: frames 0..N-1
        1st batch: frames N//2..N//2 + N - 1
    Therefore, each frame is processed twice.
    The points for each frame are concatenated across both runs.
    """
    # Field Y bounds
    y_min = min(b[1] for b in bounds.values())
    y_max = max(b[1] for b in bounds.values())

    video = cv2.VideoCapture(str(video_path))

    frames = deque(maxlen=CHUNK_SIZE)

    while True:
        for _ in range(CHUNK_SIZE // 2):
            ret, frame = video.read()
            if not ret:
                return None
            frames.append(frame)

        if len(frames) >= CHUNK_SIZE:
            optical_flow(frames, y_min, y_max)

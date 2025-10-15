"""
Utils relating to optical flow.
"""

from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

# Number of frames per optical flow chunk.
CHUNK_SIZE = 60
# Points must track for at least this many frames to be considered.
MIN_DURATION = CHUNK_SIZE // 2
# Pixels per frame threshold. Actually, this is min speed.
MIN_SPEED = 1
# Players with lower Y coords are farther from the camera, so scale velocity up.
SPEED_Y_SCALING = 4


def optical_flow(frames: deque):
    """
    Run optical flow on a sequence of frames.
    Features are detected once, on the first frame.

    Returns the trajectories of all points.
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

    return all_trajs


def filter_trajectories(all_trajs, y_min, y_max):
    """
    - Removes short trajectories.
    - Removes trajectories with insufficient speed.
    """
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
        scale = np.interp(y, [y_min, y_max], [SPEED_Y_SCALING, 1])
        scale = np.clip(scale, 1, SPEED_Y_SCALING)
        speeds[i] *= scale

    good_trajs = speeds >= MIN_SPEED
    all_trajs = [traj for i, traj in enumerate(all_trajs) if good_trajs[i]]

    return all_trajs


def chunked_optical_flow(video_path, bounds, max_frames=None):
    """
    Returns a set of salient points per frame.

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
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = deque(maxlen=CHUNK_SIZE)
    points = [[] for _ in range(total_frames)]

    pbar = tqdm(total=total_frames)
    i = 0
    while True:
        for _ in range(CHUNK_SIZE // 2):
            ret, frame = video.read()

            done = False
            done |= not ret
            done |= (max_frames is not None and i >= max_frames)
            if done:
                video.release()
                return points

            frames.append(frame)
            pbar.update(1)
            i += 1

        if len(frames) < CHUNK_SIZE:
            continue
        assert len(frames) == CHUNK_SIZE

        trajs = optical_flow(frames)
        trajs = filter_trajectories(trajs, y_min, y_max)

        for traj in trajs:
            for j, point in enumerate(traj):
                points[i - CHUNK_SIZE + j].append(point)


def vis_optical_flow(video_path, points):
    """
    Visualize salient points per frames.
    """
    video = cv2.VideoCapture(str(video_path))

    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if i >= len(points):
            break

        for p in points[i]:
            cv2.circle(frame, tuple(p.astype(int)), 2, (0, 0, 255), -1)
        cv2.imshow("frame", frame)
        cv2.waitKey(10)

        i += 1

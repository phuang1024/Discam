"""
Detect sections in between points to trim out.
The start is a sudden increase in player count.
The end is sudden decrease in separation metric (see Classifier).
"""

import numpy as np
from scipy.ndimage import median_filter

from ..utils.constants import *


def find_plateaus(data, thres, min_len):
    """
    Find sections above thres for at least min_len contiguous.
    data: ndarray (N,) float.
    return: ndarray (N,) bool.
        True if a plateau *starts* on that index.
    """
    ret = np.zeros_like(data, dtype=bool)
    # Traverse backwards.
    count = 0
    for i in range(len(data) - 1, -1, -1):
        if data[i] > thres:
            count += 1
        else:
            count = 0
        if count >= min_len:
            ret[i] = True
    return ret


def find_trim_sections(pipe_out):
    """
    Find sections to trim from pipeline output.
    pipe_out: List of CVPipeline outputs from each frame.
    return: ndarray (N, 2)
        (start, end) timestamps to trim, in seconds.
    """
    # Extract data.
    counts = np.array([len(x["active_boxes"]) for x in pipe_out], dtype=float)
    seps = np.array([x["sep_metric"] for x in pipe_out], dtype=float)
    # To detect falling edge.
    seps *= -1

    # Filtering.
    counts = median_filter(counts, size=5)
    seps = median_filter(seps, size=5)

    # Find plateaus.
    counts_pos = find_plateaus(counts, 20, DET_FPS * 10)
    speeds_pos = find_plateaus(seps, -3, DET_FPS * 10)

    sections = []
    i = 0
    while i < len(pipe_out):
        if counts_pos[i]:
            # Found end of point.
            start = i
            i += int(MIN_STOP_TIME * DET_FPS)
            while i < len(pipe_out):
                if speeds_pos[i]:
                    # Found start of point.
                    break
                if i - start >= MAX_STOP_TIME * DET_FPS:
                    break
                i += 1
            sections.append((start, i))
            i += int(MIN_PLAY_TIME * DET_FPS)
        else:
            i += 1

    sections = np.array(sections, dtype=np.float32) / DET_FPS
    if len(sections) == 0:
        return sections
    sections[:, 0] += TRIM_MARGIN
    sections[:, 1] -= TRIM_MARGIN
    return sections


def gen_timestamps(sections):
    """
    Generate timestamp string in YT format.
    sections: Sections that were trimmed, in seconds.
    """
    times = [0]
    for _, end in sections:
        times.append(end)

    ret = ""
    for t in times:
        t = int(t)
        s = int(t % 60)
        m = int((t // 60) % 60)
        h = int(t // 3600)
        if h > 0:
            string = f"{h}:{m:02d}:{s:02d}"
        else:
            string = f"{m}:{s:02d}"
        ret += string + "\n"
    return ret

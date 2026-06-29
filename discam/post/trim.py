"""
Detect sections in between points to trim.
The start of such a section is detected by a sudden increase in player count.
End is sudden increase in speed.
"""

import matplotlib.pyplot as plt

from utils import *


def smooth_data(counts, speeds):
    """
    Hardcoded.
    EMA on counts.
    HPF on speeds.
    """
    counts = EMA.run_array(counts, 0.2)
    speeds_hpf = EMA.run_array(speeds, 0.2) - EMA.run_array(speeds, 0.01)
    return counts, speeds_hpf


def find_plateaus(data, thres, min_len):
    """
    Find sections above thres for at least min_len contiguous.
    data: 1D array.
    return: bool array of same length.
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


def find_trim_sections(detect_out):
    """
    Detect in between points.
    Returns sections to trim from output video.
    return: ndarray (N, 2)
        (start, end) timestamps in seconds.
    """
    # Extract data.
    counts = [len(x["player_boxes"]) for x in detect_out]
    speeds = [x["speeds"].mean() for x in detect_out]
    # Smoothing.
    counts, speeds = smooth_data(counts, speeds)

    counts_pos = find_plateaus(counts, COUNT_THRES, FPS * PLATEAU_LEN)
    speeds_pos = find_plateaus(speeds, SPEED_THRES, FPS * PLATEAU_LEN)

    ret = []
    i = 0
    while i < len(detect_out):
        if counts_pos[i]:
            # Found end of point.
            start = i
            i += int(MIN_STOP_TIME * FPS)
            while i < len(detect_out):
                if speeds_pos[i]:
                    # Found start of point.
                    break
                if i - start >= MAX_STOP_TIME * FPS:
                    break
                i += 1
            ret.append((start, i))
            i += int(MIN_PLAY_TIME * FPS)
        else:
            i += 1

    ret = np.array(ret, dtype=np.float32) / FPS
    ret[:, 0] += TRIM_MARGIN
    ret[:, 1] -= TRIM_MARGIN
    return ret


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


def plot_data(counts, speeds):
    time = [i / FPS for i in range(len(counts))]
    plt.plot(time, counts)
    plt.plot(time, speeds)
    plt.show()

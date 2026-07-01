"""Detect sections in between points to trim out.
The start is a sudden increase in player count.
The end is sudden decrease in separation metric (see Classifier).
"""

import numpy as np
from scipy.ndimage import median_filter

from ..utils.constants import *


def find_plateaus(data, thres, above):
    """Find sections above or below ``thres`` for at least ``min_len`` contiguous.

    Args:
        data: ``ndarray float (N,)``.
        above: Whether to detect ``data > thres`` or ``data < thres``.

    Returns:
        ``ndarray bool (N,)``.
        True if a plateau *starts* on that index.
    """
    positive = (data >= thres) if above else (data <= thres)
    ret = np.zeros_like(data, dtype=bool)
    # Traverse backwards.
    count = 0
    for i in range(len(data) - 1, -1, -1):
        if positive[i]:
            count += 1
        else:
            count = 0
        if count >= CV_FPS * TRIM_PLATEAU:
            ret[i] = True
    return ret


def find_trim_sections(pipe_out):
    """Find all sections to trim from CV pipeline output.

    Args:
        pipe_out: List of CVPipeline outputs from each frame.

    Returns:
        ``ndarray float (N, 2)``.
        ``(start, end)`` timestamps to trim, in seconds.
    """
    # Extract data.
    counts = np.array([len(x["active_boxes"]) for x in pipe_out], dtype=float)
    seps = np.array([x["sep_metric"] for x in pipe_out], dtype=float)
    # Filtering.
    counts = median_filter(counts, size=TRIM_MED_FILTER)
    seps = median_filter(seps, size=TRIM_MED_FILTER)
    # Find plateaus.
    counts_high = find_plateaus(counts, TRIM_COUNT_HIGH, True)
    seps_high = find_plateaus(seps, TRIM_SEP_HIGH, True)
    seps_low = find_plateaus(seps, TRIM_SEP_LOW, False)

    sections = []
    i = 0
    while i < len(pipe_out):
        # Find next point end.
        if counts_high[i]:
            start = i

            # Find next point start.
            i += int(TRIM_MIN_STOP * CV_FPS)
            found_high = False
            for i in range(i, min(start + TRIM_MAX_STOP * CV_FPS, len(pipe_out))):
                if seps_high[i]:
                    found_high = True
                if seps_low[i] and found_high:
                    break

            sections.append((start, i))
            i += int(TRIM_MIN_PLAY * CV_FPS)

        else:
            i += 1

    sections = np.array(sections, dtype=np.float32) / CV_FPS
    if len(sections) == 0:
        return sections
    sections[:, 0] += TRIM_MARGIN_END
    sections[:, 1] -= TRIM_MARGIN_START
    return sections


def gen_timestamps(sections):
    """Generate timestamp string in YouTube format.

    Args:
        sections: ``ndarray float (N, 2)``.
        Sections that were trimmed, in seconds.
    """
    times = [0] + [x[1] for x in sections]
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

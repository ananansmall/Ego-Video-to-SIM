"""Validation for optional source-time metadata on a replay row grid."""

import numpy as np


def source_time_map(data, rows):
    """Original video frame coordinates, optionally fractional, for replay rows."""
    if 'source_time_frames' not in data:
        return None
    times = np.asarray(data['source_time_frames'], dtype=float)
    if (times.shape != np.shape(rows) or times.ndim != 1 or not len(times)
            or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('source_time_frames must give one strictly increasing source frame coordinate per replay row')
    return times

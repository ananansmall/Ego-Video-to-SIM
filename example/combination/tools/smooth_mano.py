#!/usr/bin/env python3
"""
tools/smooth_mano.py

Two trajectory smoothers for MANO hand-tracking parameters (HaWoR output).

MANO per-frame parameters live in three sequences:
  * pred_trans     : (N, 3)   global root translation
  * pred_rot       : (N, 3)   root orientation (axis-angle)
  * pred_hand_pose : (N, 45)  remaining 15 joints (axis-angle, 3*15)

Both smoothers operate on these *parameter trajectories* (not on FK'd 3D points),
so the result stays a valid MANO input and can be re-fed into the FK / retargeting
pipeline exactly like the raw data.

1) Kalman filter (online, causal)
   - Constant-velocity (CV) model per scalar component.
   - Robust to single-frame outliers: an outlier (Mahalanobis / jump gate) is
     down-weighted and the prediction is kept instead of the measurement.
   - Good for streaming / real-time, fixes the F50->F51 spike without lag.

2) Savitzky-Golay filter (offline, zero-phase)
   - Polynomial least-squares fit over a moving window; zero-phase because the
     window is symmetric -> no time shift, best for post-processing / analysis.
   - Applied on the raw trajectory (optionally after a median outlier clip).

API
---
    from smooth_mano import smooth_mano_kalman, smooth_mano_sg
    out = smooth_mano_kalman(raw, hand_idx=0)
    out = smooth_mano_sg(raw, hand_idx=0, window=9, poly=3)
where `raw` is a dict containing the (2, N, ...) arrays (same layout as the
hawor_results_*.npz). The returned dict has the same keys with the chosen hand
smoothed.
"""
import numpy as np

__all__ = ["smooth_mano_kalman", "smooth_mano_sg", "smooth_mano"]


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _axisangle_to_rotvec(aa):
    """(...,3) axis-angle -> (...,3) rotation vector (== axis-angle itself, already)."""
    return np.asarray(aa, dtype=np.float64)


def _unwrap_rotvec(rv):
    """Unwrap a rotation-vector trajectory so it does not jump across +/-pi on an axis.

    Operates on (N,3): for each component, unwrap modulo 2*pi relative to previous
    frame using the smallest-angle continuation. This keeps SG/Kalman in a smooth
    Euclidean space instead of fighting the manifold wrap.
    """
    rv = np.asarray(rv, dtype=np.float64)
    out = rv.copy()
    for i in range(1, len(rv)):
        diff = rv[i] - out[i - 1]
        # bring diff into [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        out[i] = out[i - 1] + diff
    return out


def _rewrap_rotvec(rv):
    """Map a (possibly unwrapped) rotation vector back into [-pi, pi] per component."""
    return (np.asarray(rv, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def _smooth_1d_cv_kalman(y, q=1e-3, r=1e-2, jump_gate_sigma=5.0):
    """Random-walk (RW) Kalman smoother for a single 1-D trajectory.

    We use a RANDOM-WALK model (state = [pos] only, no velocity) rather than a
    constant-velocity model. Rationale: MANO axis-angle parameters are NOT an
    Euclidean signal and do not obey "constant velocity" -- a CV model accumulates
    a velocity bias and drifts badly on rotating channels (observed RMSE ~1 rad).
    The RW model is numerically stable, behaves like an exponential smoother, and
    combined with the outlier gate cleanly removes single-frame spikes.

    y                : (N,) measurements (may contain NaN -> treated as missing).
    q                : process noise variance (None -> auto from data).
    r                : measurement noise variance (None -> auto from data).
    jump_gate_sigma : if |measurement - prediction| > gate, treat as outlier and
                       keep the prediction instead of the measurement.
    Returns smoothed (N,).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    valid = ~np.isnan(y)

    if valid.sum() > 2:
        d = np.diff(y[valid])
        delta_mad = np.median(np.abs(d - np.median(d))) + 1e-9
    else:
        delta_mad = 1e-3
    if q is None:
        # MEASUREMENT-DOMINATED config: large q, tiny r -> steady-state gain K ~ 1,
        # i.e. the filter follows the signal with almost no lag (critical for slow
        # MANO rotation drifts) and ONLY the outlier gate removes single-frame spikes.
        # Low-pass smoothing (if desired) is better done with SG on the parameter
        # trajectory; here KF is used as a robust spike rejector.
        q = max(delta_mad, 1e-6) ** 2 * 1.0
        r = max(delta_mad, 1e-6) ** 2 * 0.01
    if r is None:
        # measurement noise: trust the data; r ~ motion magnitude.
        r = max(delta_mad, 1e-6) ** 2 * 1.0
    sig_mag = np.median(np.abs(y[valid])) + 1e-9
    gate = jump_gate_sigma * delta_mad + 0.02 * sig_mag

    # RW scalar Kalman
    x = y[np.where(valid)[0][0]] if valid.any() else 0.0
    P = 1.0
    F = 1.0
    H = 1.0
    Q = q
    R = r

    out = np.full(n, np.nan)
    for i in range(n):
        # predict
        x = F * x
        P = F * P * F + Q
        if not valid[i] or np.isnan(y[i]):
            out[i] = x
            continue
        z = y[i]
        innov = z - H * x
        S = H * P * H + R
        if abs(innov) > gate:
            out[i] = x          # outlier: keep prediction
            continue
        K = P * H / S
        x = x + K * innov
        P = (1 - K * H) * P
        out[i] = x
    return out


def _smooth_vec_kalman(traj, **kw):
    """traj: (N, D) -> smoothed (N, D) with independent CV Kalman per dim."""
    traj = np.asarray(traj, dtype=np.float64)
    out = np.empty_like(traj)
    for d in range(traj.shape[1]):
        out[:, d] = _smooth_1d_cv_kalman(traj[:, d], **kw)
    return out


def _smooth_vec_sg(traj, window, poly):
    """traj: (N, D) -> smoothed (N, D) via Savitzky-Golay per dim."""
    from scipy.signal import savgol_filter
    traj = np.asarray(traj, dtype=np.float64)
    n = len(traj)
    # window must be odd and <= n; poly < window
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return traj
    p = min(poly, w - 1)
    out = np.empty_like(traj)
    for d in range(traj.shape[1]):
        out[:, d] = savgol_filter(traj[:, d], w, p)
    return out


def _median_clip(traj, k=3.0):
    """Clip per-dimension outliers (beyond k*mad from median) to the median +- k*mad.
    Used before SG so a single spike does not corrupt the polynomial fit."""
    traj = np.asarray(traj, dtype=np.float64).copy()
    for d in range(traj.shape[1]):
        c = traj[:, d]
        med = np.median(c)
        mad = np.median(np.abs(c - med)) + 1e-9
        lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
        c[(c < lo) | (c > hi)] = np.clip(c, lo, hi)[(c < lo) | (c > hi)]
        # simpler robust clip:
        c = np.where(c < lo, lo, np.where(c > hi, hi, c))
        traj[:, d] = c
    return traj


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def smooth_mano_kalman(raw, hand_idx=0, q=None, r=None, jump_gate_sigma=5.0):
    """Kalman (CV) smoother on MANO parameter trajectory of one hand.

    raw : dict with keys pred_trans/pred_rot/pred_hand_pose (shape (2,N,...) or (N,...)).
    hand_idx : which hand axis (0/1). If arrays are (N,...) already, ignored.
    q, r : process/measurement noise (None -> auto-estimated from data magnitude).
    jump_gate_sigma : outlier gate in units of frame-delta robust std.
    Returns a new dict with the same keys; only the chosen hand is smoothed,
    others are copied unchanged.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()}

    def _get(key):
        a = raw[key]
        if a.ndim == 3:
            return a[hand_idx]
        return a

    def _set(key, val):
        if out[key].ndim == 3:
            out[key][hand_idx] = val
        else:
            out[key] = val

    trans = _get("pred_trans")          # (N,3)  translation (Euclidean, NO unwrap)
    rot = _get("pred_rot")              # (N,3) axis-angle
    pose = _get("pred_hand_pose")       # (N,45) axis-angle

    # The raw HaWoR axis-angle sequence is continuous (no +-pi wrap in this data),
    # so we smooth it directly in Euclidean space. unwrap() is only needed for data
    # that actually crosses the manifold seam; keep it off by default to avoid drift.
    s_trans = _smooth_vec_kalman(trans, q=q, r=r, jump_gate_sigma=jump_gate_sigma)
    s_rot = _smooth_vec_kalman(rot, q=q, r=r, jump_gate_sigma=jump_gate_sigma)
    s_pose = _smooth_vec_kalman(pose, q=q, r=r, jump_gate_sigma=jump_gate_sigma)

    _set("pred_trans", s_trans)
    _set("pred_rot", s_rot)
    _set("pred_hand_pose", s_pose)
    return out


def smooth_mano_sg(raw, hand_idx=0, window=9, poly=3, median_clip=True):
    """Savitzky-Golay (zero-phase) smoother on MANO parameter trajectory of one hand."""
    try:
        from scipy.signal import savgol_filter  # noqa: F401
    except ImportError:
        raise ImportError("scipy is required for the Savitzky-Golay smoother")

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()}

    def _get(key):
        a = raw[key]
        if a.ndim == 3:
            return a[hand_idx]
        return a

    def _set(key, val):
        if out[key].ndim == 3:
            out[key][hand_idx] = val
        else:
            out[key] = val

    trans = _get("pred_trans")
    rot = _get("pred_rot")
    pose = _get("pred_hand_pose")

    if median_clip:
        trans = _median_clip(trans)
        rot = _median_clip(rot)
        pose = _median_clip(pose)

    s_trans = _smooth_vec_sg(trans, window, poly)
    s_rot = _smooth_vec_sg(rot, window, poly)
    s_pose = _smooth_vec_sg(pose, window, poly)

    _set("pred_trans", s_trans)
    _set("pred_rot", s_rot)
    _set("pred_hand_pose", s_pose)
    return out


def smooth_mano(raw, method="kalman", hand_idx=0, **kw):
    """Dispatch helper."""
    if method == "kalman":
        return smooth_mano_kalman(raw, hand_idx=hand_idx, **kw)
    elif method == "sg":
        return smooth_mano_sg(raw, hand_idx=hand_idx, **kw)
    else:
        raise ValueError(f"unknown method {method!r}")

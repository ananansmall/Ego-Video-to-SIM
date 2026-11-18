"""Pixel-UMI equation (3) on saved R1 command/state traces, free-space only."""
from pathlib import Path
import json
import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
CONTROLS = EXPERIMENT.parent / 'gripper-hoi-20261118-controls'
LAGS = np.arange(39) * .004
MIN_ACTIVITY_RAD_S = .05

def free_intervals(trace):
    free = ((trace['bilateral_substeps'] == 0) &
            (trace['support_collision_substeps'] == 0))
    active = np.flatnonzero(trace['bilateral_substeps'] > 0)
    if not len(active):
        raise ValueError('Expected a grasp separating the approach and withdrawal.')
    f = np.arange(len(free))
    return free & (f < active[0]), free & (f > active[-1])

def probe(path):
    with np.load(path, allow_pickle=False) as z:
        t = z['sim_time_s']
        cmd = z['arm_joint_targets']
        obs = z['q_actual'][:, z['arm_joint_indices']]
        frames = z['source_frames']
        train, test = free_intervals(z)
        indices = z['arm_joint_indices']
    # Interpolation is applied to saved frame-rate states, not to physics substeps.
    stop = np.minimum(np.searchsorted(t, t + LAGS[-1]), len(t)-1)
    moving = np.abs(np.gradient(cmd, t, axis=0)) >= MIN_ACTIVITY_RAD_S
    result = []
    for j in range(cmd.shape[1]):
        valid_fit = [i for i in np.flatnonzero(train & moving[:, j]) if train[i:stop[i]+1].all() and t[i]+LAGS[-1] <= t[-1]]
        valid_eval = [i for i in np.flatnonzero(test & moving[:, j]) if test[i:stop[i]+1].all() and t[i]+LAGS[-1] <= t[-1]]
        if len(valid_fit) < 6 or len(valid_eval) < 6:
            continue
        def curves(rows):
            shifted = np.interp(t[rows, None] + LAGS, t, obs[:, j])
            return np.sqrt(np.mean((shifted-cmd[rows,j,None])**2, axis=0))
        fit = curves(valid_fit)
        test_curve = curves(valid_eval)
        best = int(np.argmin(fit))
        result.append(dict(joint_index=int(indices[j]), arm_or_torso='arm' if j<6 else 'torso',
                           fit_points=len(valid_fit), heldout_points=len(valid_eval),
                           fit_source_range=frames[valid_fit][[0,-1]].tolist(),
                           heldout_source_range=frames[valid_eval][[0,-1]].tolist(),
                           fitted_delay_ms=float(1000*LAGS[best]),
                           heldout_zero_delay_rmse_deg=float(np.degrees(test_curve[0])),
                           heldout_calibrated_rmse_deg=float(np.degrees(test_curve[best]))))
    return dict(trace=str(path), evaluated_joints=result)

data = dict(scope='Offline free-space alignment probe on cached SAPIEN state/command pairs. Fit pre-grasp, assess post-release; no new labels, control, training or physical rerun. Grid resolution does not imply measurement resolution.',
            max_lag_s=float(LAGS[-1]), lag_grid_step_s=.004, minimum_activity_rad_s=MIN_ACTIVITY_RAD_S,
            cases={c:probe(CONTROLS/c/'multicontact_trace.npz') for c in ('456','ep4')})
(EXPERIMENT/'delay_probe.json').write_text(json.dumps(data,indent=2)+'\n')
for case, result in data['cases'].items():
    print(case, len(result['evaluated_joints']), 'eligible', '-> heldout mean joint RMSE deg', [
        round(float(np.mean([j[k] for j in result['evaluated_joints']])), 4)
        for k in ('heldout_zero_delay_rmse_deg','heldout_calibrated_rmse_deg')
    ] if result['evaluated_joints'] else None)


def cross_episode(calibration, holdout):
    delays = {j['joint_index']: j['fitted_delay_ms']/1000
              for j in data['cases'][calibration]['evaluated_joints']}
    eligible = {j['joint_index'] for j in data['cases'][holdout]['evaluated_joints']}
    with np.load(CONTROLS/holdout/'multicontact_trace.npz', allow_pickle=False) as z:
        t = z['sim_time_s']
        command = z['arm_joint_targets']
        indices = z['arm_joint_indices']
        measured = z['q_actual'][:, indices]
        _, free = free_intervals(z)
    stop = np.minimum(np.searchsorted(t, t + LAGS[-1]), len(t)-1)
    moving = np.abs(np.gradient(command, t, axis=0)) >= MIN_ACTIVITY_RAD_S
    joints = []
    for j, index in enumerate(indices):
        if index not in delays or index not in eligible:
            continue
        rows = [i for i in np.flatnonzero(moving[:, j] & free)
                if free[i:stop[i]+1].all() and t[i]+LAGS[-1] <= t[-1]]
        residual = measured[rows,j] - command[rows,j]
        aligned = np.interp(t[rows] + delays[index], t, measured[:,j]) - command[rows,j]
        raw_rmse, aligned_rmse = np.degrees(np.sqrt(np.mean([residual**2, aligned**2], axis=1)))
        joints.append(dict(joint_index=int(index), heldout_points=len(rows), delay_ms=1000*delays[index],
                           zero_delay_rmse_deg=float(raw_rmse), aligned_rmse_deg=float(aligned_rmse)))
    return dict(calibration=calibration, holdout=holdout, joints=joints,
                mean_zero_delay_rmse_deg=float(np.mean([j['zero_delay_rmse_deg'] for j in joints])),
                mean_aligned_rmse_deg=float(np.mean([j['aligned_rmse_deg'] for j in joints])))

data['cross_episode'] = [cross_episode('456','ep4'), cross_episode('ep4','456')]
(EXPERIMENT/'delay_probe.json').write_text(json.dumps(data,indent=2)+'\n')
for res in data['cross_episode']:
    print(res['calibration'], 'to', res['holdout'], len(res['joints']),
          res['mean_zero_delay_rmse_deg'], '->', res['mean_aligned_rmse_deg'])

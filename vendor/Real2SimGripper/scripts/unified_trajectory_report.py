"""Unified Stage 0 / pair / IK / physics trajectory error report.

Writes trajectory_errors.csv (per source frame) and metrics.json (per-stage and
per-segment aggregates with valid-frame counts) for one run.  Errors follow the
project convention: pad translation in mm, orientation as the SO(3) geodesic
angle in degrees.  Same-source errors compare the physics measurement with the
frozen Stage 0 trace; command-axis errors compare it with the commands the
controller actually issued.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_timing import source_time_map

ORI = Rotation


def ori_deg(R_a, R_b):
    rel = np.einsum('nij,nkj->nik', R_a, R_b)
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def read_pair(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def stats(values, mask=None):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if mask is not None:
        finite &= np.asarray(mask, bool)
    if not finite.any():
        return {"valid_frames": 0}
    v = values[finite]
    return {"valid_frames": int(finite.sum()), "median": float(np.median(v)),
            "p95": float(np.quantile(v, .95)), "max": float(v.max())}


def pair_pose_errors(a, b):
    """(position mm, orientation deg) between two (N,4,4) pose arrays."""
    d = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) * 1000
    o = ori_deg(a[:, :3, :3], b[:, :3, :3])
    return d, o


def block(name, frames, values, extra=None):
    out = {"scope": name, **stats(values), "frames": int(len(frames))}
    if extra:
        out.update(extra)
    return out


def matched_rows(reference, measured, label, allow_fractional=False):
    indices = np.searchsorted(reference, measured - 1e-8)
    in_range = indices < len(reference)
    valid = np.zeros(len(measured), bool)
    valid[in_range] = np.isclose(reference[indices[in_range]], measured[in_range], rtol=0, atol=1e-8)
    if not valid.any() or (not allow_fractional and not valid.all()):
        raise ValueError(f'{label} frames do not map onto the reference timeline')
    return indices[valid], np.flatnonzero(valid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage0-trace', type=Path, required=True)
    ap.add_argument('--stage0-csv', type=Path, help='stage0_frame_errors.csv from the Stage 0 render')
    ap.add_argument('--pair', type=Path, help='optimized paired_targets.npz')
    ap.add_argument('--ik', type=Path, help='ik_plan.npz')
    ap.add_argument('--physics', type=Path, help='directory with exec_trace.npz (+ verdict.json)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.stage0_trace, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    frames = np.asarray(trace['src_frames'])
    n = len(frames)
    metrics = {"scope": "unified trajectory report", "stage0_trace": str(args.stage0_trace.resolve()),
               "source_frames": [int(frames[0]), int(frames[-1])], "rows": n,
               "sampled_source_frames": bool(np.any(np.diff(frames) != 1))}
    columns = {'source_frame': frames.astype(int),
               'playback_t_s': frames.astype(float) / float(trace['video_fps'])}
    if 'grip_states' in trace:
        names = [str(s) for s in trace.get('grip_state_names', [str(i) for i in range(6)])]
        columns['segment'] = [names[int(s)] for s in trace['grip_states']]
        segments = np.asarray(trace['grip_states'], int)
    else:
        columns['segment'] = ['unsegmented'] * n
        segments = np.zeros(n, int)
    segment_names = sorted(set(columns['segment']), key=columns['segment'].index)

    for key in ('hand_observed', 'hand_filled', 'object_observed', 'object_filled', 'object_unavailable'):
        if key in trace:
            columns[key] = np.asarray(trace[key], bool)
    metrics['validity'] = {key: int(np.asarray(trace[key], bool).sum()) for key in
                           ('hand_observed', 'hand_filled', 'object_observed', 'object_filled',
                            'object_unavailable') if key in trace}

    stage0_pad = np.asarray(trace['T_world_pad'], float)
    stage0_obj = np.asarray(trace['T_world_object'], float)
    stage0_base = np.asarray(trace['base_se2'], float)
    phys = read_pair(args.physics / 'exec_trace.npz') if args.physics else None
    phys_time_map = source_time_map(phys, phys['source_frames']) if phys is not None else None
    plan_grid, plan_time_map = None, None
    if phys_time_map is not None:
        plan_grid, plan_time_map = phys['source_frames'], phys_time_map
        metrics['retiming'] = {'replay_rows': len(phys_time_map),
                               'source_comparison': 'exact original frame matches only'}

    # Stage 0 kinematic FK against the frozen original retarget (from the render).
    if args.stage0_csv and args.stage0_csv.is_file():
        import csv as _csv
        with args.stage0_csv.open() as stream:
            rows = list(_csv.DictReader(stream))
        csv_frames = np.asarray([int(r['source_frame']) for r in rows])
        idx = np.searchsorted(frames, csv_frames)
        if not np.array_equal(frames[idx], csv_frames):
            raise ValueError('stage0 CSV frames do not map onto the trace timeline')
        for col, out_col in (('pad_position_error_mm', 'stage0_fk_pos_mm'),
                             ('pad_orientation_error_deg', 'stage0_fk_ori_deg')):
            columns[out_col] = np.full(n, np.nan)
            columns[out_col][idx] = [float(r[col]) for r in rows]
        seg_labels = np.asarray(columns['segment'])
        metrics['stage0_fk_vs_retarget'] = {}
        for seg in segment_names:
            mask = seg_labels == seg
            metrics['stage0_fk_vs_retarget'][seg] = block(
                f'stage0_fk/{seg}', frames[mask], columns['stage0_fk_pos_mm'][mask],
                {'orientation': stats(columns['stage0_fk_ori_deg'], mask)})
        metrics['stage0_fk_vs_retarget']['all'] = block('stage0_fk/all', frames, columns['stage0_fk_pos_mm'],
                                                       {'orientation': stats(columns['stage0_fk_ori_deg'])})

    if args.pair:
        pair = read_pair(args.pair)
        pair_frames = np.asarray(pair['src_frames'])
        pair_time_map = source_time_map(pair, pair_frames)
        if pair_time_map is not None:
            if phys_time_map is not None and (not np.array_equal(pair_frames, plan_grid)
                                               or not np.array_equal(pair_time_map, phys_time_map)):
                raise ValueError('pair and physics do not share the same replay clock')
            plan_grid, plan_time_map = pair_frames, pair_time_map
            source_idx, pair_rows = matched_rows(frames, pair_time_map, 'pair', allow_fractional=True)
            if not np.array_equal(source_idx, np.arange(n)):
                raise ValueError('retimed pair must cover all exact Stage 0 source frames')
        else:
            if not np.array_equal(pair_frames, frames):
                raise ValueError('pair source frames differ from the Stage 0 trace timeline')
            pair_rows = slice(None)
        pair_pad = np.asarray(pair['T_world_pad'], float)[pair_rows]
        d, o = pair_pose_errors(pair_pad, stage0_pad)
        columns['pair_vs_stage0_pos_mm'], columns['pair_vs_stage0_ori_deg'] = d, o
        metrics['optimized_pair_vs_stage0'] = block('pair_vs_stage0/all', frames, d,
                                                    {'orientation': stats(o)})
        if 'T_world_pad_retarget' in pair:
            d, o = pair_pose_errors(pair_pad, np.asarray(pair['T_world_pad_retarget'], float)[pair_rows])
            columns['pair_vs_retarget_pos_mm'], columns['pair_vs_retarget_ori_deg'] = d, o
            metrics['optimized_pair_vs_original_retarget'] = block('pair_vs_retarget/all', frames, d,
                                                                   {'orientation': stats(o)})

    if args.ik and args.ik.is_file():
        ik = read_pair(args.ik)
        ik_frames = np.asarray(ik['source_frames'])
        if plan_time_map is not None:
            grid_idx, _ = matched_rows(plan_grid, ik_frames, 'IK replay')
            ik_source_frames = plan_time_map[grid_idx]
        else:
            ik_source_frames = ik_frames
        idx, ik_rows = matched_rows(frames, ik_source_frames, 'IK', plan_time_map is not None)
        for col, out_col in (('position_error_mm', 'ik_pos_mm'), ('orientation_error_deg', 'ik_ori_deg')):
            columns[out_col] = np.full(n, np.nan)
            columns[out_col][idx] = np.asarray(ik[col], float)[ik_rows]
        metrics['ik_fk_vs_optimized_pair'] = block('ik/all', frames, columns['ik_pos_mm'],
                                                   {'orientation': stats(columns['ik_ori_deg'])})
        if 'base_se2' in ik:
            ik_base = np.asarray(ik['base_se2'], float)[ik_rows]
            if len(ik_base) == n:
                dxy = np.linalg.norm(ik_base[:, :2] - stage0_base[:, :2], axis=1)
                dyaw = np.abs((ik_base[:, 2] - stage0_base[:, 2] + np.pi) % (2 * np.pi) - np.pi)
                columns['base_plan_vs_stage0_xy_m'], columns['base_plan_vs_stage0_yaw_deg'] = dxy, dyaw
                metrics['base_plan_vs_stage0_reference'] = block('base_plan_vs_stage0/all', frames, dxy,
                                                                 {'yaw': stats(dyaw)})

    if phys is not None:
        phys_frames = np.asarray(phys['source_frames'])
        phys_source_frames = phys_frames if phys_time_map is None else phys_time_map
        idx, phys_rows = matched_rows(frames, phys_source_frames, 'physics', phys_time_map is not None)
        if phys_time_map is not None:
            columns['replay_frame'] = np.full(n, None, dtype=object)
            columns['replay_frame'][idx] = phys_frames[phys_rows]
        if 'command_frames' in phys:
            command_indices = np.asarray(phys['command_frames'])
            if (command_indices.shape != phys_frames.shape or command_indices.dtype.kind not in 'iu'
                    or np.any(command_indices < 0) or np.any(command_indices >= len(phys_frames))):
                raise ValueError('physics command frames must index the source frames in the same trace')
            columns['command_source_frame'] = np.full(n, None, dtype=object)
            columns['command_source_frame'][idx] = phys_source_frames[command_indices[phys_rows]]
        if 'physics_substeps' in phys and 'dt' in phys:
            columns['sim_time_s'] = np.full(n, np.nan)
            world_times = np.cumsum(phys['physics_substeps']) * float(phys['dt'])
            columns['sim_time_s'][idx] = world_times[phys_rows]
        target_pad = np.asarray(phys['target_pad'], float)[phys_rows]
        actual_pad = np.asarray(phys['actual_pad'], float)[phys_rows]
        d, o = pair_pose_errors(actual_pad, target_pad)
        columns['phys_cmd_pos_mm'], columns['phys_cmd_ori_deg'] = np.full(n, np.nan), np.full(n, np.nan)
        columns['phys_cmd_pos_mm'][idx], columns['phys_cmd_ori_deg'][idx] = d, o
        metrics['physics_vs_command_axis'] = block('phys_cmd/all', frames, columns['phys_cmd_pos_mm'],
                                                   {'orientation': stats(columns['phys_cmd_ori_deg'])})
        d, o = pair_pose_errors(actual_pad, stage0_pad[idx])
        columns['phys_vs_stage0_pos_mm'], columns['phys_vs_stage0_ori_deg'] = np.full(n, np.nan), np.full(n, np.nan)
        columns['phys_vs_stage0_pos_mm'][idx], columns['phys_vs_stage0_ori_deg'][idx] = d, o
        metrics['physics_vs_stage0_same_source'] = block('phys_vs_stage0/all', frames,
                                                         columns['phys_vs_stage0_pos_mm'],
                                                         {'orientation': stats(columns['phys_vs_stage0_ori_deg'])})
        if args.pair and 'T_world_pad' in pair:
            d, o = pair_pose_errors(actual_pad, pair_pad[idx])
            columns['phys_vs_pair_pos_mm'], columns['phys_vs_pair_ori_deg'] = np.full(n, np.nan), np.full(n, np.nan)
            columns['phys_vs_pair_pos_mm'][idx], columns['phys_vs_pair_ori_deg'][idx] = d, o
            metrics['physics_vs_optimized_pair'] = block('phys_vs_pair/all', frames,
                                                         columns['phys_vs_pair_pos_mm'],
                                                         {'orientation': stats(columns['phys_vs_pair_ori_deg'])})
        obj_t = np.asarray(phys['target_object'], float)[phys_rows] if 'target_object' in phys else None
        obj_a = np.asarray(phys['actual_object'], float)[phys_rows] if 'actual_object' in phys else None
        if obj_t is not None and obj_a is not None:
            d, o = pair_pose_errors(obj_a, obj_t)
            columns['obj_phys_cmd_pos_mm'], columns['obj_phys_cmd_ori_deg'] = np.full(n, np.nan), np.full(n, np.nan)
            columns['obj_phys_cmd_pos_mm'][idx], columns['obj_phys_cmd_ori_deg'][idx] = d, o
            metrics['object_physics_vs_command'] = block('obj_cmd/all', frames, columns['obj_phys_cmd_pos_mm'],
                                                         {'orientation': stats(columns['obj_phys_cmd_ori_deg'])})
            unavailable = np.asarray(trace.get('object_unavailable', np.zeros(n, bool)), bool)
            visible_rows = np.asarray(trace.get('object_visible', ~unavailable), bool)[idx]
            d, o = pair_pose_errors(obj_a[visible_rows], stage0_obj[idx[visible_rows]])
            columns['obj_phys_vs_stage0_pos_mm'] = np.full(n, np.nan)
            columns['obj_phys_vs_stage0_ori_deg'] = np.full(n, np.nan)
            columns['obj_phys_vs_stage0_pos_mm'][idx[visible_rows]] = d
            columns['obj_phys_vs_stage0_ori_deg'][idx[visible_rows]] = o
            observed = np.asarray(trace.get('object_observed', np.zeros(n, bool)), bool)
            metrics['object_physics_vs_stage0'] = block(
                'obj_stage0/all', frames, columns['obj_phys_vs_stage0_pos_mm'],
                {'orientation': stats(columns['obj_phys_vs_stage0_ori_deg']),
                 'observed_only': {'position': stats(columns['obj_phys_vs_stage0_pos_mm'], observed),
                                   'orientation': stats(columns['obj_phys_vs_stage0_ori_deg'], observed)}})
        base_act = np.asarray(phys['base_se2'], float)[phys_rows] if 'base_se2' in phys else None
        base_tgt = np.asarray(phys['base_se2_target'], float)[phys_rows] if 'base_se2_target' in phys else None
        if base_act is not None and base_tgt is not None and len(base_act) == n:
            dxy = np.linalg.norm(base_act[:, :2] - base_tgt[:, :2], axis=1)
            dyaw = np.abs((base_act[:, 2] - base_tgt[:, 2] + np.pi) % (2 * np.pi) - np.pi)
            columns['base_phys_vs_plan_xy_m'], columns['base_phys_vs_plan_yaw_deg'] = dxy, dyaw
            metrics['base_physics_vs_plan'] = block('base_phys_vs_plan/all', frames, dxy,
                                                    {'yaw': stats(dyaw)})

    # Per-segment physics aggregates (approach / pick / lift / hold / place / released).
    if 'phys_vs_stage0_pos_mm' in columns:
        metrics['physics_vs_stage0_by_segment'] = {}
        for seg in segment_names:
            mask = np.asarray(columns['segment']) == seg
            metrics['physics_vs_stage0_by_segment'][seg] = {
                'position': stats(columns['phys_vs_stage0_pos_mm'], mask),
                'orientation': stats(columns['phys_vs_stage0_ori_deg'], mask)}
    if 'phys_cmd_pos_mm' in columns:
        metrics['physics_vs_command_by_segment'] = {}
        for seg in segment_names:
            mask = np.asarray(columns['segment']) == seg
            metrics['physics_vs_command_by_segment'][seg] = {
                'position': stats(columns['phys_cmd_pos_mm'], mask),
                'orientation': stats(columns['phys_cmd_ori_deg'], mask)}

    # Deviation attribution summary (which module the large deviations start at).
    chain = ['stage0_fk_vs_retarget', 'optimized_pair_vs_stage0', 'ik_fk_vs_optimized_pair',
             'physics_vs_command_axis']
    metrics['deviation_chain'] = {
        name: (metrics[name]['all'].get('p95') if 'all' in metrics[name] else metrics[name].get('p95'))
        for name in chain if name in metrics}

    args.out.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with (args.out / 'trajectory_errors.csv').open('w', newline='') as stream:
        writer = _csv.writer(stream)
        writer.writerow(list(columns))
        for i in range(n):
            row = []
            for key in columns:
                v = columns[key][i]
                if isinstance(v, (bool, np.bool_)):
                    v = int(v)
                row.append(v)
            writer.writerow(row)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({k: metrics[k] for k in ('deviation_chain', 'validity') if k in metrics}, indent=2))


if __name__ == '__main__':
    main()

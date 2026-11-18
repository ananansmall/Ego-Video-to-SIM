"""Apply registered, robot-retargeted AGILE corrections to a frozen pair."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copy2

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rigid(poses, count, label):
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f'{label}: expected {count} finite 4×4 poses')
    rotations = poses[:, :3, :3]
    if not (np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError(f'{label}: invalid SE(3) pose')


def interpolate_pose(old, new, weight):
    result = old.copy()
    relative = Rotation.from_matrix(old[:, :3, :3].transpose(0, 2, 1) @ new[:, :3, :3])
    result[:, :3, :3] = old[:, :3, :3] @ Rotation.from_rotvec(weight[:, None] * relative.as_rotvec()).as_matrix()
    result[:, :3, 3] += weight[:, None] * (new[:, :3, 3] - old[:, :3, 3])
    return result


def merge(baseline, windows, *, allow_sampled_source=False, freeze_object=False):
    frames = baseline['src_frames']
    count = len(frames)
    if frames.ndim != 1 or not count:
        raise ValueError('Baseline needs a source-frame axis')
    differences = np.diff(frames)
    if allow_sampled_source:
        if frames.dtype.kind not in 'iu' or np.any(frames[1:] <= frames[:-1]):
            raise ValueError('Sampled source frames must be strictly increasing integers')
    elif not np.all(differences == 1):
        raise ValueError('Baseline must use one contiguous source-frame axis')
    pad = baseline['T_world_pad_retarget']
    grip = baseline['grip_retarget']
    obj = baseline['T_world_object']
    rigid(pad, count, 'baseline retarget')
    rigid(obj, count, 'baseline object')
    if ('T_world_pad' in baseline and not np.array_equal(baseline['T_world_pad'], pad)
            or 'grip' in baseline and not np.array_equal(baseline['grip'], grip)):
        raise ValueError('The input pair must contain the original retarget commands')
    if grip.shape != (count, 2) or not np.isfinite(grip).all():
        raise ValueError('Baseline must have two calibrated gripper commands per frame')
    object_ids = baseline['object_ids']
    if object_ids.shape != (count,) or object_ids.dtype.kind != 'U':
        raise ValueError('Baseline must have a string object ID on the source-frame axis')
    weight = np.zeros(count)
    occupied = np.zeros(count, dtype=bool)
    out_pad, out_grip, out_obj = pad.copy(), grip.copy(), obj.copy()
    summary = []
    for entry, correction in windows:
        first, last = entry['first_frame'], entry['last_frame']
        selected = np.flatnonzero((frames >= first) & (frames <= last))
        if len(selected) < 3 or frames[selected[0]] != first or frames[selected[-1]] != last:
            raise ValueError(f"{entry['window_id']}: invalid source-frame interval")
        if occupied[selected].any() or not np.all(object_ids[selected] == entry['object_id']):
            raise ValueError(f"{entry['window_id']}: overlapping patch or object identity mismatch")
        occupied[selected] = True
        if not np.array_equal(correction['source_frames'], frames[selected]):
            raise ValueError(f"{entry['window_id']}: correction does not map exactly to source frames")
        if str(correction['object_id']) != entry['object_id']:
            raise ValueError(f"{entry['window_id']}: correction object identity does not match")
        evidence = np.asarray(entry['contact_evidence_frames'], dtype=int)
        if not len(evidence) or not np.isin(evidence, frames[selected]).all():
            raise ValueError(f"{entry['window_id']}: no in-range source-frame contact evidence")
        w = correction['weight']
        if (w.shape != (len(selected),) or not np.isfinite(w).all() or
                np.any((w < 0) | (w > 1)) or np.any(w[[0, -1]] != 0) or not np.any(w > 0)):
            raise ValueError(f"{entry['window_id']}: weights need nonzero interior and zero boundary")
        for field in ('T_world_pad', 'T_world_object'):
            rigid(correction[field], len(selected), field)
        if freeze_object and not np.array_equal(correction['T_world_object'], obj[selected]):
            raise ValueError('The contact plan changes the frozen object trajectory')
        g = correction['grip']
        if g.shape != (len(selected), 2) or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError('AGILE correction needs two finite nonnegative jaw commands per frame')
        active = w > 0
        j = selected[active]
        out_pad[j] = interpolate_pose(pad[j], correction['T_world_pad'][active], w[active])
        if not freeze_object:
            out_obj[j] = interpolate_pose(obj[j], correction['T_world_object'][active], w[active])
        out_grip[j] = grip[j] + w[active, None] * (g[active] - grip[j])
        weight[selected] = w
        changed = (np.any(out_pad[j] != pad[j], axis=(1, 2)) | np.any(out_grip[j] != grip[j], axis=1)
                   | np.any(out_obj[j] != obj[j], axis=(1, 2)))
        if not changed.any():
            raise ValueError(f"{entry['window_id']}: patch made no change to the pair")
        summary.append(dict(window_id=entry['window_id'], object_id=entry['object_id'],
                            source_frames=[first, last], changed_source_frames=frames[j][changed].tolist(),
                            contact_evidence_frames=evidence.tolist(),
                            native_sha256=entry.get('native_sha256'),
                            correction_sha256=entry.get('correction_sha256')))
    active = weight > 0
    original_pad_object = np.linalg.solve(pad, obj)
    pad_object = baseline.get('T_pad_object', original_pad_object).copy()
    if not np.allclose(pad_object, original_pad_object, atol=1e-5):
        raise ValueError('The baseline pad-to-object transform disagrees with its world poses')
    pad_object[active] = np.linalg.solve(out_pad[active], out_obj[active])
    updated = dict(baseline, T_world_pad=out_pad, T_world_object=out_obj, grip=out_grip,
                   T_pad_object=pad_object, contact_blend=weight)
    if 'T_world_gripper' in baseline:
        gripper = baseline['T_world_gripper']
        rigid(gripper, count, 'baseline gripper')
        out_gripper = gripper.copy()
        out_gripper[active] = out_pad[active] @ np.linalg.solve(pad[active], gripper[active])
        original_gripper_object = np.linalg.solve(gripper, obj)
        gripper_object = baseline.get('T_gripper_object', original_gripper_object).copy()
        if not np.allclose(gripper_object, original_gripper_object, atol=1e-5):
            raise ValueError('The baseline gripper-to-object transform disagrees with its world poses')
        gripper_object[active] = np.linalg.solve(out_gripper[active], out_obj[active])
        updated['T_world_gripper'] = out_gripper
        updated['T_gripper_object'] = gripper_object
    return updated, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path, help='Directory containing an unrefined R1 pair')
    parser.add_argument('windows', type=Path, help='JSON window list with correction and native AGILE provenance')
    parser.add_argument('output', type=Path, help='New output directory')
    args = parser.parse_args()
    base_path = args.baseline / 'paired_targets.npz'
    metadata = json.loads(args.windows.read_text())
    sources = []
    for entry in metadata['windows']:
        correction_path = (args.windows.parent / entry['correction']).resolve()
        native_path = (args.windows.parent / entry['native_aligned']).resolve()
        files = [native_path / name for name in ('object_poses.npz', 'optimized_mano_joints.npz', 'camera_data.npz')]
        native = {f.name: digest(f) for f in files}
        if native != entry['native_sha256']:
            raise ValueError(f"{entry['window_id']}: native AGILE artifact hash mismatch")
        with np.load(correction_path, allow_pickle=False) as correction:
            sources.append((entry, dict(correction)))
        entry['correction_sha256'] = digest(correction_path)
    if not sources:
        raise ValueError('There are no verified AGILE windows to merge')
    with np.load(base_path, allow_pickle=False) as baseline:
        updated, summary = merge(dict(baseline), sources)
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.baseline.iterdir():
        if path.is_file() and path.name != base_path.name:
            copy2(path, args.output / path.name)
    output_path = args.output / base_path.name
    np.savez_compressed(output_path, **updated)
    manifest = dict(baseline_sha256=digest(base_path), pair_sha256=digest(output_path),
                    window_spec_sha256=digest(args.windows), windows=summary)
    (args.output / 'agile_pair_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

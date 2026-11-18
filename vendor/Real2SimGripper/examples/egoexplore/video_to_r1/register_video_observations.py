"""Register EgoFound3R observations to the gravity-aligned RAS world."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def camera_center(w2c):
    return -np.einsum('nji,nj->ni', w2c[:, :3, :3], w2c[:, :3, 3])


def fit_registration(ef_w2c, ras_w2c, train):
    ef_center, ras_center = camera_center(ef_w2c), camera_center(ras_w2c)
    ef_rotation, ras_rotation = ef_w2c[:, :3, :3], ras_w2c[:, :3, :3]
    relative = Rotation.from_matrix(ras_rotation.transpose(0, 2, 1) @ ef_rotation)
    initial = relative[train].mean()
    residual = (initial.inv() * relative[train]).magnitude()
    source = train[np.argsort(residual)[: max(3, (len(train) * 3 + 3) // 4)]]
    rotation = relative[source].mean()
    x = rotation.apply(ef_center[source])
    y = ras_center[source]
    xc, yc = x - x.mean(0), y - y.mean(0)
    spread = np.sum(xc**2)
    if spread <= 1e-6:
        raise ValueError('Same-source cameras have insufficient translation to estimate scale')
    scale = float(np.sum(xc * yc) / spread)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Same-source camera registration has no positive scale')
    offset = np.median(y - scale * x, axis=0)
    position_mm = np.linalg.norm(ras_center - (scale * rotation.apply(ef_center) + offset), axis=1)
    angle_deg = np.degrees((rotation.inv() * relative).magnitude())
    return rotation, scale, offset, position_mm, angle_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--ef3r', type=Path, required=True, help='EgoFound3R inference_output.pt for the source video')
    parser.add_argument('--ras-map', type=Path, required=True, help='RAS map NPZ with exact source_frames and raw w2c')
    parser.add_argument('--ras-alignment', type=Path, required=True, help='RAS coordinate_alignment.json for that map')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-heldout-mm', type=float, default=50.)
    parser.add_argument('--max-heldout-deg', type=float, default=8.)
    args = parser.parse_args()
    import torch

    ef = torch.load(args.ef3r, map_location='cpu', weights_only=True)
    if Path(ef['video_path']).resolve() != args.video.resolve():
        raise ValueError('EgoFound3R source path differs from the requested original video')
    ef_frames = np.asarray(ef['source_frame_indices'], dtype=int)
    ef_extrinsics = np.asarray(ef['camera_pose_metric_full'], dtype=float)
    ef_valid = np.asarray(ef['camera_pose_metric_valid_full'], dtype=bool)
    with np.load(args.ras_map, allow_pickle=False) as ras:
        ras_frames = np.asarray(ras['source_frames'], dtype=int)
        ras_w2c = np.asarray(ras['w2c'], dtype=float)[:, :3, :4].copy()
    gravity = json.loads(args.ras_alignment.read_text())
    gravity_R = np.asarray(gravity['R'], dtype=float)
    gravity_t = np.asarray(gravity['t'], dtype=float)
    if np.allclose(gravity_R, np.eye(3)) and np.allclose(gravity_t, 0):
        if gravity.get('method_detail', {}).get('method') not in ('floor+wall', 'floor_only'):
            raise ValueError('RAS does not provide a verified gravity alignment')
    ras_w2c[:, :3, :3] = ras_w2c[:, :3, :3] @ gravity_R.T
    ras_w2c[:, :3, 3] -= ras_w2c[:, :3, :3] @ gravity_t
    frames, ei, ri = np.intersect1d(ef_frames, ras_frames, return_indices=True)
    valid = ef_valid[ei] & np.isfinite(ef_extrinsics[ei]).all(axis=(1, 2))
    valid &= np.isfinite(ras_w2c[ri]).all(axis=(1, 2))
    frames, ei, ri = frames[valid], ei[valid], ri[valid]
    if len(frames) < 8:
        raise ValueError(f'Only {len(frames)} same-source valid cameras for registration')
    split = np.arange(len(frames)) % 4 == 0
    train = np.flatnonzero(~split)
    rot, scale, shift, err_ras, err_angle = fit_registration(ef_extrinsics[ei], ras_w2c[ri], train)
    err_mm = 1000 * err_ras / scale
    p95_mm, p95_deg = np.quantile(err_mm[split], .95), np.quantile(err_angle[split], .95)
    accepted = p95_mm <= args.max_heldout_mm and p95_deg <= args.max_heldout_deg
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(video=str(args.video.resolve()), egofound3r=str(args.ef3r.resolve()),
                  ras_map=str(args.ras_map.resolve()), ras_alignment=str(args.ras_alignment.resolve()),
                  accepted=bool(accepted), heldout_p95_mm=float(p95_mm), heldout_p95_deg=float(p95_deg),
                  limits=[args.max_heldout_mm, args.max_heldout_deg],
                  scale_ef_m_to_ras=scale, R_ef_to_ras=rot.as_matrix().tolist(), t_ef_to_ras=shift.tolist(),
                  ras_units_to_meter=1 / scale, source_frames=frames.tolist(),
                  heldout_frames=frames[split].tolist(), residual_mm=err_mm.tolist(), residual_deg=err_angle.tolist())
    (args.out / 'registration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Aligned {len(frames)} cameras, held-out P95 {p95_mm:.1f} mm / {p95_deg:.1f}°', flush=True)
    if not accepted:
        raise SystemExit('Cross-module registration exceeds the requested tolerances')
    joints = np.asarray(ef['hand_joint_xyz_metric_full'], dtype=float)
    hand_valid = np.asarray(ef['hand_metric_valid_full'], dtype=bool) & ef_valid[:, None]
    filled = np.asarray(ef['hand_missing_fill_full'], dtype=bool)
    contact = np.asarray(ef['joint_contact_logits'], dtype=float)
    # World origin and orientation follow RAS. Scale both RAS geometry and its
    # origin offset to metres before joining any object observations.
    ef_centers = camera_center(ef_extrinsics)
    ef_world = np.einsum('nji,nhkj->nhki', ef_extrinsics[:, :3, :3], joints)
    ef_world += ef_centers[:, None, None, :]
    world_joints = rot.apply(ef_world.reshape(-1, 3)).reshape(joints.shape) + shift / scale
    np.savez_compressed(args.out / 'registered_hands.npz', source_frames=ef_frames,
                        timestamps_s=np.asarray(ef['timestamps_sec'], dtype=float),
                        world_joints=world_joints, valid=hand_valid, filled=filled,
                        contact_logits=contact)


if __name__ == '__main__':
    main()

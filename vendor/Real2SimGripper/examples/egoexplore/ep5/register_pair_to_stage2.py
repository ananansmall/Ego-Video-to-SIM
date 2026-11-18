"""Re-register the ep5 F303–720 pair and oven colliders into the stage2 frame.

The old execution frame is gravity-true but its xy/yaw and support heights
differ from the stage2 scene; the raw bridge (module-floor bridge_audit) leaves
support planes 3° off stage2 gravity, which made the bridged flask slide.  This
script:
  1. levels the bridge rotation (old Z -> stage2 Z),
  2. fits xy so the old oven door/cavity/rack/counter align with the stage2
     scene point bounds,
  3. splits the remaining 10.85 mm support-separation mismatch evenly
     (rack +5.4 mm, counter −5.4 mm; both pops are recorded),
  4. rewrites the pair world poses and base route, and re-bakes the oven door /
     cavity / rack / counter colliders in the stage2 frame.
"""
import hashlib
import json
from pathlib import Path
from shutil import copy2

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
BRIDGE = HERE / 'module-floor.jfWG8k/bridge_audit.json'
OLD_ASSETS = HERE / 'replay-oven-agile.high/scene_assets'
S2_ASSETS = HERE / 'ras-mainv3-gravity-luna-xhigh.20261116/scene_assets_stage2_final'
SRC_PKG = HERE / 'hoi-module-fixedep.4Uwlnf/package'
RUN = HERE / 'runs/stage2-merge-f303-720'

RACK_TOP_S2 = 1.0545707345874291
COUNTER_TOP_S2 = 0.72502


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_transform():
    audit = json.loads(BRIDGE.read_text())
    R = np.asarray(audit['stage2_estimated_metric_to_old_execution_R'])
    t = np.asarray(audit['stage2_estimated_metric_to_old_execution_t_m'])
    v = R.T @ np.array([0., 0., 1.])           # bridged old-Z in stage2 (3° tilt)
    axis = np.cross(v, [0., 0., 1.]); axis /= np.linalg.norm(axis)
    ang = np.arccos(np.clip(v[2], -1, 1))
    M = Rotation.from_rotvec(axis * ang).as_matrix() @ R.T
    assert abs(np.linalg.det(M) - 1) < 1e-9
    assert np.allclose(M[:, 2], [0, 0, 1], atol=1e-6)
    psi = float(np.arctan2(M[1, 0], M[0, 0]))  # base-route yaw increment

    # xy fit: old oven geometry centers -> stage2 scene point-bounds centers
    def mapped_center(mesh):
        return M @ (np.asarray(mesh.bounds.mean(axis=0)) - t)
    s2_audit = json.loads((S2_ASSETS / 'scene_audit.json').read_text())
    old_cav = trimesh.load(OLD_ASSETS / 'oven_cavity.obj', force='mesh', process=False)
    old_door = trimesh.load(OLD_ASSETS / 'oven_door.obj', force='mesh', process=False)
    cav_lo = np.asarray(s2_audit['oven_cavity_bounds']['lo'])[:2]
    cav_hi = np.asarray(s2_audit['oven_cavity_bounds']['hi'])[:2]
    door_lo = np.asarray(s2_audit['oven_door_point_bounds']['lo'])[:2]
    door_hi = np.asarray(s2_audit['oven_door_point_bounds']['hi'])[:2]
    targets = np.stack([(cav_lo + cav_hi) / 2, (door_lo + door_hi) / 2])
    mapped = np.stack([mapped_center(old_cav)[:2], mapped_center(old_door)[:2]])
    c_xy = (targets - mapped).mean(axis=0)

    # z fit: split the 10.85 mm support-separation mismatch evenly
    base_z = -(M @ t)[2]
    old_rack_top = 1.1151672184464243
    old_counter_top = 0.7747670662403107
    rack_z = old_rack_top * M[2, 2] + base_z      # with c_z = 0
    counter_z = old_counter_top * M[2, 2] + base_z
    c_z = ((RACK_TOP_S2 - rack_z) + (COUNTER_TOP_S2 - counter_z)) / 2
    c = np.array([*c_xy, c_z])
    info = {'R_up_psi_deg': np.degrees(psi), 'c_xy_m': c_xy.tolist(), 'c_z_m': float(c_z),
            'rack_top_stage2_m': rack_z + c_z, 'counter_top_stage2_m': counter_z + c_z,
            'rack_target_m': RACK_TOP_S2, 'counter_target_m': COUNTER_TOP_S2,
            'rotation_M': M.tolist(), 't_bridge': t.tolist()}
    return M, t, c, psi, info


def map_pose(T, M, t, c):
    """x_s = M (x_o - t) + c as a 4x4 similarity."""
    out = np.eye(4)
    out[:3, :3] = M @ T[:3, :3]
    out[:3, 3] = M @ (T[:3, 3] - t) + c
    return out


def main():
    M, t, c, psi, info = build_transform()
    RUN.mkdir(parents=True, exist_ok=False)
    assets = RUN / 'scene_assets'
    assets.mkdir()

    # 1. re-bake the furniture/oven colliders in the stage2 frame
    manifest = {'transform': info, 'meshes': {}}
    for name in ('table', 'chair', 'oven_door', 'oven_cavity'):
        mesh = trimesh.load(OLD_ASSETS / f'{name}.obj', force='mesh', process=False)
        verts = (M @ (np.asarray(mesh.vertices, float) - t).T + c[:, None]).T
        out = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces, int), process=False)
        out.export(assets / f'{name}.obj')
        manifest['meshes'][name] = {'vertices': len(out.vertices), 'triangles': len(out.faces),
                                    'bounds': out.bounds.round(4).tolist()}

    # 2. re-register the pair
    with np.load(SRC_PKG / 'paired_targets.npz', allow_pickle=False) as z:
        pair = {k: np.asarray(z[k]) for k in z.files}
    n = len(pair['src_frames'])
    M4 = np.eye(4); M4[:3, :3] = M; M4[:3, 3] = c - M @ t
    # Pose conventions in this package differ per key: T_world_pad/_gripper are
    # proper homogeneous matrices, while T_world_object / T_pad_object /
    # T_gripper_object carry bottom row [0,0,0,0] where the rotation is relative
    # and the translation column is the WORLD object position (so that
    # T_world_pad @ T_pad_object == T_world_object under plain matmul).  The
    # runner composes them the same way, so transform each key under its own
    # convention instead of a blanket matrix product.
    t_eff = t - M.T @ c                     # x_s = M (x_o - t_eff), z row of M is identity
    for key in ('T_world_pad', 'T_world_pad_retarget', 'T_world_gripper'):
        if key in pair:
            pair[key] = np.einsum('ij,njk->nik', M4, pair[key])
    if 'T_world_object' in pair:
        T = pair['T_world_object']
        bottom = T[:, 3, :].copy()
        T[:, :3, :3] = np.einsum('ij,njk->nik', M, T[:, :3, :3])
        T[:, :3, 3] = np.einsum('ij,nj->ni', M, T[:, :3, 3] - t_eff)
        T[:, 3, :] = bottom
    for rel_key, world_key in (('T_pad_object', 'T_world_pad'),
                               ('T_gripper_object', 'T_world_gripper')):
        # Under the package convention the runner composes
        # T_world @ T_rel == T_world_object with plain matmul, which requires
        # R_world_new @ P[:3,3] == p_object_new.  Both the world rotation and
        # the object position are already transformed at this point.
        if rel_key in pair and world_key in pair and 'T_world_object' in pair:
            P = pair[rel_key]
            bottom = P[:, 3, :].copy()
            R_new = pair[world_key][:, :3, :3]
            p_obj_new = pair['T_world_object'][:, :3, 3]
            P[:, :3, 3] = np.einsum('nji,nj->ni', R_new, p_obj_new)
            P[:, 3, :] = bottom
    base = np.asarray(pair['base_reference_se2'], float).copy()
    hom = np.c_[base[:, :2], np.zeros(len(base)), np.ones(len(base))].T
    mapped = (M4 @ hom).T
    base[:, :2] = mapped[:, :2]
    base[:, 2] = base[:, 2] + psi
    pair['base_reference_se2'] = base
    np.savez_compressed(RUN / 'paired_targets.npz', **pair)

    # 3. geometry.json for the re-registered package
    geometry = json.loads((SRC_PKG / 'geometry.json').read_text())
    rack = trimesh.load(assets / 'table.obj', force='mesh', process=False)
    counter = trimesh.load(assets / 'chair.obj', force='mesh', process=False)
    for name, mesh in (('table', rack), ('chair', counter)):
        lo, hi = mesh.bounds
        geometry['supports'][name] = {
            'center': ((lo + hi) / 2).tolist(),
            'half_extents': ((hi - lo)[:2] / 2).tolist(),
            'yaw': 0.0}
    geometry['supports']['table']['z_top'] = float(hi := rack.bounds[1][2])
    glb = S2_ASSETS / 'final_scene_world.glb'
    copy2(glb, assets / 'final_scene_world.glb')
    geometry['full_scene']['files'] = {
        'table.obj': sha256(assets / 'table.obj'),
        'chair.obj': sha256(assets / 'chair.obj'),
        'final_scene_world.glb': sha256(assets / 'final_scene_world.glb'),
        'oven:oven_door.obj': sha256(assets / 'oven_door.obj'),
        'oven:oven_cavity.obj': sha256(assets / 'oven_cavity.obj')}
    geometry['full_scene']['scene_asset_dir'] = str(assets.resolve())
    manifest = geometry['full_scene'].get('scene_asset_manifest', {})
    if manifest:
        manifest.update({'table.obj': sha256(assets / 'table.obj'),
                         'chair.obj': sha256(assets / 'chair.obj'),
                         'final_scene_world.glb': sha256(assets / 'final_scene_world.glb'),
                         'oven_door.obj': sha256(assets / 'oven_door.obj'),
                         'oven_cavity.obj': sha256(assets / 'oven_cavity.obj')})
        geometry['full_scene']['scene_asset_manifest'] = manifest
    geometry['scene_asset'] = str(assets.resolve())
    if 'mobile_release' in geometry:
        geometry['mobile_release']['scene_asset'] = str(assets.resolve())
    geometry['ras_final_glb_sha256'] = sha256(assets / 'final_scene_world.glb')
    geometry['stage2_registration'] = info
    (RUN / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')

    # 4. copy the flask visual/collision assets
    for name in ('case_metric.obj', 'case_collision.obj', 'material.mtl', 'material_0.png'):
        copy2(SRC_PKG / name, RUN / name)

    manifest['pair'] = {'rows': n, 'src_frames': [int(pair['src_frames'][0]), int(pair['src_frames'][-1])],
                        'base_yaw_delta_deg': float(np.degrees(psi))}
    manifest['sha256'] = {k: sha256(RUN / k) for k in ('paired_targets.npz', 'geometry.json')}
    (RUN / 'registration_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'c_xy': info['c_xy_m'], 'c_z': info['c_z_m'],
                      'rack_top': info['rack_top_stage2_m'], 'counter_top': info['counter_top_stage2_m'],
                      'psi_deg': info['R_up_psi_deg'], 'run': str(RUN)}, indent=2))


if __name__ == '__main__':
    main()

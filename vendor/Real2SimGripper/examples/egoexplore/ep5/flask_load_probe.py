"""Real-flask load-bearing probe on the re-registered stage2 supports.

Drops the flask (case_collision.obj, 0.08 kg) from just above its pair pose onto
the re-registered rack (run start) and counter (release target) and reports the
settled height, tilt and drift against the stage2 slab tops.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from cb_large_range_sim import _build_scene, rg  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/stage2-merge-f303-720'
DT = 1 / 250
STEPS = 250


def drop(scene, mesh_path, pose, top_z, label, friction=0.5):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type('dynamic')
    builder.add_convex_collision_from_file(str(mesh_path))
    builder.add_visual_from_file(str(mesh_path))
    actor = builder.build(name='flask')
    actor.set_pose(sapien.Pose(p=pose[:3, 3],
                               q=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]))
    for component in actor.get_components():
        if hasattr(component, 'mass'):
            component.mass = .08
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    rest = []
    for _ in range(STEPS):
        scene.step()
        T = actor.get_pose().to_transformation_matrix()
        bottom = float(np.min(np.asarray(mesh.vertices) @ T[:3, :3].T) + T[2, 3])
        rest.append(bottom)
    T = actor.get_pose().to_transformation_matrix()
    bottom = float(np.min(np.asarray(mesh.vertices) @ T[:3, :3].T) + T[2, 3])
    tilt = float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))))
    result = {'probe': label, 'support_top_m': top_z,
              'settled_bottom_m': round(bottom, 5),
              'gap_to_support_mm': round((bottom - top_z) * 1000, 2),
              'settled_tilt_deg': round(tilt, 3),
              'horizontal_drift_mm': round(float(np.linalg.norm(T[:2, 3] - pose[:2, 3])) * 1000, 2),
              'last_50_frames_range_mm': round(float((max(rest[-50:]) - min(rest[-50:])) * 1000), 3)}
    scene.remove_actor(actor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=str(RUN / 'scene_assets'))
    args = parser.parse_args()
    assets = Path(args.assets)
    geometry = json.loads((RUN / 'geometry.json').read_text())
    with np.load(RUN / 'paired_targets.npz', allow_pickle=False) as z:
        obj = np.asarray(z['T_world_object'])
    scene, _ = _build_scene()
    material = scene.create_physical_material(.5, .5, 0.)
    # static supports from the re-registered meshes
    # Rack/counter prisms only: the baked cavity/door surfaces overlap the
    # support xy footprints and would contaminate a free-body drop.
    for name in ('table', 'chair'):
        builder = scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(str(assets / f'{name}.obj'), material=material)
        builder.add_visual_from_file(str(assets / f'{name}.obj'))
        builder.build_static(name=name)
    results = []
    # The pair's F303 pose is a HELD pose (flask straddles the 18 mm rack slab),
    # so load-bearing is probed with upright free-body drops from free space:
    # spawn the flask upright with its bottom 10 mm above each support top, over
    # the support centre, and let it settle.
    case = trimesh.load(RUN / 'case_collision.obj', force='mesh', process=False)
    z_offset = float(np.max(np.asarray(case.vertices)[:, 2]))  # centre-to-bottom

    def upright_drop(mesh_path, top_z, label):
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        center_xy = mesh.bounds.mean(axis=0)[:2]
        pose = np.eye(4)
        pose[:2, 3] = center_xy
        pose[2, 3] = top_z + .010 + z_offset
        return drop(scene, RUN / 'case_collision.obj', pose, top_z, label)

    rack = trimesh.load(assets / 'table.obj', force='mesh', process=False)
    counter = trimesh.load(assets / 'chair.obj', force='mesh', process=False)
    results.append(upright_drop(assets / 'table.obj',
                                float(rack.bounds[1][2]), 'rack_drop'))
    results.append(upright_drop(assets / 'chair.obj',
                                float(counter.bounds[1][2]), 'counter_drop'))
    out = {'scope': 'real-flask load-bearing on re-registered stage2 supports',
           'dt_s': DT, 'steps': STEPS, 'mass_kg': 0.08, 'results': results,
           'stable': all(r['settled_tilt_deg'] < 2. and abs(r['gap_to_support_mm']) < 5.
                         and r['last_50_frames_range_mm'] < 0.5 for r in results)}
    (RUN / 'flask_load_probe.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

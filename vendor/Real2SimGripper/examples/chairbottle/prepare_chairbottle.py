"""Pair the chairbottle RAS scene and AGILE targets for the R1 mobile carry.

The source performs two legs: desk -> wooden chair (F12-167, rest to F184),
wooden chair -> white plastic chair (F184-588, release F588-591). The white
chair is the final release support ('chair' in the physics scene); the wooden
chair is an intermediate static support ('midchair').
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'456483158-mobile-grasp/scene_agile'))
from finish_456 import arm_info
from prepare_full_scene import sha256, support_height, w2c_to_c2w
from replay_paired_scene import bp, build_scene, lrt, package_sha256, rg


def ramp(frames, start, stop):
    u = np.clip((frames-start)/(stop-start), 0., 1.)
    return u*u*u*(10.+u*(-15.+6.*u))


def pose_sequence(position, orientation):
    out = np.repeat(np.eye(4)[None], len(position), axis=0)
    out[:, :3, 3] = position
    out[:, :3, :3] = orientation
    return out


def seg(a, b, frames, start, stop, lift=0.):
    """Linear blend between two pad positions with a vertical lift arc."""
    u = ramp(frames, start, stop)
    out = a[None]+u[:, None]*(b-a)[None]
    out[:, 2] += lift*np.sin(np.pi*u)
    return out


def prepare(args):
    manifest = json.loads((args.ras/'source_manifest.json').read_text())
    raw_glb = args.ras/'final_scene.glb'
    if manifest['source_sha256'] != sha256(args.video) or manifest['final_scene_sha256'] != sha256(raw_glb):
        raise ValueError('The RAS scene is not from this source')
    video = cv2.VideoCapture(str(args.video))
    try:
        frames = np.arange(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    if len(frames) != 592 or not np.isclose(fps, 30.):
        raise ValueError('Expected the chairbottle source with 592 frames at 30 fps')

    ids = np.asarray(manifest['extrinsic_source_frames'])
    if len(ids) < 8 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(frames))):
        raise ValueError('RAS camera entries must have distinct, verified source frame indices')
    ext = np.array([np.loadtxt(args.ras/'extrinsics'/f'{i}.txt') for i in range(len(ids))])
    cam_r, cam_c = w2c_to_c2w(ext)
    ef = np.load(args.ef3r/'camera/traj_c2w.npy')[ids]
    flip = np.diag([-1., 1., 1.])
    ef_from_ras = Rotation.from_matrix(ef[:, :3, :3] @ flip @ np.swapaxes(cam_r, 1, 2) @ flip).mean().as_matrix() @ flip
    X = cam_c @ ef_from_ras.T
    Y = ef[:, :3, 3]
    x = X-X.mean(0); y = Y-Y.mean(0)
    scale = np.sum(x*y)/np.sum(x*x)
    shift = (Y-scale*X).mean(0)
    camera_error = np.linalg.norm(scale*X+shift-Y, axis=1)
    angle_error = np.degrees(Rotation.from_matrix(
        np.swapaxes(ef[:, :3, :3], 1, 2) @ ef_from_ras @ cam_r @ flip).magnitude())
    if not .1 < scale < 5 or np.quantile(camera_error, .95) > .040 or np.quantile(angle_error, .95) > 4.:
        raise ValueError(f'Camera registration mismatch: scale={scale}, p95={np.quantile(camera_error, .95)} m, '
                         f'rotation p95={np.quantile(angle_error, .95)} deg')

    gltf_to_ras = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    seed_transform = np.eye(4)
    seed_transform[:3, :3] = scale*flip@gltf_to_ras
    scene = trimesh.load(raw_glb, force='scene', process=False)
    meshes = {}
    for node in scene.graph.nodes_geometry:
        pose, name = scene.graph[node]
        mesh = scene.geometry[name]
        if isinstance(mesh, trimesh.Trimesh):
            meshes[node] = mesh.copy().apply_transform(seed_transform@pose)

    table_nodes = manifest['support_nodes']['table']
    mid_nodes = manifest['support_nodes']['wooden chair']
    place_nodes = manifest['support_nodes']['white plastic chair']
    bottle_nodes = manifest['dynamic_nodes']
    if not table_nodes or not mid_nodes or not place_nodes or len(bottle_nodes) != 1:
        raise ValueError(f'Identify the desk, both chairs and one bottle. Nodes={list(meshes)}')
    table = trimesh.util.concatenate([meshes[n] for n in table_nodes])
    midchair = trimesh.util.concatenate([meshes[n] for n in mid_nodes])
    finalchair = trimesh.util.concatenate([meshes[n] for n in place_nodes])
    bottle = meshes[bottle_nodes[0]]

    top = table.face_normals[:, 2] > .72
    if not top.any():
        raise ValueError('Desk mesh has no upward support faces')
    up = np.average(table.face_normals[top], axis=0, weights=table.area_faces[top])
    tilt, _ = Rotation.align_vectors([[0., 0., 1.]], [up/np.linalg.norm(up)])
    eye = tilt.apply(scale*(cam_c[0]@flip))
    target = tilt.apply(bottle.bounds.mean(0))
    forward = target[:2]-eye[:2]
    yaw = np.pi/2-np.arctan2(forward[1], forward[0])
    W = (Rotation.from_euler('z', yaw)*tilt).as_matrix()
    sim_world = np.eye(4)
    sim_world[:3, :3] = W
    bottle_xy = (W @ bottle.bounds.mean(0))[:2]
    sim_world[:2, 3] = [0., .30]-bottle_xy
    for mesh in meshes.values():
        mesh.apply_transform(sim_world)
    table = trimesh.util.concatenate([meshes[n] for n in table_nodes])
    midchair = trimesh.util.concatenate([meshes[n] for n in mid_nodes])
    finalchair = trimesh.util.concatenate([meshes[n] for n in place_nodes])
    bottle = meshes[bottle_nodes[0]]
    pick_xy = bottle.bounds.mean(0)[:2]
    mid_xy = midchair.bounds.mean(0)[:2]
    place_xy = finalchair.bounds.mean(0)[:2]
    raw_pick_z = support_height(table, pick_xy)
    shift_z = .75-raw_pick_z
    for mesh in meshes.values():
        mesh.apply_translation([0., 0., shift_z])
    sim_world[2, 3] = shift_z
    table = trimesh.util.concatenate([meshes[n] for n in table_nodes])
    midchair = trimesh.util.concatenate([meshes[n] for n in mid_nodes])
    finalchair = trimesh.util.concatenate([meshes[n] for n in place_nodes])
    bottle = meshes[bottle_nodes[0]]
    table_z = support_height(table, pick_xy)
    mid_z = support_height(midchair, mid_xy)
    place_z = support_height(finalchair, place_xy)
    if not .25 <= table_z-mid_z <= .55 or not .25 <= table_z-place_z <= .55:
        raise ValueError(f'Chair seats inconsistent with the desk: mid={table_z-mid_z:.3f} m, '
                         f'final={table_z-place_z:.3f} m below the desk top')
    travel = float(np.linalg.norm(place_xy-pick_xy))
    if travel < .50:
        raise ValueError(f'Desk to white-chair transfer is only {travel:.3f} m; '
                         'the large-range gate needs >=0.50 m of net base displacement')

    out = args.out.resolve()
    assets, package = out/'scene_assets', out/'refined'
    assets.mkdir(parents=True, exist_ok=False)
    package.mkdir()
    visible = trimesh.Scene()
    for name, mesh in meshes.items():
        if name not in bottle_nodes:
            visible.add_geometry(mesh, node_name=name)
    visible.export(assets/'final_scene_world.glb')
    table.export(assets/'table.obj', include_texture=False)
    finalchair.export(assets/'chair.obj', include_texture=False)
    midchair.export(assets/'midchair.obj', include_texture=False)

    center = bottle.bounds.mean(0)
    bottle = bottle.copy().apply_translation(-center)
    bottle.export(package/'case_metric.obj')
    material = bottle.visual.material
    texture = getattr(material, 'baseColorTexture', None)
    if texture is None:
        texture = getattr(material, 'image', None)
    if texture is None:
        raise ValueError('The bottle asset has no source texture')
    texture.save(package/'material_0.png')
    (package/'material.mtl').write_text('newmtl material_0\nmap_Kd material_0.png\n')
    z_min = bottle.vertices[:, 2].min()
    pick = np.r_[pick_xy, table_z-z_min+.0005]
    mid = np.r_[mid_xy, mid_z-z_min+.0005]
    place = np.r_[place_xy, place_z-z_min+.0005]

    with np.load(args.agile) as data:
        ef_axes = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        ef_to_sim = W @ flip @ ef_from_ras.T
        orient = ef_to_sim @ ef_axes.T
        source_R = orient @ data['rot']
        source_p = data['pos'] @ orient.T - shift @ ef_to_sim.T + sim_world[:3, 3]
        source_p += rg.PAD_FWD*source_R[:, :, 0]
        source_grip = data['grip']
    source_pad = pose_sequence(source_p, source_R)

    # Scripted pad trajectory preserving the source order: desk grasp (F45),
    # wooden-chair rest (F150-184), regrasp (F184), white-chair release (F588).
    R = Rotation.from_euler('z', np.pi/2).as_matrix()
    offset = R @ [-.005, 0., args.grasp_z]
    pick_pad, mid_pad, place_pad = pick+offset, mid+offset, place+offset
    seat_R = R
    p = np.zeros((len(frames), 3))
    entry = source_p[0].copy()
    entry[:2] = pick_pad[:2]+np.clip(entry[:2]-pick_pad[:2], -.25, .25)
    entry[2] = max(entry[2], table_z+args.approach_height)
    clearance = pick_pad+[0., -.11, .110]
    for a, b, s, t in ((entry, clearance, 0, 30), (clearance, pick_pad, 30, 45),
                       (pick_pad, pick_pad, 45, 58), (pick_pad, mid_pad, 58, 135),
                       (mid_pad, mid_pad, 135, 165),
                       (mid_pad, mid_pad+[0., 0., .075], 165, 184),
                       (mid_pad+[0., 0., .075], mid_pad, 184, 196),
                       (mid_pad, mid_pad, 196, 210), (mid_pad, place_pad, 210, 500),
                       (place_pad, place_pad, 500, 591)):
        p[s:t] = seg(a, b, frames[s:t], s, t)
    # Lift arc during the two carries.
    p[58:135, 2] += .08*np.sin(np.pi*ramp(frames[58:135], 58, 135))
    p[210:500, 2] += .06*np.sin(np.pi*ramp(frames[210:500], 210, 500))
    ori = np.repeat(R[None], len(frames), axis=0)
    # Grip: open before the desk grasp, close F45-53, ease F152-165 for the
    # wooden-chair rest, re-close F184-196, closed until the physical release.
    span = np.ptp(bottle.vertices @ R[:, 1])
    closed = lrt.grasp_width_for_object(span, squeeze=args.squeeze)
    opened = min(.048, max(.034, closed+.011))
    if opened <= closed+.005:
        raise ValueError('Selected bottle grasp exceeds jaw travel')
    eased = min(.048, closed+.010)
    grip = np.full(len(frames), opened)
    grip[45:53] = opened+(closed-opened)*ramp(frames[45:53], 45, 53)
    grip[53:152] = closed
    grip[152:165] = closed+(eased-closed)*ramp(frames[152:165], 152, 165)
    grip[165:184] = eased
    grip[184:196] = eased+(closed-eased)*ramp(frames[184:196], 184, 196)
    grip[196:] = closed
    grip = np.repeat(grip[:, None], 2, axis=1)
    states = np.zeros(len(frames), int)
    states[32:45], states[45:53], states[53:130] = 1, 2, 3
    states[130:150], states[150:184], states[184:196], states[196:517] = 4, 3, 2, 3
    states[517:588], states[588:] = 4, 5
    target = pose_sequence(p, ori)
    obj = np.zeros((len(frames), 4, 4))
    obj[:, :3, :3] = np.eye(3)
    u_hold = np.clip((frames-45)/(150-45), 0., 1.)
    u_hold2 = np.clip((frames-196)/(517-196), 0., 1.)
    obj[:, :3, 3] = (pick[None]+u_hold[:, None]*(mid-pick)[None]) * (frames[:, None] <= 150)
    obj[:, :3, 3] += (mid[None]+u_hold2[:, None]*(place-mid)[None]) * (frames[:, None] > 150)
    pair = np.linalg.inv(target)@obj
    weights = np.ones(len(frames))
    np.savez_compressed(package/'paired_targets.npz', src_frames=frames, video_fps=fps, grip=grip,
                        grip_states=states, T_world_pad=target, T_pad_object=pair,
                        T_world_gripper=target, T_gripper_object=pair, T_world_object=obj,
                        T_world_pad_retarget=source_pad, grip_retarget=source_grip, contact_blend=weights)

    def support(mesh, xyz):
        return dict(center=xyz.tolist(), half_extents=(mesh.extents[:2]/2).tolist(), yaw=0.)
    supports = dict(table=support(table, np.r_[pick_xy, table_z]),
                    midchair=support(midchair, np.r_[mid_xy, mid_z]),
                    chair=support(finalchair, np.r_[place_xy, place_z]))
    geometry = dict(source_video=args.video.name, grasp_z_m=args.grasp_z,
        instruction='Carry the bottle from the desk to the wooden chair, then to the white plastic chair.',
        supports=supports,
        contact_refinement=dict(grasp_window=[45, 53], stationary_on_table_through_frame=53,
                                target_jaw_gap_max_mm=4.),
        initial_case_bottom_above_table_mm=.5, final_case_bottom_above_chair_mm=.5,
        mobile_release=dict(scene_asset=str(assets), seat_frame=540,
                            support_name='white plastic chair', open_clearance_m=args.open_clearance),
        full_scene=dict(source_sha256=manifest['source_sha256'], ras_final_glb_sha256=manifest['final_scene_sha256'],
            support_nodes=dict(table=table_nodes, midchair=mid_nodes, chair=place_nodes),
            dynamic_nodes=bottle_nodes, world_from_gltf=(sim_world@seed_transform).tolist(),
            mirror_from_source=True, scale=float(scale),
            camera_center_p95_mm=float(np.quantile(camera_error, .95)*1000),
            camera_angle_p95_deg=float(np.quantile(angle_error, .95)),
            source_windows=dict(desk_grasp=[45, 53], wooden_chair_rest=[150, 184],
                                regrasp=[184, 196], white_chair_descent=[517, 588],
                                release_request=[588, 591]),
            station_distances_m=dict(desk_to_wooden=float(np.linalg.norm(mid_xy-pick_xy)),
                                     wooden_to_white=float(np.linalg.norm(place_xy-mid_xy)),
                                     desk_to_white=travel),
            files={q.name: sha256(q) for q in assets.iterdir()}))
    (package/'geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')

    sim, robot, _ = build_scene(supports, scene_meshes=assets)
    arm, limits = arm_info(robot)
    base_pick = np.r_[pick[0]+.13, min(pick[1]-.53, table.bounds[0, 1]-.28)]
    base_mid = np.r_[mid[0]+.13, min(mid[1]-.53, midchair.bounds[0, 1]-.28)]
    base_place = np.r_[place[0]+.13, min(place[1]-.53, finalchair.bounds[0, 1]-.28)]
    base_xy = np.repeat(base_pick[None], len(frames), axis=0)
    base_xy[58:135] = base_pick+ramp(frames[58:135], 58, 135)[:, None]*(base_mid-base_pick)
    base_xy[135:210] = base_mid
    base_xy[210:500] = base_mid+ramp(frames[210:500], 210, 500)[:, None]*(base_place-base_mid)
    base_xy[500:] = base_place
    base_xy -= np.array(rg.R1_BASE_P)[:2]
    yaw_ref = np.zeros(len(frames))
    _, joints, pe, oe = lrt.solve_wbc_track(robot, 'right', arm, limits, base_xy[:, 0], base_xy[:, 1],
        list(zip(p, ori)), p, bp.se2_pose, bp._gripper_pad_fk, w_pos=55., w_ori=5., w_base=8., w_yaw=8.,
        w_posture=.2, w_limit=.5, w_cont=1., n_local=1, n_rand=0, max_nfev=75, ref_yaw=yaw_ref,
        base_range=.0004, yaw_max=.0004, base_step_max=.029, joint_step_max=5.5,
        torso_pitch=np.deg2rad([9., 11., 13.]))
    (out/'refined_ik').mkdir()
    np.savez_compressed(out/'refined_ik/ik_plan.npz', source_frames=frames,
        package_sha256=package_sha256(package), base_se2=np.c_[base_xy, yaw_ref],
        arm_joint_indices=arm, arm_joint_targets=joints, position_error_mm=pe, orientation_error_deg=oe)
    print(json.dumps(dict(pick=pick.tolist(), mid=mid.tolist(), place=place.tolist(),
        bottle_extent=bottle.extents.tolist(), width=float(span), closed=float(closed),
        desk_to_white_m=travel, base_travel=float(np.linalg.norm(base_place-base_pick)),
        ik_p95_mm=float(np.quantile(pe, .95))), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ras', 'video', 'ef3r', 'agile', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--grasp-z', type=float, default=.006)
    parser.add_argument('--approach-height', type=float, default=.160)
    parser.add_argument('--squeeze', type=float, default=.004)
    parser.add_argument('--open-clearance', type=float, default=.018)
    prepare(parser.parse_args())

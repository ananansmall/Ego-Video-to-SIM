"""Replay the RAS/AGILE pair as R1 commands and a free PhysX rigid body."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples/456483158-mobile-grasp/scene_agile'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cb_large_range_sim import _build_scene, bp, lrt, rg


def case_collision_mesh(visual_mesh):
    collision_mesh = visual_mesh.with_name('case_collision.obj')
    return collision_mesh if collision_mesh.is_file() else visual_mesh


def torso_pitch_limits(geometry):
    limits = np.asarray(geometry.get('torso_pitch_limit_deg', [12., 16., 18.]), dtype=float)
    margin = np.asarray(geometry.get('torso_pitch_tracking_margin_deg', [3., 5., 5.]), dtype=float)
    if (limits.shape != (3,) or margin.shape != (3,)
            or not np.isfinite(limits).all() or not np.isfinite(margin).all()
            or np.any(limits > 90.) or np.any(margin < 0.) or np.any(limits <= margin)):
        raise ValueError('Expected three positive torso pitch limits up to 90 degrees '
                         'and nonnegative tracking margins smaller than the limits')
    return limits, margin


def make_rigid_body(scene, obj, pose, texture, material):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type('dynamic')
    builder.add_convex_collision_from_file(str(case_collision_mesh(obj)), material=material)
    builder.add_visual_from_file(str(obj), material=rg._textured_render_material(str(texture)))
    case = builder.build(name='case')
    case.set_pose(sapien.Pose(p=pose[:3, 3], q=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]))
    for component in case.get_components():
        if hasattr(component, 'set_disable_gravity'):
            component.mass = .08
    return case


def measure_contacts(scene, impulses, case_columns, support_names, robot_links, unexpected=None):
    furniture = False
    support_vertical = {name: 0. for name in support_names}
    support_normals = {name: [] for name in support_names}
    for contact in scene.get_contacts():
        a, b = (body.get_entity().get_name() for body in contact.bodies)
        impulse = sum(np.linalg.norm(point.impulse) for point in contact.points)
        if a == 'case' or b == 'case':
            other = b if a == 'case' else a
            impulses[case_columns.get(other, len(case_columns))] += impulse
            if other in support_names:
                for point in contact.points:
                    sign = 1. if a == 'case' else -1.
                    normal = sign*np.asarray(point.normal, dtype=float)
                    support_normals[other].append(normal)
                    if normal[2] > .8:
                        support_vertical[other] += max(0., sign*float(point.impulse[2]))
        if impulse > 1e-6:
            furniture |= (a in support_names and b in robot_links) or (b in support_names and a in robot_links)
            if unexpected is not None and (a in robot_links or b in robot_links):
                link, other = (a, b) if a in robot_links else (b, a)
                wheel_ground = other == 'ground' and link.startswith('wheel_motor_link')
                bottle_grip = other == 'case' and link in case_columns
                if not (wheel_ground or bottle_grip):
                    unexpected.add(tuple(sorted((a, b))))
    return furniture, support_vertical, support_normals


def package_sha256(package):
    digest = hashlib.sha256()
    files = ['paired_targets.npz', 'geometry.json', 'case_metric.obj', 'material.mtl', 'material_0.png']
    collision_mesh = case_collision_mesh(package/'case_metric.obj')
    if collision_mesh.name != 'case_metric.obj':
        files.append(collision_mesh.name)
    for name in files:
        digest.update(name.encode())
        digest.update(hashlib.sha256((package/name).read_bytes()).digest())
    return digest.hexdigest()


def build_scene(supports, planar_base=False, scene_meshes=None):
    scene, robot = _build_scene(planar_base=planar_base)
    if not np.isclose(scene.get_timestep(), 1/250):
        raise ValueError('Expected the project physics step of 1/250 second')
    material = scene.create_physical_material(1., .9, 0.)
    for name, data in supports.items():
        if scene_meshes is not None:
            builder = scene.create_actor_builder()
            builder.add_nonconvex_collision_from_file(str(scene_meshes/f'{name}.obj'), material=material)
            builder.build_static(name=name)
            continue
        center = np.array(data['center']) - [0, 0, .0225]
        quat = Rotation.from_euler('z', data['yaw']).as_quat()[[3, 0, 1, 2]]
        color = (.82, .83, .82) if name == 'table' else (.31, .32, .33)
        rg.create_box(scene, sapien.Pose(p=center, q=quat), [*data['half_extents'], .0225],
                      color=color, name=name, is_static=True)
    return scene, robot, material


def render_trace(package, out, supports, fps, frames, digest, trace_path, scene_asset=None, scene_meshes=None):
    with np.load(trace_path) as trace:
        if str(trace['package_sha256']) != digest or not np.array_equal(trace['source_frames'], frames):
            raise ValueError('Execution trace belongs to a different paired package. Run physics again.')
        base, q, objects = trace['base_se2'], trace['q_actual'], trace['actual_object']
    scene, robot, material = build_scene(supports, scene_meshes=scene_meshes)
    if scene_asset is not None:
        builder=scene.create_actor_builder()
        glb_name = 'final_scene_world.glb' if scene_meshes is not None else 'video_scene.glb'
        builder.add_visual_from_file(str(scene_asset/glb_name))
        builder.build_static(name='same_video_scene')
        for entity in scene.entities:
            if entity.name in supports:
                for component in entity.components:
                    if hasattr(component,'visibility'):
                        component.visibility=0.
    case = make_rigid_body(scene, package/'case_metric.obj', objects[0], package/'material_0.png', material)
    link = robot.find_link_by_name('right_gripper_link')
    front, side, fpv, overhead = [
        scene.add_camera(name, 480, 360, np.deg2rad(fov), .025, 30)
        for name, fov in [('front', 51), ('side', 51), ('robot_fpv', 88), ('overhead', 54)]
    ]
    middle = (np.array(supports['table']['center']) + supports['chair']['center'])/2
    middle[2] = .94
    front.entity.set_pose(rg.look_at_pose(middle+[-1.55, -1.35, .95], middle))
    side.entity.set_pose(rg.look_at_pose(middle+[1.68, -.39, .85], middle))
    overhead.entity.set_pose(rg.look_at_pose(middle+[.03, -.10, 2.10], middle))
    camera_to_gripper = rg.look_at_pose([.005, -.095, .085], [rg.PAD_FWD, 0., -.012]).to_transformation_matrix()
    if scene_asset is not None:
        from robotwin_r1_duck_dataset import encode_jpg
        vision={view.entity.name:dict(colors=[],intrinsic=[],extrinsic=[],shape=[360,480,3])
                for view in (front,side,fpv,overhead)}
    with imageio.get_writer(out/'four_views.mp4', fps=fps, macro_block_size=8) as cam, \
         imageio.get_writer(out/'robot_fpv.mp4', fps=fps, macro_block_size=8) as ego:
        for base_pose, qpos, obj in zip(base, q, objects, strict=True):
            # This scene replays measured poses for visualization only and is never stepped.
            robot.set_root_pose(bp.se2_pose(*base_pose))
            robot.set_qpos(qpos)
            case.set_pose(sapien.Pose(obj))
            hand = link.get_entity_pose().to_transformation_matrix()
            fpv.entity.set_pose(sapien.Pose(hand @ camera_to_gripper))
            scene.update_render()
            pictures = []
            for view in (front, side, fpv, overhead):
                view.take_picture()
                pictures.append(np.uint8(255*np.clip(view.get_picture('Color')[:, :, :3], 0, 1)))
                if scene_asset is not None:
                    entry=vision[view.entity.name]
                    entry['colors'].append(encode_jpg(pictures[-1]))
                    entry['intrinsic'].append(view.get_intrinsic_matrix())
                    entry['extrinsic'].append(view.get_extrinsic_matrix())
            cam.append_data(np.vstack((np.hstack(pictures[:2]), np.hstack(pictures[2:]))))
            ego.append_data(pictures[2])
    if scene_asset is not None:
        export_dataset(out,trace_path,vision,fps, package)


def export_dataset(out,trace_path,vision,fps, package):
    from robotwin_r1_duck_dataset import write_hdf5
    import h5py

    def pose_arrays(T):
        quat=Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        return np.c_[T[:,:3,3],quat]

    with np.load(trace_path) as trace:
        joints=trace['arm_joint_indices']
        fingers=trace['finger_joint_indices']
        data={
            'state/right_arm_joint_states':trace['q_actual'][:,joints[:6]],
            'state/right_ee_joint_states':trace['q_actual'][:,fingers].mean(1,keepdims=True),
            'state/right_ee_poses':pose_arrays(trace['actual_pad']),
            'action/right_arm_joint_states':trace['arm_joint_targets'][:,:6],
            'action/right_ee_joint_states':trace['finger_joint_targets'].mean(1,keepdims=True),
            'action/right_ee_poses':pose_arrays(trace['target_pad']),
            'state/base_se2':trace['base_se2'],'action/base_se2':trace['base_se2_target'],
            'object/case_pose':pose_arrays(trace['actual_object']),
            'physics/contact_impulses':trace['contact_impulses'],
            'physics/chair_vertical_impulse':trace['chair_vertical_impulse'],
            'physics/substeps':trace['physics_substeps'],'physics/bilateral_substeps':trace['bilateral'],
            'physics/right_fingers_q':trace['q_actual'][:,fingers],
            'timeline/source_frame':trace['source_frames'],'timeline/command_frame':trace['command_frames'],
        }
    verdict=json.loads((out/'verdict.json').read_text())
    passed=bool(verdict['training_data_pass'])
    diagnostic = 'ik_only_diagnostic' in verdict
    path=out/('episode.hdf5' if passed and not diagnostic else 'episode_diagnostic.hdf5')
    geometry = json.loads((package/'geometry.json').read_text())
    write_hdf5(str(path), data, vision, dict(frequency=round(fps),
        instruction=geometry.get('instruction', 'Move the case from table to chair.'),
        source=geometry.get('source_video', '456483158.mp4')))
    with h5py.File(path,'a') as f:
        f.attrs['training_data_pass']=passed and not diagnostic
        if diagnostic:
            f.attrs['scene_validation'] = 'IK-only diagnostic on the prior supports'
        f.attrs['source_frame_count']=len(data['timeline/source_frame'])
        if 'ras_final_glb_sha256' in verdict:
            f.attrs['ras_final_glb_sha256'] = verdict['ras_final_glb_sha256']


def replay(args):
    package = args.package
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    targets = np.load(package/'paired_targets.npz')
    meta = json.loads((package/'geometry.json').read_text())
    supports = meta['supports']
    mobile_release = meta.get('mobile_release')
    open_frames, lift_frames, blend_frames = (8, 5, 12) if mobile_release is None else (4, 3, 6)
    finish=meta.get('finish_456', mobile_release)
    scene_asset=Path(finish['scene_asset']) if finish is not None else \
        (Path(meta['scene_asset']) if meta.get('scene_asset') else None)
    full_scene = meta.get('full_scene')
    scene_meshes = scene_asset if full_scene is not None else None
    if full_scene is not None:
        from prepare_full_scene import sha256
        for name in ('table.obj', 'chair.obj', 'final_scene_world.glb'):
            if sha256(scene_meshes/name) != full_scene['files'][name]:
                raise ValueError(f'Registered RAS geometry changed after packaging: {name}')
        for name in ('oven_door.obj', 'oven_cavity.obj'):
            if 'oven:'+name in full_scene.get('files', {}):
                if not (scene_meshes/name).is_file():
                    raise ValueError(f'Required oven collider missing from scene assets: '
                                     f'{scene_meshes/name}')
                if sha256(scene_meshes/name) != full_scene['files']['oven:'+name]:
                    raise ValueError(f'Oven collider hash changed after packaging: {name}')
    digest = package_sha256(package)
    fps = float(targets['video_fps'])
    frames = targets['src_frames']
    if args.render_trace is not None:
        render_trace(package, out, supports, fps, frames, digest, args.render_trace,scene_asset,scene_meshes)
        return
    retargeted = 'T_world_pad_retarget' in targets
    route = None
    if 'base_reference_se2' in targets:
        route = np.asarray(targets['base_reference_se2'], dtype=float)
        if route.shape != (len(frames), 3) or not np.isfinite(route).all():
            raise ValueError('base_reference_se2 needs world x/y (m) and yaw offset (rad) for each frame')
    scene, robot, material = build_scene(supports, planar_base=(retargeted or route is not None) and not args.plan_only,
                                          scene_meshes=scene_meshes)
    reference = targets['T_world_object']
    pair = targets['T_gripper_object']
    # The optimization's MANO SVD position is adjusted by half the two finger
    # travels. Move from that frame to the physical pad center without changing
    # the world hand or object reference. WBC's FK is measured at the pad center.
    if 'T_world_pad' in targets:
        target_T = targets['T_world_pad']
        pair_pad = targets['T_pad_object']
    else:
        target_T = targets['T_world_gripper'].copy()
        target_T[:, :3, 3] = targets['world_joints'][:, [4, 8]].mean(1)
        pair_pad = np.linalg.inv(target_T) @ targets['T_world_gripper'] @ pair
    if not np.allclose(target_T @ pair_pad, reference, atol=1e-9):
        raise ValueError('Pad-frame conversion changed the AGILE object trajectory')
    if retargeted:
        source_pad = targets['T_world_pad_retarget']
        source_grip = targets['grip_retarget']
        contact_weight = targets['contact_blend']
        free = contact_weight == 0
        if not (np.array_equal(target_T[free], source_pad[free])
                and np.array_equal(targets['grip'][free], source_grip[free])):
            raise ValueError('Free-motion pose or opening diverged from the retargeted targets')
    if args.source_window is not None:
        first, last = args.source_window
        if not frames[0] <= first <= last <= frames[-1]:
            raise ValueError('IK source window is outside the paired frames')
        selected = (frames >= first) & (frames <= last)
        frames, target_T = frames[selected], target_T[selected]
        if route is not None:
            route = route[selected]
        if retargeted:
            contact_weight = contact_weight[selected]
    target_p, target_R = target_T[:, :3, 3], target_T[:, :3, :3]

    joints = robot.get_active_joints()
    joint_index = {j.get_name(): i for i, j in enumerate(joints)}
    planar = [joint_index[name] for name in ('planar_joint_x','planar_joint_y','planar_joint_yaw') if name in joint_index]
    native = [i for i in range(len(joints)) if i not in planar]
    arm = [joint_index[f'right_arm_joint{i+1}'] for i in range(6)]
    arm += [joint_index[f'torso_joint{i+1}'] for i in range(4)]
    fingers = [joint_index[f'right_gripper_finger_joint{i+1}'] for i in range(2)]
    left = [joint_index[f'left_arm_joint{i+1}'] for i in range(6)]
    q = robot.get_qpos()
    q[left] = rg.R1_LEFT_START
    if route is not None:
        # The parked gripper must not start with its two collision meshes overlapping.
        q[[joint_index[f'left_gripper_finger_joint{i}'] for i in (1, 2)]] = .045
    robot.set_qpos(q)
    lim = np.vstack((bp._limits('right'), [joints[i].get_limits()[0] for i in arm[6:]]))
    for i, joint in enumerate(joints):
        if i in fingers:
            joint.set_drive_property(stiffness=4000., damping=120., force_limit=28.)
        elif i in planar:
            joint.set_drive_property(stiffness=35000., damping=3000., force_limit=5000.)
        else:
            joint.set_drive_property(stiffness=2400., damping=150.)
    for i in (1, 2):
        link = robot.find_link_by_name(f'right_gripper_finger_link{i}')
        for component in link.entity.components:
            for shape in getattr(component, 'get_collision_shapes', lambda: ())():
                shape.set_physical_material(material)

    n = len(target_p)
    if not np.array_equal(np.diff(frames), np.ones(n-1, dtype=int)):
        raise ValueError('Paired trajectory must use contiguous source frames')
    if not np.isfinite(args.base_lead) or args.base_lead < 0:
        raise ValueError('Base lead must be a finite nonnegative duration')
    # The base starts alongside the supports. It leads the target's movement
    # along the table-to-chair direction while the arm follows the paired pose.
    ref_yaw = np.full(n, -np.pi/2)
    lead = min(n-1, int(round(args.base_lead*fps)))
    lead_index = np.minimum(np.arange(n)+lead, n-1)
    anchor_x = min(data['center'][0] - max(data['half_extents'])
                   for data in supports.values())
    base_xy = np.c_[np.full(n, anchor_x-.25), target_p[lead_index, 1] + .15]
    if retargeted:
        local = meta['contact_refinement']
        approach,grasp = local['grasp_window']
        closing = (frames >= approach) & (frames <= local.get('local_lift_blend_end_frame',local['stationary_on_table_through_frame']))
        if np.any(closing):
            contact_idx = int(np.clip(np.searchsorted(frames, grasp), 0, n-1))
            base_xy[closing, 1] = target_p[contact_idx, 1] + .15
    base_range, yaw_max = .24, .5
    if route is not None:
        base_xy, ref_yaw = route[:, :2].copy(), np.unwrap(route[:, 2])
        base_range = float(meta.get('base_reference_xy_tolerance_m', .02))
        yaw_max = np.radians(float(meta.get('base_reference_yaw_tolerance_deg', 5.)))
        if not np.isfinite([base_range, yaw_max]).all() or min(base_range, yaw_max) <= 0:
            raise ValueError('Base reference tolerances must be finite and positive')
    # Keep positive bounds for scipy while testing a stationary base pose.
    if args.base_station is not None:
        x, y, yaw = args.base_station
        if not np.isfinite(args.base_station).all():
            raise ValueError('Base station must be finite')
        base_xy[:] = [x, y]
        ref_yaw[:] = np.radians(yaw)
        base_range, yaw_max = .0005, np.radians(.1)
    base_xy -= np.asarray(rg.R1_BASE_P)[:2]
    base_step_max = .025
    if route is not None and np.any(np.abs(np.diff(base_xy, axis=0)) > base_step_max/np.sqrt(2.) + 1e-9):
        raise ValueError('Base reference exceeds the per-frame axis travel budget; reschedule the route')
    if retargeted and route is None:
        # The solver intersects the station envelope with a per-axis velocity
        # bound. Limit the reference's rate as well so that intersection stays
        # feasible even when the hand target shifts during contact correction.
        base_xy = lrt.clip_step(base_xy, .8*base_step_max/np.sqrt(2.))
    pitch_limit, pitch_tracking_margin = torso_pitch_limits(meta)
    if args.plan is None:
        # Preserve the retargeted pose on free frames and retain the same
        # geometric bounds and torso limits for the contact window.
        weights = dict(w_pos=42., w_ori=4., w_posture=.15, w_limit=.5) if retargeted or route is not None else dict(
            w_pos=10., w_ori=2., w_posture=1., w_limit=1.)
        initial_seed = None
        if 'ik_initial_arm_seed_rad' in meta:
            seed = np.asarray(meta['ik_initial_arm_seed_rad'], dtype=float)
            if seed.shape != (len(arm),) or not np.isfinite(seed).all():
                raise ValueError('IK arm seed must contain one finite angle per arm and torso joint')
            initial_seed = np.r_[0., 0., 0., seed]
        base, q_plan, pe, oe = lrt.solve_wbc_track(
            robot, 'right', arm, lim, base_xy[:, 0], base_xy[:, 1],
            list(zip(target_p, target_R)), target_p, bp.se2_pose, bp._gripper_pad_fk,
            w_base=3., w_bcont=1., w_cont=.7 if route is not None else 2.,
            n_local=1, n_rand=0, max_nfev=70, ref_yaw=ref_yaw,
            base_range=base_range, yaw_max=yaw_max, base_step_max=base_step_max, joint_step_max=10.,
            q_neutral=np.r_[lim[:6].mean(1), np.zeros(4)],
            torso_pitch=np.radians(pitch_limit-pitch_tracking_margin), initial_seed=initial_seed,
            optimize_continuity=route is not None, **weights)
        if args.plan_only:
            names = [joints[index].get_name() for index in arm]
            margin = np.degrees(np.minimum(q_plan-lim[:, 0], lim[:, 1]-q_plan))
            pitch_deg = np.degrees(lrt.torso_link_pitch(q_plan[:, 6:9]))
            np.savez_compressed(out/'ik_plan.npz', source_frames=frames, package_sha256=digest,
                                base_lead_s=args.base_lead, base_se2=base, arm_joint_indices=arm,
                                arm_joint_targets=q_plan, position_error_mm=pe, orientation_error_deg=oe,
                                joint_names=names, joint_limits_rad=lim, joint_margin_deg=margin,
                                base_station=np.asarray(args.base_station or [], dtype=float))
            columns = ['source_frame', 'position_error_mm', 'orientation_error_deg', 'base_x_world_m',
                       'base_y_world_m', 'base_yaw_deg', 'torso_pitch_1_deg', 'torso_pitch_2_deg',
                       'torso_pitch_3_deg', 'min_joint_margin_deg']
            columns += [f'{name}_deg' for name in names] + [f'{name}_margin_deg' for name in names]
            np.savetxt(out/'ik_frames.csv', np.c_[frames, pe, oe, base[:, :2]+rg.R1_BASE_P[:2],
                        np.degrees(base[:, 2]), pitch_deg, margin.min(1), np.degrees(q_plan), margin],
                       fmt=['%d']+['%.6f']*(len(columns)-1), delimiter=',', header=','.join(columns), comments='')
            ik = dict(frames=n, base_lead_s=args.base_lead, position_p95_mm=float(np.quantile(pe, .95)),
                      orientation_p95_deg=float(np.quantile(oe, .95)),
                      source_frame_range=[int(frames[0]), int(frames[-1])], package_sha256=digest,
                      base_station=args.base_station, min_joint_margin_deg=float(margin.min()),
                      joint_limit_violation_frames=int(np.sum((margin < -1e-6).any(1))),
                      frames_within_5deg_of_joint_limit=int(np.sum((margin < 5).any(1))),
                      planned_torso_peak_deg=np.max(np.abs(pitch_deg), axis=0).tolist(),
                      planned_torso_limit_deg=(pitch_limit-pitch_tracking_margin).tolist(),
                      max_base_step_mm=float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).max(initial=0.)*1000),
                      max_base_yaw_step_deg=float(np.degrees(np.abs(np.diff(base[:, 2])).max(initial=0.))),
                      max_joint_step_deg=float(np.degrees(np.abs(np.diff(q_plan, axis=0)).max(initial=0.))))
            if retargeted:
                first_contact = np.flatnonzero(targets['contact_blend'] > 0)
                front = frames < targets['src_frames'][first_contact[0]] if len(first_contact) else contact_weight == 0
                if np.any(front):
                    ik['pre_contact_retarget'] = dict(frames=int(front.sum()),
                        position_p95_mm=float(np.quantile(pe[front], .95)),
                        orientation_p95_deg=float(np.quantile(oe[front], .95)))
            (out/'ik_metrics.json').write_text(json.dumps(ik, indent=2)+'\n')
            print(json.dumps(ik, indent=2), flush=True)
            return
    else:
        with np.load(args.plan) as plan:
            if (str(plan['package_sha256']) != digest or not np.array_equal(plan['source_frames'], frames)
                    or not np.array_equal(plan['arm_joint_indices'], [native.index(i) for i in arm])):
                raise ValueError('IK plan belongs to a different package or robot. Run IK again.')
            base, q_plan = plan['base_se2'], plan['arm_joint_targets']
            pe, oe = plan['position_error_mm'], plan['orientation_error_deg']
    print('IK position p95 (mm)', np.quantile(pe, .95), flush=True)
    q[arm], q[fingers] = q_plan[0], targets['grip'][0]
    base = base.copy()
    base[:, 2] = np.unwrap(base[:, 2])
    commanded_base=base.copy()
    if planar:
        initial_xy=np.asarray(rg.R1_BASE_P)[:2]
        initial_yaw=Rotation.from_quat(np.asarray(rg.R1_BASE_Q)[[1,2,3,0]]).as_euler('xyz')[2]
        planar_offset=np.r_[initial_xy, initial_yaw]
        robot.set_root_pose(sapien.Pose())
        q[planar]=base[0]+planar_offset
    else:
        robot.set_root_pose(bp.se2_pose(*base[0]))
    robot.set_qpos(q)
    for joint, value in zip(joints, q):
        joint.set_drive_target(float(value))
    case = make_rigid_body(scene, package/'case_metric.obj', reference[0],
                           package/'material_0.png', material)
    # No hand attachment and no object pose writes after the initial spawn.
    steps = np.diff(np.floor(np.arange(n+1)*250/fps).astype(int))
    columns = {name: i for i, name in enumerate(
        ('right_gripper_finger_link1', 'right_gripper_finger_link2', 'table', 'chair', 'ground'))}
    impulses = np.zeros((n, len(columns)+1))
    bilateral = np.zeros(n, dtype=int)
    chair_only = np.zeros(n, dtype=int)
    furniture = np.zeros(n, dtype=int)
    unexpected_by_frame = []
    chair_vertical=np.zeros(n)
    wood_vertical=np.zeros(n)
    desk_vertical=np.zeros(n)
    contact_substep_rows=[]
    chair_threshold=.5*.08*9.81*steps/250
    wood_threshold=.5*.08*9.81*steps/250
    command_frames=np.arange(n)
    release_ok=None
    events=meta.get('multi_release_events', [])
    wood_event=next((event for event in events if event.get('event_id')=='wood_drop'), None)
    regrasp_event=next((event for event in events if event.get('event_id')=='wood_regrasp'), None)
    wood_release_ok=None
    wood_release_actual_gap=None
    regrasp_contact_source_frame=None
    regrasp_object_pose=None
    carried_contact_active=False
    contact_loss_source_frame=None
    contact_loss_phase=None
    contact_loss_grip=None
    if wood_event is not None:
        wood_request=int(wood_event['request_frame'])
        wood_window_end=int(wood_event['source_window'][1])
        wood_support=str(wood_event['support_actor_id'])
        wood_seat_index=int(np.searchsorted(frames, int(wood_event['seat_frame_source'])))
    if regrasp_event is not None:
        regrasp_start,regrasp_end=map(int,regrasp_event['source_window'])
    released=np.flatnonzero(targets['grip_states']>=5)
    # Packages truncated before any release command replay as pure carry.
    planned_release=int(released[0]) if len(released) else len(frames)
    if planned_release < 3:
        raise ValueError('Expected a paired approach, manipulation and release interval')
    actual_object = np.empty((n, 4, 4))
    actual_pad = np.repeat(np.eye(4)[None], n, axis=0)
    joint_state = np.empty((n, len(joints)))
    robot_links = {link.entity.get_name() for link in robot.get_links()}
    link = robot.find_link_by_name('right_gripper_link')
    vertices = np.loadtxt([line[2:] for line in case_collision_mesh(package/'case_metric.obj').read_text().splitlines()
                           if line.startswith('v ')], usecols=(0, 1, 2))
    chair_z=supports['chair']['center'][2]
    issued_q=q_plan.copy()
    issued_pad=target_T.copy()
    issued_grip=targets['grip'].copy()
    descent=[]
    if finish is not None:
        seat_index=int(np.searchsorted(frames,finish['seat_frame']))
        nominal_seat=target_T[min(int(seat_index), len(target_T)-1)].copy()
        seat_pad=nominal_seat.copy()
        seat_q=q_plan[seat_index].copy()

        def solve_seat(P,seed):
            saved=robot.get_qpos().copy()
            speed=robot.get_qvel().copy()
            scratch=saved.copy(); scratch[planar]=0.
            robot.set_qpos(scratch)
            b=base[seat_index]
            _,solution,_,_=lrt.solve_wbc_track(robot,'right',arm,lim,b[None,0],b[None,1],
                [(P[:3,3],P[:3,:3])],P[None,:3,3],bp.se2_pose,bp._gripper_pad_fk,
                w_pos=100.,w_ori=8.,w_posture=.10,w_limit=.5,base_range=.0002,yaw_max=.0002,
                ref_yaw=b[None,2],max_nfev=55,n_local=0,n_rand=0,base_step_max=.02,joint_step_max=4.,
                q_neutral=seed,torso_pitch=np.radians(pitch_limit-pitch_tracking_margin),
                initial_state=np.r_[0.,0.,0.,seed])
            robot.set_root_pose(sapien.Pose())
            robot.set_qpos(saved); robot.set_qvel(speed)
            return solution[0]

    for i, nsteps in enumerate(steps):
        if (wood_event is not None and wood_release_ok is None and i>=wood_request
                and i<=wood_window_end and i>=3
                and np.all((wood_vertical[i-3:i]>=wood_threshold[i-3:i])
                           &(bilateral[i-3:i]==steps[i-3:i])&(furniture[i-3:i]==0))
                and contact_loss_source_frame is None):
            wood_release_ok=i
            wood_release_actual_gap=float(joint_state[i-1,fingers].max())
        command=i
        alpha=0.
        if finish is not None and i>=planned_release:
            if release_ok is None and contact_loss_source_frame is None and np.all(
                    (chair_vertical[i-3:i]>=chair_threshold[i-3:i])
                    &(bilateral[i-3:i]==steps[i-3:i])&(furniture[i-3:i]==0)):
                release_ok=i
            if release_ok is None:
                command=planned_release
            else:
                # Catch up during withdrawal so F286–299 can follow their own
                # source timestamps after the physical release.
                since=i-release_ok
                if mobile_release is None:
                    u=np.clip((since-8)/max(1,275-release_ok-8),0,1)
                    source_time=min(n-1,planned_release+since+(release_ok-planned_release)*u*u*(3-2*u))
                else:
                    source_time = i
                command=int(source_time)
                alpha=source_time-command
        next_command=min(n-1,command+1)
        base_target=base[command]
        arm_target=(1-alpha)*q_plan[command]+alpha*q_plan[next_command]
        gripper_target=(1-alpha)*targets['grip'][command]+alpha*targets['grip'][next_command]
        pad_target=target_T[command].copy()
        if wood_event is not None and wood_release_ok is None and i>=wood_request:
            command=wood_request
            base_target=base[wood_seat_index]
            arm_target=q_plan[wood_seat_index]
            gripper_target=targets['grip'][wood_request-1].copy()
            pad_target=target_T[wood_seat_index].copy()
        elif wood_event is not None and wood_release_ok is not None and i<regrasp_start:
            elapsed=i-wood_release_ok
            if elapsed < open_frames:
                base_target=base[wood_seat_index]
                arm_target=q_plan[wood_seat_index]
                pad_target=target_T[wood_seat_index].copy()
                gripper_target=np.full(2,wood_release_actual_gap+(.045-wood_release_actual_gap)*min(1.,(elapsed+1)/open_frames))
            else:
                gripper_target=np.full(2,.045)
        if (regrasp_event is not None and wood_release_ok is not None
                and i>regrasp_end and regrasp_contact_source_frame is None):
            command=regrasp_end
            base_target=base[regrasp_end]
            arm_target=q_plan[regrasp_end]
            gripper_target=targets['grip'][regrasp_end].copy()
            pad_target=target_T[regrasp_end].copy()
        if finish is not None and i>=seat_index and (wood_event is None or
                (wood_release_ok is not None and regrasp_contact_source_frame is not None)):
            # Seat and lift IK are both solved at this station. Hold it while
            # issuing those arm targets, then blend both sets of commands together.
            if release_ok is None:
                base_target=base[seat_index]
                if i >= planned_release:
                    gripper_target = targets['grip'][planned_release-1].copy()
                if (i-seat_index)%2==0 and bilateral[i-1]>0 and chair_vertical[i-1]<chair_threshold[i-1]:
                    O=actual_object[i-1]; P=actual_pad[i-1]
                    gap=O[2,3]+np.min(vertices@O[2,:3])-chair_z
                    finger_gap=P[2,3]-chair_z-np.abs(P[2,:3])@np.array([.030,.005,.020])-abs(P[2,1])*.049
                    remaining=.020-(nominal_seat[2,3]-seat_pad[2,3])
                    step=max(0.,min(gap+.0003,finger_gap-.0015,remaining,.0018))
                    if step>1e-6:
                        seat_pad[2,3]-=step
                        seat_q=solve_seat(seat_pad,seat_q)
                        descent.append([int(frames[i]),gap*1000,(nominal_seat[2,3]-seat_pad[2,3])*1000])
                arm_target=seat_q
                pad_target=seat_pad.copy()
            elif i-release_ok <= open_frames:
                base_target=base[seat_index]
                if i==release_ok:
                    actual_gap=float(joint_state[i-1,fingers].max())
                    lift_pad=seat_pad.copy(); lift_pad[2,3]+=(mobile_release or {}).get('open_clearance_m', .004)
                    lift_q=solve_seat(lift_pad,seat_q)
                    for j in fingers:
                        joints[j].set_drive_property(stiffness=4000.,damping=120.,force_limit=12.)
                elapsed=i-release_ok
                gripper_target=np.full(2,actual_gap+(.045-actual_gap)*min(1.,(elapsed+1)/open_frames))
                u=min(1.,elapsed/lift_frames)
                arm_target=(1-u)*seat_q+u*lift_q
                pad_target=seat_pad.copy(); pad_target[2,3]=(1-u)*seat_pad[2,3]+u*lift_pad[2,3]
            elif i-release_ok <= open_frames+blend_frames:
                u=(i-release_ok-open_frames)/blend_frames
                base_target=(1-u)*base[seat_index]+u*base_target
                arm_target=(1-u)*lift_q+u*arm_target
                pad_target[:3,3]=(1-u)*lift_pad[:3,3]+u*pad_target[:3,3]
        if contact_loss_source_frame is not None:
            hold=contact_loss_source_frame
            command=hold
            base_target=commanded_base[hold].copy()
            arm_target=issued_q[hold].copy()
            pad_target=issued_pad[hold].copy()
            gripper_target=np.minimum(issued_grip[hold], contact_loss_grip)
        if release_ok is not None and contact_loss_source_frame is None:
            gripper_target=np.maximum(gripper_target, np.full(2, .045))
        command_frames[i]=command
        commanded_base[i]=base_target
        issued_q[i]=arm_target
        issued_pad[i]=pad_target
        issued_grip[i]=gripper_target
        for j, value in zip(arm, arm_target):
            joints[j].set_drive_target(float(value))
        for j, value in zip(fingers, gripper_target):
            joints[j].set_drive_target(float(value))
        previous = commanded_base[max(0,i-1)]
        if planar:
            previous = previous+planar_offset
            delta = base_target+planar_offset-previous
        else:
            delta = base_target-previous
        unexpected = set() if route is not None else None
        for sub in range(nsteps):
            if planar:
                for j,value,velocity in zip(planar,previous+delta*(sub+1)/nsteps,delta*250/nsteps):
                    joints[j].set_drive_target(float(value))
                    joints[j].set_drive_velocity_target(float(velocity))
            else:
                robot.set_root_pose(bp.se2_pose(*(previous+delta*(sub+1)/nsteps)))
            rg.step_sim(scene, robot)
            measured = np.zeros(impulses.shape[1])
            hit,loads,normals=measure_contacts(scene, measured, columns, supports, robot_links, unexpected)
            furniture[i] += hit
            chair_vertical[i] += loads.get('chair', 0.)
            wood_vertical[i] += loads.get('table', 0.)
            desk_vertical[i] += loads.get('desk', 0.)
            impulses[i] += measured
            bilateral_substep=bool(np.all(measured[:2] > 1e-6))
            bilateral[i] += int(bilateral_substep)
            chair_only[i] += int(measured[3] > 1e-6 and not np.any(measured[[0, 1, 2, 4, 5]] > 1e-6))
            def z_range(name):
                values=np.asarray(normals.get(name, ()), dtype=float)
                if values.size:
                    return float(values[:,2].min()),float(values[:,2].max())
                return float('nan'),float('nan')
            desk_zmin,desk_zmax=z_range('desk')
            wood_zmin,wood_zmax=z_range('table')
            white_zmin,white_zmax=z_range('chair')
            q_sub=robot.get_qpos()
            pitch_sub=np.degrees(lrt.torso_link_pitch(q_sub[arm[6:9]]))
            object_sub=case.get_pose().to_transformation_matrix()
            source_frame=int(frames[i])
            object_bottom_sub=float(np.min(vertices@object_sub[2,:3])+object_sub[2,3])
            if (source_frame>=24 and source_frame<wood_request if wood_event is not None
                    else source_frame>=24) and bilateral_substep:
                carried_contact_active=True
            if (regrasp_event is not None and regrasp_start<=source_frame<=regrasp_end
                    and bilateral_substep):
                carried_contact_active=True
            if wood_release_ok is not None and source_frame>=wood_release_ok and source_frame<regrasp_start:
                carried_contact_active=False
            if release_ok is not None and source_frame>=release_ok:
                carried_contact_active=False
            carry_phase=('table-to-wood' if source_frame>=42 and
                         (wood_event is None or source_frame<wood_request) else
                         'wood-to-white' if regrasp_contact_source_frame is not None and
                         source_frame>regrasp_contact_source_frame and source_frame<planned_release else None)
            support_present=any(loads.get(name,0.)>1e-6 for name in ('desk','table','chair'))
            expected_floor=(float(supports['desk']['center'][2]) if carry_phase=='table-to-wood'
                            else float(supports['table']['center'][2]) if carry_phase=='wood-to-white'
                            else -float('inf'))
            airborne=carry_phase is not None and not support_present and object_bottom_sub>expected_floor+.001
            if (airborne and carried_contact_active and not bilateral_substep
                    and contact_loss_source_frame is None):
                contact_loss_source_frame=source_frame
                contact_loss_phase=carry_phase
                contact_loss_grip=np.minimum(np.asarray(issued_grip[i]),robot.get_qpos()[fingers].copy())
            expected_support=('desk' if frames[i] < 42 else
                              'table' if wood_event is not None and wood_request <= frames[i] <= regrasp_end else
                              'chair' if frames[i] >= planned_release else 'airborne')
            contact_substep_rows.append([
                int(frames[i]),int(sub),int(nsteps),expected_support,'case',
                float(measured[0]),float(measured[1]),int(np.all(measured[:2] > 1e-6)),
                float(loads.get('desk',0.)),desk_zmin,desk_zmax,
                float(loads.get('table',0.)),wood_zmin,wood_zmax,
                float(loads.get('chair',0.)),white_zmin,white_zmax,
                *object_sub[:3,3].tolist(),*pitch_sub.tolist(),
                *q_sub[fingers].tolist(),
                *(q_sub[planar].tolist() if planar else [float('nan')]*3)
            ])
        if unexpected:
            unexpected_by_frame.append({'source_frame': int(frames[i]), 'pairs': sorted(unexpected)})
        actual_object[i] = case.get_pose().to_transformation_matrix()
        hand = link.get_entity_pose().to_transformation_matrix()
        actual_pad[i, :3, :3] = hand[:3, :3]
        actual_pad[i, :3, 3] = hand[:3, 3] + rg.PAD_FWD*hand[:3, 0]
        joint_state[i] = robot.get_qpos()
        if (regrasp_event is not None and regrasp_contact_source_frame is None
                and regrasp_start <= int(frames[i]) <= regrasp_end and bilateral[i] > 0):
            regrasp_contact_source_frame=int(frames[i])
            regrasp_object_pose=actual_object[i].copy()

    if planar:
        measured_base=joint_state[:,planar]
        base=np.c_[measured_base[:, :2]-initial_xy,measured_base[:,2]-initial_yaw]

    target_T=issued_pad
    target_p=target_T[:,:3,3]
    pair_pad=np.linalg.inv(target_T)@reference[command_frames]
    executed_grip=issued_grip

    actual_pair = np.linalg.inv(actual_pad) @ actual_object
    pd = np.linalg.norm(actual_pair[:, :3, 3] - pair_pad[:, :3, 3], axis=1)
    pad_err = np.linalg.norm(actual_pad[:, :3, 3] - target_p, axis=1)
    moving = np.r_[False, np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1) > .001]
    air = (impulses[:, 2:5] <= 1e-6).all(1)
    if route is not None:
        air &= impulses[:, -1] <= 1e-6
    carried = air & moving & (bilateral > 0)
    object_bottom = np.min(vertices @ actual_object[:, 2, :3].T, axis=0) + actual_object[:, 2, 3]
    table_z, chair_z = (supports[name]['center'][2] for name in ('table', 'chair'))
    lifted = air & (bilateral > 0) & (object_bottom > table_z+.001)
    released=(targets['grip_states']>=5) if finish is None else (
        np.arange(n)>=release_ok if release_ok is not None else np.zeros(n,dtype=bool))
    stable = np.flatnonzero((chair_only == steps) & released)
    tail = stable[stable >= n-15]
    place_error = np.linalg.norm(actual_object[-1, :2, 3] - reference[-1, :2, 3])
    release_frames = np.flatnonzero(targets['grip_states'] >= 5)
    if not len(release_frames) or release_frames[0] < 3:
        raise ValueError('Expected a paired approach, manipulation and release interval')
    release = release_frames[0]
    stable_frames = np.zeros(n, dtype=bool)
    stable_frames[stable] = True
    trailing_chair_start = int(np.flatnonzero(~stable_frames)[-1])+1 if (~stable_frames).any() else 0
    bilateral_frames = frames[bilateral > 0]
    lifted_frames = frames[lifted]
    chair_frames = frames[impulses[:, 3] > 1e-6]
    last_position = actual_object[-1, :3, 3]
    drift = float(np.linalg.norm(actual_object[-15:, :3, 3] - last_position, axis=1).max())
    pitch = np.max(np.abs(np.degrees(lrt.torso_link_pitch(joint_state[:, arm[6:9]]))), axis=0)
    supported_at_open = bool(np.all(impulses[release-3:release, 3] > 1e-6))
    measured_open=None
    observed_open=None
    if finish is not None:
        supported_at_open=release_ok is not None
        before_request = np.median(joint_state[release-3:release, fingers], axis=0)
        observed = np.flatnonzero((np.arange(n)>=release)&np.all(joint_state[:, fingers]>before_request+.001,axis=1))
        observed_open=int(frames[observed[0]]) if len(observed) else None
        if release_ok is not None:
            before=np.median(joint_state[release_ok-3:release_ok,fingers],axis=0)
            opened=np.flatnonzero((np.arange(n)>=release_ok)&np.all(joint_state[:,fingers]>before+.001,axis=1))
            measured_open=int(frames[opened[0]]) if len(opened) else None
    before_release = np.flatnonzero(
        np.linalg.norm(base[:release, :2]-base[release, :2], axis=1) <= .05)
    settled_before_release = np.split(before_release, np.flatnonzero(np.diff(before_release) > 1)+1)[-1]
    lead_frames = len(settled_before_release) if release-1 in settled_before_release else 0
    contact_fit = meta.get('contact_refinement', {}).get('target_jaw_gap_max_mm')
    geometry_contact_ok = (contact_fit <= 10 if contact_fit is not None else
                           meta['carry_frames_both_within_10mm'] == meta['carry_frames'])
    result = dict(source_first=int(frames[0]), source_last=int(frames[-1]),
                  frames=n, physics_substeps=int(steps.sum()), physical_time_s=float(steps.sum()/250),
                  base_travel_m=float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).sum()),
                  ik_position_p95_mm=float(np.quantile(pe, .95)),
                  ik_orientation_p95_deg=float(np.quantile(oe, .95)),
                  executed_pad_position_p95_mm=float(np.quantile(pad_err, .95)*1000),
                  torso_limit_deg=pitch_limit.tolist(),
                  max_torso_pitch_deg=pitch.tolist(),
                  base_within_50mm_of_release_position_before_release_frames=lead_frames,
                  bilateral_contact_frames=int((bilateral > 0).sum()),
                  first_bilateral_source_frame=int(bilateral_frames[0]) if len(bilateral_frames) else None,
                  first_lift_with_bilateral_source_frame=int(lifted_frames[0]) if len(lifted_frames) else None,
                  lift_with_bilateral_frames=int(lifted.sum()),
                  first_chair_contact_source_frame=int(chair_frames[0]) if len(chair_frames) else None,
                  reference_release_source_frame=int(frames[release]),
                  f0_actual_bottom_above_table_mm=float((object_bottom[0]-table_z)*1000),
                  final_actual_bottom_above_chair_mm=float((object_bottom[-1]-chair_z)*1000),
                  continuous_chair_only_after_release_s=float(steps[trailing_chair_start:].sum()/250),
                  chair_contact_before_release=supported_at_open,
                  final_15_frame_drift_mm=1000*drift,
                  bilateral_carry_while_moving_frames=int(carried.sum()),
                  pair_translation_p95_mm=float(np.quantile(pd, .95)*1000),
                  chair_only_final_s=float(steps[tail].sum()/250),
                  ground_contact_frames=int((impulses[:, 4] > 1e-6).sum()),
                  robot_furniture_contact_frames=int((furniture > 0).sum()),
                  final_place_xy_error_mm=float(place_error*1000),
                  source_asset_gate_pass=(geometry_contact_ok
                                          and abs(meta['initial_case_bottom_above_table_mm']) <= 10
                                          and abs(meta['final_case_bottom_above_chair_mm']) <= 10))
    result['training_data_pass'] = bool(result['source_asset_gate_pass'] and carried.any()
                                        and len(tail) == 15 and place_error <= .035
                                        and supported_at_open and drift <= .003 and np.all(pitch <= pitch_limit)
                                        and result['ground_contact_frames'] == 0
                                        and result['robot_furniture_contact_frames'] == 0)
    if wood_event is not None:
        result.update(wood_release_support_actor_id=wood_support,
                      wood_release_authorized_source_frame=wood_release_ok,
                      carry_contact_loss_source_frame=contact_loss_source_frame,
                      carry_contact_loss_phase=contact_loss_phase,
                      wood_load_before_release_ns=None if wood_release_ok is None else
                          wood_vertical[max(0,wood_release_ok-3):wood_release_ok].tolist(),
                      wood_object_actor_id=wood_event.get('object_actor_id','case'),
                      wood_source_release_window=wood_event['source_window'],
                      wood_regrasp_contact_source_frame=regrasp_contact_source_frame,
                      wood_regrasp_object_actor_id=(regrasp_event or {}).get('object_actor_id','case'),
                      wood_support_impulse_frames=int(np.sum(wood_vertical[regrasp_start:planned_release] > 1e-6)),
                      desk_support_impulse_frames=int(np.sum(desk_vertical[:wood_request] > 1e-6)))
        if wood_release_ok is not None and regrasp_contact_source_frame is not None:
            a=int(np.searchsorted(frames,wood_release_ok))
            b=int(np.searchsorted(frames,regrasp_contact_source_frame))
            result['wood_regrasp_pose_change_mm']=float(np.linalg.norm(actual_object[b,:3,3]-actual_object[a,:3,3])*1000)
        result['training_data_pass'] &= (wood_release_ok is not None and
                                           regrasp_contact_source_frame is not None and
                                           contact_loss_source_frame is None)
    if mobile_release is not None:
        result.update(placement_support=mobile_release['support_name'],
                      measured_open_source_frame=measured_open,
                      observed_finger_separation_after_request_source_frame=observed_open,
                      release_authorized_source_frame=None if release_ok is None else int(frames[release_ok]),
                      placement_load_before_release_ns=None if release_ok is None else chair_vertical[release_ok-3:release_ok].tolist(),
                      measured_descent_corrections=descent)
        result['training_data_pass'] &= measured_open is not None
    elif finish is not None:
        idx=int(np.searchsorted(frames,finish['base_arrival_frame']))
        hold=slice(idx,planned_release)
        old=Path(finish['frozen_run'])
        if not old.is_absolute():
            old=Path(__file__).parent/old
        cut=slice(0,finish['frozen_ik_through']+1)
        with np.load(old/'refined_replay/exec_trace.npz') as frozen:
            prefix_error=float(np.max(np.linalg.norm(actual_pad[cut,:3,3]-frozen['actual_pad'][cut,:3,3],axis=1))*1000)
        result.update(frozen_front_pad_replay_max_mm=prefix_error,
                      measured_open_source_frame=measured_open,
                      release_authorized_source_frame=None if release_ok is None else int(frames[release_ok]),
                      chair_load_before_release_ns=None if release_ok is None else chair_vertical[release_ok-3:release_ok].tolist(),
                      base_hold_drift_mm=float(np.linalg.norm(base[hold,:2]-commanded_base[hold,:2],axis=1).max()*1000),
                      base_hold_yaw_error_deg=float(np.degrees(np.abs(base[hold,2]-commanded_base[hold,2]).max())),
                      base_turn_deg=float(np.degrees(commanded_base[idx,2]-commanded_base[finish['frozen_ik_through'],2])),
                      measured_descent_corrections=descent,
                      tail_source_commands_executed=int(np.sum(command_frames[286:]>=286)))
        result['training_data_pass'] &= measured_open is not None and result['tail_source_commands_executed']==14
    if retargeted:
        precontact = np.cumprod(contact_weight == 0).astype(bool)
        actual_rotation = np.swapaxes(source_pad[:, :3, :3], 1, 2) @ actual_pad[:, :3, :3]
        retarget_pos_mm = np.linalg.norm(actual_pad[:, :3, 3]-source_pad[:, :3, 3], axis=1)*1000
        retarget_angle_deg = np.degrees(Rotation.from_matrix(actual_rotation).magnitude())
        executed_grip_mm = np.max(np.abs(joint_state[:, fingers]-source_grip), axis=1)*1000
        if np.any(precontact):
            result['pre_contact_retarget'] = dict(frames=int(precontact.sum()),
                first_source_frame=int(frames[0]), last_source_frame=int(frames[precontact][-1]),
                executed_position_p95_mm=float(np.quantile(retarget_pos_mm[precontact], .95)),
                executed_orientation_p95_deg=float(np.quantile(retarget_angle_deg[precontact], .95)),
                executed_grip_joint_p95_mm=float(np.quantile(executed_grip_mm[precontact], .95)))
        np.savetxt(out/'retarget_frames.csv',np.c_[frames,contact_weight,source_pad[:, :3, 3],
                    target_p,actual_pad[:, :3, 3],retarget_pos_mm,retarget_angle_deg,executed_grip_mm],
                   delimiter=',',fmt=['%d']+['%.6f']*13,comments='',
                   header='source_frame,contact_blend,retarget_x,retarget_y,retarget_z,target_x,target_y,target_z,'
                          'actual_x,actual_y,actual_z,executed_retarget_position_error_mm,'
                          'executed_retarget_rotation_error_deg,executed_retarget_grip_error_mm')
    trace_path = out/'exec_trace.npz'
    np.savez_compressed(trace_path, source_frames=frames, dt=1/250, package_sha256=digest,
                        physics_substeps=steps, target_pad=target_T, actual_pad=actual_pad,
                        target_gripper_object=pair_pad, actual_gripper_object=actual_pair,
                        target_object=reference, actual_object=actual_object,
                        contact_impulses=impulses, contact_columns=np.array([*columns, 'other']),
                        chair_vertical_impulse=chair_vertical,chair_load_threshold=chair_threshold,command_frames=command_frames,
                        bilateral=bilateral, chair_only=chair_only, base_se2=base,base_se2_target=commanded_base,
                        arm_joint_indices=np.asarray([native.index(i) for i in arm]), arm_joint_targets=issued_q,
                        finger_joint_indices=np.asarray([native.index(i) for i in fingers]), finger_joint_targets=executed_grip,
                        q_actual=joint_state[:,native])
    csv_columns = ['source_frame', 'physics_substeps', 'bilateral_substeps', 'chair_only_substeps',
                   'lift_with_bilateral', 'moving_bilateral_airborne', 'case_x_m', 'case_y_m',
                   'case_z_m', 'case_bottom_z_m', 'pad_error_mm', 'robot_furniture_substeps']
    csv_columns += [f'{name}_impulse_ns' for name in (*columns, 'other')]
    np.savetxt(out/'physics_frames.csv', np.c_[frames, steps, bilateral, chair_only, lifted, carried,
                actual_object[:, :3, 3], object_bottom, pad_err*1000, furniture, impulses],
               fmt=['%d']*6+['%.8f']*5+['%d']+['%.9g']*impulses.shape[1], delimiter=',',
               header=','.join(csv_columns), comments='')
    if contact_substep_rows:
        columns_sub=[
            'source_frame','physics_substep','physics_substeps_in_frame','intended_support_actor_id','object_actor_id',
            'left_pad_impulse_Ns','right_pad_impulse_Ns','bilateral_same_substep',
            'desk_support_vertical_impulse_Ns','desk_normal_z_min','desk_normal_z_max',
            'wood_support_vertical_impulse_Ns','wood_normal_z_min','wood_normal_z_max',
            'white_support_vertical_impulse_Ns','white_normal_z_min','white_normal_z_max',
            'object_x_m','object_y_m','object_z_m','torso_pitch_1_deg','torso_pitch_2_deg','torso_pitch_3_deg',
            'finger_joint_1_rad','finger_joint_2_rad','planar_x','planar_y','planar_yaw']
        with (out/'contact_substeps.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(columns_sub);writer.writerows(contact_substep_rows)
    if route is not None:
        (out/'robot_contact_audit.json').write_text(json.dumps(unexpected_by_frame, indent=2)+'\n')
        result['unexpected_robot_contact_frames'] = len(unexpected_by_frame)
        result['training_data_pass'] &= not unexpected_by_frame
    if meta.get('ik_only_diagnostic'):
        result['ik_only_diagnostic'] = True
    if full_scene is not None:
        result['ras_final_glb_sha256'] = full_scene['ras_final_glb_sha256']
    result['package_sha256'] = digest
    result['trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if wood_event is not None:
        (out/'cycle_events.json').write_text(json.dumps({
            'wood_drop': {'source_window': wood_event['source_window'],
                          'support_actor_id': wood_support,
                          'seat_frame_source': wood_seat_index and int(frames[wood_seat_index]),
                          'object_actor_id': wood_event.get('object_actor_id','case'),
                          'authorized_source_frame': wood_release_ok,
                          'pre_open_support_vertical_impulse_Ns': None if wood_release_ok is None else
                              wood_vertical[max(0,wood_release_ok-3):wood_release_ok].tolist()},
            'wood_regrasp': {'source_window': regrasp_event['source_window'],
                             'support_actor_id': regrasp_event.get('support_actor_id','table'),
                             'object_actor_id': regrasp_event.get('object_actor_id','case'),
                             'first_bilateral_source_frame': regrasp_contact_source_frame,
                             'object_pose_unchanged_without_reset': regrasp_contact_source_frame is not None},
            'white_drop': {'source_window': (570,576), 'support_actor_id':'chair',
                           'object_actor_id':'case',
                           'authorized_source_frame': result.get('release_authorized_source_frame'),
                           'pre_open_support_vertical_impulse_Ns': result.get('placement_load_before_release_ns')}},
            indent=2)+'\n')
    (out/'verdict.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if args.render:
        render_trace(package, out, supports, fps, frames, digest, trace_path,scene_asset,scene_meshes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base-lead', type=float, default=.35)
    parser.add_argument('--source-window', type=int, nargs=2, metavar=('FIRST', 'LAST'))
    parser.add_argument('--base-station', type=float, nargs=3, metavar=('X_M', 'Y_M', 'YAW_DEG'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--plan', type=Path)
    mode.add_argument('--render-trace', type=Path)
    mode.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if (args.source_window is not None or args.base_station is not None) and not args.plan_only:
        parser.error('Source windows and stationary base candidates require --plan-only')
    replay(args)

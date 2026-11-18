"""456483158 source-timed, gravity-active tabletop to chair transfer."""
from pathlib import Path
import argparse
import json
import sys
import shutil
import struct

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import cb_large_range_sim as cb
from grasp_pose_audit import angles, source_directions, unit

rg, bp = cb.rg, cb.bp
SOURCE = Path('/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158')
SAM3D = Path('/path/to/project_data/mano_ws/AGILE/output/456483158/generated/object')
URDF_POSE = Rotation.from_quat([0, 0, np.sqrt(.5), np.sqrt(.5)])
REFERENCE_WINDOW = (75, 95)
FINGER_HALF = np.array([.030,.005,.020])
SUPPORT_CLEARANCE = .003


def setup(out):
    obj = np.load(SOURCE / 'object_instance.npz')
    gravity = np.load(SOURCE / 'gravity_align.npz')['M_sim']
    start = obj['surface_pick_V'] @ gravity.T
    end = obj['surface_place_V'] @ gravity.T
    a, b = start.mean(0), end.mean(0)
    forward = b[:2] - a[:2]
    forward /= np.linalg.norm(forward)
    rotation = np.eye(3)
    rotation[:2,:2] = [[forward[1], -forward[0]], forward]
    # The camera map does not include a calibrated floor. Use the recovered
    # support planes and place the source table at the R1 working height.
    scale = 1.0
    translation = np.r_[-rotation[:2,:2] @ a[:2], .78-a[2]]
    transform = lambda x: scale * (x @ gravity.T @ rotation.T) + translation
    table = transform(obj['surface_pick_V'])
    chair = transform(obj['surface_place_V'])
    scene_pts = np.load(SOURCE/'pointcloud/scene_cloud.npz')
    points = transform(scene_pts['points'])
    colors = scene_pts['colors']
    centers = np.array([table.mean(0), chair.mean(0)])
    (out/'geometry.json').write_text(json.dumps({
        'supports':centers.tolist(), 'gravity_to_sim':gravity.tolist(),
        'heading':rotation.tolist(), 'translation':translation.tolist(),
        'scale':scale, 'height_anchor':'nominal table 0.78m; source support height difference retained',
        'objects':'dynamic box proxy, static tabletop and chair-seat proxies'},indent=2))
    return centers, points, colors


def interp(frames, points, length=300):
    result = np.empty((length,*points.shape[1:]))
    for i,j,p,q in zip(frames[:-1],frames[1:],points[:-1],points[1:]):
        u = np.linspace(0.,1.,j-i+1)
        w = (u*u*(3-2*u)).reshape((-1,) + (1,)*(points.ndim-1))
        result[i:j+1] = p + w*(q-p)
    return result


def drive(robot, targets, bases, selected, q0, yaw, orientation, seeds_count=20, torso_plan_deg=20., z_weight=1.):
    q_min, q_max = bp._limits('right').T
    torso_limits=np.array([robot.get_active_joints()[i].get_limits()[0] for i in selected[6:]])
    tl, th=torso_limits.T
    pitch=np.full(3,np.deg2rad(torso_plan_deg))
    lo, hi = np.r_[q_min+.015,[0.,0.,0.]], np.r_[q_max-.015,[1.,1.,1.]]
    torso0=q0[6:] if len(q0)==9 else np.zeros(3)
    q0=np.r_[q0[:6],cb.lrt.torso_pitch_coordinates(torso0,tl,th,pitch,encode=True)]
    def decode(x):
        return np.r_[x[:6],cb.lrt.torso_pitch_coordinates(x[6:],tl,th,pitch)]
    rz = Rotation.from_euler('z',yaw)
    base_q = np.broadcast_to((rz*URDF_POSE).as_quat(),(len(targets),4))[:,[3,0,1,2]]
    orientations=np.broadcast_to(orientation,(len(targets),3,3))
    weights=np.broadcast_to(z_weight,(len(targets),))
    rng = np.random.default_rng(5)
    solutions=[]
    errors=[]
    for k,(target,root) in enumerate(zip(targets,bases)):
        robot.set_root_pose(sapien.Pose(p=[*root,0],q=base_q[k]))
        orientation=Rotation.from_matrix(orientations[k]).inv()
        position_weight=np.array([1.,1.,weights[k]])
        def residual(q):
            joints=decode(q)
            p,R=bp._gripper_pad_fk(robot,'right',selected,joints)
            angle=(orientation*Rotation.from_matrix(R)).as_rotvec()
            return np.r_[(p-target)*position_weight,.12*angle,.0025*(q-q0),.025*joints[6:]]
        def jac(q):
            return np.column_stack([(residual(q+h)-residual(q-h))/.0002
                                    for h in np.eye(len(q))*.0001])
        seeds=[q0] if k else [q0,*rng.uniform(lo,hi,(seeds_count,len(lo)))]
        candidates=[least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),jac=jac,
                                 max_nfev=100,ftol=1e-6) for seed in seeds]
        best=min(candidates,key=lambda fit:fit.cost)
        err=residual(best.x)
        solutions.append(decode(best.x))
        errors.append([np.linalg.norm(err[:3]/position_weight),np.linalg.norm(err[3:6])/.12])
        q0=best.x
    return np.asarray(solutions),np.asarray(errors),base_q


def plan_grasp(box, object_R, source_x, source_y, source_spread, angles_deg=None):
    import trimesh

    palm_file=Path(bp.R1_URDF).parent/'meshes/right_gripper_link_collision.STL'
    palm=np.asarray(trimesh.load_mesh(palm_file).vertices)-[rg.PAD_FWD,0.,0.]
    up=np.array([0.,0.,1.])
    ref_y=unit(source_y*np.array([1.,1.,0.]))
    # Give the arm room above the tabletop while keeping a tracking budget
    # inside the observed source cone (audit allowance is ten degrees).
    elevation=np.arcsin(source_x[2])-np.sign(source_x[2])*np.deg2rad(.6*(source_spread[0]+10.))
    candidates=[]
    for axis in (0,1):
        normal=object_R[:,axis]
        toward=ref_y if ref_y@normal>=0 else -ref_y
        turn=np.arctan2(np.cross(normal,toward)[2],normal@toward)
        # Keep both pads near face normal to reduce the twisting moment from
        # the box corners touching before the side faces.
        max_turn=np.arctan(.20*box[1-axis]/box[axis])
        yaw_options=np.linspace(0,np.clip(turn,-max_turn,max_turn),5)
        if angles_deg is not None:
            yaw_options=[np.sign(turn)*min(max_turn,np.deg2rad(angles_deg[0]))]
        for yaw in yaw_options:
            y=Rotation.from_rotvec(yaw*up).apply(normal)
            if np.cross(y,up)@source_x<0:
                y=-y
            forward=np.cross(y,up)
            tilt_options=np.linspace(0,elevation,7) if angles_deg is None else [np.deg2rad(angles_deg[1])]
            for tilt in tilt_options:
                x=np.cos(tilt)*forward+np.sin(tilt)*up
                R=np.column_stack((x,y,np.cross(x,y)))
                finger_drop=np.abs(R[2])@FINGER_HALF+.000121
                palm_drop=-(palm@R[2]).min()
                pad_height=max(finger_drop,palm_drop)+SUPPORT_CLEARANCE
                local_y=object_R.T@y
                chord=box[axis]/abs(local_y[axis])
                # Find the common vertical interval of the two projected face points
                # and the finite pad rectangles, with a 2 mm border on each pad.
                center=R[:,2]*(pad_height-box[2])/R[2,2]
                lower,upper=-box[2]+.002,box[2]-.002
                for column in (0,2):
                    coef=R[2,column]
                    if abs(coef)>1e-8:
                        extent=(FINGER_HALF[column]-.002)/abs(coef)
                        mid=(center@R[:,column])/coef
                        lower=max(lower,mid-extent)
                        upper=min(upper,mid+extent)
                width=2*(np.abs(local_y)@box)
                if upper<lower or width>2*(.044-.001547):
                    continue
                # Use the highest feasible patch for the side approach, away from the table.
                contact=np.stack((-chord*local_y,chord*local_y))
                contact[:,2]=upper
                error=np.array([angles(x,source_x),angles(y,source_y,unsigned=True)])
                score=error@error
                candidates.append((score,R,forward,pad_height,axis,contact,error,finger_drop,palm_drop,center))
    if not candidates:
        raise RuntimeError('No opposing face pair clears the table within the R1 jaw geometry')
    _,R,forward,height,axis,contact,error,finger_drop,palm_drop,center=min(candidates,key=lambda item:item[0])
    details=dict(box_half_size_m=box.tolist(),object_R_world=object_R.tolist(),face_axis=int(axis),
                 face_points_object_m=contact.tolist(),approach_object=(object_R.T@R[:,0]).tolist(),
                 jaw_axis_object=(object_R.T@R[:,1]).tolist(),
                 pad_rotation_world=R.tolist(),pad_height_above_support_m=height,
                 fingertip_gap_m=height-finger_drop,palm_gap_m=height-palm_drop,
                 pad_minus_case_center_world_m=center.tolist(),source_error_deg=error.tolist(),
                 reference_window=list(REFERENCE_WINDOW))
    return R,forward,height,details


def select_stations(scene, robot, arm, start, end, forward, rotations, target, reach, q_seed, fixed_pick=None):
    # Keep the entire translation on the outside of the table-to-chair aisle.
    aisle=unit(end[:2]-start[:2])
    outside=np.array([aisle[1],-aisle[0]])
    if outside@forward[:2]>0:
        outside=-outside
    heading=np.arctan2(-outside[1],-outside[0])
    approach=-outside
    initial_q=robot.get_qpos()
    robot_links={link.entity.get_name() for link in robot.get_links()}
    furniture={'white_table','grey_chair_seat','support_leg'}

    def colliding_links(station, root_quats, joint_plan):
        collisions=set()
        for q_frame,quat in zip(joint_plan,root_quats):
            q=initial_q.copy()
            q[arm]=q_frame
            robot.set_root_pose(sapien.Pose(p=[*station,0.],q=quat))
            robot.set_qpos(q)
            robot.set_qvel(np.zeros_like(q))
            for joint,theta in zip(robot.get_active_joints(),q):
                joint.set_drive_target(float(theta))
            rg.step_sim(scene,robot)
            for contact in scene.get_contacts():
                a,b=(body.get_entity().get_name() for body in contact.bodies)
                if ((a in robot_links and b in furniture) or (b in robot_links and a in furniture)) and any(
                        np.linalg.norm(point.impulse)>1e-6 for point in contact.points):
                    collisions.add('/'.join(sorted((a,b))))
        return sorted(collisions)

    def probe(center, frames, y_offsets, yaw_offsets, important, z_weight):
        candidates=[]
        for x in (reach,reach+.05):
            for y in y_offsets:
                station=center[:2]+x*outside+y*aisle
                bases=np.broadcast_to(station,(len(frames),2))
                for yaw in heading-np.pi/2+np.deg2rad(yaw_offsets):
                    joints,err,quats=drive(robot,target[frames],bases,arm,q_seed,yaw,
                                           rotations[frames],seeds_count=7,torso_plan_deg=17.,z_weight=z_weight)
                    collisions=colliding_links(station,quats,joints)
                    limits=bp._limits('right')
                    margin=np.min(np.minimum(joints[:,:6]-limits[:,0],limits[:,1]-joints[:,:6]))
                    weighted=np.linalg.norm(err*[1.,.12],axis=1)
                    score=float(weighted[important]+.25*weighted.max())
                    report=dict(reach_m=x,base_yaw_deg=float(np.degrees(yaw+np.pi/2)),
                                xy_m=station.tolist(),probe_frames=frames,ik_errors_m_rad=err.tolist(),
                                min_arm_limit_margin_deg=float(np.degrees(margin)),
                                furniture_collisions=collisions,score=score)
                    candidates.append(((len(collisions),score),station,yaw,report))
        return min(candidates,key=lambda item:item[0]),[item[3] for item in candidates]

    if fixed_pick is None:
        pick,pick_candidates=probe(start,[47,71,125],(-.25,-.20,-.142),(-10,20,45),1,1.)
        place,place_candidates=probe(end,[162,190,219],(-.20,-.142,-.08),(0,25,50),2,[1.,1.,2.])
    else:
        place,place_candidates=probe(end,[190,236,274],(-.20,-.142,-.08),(0,25,50),1,[1.,4.,1.])
        return place[1],place[2],dict(place=place[3],place_candidates=place_candidates)
    _,a,yaw_a,_=pick
    _,b,yaw_b,_=place
    base_frames=[0,47,125,162,244,274,299]
    bases=interp(base_frames,np.array([a-.10*approach,a,a,b,b,b-.07*approach,b-.07*approach]))
    yaw=interp([0,125,162,299],np.array([yaw_a,yaw_a,yaw_b,yaw_b]))
    return bases,yaw,dict(pick=pick[3],place=place[3],pick_candidates=pick_candidates,
                          place_candidates=place_candidates)


def join_measured_pickup(scene,robot,arm,case,start,end,box,forward,pickup_trace,target,plan):
    pickup_verdict=json.loads((pickup_trace/'verdict.json').read_text())
    with np.load(pickup_trace/'exec_trace.npz') as trace:
        if not np.array_equal(trace['source_frame'],np.arange(126)):
            raise ValueError('Expected the complete fixed-base pickup trial through F125')
        pose=trace['object_pose'][-1]
        object_at_lift=sapien.Pose(p=pose[:3],q=pose[3:]).to_transformation_matrix()
        pad_at_lift=np.eye(4)
        pad_at_lift[:3,:3]=trace['pad_R_world'][-1]
        pad_at_lift[:3,3]=trace['pad_actual'][-1]
        fixed_base=trace['base_xy']
        fixed_quat=trace['base_quat']
        fixed_q=trace['q_arm_target']
        fixed_errors=trace['ik_err']
        target[:126]=trace['pad_target']
    if not pickup_verdict['pass']:
        raise ValueError('Fixed-base trial failed contact, retention or clearance checks')
    held=np.linalg.solve(pad_at_lift,object_at_lift)
    # Keep the object at lift height while the base travels, then bring the
    # measured held orientation to a flat support pose over the chair.
    flat=np.eye(4)
    axis=object_at_lift[:3,0]
    flat[:3,:3]=Rotation.from_euler('z',np.arctan2(axis[1],axis[0])).as_matrix()
    inverse_held=np.linalg.inv(held)
    jaw_axis=(flat@inverse_held)[:3,1]
    carry=pad_at_lift.copy()
    carry[:2,3]+=end[:2]-object_at_lift[:2,3]
    waypoints=[0,125,133,168,190,207,223,236,245,274,299]
    progress=interp([0,168,190,299],np.array([0.,0.,1.,1.]))
    original_pose=case.get_pose()
    case.set_pose(sapien.Pose(p=[2.,2.,2.]))
    approaches=[]
    # Compare a flat set-down and two edge-first set-downs. The chair gate still
    # requires measured support, and the released case must settle on its face.
    for tilt in (0.,20.,28.,34.,38.):
        seat=flat.copy()
        seat[:3,:3]=Rotation.from_rotvec(np.deg2rad(tilt)*jaw_axis).as_matrix()@flat[:3,:3]
        seat[:3,3]=end+[0,0,np.abs(seat[2,:3])@box-.0004]
        goal=seat@inverse_held
        hover=goal[:3,3]+[0,0,.145]
        retreat=goal[:3,3]-.105*forward+[0,0,.14]
        planned=interp(waypoints,np.array([target[0],pad_at_lift[:3,3],pad_at_lift[:3,3],carry[:3,3],hover,
                                           goal[:3,3]+[0,0,.030],goal[:3,3]+[0,0,.006],
                                           goal[:3,3],goal[:3,3],retreat,retreat]))
        transition=Rotation.from_matrix(goal[:3,:3]@pad_at_lift[:3,:3].T).as_rotvec()
        rotations=Rotation.from_rotvec(progress[:,None]*transition).as_matrix()@pad_at_lift[:3,:3]
        station,yaw_b,stations=select_stations(scene,robot,arm,start,end,forward,rotations,planned,
                                             .29,fixed_q[-1],fixed_pick=plan['fixed_pick'])
        candidate=stations['place']
        # Include the open finger spread in the lowest predicted fingertip.
        clearance=goal[2,3]-end[2]-np.abs(goal[2,:3])@FINGER_HALF-abs(goal[2,1])*(.044+.003453)
        candidate['nominal_finger_clearance_mm']=float(clearance*1000)
        key=(len(candidate['furniture_collisions']),clearance<.001,candidate['score']+tilt*.0003)
        approaches.append((key,tilt,seat,goal,planned,rotations,station,yaw_b,stations))
    _,tilt,seat,goal,planned,rotations,station,yaw_b,stations=min(approaches,key=lambda item:item[0])
    target[126:]=planned[126:]
    case.set_pose(original_pose)
    body=next(comp for comp in case.get_components() if hasattr(comp,'set_disable_gravity'))
    body.linear_velocity=np.zeros(3)
    body.angular_velocity=np.zeros(3)
    yaw_a=np.deg2rad(plan['fixed_pick'][2])-np.pi/2
    aisle=unit(end[:2]-start[:2])
    outside=np.array([aisle[1],-aisle[0]])
    if outside@forward[:2]>0:
        outside=-outside
    # The front wheels straddle the fixture legs at the pickup yaw. Move the
    # near wheels further inside the leg line before starting along the aisle.
    lane=(fixed_base[-1]-start[:2])@outside-.030
    lane_start=fixed_base[-1]+(lane-(fixed_base[-1]-start[:2])@outside)*outside
    lane_end=station+(lane-(station-start[:2])@outside)*outside
    reference=interp([0,125,133,168,299],np.array([fixed_base[-1],fixed_base[-1],lane_start,lane_end,lane_end]))
    reference_yaw=np.full(300,yaw_a)

    def se2(x,y,yaw):
        quat=(Rotation.from_euler('z',yaw)*URDF_POSE).as_quat()[[3,0,1,2]]
        return sapien.Pose(p=[x,y,0.],q=quat)

    indices=slice(126,169)
    joints=robot.get_active_joints()
    limits=np.vstack((bp._limits('right'),[joints[i].get_limits()[0] for i in arm[6:]]))
    base,transit_q,pe,oe=cb.lrt.solve_wbc_track(
        robot,'right',arm,limits,reference[indices,0],reference[indices,1],
        list(zip(target[indices],rotations[indices])),target[indices],se2,bp._gripper_pad_fk,
        w_pos=60.,w_ori=4.,w_base=6.,w_yaw=4.,w_posture=.10,w_limit=.5,w_cont=2.,
        base_range=.012,yaw_max=np.deg2rad(4.),ref_yaw=reference_yaw[indices],
        n_local=1,n_rand=0,max_nfev=70,base_step_max=.018,joint_step_max=5.,
        q_neutral=fixed_q[-1],torso_pitch=np.full(3,np.deg2rad(17.)),
        initial_state=np.r_[0.,0.,0.,fixed_q[-1]])
    yaw_delta=np.angle(np.exp(1j*(base[-1,2]-yaw_b)))
    withdraw=station-.07*forward[:2]
    tail=interp([0,168,190,245,274,299],np.array([base[-1,:2],base[-1,:2],station,station,withdraw,withdraw]))
    tail_yaw=interp([0,168,190,299],np.array([yaw_b+yaw_delta,yaw_b+yaw_delta,yaw_b,yaw_b]))
    z_weight=interp([0,190,223,245,274,299],np.array([1.,1.,4.,4.,1.,1.]))
    tail_q,tail_errors,tail_quat=drive(robot,target[169:],tail[169:],arm,transit_q[-1],tail_yaw[169:],
                                      rotations[169:],seeds_count=0,torso_plan_deg=17.,z_weight=z_weight[169:])
    transit_quat=(Rotation.from_euler('z',base[:,2])*URDF_POSE).as_quat()[:,[3,0,1,2]]
    # Keep quaternion signs consistent for the substep interpolation at joins.
    quats=np.vstack((fixed_quat,transit_quat,tail_quat))
    for i in range(1,len(quats)):
        if quats[i-1]@quats[i]<0:
            quats[i]*=-1
    plan.update(pickup_trial=str(pickup_trace),held_object_in_pad=held.tolist(),
                seated_object_world=seat.tolist(),place_pad_world_m=goal[:3,3].tolist(),
                place_case_rotation_world=seat[:3,:3].tolist(),stations=stations,seat_approach_tilt_deg=tilt,
                seat_approaches=[dict(tilt_deg=item[1],station=item[-1]['place']) for item in approaches],
                transit_solver='large_range_track.solve_wbc_track',transit_frames=[126,168],
                max_transit_step_deg=float(np.abs(np.rad2deg(np.diff(np.vstack((fixed_q[-1],transit_q)),axis=0))).max()),
                max_transit_position_error_mm=float(pe.max()))
    return (np.vstack((fixed_base,base[:,:2],tail[169:])),target,np.vstack((fixed_q,transit_q,tail_q)),
            np.vstack((fixed_errors,np.c_[pe/1000,np.deg2rad(oe)],tail_errors)),quats)


def case_visual(builder, out, half_size):
    # Retain the source OBJ's UVs while fitting its PCA axes to the collider.
    lines=(SAM3D/'textured_mesh.obj').read_text().splitlines()
    idx=[i for i,line in enumerate(lines) if line.startswith('v ')]
    verts=np.array([[float(x) for x in lines[i].split()[1:4]] for i in idx])
    centered=verts-verts.mean(0)
    _,axes=np.linalg.eigh(centered.T@centered)
    axes=axes[:,::-1].copy()
    if np.linalg.det(axes)<0:
        axes[:,0]*=-1
    mesh=centered@axes
    low,high=mesh.min(0),mesh.max(0)
    mesh=(mesh-(low+high)/2)*(2*half_size)/(high-low)
    for i,v in zip(idx,mesh):
        lines[i]='v '+' '.join(f'{x:.7f}' for x in v)
    obj=out/'case_visual.obj'
    obj.write_text('\n'.join(lines)+'\n')
    for asset in ('material.mtl','material_0.png'):
        shutil.copy2(SAM3D/asset,out/asset)
    mat=rg._textured_render_material(str(out/'material_0.png'))
    builder.add_visual_from_file(str(obj),material=mat)


def replay(out, render, reach, down, side, object_grasp=False, fixed_pick=None, pickup_trace=None):
    out.mkdir(parents=True,exist_ok=True)
    if pickup_trace is not None:
        fixed_pick=json.loads((pickup_trace/'grasp_plan.json').read_text())['fixed_pick']
    fixed_trial=fixed_pick is not None and pickup_trace is None
    n_frames=126 if fixed_trial else 300
    object_grasp=object_grasp or fixed_pick is not None
    centers, points, colors = setup(out)
    start,end=centers
    scene,robot=cb._build_scene(planar_base=pickup_trace is not None)
    mat=scene.create_physical_material(1.0,0.9,0.)
    rg.create_box(scene,sapien.Pose(p=start-[0,0,.023]),half_size=[.172,.151,.023],
                  color=(.75,.75,.73),name='white_table',is_static=True)
    rg.create_box(scene,sapien.Pose(p=end-[0,0,.035]),half_size=[.163,.145,.035],
                  color=(.28,.29,.31),name='grey_chair_seat',is_static=True)
    # The support assets are local collision surfaces; use visual legs for scale.
    for center, half, color in ((start,[.153,.129],(.62,.62,.61)),
                                 (end,[.140,.125],(.21,.22,.24))):
        for x,y in ((-1,-1),(-1,1),(1,-1),(1,1)):
            p=[center[0]+x*half[0],center[1]+y*half[1],center[2]/2]
            rg.create_box(scene,sapien.Pose(p=p),half_size=[.013,.013,center[2]/2],
                          color=color,name='support_leg',is_static=True)
    b=scene.create_actor_builder()
    b.set_physx_body_type('dynamic')
    box=np.array([.0253,.0236,.0126])
    b.add_box_collision(sapien.Pose(),half_size=box,material=mat)
    if render:
        case_visual(b,out,box)
    case=b.build(name='case')
    object_R=np.eye(3)
    if object_grasp:
        geometry=json.loads((out/'geometry.json').read_text())
        world_R=np.array(geometry['heading'])@np.array(geometry['gravity_to_sim'])
        with np.load(SOURCE/'object_instance.npz') as instance:
            axis=world_R@instance['pca_axes'][:,0]
        # The flat collider is seated on the table. Retain only the reconstructed
        # horizontal heading, since the raw PCA axes need not be gravity-aligned.
        object_R=Rotation.from_euler('z',np.arctan2(axis[1],axis[0])).as_matrix()
    case_q=Rotation.from_matrix(object_R).as_quat()[[3,0,1,2]]
    case.set_pose(sapien.Pose(p=start+[0,0,box[2]+.001],q=case_q))
    for comp in case.get_components():
        if hasattr(comp,'set_disable_gravity'):
            comp.mass=.08
            comp.linear_damping=.10
            comp.angular_damping=.12

    joints=robot.get_active_joints()
    joint_idx={j.get_name():i for i,j in enumerate(joints)}
    planar=[joint_idx[f'planar_joint_{label}'] for label in ('x','y','yaw')] if pickup_trace is not None else []
    arm=[joint_idx[f'right_arm_joint{i+1}'] for i in range(6)]
    arm.extend(joint_idx[f'torso_joint{i+1}'] for i in range(3))
    fingers=[joint_idx[f'right_gripper_finger_joint{i+1}'] for i in range(2)]
    left=[joint_idx[f'left_arm_joint{i+1}'] for i in range(6)]
    q=robot.get_qpos()
    q[left]=rg.R1_LEFT_START
    q[fingers]=.044
    q[[joint_idx[f'left_gripper_finger_joint{i+1}'] for i in range(2)]]=.025
    robot.set_qpos(q)
    for i,j in enumerate(joints):
        if i in fingers:
            j.set_drive_property(stiffness=4000,damping=120,force_limit=28)
        elif i in planar:
            j.set_drive_property(stiffness=35000,damping=3000,force_limit=5000)
        else:
            j.set_drive_property(stiffness=2400,damping=150)
    for name in ('right_gripper_finger_link1','right_gripper_finger_link2',
                 'right_realsense_link','left_realsense_link'):
        link=robot.find_link_by_name(name)
        for comp in link.entity.components:
            for shape in getattr(comp,'get_collision_shapes',lambda:())():
                if 'realsense' in name:
                    shape.set_collision_groups([0,0,0,0])
                else:
                    shape.set_physical_material(mat)

    # Grasp above the lower edge so the tips clear the support. The source
    # phases determine the schedule and the reconstructed surfaces the endpoints.
    tilt=np.deg2rad(down)
    # The R1 finger collision box is 60 x 10 x 40 mm. Its projection on
    # gravity sets the fingertip clearance instead of the pad height alone.
    fingertip_drop=.030*np.sin(tilt)+.020*np.cos(tilt)
    pick=start+[0,0,fingertip_drop+.010]
    put=end+[0,0,fingertip_drop+.0065]
    near=pick+[-side*.100,0,.11]
    up=pick+[0,0,.145]
    over=put+[0,0,.145]
    back=put+[-side*.105,0,.14]
    yaw=-side*np.pi/2
    x=np.array([side*np.cos(tilt),0.,-np.sin(tilt)])
    y=np.array([0.,1.,0.])
    R=np.column_stack((x,y,np.cross(x,y)))
    if object_grasp:
        source_x,source_y,source_spread,_=source_directions(world_R,*REFERENCE_WINDOW)
        R,forward,pad_height,plan=plan_grasp(box,object_R,source_x,source_y,source_spread,
                                            None if fixed_pick is None else fixed_pick[3:5])
        grasp_offset=np.array(plan['pad_minus_case_center_world_m'])
        pick=start+grasp_offset+[0,0,box[2]]
        # After lift, level the wrist about the jaw axis to reach the lower
        # seat. Compute the resulting case footprint from the fixed grasp.
        tilt=np.arcsin(R[2,0])
        blend=interp([0,175,212,299],np.array([0.,0.,1.,1.]))
        turn=Rotation.from_rotvec(blend[:,None]*tilt*R[:,1]).as_matrix()
        orientations=turn@R
        place_case_R=turn[-1]@object_R
        offset=turn[-1]@grasp_offset
        put=end+offset+[0,0,np.abs(place_case_R[2])@box]
        plan['place_case_rotation_world']=place_case_R.tolist()
        plan['place_pad_world_m']=put.tolist()
        near=pick-.100*forward+[0,0,.07]
        up=pick+[0,0,.145]
        over=put+[0,0,.145]
        back=put-.105*forward+[0,0,.14]
    frames=[0,47,71,91,107,125,162,190,219,230,245,274,299]
    target=interp(frames,np.array([near,near,pick,pick,pick,up,over,over,put,put,put,back,back]))
    hold=.010 if fixed_pick is None else box[plan['face_axis']]+.001547-fixed_pick[5]/1000
    grip=interp(frames,np.array([.044,.044,.044,hold,hold,hold,hold,hold,hold,hold,.044,.044,.044]))
    base_frames=[0,47,71,105,125,162,219,244,274,299]
    base_x=-side*reach
    # The R1 right shoulder is +X in its +Y-forward default pose.
    # Rotate that offset with base yaw before placing the base beside the support.
    left_offset=side*.142
    base_start=np.array([base_x,start[1]+left_offset])
    base_end=np.array([base_x,end[1]+left_offset])
    bases=interp(base_frames,np.array([base_start+[-side*.10,0],base_start,base_start,
                                    base_start,base_start,base_end,base_end,base_end,
                                    base_end+[-side*.07,0],base_end+[-side*.07,0]]))
    q_seed=np.array([-.7,1.7,-.7,.3,.2,0])
    z_weight=np.ones(len(target))
    if fixed_pick is not None:
        bases[:]=start[:2]+fixed_pick[:2]
        yaw=np.deg2rad(fixed_pick[2])-np.pi/2
        # The fixed-base trial must use the same low-clearance correction and
        # force target that the final replay uses before joining the transit.
        z_weight[:126]=2.
        plan['fixed_pick']=list(fixed_pick)
        (out/'grasp_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif object_grasp:
        original_pose=case.get_pose()
        case.set_pose(sapien.Pose(p=[2.,2.,2.]))
        bases,yaw,stations=select_stations(scene,robot,arm,start,end,forward,orientations,target,reach,q_seed)
        case.set_pose(original_pose)
        body=next(comp for comp in case.get_components() if hasattr(comp,'set_disable_gravity'))
        body.linear_velocity=np.zeros(3)
        body.angular_velocity=np.zeros(3)
        z_weight=interp([0,190,219,245,274,299],np.array([1.,1.,2.,2.,1.,1.]))
        plan['stations']=stations
        (out/'grasp_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if pickup_trace is None:
        q_plan,ik,base_q=drive(robot,target[:n_frames],bases[:n_frames],arm,q_seed,yaw,
                               orientations[:n_frames] if object_grasp else R,
                               torso_plan_deg=17. if object_grasp else 20.,z_weight=z_weight[:n_frames])
    else:
        bases,target,q_plan,ik,base_q=join_measured_pickup(
            scene,robot,arm,case,start,end,box,forward,pickup_trace,target,plan)
        (out/'grasp_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('IK p95 mm',np.quantile(ik[:,0],.95)*1000,'degrees',np.quantile(ik[:,1],.95)*180/np.pi,flush=True)
    for i in [0,71,107,125,162,219,230,299]:
        if i>=n_frames:
            break
        print('IK event',i,np.round(ik[i],4),flush=True)
    q[arm]=q_plan[0]
    if planar:
        planar_yaw=np.unwrap(Rotation.from_quat(base_q[:,[1,2,3,0]]).as_euler('xyz')[:,2])
        q[planar]=[*bases[0],planar_yaw[0]]
        robot.set_root_pose(sapien.Pose())
    else:
        robot.set_root_pose(sapien.Pose(p=[*bases[0],0],q=base_q[0]))
    robot.set_qpos(q)
    robot.set_qvel(np.zeros_like(q))
    for joint,theta in zip(joints,q):
        joint.set_drive_target(float(theta))
    rg.step_sim(scene,robot,125)
    object_start=np.asarray(case.get_pose().p).copy()
    contact_names=('right_gripper_finger_link1','right_gripper_finger_link2','white_table','grey_chair_seat','ground','other')
    columns={name:i for i,name in enumerate(contact_names)}
    steps=np.diff((np.arange(n_frames+1)*250)//30)
    all_contact=[]; bilateral=[]; chair_only=[]; states=[]; pads=[]; q_actual=[]
    pad_rotations=[]; opposed_faces=[]; nearest_faces=[]
    executed_bases=[]
    command_frames=[]; release_start=None; release_pose=None; issued_grip=[]
    issued_q=[]; issued_pad=[]; issued_ik=[]; seat_corrections=[]
    chair_vertical=[]
    # Require the chair to carry at least half the static weight over three
    # consecutive video frames. A single collision impulse cannot open the jaws.
    chair_impulse_min=.5*.08*9.81*steps/250
    robot_environment=[]
    furniture_pairs={}
    robot_links={link.entity.get_name() for link in robot.get_links()}
    furniture={'white_table','grey_chair_seat','support_leg'}
    tool=robot.find_link_by_name('right_gripper_link')
    if pickup_trace is not None:
        nominal_seat=target[236].copy()
        seat_orientation=(np.asarray(plan['seated_object_world'])@np.linalg.inv(plan['held_object_in_pad']))[:3,:3]

        def solve_seat_pad(requested, index):
            saved_q=robot.get_qpos().copy()
            saved_v=robot.get_qvel().copy()
            scratch=saved_q.copy()
            scratch[planar]=0.
            robot.set_qpos(scratch)
            solution,err,_=drive(robot,[requested],bases[index:index+1],arm,q_plan[index],
                                  planar_yaw[index]-np.pi/2,seat_orientation,
                                  seeds_count=0,torso_plan_deg=17.,z_weight=4.)
            robot.set_root_pose(sapien.Pose())
            robot.set_qpos(saved_q)
            robot.set_qvel(saved_v)
            return solution[0],err[0]
    if render:
        c0=scene.add_camera('front',480,360,np.deg2rad(48),.02,30)
        c1=scene.add_camera('side',480,360,np.deg2rad(48),.02,30)
        c2=scene.add_camera('robot_fpv',480,360,np.deg2rad(65),.02,30)
        c3=scene.add_camera('overhead',480,360,np.deg2rad(55),.02,30)
        wrist_offset=rg.look_at_pose([.005,-.095,.085],[rg.PAD_FWD,0.,-.012]).to_transformation_matrix()
        middle=(start+end)/2
        view_center=[-.10,middle[1],.88]
        c0.entity.set_pose(rg.look_at_pose([-1.90,-.85,1.52],view_center))
        c1.entity.set_pose(rg.look_at_pose([.94,-1.65,1.50],view_center))
        c3.entity.set_pose(rg.look_at_pose([-.46,.46,2.35],[0.,middle[1],.58]))
        # Capture static reconstruction as a separate point-only view with the
        # same orientation as the top-down camera. The other tiles render physics.
        import trimesh
        keep=np.linalg.norm(points[:,:2]-middle[:2],axis=1)<.70
        key=np.floor(points[keep]/.018).astype(int)
        _,unique=np.unique(key,axis=0,return_index=True)
        cloud=trimesh.PointCloud(points[keep][unique],colors=np.uint8(255*colors[keep][unique].clip(0,1)))
        cloud.export(out/'scene_reference.ply')
        def view_cloud():
            pose=c3.entity.get_pose().to_transformation_matrix()
            pc=(cloud.vertices-pose[:3,3]) @ pose[:3,:3]
            ratio=360/(2*np.tan(np.deg2rad(55)/2))
            good=pc[:,0]>.02
            uv=np.rint(np.c_[240-ratio*pc[good,1]/pc[good,0],180-ratio*pc[good,2]/pc[good,0]]).astype(int)
            im=Image.new('RGB',(480,360),(26,30,35)); draw=ImageDraw.Draw(im)
            idx=np.argsort(pc[good,0])[::-1]
            for i in idx:
                x,y=uv[i]
                if 0<=x<480 and 0<=y<360:
                    draw.rectangle((x-1,y-1,x+1,y+1),fill=tuple(cloud.colors[good][i,:3]))
            return np.asarray(im)
        cloud_panel=view_cloud()
        Image.fromarray(cloud_panel).save(out/'point_cloud_only.png')
        visible_points=(cloud_panel != [26,30,35]).any(axis=2)
        w=imageio.get_writer(out/'four_views.mp4',fps=30,macro_block_size=8)
        fpv=imageio.get_writer(out/'robot_fpv.mp4',fps=30,macro_block_size=8)
    for k in range(n_frames):
        command=k
        place_end=236 if pickup_trace is not None else 230
        place_start=207 if pickup_trace is not None else 230
        if object_grasp and k>=place_start:
            if release_start is None and all(chair_vertical[i]>=chair_impulse_min[i] and bilateral[i]>0
                                             and (pickup_trace is None or robot_environment[i]==0)
                                             for i in range(k-3,k)):
                release_start=k
                release_pose=command_frames[-1]
            command=min(k,place_end) if release_start is None else min(299,place_end+k-release_start)
        grip_target=grip[command]
        if pickup_trace is not None and k>=place_start:
            if release_start is None:
                grip_target=hold
                if k>=236 and (k-236)%2==0 and bilateral[-1]>0 and chair_vertical[-1]<chair_impulse_min[k-1]:
                    case_T=case.get_pose().to_transformation_matrix()
                    gap=case_T[2,3]-end[2]-np.abs(case_T[2,:3])@box
                    pad_R=pad_rotations[-1]
                    clearance=pads[-1][2]-end[2]-np.abs(pad_R[2])@FINGER_HALF-abs(pad_R[2,1])*(.044+.003453)
                    correction=max(0.,min(gap+.00015,clearance-.00012,.00060,
                                           target[236,2]-nominal_seat[2]+SUPPORT_CLEARANCE))
                    if correction>1e-6:
                        requested=target[236].copy()
                        requested[2]-=correction
                        q_update,err=solve_seat_pad(requested,236)
                        target[236],q_plan[236],ik[236]=requested,q_update,err
                        seat_corrections.append(dict(frame=k,case_gap_mm=float(gap*1000),
                                                     finger_clearance_mm=float(clearance*1000),
                                                     commanded_descent_mm=float((nominal_seat[2]-requested[2])*1000)))
            else:
                since_release=k-release_start
                if since_release==0:
                    measured_gap=float(max(q_actual[-1][fingers]))
                    raised_pad=target[release_pose]+[0.,0.,SUPPORT_CLEARANCE]
                    raised_q,raised_err=solve_seat_pad(raised_pad,release_pose)
                    for i in fingers:
                        joints[i].set_drive_property(stiffness=4000,damping=120,force_limit=12)
                # Start from the jaw's measured opening under load and raise
                # the unloaded fingertips off the chair during release.
                grip_target=measured_gap+(.044-measured_gap)*min(1.,(since_release+1)/8)
                command=release_pose if since_release<=8 else min(299,237+since_release)
        issued_grip.append(grip_target)
        joint_target,pad_target,ik_error=q_plan[command],target[command],ik[command]
        if pickup_trace is not None and release_start is not None and k-release_start<=8:
            blend=min(1.,(k-release_start)/5)
            joint_target=joint_target+blend*(raised_q-joint_target)
            pad_target=pad_target+blend*(raised_pad-pad_target)
            ik_error=ik_error+blend*(raised_err-ik_error)
        issued_q.append(joint_target.copy())
        issued_pad.append(pad_target.copy())
        issued_ik.append(ik_error.copy())
        command_frames.append(command)
        for i,theta in zip(arm,joint_target):
            joints[i].set_drive_target(float(theta))
        for i in fingers:
            joints[i].set_drive_target(float(grip_target))
        j_sum=np.zeros(6); n_bilateral=n_chair=n_opposed=n_nearest=0
        environment=0
        support_z_impulse=0.
        previous=command_frames[max(k-1,0)]
        old=bases[previous]
        for j in range(steps[k]):
            portion=(j+1)/steps[k]
            root=old+(bases[command]-old)*portion
            if planar:
                delta=planar_yaw[command]-planar_yaw[previous]
                velocity=np.r_[bases[command]-old,delta]*250/steps[k]
                for i,value,v in zip(planar,np.r_[root,planar_yaw[previous]+portion*delta],velocity):
                    joints[i].set_drive_target(float(value))
                    joints[i].set_drive_velocity_target(float(v))
            else:
                base_rotation=(1-portion)*base_q[previous]+portion*base_q[command]
                robot.set_root_pose(sapien.Pose(p=[*root,0],q=unit(base_rotation)))
            rg.step_sim(scene,robot)
            ji=np.zeros(6)
            finger_centers=np.zeros((2,3))
            finger_normals=np.zeros((2,3))
            touched=False
            for contact in scene.get_contacts():
                a,b=(body.get_entity().get_name() for body in contact.bodies)
                other=b if a=='case' else a if b=='case' else None
                if (a in robot_links and b in furniture) or (b in robot_links and a in furniture):
                    if any(np.linalg.norm(p.impulse)>1e-6 for p in contact.points):
                        touched=True
                        key='/'.join(sorted((a,b)))
                        furniture_pairs[key]=furniture_pairs.get(key,0)+1
                if other is not None:
                    column=columns.get(other,5)
                    for point in contact.points:
                        impulse=np.linalg.norm(point.impulse)
                        ji[column]+=impulse
                        if column<2:
                            finger_centers[column]+=impulse*np.asarray(point.position)
                            finger_normals[column]+=impulse*np.asarray(point.normal)
                        if column==3 and abs(point.normal[2])>.8:
                            support_z_impulse+=abs(float(point.impulse[2]))
            environment+=int(touched)
            j_sum+=ji
            if np.all(ji[:2]>1e-6):
                n_bilateral+=1
                object_T=case.get_pose().to_transformation_matrix()
                local=(finger_centers/ji[:2,None]-object_T[:3,3])@object_T[:3,:3]
                axes=np.abs(np.abs(local)-box).argmin(axis=1)
                axis=axes[0]
                near=int(axes[1]==axis and local[0,axis]*local[1,axis]<0
                         and (not object_grasp or axis==plan['face_axis']))
                n_nearest+=near
                if fixed_pick is None:
                    n_opposed+=near
                else:
                    axis=plan['face_axis']
                    normals=finger_normals@object_T[:3,:3]
                    lengths=np.linalg.norm(normals,axis=1)
                    # At an edge the nearest coordinate can label a side contact
                    # as a top contact. Check the measured reaction direction
                    # and the actual position on the intended side patch.
                    normal_ok=np.abs(normals[:,axis])>=np.cos(np.deg2rad(20.))*lengths
                    on_patch=(np.all(np.abs(np.abs(local[:,axis])-box[axis])<=.002)
                              and np.all(np.abs(local)<=box+.002))
                    n_opposed+=int(np.all(lengths>1e-6)&np.all(normal_ok)&on_patch
                                   &(local[0,axis]*local[1,axis]<0))
            n_chair+=int(ji[3]>1e-6 and np.all(ji[[0,1,2,4,5]]<=1e-6))
        pose=case.get_pose()
        states.append(np.r_[pose.p,pose.q])
        T=tool.get_entity_pose().to_transformation_matrix()
        pads.append(T[:3,3]+rg.PAD_FWD*T[:3,0])
        pad_rotations.append(T[:3,:3].copy())
        opposed_faces.append(n_opposed)
        nearest_faces.append(n_nearest)
        q_actual.append(robot.get_qpos())
        executed_bases.append(q_actual[-1][planar[:2]] if planar else bases[command])
        all_contact.append(j_sum);bilateral.append(n_bilateral);chair_only.append(n_chair)
        chair_vertical.append(support_z_impulse)
        robot_environment.append(environment)
        if k in frames:
            print(k,'pad',np.round(pads[-1],3),'obj',np.asarray(pose.p).round(3),'impulse',j_sum.round(3),flush=True)
        if render:
            if object_grasp:
                c2.entity.set_pose(sapien.Pose(T@wrist_offset))
            else:
                root=np.r_[bases[command],0.]
                eye=root+[side*.20,0,robot.find_link_by_name('torso_link4').get_entity_pose().p[2]+.33]
                c2.entity.set_pose(rg.look_at_pose(eye,target[command]+[0,0,-.025]))
            scene.update_render()
            imgs=[]
            for cam in (c0,c1,c2,c3):
                cam.take_picture()
                imgs.append(np.uint8(255*np.clip(cam.get_picture('Color')[:,:,:3],0,1)))
            # Cyan points identify the reconstruction layer over the physics camera.
            overlay=imgs[3].copy()
            overlay[visible_points]=np.uint8(.58*overlay[visible_points]+.42*np.array([45,200,240]))
            w.append_data(np.vstack((np.hstack(imgs[:2]),np.hstack((imgs[2],overlay)))))
            fpv.append_data(imgs[2])
    if render:
        w.close();fpv.close()
    base_targets=bases[command_frames]
    bases=np.asarray(executed_bases)
    target,q_plan,ik=map(np.asarray,(issued_pad,issued_q,issued_ik))
    grip=np.asarray(issued_grip)
    pose=np.array(states); J=np.array(all_contact); bilateral=np.array(bilateral)
    lifted=(pose[:,2]>object_start[2]+.04)&(bilateral>0)&(J[:,2]<1e-6)
    walk=(np.linalg.norm(np.diff(bases,axis=0,prepend=bases[:1]),axis=1)>1e-5)
    walk &= np.linalg.norm(np.diff(base_targets,axis=0,prepend=base_targets[:1]),axis=1)>1e-5
    end_drift=np.linalg.norm(pose[-15:,:3]-pose[-1,:3],axis=1).max()
    opening=np.flatnonzero((np.arange(n_frames)>=place_start)&(grip>hold+1e-4)) if not fixed_trial else []
    release=int(opening[0]) if len(opening) else None
    sustained_support=chair_impulse_min if object_grasp else np.full(n_frames,1e-5)
    measured_support=np.asarray(chair_vertical) if object_grasp else J[:,3]
    bearing_frame=release_start if pickup_trace is not None else release
    result=dict(frames=n_frames,physical_time_s=float(steps.sum()/250),
                base_travel_m=float(np.linalg.norm(np.diff(bases,axis=0),axis=1).sum()),
                base_tracking_p95_mm=float(np.quantile(np.linalg.norm(bases-base_targets,axis=1),.95)*1000),
                carried_while_moving_frames=int(np.count_nonzero(
                    (bilateral>0) & (J[:,2]<1e-6) & (J[:,3]<1e-6) & walk)),
                torso_pitch_peak_deg=np.abs(np.rad2deg(cb.lrt.torso_link_pitch(
                    np.asarray(q_actual)[:,arm[6:]]))).max(axis=0).tolist(),
                lift_mm=float((pose[:,2].max()-object_start[2])*1000),
                dynamic_bilateral_lift_frames=int(lifted.sum()),
                chair_support_before_open=bool(release is not None and bearing_frame is not None and
                    np.all(measured_support[bearing_frame-3:bearing_frame]>=sustained_support[bearing_frame-3:bearing_frame])),
                release_trigger_frame=release_start,first_open_command_frame=release,
                seat_descent_corrections=seat_corrections,
                furniture_contact_pairs=furniture_pairs,
                final_chair_only_seconds=sum(chair_only[-15:])/250,
                final_drift_mm=float(end_drift*1000), ground_frames=int(np.count_nonzero(J[:,4]>1e-6)),
                pad_p95_mm=float(np.quantile(np.linalg.norm(np.array(pads)-target,axis=1),.95)*1000),
                robot_furniture_contact_frames=int(np.count_nonzero(robot_environment)),
                final_place_xy_error_mm=float(np.linalg.norm(pose[-1,:2]-end[:2])*1000))
    result['pass']=bool(result['carried_while_moving_frames'] and result['chair_support_before_open']
                        and result['final_chair_only_seconds']>=.5 and result['ground_frames']==0
                        and result['final_drift_mm']<3 and result['final_place_xy_error_mm']<35
                        and result['robot_furniture_contact_frames']==0)
    if fixed_trial:
        airborne=np.flatnonzero((bilateral>0)&(J[:,2]<1e-6))
        prelift=np.arange(n_frames)<(airborne[0] if len(airborne) else n_frames)
        contact_fraction=float(np.asarray(opposed_faces)[prelift].sum()/max(1,bilateral[prelift].sum()))
        relative=np.einsum('ni,nij->nj',pose[:,:3]-pads,pad_rotations)
        slip=float(np.linalg.norm(relative[airborne]-relative[airborne[0]],axis=1).max()*1000) if len(airborne) else None
        reference_x,reference_y,_,_=source_directions(world_R,*REFERENCE_WINDOW)
        rot=np.asarray(pad_rotations)[prelift&(bilateral>0)]
        angle=([float(np.quantile(angles(rot[:,:,0],reference_x),.95)),
                float(np.quantile(angles(rot[:,:,1],reference_y,unsigned=True),.95))] if len(rot) else [None,None])
        result.update(pickup_lift_frame=int(airborne[0]) if len(airborne) else None,
                      pickup_opposed_face_fraction=contact_fraction,pickup_slip_mm=slip,pickup_source_error_p95_deg=angle,
                      pickup_tail_bilateral_frames=int((bilateral[-6:]>0).sum()))
        result['pass']=bool(result['lift_mm']>=40 and slip is not None and slip<=.4*box.min()*1000
                            and len(airborne)>=8 and len(airborne)==n_frames-airborne[0]
                            and contact_fraction>=.95 and result['robot_furniture_contact_frames']==0
                            and max(result['torso_pitch_peak_deg'])<=20.)
    np.savez(out/'exec_trace.npz', object_pose=pose, contact_impulses=J, contact_names=contact_names,
             bilateral_substeps=bilateral, chair_only_substeps=chair_only, physics_substeps=steps,
             opposed_face_substeps=opposed_faces,nearest_face_substeps=nearest_faces,pad_R_world=pad_rotations,
             robot_furniture_substeps=robot_environment,
             source_frame=np.arange(n_frames), command_frame=command_frames, base_xy=bases,base_quat=base_q[command_frames],
             base_xy_target=base_targets,
             pad_target=target,pad_actual=pads,
             chair_load_threshold_ns=chair_impulse_min,chair_vertical_impulse_ns=chair_vertical,
             gripper_actual=np.asarray(q_actual)[:,fingers],
             q_arm_target=q_plan,q_actual=q_actual,gripper_target=grip,ik_err=ik)
    (out/'verdict.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--reach',type=float,default=.29)
    parser.add_argument('--down',type=float,default=85)
    parser.add_argument('--side',type=int,choices=[-1,1],default=1)
    parser.add_argument('--object-grasp',action='store_true')
    stage=parser.add_mutually_exclusive_group()
    stage.add_argument('--fixed-pick',nargs=6,type=float,metavar=('X','Y','YAW','JAW_YAW','TILT','PRELOAD_MM'))
    stage.add_argument('--pickup-trace',type=Path)
    a=parser.parse_args()
    replay(a.output,a.render,a.reach,a.down,a.side,a.object_grasp,a.fixed_pick,a.pickup_trace)


"""456483158 source-timed, gravity-active tabletop to chair transfer."""
from pathlib import Path
import argparse
import json
import sys

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import cb_large_range_sim as cb

rg, bp = cb.rg, cb.bp
SOURCE = Path('/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158')
URDF_POSE = Rotation.from_quat([0, 0, np.sqrt(.5), np.sqrt(.5)])


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


def drive(scene, robot, targets, bases, selected, q0, yaw, down, forward):
    q_min, q_max = bp._limits('right').T
    lo, hi = q_min+.015, q_max-.015
    rz = Rotation.from_euler('z',yaw)
    base_q = (rz*URDF_POSE).as_quat()[[3,0,1,2]]
    x = np.array([np.cos(down),0.,-np.sin(down)]) if forward > 0 else np.array([-np.cos(down),0.,-np.sin(down)])
    y = np.array([0.,1.,0.])
    orientation = Rotation.from_matrix(np.column_stack((x,y,np.cross(x,y))))
    rng = np.random.default_rng(5)
    solutions=[]
    errors=[]
    for k,(target,root) in enumerate(zip(targets,bases)):
        robot.set_root_pose(sapien.Pose(p=[*root,0],q=base_q))
        def residual(q):
            p,R=bp._gripper_pad_fk(robot,'right',selected,q)
            angle=(orientation.inv()*Rotation.from_matrix(R)).as_rotvec()
            return np.r_[p-target,.12*angle,.0015*(q-q0)]
        def jac(q):
            return np.column_stack([(residual(q+h)-residual(q-h))/.0002
                                    for h in np.eye(6)*.0001])
        seeds=[q0] if k else [q0,*rng.uniform(lo,hi,(25,6))]
        candidates=[least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),jac=jac,
                                 max_nfev=100,ftol=1e-6) for seed in seeds]
        best=min(candidates,key=lambda fit:fit.cost)
        err=residual(best.x)
        solutions.append(best.x)
        errors.append([np.linalg.norm(err[:3]),np.linalg.norm(err[3:6])/.12])
        q0=best.x
    return np.asarray(solutions),np.asarray(errors),base_q


def replay(out, render, reach, down, side):
    out.mkdir(parents=True,exist_ok=True)
    centers, points, colors = setup(out)
    start,end=centers
    scene,robot=cb._build_scene()
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
    b.add_box_visual(sapien.Pose(),half_size=box,material=(.09,.10,.12))
    case=b.build(name='case')
    case.set_pose(sapien.Pose(p=start+[0,0,box[2]+.001]))
    for comp in case.get_components():
        if hasattr(comp,'set_disable_gravity'):
            comp.mass=.08
            comp.linear_damping=.10
            comp.angular_damping=.12

    joints=robot.get_active_joints()
    joint_idx={j.get_name():i for i,j in enumerate(joints)}
    arm=[joint_idx[f'right_arm_joint{i+1}'] for i in range(6)]
    fingers=[joint_idx[f'right_gripper_finger_joint{i+1}'] for i in range(2)]
    left=[joint_idx[f'left_arm_joint{i+1}'] for i in range(6)]
    q=robot.get_qpos()
    q[left]=[1.,.6,-.4,.5,0,0]
    q[fingers]=.044
    q[[joint_idx[f'left_gripper_finger_joint{i+1}'] for i in range(2)]]=.025
    robot.set_qpos(q)
    for i,j in enumerate(joints):
        if i in fingers:
            j.set_drive_property(stiffness=4000,damping=120,force_limit=28)
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
    pick=start+[0,0,.021]
    put=end+[0,0,.021]
    near=pick+[-side*.100,0,.11]
    up=pick+[0,0,.145]
    over=put+[0,0,.145]
    back=put+[-side*.105,0,.14]
    frames=[0,47,71,91,123,151,190,219,230,245,274,299]
    target=interp(frames,np.array([near,near,pick,pick,pick,up,over,put,put,put,back,back]))
    grip=interp(frames,np.array([.044,.044,.044,.010,.010,.010,.010,.010,.010,.044,.044,.044]))
    base_frames=[0,47,71,121,152,190,219,244,274,299]
    base_x=-side*reach
    left_offset=-side*.14
    base_start=np.array([base_x,start[1]+left_offset])
    base_end=np.array([base_x,end[1]+left_offset])
    bases=interp(base_frames,np.array([base_start+[-side*.10,0],base_start,base_start,
                                    base_start,base_start,base_end,base_end,base_end,
                                    base_end+[-side*.07,0],base_end+[-side*.07,0]]))
    q_plan,ik,base_q=drive(scene,robot,target,bases,arm,np.array([-.7,1.7,-.7,.3,.2,0]),
                           -side*np.pi/2,np.radians(down),side)
    print('IK p95 mm',np.quantile(ik[:,0],.95)*1000,'degrees',np.quantile(ik[:,1],.95)*180/np.pi,flush=True)
    for i in [0,71,123,151,190,219,230,299]:
        print('IK event',i,np.round(ik[i],4),flush=True)
    q[arm]=q_plan[0]
    robot.set_root_pose(sapien.Pose(p=[*bases[0],0],q=base_q))
    robot.set_qpos(q)
    for joint,theta in zip(joints,q):
        joint.set_drive_target(float(theta))
    rg.step_sim(scene,robot,125)
    object_start=np.asarray(case.get_pose().p).copy()
    contact_names=('right_gripper_finger_link1','right_gripper_finger_link2','white_table','grey_chair_seat','ground','other')
    columns={name:i for i,name in enumerate(contact_names)}
    steps=np.diff((np.arange(301)*250)//30)
    all_contact=[]; bilateral=[]; chair_only=[]; states=[]; pads=[]; q_actual=[]
    tool=robot.find_link_by_name('right_gripper_link')
    if render:
        c0=scene.add_camera('front',480,360,np.deg2rad(48),.02,30)
        c1=scene.add_camera('side',480,360,np.deg2rad(48),.02,30)
        c2=scene.add_camera('robot_fpv',480,360,np.deg2rad(65),.02,30)
        c3=scene.add_camera('overhead',480,360,np.deg2rad(55),.02,30)
        middle=(start+end)/2
        for cam,eye in [(c0,[-1.75,-.82,1.55]),(c1,[.78,-1.38,1.50]),(c3,[-1.1,.70,1.62])]:
            cam.entity.set_pose(rg.look_at_pose(eye,middle))
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
        w=imageio.get_writer(out/'four_views.mp4',fps=30,macro_block_size=8)
        fpv=imageio.get_writer(out/'robot_fpv.mp4',fps=30,macro_block_size=8)
    for k in range(300):
        for i,theta in zip(arm,q_plan[k]):
            joints[i].set_drive_target(float(theta))
        for i in fingers:
            joints[i].set_drive_target(float(grip[k]))
        j_sum=np.zeros(6); n_bilateral=n_chair=0
        old=bases[max(k-1,0)]
        for j in range(steps[k]):
            root=old+(bases[k]-old)*(j+1)/steps[k]
            robot.set_root_pose(sapien.Pose(p=[*root,0],q=base_q))
            rg.step_sim(scene,robot)
            ji=np.zeros(6)
            for contact in scene.get_contacts():
                a,b=(body.get_entity().get_name() for body in contact.bodies)
                other=b if a=='case' else a if b=='case' else None
                if other is not None:
                    ji[columns.get(other,5)] += sum(np.linalg.norm(p.impulse) for p in contact.points)
            j_sum+=ji
            n_bilateral+=int(np.all(ji[:2]>1e-6))
            n_chair+=int(ji[3]>1e-6 and np.all(ji[[0,1,2,4,5]]<=1e-6))
        pose=case.get_pose()
        states.append(np.r_[pose.p,pose.q])
        T=tool.get_entity_pose().to_transformation_matrix()
        pads.append(T[:3,3]+rg.PAD_FWD*T[:3,0])
        q_actual.append(robot.get_qpos())
        all_contact.append(j_sum);bilateral.append(n_bilateral);chair_only.append(n_chair)
        if k in frames:
            print(k,'pad',np.round(pads[-1],3),'obj',np.asarray(pose.p).round(3),'impulse',j_sum.round(3),flush=True)
        if render:
            root=np.r_[bases[k],0.]
            eye=root+[side*.20,0,robot.find_link_by_name('torso_link4').get_entity_pose().p[2]+.33]
            c2.entity.set_pose(rg.look_at_pose(eye,target[k]+[0,0,-.025]))
            scene.update_render()
            imgs=[]
            for cam in (c0,c1,c2):
                cam.take_picture()
                imgs.append(np.uint8(255*np.clip(cam.get_picture('Color')[:,:,:3],0,1)))
            w.append_data(np.vstack((np.hstack(imgs[:2]),np.hstack((imgs[2],cloud_panel)))))
            fpv.append_data(imgs[2])
    if render:
        w.close();fpv.close()
    pose=np.array(states); J=np.array(all_contact); bilateral=np.array(bilateral)
    lifted=(pose[:,2]>object_start[2]+.04)&(bilateral>0)&(J[:,2]<1e-6)
    walk=(np.linalg.norm(np.diff(bases,axis=0,prepend=bases[:1]),axis=1)>1e-5)
    end_drift=np.linalg.norm(pose[-15:,:3]-pose[-1,:3],axis=1).max()
    release=int(np.flatnonzero(grip[231:] > grip[230]+1e-4)[0]+231)
    result=dict(frames=300,physical_time_s=float(steps.sum()/250),
                base_travel_m=float(np.linalg.norm(np.diff(bases,axis=0),axis=1).sum()),
                carried_while_moving_frames=int(np.count_nonzero(lifted & walk)),
                lift_mm=float((pose[:,2].max()-object_start[2])*1000),
                dynamic_bilateral_lift_frames=int(lifted.sum()),
                chair_support_before_open=bool(np.any(J[release-8:release,3]>1e-5)),
                final_chair_only_seconds=sum(chair_only[-15:])/250,
                final_drift_mm=float(end_drift*1000), ground_frames=int(np.count_nonzero(J[:,4]>1e-6)),
                pad_p95_mm=float(np.quantile(np.linalg.norm(np.array(pads)-target,axis=1),.95)*1000),
                final_place_xy_error_mm=float(np.linalg.norm(pose[-1,:2]-end[:2])*1000))
    result['pass']=bool(result['carried_while_moving_frames'] and result['chair_support_before_open']
                        and result['final_chair_only_seconds']>=.5 and result['ground_frames']==0
                        and result['final_drift_mm']<3 and result['final_place_xy_error_mm']<35)
    np.savez(out/'exec_trace.npz', object_pose=pose, contact_impulses=J, contact_names=contact_names,
             bilateral_substeps=bilateral, chair_only_substeps=chair_only, physics_substeps=steps,
             source_frame=np.arange(300), base_xy=bases, pad_target=target,pad_actual=pads,
             q_arm_target=q_plan,q_actual=q_actual,gripper_target=grip,ik_err=ik)
    (out/'verdict.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--reach',type=float,default=.49)
    parser.add_argument('--down',type=float,default=75)
    parser.add_argument('--side',type=int,choices=[-1,1],default=1)
    a=parser.parse_args()
    replay(a.output,a.render,a.reach,a.down,a.side)


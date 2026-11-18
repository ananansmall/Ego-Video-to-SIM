"""Experiment-local joint base/arm whole-body IK with per-link scene clearance residuals."""
import argparse,csv,json,sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT=Path('/path/to/Real2SimGripper')
OUT=ROOT/'examples/r1_multicontact/runs/pixel-umi-ear-20261118/chair_contact_self_ab'
PKG=OUT/'B/package'
SEED=OUT/'B/seed_plan.npz'
sys.path.insert(0,str(ROOT/'scripts'))
from cb_large_range_sim import _build_scene,bp,rg
from clearance_audit import shape_vertices
import large_range_track as lrt

def object_spheres(mesh_path,count=4):
    m=trimesh.load(mesh_path,force='mesh',process=False);v=np.asarray(m.vertices,float)
    c=v.mean(0);_,_,vh=np.linalg.svd(v-c,full_matrices=False);axis=vh[0];s=(v-c)@axis
    edges=np.linspace(float(s.min()),float(s.max()),count+1);out=[]
    for i in range(count):
        width=edges[i+1]-edges[i];pts=v[(s>=edges[i]-.06*width)&(s<=edges[i+1]+.06*width)]
        if len(pts):
            cc=pts.mean(0);out.append((cc,float(np.linalg.norm(pts-cc,axis=1).max())+.004))
    return out

def obstacle_trees(mesh_dir):
    np.random.seed(0);out={}
    for name in ('table','chair','desk'):
        m=trimesh.load(mesh_dir/f'{name}.obj',force='mesh',process=False)
        p,_=trimesh.sample.sample_surface(m,12000)
        out[name]=cKDTree(np.vstack([p,np.asarray(m.vertices,float)]))
    return out

def robot_collision_spheres(robot):
    out=[]
    for link in robot.get_links():
        for si,sh in enumerate(link.get_collision_shapes()):
            v=shape_vertices(sh)
            if v is None or len(v)==0: continue
            v=np.asarray(v,float);c=v.mean(0);r=float(np.linalg.norm(v-c,axis=1).max())
            out.append((link.entity.get_name(),si,link,sh.get_local_pose().to_transformation_matrix(),c,r))
    if not out: raise RuntimeError('robot collision geometry is empty')
    return out

def T_from(p,R):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--first', type=int)
    ap.add_argument('--last', type=int)
    ap.add_argument('--out',type=Path,default=OUT/'B/plan')
    ap.add_argument('--margin-mm',type=float,default=20.)
    ap.add_argument('--weight',type=float,default=42.)
    ap.add_argument('--max-nfev',type=int,default=45)
    ap.add_argument('--self-margin-mm',type=float,default=15.)
    args=ap.parse_args()
    if args.margin_mm < 0 or args.self_margin_mm < 0 or args.weight <= 0:
        raise ValueError('invalid obstacle weight or clearance margin')
    with np.load(PKG/'paired_targets.npz', allow_pickle=False) as stored:
        a = {key: stored[key] for key in stored.files}
    frames = np.asarray(a['src_frames'], dtype=int)
    if frames.ndim != 1 or not frames.size or np.any(np.diff(frames) != 1):
        raise ValueError('expected a non-empty, contiguous source-frame axis')
    start, end = int(frames[0]), int(frames[-1])
    first = start if args.first is None else args.first
    last = end if args.last is None else args.last
    if not start <= first <= last <= end:
        raise ValueError(f'source window {first}–{last} is outside {start}–{end}')
    args.first, args.last = first - start, last - start
    meta=json.loads((PKG/'geometry.json').read_text())
    with np.load(OUT/'A/plan/ik_plan.npz',allow_pickle=False) as zA:A={k:zA[k] for k in zA.files}
    with np.load(SEED,allow_pickle=False) as zB:seed_plan={k:zB[k] for k in zB.files}
    if not np.array_equal(A['source_frames'],frames): raise ValueError('A plan frame mismatch')
    # B uses the identical frozen package route as its reference; it does not use
    # A's solved base. Whole-body offsets are optimized inside the same envelope.
    route=np.asarray(a['base_reference_se2'],float)
    ref_xy=route[:,:2]-np.asarray(rg.R1_BASE_P,float)[:2]
    ref_yaw=np.unwrap(route[:,2])
    target=a['T_world_pad'];pair=a['T_pad_object']
    if not np.allclose(target@pair,a['T_world_object'],atol=1e-8):
        raise ValueError('frozen pad/object relation is inconsistent')
    # Use the exact reviewed source windows for the intended carried-object state.
    reviewed=meta['source_timing_evidence']['reviewed_source_windows']
    first_hold=int(reviewed['first_grasp_closure']['lift_visible_by'])
    wood_drop=next(e for e in meta['multi_release_events'] if e['event_id']=='wood_drop')
    regrasp=next(e for e in meta['multi_release_events'] if e['event_id']=='wood_regrasp')
    white_drop=next(e for e in meta['multi_release_events'] if e['event_id']=='white_drop')
    hold1=(first_hold,int(wood_drop['source_window'][0])-1)
    hold2=(int(regrasp['source_window'][1])+1,int(white_drop['source_window'][0])-1)
    hold=np.asarray([hold1[0]<=f<=hold1[1] or hold2[0]<=f<=hold2[1] for f in frames],bool)
    ignore_support=np.full(len(frames),'',dtype='U8')
    wood_window = (frames >= wood_drop['source_window'][0]) & (frames <= regrasp['source_window'][1])
    ignore_support[wood_window] = 'table'
    ignore_support[frames >= white_drop['source_window'][0]] = 'chair'
    trees=obstacle_trees(PKG/'scene_assets');case_spheres=object_spheres(PKG/'case_collision.obj')
    scene,robot=_build_scene()
    joints=robot.get_active_joints();ji={j.get_name():i for i,j in enumerate(joints)}
    arm=[ji[f'right_arm_joint{i}'] for i in range(1,7)]+[ji[f'torso_joint{i}'] for i in range(1,5)]
    left=[ji[f'left_arm_joint{i}'] for i in range(1,7)]
    qinit=robot.get_qpos();qinit[left]=rg.R1_LEFT_START
    qinit[[ji['left_gripper_finger_joint1'],ji['left_gripper_finger_joint2']]]=.045;robot.set_qpos(qinit)
    lim=np.vstack((bp._limits('right'),[joints[i].get_limits()[0] for i in arm[6:]]))
    spheres=robot_collision_spheres(robot)
    link_names={name for name,_,_,_,_,_ in spheres}
    arm_hand=sorted(n for n in link_names if n.startswith('right_arm_link') or n.startswith('right_gripper') or n=='right_arm_base_link')
    torso_links=sorted(n for n in link_names if n.startswith('torso_link'))
    urdf=ET.parse(bp.R1_URDF).getroot();adjacent=set()
    for joint in urdf.findall('joint'):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link')
        adjacent.add((parent,child));adjacent.add((child,parent))
    self_pairs=tuple((a,t) for a in arm_hand for t in torso_links if (a,t) not in adjacent)
    if 'right_arm_link6' not in arm_hand or 'right_gripper_link' not in arm_hand or 'torso_link1' not in torso_links:
        raise RuntimeError('required arm/gripper/torso collision links are missing')
    self_margin=args.self_margin_mm/1000.
    base_range=float(meta.get('base_reference_xy_tolerance_m',.02));yaw_max=np.radians(float(meta.get('base_reference_yaw_tolerance_deg',5.)))
    base_step=.025;yaw_step=np.radians(5.);joint_step=np.radians(10.)
    bs=base_step/np.sqrt(2.);margin=args.margin_mm/1000.
    pitch=np.radians(np.asarray(meta['torso_pitch_limit_deg'],float)-np.asarray(meta.get('torso_pitch_tracking_margin_deg',[0,0,0]),float))
    q_pref=np.r_[lim[:6].mean(1),np.zeros(4)]
    w_pos,w_ori,w_base,w_yaw,w_bcont,w_cont,w_posture,w_limit=42.,4.,3.,3.,1.,.7,.15,.5
    qmargin=np.radians(12.)
    prev=None
    if args.first>0:
        bpref=np.asarray(seed_plan['base_se2'][args.first-1],float)
        prev=np.r_[bpref[0]-ref_xy[args.first-1,0],bpref[1]-ref_xy[args.first-1,1],
                   bpref[2]-ref_yaw[args.first-1],seed_plan['arm_joint_targets'][args.first-1]]
    byframe=[];bplans=[]
    def geometry_clearance(x,k):
        base_pose=[ref_xy[k,0]+x[0],ref_xy[k,1]+x[1],ref_yaw[k]+x[2]]
        robot.set_root_pose(bp.se2_pose(*base_pose))
        q=x[3:];p,R=bp._gripper_pad_fk(robot,'right',arm,q);Tpad=T_from(p,R)
        body=[];by_link={};by_link_spheres={}
        for name,si,link,local,c,r in spheres:
            wc=(link.get_entity_pose().to_transformation_matrix()@local@np.r_[c,1.])[:3]
            ds={n:float(tree.query(wc,k=1)[0])-r for n,tree in trees.items()}
            d=min(ds.values());body.append(d);by_link.setdefault(name,[]).append(d)
            by_link_spheres.setdefault(name,[]).append((wc,r))
        selfd=[]
        for left,right in self_pairs:
            pairs=[np.linalg.norm(a-b)-ra-rb for a,ra in by_link_spheres[left] for b,rb in by_link_spheres[right]]
            if not pairs: raise RuntimeError(f'missing collision spheres for self pair {left}/{right}')
            selfd.append(min(pairs))
        obj=[]
        if hold[k]:
            O=Tpad@pair[k]
            for c,r in case_spheres:
                wc=(O@np.r_[c,1.])[:3]
                for n,tree in trees.items():
                    if n==ignore_support[k]:continue
                    obj.append(float(tree.query(wc,k=1)[0])-r)
        return p,R,np.asarray(body),np.asarray(obj),by_link,np.asarray(selfd)
    for k in range(args.first,args.last+1):
        lo=np.r_[-base_range,-base_range,-yaw_max,lim[:,0]].astype(float)
        hi=np.r_[ base_range, base_range, yaw_max,lim[:,1]].astype(float)
        if prev is not None:
            for j in (0,1):
                dref=ref_xy[k,j]-ref_xy[k-1,j]
                lo[j]=max(lo[j],prev[j]-bs-dref);hi[j]=min(hi[j],prev[j]+bs-dref)
            dyaw=ref_yaw[k]-ref_yaw[k-1];center=prev[2]-dyaw
            lo[2]=max(lo[2],center-yaw_step);hi[2]=min(hi[2],center+yaw_step)
            lo[3:]=np.maximum(lo[3:],prev[3:]-joint_step);hi[3:]=np.minimum(hi[3:],prev[3:]+joint_step)
            bad=lo>=hi
            for j in np.flatnonzero(bad):
                val=np.clip(prev[j],[-base_range,-base_range,-yaw_max,*lim[:,0]][j],
                            [base_range,base_range,yaw_max,*lim[:,1]][j])
                lo[j]=val-1e-5;hi[j]=val+1e-5
        if np.any(lo>=hi):raise RuntimeError(f'empty bounds at {frames[k]}')
        def encode(x):
            y=x.copy();y[9:12]=lrt.torso_pitch_coordinates(x[9:12],lo[9:12],hi[9:12],pitch,encode=True);return y
        def decode(y):
            x=y.copy();x[9:12]=lrt.torso_pitch_coordinates(y[9:12],lo[9:12],hi[9:12],pitch);return x
        lower,upper=lo.copy(),hi.copy();lower[9:12]=0.;upper[9:12]=1.
        if prev is None:
            ba=np.asarray(seed_plan['base_se2'][k],float)
            seed=np.r_[np.clip(ba[0]-ref_xy[k,0],-base_range,base_range),
                       np.clip(ba[1]-ref_xy[k,1],-base_range,base_range),
                       np.clip(ba[2]-ref_yaw[k],-yaw_max,yaw_max),seed_plan['arm_joint_targets'][k]]
        else:seed=prev.copy()
        seed=np.clip(seed,lo+1e-8,hi-1e-8);y0=np.clip(encode(seed),lower+1e-7,upper-1e-7)
        p0,R0=target[k,:3,3],target[k,:3,:3]
        def residual(y):
            x=decode(y);p,R,body,obj,_,selfd=geometry_clearance(x,k)
            parts=[w_pos*(p-p0),w_ori*(R-R0).reshape(-1),w_base*x[:2],np.array([w_yaw*x[2]]),
                   w_posture*(x[3:]-q_pref)]
            below=np.maximum(0.,(lim[:,0]+qmargin)-x[3:]);above=np.maximum(0.,x[3:]-(lim[:,1]-qmargin))
            parts.append(w_limit*np.r_[below,above])
            if prev is not None:
                motion=x[:3]-prev[:3]+np.r_[ref_xy[k]-ref_xy[k-1],ref_yaw[k]-ref_yaw[k-1]]
                parts.extend([w_bcont*motion,w_cont*(x[3:]-prev[3:])])
            parts.append(args.weight*np.maximum(0.,margin-body))
            if hold[k]:parts.append(args.weight*np.maximum(0.,margin-obj))
            else:parts.append(np.zeros(len(case_spheres)*len(trees)))
            parts.append(args.weight*np.maximum(0.,self_margin-selfd))
            return np.concatenate(parts)
        fit=least_squares(residual,y0,bounds=(lower,upper),method='trf',diff_step=1e-4,
                          xtol=1e-4,ftol=1e-4,gtol=1e-4,max_nfev=args.max_nfev)
        x=decode(fit.x);p,R,body,obj,by_link,selfd=geometry_clearance(x,k)
        pe=float(np.linalg.norm(p-p0)*1000);oe=float(np.degrees(np.arccos(np.clip((np.trace(R0.T@R)-1)/2.,-1.,1.))))
        bse2=np.r_[ref_xy[k]+x[:2],ref_yaw[k]+x[2]]
        row={'source_frame':int(frames[k]),'pad_position_error_mm':pe,'pad_orientation_error_deg':oe,
             'base_reference_dx_mm':float(x[0]*1000),'base_reference_dy_mm':float(x[1]*1000),
             'base_reference_dyaw_deg':float(np.degrees(x[2])),'base_step_mm':0. if prev is None else float(np.linalg.norm(bse2-np.r_[ref_xy[k-1]+prev[:2],ref_yaw[k-1]+prev[2]])*1000),
             'min_robot_sphere_clearance_mm':float(body.min()*1000),
             'min_upper_body_clearance_mm':float(min(d for n,ds in by_link.items() if not(n=='base_link' or n.startswith('wheel_motor_link') or n.startswith('steer_motor_link')) for d in ds)*1000),
             'min_arm_torso_clearance_mm':float(min(d for n,ds in by_link.items() if 'arm' in n or n.startswith('torso_link') for d in ds)*1000),
             'min_held_case_clearance_mm':float(obj.min()*1000) if len(obj) else '',
             'min_self_sphere_clearance_mm':float(selfd.min()*1000),
             'held_object_constraint_active':int(hold[k]),'optimizer_nfev':int(fit.nfev),'optimizer_cost':float(fit.cost),
             'torso_pitch_1_deg':float(np.degrees(lrt.torso_link_pitch(x[9:12])[0])),
             'torso_pitch_2_deg':float(np.degrees(lrt.torso_link_pitch(x[9:12])[1])),
             'torso_pitch_3_deg':float(np.degrees(lrt.torso_link_pitch(x[9:12])[2]))}
        for (left,right),dself in zip(self_pairs,selfd):row['self_clearance_'+left+'__'+right+'_mm']=float(dself*1000)
        for name,ds in by_link.items():row['link_'+name+'_clearance_mm']=float(min(ds)*1000)
        byframe.append(row);bplans.append((bse2,x[3:]))
        prev=x
        if (len(bplans)%10==0) or k==args.last:
            print(f'frame {frames[k]}: pad={pe:.1f}mm/{oe:.1f}deg base_offset=({x[0]*1000:.1f},{x[1]*1000:.1f})mm yaw={np.degrees(x[2]):.1f}deg arm_clear={row["min_arm_torso_clearance_mm"]:.1f}mm obj={row["min_held_case_clearance_mm"]}mm nfev={fit.nfev}',flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    Bbase=np.asarray([x[0] for x in bplans]);Bq=np.asarray([x[1] for x in bplans])
    # Also score A plan under the exact same link/object sphere model for comparison.
    arows=[]
    for k in range(args.first,args.last+1):
        ba=np.asarray(A['base_se2'][k],float);qa=np.asarray(A['arm_joint_targets'][k],float)
        x=np.r_[ba[0]-ref_xy[k,0],ba[1]-ref_xy[k,1],ba[2]-ref_yaw[k],qa]
        p,R,body,obj,links,selfd=geometry_clearance(x,k)
        arows.append({'source_frame':int(frames[k]),'variant':'A','pad_position_error_mm':float(np.linalg.norm(p-target[k,:3,3])*1000),
                      'pad_orientation_error_deg':float(np.degrees(np.arccos(np.clip((np.trace(target[k,:3,:3].T@R)-1)/2.,-1.,1.)))),
                      'base_reference_dx_mm':float(x[0]*1000),'base_reference_dy_mm':float(x[1]*1000),'base_reference_dyaw_deg':float(np.degrees(x[2])),
                      'min_robot_sphere_clearance_mm':float(body.min()*1000),
                      'min_upper_body_clearance_mm':float(min(d for n,ds in links.items() if not(n=='base_link' or n.startswith('wheel_motor_link') or n.startswith('steer_motor_link')) for d in ds)*1000),
                      'min_arm_torso_clearance_mm':float(min(d for n,ds in links.items() if 'arm' in n or n.startswith('torso_link') for d in ds)*1000),
                      'min_held_case_clearance_mm':float(obj.min()*1000) if len(obj) else '',
                      'min_self_sphere_clearance_mm':float(selfd.min()*1000),
                      'held_object_constraint_active':int(hold[k]),
                      'torso_pitch_1_deg':float(np.degrees(lrt.torso_link_pitch(qa[6:9])[0])),
                      'torso_pitch_2_deg':float(np.degrees(lrt.torso_link_pitch(qa[6:9])[1])),
                      'torso_pitch_3_deg':float(np.degrees(lrt.torso_link_pitch(qa[6:9])[2]))})
        for (left,right),dself in zip(self_pairs,selfd):arows[-1]['self_clearance_'+left+'__'+right+'_mm']=float(dself*1000)
        for name,ds in links.items():arows[-1]['link_'+name+'_clearance_mm']=float(min(ds)*1000)
    for r in byframe:r['variant']='B'
    rows=arows+byframe;fields=[]
    for row in rows:
        for f in row:
            if f not in fields:fields.append(f)
    with (args.out/'planned_clearance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    if args.first == 0 and args.last == len(frames) - 1:
        margin_q=np.degrees(np.minimum(Bq-lim[:,0],lim[:,1]-Bq))
        np.savez_compressed(args.out/'ik_plan.npz',source_frames=frames,package_sha256=A['package_sha256'],
                            base_lead_s=A['base_lead_s'],base_se2=Bbase,arm_joint_indices=A['arm_joint_indices'],
                            arm_joint_targets=Bq,position_error_mm=np.asarray([r['pad_position_error_mm'] for r in byframe]),
                            orientation_error_deg=np.asarray([r['pad_orientation_error_deg'] for r in byframe]),
                            joint_names=A['joint_names'],joint_limits_rad=lim,joint_margin_deg=margin_q,
                            base_station=A['base_station'])
    else:
        np.savez_compressed(args.out/'ik_plan_window.npz',source_frames=frames[args.first:args.last+1],
                            package_sha256=A['package_sha256'],base_lead_s=A['base_lead_s'],base_se2=Bbase,
                            arm_joint_indices=A['arm_joint_indices'],arm_joint_targets=Bq,
                            position_error_mm=np.asarray([r['pad_position_error_mm'] for r in byframe]),
                            orientation_error_deg=np.asarray([r['pad_orientation_error_deg'] for r in byframe]),
                            joint_names=A['joint_names'],joint_limits_rad=lim,
                            base_station=A['base_station'])
    def stats(name):
        rr=[r for r in byframe if name=='B'] if name=='B' else arows
        return {'pad_position_p95_mm':float(np.quantile([r['pad_position_error_mm'] for r in rr],.95)),
                'pad_orientation_p95_deg':float(np.quantile([r['pad_orientation_error_deg'] for r in rr],.95)),
                'min_robot_clearance_mm':float(min(r['min_robot_sphere_clearance_mm'] for r in rr)),
                'min_upper_clearance_mm':float(min(r['min_upper_body_clearance_mm'] for r in rr)),
                'min_arm_torso_clearance_mm':float(min(r['min_arm_torso_clearance_mm'] for r in rr)),
                'min_self_sphere_clearance_mm':float(min(r['min_self_sphere_clearance_mm'] for r in rr)),
                'min_held_object_clearance_mm':min([r['min_held_case_clearance_mm'] for r in rr if r['min_held_case_clearance_mm']!=''],default=None)}
    summary={'source_window':[int(frames[args.first]),int(frames[args.last])],'frames':len(byframe),
             'verified_source_hold_windows':[list(hold1),list(hold2)],'object_constraint_frames':int(hold[args.first:args.last+1].sum()),
             'allowed_object_support_windows':{'table':[int(wood_drop['source_window'][0]),int(regrasp['source_window'][1])],
                                               'chair':[int(white_drop['source_window'][0]),end]},
             'robot_collision_shape_spheres':len(spheres),'case_pca_spheres':len(case_spheres),
             'scene_surface_points_per_mesh':12000,'margin_mm':args.margin_mm,'obstacle_weight':args.weight,
             'base_reference_xy_tolerance_mm':base_range*1000,'base_reference_yaw_tolerance_deg':float(np.degrees(yaw_max)),
             'base_step_max_mm':base_step*1000,'yaw_step_max_deg':float(np.degrees(yaw_step)),'joint_step_max_deg':10.,
             'max_base_offset_from_reference_mm':float(np.max(np.linalg.norm(Bbase[:,:2]-ref_xy[args.first:args.last+1],axis=1))*1000),
             'max_yaw_offset_deg':float(np.degrees(np.max(np.abs(Bbase[:,2]-ref_yaw[args.first:args.last+1])))),
             'A_same_geometry_metrics':stats('A'),'B_joint_wbc_metrics':stats('B'),
             'sphere_approximation':'one conservative bounding sphere per available convex collision shape; four PCA-bin spheres for bottle; per-frame planning metric. Self residual covers all right-arm/gripper x torso pairs.',
             'self_collision_pairs':[list(x) for x in self_pairs],'self_collision_pair_count':len(self_pairs),'self_collision_margin_mm':args.self_margin_mm,
             'self_residual_active_frames':int(sum(r['min_self_sphere_clearance_mm']<args.self_margin_mm for r in byframe)),
             'A_same_geometry_metrics_self_clearance_mm':float(min(r['min_self_sphere_clearance_mm'] for r in arows)),
             'B_self_aware_metrics_min_self_clearance_mm':float(min(r['min_self_sphere_clearance_mm'] for r in byframe))}
    (args.out/'plan_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

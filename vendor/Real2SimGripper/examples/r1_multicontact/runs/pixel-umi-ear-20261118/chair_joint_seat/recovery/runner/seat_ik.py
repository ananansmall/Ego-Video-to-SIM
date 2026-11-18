"""Independent FK solver and swept safety checks for the event-local seat motion."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from clearance_audit import MeshQuery
from self_collision_plan_base import object_spheres, robot_collision_spheres, T_from
from cb_large_range_sim import bp, lrt


def _yaw_lerp(a, b, u):
    d = np.arctan2(np.sin(b[2]-a[2]), np.cos(b[2]-a[2]))
    return np.array([a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]), a[2]+u*d])


class JointSeatSafety:
    def __init__(self, robot, arm, fingers, planar, planar_offset, limits,
                 torso_pitch, package, support_data, env_margin_mm=0.5,
                 self_margin_mm=0.5, object_margin_mm=0.5):
        self.robot = robot
        self.arm = np.asarray(arm, int)
        self.fingers = np.asarray(fingers, int)
        self.planar = np.asarray(planar, int)
        self.planar_offset = np.asarray(planar_offset, float)
        self.limits = np.asarray(limits, float)
        self.torso_pitch = np.asarray(torso_pitch, float)
        self.support_data = support_data
        self.env_margin = env_margin_mm / 1000.
        self.self_margin = self_margin_mm / 1000.
        self.object_margin = object_margin_mm / 1000.
        self.shapes = robot_collision_spheres(robot)
        self.by_name = {}
        for name, _, link, local, center, radius in self.shapes:
            self.by_name.setdefault(name, []).append((link, local, center, radius))
        arm_hand = sorted(n for n in self.by_name if
                          n.startswith('right_arm_link') or
                          n.startswith('right_gripper') or n == 'right_arm_base_link')
        torso = sorted(n for n in self.by_name if n.startswith('torso_link'))
        urdf = ET.parse(bp.R1_URDF).getroot()
        adjacent = set()
        for joint in urdf.findall('joint'):
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            adjacent.update(((parent, child), (child, parent)))
        self.pairs = tuple((a, b) for a in arm_hand for b in torso if (a, b) not in adjacent)
        if not self.pairs or not any(a == 'right_gripper_finger_link2' and b == 'torso_link1'
                                     for a, b in self.pairs):
            raise RuntimeError('required right-hand/torso self-collision pairs are missing')
        mesh_dir = Path(package) / 'scene_assets'
        self.meshes = {n: MeshQuery(trimesh.load(mesh_dir / f'{n}.obj',
                                                  force='mesh', process=False))
                       for n in support_data if (mesh_dir / f'{n}.obj').is_file()}
        self.case_mesh = trimesh.load(Path(package) / 'case_collision.obj',
                                      force='mesh', process=False)
        self.case_vertices = np.asarray(self.case_mesh.vertices, float)
        self.case_spheres = object_spheres(Path(package) / 'case_collision.obj')

    def _set(self, base, q_arm, q_fingers):
        full = self.robot.get_qpos()
        full[self.arm] = q_arm
        full[self.fingers] = q_fingers
        if len(self.planar):
            full[self.planar] = np.asarray(base, float) + self.planar_offset
        else:
            self.robot.set_root_pose(bp.se2_pose(*base))
        self.robot.set_qpos(full)
        return bp._gripper_pad_fk(self.robot, 'right', self.arm, q_arm)

    def clearance(self, base, q_arm, q_fingers, pad_object, allowed_support=None):
        pad_p, pad_R = self._set(base, q_arm, q_fingers)
        Tpad = T_from(pad_p, pad_R)
        case_pose = Tpad @ pad_object
        centers = {}
        env = []
        env_worst = ('', float('inf'))
        for name, _, link, local, center, radius in self.shapes:
            world = link.get_entity_pose().to_transformation_matrix() @ local @ np.r_[center, 1.]
            world = world[:3]
            centers.setdefault(name, []).append((world, radius))
            for support, mesh in self.meshes.items():
                distance = mesh.min_distance(world[None, :], margin=radius+.02) - radius
                if distance < env_worst[1]:
                    env_worst = (f'{name}-{support}', distance)
                env.append(distance)
        self_values = []
        self_worst = ('', float('inf'))
        for a, b in self.pairs:
            ds = [float(np.linalg.norm(ca-cb)-ra-rb)
                  for ca, ra in centers[a] for cb, rb in centers[b]]
            d = min(ds)
            self_values.append(d)
            if d < self_worst[1]:
                self_worst = (f'{a}-{b}', d)
        object_values = []
        object_worst = ('', float('inf'))
        allowed = False
        if allowed_support in self.support_data:
            sd = self.support_data[allowed_support]
            bottom = float((case_pose[:3, 3] + self.case_vertices @ case_pose[:3, :3].T)[:, 2].min())
            xy = case_pose[:2, 3]
            inside = np.all(np.abs(xy - np.asarray(sd['center'][:2])) <= np.asarray(sd['half_extents']) + .02)
            allowed = inside and sd['center'][2]-.004 <= bottom <= sd['center'][2]+.025
        for center, radius in self.case_spheres:
            world = (case_pose @ np.r_[center, 1.])[:3]
            for support, mesh in self.meshes.items():
                if allowed and support == allowed_support:
                    # Keep residual dimensions fixed across the support transition.
                    object_values.append(1.0)
                    continue
                distance = mesh.min_distance(world[None, :], margin=radius+.02) - radius
                if distance < object_worst[1]:
                    object_worst = (f'case-{support}', distance)
                object_values.append(distance)
        if not env: env=[float('inf')]
        if not self_values: self_values=[float('inf')]
        if not object_values: object_values=[float('inf')]
        return {
            'min_self_m': float(min(self_values)), 'self_pair': self_worst[0],
            'min_env_m': float(min(env)), 'env_pair': env_worst[0],
            'min_object_m': float(min(object_values)), 'object_pair': object_worst[0],
            'self_values': np.asarray(self_values, float),
            'env_values': np.asarray(env, float), 'object_values': np.asarray(object_values, float),
            'pad': Tpad, 'case': case_pose, 'support_exception': allowed,
        }

    def self_clearance(self, base, q_arm, q_fingers):
        self._set(base, q_arm, q_fingers)
        centers={}
        for name,_,link,local,center,radius in self.shapes:
            world=(link.get_entity_pose().to_transformation_matrix()@local@np.r_[center,1.])[:3]
            centers.setdefault(name,[]).append((world,radius))
        result=[]
        for a,b in self.pairs:
            result.extend(float(np.linalg.norm(x-y)-rx-ry)
                          for x,rx in centers[a] for y,ry in centers[b])
        return min(result) if result else float('inf')

    def sweep_self(self, start_base, start_q, start_fingers, end_base, end_q, end_fingers,
                   samples=7):
        values=[]
        for u in np.linspace(0.,1.,samples):
            base=_yaw_lerp(np.asarray(start_base),np.asarray(end_base),float(u))
            q=(1.-u)*np.asarray(start_q)+u*np.asarray(end_q)
            fingers=(1.-u)*np.asarray(start_fingers)+u*np.asarray(end_fingers)
            values.append(self.self_clearance(base,q,fingers))
        index=int(np.argmin(values))
        return {'min_self_m':float(values[index]),'sample_index':index,'samples':int(samples)}

    def sweep(self, start_base, start_q, start_fingers, end_base, end_q, end_fingers,
              pad_object, allowed_support=None, samples=7):
        rows = []
        for u in np.linspace(0., 1., samples):
            base = _yaw_lerp(np.asarray(start_base), np.asarray(end_base), float(u))
            q = (1.-u)*np.asarray(start_q) + u*np.asarray(end_q)
            fingers = (1.-u)*np.asarray(start_fingers) + u*np.asarray(end_fingers)
            rows.append(self.clearance(base, q, fingers, pad_object, allowed_support))
        return {
            'min_self_m': min(x['min_self_m'] for x in rows),
            'min_env_m': min(x['min_env_m'] for x in rows),
            'min_object_m': min(x['min_object_m'] for x in rows),
            'self_pair': min(rows, key=lambda x:x['min_self_m'])['self_pair'],
            'env_pair': min(rows, key=lambda x:x['min_env_m'])['env_pair'],
            'object_pair': min(rows, key=lambda x:x['min_object_m'])['object_pair'],
            'samples': int(samples),
        }

    def solve_terminal(self, start_base, start_q, target_pad, target_fingers,
                       pad_object, joint_limits, allowed_support=None, search_xy=.24,
                       search_yaw=.5, max_nfev=60):
        start_base=np.asarray(start_base,float)
        start_q=np.asarray(start_q,float)
        target_pad=np.asarray(target_pad,float)
        target_fingers=np.asarray(target_fingers,float)
        limits=np.asarray(joint_limits,float)
        lo=np.r_[start_base+[-search_xy,-search_xy,-search_yaw],limits[:,0]]
        hi=np.r_[start_base+[search_xy,search_xy,search_yaw],limits[:,1]]
        pitch_idx=np.array([9,10,11])
        lower,upper=lo.copy(),hi.copy()
        lower[pitch_idx]=0.;upper[pitch_idx]=1.
        def encode(x):
            y=x.copy()
            y[pitch_idx]=lrt.torso_pitch_coordinates(x[pitch_idx],lo[pitch_idx],
                hi[pitch_idx],self.torso_pitch,encode=True)
            return y
        def decode(y):
            x=y.copy()
            x[pitch_idx]=lrt.torso_pitch_coordinates(y[pitch_idx],lo[pitch_idx],
                hi[pitch_idx],self.torso_pitch)
            return x
        target_p,target_R=target_pad[:3,3],target_pad[:3,:3]
        y0=np.clip(encode(np.r_[start_base,start_q]),lower+1e-7,upper-1e-7)
        def residual(y):
            x=decode(y);base,q=x[:3],x[3:]
            m=self.clearance(base,q,target_fingers,pad_object,allowed_support)
            p,R=m['pad'][:3,3],m['pad'][:3,:3]
            orient=Rotation.from_matrix(target_R.T@R).as_rotvec()
            return np.concatenate((100.*(p-target_p),8.*orient,
                3.*(base[:2]-start_base[:2]),
                np.array([3.*np.arctan2(np.sin(base[2]-start_base[2]),
                                        np.cos(base[2]-start_base[2]))]),
                .7*(q-start_q),42.*np.maximum(0.,.020-m['env_values']),
                42.*np.maximum(0.,.015-m['self_values']),
                42.*np.maximum(0.,.005-m['object_values'])))
        fit=least_squares(residual,y0,bounds=(lower,upper),method='trf',
            diff_step=1e-4,xtol=1e-4,ftol=1e-4,gtol=1e-4,max_nfev=max_nfev)
        x=decode(fit.x);base,q=x[:3],x[3:]
        m=self.clearance(base,q,target_fingers,pad_object,allowed_support)
        p,R=m['pad'][:3,3],m['pad'][:3,:3]
        report={'accepted':bool(m['min_self_m']>=self.self_margin),
            'position_error_mm':float(np.linalg.norm(p-target_p)*1000.),
            'orientation_error_deg':float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(target_R.T@R).as_rotvec()))),
            'min_self_mm':m['min_self_m']*1000.,
            'min_environment_mm':m['min_env_m']*1000.,
            'min_object_mm':m['min_object_m']*1000.,
            'base_displacement_m':float(np.linalg.norm(base[:2]-start_base[:2])),
            'base_goal':base.tolist(),'optimizer_nfev':int(fit.nfev)}
        return ((base,q) if report['accepted'] else None),report

    def solve_step(self, previous_base, previous_q, actual_base, actual_q, actual_fingers,
                   target_base, target_pad, target_fingers, pad_object, joint_limits,
                   allowed_support=None, max_nfev=24):
        previous_base = np.asarray(previous_base, float)
        previous_q = np.asarray(previous_q, float)
        actual_base = np.asarray(actual_base, float)
        actual_q = np.asarray(actual_q, float)
        actual_fingers = np.asarray(actual_fingers, float)
        target_base = np.asarray(target_base, float)
        target_pad = np.asarray(target_pad, float)
        target_fingers = np.asarray(target_fingers, float)
        limits = np.asarray(joint_limits, float)
        lo = np.r_[previous_base + [-.025/np.sqrt(2), -.025/np.sqrt(2), -np.radians(5.)],
                   np.maximum(limits[:, 0], previous_q-np.radians(10.))]
        hi = np.r_[previous_base + [.025/np.sqrt(2), .025/np.sqrt(2), np.radians(5.)],
                   np.minimum(limits[:, 1], previous_q+np.radians(10.))]
        if np.any(lo >= hi):
            return None, {'accepted': False, 'reason': 'empty_step_bounds'}
        pitch_idx = np.array([9, 10, 11])
        def encode(x):
            y=x.copy()
            y[pitch_idx]=lrt.torso_pitch_coordinates(x[pitch_idx],lo[pitch_idx],hi[pitch_idx],
                                                      self.torso_pitch,encode=True)
            return y
        def decode(y):
            x=y.copy()
            x[pitch_idx]=lrt.torso_pitch_coordinates(y[pitch_idx],lo[pitch_idx],hi[pitch_idx],
                                                      self.torso_pitch)
            return x
        lower, upper = lo.copy(), hi.copy()
        lower[pitch_idx]=0.;upper[pitch_idx]=1.
        target_p, target_R = target_pad[:3, 3], target_pad[:3, :3]
        target_seed = np.r_[np.clip(target_base,lo[:3],hi[:3]),np.clip(previous_q,lo[3:],hi[3:])]
        target_seed = np.clip(target_seed,lo+1e-8,hi-1e-8)
        y0 = np.clip(encode(target_seed),lower+1e-7,upper-1e-7)
        for scale in (1.,.5,.25):
            lo2,hi2=lo.copy(),hi.copy()
            lo2[:3]=previous_base+scale*(lo[:3]-previous_base)
            hi2[:3]=previous_base+scale*(hi[:3]-previous_base)
            lo2[3:]=np.maximum(limits[:,0],previous_q-scale*np.radians(10.))
            hi2[3:]=np.minimum(limits[:,1],previous_q+scale*np.radians(10.))
            lower,upper=lo2.copy(),hi2.copy()
            lower[pitch_idx]=0.;upper[pitch_idx]=1.
            if np.any(lo2>=hi2): continue
            def enc2(x):
                y=x.copy();y[pitch_idx]=lrt.torso_pitch_coordinates(x[pitch_idx],lo2[pitch_idx],hi2[pitch_idx],self.torso_pitch,encode=True);return y
            def dec2(y):
                x=y.copy();x[pitch_idx]=lrt.torso_pitch_coordinates(y[pitch_idx],lo2[pitch_idx],hi2[pitch_idx],self.torso_pitch);return x
            def res2(y):
                x=dec2(y);base=x[:3];q=x[3:]
                metrics=self.clearance(base,q,target_fingers,pad_object,allowed_support)
                p,R=metrics['pad'][:3,3],metrics['pad'][:3,:3]
                orient=Rotation.from_matrix(target_R.T@R).as_rotvec()
                parts=[100.*(p-target_p),8.*orient,3.*(base[:2]-target_base[:2]),
                       np.array([3.*np.arctan2(np.sin(base[2]-target_base[2]),np.cos(base[2]-target_base[2]))]),
                       .7*(q-previous_q),42.*np.maximum(0.,.020-metrics['env_values']),
                       42.*np.maximum(0.,.015-metrics['self_values']),
                       42.*np.maximum(0.,.005-metrics['object_values'])]
                return np.concatenate(parts)
            fit=least_squares(res2,np.clip(enc2(target_seed),lower+1e-7,upper-1e-7),
                              bounds=(lower,upper),method='trf',diff_step=1e-4,
                              xtol=1e-4,ftol=1e-4,gtol=1e-4,max_nfev=max_nfev)
            candidate=dec2(fit.x)
            base,q=candidate[:3],candidate[3:]
            sweep=self.sweep(actual_base,actual_q,actual_fingers,base,q,target_fingers,
                             pad_object,allowed_support,samples=7)
            ok=sweep['min_self_m']>=self.self_margin
            if ok:
                p,R=self._set(base,q,target_fingers)
                epos=float(np.linalg.norm(p-target_p)*1000.)
                eori=float(np.degrees(np.linalg.norm(Rotation.from_matrix(target_R.T@R).as_rotvec())))
                return (base,q),{'accepted':True,'reason':'accepted','scale':scale,
                    'position_error_mm':epos,'orientation_error_deg':eori,
                    'final_min_self_mm':float(self.clearance(base,q,target_fingers,pad_object,allowed_support)['min_self_m']*1000),
                    'swept_min_self_mm':sweep['min_self_m']*1000,'swept_pair':sweep['self_pair'],
                    'swept_min_environment_mm':sweep['min_env_m']*1000,
                    'swept_min_object_mm':sweep['min_object_m']*1000,
                    'environment_and_object_clearance_are_soft_residuals':True,
                    'sweep_samples':sweep['samples'],'optimizer_nfev':int(fit.nfev)}
        return None,{'accepted':False,'reason':'no_safe_step_within_velocity_bounds',
                     'swept_min_self_mm':float(sweep['min_self_m']*1000) if 'sweep' in locals() else None,
                     'swept_pair':sweep['self_pair'] if 'sweep' in locals() else None}

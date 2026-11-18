"""Verify the installed analytic Jacobian against SAPIEN FK; does not change solver."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import robotwin_r1_duck_grasp as G
import pinocchio


def main():
    scene = G.sapien.Scene()
    urdf = '/tmp/r1_pinocchio_probe.urdf'
    G.make_r1_urdf_abs(G.R1_URDF, urdf)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf)
    robot.set_root_pose(G.sapien.Pose(p=[0.3,-0.38,0.0], q=G.R1_BASE_Q))
    joint_idx = {j.get_name():i for i,j in enumerate(robot.get_active_joints())}
    sel = [joint_idx[n] for n in G.ARM_JOINTS]
    link_idx = next(i for i,l in enumerate(robot.get_links()) if l.get_name()==G.GRIP_LINK)
    link = robot.get_links()[link_idx]
    model = robot.create_pinocchio_model()
    base = robot.get_root_pose().to_transformation_matrix()
    q = robot.get_qpos().copy()
    rows = []
    def fk(v):
        robot.set_qpos(v)
        mat = link.get_entity_pose().to_transformation_matrix()
        return mat[:3,3] + mat[:3,0]*G.PAD_FWD, mat
    for x in (G.R1_LEFT_START, np.array([1.339,2.245,-1.533,2.465,0.415,0.982]),
              np.array([1.403,2.291,-1.561,2.659,0.635,-2.541])):
        q[sel] = x
        p, world = fk(q)
        model.compute_forward_kinematics(q)
        pin_world = base @ model.get_link_pose(link_idx).to_transformation_matrix()
        pose_error = float(np.max(np.abs(pin_world-world)))
        jl = model.compute_single_link_local_jacobian(q, link_idx)[:,sel]
        r = world[:3,:3]
        angular = r @ jl[3:]
        analytic = r @ jl[:3] + np.cross(angular.T, r[:,0]*G.PAD_FWD).T
        numeric = np.zeros((3,6))
        h = 1e-3  # SAPIEN poses are float32; smaller steps amplify pose quantisation.
        for i,j in enumerate(sel):
            plus,minus = q.copy(),q.copy()
            plus[j] += h
            minus[j] -= h
            numeric[:,i] = (fk(plus)[0]-fk(minus)[0])/(2*h)
        jac_error = float(np.max(np.abs(analytic-numeric)))
        assert pose_error < 1e-5, pose_error
        assert jac_error < 1e-3, jac_error
        rows.append(dict(q=x.tolist(), pose_max_abs_error=pose_error,
                         pad_jacobian_max_abs_error=jac_error))
    report = dict(pinocchio_version=pinocchio.__version__, checks=rows,
                  result='PASS', solver_changed=False,
                  method='body Jacobian rotated to world, including gripper pad offset')
    (ROOT/'results/ik_agent_20261022/pinocchio_probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""robotwin_r1_duck_dataset.py — real-to-sim 数据生成器 (单场景验证)。

链路:
  真实第一人称视频 -> {场景 GLB, 真人手轨迹 NPZ}
    -> 复用 robotwin_r1_duck_grasp.build_episode() 在 RoboTwin(SAPIEN) 里
       用 R1 复现抓取 (IK + PD 物理)
    -> 逐步记录 state/action/多相机 -> 写出 RoboTwin 格式 hdf5

输出 (RoboTwin schema):
  action/{left,right}_arm_joint_states   (T,6)   关节角指令
  action/{left,right}_ee_joint_states    (T,1)   夹爪指令
  action/{left,right}_ee_poses           (T,7)   末端位姿指令 xyz+quat(wxyz)
  state/  同上结构                              实际到达的状态
  vision/<cam>/{colors, intrinsic_matrix, extrinsics_matrix, shape}
  object/pink_duck_pose                  (T,7)   物体位姿 (附加, 监督用)
  additional_info/{frequency, instructions, data_format_version}

用法:
  python scripts/robotwin_r1_duck_dataset.py [--out demo_output/robotwin_r1_duck_episode0.hdf5]
"""
import argparse
import os
import sys

import numpy as np
import h5py
import cv2
import sapien.core as sapien

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import robotwin_r1_duck_grasp as G  # noqa: E402

# ---- 统一入口: 给定 --task 时所有 episode 走 pipeline.run_episode ----
TASK_CFG = None


def _run_episode(on_step=None, collect_frames=False):
    if TASK_CFG is not None:
        import pipeline as PIPE
        ep, _ = PIPE.run_episode(TASK_CFG, on_step=on_step,
                                 collect_frames=collect_frames, setup=False)
        return ep
    return G.build_episode(on_step=on_step, collect_frames=collect_frames)

# 记录用相机尺寸 (训练用, 比视频小以省空间)
CAM_W, CAM_H = 320, 240
HEAD_FOVY = np.deg2rad(50)
WRIST_FOVY = np.deg2rad(70)
# 腕部相机相对 gripper_link 的安装偏移 (相机坐标系: +x 前, +y 左, +z 上;
# gripper_link 的接近轴 = +x, 故旋转取单位阵即可让相机沿接近轴前视)
WRIST_OFF = np.eye(4)
WRIST_OFF[:3, 3] = [-0.03, 0.0, 0.06]

INSTRUCTION = "Grasp the pink duck on the table and lift it up."


def quat_wxyz(pose):
    """sapien.Pose -> [x,y,z, qw,qx,qy,qz]"""
    return np.concatenate([pose.p, pose.q]).astype(np.float32)


def encode_jpg(rgb):
    """RGB uint8 -> JPEG 编码的 uint8 字节数组 (hdf5 vlen uint8 存储)。"""
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


class Recorder:
    """逐步收集 RoboTwin 格式 episode 数据。"""

    def __init__(self):
        self.cams = {}
        self.right_arm_s, self.right_grip_s, self.right_ee_s = [], [], []
        self.left_arm_s, self.left_grip_s, self.left_ee_s = [], [], []
        self.left_arm_a, self.left_grip_a, self.left_ee_a = [], [], []
        self.duck_pose = []
        self.vision = {}
        self._qidx = None

    def _setup_idx(self, ctx):
        # 用 robot.get_active_joints() 的顺序建立 关节名 -> qpos 索引 映射
        all_j = ctx["robot"].get_active_joints()
        n2i = {j.get_name(): i for i, j in enumerate(all_j)}
        self._qidx = {
            "ra": [n2i[f"right_arm_joint{i}"] for i in range(1, 7)],
            "la": [n2i[f"left_arm_joint{i}"] for i in range(1, 7)],
            "rg": [n2i["right_gripper_finger_joint1"], n2i["right_gripper_finger_joint2"]],
            "lg": [n2i["left_gripper_finger_joint1"], n2i["left_gripper_finger_joint2"]],
        }

    def _ensure_cams(self, ctx):
        if self.cams:
            return
        scene = ctx["scene"]
        head = scene.add_camera(name="ds_cam_head", width=CAM_W, height=CAM_H,
                                fovy=HEAD_FOVY, near=0.05, far=100)
        head.entity.set_pose(G.look_at_pose([1.55, -1.35, 1.30], [0.0, -0.30, 0.95]))
        third = scene.add_camera(name="ds_cam_third_view", width=CAM_W, height=CAM_H,
                                 fovy=HEAD_FOVY, near=0.05, far=100)
        third.entity.set_pose(G.look_at_pose([-1.45, -0.35, 1.25], [0.0, -0.30, 0.95]))
        wrist = scene.add_camera(name="ds_cam_left_wrist", width=CAM_W, height=CAM_H,
                                 fovy=WRIST_FOVY, near=0.02, far=100)
        self.cams = {"cam_head": head, "cam_third_view": third, "cam_left_wrist": wrist}
        for k in self.cams:
            self.vision[k] = {"colors": [], "intrinsic": [], "extrinsic": [], "shape": None}

    def __call__(self, ctx, k, fi):
        robot = ctx["robot"]
        g_link = ctx["g_link"]
        if self._qidx is None:
            self._setup_idx(ctx)
            self._ensure_cams(ctx)

        scene = ctx["scene"]
        # 腕部相机跟随工作臂(左)夹爪 —— 本任务执行臂为左臂, 相机键与绑定臂必须一致
        gT = g_link.get_pose().to_transformation_matrix()
        self.cams["cam_left_wrist"].entity.set_pose(sapien.Pose(gT @ WRIST_OFF))

        # --- 渲染 ---
        scene.update_render()
        for name, c in self.cams.items():
            c.take_picture()
        for name, c in self.cams.items():
            rgb = (np.clip(c.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8)
            K = np.array(c.get_intrinsic_matrix(), dtype=np.float32)
            # SAPIEN 返回 3x4, pad 成 4x4 以对齐 RoboTwin schema
            E34 = np.array(c.get_extrinsic_matrix(), dtype=np.float32)
            E = np.eye(4, dtype=np.float32)
            E[:3, :] = E34
            self.vision[name]["colors"].append(encode_jpg(rgb))
            self.vision[name]["intrinsic"].append(K)
            self.vision[name]["extrinsic"].append(E)
            self.vision[name]["shape"] = list(rgb.shape)

        q = robot.get_qpos()

        # --- state (实际到达) ---
        self.right_arm_s.append(q[self._qidx["ra"]].astype(np.float32))
        self.right_grip_s.append(np.float32(q[self._qidx["rg"]].mean()))
        self.right_ee_s.append(quat_wxyz(g_link.get_pose()))
        lg_link = robot.find_link_by_name("left_gripper_link")
        self.left_arm_s.append(q[self._qidx["la"]].astype(np.float32))
        self.left_grip_s.append(np.float32(q[self._qidx["lg"]].mean()))
        self.left_ee_s.append(quat_wxyz(lg_link.get_pose()))

        # --- action (指令): 左臂为目标关节角; 末端位姿由目标关节角干跑 FK 得到 ---
        arm_t = ctx["xs"][k].astype(np.float32)
        k_close, k_open = ctx["k_close"], ctx["k_open"]
        if k < k_close:
            gfrac = 0.0
        elif k < k_open:
            gfrac = min(1.0, (k - k_close) / 3.0)
        else:
            gfrac = max(0.0, 1.0 - (k - k_open) / 2.0)
        grip_t = np.float32(ctx["grip_open_q"] * (1 - gfrac) + ctx["grip_hold_q"] * gfrac)
        self.left_arm_a.append(arm_t)
        self.left_grip_a.append(grip_t)
        q_save = q.copy()
        q_try = q.copy()
        q_try[self._qidx["la"]] = arm_t
        q_try[self._qidx["lg"]] = grip_t
        robot.set_qpos(q_try)
        self.left_ee_a.append(quat_wxyz(g_link.get_pose()))
        robot.set_qpos(q_save)          # 还原, 不影响仿真

        # --- 物体位姿 (监督) ---
        self.duck_pose.append(quat_wxyz(ctx["duck"].get_pose()))

    def to_arrays(self):
        d = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
        out = {
            "action/left_arm_joint_states": d(self.left_arm_a),
            "action/left_ee_joint_states": d(self.left_grip_a)[:, None],
            "action/left_ee_poses": d(self.left_ee_a),
            "action/right_arm_joint_states": d(self.right_arm_s),    # 右臂不动, 指令=状态
            "action/right_ee_joint_states": d(self.right_grip_s)[:, None],
            "action/right_ee_poses": d(self.right_ee_s),
            "state/left_arm_joint_states": d(self.left_arm_s),
            "state/left_ee_joint_states": d(self.left_grip_s)[:, None],
            "state/left_ee_poses": d(self.left_ee_s),
            "state/right_arm_joint_states": d(self.right_arm_s),
            "state/right_ee_joint_states": d(self.right_grip_s)[:, None],
            "state/right_ee_poses": d(self.right_ee_s),
            "object/pink_duck_pose": d(self.duck_pose),
        }
        return out


def write_hdf5(path, data, vision, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr)
        for cam, v in vision.items():
            g = f.create_group(f"vision/{cam}")
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            g.create_dataset("colors", data=np.array(v["colors"], dtype=object), dtype=dt)
            g.create_dataset("intrinsic_matrix", data=np.asarray(v["intrinsic"], dtype=np.float32))
            g.create_dataset("extrinsics_matrix", data=np.asarray(v["extrinsic"], dtype=np.float32))
            g.create_dataset("shape", data=np.array(v["shape"], dtype=np.int32))
        ai = f.create_group("additional_info")
        ai.create_dataset("frequency", data=np.int32(meta["frequency"]))
        ai.create_dataset("instructions", data=meta["instruction"])
        f.create_dataset("data_format_version", data="robotwin-r1-v1.0")
        f.attrs["source"] = meta["source"]
        f.attrs["robot"] = "Galaxea_R1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(G.OUT_DIR, "robotwin_r1_duck_episode0.hdf5"))
    ap.add_argument("--task", default=None,
                    help="任务 JSON(如 configs/duck_to_plate.json)。给定时走 pipeline 统一入口。")
    args = ap.parse_args()

    global TASK_CFG
    if args.task:
        import pipeline as PIPE
        from task_config import load_task
        TASK_CFG = load_task(args.task)
        PIPE.apply_adapters(TASK_CFG, verbose=True)
        print(f"[task] 统一入口模式: {args.task}")

    rec = Recorder()
    print("运行物理仿真并逐步记录...")
    ep = _run_episode(on_step=rec, collect_frames=False)

    data = rec.to_arrays()
    T = data["state/left_arm_joint_states"].shape[0]

    # ---- 校验 ----
    print("\n" + "=" * 66)
    print("数据校验:")
    print(f"  episode 长度 T = {T} 帧, 控制频率 = {G.CTRL_FREQ} Hz")
    ee = data["state/left_ee_poses"]
    print(f"  左末端位姿: 起={np.round(ee[0],3).tolist()}  末={np.round(ee[-1],3).tolist()}")
    grip = data["state/left_ee_joint_states"][:, 0]
    print(f"  左夹爪开合: 起={grip[0]:.4f}  最小(夹持)={grip.min():.4f}  末={grip[-1]:.4f} "
          f"(张开 {G.GRIP_OPEN_Q} -> 夹持 {G.GRIP_HOLD_Q} -> 释放)")
    duck = data["object/pink_duck_pose"]
    lift = (duck[:, 2].max() - duck[ep["k_g"], 2]) * 100
    print(f"  粉鸭抬升(max) = {lift:.2f} cm   末态高度 = {duck[-1,2]:.4f} (释放后应落回桌面 {G.TABLE_Z:.2f})")
    print(f"  action/state 关节角 最大差 = "
          f"{np.abs(data['action/left_arm_joint_states']-data['state/left_arm_joint_states']).max():.4f} rad")
    for cam, v in rec.vision.items():
        print(f"  相机 {cam}: {len(v['colors'])} 帧, shape={v['shape']}, "
              f"K.fx={v['intrinsic'][0][0,0]:.1f}, E[0,3]={v['extrinsic'][0][0,3]:.3f}")

    meta = {"frequency": G.CTRL_FREQ, "instruction": INSTRUCTION,
            "source": "final_scene.glb(toy_0=pink_duck) + mano_ras_reconstruction_0_113.npz"}
    write_hdf5(args.out, data, rec.vision, meta)
    print(f"\n已写出: {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"  组: {list(data.keys())[:6]} ...")
    print(f"  相机: {list(rec.vision.keys())}")


if __name__ == "__main__":
    main()

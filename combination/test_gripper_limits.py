#!/usr/bin/env python3
"""
test_gripper_limits.py — 三种夹爪极限标定方法的 SAPIEN 对比测试

用途:
    加载 HaWoR MANO 数据，用三种方法计算拇指-食指最大张度，
    生成不同夹爪关节 limit 的 URDF，并在 SAPIEN 中渲染对比视频。

三种方法:
    A — 直接取当前数据帧的最大指尖距 (最简单)
    B — 固定 betas，对 PCA 空间做 Monte Carlo 采样找极限 (推荐)
    C — 固定 betas，对关键 PCA 维度做系统搜索 (网格 + 极值)

用法:
    # 自动检测活跃手 (推荐)
    python test_gripper_limits.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --output-dir /tmp/gripper_test

    # 手动指定手 (0=左手, 1=右手)
    python test_gripper_limits.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --hand-idx 0 \\
        --output-dir /tmp/gripper_test

关于自动检测:
    脚本会自动调用 detect_hands() (与 002_render_scene.py 相同逻辑),
    根据 pred_valid、手腕运动幅度、位置分布来判断哪只手是真实的。
    HaWoR 数据总是有 (2, N, ...) 形状, dim0=0 对应左手, dim0=1 对应右手。
    即使只有一只手在画面中, HaWoR 也会输出两手的数据, 但不存在的那只手
    参数会接近默认值 (手腕在原点附近、betas 接近零), detect_hands() 会
    通过运动幅度(>3cm)和离原点距离(>5cm)来排除这些"幽灵手"。

输出 (每段约 30 秒视频, 1280x720):
    /tmp/gripper_test/method_A_*.mp4    — 用数据最大张度
    /tmp/gripper_test/method_B_*.mp4    — 用 Monte Carlo 极限张度
    /tmp/gripper_test/method_C_*.mp4    — 用 PCA 系统搜索张度
    /tmp/gripper_test/summary.json      — 数值汇总
    /tmp/gripper_test/gripper_urdfs/    — 三种方法的 URDF 文件
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
if os.path.exists(_nvidia_icd):
    os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
else:
    _intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
    os.environ['VK_ICD_FILENAMES'] = _intel_icd

import sys
import json
import argparse
import logging
from pathlib import Path
import math

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap
from tqdm import trange
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dex-retargeting" / "example" / "position_retargeting"))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))

from hand_track.common import detect_hands, _find_reconstruction_file

# ─── 核心常量 ────────────────────────────────────────────────────────────────
FINGER_BASE_DIST = 0.026906  # 夹爪闭合时两指尖距 (米)
MANO_THUMB_TIP = 4           # MANO 关节号: 拇指尖
MANO_INDEX_TIP = 8           # MANO 关节号: 食指尖
MANO_WRIST = 0               # MANO 关节号: 手腕

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# 原始 URDF 路径
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
R1_GRIPPER_URDF = (
    GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1"
    / "meshes" / "r1_mjcf_2nyrsmeg" / "r1_gripper_only_right.urdf"
)

logger = logging.getLogger("gripper_test")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


# ─── Step 1: 加载 HaWoR 数据 ─────────────────────────────────────────────────

def load_hawor_mano_data(hawor_dir, hand_idx=1):
    """加载 HaWoR 重建数据，对每一帧跑 MANO forward，返回 SAPIEN 坐标系关节"""
    from mano_layer import MANOLayer

    rec_dir = Path(hawor_dir) / "reconstruction"
    candidates = sorted(rec_dir.glob("hawor_results_*.npz"), reverse=True)
    if not candidates:
        candidates = sorted(Path(hawor_dir).glob("hawor_results_*.npz"), reverse=True)
    rec_file = candidates[0]
    logger.info(f"数据文件: {rec_file}")

    rec = np.load(str(rec_file), allow_pickle=True)
    trans = rec['pred_trans'][hand_idx]
    rot = rec['pred_rot'][hand_idx]
    hand_pose = rec['pred_hand_pose'][hand_idx]
    betas = rec['pred_betas'][hand_idx]
    valid = rec['pred_valid'][hand_idx]

    n_frames = trans.shape[0]
    logger.info(f"  总帧数={n_frames}, hand_idx={hand_idx}, 有效帧={valid.sum()}")

    # 用 betas 均值初始化 MANO layer
    betas_mean = np.mean(betas, axis=0)
    side = "right" if hand_idx == 1 else "left"
    mano_layer = MANOLayer(side, betas_mean)

    # 对所有有效帧做 forward
    joints_list, vertices_list = [], []
    for i in range(n_frames):
        if not valid[i]:
            continue
        # 完整 48 维 pose: 手腕旋转(3) + 手指 PCA(45)
        pose_full = np.concatenate([rot[i], hand_pose[i]])[:48]
        p = torch.from_numpy(pose_full[None, :].astype(np.float32))
        t = torch.from_numpy(trans[i:i+1, :3].astype(np.float32))

        v, j = mano_layer(p, t)
        j_sapien = (RXWORLD_TO_SAPIEN @ j.cpu().numpy()[0].T).T
        v_sapien = (RXWORLD_TO_SAPIEN @ v.cpu().numpy()[0].T).T
        joints_list.append(j_sapien)
        vertices_list.append(v_sapien)

    joints_all = np.array(joints_list)
    vertices_all = np.array(vertices_list)

    # 归一化: 第一帧手腕为原点
    origin = joints_all[0, MANO_WRIST].copy()
    joints_all = joints_all - origin[None, :]
    vertices_all = vertices_all - origin[None, :]
    logger.info(f"  有效帧数: {len(joints_all)}, 手腕原点归一化完成")

    return dict(
        joints=joints_all, vertices=vertices_all,
        betas_mean=betas_mean, side=side,
        n_frames=len(joints_all),
        mano_layer=mano_layer,
    )


# ─── Step 2: 三种 D_max 计算 ────────────────────────────────────────────────

def method_A_max_in_data(joints_all):
    """方法A: 直接取数据中拇指尖-食指尖距离的最大值"""
    thumb = joints_all[:, MANO_THUMB_TIP]
    index = joints_all[:, MANO_INDEX_TIP]
    dists = np.linalg.norm(thumb - index, axis=1)
    return {
        "D_max": float(dists.max()),
        "D_p95": float(np.percentile(dists, 95)),
        "D_p99": float(np.percentile(dists, 99)),
        "D_mean": float(dists.mean()),
        "D_min": float(dists.min()),
        "D_std": float(dists.std()),
    }


def method_B_monte_carlo(mano_layer, betas_mean, n_samples=50000, sigma_range=3.0):
    """方法B: 固定 betas, PCA 空间均匀采样, 找最大张度"""
    from manopth.manolayer import ManoLayer as ManoLayerRaw

    best_dist, best_pose = 0.0, None
    n = n_samples

    for i in range(n):
        pca = np.random.uniform(-sigma_range, sigma_range, size=45).astype(np.float32)
        wrist = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        pose = np.concatenate([wrist, pca])
        p = torch.from_numpy(pose).unsqueeze(0).float()
        t = torch.zeros(1, 3)
        _, j = mano_layer(p, t)
        dist = float(np.linalg.norm(j.cpu().numpy()[0, MANO_THUMB_TIP]
                                    - j.cpu().numpy()[0, MANO_INDEX_TIP]))
        if dist > best_dist:
            best_dist = dist
            best_pose = pose.copy()
        if (i + 1) % 20000 == 0:
            logger.info(f"  采样 {i+1}/{n}, 当前最佳 D={best_dist*1000:.2f}mm")

    # 验证最佳姿态
    p = torch.from_numpy(best_pose).unsqueeze(0).float()
    _, j = mano_layer(p, t)
    best_dist = float(np.linalg.norm(j.cpu().numpy()[0, MANO_THUMB_TIP]
                                     - j.cpu().numpy()[0, MANO_INDEX_TIP]))
    return {"D_max": best_dist, "n_samples": n}


def method_C_grid_search(mano_layer, betas_mean, key_dims=5, sigma_range=3.0):
    """方法C: 对关键 PCA 维度做网格搜索
    前 key_dims 个 PCA 维度通常控制拇指和食指的主要形态"""
    from itertools import product

    n_dims = key_dims
    # 每个维度取 7 个值: -3σ, -2σ, -σ, 0, σ, 2σ, 3σ
    grid_vals = [-sigma_range, -2.0, -1.0, 0.0, 1.0, 2.0, sigma_range]
    total = len(grid_vals) ** n_dims

    best_dist, best_pose = 0.0, None
    count = 0
    for combo in product(grid_vals, repeat=n_dims):
        pca = np.zeros(45, dtype=np.float32)
        pca[:n_dims] = np.array(combo)
        pose = np.concatenate([np.zeros(3, dtype=np.float32), pca])
        p = torch.from_numpy(pose).unsqueeze(0).float()
        t = torch.zeros(1, 3)
        _, j = mano_layer(p, t)
        dist = float(np.linalg.norm(j.cpu().numpy()[0, MANO_THUMB_TIP]
                                    - j.cpu().numpy()[0, MANO_INDEX_TIP]))
        if dist > best_dist:
            best_dist = dist
            best_pose = pose.copy()
        count += 1

    return {"D_max": best_dist, "n_searched": count, "total_combos": total}


# ─── Step 3: 生成 URDF ──────────────────────────────────────────────────────

def compute_joint_limit(d_max):
    """从指尖距反推单侧 prismatic 关节 limit"""
    single = (d_max - FINGER_BASE_DIST) / 2.0
    return max(0.0, float(single))


def generate_urdf_with_limit(joint_limit, urdf_out_dir):
    """基于原始 URDF 模板, 替换 upper 限制 + mesh 路径"""
    import re
    xml = R1_GRIPPER_URDF.read_text()

    # 替换 mesh 路径为绝对路径
    R1_MESH_DIR_STR = str(R1_MESH_DIR)
    # 原始 URDF 使用 "right_xxx.STL" 这样的相对路径
    xml = re.sub(r'mesh filename="([^"]+\.STL)"',
                 lambda m: f'mesh filename="{R1_MESH_DIR_STR}/{m.group(1)}"',
                 xml)

    # fixed → prismatic (某些版本可能是 fixed)
    for i in [1, 2]:
        pattern_fixed = f'right_gripper_finger_joint{i}" type="fixed"'
        if pattern_fixed in xml:
            xml = xml.replace(
                pattern_fixed,
                f'right_gripper_finger_joint{i}" type="prismatic"'
            )

    # 替换 upper 限制
    new_limit = f'{joint_limit:.6f}'
    xml = xml.replace('upper="0.05"', f'upper="{new_limit}"')

    # 更新 velocity 成比例
    vel_mult = max(1.0, joint_limit / 0.05)
    new_vel = 0.25 * vel_mult
    xml = xml.replace('velocity="0.25"', f'velocity="{new_vel:.4f}"')

    urdf_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = urdf_out_dir / f"r1_gripper_limit_{joint_limit:.4f}.urdf"
    out_path.write_text(xml)
    return str(out_path)


# ─── Step 4: SAPIEN 渲染 ────────────────────────────────────────────────────

def compute_smooth_shading_normal_np(vertices, indices):
    v1, v2, v3 = vertices[indices[:, 0]], vertices[indices[:, 1]], vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)
    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    norms = np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    vertex_normal = np.where(norms > 1e-12, vertex_normal / norms, vertex_normal)
    return vertex_normal


class TestRenderer:
    """渲染 MANO 手 + 夹爪 对比视频 (使用 SAPIEN 3.0 内部渲染 API)"""

    def __init__(self, gripper_urdf, method_name, video_path,
                 joint_limit, d_max, w=1280, h=720):
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)
        self.scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.25], ground_color=[0.15, 0.15, 0.2])
        )
        self.scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
        self.scene.add_directional_light([-1, 0.5, -1], [1.2, 1.2, 1.2], shadow=False)
        self.scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])

        # Ground
        gm = sapien.render.RenderMaterial()
        gm.set_base_color([0.4, 0.4, 0.4, 1]); gm.set_roughness(0.8)
        self.scene.add_ground(-0.6, render_material=gm)

        # Gripper
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self.robot = loader.load(gripper_urdf)

        # Camera: 从右侧俯视
        self.camera = self.scene.add_camera("cam", w, h, math.pi / 4, 0.01, 10.0)
        self.camera.set_local_pose(
            sapien.Pose([0.4, 0.15, 0.5], [0.7071, 0, 0, -0.7071])
        )

        # Video writer
        self.writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
        )
        self.w, self.h = w, h
        self.joint_limit = joint_limit
        self.d_max = d_max

        # Title
        self.title = (
            f" [{method_name}]  "
            f"D_max={d_max*1000:.1f}mm  |  "
            f"Limit={joint_limit*1000:.1f}mm  |  "
            f"MaxFinger={d_max*1000:.1f}mm  |  "
            f"FingerClose={FINGER_BASE_DIST*1000:.1f}mm"
        )
        logger.info(f"  {self.title}")

        # SAPIEN 3.0 内部渲染器
        self.internal = self.scene.render_system._internal_scene
        self.context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)
        self.mat_thumb = self.context.create_material(
            np.zeros(4), np.array([1.0, 0.2, 0.2, 1]), 0.0, 0.2, 0)
        self.mat_index = self.context.create_material(
            np.zeros(4), np.array([0.2, 0.8, 0.2, 1]), 0.0, 0.2, 0)
        self.mat_wrist = self.context.create_material(
            np.zeros(4), np.array([0.2, 0.4, 1.0, 1]), 0.0, 0.2, 0)
        self.mat_line = self.context.create_material(
            np.zeros(4), np.array([1.0, 1.0, 0.0, 1]), 0.0, 0.2, 0)

        # 预计算球体 mesh（避免每帧重复创建）
        self.sphere_mesh = self.context.create_uvsphere_mesh(12, 6)
        # 预计算手网法线
        self.face = None  # 在 frame 中传入

        self._nodes = []

    def _clear_objs(self):
        while self._nodes:
            node = self._nodes.pop()
            self.internal.remove_node(node)

    def render_mano_mesh(self, verts, face):
        """渲染 MANO 手部网格"""
        normals = compute_smooth_shading_normal_np(verts, face)
        mesh = self.context.create_mesh_from_array(
            np.ascontiguousarray(verts.astype(np.float32)), face,
            np.ascontiguousarray(normals.astype(np.float32)))
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal.add_node()
        node.set_position([0, 0, 0])
        obj = self.internal.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self._nodes.append(node)

    def render_sphere(self, pos, color_mat, radius=0.008):
        """渲染小球标记"""
        model = self.context.create_model([self.sphere_mesh], [color_mat])
        node = self.internal.add_node()
        node.set_position(pos.tolist())
        node.set_scale([radius, radius, radius])
        obj = self.internal.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0
        self._nodes.append(node)

    def render_line(self, p0, p1, radius=0.002):
        """在两点之间渲染胶囊体（代替旧版 LINE）"""
        mid = (p0 + p1) / 2.0
        length = np.linalg.norm(p1 - p0)
        if length < 1e-6:
            return
        capsule = self.context.create_capsule_mesh(radius, length / 2.0, 8, 4)
        model = self.context.create_model([capsule], [self.mat_line])
        node = self.internal.add_node()
        node.set_position(mid.tolist())
        direction = (p1 - p0) / length
        z_axis = np.array([0.0, 0.0, 1.0])
        from pytransform3d import rotations as pr
        rot_axis = np.cross(z_axis, direction)
        rot_axis_len = np.linalg.norm(rot_axis)
        if rot_axis_len > 1e-6:
            rot_axis_n = rot_axis / rot_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            q_wxyz = pr.quaternion_from_axis_angle(
                np.array([rot_axis_n[0], rot_axis_n[1], rot_axis_n[2], angle]))
            q_xyzw = pr.quaternion_xyzw_from_wxyz(q_wxyz)
            node.set_rotation(q_xyzw.tolist())
        elif np.dot(z_axis, direction) < 0:
            node.set_rotation([0, 1, 0, 0])
        obj = self.internal.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0
        self._nodes.append(node)

    def gripper_qpos(self, joints):
        thumb, index = joints[MANO_THUMB_TIP], joints[MANO_INDEX_TIP]
        dist = float(np.linalg.norm(thumb - index))
        req = dist - FINGER_BASE_DIST
        single = max(0.0, min(self.joint_limit, req / 2.0))
        return np.array([single, single], dtype=np.float64)

    def frame(self, joints, verts, face):
        self._clear_objs()
        self.robot.set_qpos(self.gripper_qpos(joints))
        self.scene.step()
        self.render_mano_mesh(verts, face)
        self.render_sphere(joints[MANO_WRIST], self.mat_wrist)
        self.render_sphere(joints[MANO_THUMB_TIP], self.mat_thumb)
        self.render_sphere(joints[MANO_INDEX_TIP], self.mat_index)
        self.render_line(joints[MANO_THUMB_TIP], joints[MANO_INDEX_TIP])
        self.scene.update_render()
        self.camera.take_picture()
        rgb = np.ascontiguousarray(
            (np.clip(self.camera.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
        )
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        cv2.putText(bgr, self.title, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        thumb, index = joints[MANO_THUMB_TIP], joints[MANO_INDEX_TIP]
        cur_dist = float(np.linalg.norm(thumb - index))
        cv2.putText(bgr, f"CurDist={cur_dist*1000:.1f}mm",
                     (10, self.h - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        self.writer.write(bgr)

    def close(self):
        self.writer.release()
        logger.info("  视频已保存")


def render_video(mano_data, joint_limit, d_max, method_name, video_path):
    """渲染单个方法的对比视频"""
    urdf_out_dir = Path(video_path).parent / "gripper_urdfs"
    urdf_path = generate_urdf_with_limit(joint_limit, urdf_out_dir)
    logger.info(f"  URDF: {urdf_path}")

    joints = mano_data["joints"]
    verts = mano_data["vertices"]
    face = mano_data["mano_layer"].f.cpu().numpy()
    n = joints.shape[0]

    renderer = TestRenderer(urdf_path, method_name, str(video_path),
                            joint_limit, d_max)
    logger.info(f"  渲染 {n} 帧 ...")
    for i in trange(n, desc=f"  {method_name}"):
        renderer.frame(joints[i], verts[i], face)
    renderer.close()


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="三种夹爪极限标定方法的 SAPIEN 对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
自动检测说明:
  --hand-idx 默认 None, 脚本会自动调用 detect_hands() 识别活跃手
  如需手动指定, 传入 --hand-idx 0(左手) 或 1(右手)
        """)
    parser.add_argument("--hawor-dir", required=True)
    parser.add_argument("--hand-idx", type=int, default=None,
                        help="手部索引, 默认 None=自动检测 (0=左手, 1=右手)")
    parser.add_argument("--output-dir", default="/tmp/gripper_test")
    parser.add_argument("--mc-samples", type=int, default=50000)
    parser.add_argument("--skip-mc", action="store_true",
                        help="跳过方法B (Monte Carlo 较慢)")
    parser.add_argument("--grid-dims", type=int, default=4,
                        help="方法C 网格搜索的关键 PCA 维度数 (默认4)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 自动检测活跃手 ──
    if args.hand_idx is None:
        logger.info("=" * 60)
        logger.info("自动检测活跃手...")
        logger.info("=" * 60)
        active_hands = detect_hands(args.hawor_dir)
        logger.info(f"检测到活跃手: {active_hands} (0=左手, 1=右手)")
        if not active_hands:
            logger.error("未检测到活跃手, 请检查数据或手动指定 --hand-idx")
            sys.exit(1)
        # 如果检测到双手, 只取第一只 (也可扩展为双手并行)
        if len(active_hands) > 1:
            logger.warning(f"检测到双手 {active_hands}, 仅处理第一只: {active_hands[0]}")
        hand_idx = active_hands[0]
        logger.info(f"使用 hand_idx={hand_idx}")
    else:
        hand_idx = args.hand_idx

    # ── 1. 加载 ──
    logger.info("=" * 60)
    logger.info("Step 1: 加载 HaWoR 数据 + MANO forward")
    logger.info("=" * 60)
    md = load_hawor_mano_data(args.hawor_dir, hand_idx)
    joints = md["joints"]
    summary = {}

    # ── 2. 计算 D_max ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 计算三种方法的最大张度")
    logger.info("=" * 60)

    # 方法 A
    res_a = method_A_max_in_data(joints)
    limit_a = compute_joint_limit(res_a["D_max"])
    logger.info(f"  A: D_max={res_a['D_max']*1000:.2f}mm, "
                f"P95={res_a['D_p95']*1000:.2f}mm, "
                f"P99={res_a['D_p99']*1000:.2f}mm, "
                f"mean={res_a['D_mean']*1000:.2f}mm, "
                f"min={res_a['D_min']*1000:.2f}mm, "
                f"std={res_a['D_std']*1000:.2f}mm")
    logger.info(f"  → JointLimit={limit_a*1000:.2f}mm")
    summary["method_A_data_max"] = {
        "D_max_mm": round(res_a["D_max"]*1000, 2),
        "D_p95_mm": round(res_a["D_p95"]*1000, 2),
        "D_p99_mm": round(res_a["D_p99"]*1000, 2),
        "D_mean_mm": round(res_a["D_mean"]*1000, 2),
        "D_min_mm": round(res_a["D_min"]*1000, 2),
        "D_std_mm": round(res_a["D_std"]*1000, 2),
        "joint_limit_mm": round(limit_a*1000, 2),
        "description": "数据中实际出现的最大指尖距",
    }

    # 方法 B
    d_max_b, limit_b = None, None
    if not args.skip_mc:
        logger.info("")
        res_b = method_B_monte_carlo(md["mano_layer"], md["betas_mean"],
                                     n_samples=args.mc_samples)
        limit_b = compute_joint_limit(res_b["D_max"])
        logger.info(f"  B: D_max={res_b['D_max']*1000:.2f}mm (采样{res_b['n_samples']}次)")
        logger.info(f"  → JointLimit={limit_b*1000:.2f}mm")
        summary["method_B_monte_carlo"] = {
            "D_max_mm": round(res_b["D_max"]*1000, 2),
            "joint_limit_mm": round(limit_b*1000, 2),
            "n_samples": res_b["n_samples"],
            "description": f"PCA 空间 Monte Carlo 均匀采样 {res_b['n_samples']} 次",
        }
    else:
        logger.info("  跳过方法B (add --skip-mc 已设, 或去掉该参数)")

    # 方法 C
    logger.info("")
    res_c = method_C_grid_search(md["mano_layer"], md["betas_mean"],
                                 key_dims=args.grid_dims)
    limit_c = compute_joint_limit(res_c["D_max"])
    logger.info(f"  C: D_max={res_c['D_max']*1000:.2f}mm "
                f"(搜索{res_c['n_searched']}/{res_c['total_combos']}组合, "
                f"key_dims={args.grid_dims})")
    logger.info(f"  → JointLimit={limit_c*1000:.2f}mm")
    summary["method_C_grid_search"] = {
        "D_max_mm": round(res_c["D_max"]*1000, 2),
        "joint_limit_mm": round(limit_c*1000, 2),
        "n_searched": res_c["n_searched"],
        "total_combos": res_c["total_combos"],
        "key_dims": args.grid_dims,
        "description": f"前{args.grid_dims}个PCA维度网格搜索(7值网格)",
    }

    # ── 3. 对比表 ──
    logger.info("\n" + "=" * 60)
    logger.info("方法对比")
    logger.info("=" * 60)
    logger.info(f"{'方法':<35} {'D_max(mm)':<14} {'JointLimit(mm)':<16}")
    logger.info("-" * 65)
    logger.info(f"{'A: 数据最大':<35} {res_a['D_max']*1000:<14.2f} {limit_a*1000:<16.2f}")
    if d_max_b:
        logger.info(f"{'B: Monte Carlo':<35} {res_b['D_max']*1000:<14.2f} {limit_b*1000:<16.2f}")
    logger.info(f"{'C: 网格搜索':<35} {res_c['D_max']*1000:<14.2f} {limit_c*1000:<16.2f}")

    # ── 4. 渲染 ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: 渲染对比视频 (MANO手 + 夹爪)")
    logger.info("=" * 60)
    side = md["side"]
    methods = [("method_A_data_max", res_a["D_max"], limit_a)]
    if d_max_b:
        methods.append(("method_B_monte_carlo", res_b["D_max"], limit_b))
    methods.append(("method_C_grid_search", res_c["D_max"], limit_c))

    for name, d_max, jl in methods:
        vp = str(out / f"{name}_{side}.mp4")
        logger.info(f"\n{'─'*60}")
        logger.info(f"渲染 {name}")
        render_video(md, jl, d_max, name, vp)

    # ── 5. 保存 summary ──
    sp = out / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 输出在: {out}")
    logger.info(f"Summary: {sp}")
    logger.info(f"请打开 MP4 视频, 比较夹爪开合是否覆盖了 MANO 手的运动范围")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

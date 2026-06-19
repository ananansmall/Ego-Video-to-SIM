"""
05_video_alignment.py — 视频-仿真对齐：2D重投影验证 + 视频叠加对比 + 位姿优化

核心思路:
  1. 视频叠加对比: 将仿真渲染结果半透明叠加到原始视频帧上，直观看到对齐偏差
  2. 2D重投影验证: 将3D手部关节投影到2D，与原始视频的2D关键点比较，量化对齐质量
  3. 位姿优化: 基于2D重投影误差，优化3D位姿偏移量，使仿真与视频一致

数据源:
  - 原始视频帧: hawor_dir/extracted_images/
  - 2D关键点真值: hawor_dir/cam_space/ (MANO关节在相机空间的3D位置→投影到2D)
  - 相机参数: hawor reconstruction npz 中的 R_c2w, t_c2w, img_focal
  - 仿真渲染: 04_physics_simulation.py 的输出或实时渲染

用法:
  # 模式1: 视频叠加对比 (最简单, 快速诊断)
  python 05_video_alignment.py \\
      --hawor-dir /home/an/data/hawor/7 \\
      --mode overlay \\
      --sim-video output/videos/physics_sim_physics_tracking.mp4

  # 模式2: 2D重投影误差分析
  python 05_video_alignment.py \\
      --hawor-dir /home/an/data/hawor/7 \\
      --mode reproj_analysis

  # 模式3: 位姿优化 (基于2D重投影误差优化3D偏移)
  python 05_video_alignment.py \\
      --hawor-dir /home/an/data/hawor/7 \\
      --mode optimize \\
      --ras-dir /home/an/data/ras/my_7mp4_result \\
      --transform-params ./output/alignment/transform_params.npz

  # 模式4: 完整流程 (叠加+分析+优化)
  python 05_video_alignment.py \\
      --hawor-dir /home/an/data/hawor/7 \\
      --mode full \\
      --ras-dir /home/an/data/ras/my_7mp4_result \\
      --transform-params ./output/alignment/transform_params.npz
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import minimize
from tqdm import trange, tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))

from pytransform3d import rotations as pr

try:
    import torch
    from mano_layer import MANOLayer
except ImportError:
    MANOLayer = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import sapien
    import sapien.render
except ImportError:
    sapien = None

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = [
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (0, 165, 255),
]

JOINT_TO_FINGER = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]


def _find_reconstruction_file(hawor_path):
    """在 HaWoR 目录中查找重建结果 npz 文件

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        Path: 找到的 npz 文件路径，或 None
    """
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        npz_files = list(hawor_path.glob("*.npz"))
        if npz_files:
            return npz_files[0]
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中哪只手是活跃的

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        int: 手部索引 (0=左手, 1=右手)，或 None
    """
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if 0 in detected and 1 not in detected:
            return 0
        if 1 in detected and 0 not in detected:
            return 1
        if 0 in detected:
            return 0
        if 1 in detected:
            return 1
    return None


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据 (含相机轨迹)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引 (0=左手, 1=右手)

    Returns:
        dict: 包含 pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
              R_c2w, t_c2w, img_focal
    """
    hawor_path = Path(hawor_dir)
    rec_file = _find_reconstruction_file(hawor_path)
    if rec_file is None:
        raise FileNotFoundError(f"未找到 hawor 数据: {hawor_path}")

    rec = np.load(str(rec_file), allow_pickle=True)
    data = {
        "pred_trans": rec["pred_trans"][hand_idx],
        "pred_rot": rec["pred_rot"][hand_idx],
        "pred_hand_pose": rec["pred_hand_pose"][hand_idx],
        "pred_betas": rec["pred_betas"][hand_idx],
        "pred_valid": rec["pred_valid"][hand_idx],
        "R_c2w": rec["R_c2w"] if "R_c2w" in rec else None,
        "t_c2w": rec["t_c2w"] if "t_c2w" in rec else None,
        "img_focal": float(rec["img_focal"]) if "img_focal" in rec else 600.0,
    }

    est_focal_file = hawor_path / "est_focal.txt"
    if data["img_focal"] <= 0 and est_focal_file.exists():
        try:
            data["img_focal"] = float(est_focal_file.read_text().strip())
        except Exception:
            pass

    return data


def load_cam_space_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 相机空间数据 (MANO 关节在相机坐标系下的3D位置)

    从 hawor_dir/cam_space/<hand_idx>/*.json 加载,
    包含 init_trans, init_root_orient, init_hand_pose, init_betas。

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引

    Returns:
        dict: 包含 init_trans, init_root_orient, init_hand_pose, init_betas, 或 None
    """
    cam_dir = Path(hawor_dir) / "cam_space" / str(hand_idx)
    if not cam_dir.exists():
        return None

    json_files = sorted(cam_dir.glob("*.json"))
    if not json_files:
        return None

    with open(json_files[0], "r") as f:
        data = json.load(f)

    return {
        "init_trans": np.array(data["init_trans"])[0],
        "init_root_orient": np.array(data["init_root_orient"])[0],
        "init_hand_pose": np.array(data["init_hand_pose"])[0],
        "init_betas": np.array(data["init_betas"])[0],
    }


def load_video_frames(hawor_dir):
    """加载原始视频帧 (从 hawor_dir/extracted_images/)

    Args:
        hawor_dir: HaWoR 输出目录路径

    Returns:
        tuple: (frames, img_shape) 或 (None, None)
            - frames: 视频帧列表 (BGR)
            - img_shape: (height, width)
    """
    img_dir = Path(hawor_dir) / "extracted_images"
    if not img_dir.exists():
        return None, None

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        return None, None

    frames = []
    for f in tqdm(img_files, desc="加载视频帧"):
        img = cv2.imread(str(f))
        if img is not None:
            frames.append(img)

    if not frames:
        return None, None

    return frames, frames[0].shape[:2]


def rotation_matrix_to_angle_axis(R):
    """将旋转矩阵转换为轴角表示

    Args:
        R: (3, 3) 或 (N, 3, 3) 旋转矩阵

    Returns:
        np.ndarray: (3,) 或 (N, 3) 轴角向量
    """
    if R.ndim == 2:
        R = R[np.newaxis]
    angle_axis = np.zeros((R.shape[0], 3), dtype=np.float64)
    for i in range(R.shape[0]):
        trace = R[i, 0, 0] + R[i, 1, 1] + R[i, 2, 2]
        if trace > 3 - 1e-6:
            angle_axis[i] = np.zeros(3)
        elif trace < -1 + 1e-6:
            vals = np.array([R[i, 0, 0], R[i, 1, 1], R[i, 2, 2]])
            k = np.argmax(vals)
            idx = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][k]
            v = R[i, idx[1], idx[2]] - R[i, idx[2], idx[1]]
            angle_axis[i] = v / np.linalg.norm(v) * np.pi
        else:
            theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            vx = np.array([
                R[i, 2, 1] - R[i, 1, 2],
                R[i, 0, 2] - R[i, 2, 0],
                R[i, 1, 0] - R[i, 0, 1],
            ])
            angle_axis[i] = vx / (2 * np.sin(theta)) * theta
    return angle_axis


def compute_mano_joints_cam_space(cam_data, frame_idx, mano_layer_cam=None):
    """在相机坐标系下计算 MANO 关节位置

    使用 cam_space 数据中的旋转矩阵和平移向量,
    通过 MANO FK 计算关节在相机坐标系下的3D位置。

    Args:
        cam_data: load_cam_space_data() 返回的字典
        frame_idx: 帧索引
        mano_layer_cam: MANOLayer 实例

    Returns:
        tuple: (vertices, joints) 或 (None, None)
    """
    init_trans = cam_data["init_trans"][frame_idx]
    init_root_orient = cam_data["init_root_orient"][frame_idx]
    init_hand_pose = cam_data["init_hand_pose"][frame_idx]
    init_betas = cam_data["init_betas"][frame_idx]

    if mano_layer_cam is not None:
        root_aa = rotation_matrix_to_angle_axis(init_root_orient).flatten()  # (3,)
        pose_aa = rotation_matrix_to_angle_axis(init_hand_pose.reshape(-1, 3, 3)).flatten()  # (45,)

        p = torch.from_numpy(np.concatenate([root_aa, pose_aa]).astype(np.float32)).unsqueeze(0)
        t = torch.from_numpy(init_trans.astype(np.float32)).unsqueeze(0)
        v, j = mano_layer_cam(p, t)
        return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]

    return None, None


def project_3d_to_2d(points_3d_cam, focal, cx, cy):
    """将相机坐标系下的3D点投影到2D图像平面

    使用针孔相机模型: x = f*X/Z + cx, y = f*Y/Z + cy
    只投影 Z > 0.01 的点 (在相机前方)。

    Args:
        points_3d_cam: (N, 3) 相机坐标系下的3D点
        focal: 焦距 (像素)
        cx: 主点 X (像素)
        cy: 主点 Y (像素)

    Returns:
        tuple: (pts_2d, valid)
            - pts_2d: (N, 2) 2D投影坐标, 无效点为 -1
            - valid: (N,) 有效标记
    """
    valid = points_3d_cam[:, 2] > 0.01
    pts_2d = np.full((len(points_3d_cam), 2), -1.0)
    pts_2d[valid, 0] = focal * points_3d_cam[valid, 0] / points_3d_cam[valid, 2] + cx
    pts_2d[valid, 1] = focal * points_3d_cam[valid, 1] / points_3d_cam[valid, 2] + cy
    return pts_2d, valid


def world_to_cam_hawor(points_world, R_c2w, t_c2w):
    """将世界坐标系下的点转换到 HaWoR 相机坐标系

    变换链: world → SLAM world (R_x) → camera (R_w2c, t_w2c)

    Args:
        points_world: (N, 3) 世界坐标系下的点
        R_c2w: (3, 3) camera-to-world 旋转矩阵
        t_c2w: (3,) camera-to-world 平移向量

    Returns:
        np.ndarray: (N, 3) 相机坐标系下的点
    """
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    points_world_slam = (R_x @ points_world.T).T
    points_cam = (R_w2c @ points_world_slam.T).T + t_w2c
    return points_cam


def compute_reprojection_error(joints_2d_gt, joints_2d_sim, valid_mask=None):
    """计算2D重投影误差

    比较真值2D关节和仿真2D关节之间的欧氏距离。

    Args:
        joints_2d_gt: (21, 2) 真值2D关节坐标
        joints_2d_sim: (21, 2) 仿真2D关节坐标
        valid_mask: (21,) 有效标记, None则自动判断

    Returns:
        tuple: (mean_error, info_dict)
            - mean_error: 平均误差 (像素)
            - info_dict: 包含 mean, median, max, per_joint, wrist_err, fingertip_err
    """
    if valid_mask is None:
        valid_mask = (joints_2d_gt[:, 0] > 0) & (joints_2d_sim[:, 0] > 0)
    if valid_mask.sum() == 0:
        return float("inf"), {}
    diff = joints_2d_gt[valid_mask] - joints_2d_sim[valid_mask]
    per_joint_err = np.linalg.norm(diff, axis=1)
    mean_err = per_joint_err.mean()
    return mean_err, {
        "mean": mean_err,
        "median": np.median(per_joint_err),
        "max": per_joint_err.max(),
        "per_joint": per_joint_err,
        "wrist_err": per_joint_err[0] if len(per_joint_err) > 0 else 0,
        "fingertip_err": per_joint_err[[4, 8, 12, 16, 20]].mean() if len(per_joint_err) > 20 else 0,
    }


def draw_2d_skeleton(img, joints_2d, valid, color=(0, 255, 0), radius=4, thickness=2):
    """在图像上绘制2D手部骨架

    绘制21个关节点和20条骨架线, 每根手指用不同颜色。

    Args:
        img: BGR 图像 (原地修改)
        joints_2d: (21, 2) 2D关节坐标
        valid: (21,) 有效标记
        color: 绘制颜色, "auto" 则按手指分组着色
        radius: 关键点半径
        thickness: 骨架线宽度
    """
    for i, (x, y) in enumerate(joints_2d):
        if not valid[i] or x < 0 or y < 0:
            continue
        finger = JOINT_TO_FINGER[i] if i < len(JOINT_TO_FINGER) else 0
        c = FINGER_COLORS[finger] if isinstance(color, str) and color == "auto" else color
        cv2.circle(img, (int(x), int(y)), radius, c, -1)
    for j1, j2 in MANO_SKELETON:
        if not valid[j1] or not valid[j2]:
            continue
        x1, y1 = joints_2d[j1]
        x2, y2 = joints_2d[j2]
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        finger = JOINT_TO_FINGER[j1] if j1 < len(JOINT_TO_FINGER) else 0
        c = FINGER_COLORS[finger] if isinstance(color, str) and color == "auto" else color
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, thickness)


def overlay_sim_on_video(video_frame, sim_frame, alpha=0.5):
    """将仿真帧半透明叠加到原始视频帧上

    Args:
        video_frame: 原始视频帧 (BGR)
        sim_frame: 仿真渲染帧 (BGR)
        alpha: 仿真帧的透明度 (0=仅视频, 1=仅仿真)

    Returns:
        np.ndarray: 叠加后的帧
    """
    h, w = video_frame.shape[:2]
    sim_resized = cv2.resize(sim_frame, (w, h))
    blended = cv2.addWeighted(video_frame, 1 - alpha, sim_resized, alpha, 0)
    return blended


class PoseOptimizer:
    """基于2D重投影误差的3D位姿偏移优化器

    通过最小化真值2D关节和仿真2D关节之间的距离,
    优化3D平移偏移量, 使仿真结果与视频对齐。
    """

    def __init__(self, hawor_data, cam_data, img_shape, mano_layer, logger=None):
        """初始化位姿优化器

        Args:
            hawor_data: load_hawor_data() 返回的字典
            cam_data: load_cam_space_data() 返回的字典
            img_shape: (height, width) 图像尺寸
            mano_layer: MANOLayer 实例
            logger: 日志记录器
        """
        self.hawor_data = hawor_data
        self.cam_data = cam_data
        self.img_h, self.img_w = img_shape
        self.focal = hawor_data["img_focal"]
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0
        self.mano_layer = mano_layer
        self.logger = logger or logging.getLogger("PoseOptimizer")

    def _get_gt_2d_joints(self, frame_idx):
        """获取真值2D关节 (从 cam_space 数据投影)

        Args:
            frame_idx: 帧索引

        Returns:
            tuple: (joints_2d, valid) 或 (None, None)
        """
        if self.cam_data is None:
            return None, None

        _, joints_cam = compute_mano_joints_cam_space(self.cam_data, frame_idx, self.mano_layer)
        if joints_cam is None:
            init_trans = self.cam_data["init_trans"][frame_idx]
            return None, None

        joints_2d, valid = project_3d_to_2d(joints_cam, self.focal, self.cx, self.cy)
        return joints_2d, valid

    def _get_sim_2d_joints(self, frame_idx, offset_trans=np.zeros(3), offset_rot=np.zeros(3)):
        """获取仿真2D关节 (从 HaWoR 世界坐标投影到相机)

        流程: MANO FK → 世界坐标 → 相机坐标 → 2D投影

        Args:
            frame_idx: 帧索引
            offset_trans: (3,) 平移偏移量 (优化参数)
            offset_rot: (3,) 旋转偏移量 (暂未使用)

        Returns:
            tuple: (joints_2d, valid)
        """
        pred_trans = self.hawor_data["pred_trans"][frame_idx].copy()
        pred_rot = self.hawor_data["pred_rot"][frame_idx].copy()

        pred_trans += offset_trans

        R_c2w = self.hawor_data["R_c2w"][frame_idx]
        t_c2w = self.hawor_data["t_c2w"][frame_idx]

        joints_cam = world_to_cam_hawor(pred_trans.reshape(1, 3), R_c2w, t_c2w)[0]

        p = torch.from_numpy(
            np.concatenate([pred_rot, self.hawor_data["pred_hand_pose"][frame_idx]]).astype(np.float32)
        ).unsqueeze(0)
        t = torch.from_numpy(pred_trans.astype(np.float32)).unsqueeze(0)
        _, joints_world = self.mano_layer(p, t)
        joints_world_np = joints_world.detach().cpu().numpy()[0]

        joints_cam_all = world_to_cam_hawor(joints_world_np, R_c2w, t_c2w)
        joints_2d, valid = project_3d_to_2d(joints_cam_all, self.focal, self.cx, self.cy)
        return joints_2d, valid

    def compute_frame_error(self, frame_idx, offset_trans=np.zeros(3)):
        """计算单帧的2D重投影误差

        Args:
            frame_idx: 帧索引
            offset_trans: (3,) 平移偏移量

        Returns:
            tuple: (mean_error, info_dict)
        """
        gt_2d, gt_valid = self._get_gt_2d_joints(frame_idx)
        if gt_2d is None:
            return float("inf"), {}

        sim_2d, sim_valid = self._get_sim_2d_joints(frame_idx, offset_trans)
        valid = gt_valid & sim_valid

        return compute_reprojection_error(gt_2d, sim_2d, valid)

    def optimize_offset(self, frame_indices=None, method="L-BFGS-B"):
        """优化3D平移偏移量, 最小化2D重投影误差

        使用 scipy.optimize.minimize (L-BFGS-B) 优化3个平移参数,
        目标函数为采样帧的平均2D重投影误差。

        Args:
            frame_indices: 采样帧索引列表, None则自动采样10帧
            method: 优化方法 (默认 L-BFGS-B)

        Returns:
            tuple: (offset_trans, final_error, scipy_result)
        """
        if frame_indices is None:
            n_total = len(self.hawor_data["pred_valid"])
            valid_frames = [i for i in range(n_total) if self.hawor_data["pred_valid"][i]]
            step = max(1, len(valid_frames) // 10)
            frame_indices = valid_frames[::step][:10]

        self.logger.info(f"  优化帧: {frame_indices}")

        def loss_fn(params):
            offset = np.array(params[:3])
            total_err = 0
            n_valid = 0
            for fi in frame_indices:
                err, _ = self.compute_frame_error(fi, offset)
                if np.isfinite(err):
                    total_err += err
                    n_valid += 1
            return total_err / max(n_valid, 1)

        initial = np.zeros(3)
        self.logger.info(f"  初始误差: {loss_fn(initial):.2f} px")

        result = minimize(
            loss_fn,
            initial,
            method=method,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        offset = result.x[:3]
        final_err = loss_fn(offset)
        self.logger.info(f"  优化后误差: {final_err:.2f} px")
        self.logger.info(f"  平移偏移: {offset}")
        self.logger.info(f"  优化状态: {result.message}")

        return offset, final_err, result


def run_overlay(hawor_dir, sim_video_path, output_path, hand_idx=0, alpha=0.5, logger=None):
    """模式1: 视频叠加对比 — 将仿真渲染半透明叠加到原始视频

    在原始视频帧上绘制:
    - 绿色骨架: 真值2D关节 (cam_space → 投影)
    - 红色骨架: 仿真2D关节 (world → cam → 投影)
    - 半透明叠加: 仿真视频帧

    Args:
        hawor_dir: HaWoR 输出目录
        sim_video_path: 仿真视频路径
        output_path: 输出视频路径
        hand_idx: 手部索引
        alpha: 叠加透明度
        logger: 日志记录器
    """
    logger = logger or logging.getLogger("overlay")
    logger.info("=" * 60)
    logger.info("模式: 视频叠加对比")
    logger.info("=" * 60)

    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    cam_data = load_cam_space_data(hawor_dir, hand_idx=hand_idx)
    video_frames, img_shape = load_video_frames(hawor_dir)

    if video_frames is None:
        logger.error("  ✗ 未找到原始视频帧")
        return

    logger.info(f"  原始视频: {len(video_frames)} 帧, {img_shape[1]}x{img_shape[0]}")

    cap = None
    if sim_video_path and Path(sim_video_path).exists():
        cap = cv2.VideoCapture(str(sim_video_path))
        logger.info(f"  仿真视频: {sim_video_path}")
    else:
        logger.warning(f"  ⚠ 仿真视频不存在: {sim_video_path}, 仅显示2D关键点")

    focal = hawor_data["img_focal"]
    cx, cy = img_shape[1] / 2.0, img_shape[0] / 2.0

    n_frames = len(video_frames)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_shape[1], img_shape[0]))

    mano_layer = None
    if MANOLayer is not None:
        try:
            betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
            mano_side = "left" if hand_idx == 0 else "right"
            mano_layer = MANOLayer(mano_side, betas_mean)
        except Exception as e:
            logger.warning(f"  ⚠ MANO 初始化失败: {e}")

    for frame_idx in trange(n_frames, desc="叠加渲染"):
        video_frame = video_frames[frame_idx].copy()

        if cam_data is not None and frame_idx < len(cam_data["init_trans"]) and mano_layer is not None:
            _, joints_cam = compute_mano_joints_cam_space(cam_data, frame_idx, mano_layer)
            if joints_cam is not None:
                joints_2d, valid = project_3d_to_2d(joints_cam, focal, cx, cy)
                draw_2d_skeleton(video_frame, joints_2d, valid, color=(0, 255, 0), radius=3, thickness=1)

        if hawor_data["R_c2w"] is not None and frame_idx < len(hawor_data["R_c2w"]):
            pred_trans = hawor_data["pred_trans"][frame_idx]
            R_c2w = hawor_data["R_c2w"][frame_idx]
            t_c2w = hawor_data["t_c2w"][frame_idx]

            if mano_layer is not None:
                p = torch.from_numpy(
                    np.concatenate([
                        hawor_data["pred_rot"][frame_idx],
                        hawor_data["pred_hand_pose"][frame_idx]
                    ]).astype(np.float32)
                ).unsqueeze(0)
                t = torch.from_numpy(pred_trans.astype(np.float32)).unsqueeze(0)
                _, joints_world = mano_layer(p, t)
                joints_world_np = joints_world.detach().cpu().numpy()[0]

                joints_cam_sim = world_to_cam_hawor(joints_world_np, R_c2w, t_c2w)
                joints_2d_sim, valid_sim = project_3d_to_2d(joints_cam_sim, focal, cx, cy)
                draw_2d_skeleton(video_frame, joints_2d_sim, valid_sim, color=(0, 0, 255), radius=3, thickness=1)

        if cap is not None and cap.isOpened():
            ret, sim_frame = cap.read()
            if ret:
                video_frame = overlay_sim_on_video(video_frame, sim_frame, alpha=alpha)

        h, w = video_frame.shape[:2]
        cv2.rectangle(video_frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(video_frame, f"Frame {frame_idx}  |  Green=GT  Red=Sim",
                     (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(video_frame)

    writer.release()
    if cap is not None:
        cap.release()

    logger.info(f"  ✓ 叠加视频已保存: {output_path}")


def run_reproj_analysis(hawor_dir, output_path, hand_idx=0, logger=None):
    """模式2: 2D重投影误差分析 — 量化仿真与视频的对齐质量

    逐帧计算:
    1. 真值2D关节 (cam_space MANO FK → 投影)
    2. 仿真2D关节 (world MANO FK → world_to_cam → 投影)
    3. 两者之间的欧氏距离

    输出: 误差统计 + 可视化视频 (绿=GT, 红=Sim)

    Args:
        hawor_dir: HaWoR 输出目录
        output_path: 输出视频路径
        hand_idx: 手部索引
        logger: 日志记录器

    Returns:
        list: 每帧的误差信息字典, 或 None
    """
    logger = logger or logging.getLogger("reproj_analysis")
    logger.info("=" * 60)
    logger.info("模式: 2D重投影误差分析")
    logger.info("=" * 60)

    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    cam_data = load_cam_space_data(hawor_dir, hand_idx=hand_idx)
    video_frames, img_shape = load_video_frames(hawor_dir)

    if cam_data is None:
        logger.error("  ✗ 未找到 cam_space 数据")
        return None

    focal = hawor_data["img_focal"]
    cx, cy = img_shape[1] / 2.0, img_shape[0] / 2.0

    mano_layer = None
    if MANOLayer is not None:
        try:
            betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
            mano_side = "left" if hand_idx == 0 else "right"
            mano_layer = MANOLayer(mano_side, betas_mean)
        except Exception as e:
            logger.error(f"  ✗ MANO 初始化失败: {e}")
            return None

    n_frames = len(hawor_data["pred_valid"])
    errors = []

    for frame_idx in trange(n_frames, desc="重投影分析"):
        if not hawor_data["pred_valid"][frame_idx]:
            errors.append(None)
            continue

        _, joints_cam_gt = compute_mano_joints_cam_space(cam_data, frame_idx, mano_layer)
        if joints_cam_gt is None:
            errors.append(None)
            continue

        gt_2d, gt_valid = project_3d_to_2d(joints_cam_gt, focal, cx, cy)

        if hawor_data["R_c2w"] is not None and frame_idx < len(hawor_data["R_c2w"]):
            pred_trans = hawor_data["pred_trans"][frame_idx]
            R_c2w = hawor_data["R_c2w"][frame_idx]
            t_c2w = hawor_data["t_c2w"][frame_idx]

            p = torch.from_numpy(
                np.concatenate([
                    hawor_data["pred_rot"][frame_idx],
                    hawor_data["pred_hand_pose"][frame_idx]
                ]).astype(np.float32)
            ).unsqueeze(0)
            t = torch.from_numpy(pred_trans.astype(np.float32)).unsqueeze(0)
            _, joints_world = mano_layer(p, t)
            joints_world_np = joints_world.detach().cpu().numpy()[0]

            joints_cam_sim = world_to_cam_hawor(joints_world_np, R_c2w, t_c2w)
            sim_2d, sim_valid = project_3d_to_2d(joints_cam_sim, focal, cx, cy)

            valid = gt_valid & sim_valid
            err, info = compute_reprojection_error(gt_2d, sim_2d, valid)
            errors.append(info)
        else:
            errors.append(None)

    valid_errors = [e for e in errors if e is not None and np.isfinite(e.get("mean", float("inf")))]
    if not valid_errors:
        logger.error("  ✗ 无有效重投影误差")
        return None

    mean_errs = [e["mean"] for e in valid_errors]
    logger.info(f"\n  重投影误差统计:")
    logger.info(f"    有效帧: {len(valid_errors)}/{n_frames}")
    logger.info(f"    平均误差: {np.mean(mean_errs):.2f} px")
    logger.info(f"    中位误差: {np.median(mean_errs):.2f} px")
    logger.info(f"    最大误差: {np.max(mean_errs):.2f} px")
    logger.info(f"    最小误差: {np.min(mean_errs):.2f} px")

    wrist_errs = [e["wrist_err"] for e in valid_errors]
    fingertip_errs = [e["fingertip_err"] for e in valid_errors]
    logger.info(f"    手腕平均: {np.mean(wrist_errs):.2f} px")
    logger.info(f"    指尖平均: {np.mean(fingertip_errs):.2f} px")

    if video_frames is not None:
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (img_shape[1], img_shape[0]))

        for frame_idx in range(min(n_frames, len(video_frames))):
            frame = video_frames[frame_idx].copy()

            if cam_data is not None and frame_idx < len(cam_data["init_trans"]) and mano_layer is not None:
                _, joints_cam_gt = compute_mano_joints_cam_space(cam_data, frame_idx, mano_layer)
                if joints_cam_gt is not None:
                    gt_2d, gt_valid = project_3d_to_2d(joints_cam_gt, focal, cx, cy)
                    draw_2d_skeleton(frame, gt_2d, gt_valid, color=(0, 255, 0), radius=3, thickness=1)

            if hawor_data["R_c2w"] is not None and frame_idx < len(hawor_data["R_c2w"]) and mano_layer is not None:
                pred_trans = hawor_data["pred_trans"][frame_idx]
                R_c2w = hawor_data["R_c2w"][frame_idx]
                t_c2w = hawor_data["t_c2w"][frame_idx]

                p = torch.from_numpy(
                    np.concatenate([
                        hawor_data["pred_rot"][frame_idx],
                        hawor_data["pred_hand_pose"][frame_idx]
                    ]).astype(np.float32)
                ).unsqueeze(0)
                t = torch.from_numpy(pred_trans.astype(np.float32)).unsqueeze(0)
                _, joints_world = mano_layer(p, t)
                joints_world_np = joints_world.detach().cpu().numpy()[0]

                joints_cam_sim = world_to_cam_hawor(joints_world_np, R_c2w, t_c2w)
                sim_2d, sim_valid = project_3d_to_2d(joints_cam_sim, focal, cx, cy)
                draw_2d_skeleton(frame, sim_2d, sim_valid, color=(0, 0, 255), radius=3, thickness=1)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            err_info = errors[frame_idx]
            if err_info is not None:
                err_val = err_info["mean"]
                err_color = (0, 255, 0) if err_val < 10 else (0, 255, 255) if err_val < 30 else (0, 0, 255)
                cv2.putText(frame, f"Frame {frame_idx}  Err:{err_val:.1f}px  Wrist:{err_info['wrist_err']:.1f}px  Tip:{err_info['fingertip_err']:.1f}px",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 1)
                cv2.putText(frame, f"Green=GT(cam_space)  Red=Sim(world→cam)",
                            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"Frame {frame_idx}  No data",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            writer.write(frame)

        writer.release()
        logger.info(f"  ✓ 分析视频已保存: {output_path}")

    return errors


def run_optimize(hawor_dir, output_dir, hand_idx=0, logger=None):
    """模式3: 位姿优化 — 基于2D重投影误差优化3D平移偏移

    3个步骤:
    1. 优化前误差分析
    2. L-BFGS-B 优化平移偏移量
    3. 验证优化结果 + 可视化

    输出:
    - pose_offset.npz: 优化后的偏移量
    - optimization_vis.mp4: 优化前后对比视频 (绿=GT, 品红=Optimized)

    Args:
        hawor_dir: HaWoR 输出目录
        output_dir: 输出目录
        hand_idx: 手部索引
        logger: 日志记录器

    Returns:
        np.ndarray: (3,) 优化后的平移偏移量, 或 None
    """
    logger = logger or logging.getLogger("optimize")
    logger.info("=" * 60)
    logger.info("模式: 位姿优化")
    logger.info("=" * 60)

    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    cam_data = load_cam_space_data(hawor_dir, hand_idx=hand_idx)
    video_frames, img_shape = load_video_frames(hawor_dir)

    if cam_data is None:
        logger.error("  ✗ 未找到 cam_space 数据, 无法优化")
        return None

    if hawor_data["R_c2w"] is None:
        logger.error("  ✗ 未找到 R_c2w 数据, 无法优化")
        return None

    focal = hawor_data["img_focal"]
    cx, cy = img_shape[1] / 2.0, img_shape[0] / 2.0

    mano_layer = None
    if MANOLayer is not None:
        try:
            betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
            mano_side = "left" if hand_idx == 0 else "right"
            mano_layer = MANOLayer(mano_side, betas_mean)
        except Exception as e:
            logger.error(f"  ✗ MANO 初始化失败: {e}")
            return None

    optimizer = PoseOptimizer(hawor_data, cam_data, img_shape, mano_layer, logger=logger)

    logger.info("\n[1/3] 优化前误差分析 ...")
    n_total = len(hawor_data["pred_valid"])
    valid_frames = [i for i in range(n_total) if hawor_data["pred_valid"][i]]
    sample_frames = valid_frames[::max(1, len(valid_frames) // 10)][:10]

    before_errors = []
    for fi in sample_frames:
        err, info = optimizer.compute_frame_error(fi)
        if np.isfinite(err):
            before_errors.append(err)
    if before_errors:
        logger.info(f"  优化前平均重投影误差: {np.mean(before_errors):.2f} px")

    logger.info("\n[2/3] 优化平移偏移 ...")
    offset_trans, final_err, result = optimizer.optimize_offset(sample_frames)

    logger.info("\n[3/3] 验证优化结果 ...")
    after_errors = []
    for fi in sample_frames:
        err, info = optimizer.compute_frame_error(fi, offset_trans)
        if np.isfinite(err):
            after_errors.append(err)
    if after_errors:
        logger.info(f"  优化后平均重投影误差: {np.mean(after_errors):.2f} px")
        if before_errors:
            improvement = (np.mean(before_errors) - np.mean(after_errors)) / np.mean(before_errors) * 100
            logger.info(f"  改善: {improvement:.1f}%")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    offset_path = output_dir / "pose_offset.npz"
    np.savez(str(offset_path),
             offset_trans=offset_trans,
             before_mean_err=np.mean(before_errors) if before_errors else float("inf"),
             after_mean_err=np.mean(after_errors) if after_errors else float("inf"))
    logger.info(f"  ✓ 偏移量已保存: {offset_path}")

    if video_frames is not None:
        vis_path = output_dir / "optimization_vis.mp4"
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vis_path), fourcc, fps, (img_shape[1], img_shape[0]))

        for frame_idx in trange(min(n_total, len(video_frames)), desc="优化可视化"):
            if not hawor_data["pred_valid"][frame_idx]:
                writer.write(video_frames[frame_idx])
                continue

            frame = video_frames[frame_idx].copy()

            if cam_data is not None and frame_idx < len(cam_data["init_trans"]) and mano_layer is not None:
                _, joints_cam_gt = compute_mano_joints_cam_space(cam_data, frame_idx, mano_layer)
                if joints_cam_gt is not None:
                    gt_2d, gt_valid = project_3d_to_2d(joints_cam_gt, focal, cx, cy)
                    draw_2d_skeleton(frame, gt_2d, gt_valid, color=(0, 255, 0), radius=3, thickness=1)

            if mano_layer is not None and frame_idx < len(hawor_data["R_c2w"]):
                pred_trans_opt = hawor_data["pred_trans"][frame_idx] + offset_trans
                R_c2w = hawor_data["R_c2w"][frame_idx]
                t_c2w = hawor_data["t_c2w"][frame_idx]

                p = torch.from_numpy(
                    np.concatenate([
                        hawor_data["pred_rot"][frame_idx],
                        hawor_data["pred_hand_pose"][frame_idx]
                    ]).astype(np.float32)
                ).unsqueeze(0)
                t = torch.from_numpy(pred_trans_opt.astype(np.float32)).unsqueeze(0)
                _, joints_world = mano_layer(p, t)
                joints_world_np = joints_world.detach().cpu().numpy()[0]

                joints_cam_opt = world_to_cam_hawor(joints_world_np, R_c2w, t_c2w)
                opt_2d, opt_valid = project_3d_to_2d(joints_cam_opt, focal, cx, cy)
                draw_2d_skeleton(frame, opt_2d, opt_valid, color=(255, 0, 255), radius=3, thickness=1)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame {frame_idx}  |  Green=GT  Magenta=Optimized  offset={np.linalg.norm(offset_trans)*100:.1f}cm",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Offset: [{offset_trans[0]:.4f}, {offset_trans[1]:.4f}, {offset_trans[2]:.4f}]",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            writer.write(frame)

        writer.release()
        logger.info(f"  ✓ 优化可视化已保存: {vis_path}")

    return offset_trans


def main():
    """命令行入口: 视频-仿真对齐

    支持四种模式:
      overlay:         视频叠加对比 (快速诊断)
      reproj_analysis: 2D重投影误差分析 (量化对齐)
      optimize:        位姿优化 (优化3D偏移)
      full:            完整流程 (叠加+分析+优化)

    用法示例:
      python 05_video_alignment.py --hawor-dir /path/to/hawor --mode overlay --sim-video output/videos/xxx.mp4
      python 05_video_alignment.py --hawor-dir /path/to/hawor --mode reproj_analysis
      python 05_video_alignment.py --hawor-dir /path/to/hawor --mode optimize
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="视频-仿真对齐: 2D重投影验证 + 视频叠加对比 + 位姿优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="overlay",
                        choices=["overlay", "reproj_analysis", "optimize", "full"],
                        help="运行模式")
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建输出目录")
    parser.add_argument("--ras-dir", type=str, default=None,
                        help="RAS 场景重建输出目录 (optimize/full 模式需要)")
    parser.add_argument("--transform-params", type=str, default=None,
                        help="transform_params.npz 路径 (optimize/full 模式需要)")
    parser.add_argument("--sim-video", type=str, default=None,
                        help="仿真视频路径 (overlay 模式)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手的索引: 0=左手, 1=右手, -1=自动检测")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="叠加透明度 (0=仅视频, 1=仅仿真)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径")
    parser.add_argument("--output-dir", type=str, default="output/alignment_analysis",
                        help="输出目录")
    args = parser.parse_args()

    if args.hand_idx < 0:
        detected = _detect_hand_idx(Path(args.hawor_dir))
        args.hand_idx = detected if detected is not None else 0
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        print(f"自动检测到手: {hand_label} (idx={args.hand_idx})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("VideoAlignment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

    if args.mode == "overlay":
        output = args.output or str(output_dir / "overlay_comparison.mp4")
        run_overlay(args.hawor_dir, args.sim_video, output,
                    hand_idx=args.hand_idx, alpha=args.alpha, logger=logger)

    elif args.mode == "reproj_analysis":
        output = args.output or str(output_dir / "reproj_analysis.mp4")
        run_reproj_analysis(args.hawor_dir, output,
                           hand_idx=args.hand_idx, logger=logger)

    elif args.mode == "optimize":
        run_optimize(args.hawor_dir, str(output_dir),
                    hand_idx=args.hand_idx, logger=logger)

    elif args.mode == "full":
        logger.info("=" * 60)
        logger.info("完整流程: 叠加 → 分析 → 优化")
        logger.info("=" * 60)

        overlay_path = str(output_dir / "overlay_comparison.mp4")
        run_overlay(args.hawor_dir, args.sim_video, overlay_path,
                    hand_idx=args.hand_idx, alpha=args.alpha, logger=logger)

        analysis_path = str(output_dir / "reproj_analysis.mp4")
        errors = run_reproj_analysis(args.hawor_dir, analysis_path,
                                     hand_idx=args.hand_idx, logger=logger)

        offset = run_optimize(args.hawor_dir, str(output_dir),
                             hand_idx=args.hand_idx, logger=logger)

        if offset is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"优化结果汇总:")
            logger.info(f"  平移偏移: {offset}")
            logger.info(f"  偏移量: {np.linalg.norm(offset)*100:.2f} cm")
            logger.info(f"  使用方法: 在04脚本中加载 output/alignment_analysis/pose_offset.npz")
            logger.info(f"  并将 offset_trans 加到 pred_trans 上")
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

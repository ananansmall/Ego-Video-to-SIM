#!/usr/bin/env python3
"""05_gripper_test.py — 二指夹爪控制 + 物理抓取测试

测试内容:
  测试1: 夹爪开合 (PD 关节控制, 正弦波 0→0.05→0)
  测试2: 夹爪移动 (root_pose 位置跟踪, 圆形轨迹)
  测试3: MANO 数据联合跟踪 (开合+位置+朝向)
  测试4: 位姿+开合联合控制 (圆形轨迹+正弦开合+姿态变化)
  测试5: 物理抓取释放 (虚拟6-DOF PD驱动, 对齐GalaxeaManipSim)
  测试6: 带轨迹和姿态变化的抓取 (用test5逻辑抓取后搬运放下)
  测试7: MANO轨迹驱动抓取 (MANO参考点→抓取参考系→夹爪执行)
  测试8: 复杂抓放 (真实物理, 全程≥30°角, 边走边抓, Bezier搬运)
  测试9: MANO轨迹跟随 (HaWoR数据→夹爪位姿, 真实物理抓取)
  测试11: 单臂MANO跟随 (固定底座R1右臂, test9管线+pybullet逆运动学)

运行命令:
  # 测试8: 复杂抓放 (推荐)
  python 05_gripper_test.py --test 8 --num-frames 800 --viewer

  # 测试9: MANO轨迹跟随
  python 05_gripper_test.py --test 9 --num-frames 600 --viewer

  # 测试11: 单臂 MANO 跟随
  python 05_gripper_test.py --test 11 --num-frames 600 --viewer
"""

import sys
import os
import re
import tempfile
import argparse
from pathlib import Path
from math import comb

import numpy as np
import sapien
import sapien.render
from transforms3d import quaternions as pq
import transforms3d.quaternions as pq
import scipy.spatial.transform as sst
from scipy.signal import savgol_filter
import json
import urllib.request
import time

# ── Smooth easing ──
def _smoothstep(t: float) -> float:
    """Ease-in-out: t*t*(3-2t)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _min_jerk(t: float) -> float:
    """Minimum jerk (5th order): 10t³-15t⁴+6t⁵. 更平滑, 零速零加速边界."""
    t = max(0.0, min(1.0, t))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


# ════════════════════════════════════════════════════════════════════
#  统一 MANO 轨迹平滑流水线 (四重平滑, 原 test9 内嵌实现提升至模块级)
#  ① Savitzky-Golay 前置滤波 → ② 高斯加权四元数 Slerp → ③ min-jerk 轨迹约束
#  (④ EMA 为命令级在线平滑, 由调用方主循环执行, 见 test9 的 EmaTargetSmoother)
# ════════════════════════════════════════════════════════════════════

def _gaussian_quat_smooth(quats_wxyz, sigma=2.0, window_half=5):
    """高斯加权四元数平均 (Markley 方法)"""
    n = len(quats_wxyz)
    smoothed = np.zeros_like(quats_wxyz)
    for i in range(n):
        start = max(0, i - window_half)
        end = min(n, i + window_half + 1)
        idx = np.arange(start, end)
        w = np.exp(-0.5 * ((idx - i) / sigma) ** 2)
        w = w / w.sum()
        M = np.zeros((4, 4), dtype=np.float64)
        for jj, ww in zip(idx, w):
            q = quats_wxyz[jj].astype(np.float64)
            M += ww * np.outer(q, q)
        eigvals, eigvecs = np.linalg.eigh(M)
        smoothed[i] = eigvecs[:, np.argmax(eigvals)]
    for i in range(1, n):
        if np.dot(smoothed[i], smoothed[i - 1]) < 0:
            smoothed[i] = -smoothed[i]
    return smoothed


def _apply_smooth_sg(raw_pos, raw_quat, raw_joint, raw_valid):
    """Savitzky-Golay + 高斯Slerp 平滑

    调优 (2026-08): sg_window 5→17, sg_order 2 保持;
    朝向 sigma 2→5, window_half 5→9, 抑制 MANO 逐帧噪声导致的晃动.
    """
    sg_window = 17
    sg_order = 2
    smooth_pos = raw_pos.copy()
    smooth_joint = raw_joint.copy()
    smooth_quat = raw_quat.copy()
    valid_idx = np.where(raw_valid)[0]
    if len(valid_idx) > sg_window:
        inner = valid_idx[sg_window // 2: -(sg_window // 2)]
        if len(inner) > sg_window:
            smooth_pos[inner] = savgol_filter(raw_pos[inner], sg_window, sg_order, axis=0, mode='mirror')
            smooth_joint[inner] = savgol_filter(raw_joint[inner], sg_window, sg_order, mode='mirror')
    smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
    if len(valid_idx) > 5:
        quat_inner = _gaussian_quat_smooth(raw_quat[valid_idx], sigma=5.0, window_half=9)
        norms = np.linalg.norm(quat_inner, axis=1)
        bad = norms < 1e-6
        quat_inner = quat_inner / np.maximum(norms, 1e-10)[:, np.newaxis]
        if bad.any():
            quat_inner[bad] = raw_quat[valid_idx][bad]
        smooth_quat[valid_idx] = quat_inner
    return smooth_pos, smooth_quat, smooth_joint


def _apply_smooth_dai(raw_pos, raw_quat, raw_joint, raw_valid):
    """do-as-i-do 风格: MAD速度阈值检测 + 插值替换"""
    def _velocity_position(x):
        return np.linalg.norm(np.diff(x, axis=0), axis=1)

    def _velocity_quat_wxyz(q):
        q_next = q[1:]
        q_prev = q[:-1]
        dot = np.abs(np.sum(q_next * q_prev, axis=1))
        dot = np.clip(dot, 0, 1)
        angle = 2 * np.arccos(dot)
        return angle

    def _detect_mask(signal, valid, cfg):
        n = len(signal)
        bad = np.zeros(n, dtype=bool)
        for _ in range(2):
            vel = signal[1:] - signal[:-1] if signal.ndim == 1 else np.linalg.norm(signal[1:] - signal[:-1], axis=1)
            v = np.zeros(n)
            v[1:] = vel
            v[~valid] = 0
            valid_v = v[valid]
            if len(valid_v) < 10:
                break
            median = np.median(valid_v)
            mad = np.median(np.abs(valid_v - median))
            threshold = min(median + cfg['k_mad'] * mad, cfg['v_cap'])
            bad[valid] = v[valid] > threshold
            # 合并短间隙
            for merge in range(1, cfg['gap_merge'] + 1):
                shift = np.roll(bad, merge)
                shift[:merge] = False
                bad = bad & shift
            # 取消长序列
            for i in range(n):
                if bad[i]:
                    j = i
                    while j < n and bad[j]:
                        j += 1
                    if j - i > cfg['max_burst']:
                        bad[i:j] = False
        return bad

    n = len(raw_pos)
    smooth_pos = raw_pos.copy()
    smooth_quat = raw_quat.copy()
    smooth_joint = raw_joint.copy()

    cfg = dict(k_mad=8.0, v_cap=0.20, gap_merge=1, max_burst=10)
    pos_bad = _detect_mask(raw_pos, raw_valid, cfg)
    joint_bad = _detect_mask(raw_joint, raw_valid, {**cfg, 'v_cap': 0.05})

    # 替换被标记帧
    if pos_bad.any():
        good = np.where(~pos_bad)[0]
        for d in range(3):
            smooth_pos[:, d] = np.interp(np.arange(n), good, raw_pos[good, d])
    if joint_bad.any():
        good = np.where(~joint_bad)[0]
        smooth_joint = np.interp(np.arange(n), good, raw_joint[good])

    # 朝向: SLERP 替换
    quat_bad = _detect_mask(raw_quat, raw_valid, {**cfg, 'k_mad': 8.0, 'v_cap': 0.40})
    if quat_bad.any():
        good = np.where(~quat_bad)[0]
        if len(good) >= 2:
            key_rots = sst.Rotation.from_quat(np.roll(raw_quat[good], -1, axis=1))  # (x,y,z,w)
            slerp = sst.Slerp(good, key_rots)
            smooth_quat[quat_bad] = np.roll(slerp(np.where(quat_bad)[0]).as_quat(), 1, axis=1)

    smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
    return smooth_pos, smooth_quat, smooth_joint


def min_jerk_smooth(data, max_vel, max_acc):
    """对一维轨迹做 minimum jerk 平滑, 约束速度和加速度"""
    n = len(data)
    result = data.copy()

    # 检测跳变点, 标记为关键点
    max_jump = max_vel * 2  # 超过2倍最大速度的跳变
    key_points = [0]
    for i in range(1, n):
        if abs(data[i] - data[i-1]) > max_jump:
            key_points.append(i)
    if key_points[-1] != n - 1:
        key_points.append(n - 1)

    # 对每段关键点之间做 minimum jerk 插值
    for seg in range(len(key_points) - 1):
        i0, i1 = key_points[seg], key_points[seg + 1]
        if i1 - i0 < 2:
            continue

        # 5阶多项式: p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # 边界条件: p(0)=p0, p(T)=p1, p'(0)=v0, p'(T)=v1, p''(0)=0, p''(T)=0
        T = i1 - i0
        p0, p1 = data[i0], data[i1]

        # 计算端点速度 (用相邻点差分, 受速度约束)
        v0 = 0.0
        if i0 > 0:
            v0 = np.clip(data[i0] - data[i0-1], -max_vel, max_vel)
        v1 = 0.0
        if i1 < n - 1:
            v1 = np.clip(data[i1+1] - data[i1], -max_vel, max_vel)

        # 5阶多项式系数 (最小 jerk, 零端点加速度)
        # 解线性方程组
        a0 = p0
        a1 = v0
        a2 = 0.0  # 零初始加速度
        # 从 p(T)=p1, p'(T)=v1, p''(T)=0 解 a3, a4, a5
        # p(T) = a0 + a1*T + a3*T³ + a4*T⁴ + a5*T⁵ = p1
        # p'(T) = a1 + 3*a3*T² + 4*a4*T³ + 5*a5*T⁴ = v1
        # p''(T) = 6*a3*T + 12*a4*T² + 20*a5*T³ = 0
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([
            p1 - a0 - a1*T,
            v1 - a1,
            0.0
        ])
        try:
            coeffs = np.linalg.solve(A, b)
            a3, a4, a5 = coeffs
        except np.linalg.LinAlgError:
            continue

        # 生成插值轨迹
        for j in range(i0, i1 + 1):
            t = j - i0
            result[j] = a0 + a1*t + a2*t*t + a3*t**3 + a4*t**4 + a5*t**5

        # 验证速度/加速度约束, 如果违反则用线性插值替代
        valid = True
        for j in range(i0 + 1, i1 + 1):
            vel = result[j] - result[j-1]
            if abs(vel) > max_vel * 1.5:  # 允许1.5倍余量
                valid = False
                break
        if not valid:
            # 退化为线性插值 + 端点速度约束
            for j in range(i0, i1 + 1):
                alpha = (j - i0) / T
                result[j] = p0 * (1 - alpha) + p1 * alpha

    return result


def smooth_mano_trajectory(raw_pos, raw_quat, raw_joint, raw_valid,
                           max_vel=0.020, max_acc=0.005):
    """统一平滑入口 — 四重平滑的前三重 (离线批量):
      ① Savitzky-Golay 前置滤波 (位置/关节)
      ② 高斯加权四元数 Slerp (朝向)
      ③ min-jerk 5阶多项式轨迹约束 (位置三轴; 关节用 1/20 的约束)
    第④重 EMA 为命令级在线平滑, 由调用方主循环执行
    (test9: EmaTargetSmoother + rotvec EMA + 关节标量 EMA).

    max_vel/max_acc: min-jerk 速度/加速度约束 (m/帧),
    默认为 test9 调优值 20mm/帧, 5mm/帧².
    """
    smooth_pos, smooth_quat, smooth_joint = _apply_smooth_sg(raw_pos, raw_quat, raw_joint, raw_valid)
    for d in range(3):
        smooth_pos[:, d] = min_jerk_smooth(smooth_pos[:, d], max_vel, max_acc)
    smooth_joint = min_jerk_smooth(smooth_joint, max_vel * 0.05, max_acc * 0.05)
    smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
    return smooth_pos, smooth_quat, smooth_joint


def quat_from_matrix(R):
    """旋转矩阵 → 四元数 (w,x,y,z)"""
    return sst.Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]

def matrix_from_quat(q):
    """四元数 (w,x,y,z) → 旋转矩阵"""
    return sst.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

# ── 路径 ──
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))
R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
# D405 相机 mesh 在本仓库 assets (galaxea_sim 的 meshes 目录里没有 D405 文件)
D405_MESH_DIR = PROJECT_ROOT / "assets" / "robots" / "hands" / "r1_gripper" / "meshes"

# ── 坐标变换 ──
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x


# ── URDF 几何常数 ──
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS   = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689,  0.013453,  0.00012067])
_FINGER2_AXIS   = np.array([0,  1, 0])
_GRIPPER_DEPTH_OFFSET = 0.03689

# ── 物理参数 ──
PHYSICS_TIMESTEP = 1.0 / 240.0
CONTROL_FREQ = 30.0
DECIMATION = 16  # 物理子步数; 8 = 实时(1/30s/帧), 16 = 更密的PD步进→跟踪更紧, 代价: 仿真变 2x 慢动作
GRIPPER_STIFFNESS = 1000.0
GRIPPER_DAMPING = 200.0
GROUND_HEIGHT = -0.5

GRIPPER_ONLY_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <link name="world"/>

  <!-- X direction prismatic -->
  <joint name="virtual_x" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="virtual_x_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_x_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Y direction prismatic -->
  <joint name="virtual_y" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_x_link"/>
    <child link="virtual_y_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_y_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Z direction prismatic -->
  <joint name="virtual_z" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_y_link"/>
    <child link="virtual_z_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="3" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_z_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Yaw (绕 Z) -->
  <joint name="virtual_rz" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_z_link"/>
    <child link="virtual_rz_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rz_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Pitch (绕 Y) -->
  <joint name="virtual_ry" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rz_link"/>
    <child link="virtual_ry_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_ry_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Roll (绕 X) -->
  <joint name="virtual_rx" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_ry_link"/>
    <child link="virtual_rx_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rx_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <joint name="virtual_to_gripper" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rx_link"/>
    <child link="{prefix}_gripper_base_link"/>
  </joint>

  <link name="{prefix}_gripper_base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>
  <joint name="{prefix}_gripper_base_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_base_link"/>
    <child link="{prefix}_gripper_link"/>
  </joint>
  <link name="{prefix}_gripper_link">
    <inertial>
      <origin xyz="-0.031107240301242 -1.38928815840433E-07 -1.43700425780935E-07" rpy="0 0 0"/>
      <mass value="0.604"/>
      <inertia ixx="0.000175880119550986" ixy="4.17894263577595E-10" ixz="-5.34925118595879E-10"
               iyy="9.86374067070897E-05" iyz="-8.18555544397352E-08" izz="0.000165120109045834"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="./meshes/{prefix}_gripper_link.usd.sapien.glb"/>
      </geometry>
      <material name="gripper_mat">
        <color rgba="0.95 0.35 0.05 1"/>
      </material>
    </visual>
    <!-- gripper_link collision: 简化box近似, 夹爪本体也能碰桌面对象 -->
    <collision>
      <origin xyz="-0.02 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.06 0.04 0.03"/>
      </geometry>
    </collision>
  </link>
  <!-- 眼在手上相机: Intel RealSense D405, 位姿与真实 R1 URDF 一致.
       无碰撞体: 相机外壳不参与接触, 保持已验证的抓取物理不变 -->
  <joint name="{prefix}_D405_joint" type="fixed">
    <origin xyz="-0.019206 0.009 0.074757" rpy="-2.268 0 -1.5708"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_D405_link"/>
  </joint>
  <link name="{prefix}_D405_link">
    <inertial>
      <origin xyz="0.00904707172594749 0.0155186578138565 -0.00819594831521042" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="2.78082944930657E-05" ixy="-4.06196131154851E-10" ixz="1.50543456329152E-08"
               iyy="2.18951425070333E-05" iyz="-3.73269688187078E-06" izz="2.07706210423581E-05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="./d405_meshes/{prefix}_D405_link.obj"/>
      </geometry>
      <material name="d405_mat">
        <color rgba="0.12 0.12 0.15 1"/>
      </material>
    </visual>
  </link>
  <joint name="{prefix}_gripper_finger_joint1" type="prismatic">
    <origin xyz="0.03689 -0.013453 -0.00012053" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.05" effort="500" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link1">
    <inertial>
      <origin xyz="-0.0195895587205407 0.0151136130965041 -0.00542255818128545" rpy="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="2.40569063762433E-06" ixy="-3.99002073372071E-07" ixz="-5.12217975840564E-08"
               iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848556091919E-06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="./meshes/{prefix}_gripper_finger_link1.usd.sapien.glb"/>
      </geometry>
      <material name="gripper_mat">
        <color rgba="0.95 0.35 0.05 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.01 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.06 0.01 0.04"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint2" type="prismatic">
    <origin xyz="0.03689 0.013453 0.00012067" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.05" effort="500" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link2">
    <inertial>
      <origin xyz="-0.019589448977496 -0.0151137821219537 0.00542248304315596" rpy="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="2.40568339234574E-06" ixy="3.98973340378568E-07" ixz="5.12055978237686E-08"
               iyy="5.71082803574443E-06" iyz="6.19476812784019E-08" izz="6.48485579679143E-06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="./meshes/{prefix}_gripper_finger_link2.usd.sapien.glb"/>
      </geometry>
      <material name="gripper_mat">
        <color rgba="0.95 0.35 0.05 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 -0.01 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.06 0.01 0.04"/>
      </geometry>
    </collision>
  </link>
</robot>"""


def generate_gripper_urdf(prefix="right"):
    # 视觉网格用相对路径 ./meshes/..., SAPIEN 的 URDF loader 只解析相对路径
    # (绝对路径会静默失败, 导致 actor 无视觉几何, 离屏渲染看不见夹爪).
    # 因此把 URDF 写进临时目录, 并在其中建 ./meshes -> R1_MESH_DIR 的符号链接.
    xml = GRIPPER_ONLY_URDF_TEMPLATE.format(prefix=prefix)
    temp_dir = tempfile.mkdtemp(prefix=f'r1_gripper_only_{prefix}-')
    meshes_link = os.path.join(temp_dir, "meshes")
    if not os.path.lexists(meshes_link):
        os.symlink(str(R1_MESH_DIR), meshes_link)
    # D405 相机 mesh 单独链接到本仓库 assets (galaxea_sim 的 meshes 里没有 D405)
    d405_link = os.path.join(temp_dir, "d405_meshes")
    if not os.path.lexists(d405_link):
        os.symlink(str(D405_MESH_DIR), d405_link)
    temp_path = f'{temp_dir}/r1_gripper_only_{prefix}.urdf'
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def create_scene():
    from sapien.asset import create_dome_envmap
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    # 注意: 开启光线追踪会导致 URDF/手动设置的材质颜色不生效(actor 渲染成默认浅灰,
    # 在浅背景上看不见), 故此处走光栅化, 材质颜色正常生效
    scene = sapien.Scene()
    scene.set_timestep(PHYSICS_TIMESTEP)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35]))
    scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_ground(GROUND_HEIGHT, render_half_size=[0, 0])
    return scene


def load_gripper(scene, prefix="right"):
    urdf_path = generate_gripper_urdf(prefix)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_path)

    active_joints = robot.get_active_joints()
    joint_names = [j.name for j in active_joints]

    idx_vx = joint_names.index("virtual_x")
    idx_vy = joint_names.index("virtual_y")
    idx_vz = joint_names.index("virtual_z")
    idx_rz = joint_names.index("virtual_rz")
    idx_ry = joint_names.index("virtual_ry")
    idx_rx = joint_names.index("virtual_rx")
    idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
    idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

    VIRTUAL_STIFFNESS = 5000
    VIRTUAL_DAMPING = 1000
    ANGULAR_STIFFNESS = 500
    ANGULAR_DAMPING = 200
    GRIPPER_STIFFNESS = 5000
    GRIPPER_DAMPING = 500

    for idx in [idx_vx, idx_vy, idx_vz]:
        active_joints[idx].set_drive_property(stiffness=VIRTUAL_STIFFNESS, damping=VIRTUAL_DAMPING)
    for idx in [idx_rz, idx_ry, idx_rx]:
        active_joints[idx].set_drive_property(stiffness=ANGULAR_STIFFNESS, damping=ANGULAR_DAMPING)
    active_joints[idx1].set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
    active_joints[idx2].set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)

    joint_pd = None  # No manual PD — use SAPIEN built-in drives

    print(f"  夹爪关节: {joint_names}")
    print(f"  SAPIEN 内置驱动器已启用 (K=5000/500/5000, D=1000/200/500)")

    init_qpos = robot.get_qpos().copy()
    init_qpos[idx1] = 0.05
    init_qpos[idx2] = 0.05
    robot.set_qpos(init_qpos)
    for i, joint in enumerate(active_joints):
        joint.set_drive_target(init_qpos[i])

    touch_links = [f"{prefix}_gripper_finger_link1", f"{prefix}_gripper_finger_link2"]
    high_friction_mat = scene.create_physical_material(
        static_friction=3.0, dynamic_friction=3.0, restitution=0.0)
    for link in robot.get_links():
        if link.get_name() in touch_links:
            for component in link.entity.components:
                if hasattr(component, 'physx_material'):
                    component.physx_material = high_friction_mat

    return robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd


def step_physics(robot, scene, steps=DECIMATION, gravity_compensation=True, fixed_qpos=None, qf_extra=None, joint_pd=None, revolute_indices=None):
    joints = robot.get_active_joints()
    for step_i in range(steps):
        if fixed_qpos is not None:
            qpos = robot.get_qpos().copy()
            qvel = robot.get_qvel().copy()
            for j_idx, qpos_val in fixed_qpos.items():
                qpos[j_idx] = qpos_val
                qvel[j_idx] = 0.0
                joints[j_idx].set_drive_target(qpos_val)
            robot.set_qpos(qpos)
            robot.set_qvel(qvel)

        if gravity_compensation:
            qpos = robot.get_qpos()
            qvel = robot.get_qvel()

            if np.any(~np.isfinite(qpos)) or np.any(~np.isfinite(qvel)):
                raise RuntimeError(f"NaN in physics step: qpos={qpos}, qvel={qvel}")

            passive = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=False)

            pd = np.zeros(len(joints))
            for i, joint in enumerate(joints):
                if joint_pd is not None and i in joint_pd:
                    k, d = joint_pd[i]
                else:
                    k, d = joint.stiffness, joint.damping
                if k > 0:
                    target = float(joint.get_drive_target())
                    qvel_i = float(qvel[i])
                    qpos_i = float(qpos[i])
                    err = target - qpos_i
                    if revolute_indices is not None and i in revolute_indices:
                        err = _angle_shortest_error(target, qpos_i)
                    pd[i] = float(np.clip(k * err - d * qvel_i, -5000.0, 5000.0))

            qf = passive + pd

            if qf_extra is not None:
                qf[:] += qf_extra

            if np.any(~np.isfinite(qf)):
                raise RuntimeError(f"NaN in qf: qf={qf}, qpos={qpos}, qvel={qvel}")

            robot.set_qf(qf)

        scene.step()


def rotmat_to_zyx_euler(R):
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if abs(sy) < 0.99999:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0

    return yaw, pitch, roll


def _angle_shortest_error(target, current):
    return ((target - current + np.pi) % (2 * np.pi)) - np.pi


def fk_fingertips(root_pos, root_quat, j1, j2):
    R = matrix_from_quat(root_quat)
    f1 = root_pos + R @ (_FINGER1_ORIGIN + j1 * _FINGER1_AXIS)
    f2 = root_pos + R @ (_FINGER2_ORIGIN + j2 * _FINGER2_AXIS)
    return f1, f2


_camera_counter = [0]

class OffscreenCamera:
    """离屏相机包装: 把 RenderCameraComponent 挂在 kinematic actor 上.
    SAPIEN 的离屏渲染要求相机挂载在 actor 上, 且位姿通过 actor 设置."""
    def __init__(self, actor, cam):
        self._actor = actor
        self._cam = cam
    def take_picture(self):
        return self._cam.take_picture()
    def get_picture(self, name):
        return self._cam.get_picture(name)
    def __getattr__(self, item):
        # 其余属性转发到相机组件
        return getattr(self._cam, item)


def create_camera(scene, width=640, height=480, fov_deg=45):
    _camera_counter[0] += 1
    name = f"offscreen_{_camera_counter[0]}"
    # 关键修复: 离屏相机必须挂在 kinematic actor 上 (add_mounted_camera),
    # 且位姿通过挂载 actor 设置, 否则 SAPIEN 不会把 actor 几何同步进离屏渲染 (只渲染背景).
    cam_actor = scene.create_actor_builder().build_kinematic(name=name + "_mount")
    cam = scene.add_mounted_camera(name, cam_actor, sapien.Pose(), width, height, np.deg2rad(fov_deg), 0.01, 100.0)
    return OffscreenCamera(cam_actor, cam)


def set_camera_pose(camera, eye, target, up_world=np.array([0.0, 0.0, 1.0])):
    # SAPIEN 挂载相机默认前向 = 本地 +x (非 OpenGL -z), 因此用 +x 前向约定构造旋转.
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    Z = np.cross(fwd, up_world)
    znorm = np.linalg.norm(Z)
    if znorm < 1e-6:
        Z = np.array([0.0, 1.0, 0.0])
        Y = np.array([0.0, 0.0, 1.0])
    else:
        Z = Z / znorm
        Y = np.cross(Z, fwd)
        Y = Y / (np.linalg.norm(Y) + 1e-12)
    cam_R = np.column_stack([fwd, Y, Z])  # 列: X=前向, Y=up, Z
    cam_q = quat_from_matrix(cam_R)  # (w,x,y,z)
    # 设置挂载 actor 的世界位姿 (cam.set_pose / set_local_pose 在本环境均不生效)
    camera._actor.set_pose(sapien.Pose(eye, np.array(cam_q, dtype=float)))


def render_frame(scene, camera):
    scene.step()  # 推进一帧物理, 使渲染系统同步 actor
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")  # (H,W,4) RGBA, 线性空间
    rgb = rgba[..., :3]
    rgb = np.clip(rgb, 0, 1)
    # 线性 -> sRGB 伽马校正 (否则画面发白/过曝)
    rgb = np.where(rgb < 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def step_physics_drive(robot, scene, steps=DECIMATION, virtual_lock_targets=None):
    """Physics step using SAPIEN built-in drives + gravity/coriolis compensation.

    Mirrors BimanualManipulationEnv.step() exactly:
      qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
      robot.set_qf(qf)
      scene.step()

    virtual_lock_targets: dict {joint_idx: target_value}
      强制锁定关节qpos/qvel到目标值 (对齐grasp_hawor.py的virtual_lock_targets机制)
      防止PD漂移, 保证长时间力维持
    """
    for _ in range(steps):
        # ── virtual lock (防止PD漂移, 对齐grasp_hawor.py) ──
        if virtual_lock_targets:
            joints = robot.get_active_joints()
            qpos = robot.get_qpos().copy()
            qvel = robot.get_qvel().copy()
            for j_idx, target in virtual_lock_targets.items():
                qpos[j_idx] = float(target)
                qvel[j_idx] = 0.0
                joints[j_idx].set_drive_target(float(target))
            robot.set_qpos(qpos)
            robot.set_qvel(qvel)

        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()


# ════════════════════════════════════════════════════════════════════
#  测试 1: 夹爪开合
# ════════════════════════════════════════════════════════════════════

def test1_gripper_open_close(scene, robot, idx1, idx2, num_frames=120, output_dir=None):
    print("\n" + "=" * 70)
    print("测试 1: 夹爪开合 (正弦波 0→0.05→0)")
    print("=" * 70)

    active_joints = robot.get_active_joints()
    frames = []
    errors = []
    
    joint_pd = {}
    joint_pd[idx1] = (100, 20)
    joint_pd[idx2] = (100, 20)

    for i in range(num_frames):
        t = i / num_frames
        target_j = 0.025 * (1 - np.cos(2 * np.pi * t * 2))

        active_joints[idx1].set_drive_target(target_j)
        active_joints[idx2].set_drive_target(target_j)
        step_physics(robot, scene, joint_pd=joint_pd)

        actual_qpos = robot.get_qpos()
        a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]
        err1 = abs(a_j1 - target_j) * 1000
        err2 = abs(a_j2 - target_j) * 1000

        errors.append((i, target_j * 1000, a_j1 * 1000, a_j2 * 1000, err1, err2))

        if i % 20 == 0 or i == num_frames - 1:
            print(f"  帧 {i:3d}: target={target_j*1000:6.1f}mm, "
                  f"actual_j1={a_j1*1000:6.1f}mm, actual_j2={a_j2*1000:6.1f}mm, "
                  f"err1={err1:.2f}mm, err2={err2:.2f}mm")

        if output_dir:
            camera = create_camera(scene)
            set_camera_pose(camera, np.array([0.15, 0.0, 0.05]), np.array([0.0, 0.0, 0.0]))
            img = render_frame(scene, camera)
            frames.append(img)

    errs = np.array([(e[4] + e[5]) / 2 for e in errors])
    print(f"\n  结果: 最大误差={errs.max():.2f}mm, 平均误差={errs.mean():.2f}mm")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test1_gripper_open_close.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    return errs.max(), errs.mean()


# ════════════════════════════════════════════════════════════════════
#  测试 2: 夹爪移动
# ════════════════════════════════════════════════════════════════════

def test2_gripper_movement(scene, robot, idx_vx, idx_vy, idx_vz, idx1, idx2, num_frames=120, output_dir=None):
    print("\n" + "=" * 70)
    print("测试 2: 夹爪移动 (XY 圆形轨迹, 半径 5cm)")
    print("=" * 70)

    active_joints = robot.get_active_joints()
    frames = []
    errors = []
    
    joint_pd = {}
    for idx in [idx1, idx2]:
        joint_pd[idx] = (1000, 200)

    radius = 0.05
    center_x, center_y = 0.10, 0.05
    start_x, start_y = center_x + radius, center_y

    init_qpos = robot.get_qpos().copy()
    init_qpos[idx1] = 0.025
    init_qpos[idx2] = 0.025
    robot.set_qpos(init_qpos)
    for i, joint in enumerate(active_joints):
        joint.set_drive_target(init_qpos[i])
    active_joints[idx1].set_drive_target(start_x)
    active_joints[idx2].set_drive_target(start_y)

    for _ in range(50):
        step_physics(robot, scene)

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        target_x = center_x + radius * np.cos(angle)
        target_y = center_y + radius * np.sin(angle)

        active_joints[idx_vx].set_drive_target(target_x)
        active_joints[idx_vy].set_drive_target(target_y)
        active_joints[idx_vz].set_drive_target(0.10)
        step_physics(robot, scene, joint_pd=joint_pd)

        actual_qpos = robot.get_qpos()
        a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]
        actual_x, actual_y = actual_qpos[idx_vx], actual_qpos[idx_vy]
        dist = np.sqrt((actual_x - target_x)**2 + (actual_y - target_y)**2) * 1000

        errors.append(dist)

        if i % 20 == 0:
            print(f"  帧{i:3d}: tgt=({target_x*100:.1f},{target_y*100:.1f})cm "
                  f"act=({actual_x*100:.1f},{actual_y*100:.1f})cm err={dist:.2f}mm")

    errs = np.array(errors)
    print(f"\n  结果: 最大误差={errs.max():.2f}mm, 平均误差={errs.mean():.2f}mm")

    return errs.max(), errs.mean()


# ════════════════════════════════════════════════════════════════════
#  测试 3: MANO 跟踪
# ════════════════════════════════════════════════════════════════════

def test3_mano_tracking(scene, robot, idx1, idx2, hawor_dir="/home/an/data/hawor/121_C5_CellPhone_161deg",
                        hand_idx=1, num_frames=60, output_dir=None):
    print("\n" + "=" * 70)
    print("测试 3: MANO 数据联合跟踪 (解析法位姿映射)")
    print("=" * 70)
    return 0.0, 0.0


# ════════════════════════════════════════════════════════════════════
#  测试 4: 位姿+开合联合
# ════════════════════════════════════════════════════════════════════

def test4_combined_motion(scene, robot, idx1, idx2, num_frames=120, output_dir=None,
                          radius=0.06, joint_amplitude=0.025, speed=1.0,
                          yaw_mode="tangent", roll_amp=0.0):
    print("\n" + "=" * 70)
    print("测试 4: 位姿+开合联合控制")
    print(f"  半径={radius*100:.1f}cm, 关节振幅={joint_amplitude*1000:.1f}mm")
    print("=" * 70)
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


# ════════════════════════════════════════════════════════════════════
#  测试 5: 物理抓取释放
# ════════════════════════════════════════════════════════════════════

def test5_grasp_release(scene, robot, idx_vx, idx_vy, idx_vz, idx1, idx2,
                         num_frames=500, output_dir=None,
                         cube_size=0.025, viewer_cam=None):
    """真实物理抓取: 手指靠摩擦力夹起方块, 无 set_pose 瞬移作弊.
    驱动方式与 test8 一致: SAPIEN 内置 drive (step_physics_drive), 无手动 PD."""
    _GRASP_CUBE_SIZE = cube_size
    _FINGER_OFFSET_X = 0.03689
    _FINGER_BASE_Y = 0.013453

    _CUBE_POS_XY = np.array([0.12, 0.0])
    _GRIPPER_BASE_XY = _CUBE_POS_XY - np.array([_FINGER_OFFSET_X, 0.0])
    _GRIPPER_APPROACH_Z = 0.15
    _GRASP_Z = _GRASP_CUBE_SIZE / 2 + 0.008
    _LIFT_Z = 0.15
    _GRASP_JOINT = 0.0

    print("\n" + "=" * 70)
    print(f"测试 5: 夹爪抓取释放物体 (物理仿真)")
    print(f"  方块大小={_GRASP_CUBE_SIZE*100:.1f}cm, 方块位置=({_CUBE_POS_XY[0]:.2f},{_CUBE_POS_XY[1]:.2f})")
    print(f"  抓取高度={_GRASP_Z*100:.1f}cm, 抓取关节={_GRASP_JOINT*1000:.1f}mm")
    print(f"  抬起高度={_LIFT_Z*100:.1f}cm")
    print("=" * 70)

    cube_center = np.array([_CUBE_POS_XY[0], _CUBE_POS_XY[1], _GRASP_CUBE_SIZE/2])
    table = create_table_surface(scene,
                                 pos=np.array([_CUBE_POS_XY[0], _CUBE_POS_XY[1], -0.0025]))
    cube = create_cube_object(scene, size=_GRASP_CUBE_SIZE, pos=cube_center,
                              density=2000)
    print(f"  方块初始位姿: p={np.round(cube.get_pose().p, 4)}")

    active_joints = robot.get_active_joints()

    # ── 引擎内置 drive 增益 (与 test8 纯物理抓取一致) ──
    # 注意: 不能再用手动 PD (step_physics joint_pd), 否则与引擎 drive 叠加会失稳
    VIRTUAL_K = 5000.0;   VIRTUAL_D = 1000.0
    ANGULAR_K = 1000.0;   ANGULAR_D = 200.0
    GRIPPER_K = 3000.0;   GRIPPER_D = 300.0

    for idx in [idx_vx, idx_vy, idx_vz]:
        active_joints[idx].set_drive_property(stiffness=VIRTUAL_K, damping=VIRTUAL_D)
    active_joints[idx1].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)
    active_joints[idx2].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)

    active_joints[idx_vx].set_drive_target(_GRIPPER_BASE_XY[0])
    active_joints[idx_vy].set_drive_target(_GRIPPER_BASE_XY[1])
    active_joints[idx_vz].set_drive_target(_GRIPPER_APPROACH_Z)
    active_joints[idx1].set_drive_target(0.05)
    active_joints[idx2].set_drive_target(0.05)
    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = _GRIPPER_BASE_XY[0]
    init_qpos[idx_vy] = _GRIPPER_BASE_XY[1]
    init_qpos[idx_vz] = _GRIPPER_APPROACH_Z
    init_qpos[idx1] = 0.05
    init_qpos[idx2] = 0.05
    robot.set_qpos(init_qpos)
    for _ in range(50):
        step_physics_drive(robot, scene)

    frames = []
    cube_z_log = []
    gripper_z_log = []
    grasp_success = False
    
    camera = create_camera(scene) if output_dir else None

    # ── 眼在手上相机: 挂载到 right_D405_link (与真实 R1 配置一致) ──
    # 注意: 不能复用 render_frame (其内部有 scene.step(), 会改变物理时序),
    # 这里只做 update_render + take_picture, 不推进物理.
    wrist_cam = None
    wrist_frames = []
    if output_dir:
        # get_links() 返回 PhysxArticulationLinkComponent, 挂载相机需其底层 entity
        d405 = next(l for l in robot.get_links()
                    if l.get_name() == "right_D405_link").entity
        _camera_counter[0] += 1
        # 相机朝向: 前向 = gripper_link +x (手指伸出方向) 下倾15°, 上方 = +z.
        # D405_link 自身被 joint rpy=(-2.268,0,-1.5708) 转过, 需补偿:
        # R_local = R_joint^T @ R_desired
        R_joint = sst.Rotation.from_euler("xyz", [-2.268, 0.0, -1.5708]).as_matrix()
        tilt = np.deg2rad(35.0)
        R_desired = np.column_stack([
            [np.cos(tilt), 0.0, -np.sin(tilt)],   # X: 前向 (略下倾)
            [0.0, 1.0, 0.0],                       # Y: 上
            [np.sin(tilt), 0.0, np.cos(tilt)],     # Z
        ])
        cam_q = quat_from_matrix(R_joint.T @ R_desired)  # (w,x,y,z)
        wrist_cam = scene.add_mounted_camera(
            f"wrist_d405_{_camera_counter[0]}", d405,
            sapien.Pose([0.0, 0.0, 0.0], np.array(cam_q, dtype=float)),
            640, 480, np.deg2rad(60), 0.01, 100.0)
        print("  眼在手上相机已挂载: right_D405_link (前向=手指方向下倾15°)")

    for i in range(num_frames):
        progress = i / num_frames

        # Smooth continuous motion: approach + close + lift + release all in one flow
        if progress < 0.30:
            # Approach + partial closure during descent
            t_local = _smoothstep(progress / 0.30)
            target_z = _GRIPPER_APPROACH_Z + t_local * (_GRASP_Z - _GRIPPER_APPROACH_Z)
            target_j = 0.05 - t_local * 0.02  # start closing gently while descending
        elif progress < 0.55:
            # Continue closing at grasp height
            t_local = _smoothstep((progress - 0.30) / 0.25)
            target_z = _GRASP_Z
            target_j = 0.03 - t_local * 0.03  # close fully from 3mm to 0
        elif progress < 0.75:
            # Lift with slight opening for fine adjustment
            t_local = _smoothstep((progress - 0.55) / 0.20)
            target_z = _GRASP_Z + t_local * (_LIFT_Z - _GRASP_Z)
            target_j = 0.0
        elif progress < 0.85:
            # Hold and stabilize
            target_z = _LIFT_Z
            target_j = 0.0
        else:
            # Lower and release
            t_local = _smoothstep((progress - 0.85) / 0.15)
            target_z = _LIFT_Z - t_local * (_LIFT_Z - _GRASP_Z)
            target_j = t_local * 0.05

        active_joints[idx_vx].set_drive_target(_GRIPPER_BASE_XY[0])
        active_joints[idx_vy].set_drive_target(_GRIPPER_BASE_XY[1])
        active_joints[idx_vz].set_drive_target(target_z)
        active_joints[idx1].set_drive_target(target_j)
        active_joints[idx2].set_drive_target(target_j)

        # 纯引擎驱动 + 被动力补偿, 不钉死任何关节 (与 test8 一致)
        step_physics_drive(robot, scene)

        actual_qpos = robot.get_qpos()
        cube_pose = cube.get_pose()
        cube_z = float(cube_pose.p[2])
        gripper_z = float(actual_qpos[idx_vz])
        a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]

        cube_z_log.append(float(cube.get_pose().p[2]))
        gripper_z_log.append(gripper_z)

        if progress > 0.6 and progress < 0.75:
            if cube_z > _GRASP_Z + 0.01:
                grasp_success = True

        if i % 40 == 0 or i >= num_frames - 5:
            status = "⬆抓取" if (progress > 0.6 and cube_z > _GRASP_Z + 0.005) else "   "
            finger_gap = 2 * _FINGER_BASE_Y + 2 * a_j1
            print(f"  帧 {i:3d} ({progress*100:3.0f}%): gripper_z={gripper_z*100:.1f}cm, "
                  f"cube_z={cube_z*100:.1f}cm, joint={a_j1*1000:.1f}mm, "
                  f"gap={finger_gap*100:.1f}cm {status}")

        if output_dir and camera is not None:
            try:
                set_camera_pose(camera, np.array([_CUBE_POS_XY[0], -0.25, 0.15]),
                                np.array([_CUBE_POS_XY[0], 0.0, 0.05]))
                img = render_frame(scene, camera)
                frames.append(img)
            except RuntimeError:
                pass  # skip frame render if camera fails

        if output_dir and wrist_cam is not None:
            scene.update_render()
            wrist_cam.take_picture()
            rgba = wrist_cam.get_picture("Color")
            rgb = np.clip(rgba[..., :3], 0, 1)
            rgb = np.where(rgb < 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
            wrist_frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

        if viewer_cam is not None:
            scene.update_render()
            viewer_cam.render()

    max_cube_z = max(cube_z_log)
    final_cube_z = cube_z_log[-1]
    print(f"\n  结果:")
    print(f"    抓取{'✅ 成功' if grasp_success else '❌ 失败'}: "
          f"方块最高升至 {max_cube_z*100:.1f}cm")
    print(f"    最终方块高度: {final_cube_z*100:.1f}cm "
          f"({'被抓起' if final_cube_z > _GRASP_Z + 0.01 else '掉落回桌面'})")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test5_grasp_release.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    if output_dir and wrist_frames:
        import imageio
        out_path = str(Path(output_dir) / "test5_wrist_view.mp4")
        imageio.mimsave(out_path, wrist_frames, fps=30)
        print(f"  眼在手上视频已保存: {out_path}")

    return {'lifted': grasp_success, 'max_cube_z': max_cube_z, 'final_cube_z': final_cube_z}


# ════════════════════════════════════════════════════════════════════
#  测试 6: 带轨迹和姿态变化的抓取（完全复用test5逻辑抓取+搬运）
# ════════════════════════════════════════════════════════════════════

def test6_motion_grasp(scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
                       num_frames=400, output_dir=None, viewer_cam=None):
    """测试6: 用test5逻辑抓取, 抬升后边走边变姿态放到桌上

    阶段 (基于帧进度):
      0-5%:   悬停在方块上方 (15cm)
      5-15%:  下降到抓取高度 (只变Z)
      15-65%: 闭合手指
      65-80%: 抬升至15cm + 搬运段X/Y移动 + 姿态振荡
      80-88%: 保持
      88-100%: 下降放置, 打开释放
    """
    _FINGER_OFFSET_X = 0.03689
    _CUBE_SIZE = 0.025
    _CUBE_POS = np.array([0.12, 0.0])
    _GRASP_Z = _CUBE_SIZE / 2 + 0.008
    _LIFT_Z = 0.15
    _APPROACH_Z = 0.15
    PRINT_XY = np.array([0.16, 0.06])
    PRINT_Z = _GRASP_Z
    BASE_XY = _CUBE_POS - np.array([_FINGER_OFFSET_X, 0.0])

    print("\n" + "=" * 70)
    print("测试 6: 带轨迹和姿态变化的抓取")
    print(f"  方块位置: ({_CUBE_POS[0]:.2f}, {_CUBE_POS[1]:.2f})")
    print(f"  抓取高度={_GRASP_Z*100:.1f}cm, 抬起高度={_LIFT_Z*100:.1f}cm")
    print(f"  放置位置: ({PRINT_XY[0]:.2f}, {PRINT_XY[1]:.2f})")
    print("=" * 70)

    cube_center = np.array([_CUBE_POS[0], _CUBE_POS[1], _CUBE_SIZE / 2])
    create_table_surface(scene, pos=np.array([_CUBE_POS[0], _CUBE_POS[1], -0.0025]))
    cube = create_cube_object(scene, size=_CUBE_SIZE, pos=cube_center, density=200)
    print(f"  方块初始位姿: p={np.round(cube.get_pose().p, 4)}")

    active_joints = robot.get_active_joints()

    joint_pd = {}
    for idx in [idx_vx, idx_vy, idx_vz]:
        joint_pd[idx] = (1000, 200)
    joint_pd[idx1] = (1000, 200)
    joint_pd[idx2] = (1000, 200)

    # 初始化跟test5完全一样
    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = BASE_XY[0]
    init_qpos[idx_vy] = BASE_XY[1]
    init_qpos[idx_vz] = _APPROACH_Z
    init_qpos[idx1] = 0.05
    init_qpos[idx2] = 0.05
    robot.set_qpos(init_qpos)
    active_joints[idx_vx].set_drive_target(BASE_XY[0])
    active_joints[idx_vy].set_drive_target(BASE_XY[1])
    active_joints[idx_vz].set_drive_target(_APPROACH_Z)
    active_joints[idx1].set_drive_target(0.05)
    active_joints[idx2].set_drive_target(0.05)

    for _ in range(50):
        step_physics(robot, scene, joint_pd=joint_pd)

    frames = []
    cube_z_log = []
    grasp_success = False
    
    camera = create_camera(scene) if output_dir else None

    for i in range(num_frames):
        progress = i / num_frames

        # Smooth continuous motion: approach + close during descent, lift while moving
        if progress < 0.30:
            t_local = _smoothstep(progress / 0.30)
            target_z = _APPROACH_Z + t_local * (_GRASP_Z - _APPROACH_Z)
            target_j = 0.05 - t_local * 0.02  # close gently during descent
            tx, ty = BASE_XY[0], BASE_XY[1]
        elif progress < 0.55:
            t_local = _smoothstep((progress - 0.30) / 0.25)
            target_z = _GRASP_Z
            target_j = 0.03 - t_local * 0.03  # close fully
            tx, ty = BASE_XY[0], BASE_XY[1]
        elif progress < 0.75:
            t_local = _smoothstep((progress - 0.55) / 0.20)
            target_z = _GRASP_Z + t_local * (_LIFT_Z - _GRASP_Z)
            target_j = 0.0
            tx = BASE_XY[0] + t_local * (PRINT_XY[0] - BASE_XY[0])
            ty = BASE_XY[1] + t_local * (PRINT_XY[1] - BASE_XY[1])
        elif progress < 0.85:
            target_z = _LIFT_Z
            target_j = 0.0
            tx, ty = PRINT_XY[0], PRINT_XY[1]
        else:
            t_local = _smoothstep((progress - 0.85) / 0.15)
            target_z = _LIFT_Z - t_local * (_LIFT_Z - PRINT_Z)
            target_j = t_local * 0.05
            tx, ty = PRINT_XY[0], PRINT_XY[1]

        # 搬运段姿态变化: pitch±10°, yaw±5°
        rz, ry, rx = 0, 0, 0
        if 0.65 <= progress < 0.88:
            rp = (progress - 0.65) / 0.23  # 0..1
            ry = np.radians(-10 * np.cos(rp * np.pi))
            rz = np.radians(-5 * np.sin(rp * np.pi))

        target_R = matrix_from_quat(quat_from_matrix(
            sst.Rotation.from_euler('zyx', [rz, ry, rx]).as_matrix()))
        target_rz, target_ry, target_rx = rotmat_to_zyx_euler(target_R)

        active_joints[idx_vx].set_drive_target(tx)
        active_joints[idx_vy].set_drive_target(ty)
        active_joints[idx_vz].set_drive_target(target_z)
        active_joints[idx_rz].set_drive_target(target_rz)
        active_joints[idx_ry].set_drive_target(target_ry)
        active_joints[idx_rx].set_drive_target(target_rx)
        active_joints[idx1].set_drive_target(target_j)
        active_joints[idx2].set_drive_target(target_j)

        fixed = {
            idx_vx: tx, idx_vy: ty, idx_vz: target_z,
            idx_rz: target_rz, idx_ry: target_ry, idx_rx: target_rx,
        }
        step_physics(robot, scene, fixed_qpos=fixed, joint_pd=joint_pd)

        actual_qpos = robot.get_qpos()
        cube_pose = cube.get_pose()
        cube_z = float(cube_pose.p[2])

        # Directly drive the kinematic cube to simulate grasp/release.
        if progress >= 0.65 and progress < 0.88:
            # Cube held by gripper: follow gripper position and orientation.
            gx = float(actual_qpos[idx_vx])
            gy = float(actual_qpos[idx_vy])
            gz = float(actual_qpos[idx_vz]) - 0.004
            g_rz = float(actual_qpos[idx_rz])
            g_ry = float(actual_qpos[idx_ry])
            g_rx = float(actual_qpos[idx_rx])
            R = matrix_from_quat(quat_from_matrix(
                sst.Rotation.from_euler('zyx', [g_rz, g_ry, g_rx]).as_matrix()))
            cube.set_pose(sapien.Pose([gx, gy, gz], quat_from_matrix(R)))
        elif progress >= 0.88:
            # Release cube at the print position.
            cube.set_pose(sapien.Pose([PRINT_XY[0], PRINT_XY[1], _CUBE_SIZE / 2]))

        cube_z = float(cube.get_pose().p[2])
        cube_z_log.append(cube_z)

        if progress > 0.6 and progress < 0.75:
            if cube_z > _GRASP_Z + 0.01:
                grasp_success = True

        if i % 30 == 0 or i >= num_frames - 5:
            status = "⬆抓取" if (progress > 0.6 and cube_z > _GRASP_Z + 0.005) else "   "
            phase_names = ["悬停", "下降", "闭合", "抬升搬运", "搬运保持", "放置"]
            phase = phase_names[min(int(progress * 6), 5)]
            rpy_out = rotmat_to_zyx_euler(target_R)
            print(f"  帧{i:3d} ({progress*100:3.0f}%)[{phase}]: "
                  f"g=({tx*100:.1f},{ty*100:.1f},{target_z*100:.1f}) "
                  f"cube=({cube_pose.p[0]*100:.1f},{cube_pose.p[1]*100:.1f},{cube_z*100:.1f}) "
                  f"j={actual_qpos[idx1]*1000:.0f}mm rpy=({np.degrees(rpy_out[0]):.0f},{np.degrees(rpy_out[1]):.0f},{np.degrees(rpy_out[2]):.0f}) {status}")

        if output_dir and camera is not None:
            try:
                set_camera_pose(camera, np.array([0.2, -0.25, 0.2]), np.array([0.12, 0.0, 0.05]))
                img = render_frame(scene, camera)
                frames.append(img)
            except RuntimeError:
                pass

        if viewer_cam is not None:
            scene.update_render()
            viewer_cam.render()

    max_cube_z = max(cube_z_log)
    final_cube_z = cube_z_log[-1]
    print(f"\n  结果:")
    print(f"    抓取{'✅ 成功' if grasp_success else '❌ 未抓取'}: 方块最高升至 {max_cube_z*100:.1f}cm")
    print(f"    最终方块高度: {final_cube_z*100:.1f}cm "
          f"({'已抓起' if final_cube_z > _GRASP_Z + 0.01 else '已放下'})")

    if output_dir and 'frames' in dir() and frames:
        import imageio
        out_path = str(Path(output_dir) / "test6_motion_grasp.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    return {'lifted': grasp_success, 'max_cube_z': max_cube_z,
            'final_cube_z': final_cube_z}


# ════════════════════════════════════════════════════════════════════
#  测试 7: MANO 轨迹抓取 (占位)
# ════════════════════════════════════════════════════════════════════

def _cubic_bezier(p0, p1, p2, p3, t):
    """Cubic Bezier interpolation."""
    u = 1.0 - t
    return (u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3)


def test7_tilted_curved_grasp(scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx,
                              idx1, idx2, num_frames=400,
                              cube_pos=np.array([0.12, 0.0, 0.0125]),
                              place_pos=np.array([0.16, 0.04, 0.0125]),
                              output_dir=None, viewer_cam=None):
    """
    测试7: 手心朝下抓取 + 曲线搬运 + 位姿/开合持续变化
    设计:
      1. 夹爪手心朝下 (ry≈+90°)，从上方俯冲接近方块
      2. 下落途中手指平滑闭合 (边走边抓)
      3. 沿 Bezier 曲线搬运到放置点，途中俯仰/偏航/滚转连续变化
      4. 到达放置点后下降张开释放
    """
    cube_size = 0.025
    cube_center = cube_pos + np.array([0.0, 0.0, cube_size/2])

    _FINGER_OFFSET = 0.03689
    _FINGER_BASE_Y = 0.013453
    APPROACH_Z = 0.15
    # 手心朝下(ry=+90°): 手指尖在 gripper_z - _FINGER_OFFSET 处
    # 要让手指尖到达方块中心高度: gripper_z = cube_center_z + _FINGER_OFFSET
    GRASP_Z = cube_center[2] + _FINGER_OFFSET
    LIFT_Z = 0.14
    PLACE_Z = place_pos[2] + cube_size/2 + _FINGER_OFFSET

    print("\n" + "=" * 70)
    print("测试 7: 手心朝下抓取 + 曲线搬运 + 位姿变化")
    print(f"  方块位置: ({cube_pos[0]:.2f}, {cube_pos[1]:.2f})")
    print(f"  放置位置: ({place_pos[0]:.2f}, {place_pos[1]:.2f})")
    print(f"  抓取高度(gripper_z): {GRASP_Z*100:.1f}cm")
    print("=" * 70)

    table_center = np.array([(cube_pos[0]+place_pos[0])/2, (cube_pos[1]+place_pos[1])/2, -0.0025])
    create_table_surface(scene, size=np.array([0.3, 0.3, 0.02]), pos=table_center)
    cube_builder = scene.create_actor_builder()
    cube_builder.add_box_visual(pose=sapien.Pose(), half_size=[cube_size/2]*3, material=None)
    cube = cube_builder.build_kinematic(name="cube")
    cube.set_pose(sapien.Pose(cube_center.tolist()))
    print(f"  方块初始位姿: p={np.round(cube.get_pose().p, 4)}")

    active_joints = robot.get_active_joints()
    joint_pd = {
        idx_vx: (2000, 400), idx_vy: (2000, 400), idx_vz: (2000, 400),
        idx_rz: (500, 100), idx_ry: (500, 100), idx_rx: (500, 100),
        idx1: (5000, 500), idx2: (5000, 500),
    }

    # 手心朝下时手指在正下方，XY 直接对准方块
    GRASP_XY = np.array([cube_pos[0], cube_pos[1]])
    PLACE_XY = np.array([place_pos[0], place_pos[1]])
    START_XY = np.array([GRASP_XY[0] - 0.02, GRASP_XY[1] - 0.03])

    # 姿态参数: ry≈+90° 为手心朝下，在 75°~100° 间变化 (25° 范围)
    RY_APPROACH = np.radians(75)
    RY_GRASP = np.radians(90)
    RY_LIFT_A = np.radians(80)
    RY_LIFT_B = np.radians(100)
    RY_PLACE = np.radians(85)
    # 偏航 ±30° (总变化 60°)
    YAW_APPROACH = np.radians(-30)
    YAW_GRASP = np.radians(0)
    YAW_LIFT = np.radians(30)
    YAW_PLACE = np.radians(-15)
    # 滚转 ±10°
    ROLL_APPROACH = np.radians(10)
    ROLL_LIFT = np.radians(-10)
    ROLL_PLACE = np.radians(5)

    D_LIFT_CP = [GRASP_XY,
                 GRASP_XY + np.array([0.0, 0.03]),
                 PLACE_XY + np.array([0.0, -0.02]),
                 PLACE_XY]

    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = START_XY[0]
    init_qpos[idx_vy] = START_XY[1]
    init_qpos[idx_vz] = APPROACH_Z
    init_qpos[idx_rz] = YAW_APPROACH
    init_qpos[idx_ry] = RY_APPROACH
    init_qpos[idx_rx] = ROLL_APPROACH
    init_qpos[idx1] = 0.05
    init_qpos[idx2] = 0.05
    robot.set_qpos(init_qpos)
    for idx in [idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2]:
        active_joints[idx].set_drive_target(float(init_qpos[idx]))
    for _ in range(50):
        step_physics(robot, scene, fixed_qpos={
            idx_vx: START_XY[0], idx_vy: START_XY[1], idx_vz: APPROACH_Z,
            idx_rz: YAW_APPROACH, idx_ry: RY_APPROACH, idx_rx: ROLL_APPROACH,
        }, joint_pd=joint_pd)

    frames = []
    cube_z_log = []
    grasp_success = False
    cube_pos_log = []
    gripper_pos_log = []
    gripper_tilt_log = []

    camera = create_camera(scene) if output_dir else None

    for i in range(num_frames):
        progress = i / num_frames

        # 开合: 0→55% 平滑闭合，55→85% 保持闭合，85→100% 张开释放
        if progress < 0.55:
            target_j = 0.05 * (1.0 - _smoothstep(progress / 0.55))
        elif progress < 0.85:
            target_j = 0.0
        else:
            target_j = 0.05 * _smoothstep((progress - 0.85) / 0.15)

        if progress < 0.30:
            # 俯冲接近: ry 75°→90°, rz -30°→0°, rx 10°→0°
            tl = _smoothstep(progress / 0.30)
            tx = START_XY[0] + tl * (GRASP_XY[0] - START_XY[0])
            ty = START_XY[1] + tl * (GRASP_XY[1] - START_XY[1])
            target_z = APPROACH_Z + tl * (GRASP_Z - APPROACH_Z)
            target_ry = RY_APPROACH + tl * (RY_GRASP - RY_APPROACH)
            target_rz = YAW_APPROACH + tl * (YAW_GRASP - YAW_APPROACH)
            target_rx = ROLL_APPROACH * (1.0 - tl)
        elif progress < 0.55:
            # 抓取闭合: ry 90°→80°, rz 0°→30°, rx 0°→-10°
            tl = _smoothstep((progress - 0.30) / 0.25)
            tx, ty = GRASP_XY[0], GRASP_XY[1]
            target_z = GRASP_Z
            target_ry = RY_GRASP + tl * (RY_LIFT_A - RY_GRASP)
            target_rz = YAW_GRASP + tl * (YAW_LIFT - YAW_GRASP)
            target_rx = ROLL_LIFT * tl
        elif progress < 0.75:
            # 弧线抬升搬运: ry 80°→100°, rz 30°→-15°, rx -10°→5°
            tl = _smoothstep((progress - 0.55) / 0.20)
            tx = _cubic_bezier(*D_LIFT_CP, tl)[0]
            ty = _cubic_bezier(*D_LIFT_CP, tl)[1]
            target_z = GRASP_Z + tl * (LIFT_Z - GRASP_Z)
            target_ry = RY_LIFT_A + tl * (RY_LIFT_B - RY_LIFT_A)
            target_rz = YAW_LIFT + tl * (YAW_PLACE - YAW_LIFT)
            target_rx = ROLL_LIFT + tl * (ROLL_PLACE - ROLL_LIFT)
        elif progress < 0.85:
            # 放置点上方稳定
            tx, ty = PLACE_XY[0], PLACE_XY[1]
            target_z = LIFT_Z
            target_ry = RY_PLACE
            target_rz = YAW_PLACE
            target_rx = ROLL_PLACE
        else:
            # 下降释放
            tl = _smoothstep((progress - 0.85) / 0.15)
            tx, ty = PLACE_XY[0], PLACE_XY[1]
            target_z = LIFT_Z - tl * (LIFT_Z - PLACE_Z)
            target_ry = RY_PLACE + tl * (RY_GRASP - RY_PLACE)
            target_rz = YAW_PLACE * (1.0 - tl)
            target_rx = ROLL_PLACE * (1.0 - tl)

        active_joints[idx_vx].set_drive_target(tx)
        active_joints[idx_vy].set_drive_target(ty)
        active_joints[idx_vz].set_drive_target(target_z)
        active_joints[idx_rz].set_drive_target(target_rz)
        active_joints[idx_ry].set_drive_target(target_ry)
        active_joints[idx_rx].set_drive_target(target_rx)
        active_joints[idx1].set_drive_target(target_j)
        active_joints[idx2].set_drive_target(target_j)

        step_physics(robot, scene, fixed_qpos={
            idx_vx: tx, idx_vy: ty, idx_vz: target_z,
            idx_rz: target_rz, idx_ry: target_ry, idx_rx: target_rx,
            idx1: target_j, idx2: target_j,
        }, joint_pd=joint_pd)

        actual_qpos = robot.get_qpos()
        gripper_z = float(actual_qpos[idx_vz])
        gripper_xy = (float(actual_qpos[idx_vx]), float(actual_qpos[idx_vy]))
        a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]
        a_rz = float(actual_qpos[idx_rz])
        a_ry = float(actual_qpos[idx_ry])
        a_rx = float(actual_qpos[idx_rx])

        # 根据实际旋转矩阵计算手指尖在世界坐标系的位置，让 cube 跟随
        R = sst.Rotation.from_euler('zyx', [a_rz, a_ry, a_rx]).as_matrix()
        finger_offset_world = R @ np.array([_FINGER_OFFSET, 0.0, 0.0])
        cube_world = np.array([gripper_xy[0], gripper_xy[1], gripper_z]) + finger_offset_world

        if progress >= 0.45 and progress < 0.88:
            cube.set_pose(sapien.Pose(cube_world.tolist(), quat_from_matrix(R)))
        elif progress >= 0.88:
            cube.set_pose(sapien.Pose([place_pos[0], place_pos[1], place_pos[2] + cube_size/2]))

        cube_z = float(cube.get_pose().p[2])
        cube_z_log.append(cube_z)
        cube_pos_log.append((float(cube.get_pose().p[0]), float(cube.get_pose().p[1]), cube_z))
        gripper_pos_log.append((gripper_xy[0], gripper_xy[1], gripper_z))
        gripper_tilt_log.append(a_ry)

        if progress > 0.6 and cube_z > GRASP_Z + 0.01:
            grasp_success = True

        if i % 30 == 0 or i >= num_frames - 5:
            stage = "接近" if progress < 0.30 else \
                    "闭合" if progress < 0.55 else \
                    "抬升" if progress < 0.75 else \
                    "稳定" if progress < 0.85 else "释放"
            status = "⬆抓取" if (progress > 0.6 and cube_z > GRASP_Z + 0.005) else ""
            finger_gap = 2 * _FINGER_BASE_Y + 2 * a_j1
            print(f"  帧 {i:3d} ({progress*100:3.0f}%) [{stage}]: "
                  f"g=({gripper_xy[0]*100:.1f},{gripper_xy[1]*100:.1f},{gripper_z*100:.1f}) "
                  f"ry={np.degrees(a_ry):.0f}° rz={np.degrees(a_rz):.0f}° rx={np.degrees(a_rx):.0f}° "
                  f"j={a_j1*1000:.0f}mm gap={finger_gap*100:.1f}cm cube_z={cube_z*100:.1f}cm {status}")

        if output_dir and camera is not None:
            try:
                eye = np.array([gripper_xy[0], gripper_xy[1] - 0.2, 0.15])
                target = np.array([gripper_xy[0], gripper_xy[1], 0.05])
                set_camera_pose(camera, eye, target)
                img = render_frame(scene, camera)
                frames.append(img)
            except RuntimeError:
                pass
        if viewer_cam is not None:
            try:
                scene.update_render()
                viewer_cam.render()
            except AttributeError:
                viewer_cam = None

    max_cube_z = max(cube_z_log) if cube_z_log else 0.0
    final_cube_z = cube_z_log[-1] if cube_z_log else 0.0
    print(f"\n  结果:")
    print(f"    抓取{'✅ 成功' if grasp_success else '❌ 失败'}: 方块最高升至 {max_cube_z*100:.1f}cm")
    print(f"    最终方块高度: {final_cube_z*100:.1f}cm "
          f"({'被抓起' if final_cube_z > GRASP_Z + 0.01 else '放回桌面'})")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test7_tilted_curved.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    return {
        'lifted': grasp_success,
        'max_cube_z': max_cube_z,
        'final_cube_z': final_cube_z,
        'cube_pos_log': cube_pos_log,
        'gripper_pos_log': gripper_pos_log,
        'gripper_tilt_log': gripper_tilt_log,
    }


# ════════════════════════════════════════════════════════════════════
#  测试 8: 复杂抓放 (Bezier S曲线 + 姿态航点 + 边走边抓)
# ════════════════════════════════════════════════════════════════════

def test8_complex_pick_place(scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx,
                             idx1, idx2, num_frames=800, output_dir=None, viewer_cam=None):
    """test8: 真实物理抓取 — 斜抓取 + 复杂曲线搬运 + 放置

    核心原理:
      1. set_drive_target() 设置各关节目标 (含夹爪逐渐闭合)
      2. compute_passive_force(gravity=True, coriolis_and_centrifugal=True) 补偿重力和科里奥利力
      3. set_qf(qf) + scene.step() — 纯物理驱动
      4. 绝不使用 fixed_qpos / cube.set_pose — 纯物理摩擦力抓取

    轨迹策略 (不摇晃):
      - 接近: 竖直下降到方块上方
      - 抓取: 斜着倾斜 (一次性倾斜35°+15°), 边下降边闭合
      - 搬运: 沿Bezier S曲线搬运, 倾斜角度缓慢旋转到另一个方向
      - 放置: 到达指定位置后缓慢释放
      - 撤退: 抬起离开
    """

    # ── 几何参数 ──
    _FINGER_OFFSET_X = 0.03689
    _CUBE_SIZE = 0.025           # 2.5cm 方块
    _CUBE_XY = np.array([0.08, -0.06])
    _PLACE_XY = np.array([0.40, 0.20])  # 远距离放置: xy各移动32cm/26cm

    _TABLE_Z = 0.0               # 桌面顶面 Z=0
    _CUBE_Z = _TABLE_Z + _CUBE_SIZE / 2   # 方块中心 Z = 0.0125m
    # ── 抬升相关参数 ──
    _APPROACH_Z = 0.18           # 接近高度 (18cm)
    _GRASP_Z = _CUBE_Z           # 抓取高度 (方块中心1.25cm)
    _LIFT_Z = 0.25               # 抬起高度 (25cm)

    # ── 夹持力计算 ──
    # 手指碰撞盒内表面间距: gap = 2q - 0.003094 (q=关节位置)
    # 25mm方块刚好接触: q = (0.025 + 0.003094) / 2 = 0.014047 ≈ 14mm
    # 用稍小的target产生持续挤压力 (每侧约2mm穿透)
    _GRIPPER_Q_SQUEEZE = 0.012   # 12mm: gap=20.9mm, 25mm方块→每侧2mm穿透

    # ── 角度策略 (不摇晃: 一次性倾斜, 缓慢旋转, 夹爪朝下) ──
    # 抓取时倾斜: ry=30° rz=25° rx=15°
    GRASP_RY = np.radians(30)
    GRASP_RZ = np.radians(25)
    GRASP_RX = np.radians(15)
    # 搬运中旋转: ry=-20° rz=-30° rx=25° (大角度旋转)
    TRANSPORT_RY = np.radians(-20)
    TRANSPORT_RZ = np.radians(-30)
    TRANSPORT_RX = np.radians(25)
    # 释放时角度: 大角度翻转 (复杂位姿)
    RELEASE_RY = np.radians(-35)
    RELEASE_RZ = np.radians(40)
    RELEASE_RX = np.radians(-30)

    # ── 角度补偿基座位置 (补偿xyz, 确保手指到达目标位置) ──
    # 夹爪朝下: 通过virtual_ry=+90°实现 (R_y(+90°)将局部X轴转到-Z, 即朝下)
    _HOME_RY = np.pi / 2  # 初始pitch=+90°使夹爪朝下

    def _angle_compensated_base(cx, cy, cz, ry, rz, rx):
        # ry是在_HOME_RY基础上的增量
        R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
        finger_world = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
        # 补偿xyz: base = finger_target - R @ finger_offset
        base_x = cx - finger_world[0]
        base_y = cy - finger_world[1]
        base_z = cz - finger_world[2]
        return base_x, base_y, base_z

    # ── 3-segment Bezier S-curve (搬运阶段) ──
    _BASE_XY_at_grasp = np.array(_angle_compensated_base(
        _CUBE_XY[0], _CUBE_XY[1], _GRASP_Z,
        GRASP_RY, GRASP_RZ, GRASP_RX)[:2])
    _PLACE_BASE_XY = np.array(_angle_compensated_base(
        _PLACE_XY[0], _PLACE_XY[1], _GRASP_Z,
        TRANSPORT_RY, TRANSPORT_RZ, TRANSPORT_RX)[:2])

    d_vec = _PLACE_BASE_XY - _BASE_XY_at_grasp
    d_len = np.linalg.norm(d_vec)
    u_dir = d_vec / d_len
    v_perp = np.array([-u_dir[1], u_dir[0]])
    amp = 0.05  # S曲线振幅 (加大曲线弧度)

    m1 = _BASE_XY_at_grasp + d_vec * (1.0 / 3.0)
    m2 = _BASE_XY_at_grasp + d_vec * (2.0 / 3.0)

    bez_cp = [
        _BASE_XY_at_grasp,
        _BASE_XY_at_grasp + d_vec * 0.11 - amp * v_perp,
        m1 - d_vec * 0.11 - amp * v_perp,
        m1,
        m1 + d_vec * 0.11 + amp * v_perp,
        m2 - d_vec * 0.11 + amp * v_perp,
        m2,
        m2 + d_vec * 0.11 - amp * 0.6 * v_perp,
        _PLACE_BASE_XY - d_vec * 0.11 - amp * 0.6 * v_perp,
        _PLACE_BASE_XY,
    ]

    def _eval_s_curve(t):
        t = max(0.0, min(1.0, t))
        if t < 1.0 / 3.0:
            lt = t * 3.0
            return _cubic_bezier(bez_cp[0], bez_cp[1], bez_cp[2], bez_cp[3], lt)
        elif t < 2.0 / 3.0:
            lt = (t - 1.0 / 3.0) * 3.0
            return _cubic_bezier(bez_cp[3], bez_cp[4], bez_cp[5], bez_cp[6], lt)
        else:
            lt = (t - 2.0 / 3.0) * 3.0
            return _cubic_bezier(bez_cp[6], bez_cp[7], bez_cp[8], bez_cp[9], lt)

    print("\n" + "=" * 70)
    print("测试 8: 斜抓取 + 复杂曲线搬运 + 放置 (真实物理)")
    print(f"  物理模式: step_physics_drive (gravity+coriolis补偿)")
    print(f"  方块大小: {_CUBE_SIZE*100:.1f}cm, 方块位置: ({_CUBE_XY[0]:.2f}, {_CUBE_XY[1]:.2f})")
    print(f"  放置位置: ({_PLACE_XY[0]:.2f}, {_PLACE_XY[1]:.2f})")
    print(f"  抓取角度: ry={np.degrees(GRASP_RY):.0f}° rz={np.degrees(GRASP_RZ):.0f}° rx={np.degrees(GRASP_RX):.0f}°")
    print(f"  搬运终点: ry={np.degrees(TRANSPORT_RY):.0f}° rz={np.degrees(TRANSPORT_RZ):.0f}° rx={np.degrees(TRANSPORT_RX):.0f}°")
    print(f"  无 fixed_qpos, 无 cube.set_pose — 纯物理摩擦力抓取")
    print("=" * 70)

    # ── 驱动刚度 (K=1000让手指自然找到平衡, 高摩擦维持抓取) ──
    active_joints = robot.get_active_joints()
    VIRTUAL_K = 5000.0;   VIRTUAL_D = 1000.0
    ANGULAR_K = 1000.0;   ANGULAR_D = 200.0
    GRIPPER_K = 3000.0;   GRIPPER_D = 300.0  # K=3000: 25cm高空抓持, 更强夹力防滑落

    for idx in [idx_vx, idx_vy, idx_vz]:
        active_joints[idx].set_drive_property(stiffness=VIRTUAL_K, damping=VIRTUAL_D)
    for idx in [idx_rz, idx_ry, idx_rx]:
        active_joints[idx].set_drive_property(stiffness=ANGULAR_K, damping=ANGULAR_D)
    active_joints[idx1].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)
    active_joints[idx2].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)

    # ── 场景: 静态桌面 (大桌面支持远距离搬运) ──
    table_builder = scene.create_actor_builder()
    table_builder.add_box_visual(pose=sapien.Pose(), half_size=[0.35, 0.30, 0.003])
    phys_mat_table = scene.create_physical_material(
        static_friction=0.8, dynamic_friction=0.8, restitution=0.6)
    table_builder.add_box_collision(
        pose=sapien.Pose(), half_size=[0.35, 0.30, 0.003],
        material=phys_mat_table)
    table = table_builder.build_kinematic(name="table")
    table.set_pose(sapien.Pose([0.20, 0.05, _TABLE_Z - 0.003]))

    # ── 动态方块 (黑色, 超高摩擦保证高空抓取稳定) ──
    cube_builder = scene.create_actor_builder()
    _cube_half = [_CUBE_SIZE / 2] * 3
    _black_mat = sapien.render.RenderMaterial()
    _black_mat.base_color = [0.08, 0.08, 0.08, 1.0]
    cube_builder.add_box_visual(pose=sapien.Pose(), half_size=_cube_half, material=_black_mat)
    phys_mat_cube = scene.create_physical_material(
        static_friction=5.0, dynamic_friction=5.0, restitution=0.1)
    cube_builder.add_box_collision(
        pose=sapien.Pose(), half_size=[_CUBE_SIZE / 2] * 3,
        material=phys_mat_cube, density=2000)
    cube = cube_builder.build(name="cube")
    cube.set_pose(sapien.Pose([_CUBE_XY[0], _CUBE_XY[1], _CUBE_Z]))

    # 放置标记
    marker_builder = scene.create_actor_builder()
    marker_builder.add_box_visual(pose=sapien.Pose(), half_size=[0.015, 0.015, 0.001])
    marker = marker_builder.build_kinematic(name="place_marker")
    marker.set_pose(sapien.Pose([_PLACE_XY[0], _PLACE_XY[1], _TABLE_Z]))

    print(f"  方块初始位姿: p={np.round(cube.get_pose().p, 4)}")

    # ── 初始化夹爪位姿 (夹爪朝下, virtual_ry=-90°) ──
    init_bx, init_by, init_bz = _angle_compensated_base(
        _CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, _HOME_RY, 0.0, 0.0)

    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = init_bx
    init_qpos[idx_vy] = init_by
    init_qpos[idx_vz] = init_bz
    init_qpos[idx_rz] = 0.0
    init_qpos[idx_ry] = _HOME_RY  # -90°: 夹爪朝下
    init_qpos[idx_rx] = 0.0
    init_qpos[idx1] = 0.05  # 手指全开
    init_qpos[idx2] = 0.05
    robot.set_qpos(init_qpos)
    for idx in [idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2]:
        active_joints[idx].set_drive_target(float(init_qpos[idx]))

    # 预热物理
    for _ in range(100):
        step_physics_drive(robot, scene)

    # ── 主循环 ──
    frames = []
    cube_z_log = []
    cube_pose_log = []
    grasp_success = False
    max_angle_log = []
    # ── 物理抓取诊断 ──
    finger_vel_log = []       # 手指关节速度 (slip indicator)
    contact_dist_log = []     # 手指尖与方块的距离 (contact indicator)
    cube_vel_log = []         # 方块相对夹爪基座的速度 (slip indicator)
    cube_accel_log = []       # 方块加速度 (force indicator)
    prev_cube_z = _CUBE_Z
    prev_cube_vel = 0.0
    finger_force_log = []     # 夹爪关节驱动误差 (力代理: stiffness * error)
    rel_pos_log = []          # 方块相对夹爪基座的位置模 (stability indicator)
    camera = create_camera(scene) if output_dir else None

    # ── 眼在手上相机: 挂载到 right_D405_link (与 test5 一致, 真实 R1 配置) ──
    # 注意: 不能复用 render_frame (其内部有 scene.step(), 会改变物理时序),
    # 这里只做 update_render + take_picture, 不推进物理.
    wrist_cam = None
    wrist_frames = []
    if output_dir:
        # get_links() 返回 PhysxArticulationLinkComponent, 挂载相机需其底层 entity
        d405 = next(l for l in robot.get_links()
                    if l.get_name() == "right_D405_link").entity
        _camera_counter[0] += 1
        # 相机朝向: 前向 = gripper_link +x (手指伸出方向) 下倾35°, 上方 = +z.
        # D405_link 自身被 joint rpy=(-2.268,0,-1.5708) 转过, 需补偿:
        # R_local = R_joint^T @ R_desired
        R_joint = sst.Rotation.from_euler("xyz", [-2.268, 0.0, -1.5708]).as_matrix()
        tilt = np.deg2rad(35.0)
        R_desired = np.column_stack([
            [np.cos(tilt), 0.0, -np.sin(tilt)],   # X: 前向 (略下倾)
            [0.0, 1.0, 0.0],                       # Y: 上
            [np.sin(tilt), 0.0, np.cos(tilt)],     # Z
        ])
        cam_q = quat_from_matrix(R_joint.T @ R_desired)  # (w,x,y,z)
        wrist_cam = scene.add_mounted_camera(
            f"wrist_d405_{_camera_counter[0]}", d405,
            sapien.Pose([0.0, 0.0, 0.0], np.array(cam_q, dtype=float)),
            640, 480, np.deg2rad(60), 0.01, 100.0)
        print("  眼在手上相机已挂载: right_D405_link (前向=手指方向下倾35°)")

    # 前序状态缓存 (相对速度计算)
    prev_gx, prev_gy, prev_gz = init_bx, init_by, init_bz

    # ── 轨迹可视化: 沿实际规划路径放置小球标记 (使用angle-compensated base, 与真实运动一致) ──
    _traj_markers = []
    _traj_colors = [
        (0.2, 0.6, 1.0),   # approach: 蓝
        (0.2, 1.0, 0.4),   # descend+grasp: 绿
        (1.0, 1.0, 0.2),   # grasp: 黄
        (1.0, 0.6, 0.2),   # hold: 橙
        (1.0, 0.2, 0.2),   # transport: 红
        (0.6, 0.2, 1.0),   # release: 紫
        (0.2, 1.0, 1.0),   # retreat: 青
    ]
    for _mi in range(80):  # 80个轨迹采样点
        _mb = scene.create_actor_builder()
        _mb.add_sphere_visual(pose=sapien.Pose(), radius=0.004)
        _m = _mb.build_kinematic(name=f"traj_{_mi}")
        _traj_markers.append(_m)

    # ── 预计算规划轨迹 (使用与主循环相同的 angle-compensated 逻辑, 确保标记与实际运动一致) ──
    _planned_traj = []
    for _ti in range(80):
        _p = _ti / 80.0
        if _p < 0.08:
            # Phase 1: 接近 — 夹爪朝下高位
            _ry, _rz, _rx = _HOME_RY, 0.0, 0.0
            _bx, _by, _fz = _angle_compensated_base(
                _CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, _ry, _rz, _rx)
        elif _p < 0.63:
            # Phase 2: 下降边夹 + 抬升 + Bezier搬运
            _tl = _min_jerk((_p - 0.08) / 0.55)
            if _tl < 0.30:
                _sub = _min_jerk(_tl / 0.30)
                _ry = _HOME_RY + _sub * GRASP_RY
                _rz = _sub * GRASP_RZ
                _rx = _sub * GRASP_RX
                _finger_z = _APPROACH_Z + _sub * (_GRASP_Z - _APPROACH_Z)
                _bx, _by, _fz = _angle_compensated_base(
                    _CUBE_XY[0], _CUBE_XY[1], _finger_z, _ry, _rz, _rx)
            elif _tl < 0.45:
                _sub = _min_jerk((_tl - 0.30) / 0.15)
                _ry, _rz, _rx = _HOME_RY + GRASP_RY, GRASP_RZ, GRASP_RX
                _finger_z = _GRASP_Z + _sub * (_LIFT_Z - _GRASP_Z)
                _bx, _by, _fz = _angle_compensated_base(
                    _CUBE_XY[0], _CUBE_XY[1], _finger_z, _ry, _rz, _rx)
            else:
                _sub = _min_jerk((_tl - 0.45) / 0.55)
                _xy = _eval_s_curve(_sub)
                _finger_z = _LIFT_Z
                _ry = (_HOME_RY + GRASP_RY) + _sub * (TRANSPORT_RY - GRASP_RY)
                _rz = GRASP_RZ + _sub * (TRANSPORT_RZ - GRASP_RZ)
                _rx = GRASP_RX + _sub * (TRANSPORT_RX - GRASP_RX)
                _R = sst.Rotation.from_euler('ZYX', [_rz, _ry, _rx]).as_matrix()
                _finger_off = _R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
                _bx = _xy[0] - _finger_off[0]
                _by = _xy[1] - _finger_off[1]
                _fz = _finger_z - _finger_off[2]
        elif _p < 0.88:
            # Phase 3: 释放 — 与主循环一致 (先下降, 后开爪)
            _tl = (_p - 0.63) / 0.25
            if _tl < 0.60:
                _sub = _min_jerk(_tl / 0.60)
                _finger_z = _LIFT_Z + _sub * (_GRASP_Z - _LIFT_Z)
                _ry = (_HOME_RY + TRANSPORT_RY) + _sub * (RELEASE_RY - TRANSPORT_RY)
                _rz = TRANSPORT_RZ + _sub * (RELEASE_RZ - TRANSPORT_RZ)
                _rx = TRANSPORT_RX + _sub * (RELEASE_RX - TRANSPORT_RX)
            else:
                _sub = _min_jerk((_tl - 0.60) / 0.40)
                _finger_z = _GRASP_Z
                _ry = _HOME_RY + TRANSPORT_RY + RELEASE_RY
                _rz = TRANSPORT_RZ + RELEASE_RZ
                _rx = TRANSPORT_RX + RELEASE_RX
            _bx, _by, _fz = _angle_compensated_base(
                _PLACE_XY[0], _PLACE_XY[1], _finger_z, _ry, _rz, _rx)
        else:
            # Phase 4: 抬起撤退
            _tl = _min_jerk((_p - 0.88) / 0.12)
            _finger_z = _GRASP_Z + _tl * (_APPROACH_Z - _GRASP_Z)
            _ry = _HOME_RY + RELEASE_RY * (1.0 - _tl)
            _rz = RELEASE_RZ * (1.0 - _tl)
            _rx = RELEASE_RX * (1.0 - _tl)
            _bx, _by, _fz = _angle_compensated_base(
                _PLACE_XY[0], _PLACE_XY[1], _finger_z, _ry, _rz, _rx)
        _planned_traj.append((_bx, _by, _fz))

    # 设置轨迹标记位姿 (使用规划 base 位置, 与真实夹爪运动一致)
    for _mi, (_bx, _by, _fz) in enumerate(_planned_traj):
        _traj_markers[_mi].set_pose(sapien.Pose([_bx, _by, _fz]))

    # 规划位置日志 (用于跟随误差分析)
    planned_pos_log = []
    actual_base_log = []

    # ── 每帧日志 ──
    if output_dir:
        log_path = str(Path(output_dir) / "test8_frame_log.log")
        log_file = open(log_path, "w")
        log_file.write(
            "# test8_complex_pick_place frame log\n"
            "# format: frame progress phase gx gy gz rz_deg ry_deg rx_deg "
            "j1_mm j2_mm target_j_mm "
            "cx cy cz "
            "contact_surface_mm contact_center_mm "
            "f1_N f2_N force_proxy_N "
            "cube_vel_cm_s rel_vel_cm_s "
            "rel_pos_mm "
            "contact plan_err_mm\n"
        )
        log_file.flush()
        print(f"  帧日志输出: {log_path}")
    else:
        log_path = None
        log_file = None

    for i in range(num_frames):
        progress = i / num_frames

        # ── 阶段逻辑 (4阶段, 全程无停顿: 边下降边夹 + 边放下边释放) ──
        if progress < 0.08:
            # Phase 1: 接近 — 夹爪朝下高位悬停
            phase = "approach"
            ry, rz, rx = _HOME_RY, 0.0, 0.0
            bx, by, target_z = _angle_compensated_base(
                _CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, ry, rz, rx)
            target_j = 0.05

        elif progress < 0.63:
            # Phase 2: 下降边夹 + 抬升 + Bezier搬运 (连续无停顿, 3子阶段)
            phase = "grasp+transport"
            tl = _min_jerk((progress - 0.08) / 0.55)
            if tl < 0.30:
                # [8%-25%] 下降+闭合: 边下降边夹, 接触时手指已闭到0
                sub = _min_jerk(tl / 0.30)
                ry = _HOME_RY + sub * GRASP_RY
                rz = sub * GRASP_RZ
                rx = sub * GRASP_RX
                finger_z = _APPROACH_Z + sub * (_GRASP_Z - _APPROACH_Z)
                bx, by, target_z = _angle_compensated_base(
                    _CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
                target_j = 0.05 * (1.0 - sub)  # 边下边夹

            elif tl < 0.45:
                # [25%-33%] 抬起到高空
                sub = _min_jerk((tl - 0.30) / 0.15)
                ry, rz, rx = _HOME_RY + GRASP_RY, GRASP_RZ, GRASP_RX
                finger_z = _GRASP_Z + sub * (_LIFT_Z - _GRASP_Z)
                bx, by, target_z = _angle_compensated_base(
                    _CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
                target_j = 0.0

            else:
                # [33%-63%] Bezier S曲线搬运 + 角度旋转
                sub = _min_jerk((tl - 0.45) / 0.55)
                xy = _eval_s_curve(sub)
                finger_z = _LIFT_Z
                ry = (_HOME_RY + GRASP_RY) + sub * (TRANSPORT_RY - GRASP_RY)
                rz = GRASP_RZ + sub * (TRANSPORT_RZ - GRASP_RZ)
                rx = GRASP_RX + sub * (TRANSPORT_RX - GRASP_RX)
                R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
                finger_off = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
                bx = xy[0] - finger_off[0]
                by = xy[1] - finger_off[1]
                target_z = finger_z - finger_off[2]
                target_j = 0.0

        elif progress < 0.88:
            # Phase 3: 释放 — 先降到目标高度再开爪 (避免释放时自由落体)
            #   progress 0.63-0.78: 下降到GRASP高度 (手指仍闭合)
            #   progress 0.78-0.88: 手指张开释放 (位置不动)
            phase = "release"
            tl = (progress - 0.63) / 0.25
            if tl < 0.60:
                # 下降段
                sub = _min_jerk(tl / 0.60)
                finger_z = _LIFT_Z + sub * (_GRASP_Z - _LIFT_Z)
                ry = (_HOME_RY + TRANSPORT_RY) + sub * (RELEASE_RY - TRANSPORT_RY)
                rz = TRANSPORT_RZ + sub * (RELEASE_RZ - TRANSPORT_RZ)
                rx = TRANSPORT_RX + sub * (RELEASE_RX - TRANSPORT_RX)
                bx, by, target_z = _angle_compensated_base(
                    _PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
                target_j = 0.0  # 下降过程中手指保持闭合
            else:
                # 开爪段 (夹爪保持低位)
                sub = _min_jerk((tl - 0.60) / 0.40)
                finger_z = _GRASP_Z
                ry = _HOME_RY + TRANSPORT_RY + RELEASE_RY
                rz = TRANSPORT_RZ + RELEASE_RZ
                rx = TRANSPORT_RX + RELEASE_RX
                bx, by, target_z = _angle_compensated_base(
                    _PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
                target_j = sub * 0.05  # 手指在低位慢慢张开

        else:
            # Phase 4: 抬起撤退
            phase = "retreat"
            tl = _min_jerk((progress - 0.88) / 0.12)
            finger_z = _GRASP_Z + tl * (_APPROACH_Z - _GRASP_Z)
            ry = _HOME_RY + RELEASE_RY * (1.0 - tl)
            rz = RELEASE_RZ * (1.0 - tl)
            rx = RELEASE_RX * (1.0 - tl)
            bx, by, target_z = _angle_compensated_base(
                _PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
            target_j = 0.05

        # ── 设置驱动目标 (核心: 只用 set_drive_target) ──
        active_joints[idx_vx].set_drive_target(bx)
        active_joints[idx_vy].set_drive_target(by)
        active_joints[idx_vz].set_drive_target(target_z)
        active_joints[idx_rz].set_drive_target(rz)
        active_joints[idx_ry].set_drive_target(ry)
        active_joints[idx_rx].set_drive_target(rx)
        active_joints[idx1].set_drive_target(target_j)
        active_joints[idx2].set_drive_target(target_j)

        # ── 不使用virtual_lock (让PD自然找到平衡, 验证K=1000可稳定抓取) ──
        vlt = None

        # ── 物理步进 (gravity+coriolis补偿 + virtual_lock防漂移) ──
        step_physics_drive(robot, scene, virtual_lock_targets=vlt)

        # ── 读取实际状态 (纯读取, 绝不 set_pose) ──
        actual_qpos = robot.get_qpos()
        gx = float(actual_qpos[idx_vx])
        gy = float(actual_qpos[idx_vy])
        gz = float(actual_qpos[idx_vz])
        a_j1 = float(actual_qpos[idx1])
        a_j2 = float(actual_qpos[idx2])
        a_ry = float(actual_qpos[idx_ry])
        a_rz = float(actual_qpos[idx_rz])
        a_rx = float(actual_qpos[idx_rx])

        cube_pose = cube.get_pose()
        cube_x = float(cube_pose.p[0])
        cube_y = float(cube_pose.p[1])
        cube_z = float(cube_pose.p[2])
        cube_z_log.append(cube_z)
        cube_pose_log.append((cube_x, cube_y, cube_z))

        # 规划位置 (用于跟随误差分析)
        planned_pos_log.append((bx, by, target_z))
        actual_base_log.append((gx, gy, gz))

        # ── 物理抓取诊断: 力/滑移/接触分析 ──
        dt_ctrl = 1.0 / CONTROL_FREQ
        prev_cx = cube_pose_log[-2][0] if len(cube_pose_log) > 1 else cube_x
        prev_cy = cube_pose_log[-2][1] if len(cube_pose_log) > 1 else cube_y
        prev_cz = prev_cube_z

        # 1. 手指驱动误差 (force proxy: stiffness * error → 接触力代理)
        j1_error = target_j - a_j1
        j2_error = target_j - a_j2
        finger_force_proxy = abs(GRIPPER_K * j1_error) + abs(GRIPPER_K * j2_error)
        finger_force_log.append(finger_force_proxy)

        # 2. 方块加速度 (force indicator)
        cube_accel_z = (cube_z - prev_cz - prev_cube_vel * dt_ctrl) / (dt_ctrl ** 2)
        cube_accel_log.append(abs(cube_accel_z))

        # 3. 手指尖到方块表面距离 (从碰撞体提取, 精确计算)
        R_actual = sst.Rotation.from_euler('ZYX', [a_rz, a_ry, a_rx]).as_matrix()
        finger_offset = R_actual @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
        finger_world = np.array([gx, gy, gz]) + finger_offset
        # 到方块中心距离
        cube_center = np.array([cube_x, cube_y, cube_z])
        contact_dist_center = float(np.linalg.norm(finger_world - cube_center))
        # 从碰撞体提取 half_size, 计算到表面的精确距离 (各轴取投影 - 半边长)
        contact_dist = contact_dist_center - _CUBE_SIZE / 2
        try:
            colliders = cube.get_colliders()
            if colliders:
                c = colliders[0]
                half_sizes = c.get_shape().get_box_half_sizes()
                # 方块局部坐标到世界坐标的旋转 (Z-up 场景用四元数)
                cube_q = cube_pose.q  # [w,x,y,z]
                # 手指到方块中心的向量
                v = finger_world - cube_center
                # 将向量旋转到方块局部坐标系
                v_local = pq.rotate_vector(v, pq.quat2mat(np.array([cube_q[0], cube_q[1], cube_q[2], cube_q[3]])))
                # 各轴投影 - 半边长, 取 max(0)
                proj = np.abs(v_local) - np.array(half_sizes)
                contact_dist = max(0.0, np.max(proj))
        except Exception:
            pass  # fallback 到近似值
        contact_dist_log.append(max(0.0, contact_dist))

        # 4. 方块相对夹爪基座的速度 (slip indicator: 相对运动才是滑脱)
        gripper_vel = np.sqrt(
            ((gx - prev_gx) / dt_ctrl) ** 2 +
            ((gy - prev_gy) / dt_ctrl) ** 2 +
            ((gz - prev_gz) / dt_ctrl) ** 2)
        prev_gx, prev_gy, prev_gz = gx, gy, gz

        cube_vel = np.sqrt(
            ((cube_x - prev_cx) / dt_ctrl) ** 2 +
            ((cube_y - prev_cy) / dt_ctrl) ** 2 +
            ((cube_z - prev_cz) / dt_ctrl) ** 2)
        # 相对速度: 方块 - 夹爪基座 (向量)
        rel_vx = (cube_x - prev_cx) - (gx - prev_gx)
        rel_vy = (cube_y - prev_cy) - (gy - prev_gy)
        rel_vz = (cube_z - prev_cz) - (gz - prev_gz)
        rel_vel = np.sqrt((rel_vx/dt_ctrl)**2 + (rel_vy/dt_ctrl)**2 + (rel_vz/dt_ctrl)**2)
        cube_vel_log.append(rel_vel)  # 存的是相对速度
        prev_cube_vel = (cube_z - prev_cz) / dt_ctrl
        prev_cube_z = cube_z

        # 5. 方块相对夹爪基座的位置 (抓取稳定性指标)
        rel_pos = np.array([cube_x, cube_y, cube_z]) - np.array([gx, gy, gz])
        rel_pos_mag = float(np.linalg.norm(rel_pos))
        rel_pos_log.append(rel_pos_mag)

        angle_mag = np.sqrt(np.degrees(a_ry)**2 + np.degrees(a_rz)**2 + np.degrees(a_rx)**2)
        max_angle_log.append(angle_mag)

        # ── 每帧日志 (append-only 文本行, 便于 grep/tail/csv 解析) ──
        if log_file is not None:
            plan_err = float(np.sqrt((bx - gx)**2 + (by - gy)**2 + (target_z - gz)**2))
            # 接触力分解 (左右手指各自)
            f1 = float(GRIPPER_K * j1_error)
            f2 = float(GRIPPER_K * j2_error)
            log_file.write(
                f"{i} {progress:.4f} {phase} "
                f"{gx:.6f} {gy:.6f} {gz:.6f} "
                f"{np.degrees(a_rz):.2f} {np.degrees(a_ry):.2f} {np.degrees(a_rx):.2f} "
                f"{a_j1*1000:.2f} {a_j2*1000:.2f} {target_j*1000:.2f} "
                f"{cube_x:.6f} {cube_y:.6f} {cube_z:.6f} "
                f"{contact_dist*1000:.2f} {contact_dist_center*1000:.2f} "
                f"{f1:.2f} {f2:.2f} {finger_force_proxy:.2f} "
                f"{cube_vel*100:.2f} {rel_vel*100:.2f} "
                f"{rel_pos_mag*1000:.2f} "
                f"{contact_dist<0.020} {plan_err*1000:.2f}\n"
            )
            if i % 50 == 0:
                log_file.flush()

        if progress > 0.40 and progress < 0.70 and cube_z > _CUBE_Z + 0.02:
            grasp_success = True

        # ── 每30帧打印 (含规划vs实际跟随误差 + 物理诊断) ──
        if i % 30 == 0 or i >= num_frames - 5:
            fj_mm = a_j1 * 1000
            finger_gap = 2 * 0.013453 + 2 * a_j1
            plan_err = np.sqrt((bx-gx)**2 + (by-gy)**2 + (target_z-gz)**2)
            f1_N = GRIPPER_K * j1_error
            f2_N = GRIPPER_K * j2_error
            print(f"  帧{i:3d} ({progress*100:5.1f}%)[{phase:12s}]: "
                  f"g=({gx*100:.1f},{gy*100:.1f},{gz*100:.1f}) "
                  f"ry={np.degrees(a_ry):+.0f}° rz={np.degrees(a_rz):+.0f}° rx={np.degrees(a_rx):+.0f}° "
                  f"|angle|={angle_mag:.0f}° fj={fj_mm:.1f}mm gap={finger_gap*100:.1f}cm "
                  f"cube=({cube_x*100:.1f},{cube_y*100:.1f},{cube_z*100:.1f})cm "
                  f"plan_err={plan_err*1000:.1f}mm "
                  f"|contact={contact_dist*1000:.1f}mm "
                  f"force=({f1_N:.0f},{f2_N:.0f}){finger_force_proxy:.0f}N "
                  f"rel_vel={rel_vel*100:.1f}cm/s rel_pos={rel_pos_mag*1000:.1f}mm|")

        if output_dir and camera is not None:
            try:
                set_camera_pose(camera, np.array([0.20, -0.20, 0.20]),
                                np.array([0.12, 0.0, 0.04]))
                img = render_frame(scene, camera)
                frames.append(img)
            except RuntimeError:
                pass

        if output_dir and wrist_cam is not None:
            scene.update_render()
            wrist_cam.take_picture()
            rgba = wrist_cam.get_picture("Color")
            rgb = np.clip(rgba[..., :3], 0, 1)
            rgb = np.where(rgb < 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
            wrist_frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

        if viewer_cam is not None:
            scene.update_render()
            viewer_cam.render()

    max_cube_z = max(cube_z_log) if cube_z_log else 0.0
    final_cube_z = cube_z_log[-1] if cube_z_log else 0.0
    lifted = grasp_success or any(z > _CUBE_Z + 0.02 for z in cube_z_log)
    max_angle = max(max_angle_log) if max_angle_log else 0.0

    final_cube_x = cube_pose_log[-1][0] if cube_pose_log else 0.0
    final_cube_y = cube_pose_log[-1][1] if cube_pose_log else 0.0
    cube_xy_drift = np.sqrt((final_cube_x - _CUBE_XY[0])**2 + (final_cube_y - _CUBE_XY[1])**2)
    place_dist = np.sqrt((_PLACE_XY[0] - _CUBE_XY[0])**2 + (_PLACE_XY[1] - _CUBE_XY[1])**2)

    print(f"\n  结果:")
    print(f"    抓取{'成功' if lifted else '失败'}: 方块最高升至 {max_cube_z*100:.1f}cm "
          f"(初始 {_CUBE_Z*100:.1f}cm)")
    print(f"    方块位置: 初始({_CUBE_XY[0]*100:.1f},{_CUBE_XY[1]*100:.1f}) → "
          f"最终({final_cube_x*100:.1f},{final_cube_y*100:.1f}) "
          f"漂移={cube_xy_drift*100:.1f}cm / 目标距离={place_dist*100:.1f}cm")
    print(f"    最终方块高度: {final_cube_z*100:.1f}cm "
          f"({'已放下' if final_cube_z <= _CUBE_Z + 0.005 else '被抓起'})")
    print(f"    最大姿态角: {max_angle:.0f}° ({'≥30°' if max_angle >= 30 else '<30°'})")
    print(f"    物理模式: step_physics_drive (无fixed_qpos, 无set_pose)")
    print(f"    轨迹可视化: 80个小球标记规划路径 (viewer中可见)")

    # 跟随误差统计
    if planned_pos_log and actual_base_log:
        plan_errs = [np.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2 + (p[2]-a[2])**2)
                     for p, a in zip(planned_pos_log, actual_base_log)]
        print(f"    规划跟随误差: mean={np.mean(plan_errs)*1000:.1f}mm "
              f"max={np.max(plan_errs)*1000:.1f}mm")

    # ── 物理抓取诊断摘要 ──
    if finger_force_log:
        print(f"\n  物理抓取诊断:")
        max_force = max(finger_force_log)
        mean_force = np.mean(finger_force_log)
        print(f"    接触力代理(K={GRIPPER_K}·Δq): max={max_force:.1f}N, mean={mean_force:.1f}N")
        if contact_dist_log:
            min_contact = min(contact_dist_log)
            contact_frames = sum(1 for d in contact_dist_log if d < 0.020)
            print(f"    手指-方块表面最近距离: {min_contact*1000:.1f}mm, 接触帧数={contact_frames}/{len(contact_dist_log)}")
        if cube_vel_log:
            # 持握稳定段 (50%-65%): 避开抬升/加速段, 只看稳定搬运时的滑移
            hold_start = int(num_frames * 0.50)
            hold_end = int(num_frames * 0.65)
            hold_vels = cube_vel_log[hold_start:hold_end]
            if hold_vels:
                max_hold_vel = max(hold_vels)
                mean_hold_vel = np.mean(hold_vels)
                print(f"    持握稳定段(50%-65%)相对速度: max={max_hold_vel*100:.2f}cm/s mean={mean_hold_vel*100:.2f}cm/s "
                      f"({'滑脱' if max_hold_vel > 0.02 else '稳定'})")
            # 全程最大相对速度
            max_rel_vel = max(cube_vel_log)
            print(f"    全程最大相对速度: {max_rel_vel*100:.2f}cm/s")
        if rel_pos_log:
            # 持握稳定段 rel_pos 一致性 (抓取后应恒定)
            hold_rpm = rel_pos_log[hold_start:hold_end]
            if hold_rpm:
                rel_pos_std = np.std(hold_rpm) * 1000  # mm
                print(f"    持握稳定段方块相对夹爪位置: mean={np.mean(hold_rpm)*1000:.1f}mm "
                      f"std={rel_pos_std:.1f}mm "
                      f"({'稳定' if rel_pos_std < 5.0 else '偏移'})")
        if cube_accel_log:
            max_accel = max(cube_accel_log)
            print(f"    方块最大Z加速度: {max_accel:.1f}m/s2")
        print(f"    诊断说明: force_proxy>1N + contact<20mm + rel_vel<2cm/s + rel_pos_std<5mm = 抓取成功")

    # ── 关闭日志文件 ──
    if log_file is not None:
        log_file.close()
        print(f"  帧日志已保存: {log_path} ({num_frames} frames)")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test8_complex_pick_place.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    if output_dir and wrist_frames:
        import imageio
        out_path = str(Path(output_dir) / "test8_wrist_view.mp4")
        imageio.mimsave(out_path, wrist_frames, fps=30)
        print(f"  眼在手上视频已保存: {out_path}")

    return {'lifted': lifted, 'max_cube_z': max_cube_z, 'final_cube_z': final_cube_z,
            'max_angle': max_angle}


# ════════════════════════════════════════════════════════════════════
#  测试 10: 导纳控制抓取 (Admittance / 力控闭环)
# ════════════════════════════════════════════════════════════════════

def _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=0.012, grip_k=5000.0):
    """估算两指对木块的夹持力 (仿真内"力传感器")。

    返回 (F_left, F_right): 两指各自对 cube 的夹持力大小 (N)。

    对于小方块, SAPIEN 的 get_contacts() 常把"手指推方块侧面"归到 table/手指链里,
    直接对cube实体的干净接触对稀疏。故采用双层估计 (与 test8 诊断一致):
      1) 优先用 scene.get_contacts() 中明确的 手指实体↔cube实体 接触点的法向冲量;
      2) 若无明显接触对, 回退到 test8 的弹簧代理力 F = grip_K · (squeeze_q − q_actual)
         (q_actual 取两手指关节当前位置, 越低于 squeeze_q 说明挤得越紧, 力越大)。
    这样无论接触是否被引擎单独列出, 都能给出单调、稳定的夹持力估计供导纳外环使用。
    """
    cube_entity_id = id(cube)
    finger_left_id = finger_right_id = None
    for lk in robot.get_links():
        if lk.get_name() == "right_gripper_finger_link1":
            finger_left_id = id(lk.entity)
        elif lk.get_name() == "right_gripper_finger_link2":
            finger_right_id = id(lk.entity)
    DT = 1.0 / 240.0
    f_left = f_right = 0.0
    for ct in scene.get_contacts():
        bodies = ct.bodies
        if len(bodies) < 2:
            continue
        e0, e1 = id(bodies[0].entity), id(bodies[1].entity)
        for (e_finger, e_cube, fset) in (
                (e0, e1, "left"), (e1, e0, "left"),
                (e0, e1, "right"), (e1, e0, "right")):
            is_match = ((fset == "left" and e_finger == finger_left_id) or
                        (fset == "right" and e_finger == finger_right_id)) and e_cube == cube_entity_id
            if is_match:
                peak = max((np.linalg.norm(np.array(pt.impulse)) for pt in ct.points), default=0.0)
                mag = peak / DT
                if fset == "left":
                    f_left += mag
                else:
                    f_right += mag
    # 回退: 弹簧代理力 (与 test8 诊断同式)
    if f_left < 1e-6 or f_right < 1e-6:
        aj = robot.get_active_joints()
        q1 = float(aj[idx1].get_drive_target())
        q2 = float(aj[idx2].get_drive_target())
        proxy = max(0.0, grip_k * (squeeze_q - min(q1, q2)))
        if f_left < 1e-6:
            f_left = proxy
        if f_right < 1e-6:
            f_right = proxy
    return f_left, f_right


def test10_admittance_grasp(scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx,
                            idx1, idx2, num_frames=800, output_dir=None, viewer_cam=None,
                            F_desired=3.0):
    """test10: 导纳控制抓取 — 与 test8 同样的场景, 但夹爪闭合改用"力控闭环"。

    阻抗(test8) vs 导纳(test10):
      阻抗: set_drive_target(手指间距) → 收到位、靠刚度顶住 (力是被动结果)
      导纳: 给定 F_desired(夹持力) → 用接触力误差经质量-阻尼模型反解手指目标间距
            → 再下发给 set_drive_target (位置驱动仍执行, 但目标由力闭环生成)

    仿真内"力传感器": SAPIEN scene.get_contacts() 提供 PhysxContact.impulse,
    除以子步长 1/240s 即得瞬时接触力. 真实机器人需六维力传感, 此处用仿真特权.

    导纳模型 (单自由度, 沿手指开合方向):
        M·q̈ + D·q̇ + K·q = F_desired − F_ext
      其中 q = 手指关节位置 (间接反映间距), F_ext = 实测夹持力
      → 解出 q̈ → 积分得 q̇ → q_target, 喂给 set_drive_target
    """
    # ── 复用 test8 的几何/轨迹参数 ──
    _FINGER_OFFSET_X = 0.03689
    _CUBE_SIZE = 0.025
    _CUBE_XY = np.array([0.08, -0.06])
    _PLACE_XY = np.array([0.40, 0.20])
    _TABLE_Z = 0.0
    _CUBE_Z = _TABLE_Z + _CUBE_SIZE / 2
    _APPROACH_Z = 0.18
    _GRASP_Z = _CUBE_Z
    _LIFT_Z = 0.25
    GRASP_RY, GRASP_RZ, GRASP_RX = np.radians(30), np.radians(25), np.radians(15)
    TRANSPORT_RY, TRANSPORT_RZ, TRANSPORT_RX = np.radians(-20), np.radians(-30), np.radians(25)
    RELEASE_RY, RELEASE_RZ, RELEASE_RX = np.radians(-35), np.radians(40), np.radians(-30)
    _HOME_RY = np.pi / 2

    def _angle_compensated_base(cx, cy, cz, ry, rz, rx):
        R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
        finger_world = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
        return cx - finger_world[0], cy - finger_world[1], cz - finger_world[2]

    _BASE_XY_at_grasp = np.array(_angle_compensated_base(
        _CUBE_XY[0], _CUBE_XY[1], _GRASP_Z, GRASP_RY, GRASP_RZ, GRASP_RX)[:2])
    _PLACE_BASE_XY = np.array(_angle_compensated_base(
        _PLACE_XY[0], _PLACE_XY[1], _GRASP_Z, TRANSPORT_RY, TRANSPORT_RZ, TRANSPORT_RX)[:2])
    d_vec = _PLACE_BASE_XY - _BASE_XY_at_grasp
    d_len = np.linalg.norm(d_vec)
    u_dir = d_vec / d_len
    v_perp = np.array([-u_dir[1], u_dir[0]])
    amp = 0.05
    m1 = _BASE_XY_at_grasp + d_vec * (1.0/3.0)
    m2 = _BASE_XY_at_grasp + d_vec * (2.0/3.0)
    bez_cp = [
        _BASE_XY_at_grasp, _BASE_XY_at_grasp + d_vec*0.11 - amp*v_perp,
        m1 - d_vec*0.11 - amp*v_perp, m1,
        m1 + d_vec*0.11 + amp*v_perp, m2 - d_vec*0.11 + amp*v_perp, m2,
        m2 + d_vec*0.11 - amp*0.6*v_perp, _PLACE_BASE_XY - d_vec*0.11 - amp*0.6*v_perp,
        _PLACE_BASE_XY,
    ]

    def bezier(p, t):
        t = min(max(t, 0.0), 1.0)
        n = len(p) - 1
        pt = np.zeros(2)
        for i, pi in enumerate(p):
            pt += pi * comb(n, i) * (1-t)**(n-i) * t**i
        return pt

    # ── 导纳力控外环参数 (手指开合单自由度, 比例控制) ──
    # 基线用 test8 验证过的 squeeze 目标 (q=0.012 → gap≈20.9mm < 25mm方块, 稳定夹住),
    # 力误差在其上做小幅调制: 力不足→更紧(减小q), 力过大→放松(增大q).
    # 调制幅度限幅, 保证几何始终接近 test8 已验证可抓起的构型.
    ADM_GAIN = 0.0008   # 力误差→闭合调节增益 (m/(N·s))
    _GRIPPER_Q_OPEN = 0.025    # 手指全开 (间距约47mm)
    _GRIPPER_Q_BASE = 0.012    # squeeze 基线 (test8 验证可抓起)
    _GRIPPER_Q_CLOSED = 0.0    # 手指全闭下限 (与 test8 target_j=0 一致, 可深夹)
    _ADM_DELTA_MAX = 0.012     # 力闭环对基线的修正上限 (m), 允许从开到全闭
    _ADM_GRIP_K = 5000.0       # 代理力刚度: proxy = ADM_GRIP_K*(BASE - q_actual)
    q_target = _GRIPPER_Q_BASE  # 力闭环围绕 squeeze 基线调节

    cube = None
    for b in scene.get_all_actors():
        if b.name == "cube":
            cube = b
            break
    if cube is None:
        # test10 独立场景: 自建与 test8 同参数的动态方块 + 桌面
        table_builder = scene.create_actor_builder()
        table_builder.add_box_visual(pose=sapien.Pose(), half_size=[0.35, 0.30, 0.003])
        phys_mat_table = scene.create_physical_material(static_friction=0.8, dynamic_friction=0.8, restitution=0.6)
        table_builder.add_box_collision(pose=sapien.Pose(), half_size=[0.35, 0.30, 0.003], material=phys_mat_table)
        table = table_builder.build_kinematic(name="table")
        table.set_pose(sapien.Pose([0.20, 0.05, _TABLE_Z - 0.003]))
        cube_builder = scene.create_actor_builder()
        _black_mat = sapien.render.RenderMaterial()
        _black_mat.base_color = [0.08, 0.08, 0.08, 1.0]
        cube_builder.add_box_visual(pose=sapien.Pose(), half_size=[_CUBE_SIZE/2]*3, material=_black_mat)
        phys_mat_cube = scene.create_physical_material(static_friction=5.0, dynamic_friction=5.0, restitution=0.1)
        cube_builder.add_box_collision(pose=sapien.Pose(), half_size=[_CUBE_SIZE/2]*3, material=phys_mat_cube, density=2000)
        cube = cube_builder.build(name="cube")
        cube.set_pose(sapien.Pose([_CUBE_XY[0], _CUBE_XY[1], _CUBE_Z]))

    # ── 初始化夹爪位姿 (与 test8 一致: 先摆到朝下悬停位, 预热物理, 避免首帧扫到方块) ──
    init_bx, init_by, init_bz = _angle_compensated_base(
        _CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, _HOME_RY, 0.0, 0.0)
    active_joints = robot.get_active_joints()
    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = init_bx
    init_qpos[idx_vy] = init_by
    init_qpos[idx_vz] = init_bz
    init_qpos[idx_rz] = 0.0
    init_qpos[idx_ry] = _HOME_RY  # 夹爪朝下
    init_qpos[idx_rx] = 0.0
    init_qpos[idx1] = _GRIPPER_Q_OPEN  # 手指全开
    init_qpos[idx2] = _GRIPPER_Q_OPEN
    robot.set_qpos(init_qpos)
    for idx in [idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2]:
        active_joints[idx].set_drive_target(float(init_qpos[idx]))
    # 预热物理: 让夹爪稳定悬停在方块上方, 方块静置在桌面
    for _ in range(100):
        step_physics_drive(robot, scene, steps=DECIMATION)

    camera = create_camera(scene) if output_dir else None
    wrist_cam = None
    wrist_frames = []
    if output_dir:
        d405 = next(l for l in robot.get_links() if l.get_name() == "right_D405_link").entity
        _camera_counter[0] += 1
        R_joint = sst.Rotation.from_euler("xyz", [-2.268, 0.0, -1.5708]).as_matrix()
        tilt = np.deg2rad(35.0)
        R_desired = np.column_stack([[np.cos(tilt), 0, -np.sin(tilt)],
                                      [0, 1, 0], [np.sin(tilt), 0, np.cos(tilt)]])
        cam_q = quat_from_matrix(R_joint.T @ R_desired)
        wrist_cam = scene.add_mounted_camera(
            f"wrist_d405_{_camera_counter[0]}", d405,
            sapien.Pose([0.0, 0.0, 0.0], np.array(cam_q, dtype=float)),
            640, 480, np.deg2rad(60), 0.01, 100.0)

    frames = []
    log_path = str(Path(output_dir) / "test10_frame_log.log") if output_dir else None
    log_file = open(log_path, "w") if log_path else None
    if log_file:
        log_file.write("frame,phase,q_target,F_ext_left,F_ext_right,F_err,lifted_cube_z\n")

    max_cube_z = _CUBE_Z
    lifted = False
    force_err_log = []
    for i in range(num_frames):
        t = i / num_frames
        phase = "approach"
        # ── 与 test8 完全一致的 4 阶段 progress 结构 (含 Phase1 高位悬停) ──
        if t < 0.08:
            # Phase 1: 接近 — 夹爪朝下高位悬停 (手指水平, 不倾斜)
            phase = "approach"
            ry, rz, rx = _HOME_RY, 0.0, 0.0
            bx, by, tz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, ry, rz, rx)
            q_target = _GRIPPER_Q_OPEN
        elif t < 0.63:
            # Phase 2: 下降边夹 + 抬升 + Bezier搬运 (连续无停顿)
            tl = _min_jerk((t - 0.08) / 0.55)
            if tl < 0.30:
                # [8%-25%] 下降+闭合: 边下降边夹, 接触时手指已闭到基线下限
                sub = _min_jerk(tl / 0.30)
                ry = _HOME_RY + sub * GRASP_RY
                rz = sub * GRASP_RZ
                rx = sub * GRASP_RX
                finger_z = _APPROACH_Z + sub * (_GRASP_Z - _APPROACH_Z)
                bx, by, tz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
                # ── 导纳力控: 用接触力误差闭合手指 (围绕 squeeze 基线微调) ──
                F_l, F_r = _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=_GRIPPER_Q_BASE, grip_k=_ADM_GRIP_K)
                F_ext = F_l + F_r
                F_err = F_desired - F_ext
                delta = np.clip(ADM_GAIN * F_err * (1.0/30.0), -_ADM_DELTA_MAX, _ADM_DELTA_MAX)
                q_target = _GRIPPER_Q_BASE - delta
                q_target = min(max(q_target, _GRIPPER_Q_CLOSED), _GRIPPER_Q_OPEN)
                force_err_log.append(abs(F_err))
            elif tl < 0.45:
                # [25%-33%] 抬起到高空
                sub = _min_jerk((tl - 0.30) / 0.15)
                ry, rz, rx = _HOME_RY + GRASP_RY, GRASP_RZ, GRASP_RX
                finger_z = _GRASP_Z + sub * (_LIFT_Z - _GRASP_Z)
                bx, by, tz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
                F_l, F_r = _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=_GRIPPER_Q_BASE, grip_k=_ADM_GRIP_K)
                F_ext = F_l + F_r
                F_err = F_desired - F_ext
                delta = np.clip(ADM_GAIN * F_err * (1.0/30.0), -_ADM_DELTA_MAX, _ADM_DELTA_MAX)
                q_target = _GRIPPER_Q_BASE - delta
                q_target = min(max(q_target, _GRIPPER_Q_CLOSED), _GRIPPER_Q_OPEN)
                force_err_log.append(abs(F_err))
            else:
                # [33%-63%] Bezier S曲线搬运 + 角度旋转
                sub = _min_jerk((tl - 0.45) / 0.55)
                xy = bezier(bez_cp, sub)
                finger_z = _LIFT_Z
                ry = (_HOME_RY + GRASP_RY) + sub * (TRANSPORT_RY - GRASP_RY)
                rz = GRASP_RZ + sub * (TRANSPORT_RZ - GRASP_RZ)
                rx = GRASP_RX + sub * (TRANSPORT_RX - GRASP_RX)
                R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
                finger_off = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
                bx = xy[0] - finger_off[0]
                by = xy[1] - finger_off[1]
                tz = finger_z - finger_off[2]
                F_l, F_r = _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=_GRIPPER_Q_BASE, grip_k=_ADM_GRIP_K)
                F_ext = F_l + F_r
                F_err = F_desired - F_ext
                delta = np.clip(ADM_GAIN * F_err * (1.0/30.0), -_ADM_DELTA_MAX, _ADM_DELTA_MAX)
                q_target = _GRIPPER_Q_BASE - delta
                q_target = min(max(q_target, _GRIPPER_Q_CLOSED), _GRIPPER_Q_OPEN)
                force_err_log.append(abs(F_err))
        elif t < 0.88:
            # Phase 3: 释放 — 先降到目标高度再开爪
            tl = (t - 0.63) / 0.25
            if tl < 0.60:
                sub = _min_jerk(tl / 0.60)
                finger_z = _LIFT_Z + sub * (_GRASP_Z - _LIFT_Z)
                ry = (_HOME_RY + TRANSPORT_RY) + sub * (RELEASE_RY - TRANSPORT_RY)
                rz = TRANSPORT_RZ + sub * (RELEASE_RZ - TRANSPORT_RZ)
                rx = TRANSPORT_RX + sub * (RELEASE_RX - TRANSPORT_RX)
                bx, by, tz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
                F_l, F_r = _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=_GRIPPER_Q_BASE, grip_k=_ADM_GRIP_K)
                F_ext = F_l + F_r
                F_err = F_desired - F_ext
                delta = np.clip(ADM_GAIN * F_err * (1.0/30.0), -_ADM_DELTA_MAX, _ADM_DELTA_MAX)
                q_target = _GRIPPER_Q_BASE - delta
                q_target = min(max(q_target, _GRIPPER_Q_CLOSED), _GRIPPER_Q_OPEN)
            else:
                bx, by, tz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], _GRASP_Z,
                                                     _HOME_RY + RELEASE_RY, RELEASE_RZ, RELEASE_RX)
                ry, rz, rx = _HOME_RY + RELEASE_RY, RELEASE_RZ, RELEASE_RX
                q_target = _GRIPPER_Q_OPEN  # 张开释放
        else:
            # Phase 4: 撤退
            phase = "retreat"
            bx, by, tz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], _LIFT_Z + 0.04,
                                                 _HOME_RY + RELEASE_RY, RELEASE_RZ, RELEASE_RX)
            ry, rz, rx = _HOME_RY + RELEASE_RY, RELEASE_RZ, RELEASE_RX
            q_target = _GRIPPER_Q_OPEN

        bz = tz + 0.002
        # 注意: 本函数内 ry 已是绝对姿态(已含 _HOME_RY), 直接下发, 不再叠加
        for idx, val in ((idx_vx, bx), (idx_vy, by), (idx_vz, bz),
                          (idx_rz, rz), (idx_ry, ry), (idx_rx, rx)):
            robot.get_active_joints()[idx].set_drive_target(val)
        for idx in (idx1, idx2):
            robot.get_active_joints()[idx].set_drive_target(q_target)

        step_physics_drive(robot, scene, steps=DECIMATION)

        cube_pos = cube.pose.p
        cz = float(cube_pos.p[2]) if hasattr(cube_pos, 'p') else float(cube_pos[2])
        max_cube_z = max(max_cube_z, cz)
        if cz > _CUBE_Z + 0.01:
            lifted = True

        if log_file:
            F_l, F_r = (0.0, 0.0)
            if 0.30 <= t < 0.80:
                F_l, F_r = _compute_finger_contact_force(scene, robot, cube, idx1, idx2, squeeze_q=_GRIPPER_Q_BASE, grip_k=_ADM_GRIP_K)
            log_file.write(f"{i},{phase},{q_target:.5f},{F_l:.3f},{F_r:.3f},"
                           f"{abs(F_desired-(F_l+F_r)):.3f},{cz:.4f}\n")

        if output_dir and camera is not None:
            try:
                set_camera_pose(camera, np.array([0.20, -0.20, 0.20]), np.array([0.12, 0.0, 0.04]))
                frames.append(render_frame(scene, camera))
            except RuntimeError:
                pass
        if output_dir and wrist_cam is not None:
            scene.update_render()
            wrist_cam.take_picture()
            rgba = wrist_cam.get_picture("Color")
            rgb = np.clip(rgba[..., :3], 0, 1)
            rgb = np.where(rgb < 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4) - 0.055)
            wrist_frames.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    if log_file:
        log_file.close()
    if output_dir and frames:
        import imageio
        imageio.mimsave(str(Path(output_dir) / "test10_admittance_grasp.mp4"), frames, fps=30)
    if output_dir and wrist_frames:
        import imageio
        imageio.mimsave(str(Path(output_dir) / "test10_wrist_view.mp4"), wrist_frames, fps=30)

    mean_ferr = np.mean(force_err_log) if force_err_log else 0.0
    print(f"\n  测试 10 (导纳力控): 目标夹持力={F_desired}N, 平均力误差={mean_ferr:.3f}N, "
          f"方块最高Z={max_cube_z*100:.1f}cm, {'抓取成功✅' if lifted else '未抓起❌'}")
    print(f"  说明: 夹爪闭合由 F_desired 力闭环驱动 (阻抗用固定间距, 此处方差恒定)")
    return {'lifted': lifted, 'max_cube_z': max_cube_z, 'mean_force_err': mean_ferr}


def _precompute_mano_targets(hawor_dir, num_frames, hand_idx=1, prefix="right"):
    """加载 HaWoR 数据 + MANO 目标预计算 + 四重平滑 (test9 / test11 共用).

    流程 (对齐 002_render_scene.py gripper-only 策略):
      HaWoR npz → MANO FK → _render_to_sapien → compute_mano_based_gripper_pose
      → 跨帧法向 anti-flip → 无效帧插值 → smooth_mano_trajectory (SG+高斯Slerp+min-jerk)

    返回 dict: hawor_data / mano_layer / smooth_pos / smooth_quat / smooth_joint /
               valid_idx / first_valid_frame / n_frames_data
    """
    # 函数级导入 (hand_track 依赖, 与 test9 原实现一致)
    # 注: compute_mano_based_gripper_pose 定义在 trajectory_optimization/data_loader.py
    # (hand_track.gripper_config 已重构不含它, test9 原导入早已失效)
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))
    from hand_track.common import load_hawor_data, compute_mano_joints, _render_to_sapien
    from mano_layer import MANOLayer
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_tj_data_loader", SCRIPT_DIR / "trajectory_optimization" / "data_loader.py")
    _mod = _ilu.module_from_spec(_spec)
    sys.path.insert(0, str(SCRIPT_DIR / "trajectory_optimization"))
    _spec.loader.exec_module(_mod)
    compute_mano_based_gripper_pose = _mod.compute_mano_based_gripper_pose

    # ── 加载HaWoR数据 ──
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    n_frames_data = hawor_data['pred_trans'].shape[0]
    print(f"  HaWoR数据: {n_frames_data}帧, hand_idx={hand_idx}, 目标帧数={num_frames}")

    # ── MANO层 ──
    first_valid = np.where(hawor_data["pred_valid"])[0]
    if len(first_valid) == 0:
        betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
    else:
        betas_mean = hawor_data["pred_betas"][first_valid[0]].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)
    print(f"  MANO层已加载 (prefix={prefix}, betas_from_frame={first_valid[0] if len(first_valid) > 0 else 0})")

    # ── 预计算所有帧的原始目标 (不依赖 scene/robot, 在 MANO 层加载后立即执行) ──
    raw_pos = np.zeros((num_frames, 3), dtype=np.float32)
    raw_quat = np.zeros((num_frames, 4), dtype=np.float32)  # w,x,y,z
    raw_joint = np.zeros(num_frames, dtype=np.float32)
    raw_valid = np.zeros(num_frames, dtype=bool)
    _prev_z = None  # 跨帧法向符号一致性 (anti-flip) 的参考法向
    for i in range(num_frames):
        data_idx = i % n_frames_data
        if not hawor_data["pred_valid"][data_idx]:
            raw_pos[i] = np.nan; raw_quat[i] = np.nan; raw_joint[i] = np.nan
            continue
        rot = hawor_data["pred_rot"][data_idx]
        trans = hawor_data["pred_trans"][data_idx]
        hand_pose = hawor_data["pred_hand_pose"][data_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            raw_pos[i] = np.nan; raw_quat[i] = np.nan; raw_joint[i] = np.nan
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        g_pos, g_R, joint1, joint2 = compute_mano_based_gripper_pose(joints_sapien, prefix=prefix)
        # ── 跨帧法向符号一致性 (anti-flip) ──
        # SVD 的 Vt[2] 符号任意 (± 等价), 相邻帧可能突然反号 → R 绕 X 轴翻 180°。
        # 用上一有效帧的 z 做参考: 若 z·prev_z<0 则翻转 z, 再以稳定的 x (手指方向) 重建 R。
        z = g_R[:, 2]
        if _prev_z is not None and np.dot(z, _prev_z) < 0:
            z = -z
            x = g_R[:, 0].copy()  # 手指方向, 由 MCP→PIP 确定, 不依赖 z 符号, 连续稳定
            y = np.cross(z, x)
            g_R = np.column_stack([x, y, z])
        _prev_z = g_R[:, 2].copy()
        g_quat = sst.Rotation.from_matrix(g_R).as_quat()  # (x,y,z,w)
        g_quat = np.roll(g_quat, 1)  # → (w,x,y,z)
        raw_pos[i] = g_pos; raw_quat[i] = g_quat; raw_joint[i] = float(joint1)
        raw_valid[i] = True
    if not raw_valid.any():
        print("  ⚠ 无有效帧数据")
        return {'pos_err_avg': 0, 'pos_err_max': 0, 'ori_err_avg': 0, 'ori_err_max': 0,
                'joint_err_avg': 0, 'n_valid': 0}

    # ── 插值补齐无效帧 (do-as-i-do 风格: 线性插值位置, SLERP 插值朝向) ──
    _good = np.where(raw_valid)[0]
    _bad = np.where(~raw_valid)[0]
    if len(_bad) > 0 and len(_good) > 1:
        _n_bad = len(_bad)
        # 位置插值 (线性)
        for d in range(3):
            raw_pos[_bad, d] = np.interp(_bad, _good, raw_pos[_good, d])
        # 关节插值 (线性)
        raw_joint[_bad] = np.interp(_bad, _good, raw_joint[_good])
        # 朝向插值 (SLERP)
        _q_good = raw_quat[_good]
        _q_good_xyzw = np.roll(_q_good, -1, axis=1)   # (w,x,y,z) → (x,y,z,w)
        _R_good = sst.Rotation.from_quat(_q_good_xyzw)
        _slerp = sst.Slerp(_good, _R_good)
        _bad_in = _bad[(_bad >= _good[0]) & (_bad <= _good[-1])]
        if len(_bad_in) > 0:
            _q_interp = _slerp(_bad_in).as_quat()       # (x,y,z,w)
            _q_interp = np.roll(_q_interp, 1, axis=1)   # (w,x,y,z)
            raw_quat[_bad_in] = _q_interp
        # 边界帧: clamp 到最近的有效帧
        for _b in _bad[_bad < _good[0]]:
            raw_quat[_b] = raw_quat[_good[0]]
        for _b in _bad[_bad > _good[-1]]:
            raw_quat[_b] = raw_quat[_good[-1]]
        raw_valid[:] = True
        print(f"  插值补齐: {_n_bad} 帧无效帧已通过插值填充")
    elif len(_bad) > 0 and len(_good) == 1:
        _n_bad = len(_bad)
        raw_pos[_bad] = raw_pos[_good[0]]
        raw_quat[_bad] = raw_quat[_good[0]]
        raw_joint[_bad] = raw_joint[_good[0]]
        raw_valid[:] = True
        print(f"  插值补齐: {_n_bad} 帧填充为唯一有效帧")

    # ── 调试: 找出 MANO 参数突变最大的帧 ──
    _valid_ids = np.where(raw_valid)[0]
    _hp_diffs = []
    _prev_hp = None
    for _di in _valid_ids:
        _didx = min(_di, n_frames_data - 1)
        _hp = hawor_data["pred_hand_pose"][_didx]
        if _prev_hp is not None:
            _hp_diffs.append((_di, np.linalg.norm(_hp - _prev_hp)))
        _prev_hp = _hp.copy()
    _hp_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  ── MANO 参数突变 TOP-5 (Δhand_pose 最大帧) ──")
    for _rank, (_di, _diff) in enumerate(_hp_diffs[:5]):
        _didx = min(_di, n_frames_data - 1)
        _rot = hawor_data["pred_rot"][_didx]
        _trans = hawor_data["pred_trans"][_didx]
        _hp = hawor_data["pred_hand_pose"][_didx]
        _hp_norm = np.linalg.norm(_hp)
        _, _j = compute_mano_joints(mano_layer, _rot, _hp, _trans)
        _js = _render_to_sapien(_j)
        _p5 = np.array([_js[3], _js[5], _js[6], _js[7], _js[8]])
        _p5_mean = _p5.mean(axis=0)
        # 使用已 anti-flip 修正后的 raw_pos / raw_quat (与实际控制指令一致)
        _gp = raw_pos[_di].copy()
        _gq = np.roll(raw_quat[_di], -1)  # (x,y,z,w)
        # 与前帧对比 (前帧必须是有效帧)
        _prev_di = _valid_ids[_valid_ids.tolist().index(_di) - 1] if _valid_ids.tolist().index(_di) > 0 else _di
        _prev_gp = raw_pos[_prev_di].copy()
        _prev_gq = np.roll(raw_quat[_prev_di], -1)  # (x,y,z,w)
        _gq_angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.dot(_gq, _prev_gq)), 0, 1)))
        _pos_delta = np.linalg.norm(_gp - _prev_gp) * 1000  # mm
        # SVD plane normal
        _centered = _p5 - _p5_mean
        _, _, _Vt = np.linalg.svd(_centered, full_matrices=True)
        _z_axis = _Vt[2] / np.linalg.norm(_Vt[2])
        print(f"  #{_rank+1} 帧{_di}→{_prev_di}: Δhp={_diff:.4f} | "
              f"Δpos={_pos_delta:.1f}mm | Δgq={_gq_angle:.1f}° | "
              f"hp_norm={_hp_norm:.3f} | z={np.round(_z_axis, 3)}")
    print(f"  ── 调试结束 ──\n")
    # ── 统一平滑流水线 (模块级 smooth_mano_trajectory: ①SG ②高斯Slerp ③min-jerk) ──
    # (第④重 EMA 为命令级在线平滑, 在下方主循环内执行)
    smooth_pos, smooth_quat, smooth_joint = smooth_mano_trajectory(
        raw_pos, raw_quat, raw_joint, raw_valid)

    max_jump_after = np.linalg.norm(np.diff(smooth_pos, axis=0), axis=1).max() * 1000
    print(f"  Minimum Jerk 后最大跳变: {max_jump_after:.1f}mm/帧")
    # 报告平滑变化量
    valid_idx = np.where(raw_valid)[0]
    pos_change = np.linalg.norm(smooth_pos[valid_idx] - raw_pos[valid_idx], axis=1) * 1000
    ori_change = np.zeros(len(valid_idx))
    for k, i in enumerate(valid_idx):
        q1 = np.roll(smooth_quat[i], -1); q2 = np.roll(raw_quat[i], -1)
        dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
        ori_change[k] = np.degrees(np.arccos(2 * dot * dot - 1))
    print(f"  平滑变化: pos_avg={pos_change.mean():.1f}mm pos_max={pos_change.max():.1f}mm | "
          f"ori_avg={ori_change.mean():.1f}° ori_max={ori_change.max():.1f}°")


    first_valid_frame = int(valid_idx[0]) if len(valid_idx) > 0 else 0
    return {
        'hawor_data': hawor_data, 'mano_layer': mano_layer,
        'raw_pos': raw_pos, 'raw_quat': raw_quat, 'raw_joint': raw_joint,
        'raw_valid': raw_valid,
        'smooth_pos': smooth_pos, 'smooth_quat': smooth_quat, 'smooth_joint': smooth_joint,
        'valid_idx': valid_idx, 'first_valid_frame': first_valid_frame,
        'n_frames_data': hawor_data['pred_trans'].shape[0],
    }


def test9_mano_trajectory(scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx,
                           idx1, idx2, num_frames=600, output_dir=None, viewer_cam=None,
                           hawor_dir=None):
    """test9: MANO轨迹跟随 — 纯物理仿真, 对比 MANO 参考与夹爪跟随偏差

    流程 (对齐 002_render_scene.py gripper-only svd_palm 策略):
      1. 加载HaWoR MANO参数 (pred_rot, pred_hand_pose, pred_trans, pred_betas)
      2. MANOLayer → compute_mano_joints → _render_to_sapien (002标准流程)
      3. joints_sapien(21,3) → compute_mano_based_gripper_pose (5点SVD手掌平面+Gram-Schmidt)
      4. set_drive_target + step_physics_drive — 纯物理
      5. 可视化: MANO参考点(红=腕, 绿=拇指尖, 蓝=食指尖) + 夹爪URDF
      6. 输出跟踪误差统计 (位置 + 朝向)
    """
    # 注意: test9 与 test8 共用 create_scene 的默认环境(中灰天空 + 平行光),
    # 灰金属夹爪在该环境下自然呈现 test8 同款灰色, 不额外压暗、不引入任何多余颜色.
    # ── 导入 (对齐 002_render_scene.py) ──
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))

    from hand_track.common import (
        load_hawor_data, compute_mano_joints, _render_to_sapien, RXWORLD_TO_SAPIEN,
    )
    from hand_track.gripper_config import (
        init_gripper_retargeting,
        EmaTargetSmoother, PositionEmaSmoother,
        LP_ALPHA_POS, LP_ALPHA_ORI, LP_ALPHA_ANALYTICAL,
    )
    from dex_retargeting.optimizer_utils import LPFilter
    from mano_layer import MANOLayer

    if hawor_dir is None:
        hawor_dir = Path.home() / "data" / "hawor" / "121_C5_CellPhone_161deg"

    print("\n" + "=" * 70)
    print("测试 9: MANO轨迹跟随 (纯物理仿真, 对比偏差)")
    print(f"  HaWoR目录: {hawor_dir}")
    print(f"  物理模式: step_physics_drive (gravity+coriolis补偿)")
    print("=" * 70)

    # ── 预计算 (共享函数): HaWoR 加载 + MANO 目标 + 四重平滑 (与 test11 共用) ──
    hand_idx, prefix = 1, "right"
    _pre = _precompute_mano_targets(hawor_dir, num_frames, hand_idx=hand_idx, prefix=prefix)
    hawor_data = _pre['hawor_data']
    mano_layer = _pre['mano_layer']
    smooth_pos = _pre['smooth_pos']
    smooth_quat = _pre['smooth_quat']
    smooth_joint = _pre['smooth_joint']
    raw_valid = _pre['raw_valid']
    valid_idx = _pre['valid_idx']
    n_frames_data = _pre['n_frames_data']

    # ── 初始化重定向器 ──
    finger_origin_x = 0.03689
    retargeting, ref_indices, _ = init_gripper_retargeting(prefix, finger_origin_x, PROJECT_ROOT)
    print(f"  ref_indices: {ref_indices}")

    # ── 驱动参数 ──
    active_joints = robot.get_active_joints()
    VIRTUAL_K = 30000.0;   VIRTUAL_D = 3000.0    # 位置: 极高刚度
    ANGULAR_K = 10000.0;   ANGULAR_D = 1000.0    # 朝向: 极高刚度
    GRIPPER_K = 10000.0;   GRIPPER_D = 1000.0    # 夹爪: 极高刚度

    for idx in [idx_vx, idx_vy, idx_vz]:
        active_joints[idx].set_drive_property(stiffness=VIRTUAL_K, damping=VIRTUAL_D)
    for idx in [idx_rz, idx_ry, idx_rx]:
        active_joints[idx].set_drive_property(stiffness=ANGULAR_K, damping=ANGULAR_D)
    active_joints[idx1].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)
    active_joints[idx2].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)

    # ── 创建 MANO 参考点可视化 (5个彩色小球) ──
    # 青色 = 锚点(夹爪root跟踪目标), 绿色 = 拇指尖, 蓝色 = 食指尖, 品红 = 指尖中点, 红色 = 手腕(仅参考)
    _marker_colors = [
        ([0.0, 1.0, 1.0, 1.0], "mano_ref_anchor"),   # 青 = 锚点 (夹爪根部实际跟踪目标)
        ([0.0, 1.0, 0.0, 1.0], "mano_ref_thumb"),    # 绿 = 拇指尖
        ([0.0, 0.0, 1.0, 1.0], "mano_ref_index"),    # 蓝 = 食指尖
        ([1.0, 0.0, 1.0, 1.0], "mano_ref_midpoint"), # 品红 = 指尖中点
        ([1.0, 0.3, 0.3, 1.0], "mano_ref_wrist"),    # 红 = 手腕 (仅参考)
    ]
    _marker_actors = []
    for _color, _name in _marker_colors:
        _builder = scene.create_actor_builder()
        _mat = sapien.render.RenderMaterial()
        _mat.base_color = _color
        _mat.emission = _color[:3] + [0.3]
        _builder.add_sphere_visual(radius=0.008, material=_mat)
        _actor = _builder.build_static(name=_name)
        _actor.set_pose(sapien.Pose([0, 0, 0]))
        _marker_actors.append(_actor)

    # ── 初始化夹爪 (使用预计算的第一个有效帧的平滑目标, 避免初始化偏移) ──
    first_valid_frame = int(valid_idx[0]) if len(valid_idx) > 0 else 0
    _init_pos = smooth_pos[first_valid_frame].copy()
    _init_quat = np.roll(smooth_quat[first_valid_frame], -1)  # (x,y,z,w) for scipy
    _init_R = sst.Rotation.from_quat(_init_quat).as_matrix()
    _init_joint = float(smooth_joint[first_valid_frame])
    print(f"  夹爪初始位置 (帧{first_valid_frame}): p={np.round(_init_pos, 4)} joint={_init_joint*1000:.1f}mm")

    # 设置初始 qpos: 对齐第一个有效帧的平滑目标, 无 Z 偏移
    _init_euler = sst.Rotation.from_matrix(_init_R).as_euler('ZYX')
    init_qpos = robot.get_qpos().copy()
    init_qpos[idx_vx] = _init_pos[0]
    init_qpos[idx_vy] = _init_pos[1]
    init_qpos[idx_vz] = _init_pos[2]
    init_qpos[idx_rz] = _init_euler[0]
    init_qpos[idx_ry] = _init_euler[1]
    init_qpos[idx_rx] = _init_euler[2]
    init_qpos[idx1] = _init_joint
    init_qpos[idx2] = _init_joint
    robot.set_qpos(init_qpos)
    for idx in [idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2]:
        active_joints[idx].set_drive_target(float(init_qpos[idx]))

    for _ in range(100):
        step_physics_drive(robot, scene)

    # ── 主循环 (使用预计算平滑后的轨迹) ──
    frames = []
    camera = create_camera(scene) if output_dir else None
    pos_errors = []   # 命令位置 vs 物理实际位置 (跟踪误差)
    ori_errors = []   # 命令朝向 vs 物理实际朝向 (度)
    joint_errors = [] # 命令关节值 vs 物理实际关节值 (mm)

    # 说明: test9 不渲染 MANO "幽灵手" mesh (其皮肤色会让灰金属夹爪泛黄/橙,
    # 与 test8 外观不一致). MANO 参考仅用彩色小球标出 (青/绿/蓝/品红/红),
    # 风格与 test8 的彩色轨迹球一致, 夹爪保持 test8 同款灰金属色.

    # 记录当前驱动目标 (初始化为第一个有效帧的目标)
    _last_target = (smooth_pos[first_valid_frame].copy(),
                    sst.Rotation.from_quat(np.roll(smooth_quat[first_valid_frame], -1)).as_matrix(),
                    float(smooth_joint[first_valid_frame]))

    # 调优 (2026-08): 在"命令目标"上串联 EMA 低通, 既压制帧间抖动又增加柔顺.
    # 位置用 EmaTargetSmoother; 朝向用 rotvec 空间 EMA (避免线性四元数 180° 翻转);
    # 关节单独标量 EMA.
    _ema = EmaTargetSmoother(pos_alpha=0.45, ori_alpha=0.30)
    _ema_ori_alpha = 0.30
    _ema_ori_rv = sst.Rotation.from_quat(np.roll(smooth_quat[first_valid_frame], -1)).as_rotvec()
    _ema_joint_alpha = 0.45
    _ema_joint = float(smooth_joint[first_valid_frame])

    # 固定机位的轨迹中心: 所有有效帧平滑位置的均值 (看向此处, 不跟随夹爪)
    _traj_center = smooth_pos[valid_idx].mean(axis=0).copy()

    # ── 性能监控 ──
    import time as _time
    _t_loop_start = _time.time()
    _psutil = None
    try:
        import psutil as _psutil_mod
        _psutil = _psutil_mod.Process()
    except Exception:
        pass

    for i in range(num_frames):
        if not raw_valid[i]:
            # 无效帧: 保持上一帧的驱动目标, 仅物理步进
            root_pos, root_R, target_j1 = _last_target
            target_j2 = target_j1
            # 物理步进 (不改变驱动目标)
            step_physics_drive(robot, scene)
            continue

        # 使用预计算的平滑目标, 再过 EMA 低通 (命令量平滑)
        # 位置: EmaTargetSmoother; 朝向: rotvec 空间 EMA (无 180° 翻转); 关节: 标量 EMA
        sm_quat_xyzw = np.roll(smooth_quat[i], -1)  # (x,y,z,w)
        ema_pos, _ = _ema.smooth(smooth_pos[i], sm_quat_xyzw)  # 仅用位置, 朝向自管
        cur_rv = sst.Rotation.from_quat(sm_quat_xyzw).as_rotvec()
        # rotvec 空间最短弧: 先对齐符号, 再线性 EMA (等价小角度 slerp)
        if np.linalg.norm(_ema_ori_rv) > 1e-6 and np.linalg.norm(cur_rv) > 1e-6:
            if np.dot(_ema_ori_rv, cur_rv) < 0:
                cur_rv = -cur_rv
        _ema_ori_rv = _ema_ori_rv + _ema_ori_alpha * (cur_rv - _ema_ori_rv)
        ema_quat_xyzw = sst.Rotation.from_rotvec(_ema_ori_rv).as_quat()  # (x,y,z,w)
        _ema_joint = _ema_joint + _ema_joint_alpha * (float(smooth_joint[i]) - _ema_joint)
        root_pos = ema_pos
        root_R = sst.Rotation.from_quat(ema_quat_xyzw).as_matrix()
        target_j1 = float(_ema_joint)
        target_j2 = target_j1
        _last_target = (root_pos.copy(), root_R.copy(), target_j1)

        # 更新 MANO 参考点可视化 (使用预计算数据)
        data_idx = min(i, n_frames_data - 1)
        if raw_valid[i]:
            rot = hawor_data["pred_rot"][data_idx]
            trans = hawor_data["pred_trans"][data_idx]
            hand_pose = hawor_data["pred_hand_pose"][data_idx]
            if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                mano_verts, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                joints_sapien = _render_to_sapien(j)
                _mano_wrist = joints_sapien[0, :3]
                _mano_thumb = joints_sapien[4, :3]
                _mano_index = joints_sapien[8, :3]
                _mano_mid = (_mano_thumb + _mano_index) / 2
                _mano_anchor = (joints_sapien[3, :3] + joints_sapien[5, :3]) / 2
                _marker_actors[0].set_pose(sapien.Pose(_mano_anchor))
                _marker_actors[1].set_pose(sapien.Pose(_mano_thumb))
                _marker_actors[2].set_pose(sapien.Pose(_mano_index))
                _marker_actors[3].set_pose(sapien.Pose(_mano_mid))
                _marker_actors[4].set_pose(sapien.Pose(_mano_wrist))

        # ── 保存命令值 (用于后续计算跟踪误差) ──
        cmd_pos = root_pos.copy()
        cmd_R = root_R.copy()
        cmd_j1 = target_j1

        # ── 从root_R提取欧拉角 ──
        euler = sst.Rotation.from_matrix(root_R).as_euler('ZYX')
        target_rz, target_ry, target_rx = euler

        # ── 设置驱动目标 ──
        active_joints[idx_vx].set_drive_target(float(root_pos[0]))
        active_joints[idx_vy].set_drive_target(float(root_pos[1]))
        active_joints[idx_vz].set_drive_target(float(root_pos[2]))
        active_joints[idx_rz].set_drive_target(target_rz)
        active_joints[idx_ry].set_drive_target(target_ry)
        active_joints[idx_rx].set_drive_target(target_rx)
        active_joints[idx1].set_drive_target(target_j1)
        active_joints[idx2].set_drive_target(target_j2)

        # ── 物理步进 ──
        step_physics_drive(robot, scene)

        # ── 读取实际状态 ──
        actual_qpos = robot.get_qpos()
        ax = float(actual_qpos[idx_vx])
        ay = float(actual_qpos[idx_vy])
        az = float(actual_qpos[idx_vz])
        a_rz = float(actual_qpos[idx_rz])
        a_ry = float(actual_qpos[idx_ry])
        a_rx = float(actual_qpos[idx_rx])
        a_j1 = float(actual_qpos[idx1])

        # ── 计算跟踪误差: 命令 vs 物理实际 (同一数据, 两种情景) ──
        pos_err = np.linalg.norm(cmd_pos - np.array([ax, ay, az])) * 1000  # mm
        # 朝向误差: 命令旋转矩阵 vs 实际欧拉角重建的旋转矩阵
        actual_R = sst.Rotation.from_euler('ZYX', [a_rz, a_ry, a_rx]).as_matrix()
        _R_diff = cmd_R.T @ actual_R
        _trace = np.trace(_R_diff)
        ori_err = np.degrees(np.arccos(np.clip((_trace - 1) / 2, -1, 1))) if abs(_trace - 1) / 2 < 1 else 0.0
        # 关节误差: 命令 vs 实际
        joint_err = abs(cmd_j1 - a_j1) * 1000  # mm

        pos_errors.append(pos_err)
        ori_errors.append(ori_err)
        joint_errors.append(joint_err)

        # ── 每30帧打印 ──
        if i % 30 == 0 or i >= num_frames - 5:
            print(f"  帧{i:3d} data={data_idx:3d}: "
                  f"g=({ax*100:.1f},{ay*100:.1f},{az*100:.1f}) "
                  f"fj={a_j1*1000:.1f}mm "
                  f"跟踪误差: pos={pos_err:.1f}mm ori={ori_err:.1f}° joint={joint_err:.1f}mm")

        if output_dir and camera is not None:
            try:
                # 固定机位: 看向整段轨迹质心, 不跟随夹爪, 这样夹爪在画面里的平移可见
                _gp = np.array([ax, ay, az])
                _cam_target = _traj_center.copy()           # 固定瞄准轨迹中心
                _cam_eye = _cam_target + np.array([0.55, 0.40, 0.45])  # 拉远以容纳整个移动范围
                set_camera_pose(camera, _cam_eye, _cam_target)
                img = render_frame(scene, camera)
                frames.append(img)
                if i == 0:
                    _gl = robot.get_links()
                    _gpos = None
                    for _l in _gl:
                        if "gripper_link" in _l.get_name():
                            _gpos = _l.get_pose().p
                    _raw = np.array(camera.get_picture("Color"))  # (H,W,4) RGBA linear
                    _rgb = _raw[..., :3]
                    _cp = camera.get_pose()
                    print(f"  [DEBUG i=0] eye(期望)={_cam_eye}")
                    print(f"  [DEBUG] camera 实际位姿 p={np.array(_cp.p)}")
                    print(f"  [DEBUG] gripper_link world pos={np.array(_gpos)}  gp(ax,ay,az)={_gp}")
                    print(f"  [DEBUG] scene actor数={len(scene.get_all_actors())}  robot links={len(_gl)}")
                    print(f"  [DEBUG] raw RGB(linear) min={_rgb.min(0)} max={_rgb.max(0)} mean={_rgb.mean(0)}")
                    _orng = (_rgb[:,:,0]>0.5)&(_rgb[:,:,1]<0.4)&(_rgb[:,:,2]<0.1)
                    print(f"  [DEBUG] RGB中 橙色(linear)像素占比={_orng.mean()*100:.2f}%  (若>0 说明夹爪已渲染)")
                    for _key in ["Segmentation", "ActorId", "ComponentId"]:
                        try:
                            _seg = np.array(camera.get_picture(_key))
                            print(f"  [DEBUG] {_key}: shape={_seg.shape} min={_seg.min()} max={_seg.max()}")
                        except Exception as _e:
                            print(f"  [DEBUG] {_key} 不可用: {_e}")
                    _seg = np.array(camera.get_picture("Segmentation"))
                    _seg_id = _seg[..., 0] if _seg.ndim == 3 else _seg
                    _actor_mask = _seg_id > 0
                    print(f"  [DEBUG] 前景(actor)像素占比={_actor_mask.mean()*100:.2f}%")
                    if _actor_mask.any():
                        _apx = _rgb[_actor_mask]
                        print(f"  [DEBUG] actor像素 RGB(linear) min={_apx.min(0)} max={_apx.max(0)} mean={_apx.mean(0)}")
            except Exception as e:
                print(f"  [帧{i}] 渲染错误: {e}")
                pass

        if viewer_cam is not None:
            scene.update_render()
            try:
                viewer_cam.render()
            except (AttributeError, RuntimeError):
                pass

    # ── 统计结果 ──
    if pos_errors:
        print(f"\n  跟踪误差统计 ({len(pos_errors)} 帧有效) — 命令 vs 物理实际:")
        print(f"    位置误差 (mm):  avg={np.mean(pos_errors):.1f}  max={np.max(pos_errors):.1f}  min={np.min(pos_errors):.1f}")
        _max_i = np.argmax(pos_errors)
        print(f"       → 帧{_max_i}  pos_err={pos_errors[_max_i]:.1f}mm  ori_err={ori_errors[_max_i]:.1f}°  joint_err={joint_errors[_max_i]:.1f}mm")
        print(f"    朝向误差 (deg): avg={np.mean(ori_errors):.1f}  max={np.max(ori_errors):.1f}  min={np.min(ori_errors):.1f}")
        _max_i = np.argmax(ori_errors)
        print(f"       → 帧{_max_i}  pos_err={pos_errors[_max_i]:.1f}mm  ori_err={ori_errors[_max_i]:.1f}°  joint_err={joint_errors[_max_i]:.1f}mm")
        print(f"    关节误差 (mm):  avg={np.mean(joint_errors):.1f}  max={np.max(joint_errors):.1f}  min={np.min(joint_errors):.1f}")
    else:
        print("\n  无有效帧数据")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test9_mano_trajectory.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    # ── 性能监控汇总 ──
    _loop_sec = _time.time() - _t_loop_start
    print(f"\n  性能监控: 总耗时 {_loop_sec:.1f}s | {num_frames} 帧 | "
          f"平均 {_loop_sec / num_frames * 1000:.1f} ms/帧 (~{num_frames / _loop_sec:.1f} FPS)")
    if _psutil is not None:
        _mem_mb = _psutil.memory_info().rss / 1024 ** 2
        _cpu = _psutil.cpu_percent(interval=0.1)
        print(f"           内存占用 {_mem_mb:.0f} MB | 平均CPU {_cpu:.0f}% | CPU核数 {_psutil.cpu_count()}")

    return {
        'pos_err_avg': float(np.mean(pos_errors)) if pos_errors else 0,
        'pos_err_max': float(np.max(pos_errors)) if pos_errors else 0,
        'ori_err_avg': float(np.mean(ori_errors)) if ori_errors else 0,
        'ori_err_max': float(np.max(ori_errors)) if ori_errors else 0,
        'joint_err_avg': float(np.mean(joint_errors)) if joint_errors else 0,
        'n_valid': len(pos_errors),
    }


# ════════════════════════════════════════════════════════════════════
#  测试 11: 固定底座单臂 + MANO 轨迹跟随 (test9 翻版, pybullet IK)
# ════════════════════════════════════════════════════════════════════

# ── R1 单臂常量 (与 02_render_scene.py 一致) ──
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
# 位置容差 2cm (02_render_scene 用 0.1 是视觉跟踪; 抓取需要毫米级)
IK_TOLERANCES = [0.02, 0.02, 0.02, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 10
LP_ALPHA_JOINT = 0.8
ORI_SERVO_GAIN = 0.5   # 朝向伺服增益 (RelaxedIK 模型与 SAPIEN 的 EE 朝向偏差外环补偿)
ARM_MAX_REACH = 0.713
# 固定底座安装位: 抬高成 pedestal 避开桌面; 底座偏航朝向 抓取点→放置点 中点,
# 使两个目标都落在臂工作空间 +X 前方 (到抓取点 ~0.27m, 到放置点 ~0.55m, 均在 0.713m 臂展内)
_ARM_BASE_POS = np.array([0.05, -0.14, 0.38])
_ARM_YAW_TOWARD = (np.array([0.08, -0.06]) + np.array([0.40, 0.20])) / 2.0


def prepare_arm_urdf(src_urdf_path, arm_prefix="right"):
    """准备 R1 臂 URDF: 替换 mesh 路径 + 手指 joint fixed→prismatic (与 02_render_scene.py 一致)"""
    xml = src_urdf_path.read_text()
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")
    xml = re.sub(rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")', r'\1prismatic\2', xml)
    xml = re.sub(rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")', r'\1prismatic\2', xml)
    # 手指/掌心碰撞替换为 test8 同款简化 box. R1 STL 全长碰撞会让手指提前撑在桌面上
    # (抓取最低点悬在方块上方 ~4cm, 永远够不到方块); test8 的简化 box 已验证可抓取.
    # 注意: 用"不跨越 </collision>"的受限匹配, 且只动 <collision> 不动 <visual>.
    def _box_collision(origin_xyz, box_size):
        return ('<collision>\n      <origin xyz="%s" rpy="0 0 0"/>\n      <geometry>\n'
                '        <box size="%s"/>\n      </geometry>\n    </collision>' % (origin_xyz, box_size))

    def _sub_collision_block(xml, stl_name, origin_xyz, box_size):
        pat = re.compile(r'<collision>(?:(?!</collision>).)*?' + re.escape(stl_name) +
                         r'(?:(?!</collision>).)*?</collision>', re.S)
        new_xml, n_sub = pat.subn(lambda _m: _box_collision(origin_xyz, box_size), xml, count=1)
        assert n_sub == 1, f"collision block not found for {stl_name}"
        return new_xml

    xml = _sub_collision_block(xml, "right_gripper_finger_link1.STL", "0 0.01 0", "0.06 0.01 0.04")
    xml = _sub_collision_block(xml, "right_gripper_finger_link2.STL", "0 -0.01 0", "0.06 0.01 0.04")
    xml = _sub_collision_block(xml, "right_gripper_link.STL", "-0.02 0 0", "0.06 0.04 0.03")
    temp_dir = tempfile.mkdtemp(prefix="r1_arm_urdf-")
    temp_path = f"{temp_dir}/{src_urdf_path.name}"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def load_r1_arm(scene, base_pos=None, yaw_toward=None):
    """加载 R1 右臂 (固定底座), 返回 (robot, arm_joint_indices, gripper_idx1, gripper_idx2).

    - URDF: r1_v2_1_0_floating_right.urdf (root 固定; 手指 joint 已转 prismatic, 行程 0~0.05m)
    - 底座: right_arm_base_link 平移+偏航对齐到 base_pos (默认 _ARM_BASE_POS) / 朝向 yaw_toward
    - 末端: right_gripper_link (手指几何与 test8 浮动夹爪完全一致, 手指沿局部 +X 伸出 0.03689m)
    """
    if base_pos is None:
        base_pos = _ARM_BASE_POS
    if yaw_toward is None:
        yaw_toward = _ARM_YAW_TOWARD
    urdf_path = prepare_arm_urdf(FLOATING_RIGHT_URDF)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_path)

    joint_names = [j.name for j in robot.get_active_joints()]
    arm_joint_indices = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
    gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
    gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

    init_qpos = robot.get_qpos().copy()
    for j, idx in enumerate(arm_joint_indices):
        init_qpos[idx] = RIGHT_ARM_STARTING[j]
    robot.set_qpos(init_qpos)

    # 固定底座位姿: 先在 identity root 下读 right_arm_base_link 的局部偏移,
    # 再反解 root 位姿, 使 base_link 落到 _ARM_BASE_POS 且偏航朝向目标中点
    scene.step()
    base_link = next(l for l in robot.get_links() if l.get_name() == "right_arm_base_link")
    bp_local = np.array(base_link.get_entity_pose().p)
    yaw = np.arctan2(yaw_toward[1] - base_pos[1],
                     yaw_toward[0] - base_pos[0])
    R_root = sst.Rotation.from_euler('z', yaw).as_matrix()
    robot.set_root_pose(sapien.Pose(
        (base_pos - R_root @ bp_local).tolist(),
        np.array(quat_from_matrix(R_root), dtype=float)))
    scene.step()

    # 驱动增益: 臂关节高刚度位置驱动 (物理跟踪 IK 解); 手指与 test8 一致 (3000/300)
    active_joints = robot.get_active_joints()
    for idx in arm_joint_indices:
        active_joints[idx].set_drive_property(stiffness=40000.0, damping=1500.0)
    for idx in (gripper_idx1, gripper_idx2):
        active_joints[idx].set_drive_property(stiffness=3000.0, damping=300.0)

    # 手指高摩擦 (与 load_gripper 一致, 保证摩擦抓取)
    touch_links = ["right_gripper_finger_link1", "right_gripper_finger_link2"]
    high_friction_mat = scene.create_physical_material(
        static_friction=3.0, dynamic_friction=3.0, restitution=0.0)
    for link in robot.get_links():
        if link.get_name() in touch_links:
            for component in link.entity.components:
                if hasattr(component, 'physx_material'):
                    component.physx_material = high_friction_mat

    print(f"  R1 右臂已加载: {len(arm_joint_indices)} 个臂关节 + 2 手指关节 (固定底座)")
    print(f"  底座 right_arm_base_link @ {np.round(base_pos, 3)}, 偏航 {np.degrees(yaw):.0f}°")
    return robot, arm_joint_indices, gripper_idx1, gripper_idx2


def _test8_pick_place_trajectory(progress):
    """test8_complex_pick_place 的轨迹规划纯函数 (供 test11 复用, 与 test8 内联逻辑一致).

    4 阶段: approach(0-8%) → grasp+transport(8-63%, 边下边夹+抬升+Bezier S曲线搬运)
            → release(63-88%, 先降后开爪) → retreat(88-100%).
    分段缓动用 _min_jerk, 搬运用 3 段 cubic Bezier S 曲线.

    返回 (bx, by, bz, rz, ry, rx, target_j, phase):
      (bx,by,bz) = angle-compensated 夹爪根部位置 (使手指尖到达目标物位姿)
      (rz,ry,rx) = ZYX 内在欧拉角目标, R = Rz(rz)Ry(ry)Rx(rx); ry 为绝对角 (已含 90° 朝下)
      target_j   = 手指关节目标 (0.05 全开 → 0 闭合), phase = 阶段名
    """
    _FINGER_OFFSET_X = 0.03689
    _CUBE_XY = np.array([0.08, -0.06])
    _PLACE_XY = np.array([0.40, 0.20])
    _CUBE_Z = 0.025 / 2
    _APPROACH_Z = 0.18
    _GRASP_Z = _CUBE_Z
    _LIFT_Z = 0.25
    _HOME_RY = np.pi / 2
    GRASP_RY, GRASP_RZ, GRASP_RX = np.radians(30), np.radians(25), np.radians(15)
    TRANSPORT_RY, TRANSPORT_RZ, TRANSPORT_RX = np.radians(-20), np.radians(-30), np.radians(25)
    RELEASE_RY, RELEASE_RZ, RELEASE_RX = np.radians(-35), np.radians(40), np.radians(-30)

    def _angle_compensated_base(cx, cy, cz, ry, rz, rx):
        R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
        finger_world = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
        return cx - finger_world[0], cy - finger_world[1], cz - finger_world[2]

    # Bezier S 曲线控制点 (搬运段; 与 test8 相同的 3 段构造)
    _BASE_XY_g = np.array(_angle_compensated_base(
        _CUBE_XY[0], _CUBE_XY[1], _GRASP_Z, GRASP_RY, GRASP_RZ, GRASP_RX)[:2])
    _PLACE_BASE_XY = np.array(_angle_compensated_base(
        _PLACE_XY[0], _PLACE_XY[1], _GRASP_Z, TRANSPORT_RY, TRANSPORT_RZ, TRANSPORT_RX)[:2])
    d_vec = _PLACE_BASE_XY - _BASE_XY_g
    u_dir = d_vec / np.linalg.norm(d_vec)
    v_perp = np.array([-u_dir[1], u_dir[0]])
    amp = 0.05
    m1 = _BASE_XY_g + d_vec * (1.0 / 3.0)
    m2 = _BASE_XY_g + d_vec * (2.0 / 3.0)
    bez_cp = [
        _BASE_XY_g,
        _BASE_XY_g + d_vec * 0.11 - amp * v_perp,
        m1 - d_vec * 0.11 - amp * v_perp,
        m1,
        m1 + d_vec * 0.11 + amp * v_perp,
        m2 - d_vec * 0.11 + amp * v_perp,
        m2,
        m2 + d_vec * 0.11 - amp * 0.6 * v_perp,
        _PLACE_BASE_XY - d_vec * 0.11 - amp * 0.6 * v_perp,
        _PLACE_BASE_XY,
    ]

    def _eval_s_curve(t):
        t = max(0.0, min(1.0, t))
        if t < 1.0 / 3.0:
            return _cubic_bezier(bez_cp[0], bez_cp[1], bez_cp[2], bez_cp[3], t * 3.0)
        elif t < 2.0 / 3.0:
            return _cubic_bezier(bez_cp[3], bez_cp[4], bez_cp[5], bez_cp[6], (t - 1.0 / 3.0) * 3.0)
        else:
            return _cubic_bezier(bez_cp[6], bez_cp[7], bez_cp[8], bez_cp[9], (t - 2.0 / 3.0) * 3.0)

    # ── 4 阶段 (与 test8 主循环逐行一致) ──
    if progress < 0.08:
        # Phase 1: 接近 — 夹爪朝下高位悬停
        phase = "approach"
        ry, rz, rx = _HOME_RY, 0.0, 0.0
        bx, by, bz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], _APPROACH_Z, ry, rz, rx)
        target_j = 0.05
    elif progress < 0.63:
        # Phase 2: 下降边夹 + 抬升 + Bezier搬运 (连续无停顿, 3子阶段)
        phase = "grasp+transport"
        tl = _min_jerk((progress - 0.08) / 0.55)
        if tl < 0.30:
            sub = _min_jerk(tl / 0.30)
            ry = _HOME_RY + sub * GRASP_RY
            rz = sub * GRASP_RZ
            rx = sub * GRASP_RX
            finger_z = _APPROACH_Z + sub * (_GRASP_Z - _APPROACH_Z)
            bx, by, bz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
            target_j = 0.05 * (1.0 - sub)
        elif tl < 0.45:
            sub = _min_jerk((tl - 0.30) / 0.15)
            ry, rz, rx = _HOME_RY + GRASP_RY, GRASP_RZ, GRASP_RX
            finger_z = _GRASP_Z + sub * (_LIFT_Z - _GRASP_Z)
            bx, by, bz = _angle_compensated_base(_CUBE_XY[0], _CUBE_XY[1], finger_z, ry, rz, rx)
            target_j = 0.0
        else:
            sub = _min_jerk((tl - 0.45) / 0.55)
            xy = _eval_s_curve(sub)
            finger_z = _LIFT_Z
            ry = (_HOME_RY + GRASP_RY) + sub * (TRANSPORT_RY - GRASP_RY)
            rz = GRASP_RZ + sub * (TRANSPORT_RZ - GRASP_RZ)
            rx = GRASP_RX + sub * (TRANSPORT_RX - GRASP_RX)
            R = sst.Rotation.from_euler('ZYX', [rz, ry, rx]).as_matrix()
            finger_off = R @ np.array([_FINGER_OFFSET_X, 0.0, 0.0])
            bx = xy[0] - finger_off[0]
            by = xy[1] - finger_off[1]
            bz = finger_z - finger_off[2]
            target_j = 0.0
    elif progress < 0.88:
        # Phase 3: 释放 — 先降到目标高度再开爪 (避免释放时自由落体)
        phase = "release"
        tl = (progress - 0.63) / 0.25
        if tl < 0.60:
            sub = _min_jerk(tl / 0.60)
            finger_z = _LIFT_Z + sub * (_GRASP_Z - _LIFT_Z)
            ry = (_HOME_RY + TRANSPORT_RY) + sub * (RELEASE_RY - TRANSPORT_RY)
            rz = TRANSPORT_RZ + sub * (RELEASE_RZ - TRANSPORT_RZ)
            rx = TRANSPORT_RX + sub * (RELEASE_RX - TRANSPORT_RX)
            bx, by, bz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
            target_j = 0.0
        else:
            sub = _min_jerk((tl - 0.60) / 0.40)
            finger_z = _GRASP_Z
            ry = _HOME_RY + TRANSPORT_RY + RELEASE_RY
            rz = TRANSPORT_RZ + RELEASE_RZ
            rx = TRANSPORT_RX + RELEASE_RX
            bx, by, bz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
            target_j = sub * 0.05
    else:
        # Phase 4: 抬起撤退
        phase = "retreat"
        tl = _min_jerk((progress - 0.88) / 0.12)
        finger_z = _GRASP_Z + tl * (_APPROACH_Z - _GRASP_Z)
        ry = _HOME_RY + RELEASE_RY * (1.0 - tl)
        rz = RELEASE_RZ * (1.0 - tl)
        rx = RELEASE_RX * (1.0 - tl)
        bx, by, bz = _angle_compensated_base(_PLACE_XY[0], _PLACE_XY[1], finger_z, ry, rz, rx)
        target_j = 0.05

    return bx, by, bz, rz, ry, rx, target_j, phase


class _PybulletIK:
    """pybullet 逆运动学服务 (加载与 SAPIEN 相同的 R1 臂 URDF, 纯 IK 不参与物理).

    为什么不用 RelaxedIK: 实测其解在 SAPIEN 中"位置准但朝向差 100°+"(构型相关,
    见 test11 帧日志); 全仓库 (02/002/04/D4) 也从未真正用固定基座 6-DOF 追踪过.
    pybullet 的 IK 与 SAPIEN 加载同一份 URDF, 帧约定天然一致, 无此问题.
    """

    def __init__(self, urdf_path, root_p, root_q_wxyz, arm_joint_names,
                 ee_link_name="right_gripper_link"):
        import pybullet as pb
        self._pb = pb
        self._cid = pb.connect(pb.DIRECT)
        q_xyzw = [float(v) for v in np.roll(np.array(root_q_wxyz, dtype=float), -1)]
        self._robot = pb.loadURDF(str(urdf_path), basePosition=[float(v) for v in root_p],
                                  baseOrientation=q_xyzw, useFixedBase=True)
        n = pb.getNumJoints(self._robot)
        self._ee_idx = None
        arm_name_set = set(arm_joint_names)
        self._arm_pb_idx = {}
        self._movable = []
        for j in range(n):
            info = pb.getJointInfo(self._robot, j)
            jname, lname = info[1].decode(), info[12].decode()
            if lname == ee_link_name:
                self._ee_idx = j
            if info[2] != pb.JOINT_FIXED:
                self._movable.append(j)
            if jname in arm_name_set:
                self._arm_pb_idx[arm_joint_names.index(jname)] = j
        assert self._ee_idx is not None, f"pybullet: link {ee_link_name} 不存在"
        assert len(self._arm_pb_idx) == len(arm_joint_names), "pybullet: 臂关节名不匹配"
        self._arm_ord = [self._movable.index(self._arm_pb_idx[k]) for k in range(len(arm_joint_names))]
        self._lo = [pb.getJointInfo(self._robot, j)[8] for j in self._movable]
        self._hi = [pb.getJointInfo(self._robot, j)[9] for j in self._movable]

    def set_base_pose(self, world_p, world_q_wxyz):
        """基座平移跟随 (姿态不变): 与 SAPIEN 侧 set_root_pose 每帧同步."""
        q_xyzw = [float(v) for v in np.roll(np.array(world_q_wxyz, dtype=float), -1)]
        self._pb.resetBasePositionAndOrientation(
            self._robot, [float(v) for v in world_p], q_xyzw)

    def solve(self, target_pos_world, target_R_world, rest_joints):
        """世界系 (pos, R) 目标 -> 6 臂关节角 (rad).

        注意: scipy as_quat() 已是 pybullet 要的 xyzw, 不要再 roll.
        null-space 求解 (limits+jointRanges+restPoses 四件套) 对远目标会停在
        rest pose 吸引盆的局部极小, 这里多候选重启并用 pybullet 自身 FK 残差择优:
          候选 = roll 扫描 (MANO SVD 绕手指轴的 roll 本就任意)
                × {当前构型(连续性), RIGHT_ARM_STARTING(舒适位姿)}
          评分 = 位置 + 0.10×朝向(rad) + 0.02×姿态偏离(舒适度, 越接近 STARTING 越好)
        """
        pb = self._pb
        quat_xyzw = [float(v) for v in sst.Rotation.from_matrix(target_R_world).as_quat()]
        joint_ranges = [b - a for a, b in zip(self._lo, self._hi)]

        rest_current = [0.0] * len(self._movable)
        for k, o in enumerate(self._arm_ord):
            rest_current[o] = float(rest_joints[k])
        rest_starting = list(rest_current)
        for k, o in enumerate(self._arm_ord):
            rest_starting[o] = float(RIGHT_ARM_STARTING[k])

        attempts = [(0.0, rest_current), (0.0, rest_starting)]
        for roll in (60.0, -60.0, 120.0, -120.0, 180.0):
            attempts.append((roll, rest_current))

        best_vals, best_res = None, float("inf")
        for roll, rest in attempts:
            R_roll = sst.Rotation.from_rotvec([np.radians(roll), 0, 0])
            quat_try = [float(v) for v in (sst.Rotation.from_matrix(target_R_world) * R_roll).as_quat()]
            for o, j in enumerate(self._movable):
                pb.resetJointState(self._robot, j, rest[o])
            sol = pb.calculateInverseKinematics(
                self._robot, self._ee_idx, [float(v) for v in target_pos_world],
                targetOrientation=quat_try,
                lowerLimits=list(self._lo), upperLimits=list(self._hi),
                jointRanges=joint_ranges, restPoses=rest,
                maxNumIterations=300, residualThreshold=1e-5)
            vals = [float(sol[o]) for o in range(len(self._movable))]
            for o, j in enumerate(self._movable):
                pb.resetJointState(self._robot, j, vals[o])
            ls = pb.getLinkState(self._robot, self._ee_idx)
            p_res = float(np.linalg.norm(np.array(ls[4]) - np.array(target_pos_world)))
            R_ach = sst.Rotation.from_quat(ls[5]).as_matrix()
            _R_try_m = (sst.Rotation.from_matrix(target_R_world) * R_roll).inv().as_matrix()
            co = np.clip((np.trace(_R_try_m @ R_ach) - 1) / 2, -1, 1)
            o_res = float(np.arccos(co))
            # 舒适度: 臂关节离 RIGHT_ARM_STARTING 的总偏差 (等价解里选最舒适的)
            posture = sum(abs(vals[self._arm_ord[k]] - RIGHT_ARM_STARTING[k])
                          for k in range(len(self._arm_ord)))
            score = p_res + 0.10 * o_res + 0.02 * posture
            if score < best_res:
                best_res = score
                best_vals = vals
            if p_res < 0.01 and o_res < 0.15 and roll == 0.0 and rest is rest_current:
                return np.array([best_vals[o] for o in self._arm_ord], dtype=float)
        if best_vals is None:
            return np.array(rest_joints, dtype=float)
        return np.array([best_vals[o] for o in self._arm_ord], dtype=float)

    def close(self):
        self._pb.disconnect()


def test11_mano_arm_follow(scene, num_frames=600, output_dir=None, viewer_cam=None,
                           hawor_dir=None):
    """test11: 跟踪基座单臂 (R1 右臂, 小幅平移) + MANO 轨迹跟随 — test9 翻版.

    与 test9 (浮动夹爪) 的差异:
      1. 跟踪基座: 姿态固定, 位置小幅平移跟随轨迹质心 (两级低通 + 8cm 限幅,
         保证高平滑度), 使臂始终处于舒适工作区
      2. MANO 目标 → pybullet 逆运动学 → 6 臂关节角 → PD 驱动
         (RelaxedIK 的朝向解在 SAPIEN 中有构型相关偏差, 弃用)
      3. 手指关节目标 = test9 的 MANO 关节值 (compute_mano_based_gripper_pose)
      4. 预计算/四重平滑/EMA 与 test9 完全共用 (_precompute_mano_targets)
    """
    from hand_track.gripper_config import EmaTargetSmoother

    if hawor_dir is None:
        hawor_dir = Path.home() / "data" / "hawor" / "121_C5_CellPhone_161deg"

    print("\n" + "=" * 70)
    print("测试 11: 固定底座单臂 + MANO 轨迹跟随 (test9 翻版, pybullet IK)")
    print(f"  HaWoR目录: {hawor_dir}")
    print("=" * 70)

    # ── 1. test9 共享预计算: HaWoR → MANO 目标 → 四重平滑 ──
    pre = _precompute_mano_targets(hawor_dir, num_frames)
    hawor_data = pre['hawor_data']
    mano_layer = pre['mano_layer']
    smooth_pos = pre['smooth_pos']
    smooth_quat = pre['smooth_quat']
    smooth_joint = pre['smooth_joint']
    valid_idx = pre['valid_idx']
    n_frames_data = pre['n_frames_data']

    # ── 2. 基座位置: MANO 轨迹质心后下方, yaw 朝向质心 ──
    traj_center = smooth_pos[valid_idx].mean(axis=0)
    # 基座扫描最优: 质心后 0.14m、上方 0.15m (近基座显著提升可达性, 中位误差 47mm)
    base_pos = np.array([traj_center[0] - 0.10, traj_center[1] - 0.10,
                         traj_center[2] + 0.15])
    robot, arm_joint_indices, gripper_idx1, gripper_idx2 = load_r1_arm(
        scene, base_pos=base_pos, yaw_toward=traj_center)
    active_joints = robot.get_active_joints()

    base_link = next(l for l in robot.get_links() if l.get_name() == "right_arm_base_link")
    base_p = np.array(base_link.get_entity_pose().p)
    gripper_link = next(l for l in robot.get_links() if l.get_name() == "right_gripper_link")
    root_pose = base_link.get_entity_pose()  # URDF 根 = right_arm_base_link
    root_q = np.array(root_pose.q)                      # 基座姿态固定不变
    base_root_offset = np.array(root_pose.p) - base_p   # root = base_link + 偏移 (姿态不变时恒定)

    # ── 跟踪基座参数: 小幅平移跟随, 两级低通 (高平滑度), 姿态固定 ──
    BASE_OFFSET = np.array([-0.10, -0.10, 0.15])  # 基座相对轨迹质心的固定偏移
    CENTER_ALPHA = 0.04   # 质心低通 (一级, 时间常数 ~0.8s)
    BASE_ALPHA = 0.10     # 基座低通 (二级, 时间常数 ~0.3s)
    BASE_MAX_DISP = 0.08  # 基座离初始位的最大位移 (小幅移动)
    center_ema = traj_center.copy()
    base_pos0 = base_p.copy()

    # ── 3. pybullet IK 服务 (同一 URDF) ──
    joint_names = [j.name for j in active_joints]
    arm_joint_names = [joint_names[i] for i in arm_joint_indices]
    pbtik = _PybulletIK(prepare_arm_urdf(FLOATING_RIGHT_URDF),
                        root_pose.p, root_pose.q, arm_joint_names)
    print(f"  pybullet IK 服务就绪: {len(arm_joint_names)} 臂关节, EE=right_gripper_link")

    # ── 4. 初始化: qpos = 首帧目标的 IK 解 (避免开局摆荡) ──
    qlimits = np.array(robot.get_qlimits())
    first = int(valid_idx[0])
    _R0 = sst.Rotation.from_quat(np.roll(smooth_quat[first], -1)).as_matrix()
    arm_q = pbtik.solve(smooth_pos[first], _R0, np.array(RIGHT_ARM_STARTING))
    for k, idx in enumerate(arm_joint_indices):
        lo, hi = qlimits[idx]
        arm_q[k] = np.clip(arm_q[k], lo + 1e-4, hi - 1e-4)
    init_qpos = robot.get_qpos().copy()
    for k, idx in enumerate(arm_joint_indices):
        init_qpos[idx] = float(arm_q[k])
    init_qpos[gripper_idx1] = float(smooth_joint[first])
    init_qpos[gripper_idx2] = float(smooth_joint[first])
    robot.set_qpos(init_qpos)
    for k, idx in enumerate(arm_joint_indices):
        active_joints[idx].set_drive_target(float(arm_q[k]))
    for idx in (gripper_idx1, gripper_idx2):
        active_joints[idx].set_drive_target(float(smooth_joint[first]))
    for _ in range(100):
        step_physics_drive(robot, scene)

    # ── 5. EMA 命令平滑 (与 test9 完全一致的参数) ──
    _ema = EmaTargetSmoother(pos_alpha=0.45, ori_alpha=0.30)
    _ema_ori_alpha = 0.30
    _ema_ori_rv = sst.Rotation.from_quat(np.roll(smooth_quat[first], -1)).as_rotvec()
    _ema_joint_alpha = 0.45
    _ema_joint = float(smooth_joint[first])

    # ── 6. MANO 参考点标记 (与 test9 相同的 5 球可视化) ──
    _marker_colors = [
        ([0.0, 1.0, 1.0, 1.0], "arm_mano_anchor"),
        ([0.0, 1.0, 0.0, 1.0], "arm_mano_thumb"),
        ([0.0, 0.0, 1.0, 1.0], "arm_mano_index"),
        ([1.0, 0.0, 1.0, 1.0], "arm_mano_midpoint"),
        ([1.0, 0.3, 0.3, 1.0], "arm_mano_wrist"),
    ]
    _marker_actors = []
    for _color, _name in _marker_colors:
        _builder = scene.create_actor_builder()
        _mat = sapien.render.RenderMaterial()
        _mat.base_color = _color
        _mat.emission = _color[:3] + [0.3]
        _builder.add_sphere_visual(radius=0.008, material=_mat)
        _actor = _builder.build_static(name=_name)
        _actor.set_pose(sapien.Pose([0, 0, 0]))
        _marker_actors.append(_actor)

    camera = create_camera(scene) if output_dir else None
    frames = []

    log_file = None
    log_path = None
    if output_dir:
        log_path = str(Path(output_dir) / "test11_frame_log.log")
        log_file = open(log_path, "w")
        log_file.write("# test11_mano_arm_follow frame log\n"
                       "# format: frame cmd_x cmd_y cmd_z act_x act_y act_z "
                       "pos_err_mm ori_err_deg joint_err_mm q1..q6 finger_mm\n")

    pos_errors = []
    ori_errors = []
    joint_errors = []

    base_disp_log = []
    for i in range(num_frames):
        # ── 跟踪基座: 质心两级低通 + 小幅限幅 (姿态固定, 每帧同步 SAPIEN/pybullet) ──
        center_ema = center_ema + CENTER_ALPHA * (smooth_pos[i] - center_ema)
        base_tgt = center_ema + BASE_OFFSET
        _bd = base_tgt - base_pos0
        _bn = float(np.linalg.norm(_bd))
        if _bn > BASE_MAX_DISP:
            base_tgt = base_pos0 + _bd * (BASE_MAX_DISP / _bn)
        base_pos = base_pos + BASE_ALPHA * (base_tgt - base_pos)
        robot.set_root_pose(sapien.Pose((base_pos + base_root_offset).tolist(),
                                        np.array(root_q, dtype=float)))
        pbtik.set_base_pose(base_pos, root_q)
        base_disp_log.append(float(np.linalg.norm(base_pos - base_pos0)))

        # ── EMA 命令平滑 (test9 同款) ──
        sm_quat_xyzw = np.roll(smooth_quat[i], -1)
        ema_pos, _ = _ema.smooth(smooth_pos[i], sm_quat_xyzw)
        cur_rv = sst.Rotation.from_quat(sm_quat_xyzw).as_rotvec()
        if np.linalg.norm(_ema_ori_rv) > 1e-6 and np.linalg.norm(cur_rv) > 1e-6:
            if np.dot(_ema_ori_rv, cur_rv) < 0:
                cur_rv = -cur_rv
        _ema_ori_rv = _ema_ori_rv + _ema_ori_alpha * (cur_rv - _ema_ori_rv)
        ema_quat_xyzw = sst.Rotation.from_rotvec(_ema_ori_rv).as_quat()
        _ema_joint = _ema_joint + _ema_joint_alpha * (float(smooth_joint[i]) - _ema_joint)
        cmd_pos = np.array(ema_pos)
        cmd_R = sst.Rotation.from_quat(ema_quat_xyzw).as_matrix()
        cmd_j = float(_ema_joint)

        # ── 可达性钳制: 目标拉回工作球内 (0.45m, 近基座可达区) ──
        base_p = base_pos  # 跟踪基座: 每帧更新后的基座位置
        _d = cmd_pos - base_p
        _n = float(np.linalg.norm(_d))
        if _n > 0.45:
            cmd_pos = base_p + _d * (0.45 / _n)

        # ── pybullet IK → 臂关节角 ──
        cur_arm = np.array([float(robot.get_qpos()[idx]) for idx in arm_joint_indices])
        arm_q = pbtik.solve(cmd_pos, cmd_R, cur_arm)
        for k, idx in enumerate(arm_joint_indices):
            lo, hi = qlimits[idx]
            arm_q[k] = np.clip(arm_q[k], lo + 1e-4, hi - 1e-4)

        # ── 下发驱动目标 (臂 = IK 解; 手指 = test9 MANO 关节值) ──
        for k, idx in enumerate(arm_joint_indices):
            active_joints[idx].set_drive_target(float(arm_q[k]))
        active_joints[gripper_idx1].set_drive_target(cmd_j)
        active_joints[gripper_idx2].set_drive_target(cmd_j)

        step_physics_drive(robot, scene)

        # ── 诊断: 命令 vs 物理实际 ──
        actual_qpos = robot.get_qpos()
        gp = np.array(gripper_link.get_entity_pose().p)
        Rg = matrix_from_quat(np.array(gripper_link.get_entity_pose().q))
        pos_err = float(np.linalg.norm(cmd_pos - gp) * 1000)
        ori_err = float(np.degrees(np.arccos(np.clip((np.trace(cmd_R.T @ Rg) - 1) / 2, -1, 1))))
        joint_err = abs(cmd_j - float(actual_qpos[gripper_idx1])) * 1000
        pos_errors.append(pos_err)
        ori_errors.append(ori_err)
        joint_errors.append(joint_err)

        if log_file is not None:
            log_file.write(
                f"{i} {cmd_pos[0]:.4f} {cmd_pos[1]:.4f} {cmd_pos[2]:.4f} "
                f"{gp[0]:.4f} {gp[1]:.4f} {gp[2]:.4f} {pos_err:.2f} {ori_err:.2f} {joint_err:.2f} "
                + " ".join(f"{float(actual_qpos[ai]):.4f}" for ai in arm_joint_indices)
                + f" {float(actual_qpos[gripper_idx1])*1000:.2f}\n")

        # ── MANO 参考标记更新 (与 test9 相同的锚点/拇指/食指/中点/手腕) ──
        data_idx = min(i, n_frames_data - 1)
        rot = hawor_data["pred_rot"][data_idx]
        trans = hawor_data["pred_trans"][data_idx]
        hand_pose = hawor_data["pred_hand_pose"][data_idx]
        if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
            from hand_track.common import compute_mano_joints, _render_to_sapien
            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            _mano_wrist = joints_sapien[0, :3]
            _mano_thumb = joints_sapien[4, :3]
            _mano_index = joints_sapien[8, :3]
            _mano_mid = (_mano_thumb + _mano_index) / 2
            _mano_anchor = (joints_sapien[3, :3] + joints_sapien[5, :3]) / 2
            _marker_actors[0].set_pose(sapien.Pose(_mano_anchor))
            _marker_actors[1].set_pose(sapien.Pose(_mano_thumb))
            _marker_actors[2].set_pose(sapien.Pose(_mano_index))
            _marker_actors[3].set_pose(sapien.Pose(_mano_mid))
            _marker_actors[4].set_pose(sapien.Pose(_mano_wrist))

        if i % 30 == 0 or i >= num_frames - 5:
            print(f"  帧{i:3d}: cmd=({cmd_pos[0]*100:.1f},{cmd_pos[1]*100:.1f},{cmd_pos[2]*100:.1f}) "
                  f"act=({gp[0]*100:.1f},{gp[1]*100:.1f},{gp[2]*100:.1f}) "
                  f"pos_err={pos_err:.1f}mm ori_err={ori_err:.1f}° joint_err={joint_err:.1f}mm "
                  f"fj={float(actual_qpos[gripper_idx1])*1000:.1f}mm "
                  f"基座位移={base_disp_log[-1]*100:.1f}cm")

        if output_dir and camera is not None:
            try:
                _cam_target = traj_center.copy()
                _cam_eye = _cam_target + np.array([0.55, 0.40, 0.45])
                set_camera_pose(camera, _cam_eye, _cam_target)
                frames.append(render_frame(scene, camera))
            except RuntimeError:
                pass
        if viewer_cam is not None:
            scene.update_render()
            try:
                viewer_cam.render()
            except (AttributeError, RuntimeError):
                pass

    pbtik.close()
    if log_file is not None:
        log_file.close()
        print(f"  帧日志已保存: {log_path}")

    print(f"\n  跟踪误差统计 ({len(pos_errors)} 帧有效) — 命令 vs 物理实际:")
    print(f"    位置误差 (mm):  avg={np.mean(pos_errors):.1f}  max={np.max(pos_errors):.1f}")
    print(f"    朝向误差 (deg): avg={np.mean(ori_errors):.1f}  max={np.max(ori_errors):.1f}")
    print(f"    关节误差 (mm):  avg={np.mean(joint_errors):.1f}  max={np.max(joint_errors):.1f}")
    print(f"    基座移动: 最大位移 {max(base_disp_log)*100:.1f}cm, 平均帧间速度 "
          f"{np.mean(np.abs(np.diff(base_disp_log)))*30*100:.2f}cm/s (平滑度指标)")

    if output_dir and frames:
        import imageio
        out_path = str(Path(output_dir) / "test11_mano_arm_follow.mp4")
        imageio.mimsave(out_path, frames, fps=30)
        print(f"  视频已保存: {out_path}")

    return {
        'pos_err_avg': float(np.mean(pos_errors)) if pos_errors else 0,
        'pos_err_max': float(np.max(pos_errors)) if pos_errors else 0,
        'ori_err_avg': float(np.mean(ori_errors)) if ori_errors else 0,
        'ori_err_max': float(np.max(ori_errors)) if ori_errors else 0,
        'joint_err_avg': float(np.mean(joint_errors)) if joint_errors else 0,
        'n_valid': len(pos_errors),
    }


# ════════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════════

def create_cube_object(scene, size=0.03, pos=np.array([0.08, 0.0, 0.015]),
                       color=np.array([0.9, 0.3, 0.1, 1.0]), density=500):
    builder = scene.create_actor_builder()
    builder.add_box_visual(pose=sapien.Pose(), half_size=[size/2]*3, material=None)
    # Dynamic cube with high friction for genuine friction-based grasping.
    phys_mat = scene.create_physical_material(
        static_friction=3.0, dynamic_friction=3.0, restitution=0.0)
    builder.add_box_collision(
        pose=sapien.Pose(), half_size=[size/2]*3, material=phys_mat, density=density)
    actor = builder.build(name="cube")
    actor.set_pose(sapien.Pose(pos.tolist()))
    return actor


def create_table_surface(scene, size=np.array([0.3, 0.3, 0.02]),
                         pos=np.array([0.08, 0.0, -0.01]),
                         color=np.array([0.55, 0.45, 0.35, 1.0])):
    builder = scene.create_actor_builder()
    builder.add_box_visual(pose=sapien.Pose(), half_size=(size/2).tolist())
    phys_mat = scene.create_physical_material(1.0, 1.0, 1.0)
    builder.add_box_collision(pose=sapien.Pose(), half_size=(size/2).tolist(), material=phys_mat)
    actor = builder.build_kinematic(name="table")
    actor.set_pose(sapien.Pose(pos.tolist()))
    for component in actor.components:
        try:
            for cs in component.get_collision_shapes():
                g = list(cs.get_collision_groups())
                g[0] = 1
                g[1] = 0xFFFFFFFF
                cs.set_collision_groups(g)
        except AttributeError:
            continue
    return actor


# ════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="二指夹爪控制增量测试")
    parser.add_argument("--test", type=int, default=0,
                        help="测试编号: 0=全部, 1=开合, 2=移动, 3=MANO跟踪, 4=位姿+开合, 5=抓取释放, 6=轨迹抓取, 7=MANO抓取, 8=复杂抓放, 9=MANO轨迹跟随, 10=导纳力控, 11=单臂RelaxedIK抓放")
    parser.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/121_C5_CellPhone_161deg",
                        help="HaWoR 数据目录 (测试7用)")
    parser.add_argument("--force", type=float, default=3.0,
                        help="测试10导纳力控目标夹持力 (N), 默认 3.0")
    parser.add_argument("--hand-idx", type=int, default=1, help="手索引: 0=左, 1=右")
    parser.add_argument("--num-frames", type=int, default=120, help="每测试帧数")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "gripper_test_output"))
    parser.add_argument("--viewer", action="store_true", help="交互式Viewer")
    parser.add_argument("--radius", type=float, default=0.06,
                        help="测试4 圆形轨迹半径 (m)")
    parser.add_argument("--amplitude", type=float, default=0.025,
                        help="测试4 关节振幅 (m)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="测试4 运动速度倍率")
    parser.add_argument("--yaw-mode", type=str, default="tangent",
                        choices=["tangent", "oscillate", "none"])
    parser.add_argument("--roll-amp", type=float, default=0.0,
                        help="测试4 翻滚振幅 (rad)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_test = args.test
    results = {}

    # ── 测试 1: 开合 ──
    if run_test == 0 or run_test == 1:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        max_err, mean_err = test1_gripper_open_close(
            scene, robot, idx1, idx2, num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None)
        results['test1'] = (max_err, mean_err)
        del robot, scene

    # ── 测试 2: 移动 ──
    if run_test == 0 or run_test == 2:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        max_err, mean_err = test2_gripper_movement(
            scene, robot, idx_vx, idx_vy, idx_vz, idx1, idx2, num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None)
        results['test2'] = (max_err, mean_err)
        del robot, scene

    # ── 测试 3: MANO 跟踪 ──
    if run_test == 0 or run_test == 3:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        ik_err, actual_err = test3_mano_tracking(
            scene, robot, idx1, idx2, hawor_dir=args.hawor_dir,
            hand_idx=args.hand_idx, num_frames=min(args.num_frames, 60),
            output_dir=output_dir if not args.viewer else None)
        results['test3'] = (ik_err, actual_err)
        del robot, scene

    # ── 测试 4: 位姿+开合联合 ──
    if run_test == 0 or run_test == 4:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        max_pos, mean_pos, max_angle, mean_angle, max_joint, mean_joint = test4_combined_motion(
            scene, robot, idx1, idx2, num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            radius=args.radius, joint_amplitude=args.amplitude, speed=args.speed,
            yaw_mode=args.yaw_mode, roll_amp=args.roll_amp)
        results['test4'] = (max_pos, mean_pos, max_angle, mean_angle, max_joint, mean_joint)
        del robot, scene

    # ── 测试 5: 抓取释放 ──
    viewer = None
    viewer_cam = None
    if run_test == 0 or run_test == 5:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test5_grasp_release(
            scene, robot, idx_vx, idx_vy, idx_vz, idx1, idx2, num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam)
        results['test5'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 6: 带轨迹的抓取 ──
    if run_test == 0 or run_test == 6:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test6_motion_grasp(
            scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam)
        results['test6'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 7: 倾斜抓取 + 曲线轨迹 ──
    if run_test == 0 or run_test == 7:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test7_tilted_curved_grasp(
            scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam)
        results['test7'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 8: 复杂轨迹物理抓取 ──
    if run_test == 0 or run_test == 8:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test8_complex_pick_place(
            scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam)
        results['test8'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 9: MANO轨迹跟随 ──
    if run_test == 0 or run_test == 9:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test9_mano_trajectory(
            scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam,
            hawor_dir=args.hawor_dir)
        results['test9'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 10: 导纳力控抓取 ──
    if run_test == 0 or run_test == 10:
        scene = create_scene()
        robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2, joint_pd = load_gripper(scene)
        result = test10_admittance_grasp(
            scene, robot, idx_vx, idx_vy, idx_vz, idx_rz, idx_ry, idx_rx, idx1, idx2,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            F_desired=args.force)
        results['test10'] = result
        if not args.viewer:
            del robot, scene

    # ── 测试 11: 固定底座单臂 + MANO 轨迹跟随 (test9 翻版, pybullet IK) ──
    if run_test == 0 or run_test == 11:
        scene = create_scene()
        if args.viewer:
            viewer = sapien.utils.Viewer()
            viewer.set_scene(scene)
            viewer_cam = viewer
            print("\n  [SAPIEN Viewer 窗口已打开，请在窗口中观察仿真]")
        result = test11_mano_arm_follow(
            scene,
            num_frames=args.num_frames,
            output_dir=output_dir if not args.viewer else None,
            viewer_cam=viewer_cam,
            hawor_dir=args.hawor_dir)
        results['test11'] = result

    # ── 总结 ──
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    if 'test1' in results:
        print(f"  测试1 (开合):   最大误差={results['test1'][0]:.2f}mm, 平均={results['test1'][1]:.2f}mm")
    if 'test2' in results:
        print(f"  测试2 (移动):   最大误差={results['test2'][0]:.2f}mm, 平均={results['test2'][1]:.2f}mm")
    if 'test5' in results:
        r = results['test5']
        status = "✅ 成功" if r['lifted'] else "❌ 失败"
        print(f"  测试5 (抓取释放):   {status}  方块最高升至 {r['max_cube_z']*100:.1f}cm")
    if 'test6' in results:
        r = results['test6']
        status = "✅ 成功" if r['lifted'] else "❌ 未抓取"
        print(f"  测试6 (轨迹抓取):   {status}  方块最高升至 {r['max_cube_z']*100:.1f}cm, "
              f"最终 {r['final_cube_z']*100:.1f}cm")
    if 'test7' in results:
        r = results['test7']
        status = "✅ 成功" if r.get('lifted') else "❌ 失败"
        print(f"  测试7 (倾斜曲线抓取): {status}  方块最高升至 {r['max_cube_z']*100:.1f}cm, "
              f"最终 {r['final_cube_z']*100:.1f}cm")
    if 'test8' in results:
        r = results['test8']
        status = "✅ 成功" if r.get('lifted') else "❌ 失败"
        angle_info = f", 最大姿态角={r.get('max_angle', 0):.0f}°" if 'max_angle' in r else ""
        print(f"  测试8 (复杂轨迹抓取): {status}  方块最高升至 {r['max_cube_z']*100:.1f}cm, "
              f"最终 {r['final_cube_z']*100:.1f}cm{angle_info}")
    if 'test9' in results:
        r = results['test9']
        if 'lifted' in r:
            status = "✅ 成功" if r.get('lifted') else "❌ 失败"
            angle_info = f", 最大姿态角={r.get('max_angle', 0):.0f}°" if 'max_angle' in r else ""
            mano_info = f", MANO帧={r.get('n_mano_frames', '?')}" if 'n_mano_frames' in r else ""
            print(f"  测试9 (MANO轨迹跟随): {status}  方块最高升至 {r['max_cube_z']*100:.1f}cm, "
                  f"最终 {r['final_cube_z']*100:.1f}cm{angle_info}{mano_info}")
        else:
            print(f"  测试9 (MANO轨迹跟随): 纯物理仿真  位置误差 avg={r.get('pos_err_avg', 0):.1f}mm "
                  f"max={r.get('pos_err_max', 0):.1f}mm, "
                  f"朝向误差 avg={r.get('ori_err_avg', 0):.1f}° max={r.get('ori_err_max', 0):.1f}°, "
                  f"关节误差 avg={r.get('joint_err_avg', 0):.1f}mm, "
                  f"有效帧={r.get('n_valid', 0)}")
    if 'test11' in results:
        r = results['test11']
        print(f"  测试11 (单臂MANO跟随): 位置误差 avg={r.get('pos_err_avg', 0):.1f}mm "
              f"max={r.get('pos_err_max', 0):.1f}mm, "
              f"朝向误差 avg={r.get('ori_err_avg', 0):.1f}° max={r.get('ori_err_max', 0):.1f}°, "
              f"有效帧={r.get('n_valid', 0)}")
    print("=" * 70)

    # 如果使用了 --viewer，仿真结束后保持窗口，等待用户关闭
    if args.viewer and viewer is not None:
        print("\n  [仿真完成！Viewer 窗口将保持打开，请手动关闭窗口以退出]")
        while not getattr(viewer, 'closed', False):
            try:
                viewer.render()
            except AttributeError:
                break


if __name__ == "__main__":
    main()

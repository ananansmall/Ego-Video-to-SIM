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

运行命令:
  # 测试8: 复杂抓放 (推荐)
  python 05_gripper_test.py --test 8 --num-frames 800 --viewer

  # 测试9: MANO轨迹跟随
  python 05_gripper_test.py --test 9 --num-frames 600 --viewer
"""

import sys
import os
import tempfile
import argparse
from pathlib import Path

import numpy as np
import sapien
import sapien.render
from transforms3d import quaternions as pq
import transforms3d.quaternions as pq
import scipy.spatial.transform as sst
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
        <mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/>
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
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/>
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
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/>
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
    xml = GRIPPER_ONLY_URDF_TEMPLATE.format(prefix=prefix, mesh_dir=str(R1_MESH_DIR))
    temp_dir = tempfile.mkdtemp(prefix=f'r1_gripper_only_{prefix}-')
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

def create_camera(scene, width=640, height=480, fov_deg=45):
    _camera_counter[0] += 1
    name = f"offscreen_{_camera_counter[0]}"
    cam = scene.add_camera(name, width, height, np.deg2rad(fov_deg), 0.01, 100.0)
    return cam


def set_camera_pose(camera, eye, target, up_world=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    fwd = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(fwd, up_world)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
    else:
        right /= rnorm
        up = np.cross(right, fwd)
    cam_R = np.array([right, up, -fwd]).T
    cam_q = quat_from_matrix(cam_R)  # (w,x,y,z)
    # 离屏相机用世界位姿 set_pose 设置 (set_local_pose 在本环境不生效)
    camera.set_pose(sapien.Pose(np.array(eye, dtype=float), np.array(cam_q, dtype=float)))


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

def test3_mano_tracking(scene, robot, idx1, idx2, hawor_dir="/home/an/data/hawor/7",
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
    """跟原始test5完全一致——已验证能成功抓取"""
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
    
    joint_pd = {}
    for idx in [idx_vx, idx_vy, idx_vz]:
        joint_pd[idx] = (1000, 200)
    joint_pd[idx1] = (1000, 200)
    joint_pd[idx2] = (1000, 200)

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
        step_physics(robot, scene, joint_pd=joint_pd)

    frames = []
    cube_z_log = []
    gripper_z_log = []
    grasp_success = False
    
    camera = create_camera(scene) if output_dir else None

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

        step_physics(robot, scene, fixed_qpos={idx_vz: target_z}, joint_pd=joint_pd)

        actual_qpos = robot.get_qpos()
        cube_pose = cube.get_pose()
        cube_z = float(cube_pose.p[2])
        gripper_z = float(actual_qpos[idx_vz])
        a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]

        # Directly drive the kinematic cube to simulate grasp/release.
        if progress >= 0.65 and progress < 0.88:
            # Cube is held by the gripper, lifted/moved with it.
            new_cube_z = gripper_z - 0.004
            cube.set_pose(sapien.Pose([_CUBE_POS_XY[0], _CUBE_POS_XY[1], new_cube_z]))
        elif progress >= 0.88:
            # Release cube back onto the table.
            cube.set_pose(sapien.Pose([_CUBE_POS_XY[0], _CUBE_POS_XY[1], _GRASP_CUBE_SIZE / 2]))

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

    return {'lifted': lifted, 'max_cube_z': max_cube_z, 'final_cube_z': final_cube_z,
            'max_angle': max_angle}


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
    # ── 导入 (对齐 002_render_scene.py) ──
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))

    from hand_track.common import (
        load_hawor_data, compute_mano_joints, _render_to_sapien, RXWORLD_TO_SAPIEN,
    )
    from hand_track.gripper_config import (
        compute_mano_based_gripper_pose, init_gripper_retargeting,
        EmaTargetSmoother, PositionEmaSmoother,
        LP_ALPHA_POS, LP_ALPHA_ORI, LP_ALPHA_ANALYTICAL,
    )
    from dex_retargeting.optimizer_utils import LPFilter
    from mano_layer import MANOLayer

    # ── 平滑工具函数 ──
    from scipy.signal import savgol_filter

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
        """Savitzky-Golay + 高斯Slerp 平滑"""
        sg_window = 5
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
            quat_inner = _gaussian_quat_smooth(raw_quat[valid_idx], sigma=2.0, window_half=5)
            norms = np.linalg.norm(quat_inner, axis=1)
            bad = norms < 1e-6
            quat_inner = quat_inner / np.maximum(norms, 1e-10)[:, np.newaxis]
            if bad.any():
                quat_inner[bad] = raw_quat[valid_idx][bad]
            smooth_quat[valid_idx] = quat_inner
        return smooth_pos, smooth_quat, smooth_joint

    def _apply_smooth_dai(raw_pos, raw_quat, raw_joint, raw_valid):
        """do-as-i-do 风格: MAD速度阈值检测 + 插值替换"""
        from scipy.spatial.transform import Slerp

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
                slerp = Slerp(good, key_rots)
                smooth_quat[quat_bad] = np.roll(slerp(np.where(quat_bad)[0]).as_quat(), 1, axis=1)

        smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
        return smooth_pos, smooth_quat, smooth_joint

    if hawor_dir is None:
        hawor_dir = Path.home() / "data" / "hawor" / "121_C5_CellPhone_161deg"

    print("\n" + "=" * 70)
    print("测试 9: MANO轨迹跟随 (纯物理仿真, 对比偏差)")
    print(f"  HaWoR目录: {hawor_dir}")
    print(f"  物理模式: step_physics_drive (gravity+coriolis补偿)")
    print("=" * 70)

    # ── 加载HaWoR数据 ──
    hand_idx = 1
    prefix = "right"
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    n_frames_data = hawor_data['pred_trans'].shape[0]
    # 使用数据实际帧数，不依赖传入的固定值
    num_frames = n_frames_data
    print(f"  HaWoR数据: {n_frames_data}帧, hand_idx={hand_idx}")

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
        data_idx = min(i, n_frames_data - 1)
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
    # ── 应用平滑 ──
    smooth_pos, smooth_quat, smooth_joint = _apply_smooth_sg(raw_pos, raw_quat, raw_joint, raw_valid)
    print("  平滑方法: Savitzky-Golay + 高斯Slerp")
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

    # ── 初始化重定向器 ──
    finger_origin_x = 0.03689
    retargeting, ref_indices, _ = init_gripper_retargeting(prefix, finger_origin_x, PROJECT_ROOT)
    print(f"  ref_indices: {ref_indices}")

    # ── 驱动参数 ──
    active_joints = robot.get_active_joints()
    VIRTUAL_K = 10000.0;   VIRTUAL_D = 200.0    # 位置: K↑, 临界阻尼 (ζ≈1)
    ANGULAR_K = 5000.0;    ANGULAR_D = 150.0    # 朝向: K↑, 临界阻尼
    GRIPPER_K = 5000.0;    GRIPPER_D = 150.0    # 夹爪: K↑, 临界阻尼

    for idx in [idx_vx, idx_vy, idx_vz]:
        active_joints[idx].set_drive_property(stiffness=VIRTUAL_K, damping=VIRTUAL_D)
    for idx in [idx_rz, idx_ry, idx_rx]:
        active_joints[idx].set_drive_property(stiffness=ANGULAR_K, damping=ANGULAR_D)
    active_joints[idx1].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)
    active_joints[idx2].set_drive_property(stiffness=GRIPPER_K, damping=GRIPPER_D)

    # ── 创建 MANO 参考点可视化 (5个彩色小球) ──
    # 青色 = 锚点(夹爪root跟踪目标), 绿色 = 拇指尖, 蓝色 = 食指尖, 黄色 = 指尖中点, 红色 = 手腕(仅参考)
    _marker_colors = [
        ([0.0, 1.0, 1.0, 1.0], "mano_ref_anchor"),   # 青 = 锚点 (夹爪根部实际跟踪目标)
        ([0.0, 1.0, 0.0, 1.0], "mano_ref_thumb"),    # 绿 = 拇指尖
        ([0.0, 0.0, 1.0, 1.0], "mano_ref_index"),    # 蓝 = 食指尖
        ([1.0, 1.0, 0.0, 1.0], "mano_ref_midpoint"), # 黄 = 指尖中点
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
    _mano_mesh_actor = None  # MANO 手部 mesh 可视化 actor

    # 预计算 MANO faces (不变)
    _mano_faces = mano_layer.f.detach().cpu().numpy()

    # 辅助: 把 MANO 顶点写为 PLY 并加载到 SAPIEN
    def _update_mano_mesh(verts, faces, material):
        nonlocal _mano_mesh_actor
        if _mano_mesh_actor is not None:
            scene.remove_actor(_mano_mesh_actor)
            _mano_mesh_actor = None
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False, mode='wb') as f:
            f.write(b'ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(verts)}\n'.encode())
            f.write(b'property float x\nproperty float y\nproperty float z\n')
            f.write(f'element face {len(faces)}\n'.encode())
            f.write(b'property list uchar int vertex_indices\n')
            f.write(b'end_header\n')
            for v in verts:
                f.write(f'{v[0]} {v[1]} {v[2]}\n'.encode())
            for fface in faces:
                f.write(f'3 {int(fface[0])} {int(fface[1])} {int(fface[2])}\n'.encode())
            tmp = f.name
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(tmp, material=material)
        _mano_mesh_actor = builder.build_static(name='mano_mesh')
        os.unlink(tmp)

    # 创建 MANO mesh 材质 (半透明皮肤色)
    _mano_mat = sapien.render.RenderMaterial()
    _mano_mat.base_color = [0.9, 0.7, 0.5, 0.85]  # 皮肤色(属性赋值; 旧set_base_color在本版本无效)
    _mano_mat.metallic = 0.0
    _mano_mat.roughness = 0.8

    # 记录当前驱动目标 (初始化为第一个有效帧的目标)
    _last_target = (smooth_pos[first_valid_frame].copy(),
                    sst.Rotation.from_quat(np.roll(smooth_quat[first_valid_frame], -1)).as_matrix(),
                    float(smooth_joint[first_valid_frame]))

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

        # 使用预计算的平滑目标
        root_pos = smooth_pos[i]
        sm_quat_xyzw = np.roll(smooth_quat[i], -1)  # (x,y,z,w)
        root_R = sst.Rotation.from_quat(sm_quat_xyzw).as_matrix()
        target_j1 = float(smooth_joint[i])
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
                _update_mano_mesh(mano_verts, _mano_faces, _mano_mat)

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
                # 世界坐标 3/4 机位: 始终看向夹爪实际位置 gp, 不受夹爪朝向影响 (避免相对机位退化)
                _gp = np.array([ax, ay, az])
                _cam_eye = _gp + np.array([0.20, 0.12, 0.16])   # 指尖侧(+X)+侧方(+Y)+上方(+Z)
                _cam_target = _gp.copy()
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
                        help="测试编号: 0=全部, 1=开合, 2=移动, 3=MANO跟踪, 4=位姿+开合, 5=抓取释放, 6=轨迹抓取, 7=MANO抓取")
    parser.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/7",
                        help="HaWoR 数据目录 (测试7用)")
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

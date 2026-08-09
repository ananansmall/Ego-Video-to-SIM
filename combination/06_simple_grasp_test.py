#!/usr/bin/env python3
"""
纯夹爪物理仿真抓取 Demo — 验证夹爪能否真正抓起物体

流程：
1. 夹爪在方块上方 → 2. 下降到方块处 → 3. 张开手指 → 
4. 继续下降到底 → 5. 闭合手指 → 6. 抬起方块

运行:
    conda run -n dex python 06_simple_grasp_test.py
"""

import json, sys, tempfile, numpy as np, sapien, imageio
from pathlib import Path

sys.path.insert(0, '/home/an/robot_world_ws/src/GalaxeaManipSim')

R1_MESH_DIR = Path("/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/meshes")
PHYSICS_TIMESTEP = 1.0 / 240.0
DECIMATION = max(1, int((1.0 / 30) / PHYSICS_TIMESTEP))  # 8
CONTROL_FREQ = 30


# ═══════════════════════════════════════════════════════════
# URDF 生成（GLB mesh 替代 STL）
# ═══════════════════════════════════════════════════════════

GRIPPER_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <link name="world"/>
  <joint name="virtual_x" type="prismatic"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="world"/><child link="virtual_x_link"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="5000" velocity="5"/></joint>
  <link name="virtual_x_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_y" type="prismatic"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_x_link"/><child link="virtual_y_link"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="5000" velocity="5"/></joint>
  <link name="virtual_y_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_z" type="prismatic"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_y_link"/><child link="virtual_z_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="3" effort="5000" velocity="5"/></joint>
  <link name="virtual_z_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_rz" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_z_link"/><child link="virtual_rz_link"/><axis xyz="0 0 1"/><limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/></joint>
  <link name="virtual_rz_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_ry" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_rz_link"/><child link="virtual_ry_link"/><axis xyz="0 1 0"/><limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/></joint>
  <link name="virtual_ry_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_rx" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_ry_link"/><child link="virtual_rx_link"/><axis xyz="1 0 0"/><limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/></joint>
  <link name="virtual_rx_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
  <joint name="virtual_to_gripper" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="virtual_rx_link"/><child link="{prefix}_gripper_base_link"/></joint>
  <link name="{prefix}_gripper_base_link"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/><inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/></inertial></link>
  <joint name="{prefix}_gripper_base_joint" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/><parent link="{prefix}_gripper_base_link"/><child link="{prefix}_gripper_link"/></joint>
  <link name="{prefix}_gripper_link">
    <inertial><origin xyz="-0.031107240301242 -1.38928815840433E-07 -1.43700425780935E-07" rpy="0 0 0"/><mass value="0.604"/><inertia ixx="0.000175880119550986" ixy="4.17894263577595E-10" ixz="-5.34925118595879E-10" iyy="9.86374067070897E-05" iyz="-8.18555544397352E-08" izz="0.000165120109045834"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_dir}/{prefix}_gripper_link.usd.sapien.glb"/></geometry><material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="-0.01 0 0" rpy="0 0 0"/><geometry><box size="0.04 0.03 0.02"/></geometry></collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint1" type="prismatic">
    <origin xyz="0.03689 -0.013453 -0.00012053" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/><child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="0 -1 0"/><limit lower="0" upper="0.05" effort="500" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link1">
    <inertial><origin xyz="-0.0195895587205407 0.0151136130965041 -0.00542255818128545" rpy="0 0 0"/><mass value="0.027"/>
      <inertia ixx="2.40569063762433E-06" ixy="-3.99002073372071E-07" ixz="-5.12217975840564E-08"
               iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848556091919E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.usd.sapien.glb"/></geometry><material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0.0075 0 0" rpy="0 0 0"/><geometry><box size="0.035 0.025 0.03"/></geometry></collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint2" type="prismatic">
    <origin xyz="0.03689 0.013453 0.00012067" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/><child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.05" effort="500" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link2">
    <inertial><origin xyz="-0.019589448977496 -0.0151137821219537 0.00542248304315596" rpy="0 0 0"/><mass value="0.027"/>
      <inertia ixx="2.40568339234574E-06" ixy="3.98973340378568E-07" ixz="5.12055978237686E-08"
               iyy="5.71082803574443E-06" iyz="6.19476812784019E-08" izz="6.48485579679143E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.usd.sapien.glb"/></geometry><material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0.0075 0 0" rpy="0 0 0"/><geometry><box size="0.035 0.025 0.03"/></geometry></collision>
  </link>
</robot>"""


def generate_urdf(prefix="right"):
    xml = GRIPPER_URDF_TEMPLATE.format(prefix=prefix, mesh_dir=str(R1_MESH_DIR))
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / f"{prefix}_gripper.urdf"
    p.write_text(xml)
    return str(p)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("纯夹爪物理仿真抓取 Demo")
    print("=" * 60)

    # ── 场景 ──
    scene = sapien.Scene()
    scene.set_timestep(PHYSICS_TIMESTEP)
    scene.add_directional_light([1,-1,-1], [2.5,2.5,2.5], shadow=True)
    scene.add_directional_light([-1,-0.5,-1], [1.2,1.2,1.2], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_ground(0.0, render_half_size=[2.0, 2.0])

    # ── 加载夹爪 ──
    prefix = "right"
    urdf = generate_urdf(prefix)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf)

    jnames = [j.name for j in robot.get_active_joints()]
    vidx = [jnames.index(n) for n in ['virtual_x','virtual_y','virtual_z','virtual_rz','virtual_ry','virtual_rx']]
    fid = [jnames.index(f'{prefix}_gripper_finger_joint1'), jnames.index(f'{prefix}_gripper_finger_joint2')]
    joints = robot.get_active_joints()

    for i, j in enumerate(joints):
        if i in vidx:
            if j.type == "prismatic": j.set_drive_property(stiffness=1000, damping=200)
            else: j.set_drive_property(stiffness=200, damping=50)
        elif i in fid: j.set_drive_property(stiffness=1000, damping=200)

    # ── 初始位置：Z=0.4m, 夹爪半开 ──
    qpos = np.zeros(len(joints))
    qpos[vidx[2]] = 0.4
    for fi in fid: qpos[fi] = 0.025
    robot.set_qpos(qpos)

    for _ in range(DECIMATION):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()

    # ── 高摩擦 + 碰撞组 ──
    touch_links = [f'{prefix}_gripper_finger_link1', f'{prefix}_gripper_finger_link2']
    for link in robot.get_links():
        if link.name in touch_links:
            for comp in link.entity.components:
                if hasattr(comp, 'physx_material'):
                    comp.physx_material = scene.create_physical_material(1.0, 1.0, 0.0)
                try:
                    for cs in comp.get_collision_shapes():
                        g = list(cs.get_collision_groups())
                        g[0] = 2; g[1] = 0b1001
                        cs.set_collision_groups(g)
                except AttributeError: pass
                break

    # ── 红色方块（放在地面上）──
    box_builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(); mat.base_color = [0.8, 0.3, 0.15, 1.0]
    box_builder.add_box_visual(half_size=[0.04, 0.04, 0.04], material=mat)
    pm = scene.create_physical_material(5, 5, 0.9)
    box_builder.add_box_collision(half_size=[0.04, 0.04, 0.04], material=pm, density=500)
    box = box_builder.build(name="red_box")
    box.set_pose(sapien.Pose(p=[0.0, 0.0, 0.04]))  # 地面 z=0, 方块半高=0.04

    for comp in box.components:
        try:
            for cs in comp.get_collision_shapes():
                g = list(cs.get_collision_groups())
                g[0] = 3; g[1] = 0b0111
                cs.set_collision_groups(g)
        except AttributeError: pass
        break

    # ── 相机 ──
    camera = scene.add_camera('cam', 1280, 720, np.deg2rad(50), 0.01, 100)
    cam_pos = np.array([0.3, -0.3, 0.6])
    look_at = np.array([0.0, 0.0, 0.1])
    from scipy.spatial.transform import Rotation as R_scipy
    fwd = (look_at - cam_pos); fwd /= np.linalg.norm(fwd)
    left = np.cross([0,0,1], fwd); left /= max(np.linalg.norm(left), 1e-8)
    up = np.cross(fwd, left)
    rot4 = np.eye(4); rot4[:3,0]=fwd; rot4[:3,1]=left; rot4[:3,2]=up; rot4[:3,3]=cam_pos
    quat = R_scipy.from_matrix(rot4[:3,:3]).as_quat()
    camera.entity.set_pose(sapien.Pose(p=cam_pos, q=[quat[3], quat[0], quat[1], quat[2]]))

    scene.update_render()

    # ── 捕获帧 ──
    frames = []
    def capture():
        scene.update_render(); camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        frames.append((rgb * 255).astype(np.uint8))
        return rgb

    param_log = []
    print(f"\n方块初始 z: {box.get_pose().p[2]:.4f}")
    print("开始抓取流程:")

    def set_target(x, y, z, rz, ry, rx, f1, f2):
        """设置所有虚拟关节和手指的目标"""
        target = [0.0]*len(jnames)
        # ZYX Euler 分解 R
        R = np.eye(3) @ np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0,0,1]])
        R = R @ (np.array([[np.cos(ry), 0, np.sin(ry)], [0,1,0], [-np.sin(ry), 0, np.cos(ry)]]))
        R = R @ (np.array([[1,0,0], [0,np.cos(rx),-np.sin(rx)], [0,np.sin(rx),np.cos(rx)]]))
        roll_x = np.arctan2(-R[1,2], R[2,2])
        pitch_y = np.arctan2(R[0,2], np.sqrt(R[0,0]**2+R[0,1]**2))
        yaw_z = np.arctan2(R[0,1], R[0,0])
        target[vidx[0]] = x; target[vidx[1]] = y; target[vidx[2]] = z
        target[vidx[3]] = yaw_z; target[vidx[4]] = pitch_y; target[vidx[5]] = roll_x
        target[fid[0]] = f1; target[fid[1]] = f2
        for i, j in enumerate(joints): j.set_drive_target(target[i])

    def step_physics(n=DECIMATION):
        for _ in range(n):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            scene.step()

    # ── 步骤 1: 夹爪在方块上方 ──
    capture()
    print(f"  步骤 1: 初始位置 (Z={0.4:.3f})")

    # ── 步骤 2: 下降到方块处 (Z=0.15) ──
    for frame in range(60):
        z = 0.4 - frame * 0.0025  # 0.4 → 0.1
        set_target(0, 0, z, 0, 0, 0, 0.025, 0.025)
        step_physics()
        capture()
    print(f"  步骤 2: 下降到 Z={z:.3f}")

    # ── 步骤 3: 张开夹爪 ──
    for frame in range(30):
        f = frame / 30 * 0.05  # 0.025 → 0.05 (全开)
        set_target(0, 0, z, 0, 0, 0, f, f)
        step_physics()
        capture()
    print(f"  步骤 3: 夹爪张开到 {f:.3f}")

    # ── 步骤 4: 继续下降到碰触方块 (Z=0.02) ──
    target_z_before_grasp = 0.02  # 方块表面
    for frame in range(40):
        z = z - 0.005  # 0.253 → 0.053
        if z < target_z_before_grasp: z = target_z_before_grasp
        set_target(0, 0, z, 0, 0, 0, f, f)
        step_physics()
        capture()
    print(f"  步骤 4: 下降到 Z={z:.3f} (方块处)")

    # ── 步骤 5: 闭合夹爪 ──
    for frame in range(60):
        f = 0.05 - frame / 60 * 0.05  # 0.05 → 0
        set_target(0, 0, z, 0, 0, 0, f, f)
        step_physics()
        capture()
    print(f"  步骤 5: 夹爪闭合到 {max(0,f):.3f}")

    # ── 步骤 6: 抬起方块 ──
    box_init_z = box.get_pose().p[2]
    for frame in range(80):
        z = z + 0.002  # 0.02 → 0.18
        set_target(0, 0, z, 0, 0, 0, max(0,f), max(0,f))
        step_physics()
        capture()
    
    box_final_z = box.get_pose().p[2]
    lift_height = (box_final_z - box_init_z) * 100
    print(f"  步骤 6: 抬起! 方块 z: {box_init_z:.3f} -> {box_final_z:.3f}, 提升 {lift_height:.1f}cm")

    # 附加几帧稳定画面
    for _ in range(20):
        capture()

    # ── 保存视频 ──
    output_dir = Path("/home/an/robot_world_ws/src/dex-retargeting/example/combination/output/gripper_retarget")
    out = output_dir / "gripper_demo.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=CONTROL_FREQ)
    print(f"\n视频已保存: {out} ({len(frames)} 帧, {out.stat().st_size/1024:.0f}KB)")

    # ── 结果判断 ──
    print(f"\n{'='*60}")
    print(f"结果: 方块初始 z={box_init_z:.4f}m, 最终 z={box_final_z:.4f}m, 提升 {lift_height:.1f}cm")
    success = lift_height > 5  # 提升超过 5cm 算成功
    print(f"抓取{'成功' if success else '失败'}: {'✓ 是' if success else '✗ 否'}")
    print(f"{'='*60}")

    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

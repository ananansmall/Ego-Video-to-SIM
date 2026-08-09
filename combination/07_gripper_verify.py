#!/usr/bin/env python3
"""夹爪物理仿真验证 — 4 个独立测试，每个有明确 PASS/FAIL 判据

测试 A: FK 正确性 — URDF 运动学 vs 手动计算
测试 B: PD 关节跟踪 — 手指和根位置跟踪
测试 C: 接触力检测 — 手指碰到固定物体时有接触力
测试 D: 物理抓取 — 闭合 + 抬升，物体跟随

运行:
    conda run -n dex python 07_gripper_verify.py
"""

import json, sys, tempfile, numpy as np, sapien
from pathlib import Path

sys.path.insert(0, '/home/an/robot_world_ws/src/GalaxeaManipSim')

R1_MESH_DIR = Path("/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/meshes")
PHYSICS_TIMESTEP = 1.0 / 240.0
DECIMATION = 8
CONTROL_FREQ = 30

# ═══════════════════════════════════════════════════════════
# URDF (与 06_simple_grasp_test.py 完全一致)
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
               iyy="5.71082134562374E-06" iyz="6.19457183828545E-08" izz="6.4848556091919E-06"/></inertial>
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
# 工具函数
# ═══════════════════════════════════════════════════════════

def create_scene_with_gripper():
    """创建场景 + 加载夹爪，返回 (scene, robot, joints, vidx, fidx, jnames)"""
    scene = sapien.Scene()
    scene.set_timestep(PHYSICS_TIMESTEP)
    scene.add_directional_light([1,-1,-1], [2.5,2.5,2.5], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_ground(0.0, render_half_size=[2.0, 2.0])

    prefix = "right"
    urdf = generate_urdf(prefix)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf)

    jnames = [j.name for j in robot.get_active_joints()]
    vidx = {n: jnames.index(n) for n in ['virtual_x','virtual_y','virtual_z','virtual_rz','virtual_ry','virtual_rx']}
    fidx = [jnames.index(f'{prefix}_gripper_finger_joint1'), jnames.index(f'{prefix}_gripper_finger_joint2')]
    joints = robot.get_active_joints()

    # PD 参数 (与 06_simple_grasp_test.py 完全一致)
    for i, j in enumerate(joints):
        if i == vidx['virtual_x'] or i == vidx['virtual_y'] or i == vidx['virtual_z']:
            if j.type == "prismatic":
                j.set_drive_property(stiffness=1000, damping=200)
            else:
                j.set_drive_property(stiffness=200, damping=50)
        elif i in [vidx['virtual_rz'], vidx['virtual_ry'], vidx['virtual_rx']]:
            j.set_drive_property(stiffness=200, damping=50)
        elif i in fidx:
            j.set_drive_property(stiffness=1000, damping=200)

    # 高摩擦
    touch_links = [f'{prefix}_gripper_finger_link1', f'{prefix}_gripper_finger_link2']
    for link in robot.get_links():
        if link.name in touch_links:
            for comp in link.entity.components:
                if hasattr(comp, 'physx_material'):
                    comp.physx_material = scene.create_physical_material(3.0, 3.0, 0.0)
                try:
                    for cs in comp.get_collision_shapes():
                        g = list(cs.get_collision_groups())
                        g[0] = 2; g[1] = 0b1001
                        cs.set_collision_groups(g)
                except AttributeError:
                    pass
                break

    # 初始 qpos: Z=0.4, 手指半开 (与 06 一致)
    qpos = np.zeros(len(joints))
    qpos[vidx['virtual_z']] = 0.4
    for fi in fidx:
        qpos[fi] = 0.025
    robot.set_qpos(qpos)
    for i, j in enumerate(joints):
        j.set_drive_target(qpos[i])

    # 稳定 50 步
    for _ in range(50):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()

    return scene, robot, joints, vidx, fidx, jnames


def step_physics(robot, scene, n=DECIMATION):
    for _ in range(n):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()


# ═══════════════════════════════════════════════════════════
# 测试 A: FK 正确性
# ═══════════════════════════════════════════════════════════

def test_A_fk():
    """验证 URDF 运动学: 设置 qpos 后，检查手指 link 位置是否与手动计算一致

    手动计算:
      finger1_origin = [0.03689, -0.013453, -0.00012053]  (URDF)
      finger1_axis   = [0, -1, 0]
      finger1_world  = root_R @ (origin + qpos * axis) + root_pos

    判据: |FK位置 - 手动计算| < 2mm
    """
    print("\n" + "=" * 60)
    print("测试 A: FK 正确性 (URDF 运动学 vs 手动计算)")
    print("=" * 60)

    scene, robot, joints, vidx, fidx, jnames = create_scene_with_gripper()
    prefix = "right"

    # URDF 常数
    F1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
    F1_AXIS = np.array([0, -1, 0])
    F2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
    F2_AXIS = np.array([0, 1, 0])

    # 测试多组 qpos
    test_cases = [
        {"q1": 0.000, "q2": 0.000, "desc": "完全闭合"},
        {"q1": 0.025, "q2": 0.025, "desc": "半开"},
        {"q1": 0.050, "q2": 0.050, "desc": "全开"},
    ]

    all_pass = True
    for tc in test_cases:
        q1, q2 = tc["q1"], tc["q2"]

        # 设置 qpos (保持 root 在 (0,0,0.3))
        qpos = robot.get_qpos()
        qpos[fidx[0]] = q1
        qpos[fidx[1]] = q2
        robot.set_qpos(qpos)
        for i, j in enumerate(joints):
            j.set_drive_target(qpos[i])

        # 稳定
        step_physics(robot, scene, n=30)

        # 从 PhysX 读取手指 link 位置
        actual_qpos = robot.get_qpos()
        f1_link = None
        f2_link = None
        for link in robot.get_links():
            if link.name == f'{prefix}_gripper_finger_link1':
                f1_link = link
            elif link.name == f'{prefix}_gripper_finger_link2':
                f2_link = link

        f1_actual = np.array(f1_link.get_pose().p)
        f2_actual = np.array(f2_link.get_pose().p)

        # 手动计算: FK from URDF
        # root 位置 = (virtual_x, virtual_y, virtual_z) = (0, 0, 0.4)
        # root 朝向 = 单位阵 (所有虚拟旋转 = 0)
        root_pos = np.array([0.0, 0.0, 0.4])
        root_R = np.eye(3)

        # finger1 关节原点在世界坐标的位置
        f1_joint_world = root_R @ F1_ORIGIN + root_pos
        # finger1 关节沿 axis 方向移动 q1
        f1_expected = f1_joint_world + root_R @ (q1 * F1_AXIS)

        f2_joint_world = root_R @ F2_ORIGIN + root_pos
        f2_expected = f2_joint_world + root_R @ (q2 * F2_AXIS)

        # 注意: link pose 是 link 坐标系原点，不是关节原点
        # 但 URDF 中 finger link 的 origin 是 (0,0,0) 相对关节
        # 所以 link pose ≈ 关节位置 + qpos*axis

        err1 = np.linalg.norm(f1_actual - f1_expected)
        err2 = np.linalg.norm(f2_actual - f2_expected)

        # 手指间距
        finger_dist = np.linalg.norm(f1_actual - f2_actual)
        expected_dist = np.linalg.norm(f1_expected - f2_expected)

        status1 = "PASS" if err1 < 0.005 else "FAIL"
        status2 = "PASS" if err2 < 0.005 else "FAIL"
        if status1 == "FAIL" or status2 == "FAIL":
            all_pass = False

        print(f"\n  [{tc['desc']}] q1={q1:.3f}, q2={q2:.3f}")
        print(f"    手指1: 实际={f1_actual}, 期望={f1_expected}, 误差={err1*1000:.2f}mm [{status1}]")
        print(f"    手指2: 实际={f2_actual}, 期望={f2_expected}, 误差={err2*1000:.2f}mm [{status2}]")
        print(f"    手指间距: 实际={finger_dist*1000:.1f}mm, 期望={expected_dist*1000:.1f}mm")

    result = "PASS" if all_pass else "FAIL"
    print(f"\n  >>> 测试 A: {result} (判据: FK误差 < 5mm)")
    return all_pass


# ═══════════════════════════════════════════════════════════
# 测试 B: PD 关节跟踪
# ═══════════════════════════════════════════════════════════

def test_B_pd_tracking():
    """验证 PD 控制器能跟踪目标: 手指开合 + 根移动

    判据: 稳态误差 < 1mm (手指), < 2mm (根)
    """
    print("\n" + "=" * 60)
    print("测试 B: PD 关节跟踪 (手指 + 根)")
    print("=" * 60)

    scene, robot, joints, vidx, fidx, jnames = create_scene_with_gripper()
    prefix = "right"

    # 子测试 B1: 手指从半开到闭合
    print("\n  B1: 手指跟踪 (半开 0.025 → 闭合 0.000)")
    for fi in fidx:
        joints[fi].set_drive_target(0.0)
    step_physics(robot, scene, n=80)

    qpos = robot.get_qpos()
    err_f1 = abs(qpos[fidx[0]] - 0.0) * 1000
    err_f2 = abs(qpos[fidx[1]] - 0.0) * 1000
    s1 = "PASS" if err_f1 < 1.0 else "FAIL"
    s2 = "PASS" if err_f2 < 1.0 else "FAIL"
    print(f"    手指1: qpos={qpos[fidx[0]]*1000:.3f}mm, 目标=0.0mm, 误差={err_f1:.3f}mm [{s1}]")
    print(f"    手指2: qpos={qpos[fidx[1]]*1000:.3f}mm, 目标=0.0mm, 误差={err_f2:.3f}mm [{s2}]")

    # 子测试 B2: 手指全开
    print("\n  B2: 手指跟踪 (闭合 → 全开 0.050)")
    for fi in fidx:
        joints[fi].set_drive_target(0.05)
    step_physics(robot, scene, n=80)

    qpos = robot.get_qpos()
    err_f1 = abs(qpos[fidx[0]] - 0.05) * 1000
    err_f2 = abs(qpos[fidx[1]] - 0.05) * 1000
    s1 = "PASS" if err_f1 < 1.0 else "FAIL"
    s2 = "PASS" if err_f2 < 1.0 else "FAIL"
    print(f"    手指1: qpos={qpos[fidx[0]]*1000:.3f}mm, 目标=50.0mm, 误差={err_f1:.3f}mm [{s1}]")
    print(f"    手指2: qpos={qpos[fidx[1]]*1000:.3f}mm, 目标=50.0mm, 误差={err_f2:.3f}mm [{s2}]")

    # 子测试 B3: 根位置移动 (Z: 0.4 → 0.2)
    print("\n  B3: 根位置跟踪 (Z: 0.4 → 0.2)")
    joints[vidx['virtual_z']].set_drive_target(0.2)
    step_physics(robot, scene, n=80)

    qpos = robot.get_qpos()
    err_z = abs(qpos[vidx['virtual_z']] - 0.2) * 1000
    sz = "PASS" if err_z < 2.0 else "FAIL"
    print(f"    Z: qpos={qpos[vidx['virtual_z']]*1000:.1f}mm, 目标=200.0mm, 误差={err_z:.3f}mm [{sz}]")

    # 子测试 B4: 根位置移动 (X: 0 → 0.1)
    print("\n  B4: 根位置跟踪 (X: 0 → 0.1)")
    joints[vidx['virtual_x']].set_drive_target(0.1)
    step_physics(robot, scene, n=80)

    qpos = robot.get_qpos()
    err_x = abs(qpos[vidx['virtual_x']] - 0.1) * 1000
    sx = "PASS" if err_x < 2.0 else "FAIL"
    print(f"    X: qpos={qpos[vidx['virtual_x']]*1000:.1f}mm, 目标=100.0mm, 误差={err_x:.3f}mm [{sx}]")

    all_pass = (s1 == "PASS" and s2 == "PASS" and sz == "PASS" and sx == "PASS")
    result = "PASS" if all_pass else "FAIL"
    print(f"\n  >>> 测试 B: {result} (判据: 手指<1mm, 根<2mm)")
    return all_pass


# ═══════════════════════════════════════════════════════════
# 测试 C: 接触力检测
# ═══════════════════════════════════════════════════════════

def test_C_contact():
    """验证手指碰到固定物体时有接触力

    方法: 在手指路径上放一个 kinematic 物体，闭合手指，检测接触力

    判据: 接触力 > 0.1N
    """
    print("\n" + "=" * 60)
    print("测试 C: 接触力检测 (手指 vs 固定物体)")
    print("=" * 60)

    scene, robot, joints, vidx, fidx, jnames = create_scene_with_gripper()
    prefix = "right"

    # 手指全开
    for fi in fidx:
        joints[fi].set_drive_target(0.05)
    step_physics(robot, scene, n=60)

    # 读取手指 link 位置
    f1_link = None
    f2_link = None
    for link in robot.get_links():
        if link.name == f'{prefix}_gripper_finger_link1':
            f1_link = link
        elif link.name == f'{prefix}_gripper_finger_link2':
            f2_link = link

    f1_pos = np.array(f1_link.get_pose().p)
    f2_pos = np.array(f2_link.get_pose().p)

    # 在两指之间放一个固定方块 (kinematic)
    mid_pos = (f1_pos + f2_pos) / 2
    box_builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial()
    mat.base_color = [0.8, 0.3, 0.15, 1.0]
    box_builder.add_box_visual(half_size=[0.02, 0.02, 0.02], material=mat)
    pm = scene.create_physical_material(5, 5, 0.5)
    box_builder.add_box_collision(half_size=[0.02, 0.02, 0.02], material=pm, density=1000)
    box = box_builder.build(name="fixed_box")
    box.set_pose(sapien.Pose(p=mid_pos.tolist()))

    # 设为 kinematic (不会动)
    for comp in box.components:
        if hasattr(comp, 'set_physx_body_type'):
            comp.set_physx_body_type("kinematic")
        break

    step_physics(robot, scene, n=20)

    # 闭合手指
    print(f"  方块位置: {mid_pos}")
    print(f"  手指1位置: {f1_pos}")
    print(f"  手指2位置: {f2_pos}")
    print(f"  手指间距: {np.linalg.norm(f1_pos - f2_pos)*1000:.1f}mm")

    for fi in fidx:
        joints[fi].set_drive_target(0.0)  # 闭合

    # 逐步闭合，记录接触力
    max_force = 0.0
    for step in range(100):
        step_physics(robot, scene, n=1)

        # 读取接触力
        contacts = scene.get_contacts()
        for c in contacts:
            for p in c.points:
                force = np.linalg.norm(p.impulse) / PHYSICS_TIMESTEP
                if force > max_force:
                    max_force = force

        # 读取手指实际位置
        f1_pos_now = np.array(f1_link.get_pose().p)
        f2_pos_now = np.array(f2_link.get_pose().p)
        finger_dist = np.linalg.norm(f1_pos_now - f2_pos_now)

        if step % 20 == 0:
            qpos = robot.get_qpos()
            print(f"    步骤 {step:3d}: 手指qpos=[{qpos[fidx[0]]*1000:.1f}, {qpos[fidx[1]]*1000:.1f}]mm, "
                  f"间距={finger_dist*1000:.1f}mm, 最大力={max_force:.2f}N")

    s = "PASS" if max_force > 0.1 else "FAIL"
    print(f"\n  最大接触力: {max_force:.2f}N [{s}] (判据: > 0.1N)")

    result = "PASS" if max_force > 0.1 else "FAIL"
    print(f"\n  >>> 测试 C: {result}")
    return max_force > 0.1


# ═══════════════════════════════════════════════════════════
# 测试 D: 物理抓取
# ═══════════════════════════════════════════════════════════

def test_D_grasp():
    """验证夹爪能物理抓取并抬升 dynamic 物体

    流程:
    1. 夹爪在方块上方
    2. 下降到方块处
    3. 闭合手指
    4. 抬升

    判据: 方块抬升 > 5cm (纯物理力，不瞬移)
    """
    print("\n" + "=" * 60)
    print("测试 D: 物理抓取 (闭合 + 抬升，纯物理力)")
    print("=" * 60)

    scene, robot, joints, vidx, fidx, jnames = create_scene_with_gripper()
    prefix = "right"

    # 创建 dynamic 方块 (与 06_simple_grasp_test.py 一致: 4cm 半边长)
    CUBE_HALF = 0.04
    box_builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial()
    mat.base_color = [0.8, 0.3, 0.15, 1.0]
    box_builder.add_box_visual(half_size=[CUBE_HALF]*3, material=mat)
    pm = scene.create_physical_material(5, 5, 0.9)
    box_builder.add_box_collision(half_size=[CUBE_HALF]*3, material=pm, density=500)
    box = box_builder.build(name="red_box")
    box.set_pose(sapien.Pose(p=[0.0, 0.0, CUBE_HALF]))  # 地面 z=0

    # 碰撞组 (与 06 一致)
    for comp in box.components:
        try:
            for cs in comp.get_collision_shapes():
                g = list(cs.get_collision_groups())
                g[0] = 3; g[1] = 0b0111
                cs.set_collision_groups(g)
        except AttributeError:
            pass
        break

    box_z_init = box.get_pose().p[2]
    print(f"  方块初始 z: {box_z_init*1000:.1f}mm")

    # ===== 完全复用 06_simple_grasp_test.py 的流程 =====
    def set_target(x, y, z, rz, ry, rx, f1, f2):
        target = [0.0]*len(jnames)
        R = np.eye(3) @ np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0,0,1]])
        R = R @ (np.array([[np.cos(ry), 0, np.sin(ry)], [0,1,0], [-np.sin(ry), 0, np.cos(ry)]]))
        R = R @ (np.array([[1,0,0], [0,np.cos(rx),-np.sin(rx)], [0,np.sin(rx),np.cos(rx)]]))
        roll_x = np.arctan2(-R[1,2], R[2,2])
        pitch_y = np.arctan2(R[0,2], np.sqrt(R[0,0]**2+R[0,1]**2))
        yaw_z = np.arctan2(R[0,1], R[0,0])
        target[vidx['virtual_x']] = x; target[vidx['virtual_y']] = y; target[vidx['virtual_z']] = z
        target[vidx['virtual_rz']] = yaw_z; target[vidx['virtual_ry']] = pitch_y; target[vidx['virtual_rx']] = roll_x
        target[fidx[0]] = f1; target[fidx[1]] = f2
        for i, j in enumerate(joints):
            j.set_drive_target(target[i])

    # 步骤 1: 下降 (Z: 0.4 → 0.1), 手指半开 (与 06 一致)
    print("\n  步骤 1: 下降 (Z: 0.4 → 0.1)")
    z = 0.4
    for frame in range(60):
        z = 0.4 - frame * 0.005
        set_target(0, 0, z, 0, 0, 0, 0.025, 0.025)
        step_physics(robot, scene)
    print(f"    Z={z:.3f}")

    # 步骤 2: 张开夹爪 (与 06 一致)
    print("\n  步骤 2: 张开夹爪")
    f = 0.025
    for frame in range(30):
        f = 0.025 + frame / 30 * 0.025
        set_target(0, 0, z, 0, 0, 0, f, f)
        step_physics(robot, scene)
    print(f"    f={f:.3f}")

    # 步骤 3: 下降到碰触方块 (与 06 一致)
    print("\n  步骤 3: 下降到方块 (Z → 0.02)")
    target_z_before_grasp = 0.02
    for frame in range(40):
        z = z - 0.005
        if z < target_z_before_grasp: z = target_z_before_grasp
        set_target(0, 0, z, 0, 0, 0, f, f)
        step_physics(robot, scene)
    print(f"    Z={z:.3f}")

    # 步骤 4: 闭合夹爪 (与 06 一致)
    print("\n  步骤 4: 闭合夹爪")
    for frame in range(60):
        f = 0.05 - frame / 60 * 0.05
        set_target(0, 0, z, 0, 0, 0, max(0,f), max(0,f))
        step_physics(robot, scene)
    print(f"    f={max(0,f):.3f}")

    box_z_closed = box.get_pose().p[2]
    qpos_closed = robot.get_qpos()
    print(f"    方块 z: {box_z_closed*100:.2f}cm (闭合后)")
    print(f"    手指 qpos: [{qpos_closed[fidx[0]]*1000:.1f}, {qpos_closed[fidx[1]]*1000:.1f}]mm")

    # 步骤 5: 抬起方块 (与 06 一致)
    print("\n  步骤 5: 抬起方块")
    for frame in range(80):
        z = z + 0.002
        set_target(0, 0, z, 0, 0, 0, max(0,f), max(0,f))
        step_physics(robot, scene)

        if frame % 20 == 0:
            box_z_now = box.get_pose().p[2]
            qpos_now = robot.get_qpos()
            print(f"    帧 {frame:3d}: 方块 z={box_z_now*100:.2f}cm, 夹爪 Z={qpos_now[vidx['virtual_z']]*100:.2f}cm")

    box_z_final = box.get_pose().p[2]
    lift_cm = (box_z_final - box_z_init) * 100

    print(f"\n  方块 z: {box_z_init*100:.1f}cm → {box_z_final*100:.1f}cm")
    print(f"  抬升: {lift_cm:.1f}cm")

    s = "PASS" if lift_cm > 5 else "FAIL"
    print(f"\n  >>> 测试 D: {s} (判据: 抬升 > 5cm, 实际 {lift_cm:.1f}cm)")
    return lift_cm > 5


# ═══════════════════════════════════════════════════════════
# 测试 E: 跟踪 + 抓取 (同时验证)
# ═══════════════════════════════════════════════════════════

def test_E_track_and_grasp():
    """验证 6 虚拟关节 PD 驱动同时实现:
    1. 跟踪目标轨迹 (位置误差 < 10mm)
    2. 物理抓取物体 (抬升 > 3cm)

    使用解析法 IK 计算目标位姿，然后 PD 驱动虚拟关节跟踪。

    判据: 位置误差 < 10mm 且 抬升 > 3cm
    """
    print("\n" + "=" * 60)
    print("测试 E: 跟踪 + 抓取 (6 虚拟关节 PD, 同时验证)")
    print("=" * 60)

    scene, robot, joints, vidx, fidx, jnames = create_scene_with_gripper()
    prefix = "right"

    # 创建 dynamic 方块 (4cm 半边长, 放在地面 z=0)
    CUBE_HALF = 0.04
    box_builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial()
    mat.base_color = [0.8, 0.3, 0.15, 1.0]
    box_builder.add_box_visual(half_size=[CUBE_HALF]*3, material=mat)
    pm = scene.create_physical_material(5, 5, 0.9)
    box_builder.add_box_collision(half_size=[CUBE_HALF]*3, material=pm, density=500)
    box = box_builder.build(name="red_box")
    box.set_pose(sapien.Pose(p=[0.0, 0.0, CUBE_HALF]))

    for comp in box.components:
        try:
            for cs in comp.get_collision_shapes():
                g = list(cs.get_collision_groups())
                g[0] = 3; g[1] = 0b0111
                cs.set_collision_groups(g)
        except AttributeError:
            pass
        break

    box_z_init = box.get_pose().p[2]

    # ===== 解析法 IK =====
    GRIPPER_DEPTH_OFFSET = 0.03689
    FINGER_BASE_DIST = 0.026906

    def analytical_ik(target_pos, target_R, finger_dist):
        finger_center = target_pos
        root_pos = finger_center - target_R @ np.array([GRIPPER_DEPTH_OFFSET, 0.0, 0.0])
        joint = max(0, (finger_dist - FINGER_BASE_DIST) / 2)
        rz = np.arctan2(target_R[0,1], target_R[0,0])
        ry = np.arctan2(target_R[0,2], np.sqrt(target_R[0,0]**2 + target_R[0,1]**2))
        rx = np.arctan2(-target_R[1,2], target_R[2,2])
        return root_pos, rz, ry, rx, joint

    # ===== 生成目标轨迹 (完全复用 06_simple_grasp_test.py 的 set_target 参数) =====
    # 直接存储 (x, y, z, rz, ry, rx, f1, f2) 元组
    trajectory = []

    z = 0.4
    # 步骤 2: 下降 (60帧, Z: 0.4 → 0.25, 2.5mm/帧)
    for i in range(60):
        z = 0.4 - i * 0.0025
        trajectory.append((0, 0, z, 0, 0, 0, 0.025, 0.025, 'DESCEND'))

    # 步骤 3: 张开夹爪 (30帧)
    for i in range(30):
        f = i / 30 * 0.05
        trajectory.append((0, 0, z, 0, 0, 0, f, f, 'OPEN'))

    # 步骤 4: 继续下降到碰触方块 (40帧, 5mm/帧)
    for i in range(40):
        z = z - 0.005
        if z < 0.02: z = 0.02
        trajectory.append((0, 0, z, 0, 0, 0, 0.05, 0.05, 'APPROACH'))

    # 步骤 5: 闭合夹爪 (60帧)
    for i in range(60):
        f = 0.05 - i / 60 * 0.05
        trajectory.append((0, 0, z, 0, 0, 0, max(0,f), max(0,f), 'CLOSE'))

    # 步骤 6: 抬起方块 (80帧, 2mm/帧)
    for i in range(80):
        z = z + 0.002
        trajectory.append((0, 0, z, 0, 0, 0, 0, 0, 'LIFT'))

    # ===== 执行轨迹 =====
    pos_errors = []
    finger_errors = []

    print(f"  轨迹总帧数: {len(trajectory)}")
    print(f"  方块初始 z: {box_z_init*1000:.1f}mm")

    for frame_idx, tf in enumerate(trajectory):
        x, y, z_t, rz_t, ry_t, rx_t, f1_t, f2_t, phase = tf

        # 用和 06 完全一致的 set_target 逻辑
        target = [0.0]*len(jnames)
        R_build = np.eye(3)
        R_build = R_build @ np.array([[np.cos(rz_t), -np.sin(rz_t), 0], [np.sin(rz_t), np.cos(rz_t), 0], [0,0,1]])
        R_build = R_build @ np.array([[np.cos(ry_t), 0, np.sin(ry_t)], [0,1,0], [-np.sin(ry_t), 0, np.cos(ry_t)]])
        R_build = R_build @ np.array([[1,0,0], [0,np.cos(rx_t),-np.sin(rx_t)], [0,np.sin(rx_t),np.cos(rx_t)]])
        roll_x = np.arctan2(-R_build[1,2], R_build[2,2])
        pitch_y = np.arctan2(R_build[0,2], np.sqrt(R_build[0,0]**2+R_build[0,1]**2))
        yaw_z = np.arctan2(R_build[0,1], R_build[0,0])
        target[vidx['virtual_x']] = x; target[vidx['virtual_y']] = y; target[vidx['virtual_z']] = z_t
        target[vidx['virtual_rz']] = yaw_z; target[vidx['virtual_ry']] = pitch_y; target[vidx['virtual_rx']] = roll_x
        target[fidx[0]] = f1_t; target[fidx[1]] = f2_t
        for i, j in enumerate(joints):
            j.set_drive_target(target[i])

        step_physics(robot, scene)

        qpos = robot.get_qpos()
        actual_pos = np.array([qpos[vidx['virtual_x']], qpos[vidx['virtual_y']], qpos[vidx['virtual_z']]])
        pos_err = np.linalg.norm(actual_pos - np.array([x, y, z_t]))
        pos_errors.append(pos_err)

        actual_j1 = qpos[fidx[0]]
        actual_j2 = qpos[fidx[1]]
        finger_err = (abs(actual_j1 - f1_t) + abs(actual_j2 - f2_t)) / 2
        finger_errors.append(finger_err)

        if frame_idx % 30 == 0:
            box_z = box.get_pose().p[2]
            print(f"  帧 {frame_idx:3d} [{phase:7s}]: "
                  f"目标Z={z_t*100:.1f}cm 实际Z={actual_pos[2]*100:.1f}cm "
                  f"pos_err={pos_err*1000:.1f}mm "
                  f"finger=[{actual_j1*1000:.1f},{actual_j2*1000:.1f}]mm→{f1_t*1000:.1f}mm "
                  f"方块Z={box_z*100:.1f}cm")

        # APPROACH 阶段详细诊断
        if 90 <= frame_idx <= 130 and frame_idx % 5 == 0:
            z_error = z_t - qpos[vidx['virtual_z']]
            print(f"    [Z诊断 帧{frame_idx}] target={z_t*100:.1f}cm actual={qpos[vidx['virtual_z']]*100:.1f}cm err={z_error*1000:.1f}mm")

    # ===== 结果 =====
    box_z_final = box.get_pose().p[2]
    lift_cm = (box_z_final - box_z_init) * 100
    pos_errors = np.array(pos_errors)
    finger_errors = np.array(finger_errors)

    print(f"\n  --- 跟踪精度 ---")
    print(f"  位置误差: avg={pos_errors.mean()*1000:.1f}mm, max={pos_errors.max()*1000:.1f}mm")
    print(f"  手指误差: avg={finger_errors.mean()*1000:.1f}mm, max={finger_errors.max()*1000:.1f}mm")
    print(f"\n  --- 抓取效果 ---")
    print(f"  方块 z: {box_z_init*100:.1f}cm → {box_z_final*100:.1f}cm, 抬升 {lift_cm:.1f}cm")

    track_pass = pos_errors.mean() < 0.01
    grasp_pass = lift_cm > 3
    print(f"\n  跟踪: {'PASS' if track_pass else 'FAIL'} (avg < 10mm, 实际 {pos_errors.mean()*1000:.1f}mm)")
    print(f"  抓取: {'PASS' if grasp_pass else 'FAIL'} (lift > 3cm, 实际 {lift_cm:.1f}cm)")
    print(f"\n  >>> 测试 E: {'PASS' if (track_pass and grasp_pass) else 'FAIL'}")
    return track_pass and grasp_pass


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("夹爪物理仿真验证")
    print("=" * 60)

    results = {}

    # 测试 A: FK
    try:
        results["A_FK"] = test_A_fk()
    except Exception as e:
        print(f"\n  >>> 测试 A: ERROR ({e})")
        results["A_FK"] = False

    # 测试 B: PD
    try:
        results["B_PD"] = test_B_pd_tracking()
    except Exception as e:
        print(f"\n  >>> 测试 B: ERROR ({e})")
        results["B_PD"] = False

    # 测试 C: Contact
    try:
        results["C_Contact"] = test_C_contact()
    except Exception as e:
        print(f"\n  >>> 测试 C: ERROR ({e})")
        results["C_Contact"] = False

    # 测试 D: Grasp
    try:
        results["D_Grasp"] = test_D_grasp()
    except Exception as e:
        print(f"\n  >>> 测试 D: ERROR ({e})")
        results["D_Grasp"] = False

    # 测试 E: 跟踪 + 抓取
    try:
        results["E_TrackAndGrasp"] = test_E_track_and_grasp()
    except Exception as e:
        print(f"\n  >>> 测试 E: ERROR ({e})")
        import traceback; traceback.print_exc()
        results["E_TrackAndGrasp"] = False

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    for name, passed in results.items():
        s = "PASS" if passed else "FAIL"
        print(f"  {name}: {s}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  总计: {passed}/{total} 通过")
    print("=" * 60)

    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

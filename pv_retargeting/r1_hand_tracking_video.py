#!/usr/bin/env python3
"""
================================================================================
  R1 手部追踪视频生成管线
  DexYCB → Dex Retargeting (手部) → RelaxedIK (R1右臂) → 生成视频
================================================================================

管线总览 (Pipeline Overview)
────────────────────────────────────────────────────────────────────────────────

  DexYCB数据集          Dex Retargeting          RelaxedIK           视频
  ┌──────────┐      ┌──────────────────┐    ┌──────────────┐    ┌───────┐
  │ hand_pose │─→MANO→│ joints[0,4,8,...] │─→→│ 手腕→IK目标  │─IK→│ 关节角 │─→│ MP4  │
  │ hand_shape│      │ 21个3D关节点      │    │ +映射+安全距 │    │ qpos  │    │      │
  └──────────┘      └──────────────────┘    └──────────────┘    └───────┘    └─────┘

阶段 1: 数据加载 (run → step [1/7])
  - 加载 DexYCB 数据集, 获取 hand_pose[48+3], hand_shape[10], object_pose, extrinsics
  - hand_pose: 前3维=手腕compact axis-angle, 中45维=手指PCA, 后3维=平移

阶段 2: Dex Retargeting 初始化 (run → step [2/7])
  - 创建 RobotHandDatasetSAPIENViewer (SAPIEN场景 + R1机器人 + 人手mesh + YCB物体)
  - 初始化 SeqRetargeting (NLopt SLSQP 位置优化器)
  - 优化目标: 最小化人手参考点与R1夹爪对应点的3D位置误差

阶段 3: 手部轨迹分析 (run → step [3/7] → _analyze_hand_trajectory)
  - 遍历所有帧, 通过 MANOLayer 计算每帧的21个3D关节点
  - 提取手腕关节(joint[0])位置, 计算质心/范围/标准差
  - 用于后续机器人放置和工作空间映射

阶段 4: 机器人放置 (run → step [4/7])
  - 基座固定地面 Z=0, 绕Z轴旋转180°(背向操作者)
  - Y方向根据手腕质心调整, 确保臂基座在手腕侧方
  - 获取 right_arm_base_link 实际世界位姿 (通过 SAPIEN FK)

阶段 5: 工作空间映射 (run → step [5/7] → _compute_workspace_mapping)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 问题: R1臂基座高Z≈0.97m, 臂展0.71m, 最低可达Z≈0.26m              │
  │       DexYCB手腕高Z≈0.08m, 低于臂最低可达范围                      │
  │ 解决: 工作空间映射 — 将手部轨迹平移到臂可达空间内                   │
  │                                                                 │
  │ 1. 舒适目标(base帧): [0.30, 0.0, -0.25] (前方30cm, 下方25cm)     │
  │ 2. 舒适目标(世界帧): base_link_R @ [0.30,0,-0.25] + base_link_p  │
  │ 3. 映射偏移 = 舒适目标(世界帧) - 手腕质心                         │
  │ 4. 安全偏移 = normalize(base_link_p - 舒适目标) × 0.075m         │
  │ 5. 每帧: ik_target = wrist_pos + mapping_offset + safety_offset  │
  └─────────────────────────────────────────────────────────────────────┘

阶段 6: RelaxedIK 初始化 (run → step [6/7])
  - 加载浮动URDF (r1_v2_1_0_floating_right.urdf), base_links=[right_arm_base_link]
  - IK输入: right_arm_base_link 帧坐标 (非世界坐标!)
  - IK输出: 6个关节角度

阶段 7: 预计算 (run → step [7/7] → _precompute)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 每帧处理流程:                                                       │
  │                                                                 │
  │ ① MANO正运动学: hand_pose → MANOLayer → vertex[778,3], joints[21,3]│
  │                                                                 │
  │ ② Dex Retargeting (手部映射):                                     │
  │    ref_value = joints[ref_indices]  (拇指尖/食指尖等参考点)        │
  │    retarget_qpos = SeqRetargeting.retarget(ref_value, fixed_qpos) │
  │    → NLopt SLSQP 最小化位置误差 → R1夹爪关节角                     │
  │    → gripper1, gripper2 = retarget_qpos[retarget2sapien]         │
  │                                                                 │
  │ ③ 工作空间映射 (IK目标计算):                                       │
  │    ik_target_world = wrist_pos + mapping_offset + safety_offset  │
  │    ik_target_world = LPFilter(α=0.6).next(ik_target_world)      │
  │    ik_target_base = base_link_R⁻¹ @ (ik_target_world - base_p)  │
  │                                                                 │
  │ ④ RelaxedIK (臂IK求解):                                           │
  │    # 用重定向优化器FK自动获取夹爪朝向（与visualize_hand_object.py一致）│
  │    r1_robot.set_qpos(retarget_qpos[retarget2sapien])               │
  │    ee_pose = right_gripper_link.get_entity_pose()                  │
  │    R_ee_world = matrix_from_quaternion(ee_pose.q)                  │
  │    r1_robot.set_qpos(saved_qpos)  # 恢复                           │
  │    # 变换到base_link帧                                              │
  │    R_ee_base = base_link_R⁻¹ @ R_ee_world                         │
  │    ee_quat_base = quat_from_matrix(R_ee_base)  # [w,x,y,z]        │
  │    right_joints = ik_solver.solve_position_right(                  │
  │        ik_target_base.tolist(),   # base_link帧坐标                │
  │        ee_quat_base.tolist()      # 末端朝向(wxyz)                 │
  │    )                                                            │
  │    right_joints = LPFilter(α=0.5).next(right_joints)  # 关节平滑 │
  │                                                                 │
  │ ⑤ 组装qpos:                                                      │
  │    r1_qpos[right_arm_indices] = right_joints                     │
  │    r1_qpos[gripper_idx1] = gripper1                              │
  │    r1_qpos[gripper_idx2] = gripper2                              │
  └─────────────────────────────────────────────────────────────────────┘

阶段 8: Warmup过渡 (_precompute → warmup阶段)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 初始跳变问题:                                                       │
  │   机器人初始关节角 = URDF默认值 (手臂伸直)                          │
  │   第一帧IK结果 = 完全不同的构型 (手臂弯曲)                          │
  │   → 直接设置导致大幅度跳变                                          │
  │                                                                 │
  │ 解决方案: Warmup过渡帧                                              │
  │   1. 先求解第一帧的IK目标, 得到 target_joints                      │
  │   2. 获取当前机器人右臂关节角, 得到 current_joints                  │
  │   3. 在 current_joints → target_joints 之间线性插值 N 帧           │
  │   4. LPFilter 从 current_joints 初始化, 避免首帧跳变               │
  └─────────────────────────────────────────────────────────────────────┘

阶段 9: 渲染视频 (_render_video)
  - 逐帧: set_qpos → scene.step() → camera.take_picture() → 写入视频
  - FK验证: 比较 right_gripper_link 实际位置与IK目标位置
  - 关节舒适度: 计算关节角到中位的归一化距离

阶段 10: 评估报告 (_output_evaluation)
  - 手部轨迹分析 / 工作空间映射 / IK求解统计 / FK验证 / 关节舒适度 / 运动平滑度(含合格判定) / 逐帧详情

阶段 8.5: 轨迹后处理平滑 (TrajectorySmoother)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 五阶段管线 + 迭代收敛:                                              │
  │                                                                 │
  │ A. 双向二阶Butterworth低通滤波 (零相位滞后, -40dB/dec衰减)          │
  │    前向: s1=s1+α*(x-s1), s2=s2+α*(s1-s2)                        │
  │    后向: s1'=s1'+α*(s2-s1'), s2'=s2'+α*(s1'-s2')                │
  │    → 比一阶指数滤波衰减更快, 高频噪声抑制能力提升约2倍              │
  │                                                                 │
  │ B. 速度限幅: |Δq/Δt| ≤ v_max (1.5 rad/s)                        │
  │ C. 加速度限幅: |Δv/Δt| ≤ a_max (4.0 rad/s²)                     │
  │ D. 加加速度限幅: |Δa/Δt| ≤ j_max (20.0 rad/s³)  ← 核心改进      │
  │                                                                 │
  │ E. 迭代收敛: B→C→D循环, 直到轨迹变化量 < ε                       │
  │    → 保证速度/加速度/加加速度约束同时满足                           │
  │                                                                 │
  │ 平滑度合格标准:                                                    │
  │   速度≤3.0 rad/s, 加速度≤8.0 rad/s², 加加速度≤80.0 rad/s³       │
  │   SI改善≥50%                                                     │
  └─────────────────────────────────────────────────────────────────────┘

视角模式 (--view)：
  - behind:  机器人背后, 相机在+X侧, 固定四元数[0,0,1,0]
  - front:   机器人前方, 相机在-X侧, 固定四元数[1,0,0,0]
  - topdown: 高空俯视, 相机在+Z侧, 固定四元数[0.7071,0,0.7071,0]

视频标注:
  - 0.75x慢放: 视频帧率=原始FPS*0.75, 每帧写入1次
  - Warmup阶段: 顶部进度条 + 百分比
  - 映射开始: ">>> MAPPING START <<<" 橙色横幅
  - 坐标系(放在机器人前方, front视角可见):
    WORLD:      世界坐标系原点 (白色标签)
    ROBOT_BASE: 机器人底座坐标系 (黄色标签, 绕Z旋转180°)
    ARM_BASE:   右臂基座坐标系 (青色标签, IK求解坐标系)
    EE:         机械臂末端坐标系 (绿色标签, right_gripper_link)
    HAND:       手部手腕坐标系 (橙色标签, DexYCB手部数据)
  - 轨迹线:
    EE trail:   机械臂末端轨迹 (渐变色蓝→绿)
    IK trail:   IK目标轨迹 (黄色)
    Hand trail: 手部手腕轨迹 (橙色)
  - 当前位置标记: 绿色圆点(EE), 青色圆点(IK target), 橙色圆点(Hand wrist)
  - EE-IK距离: 实时显示(绿<2cm, 黄<5cm, 红>5cm)
  - 图例: 左下角标注各可视化元素含义

IK坐标系对应关系:
  DexYCB数据集坐标系 = SAPIEN世界坐标系 (右手系, Z朝上)
  机器人root绕Z旋转180°, 故root的X轴=世界-X, root的Y轴=世界-Y
  right_arm_base_link在root坐标系中, 继承180°旋转
  IK求解器使用浮动URDF (r1_v2_1_0_floating_right.urdf), 坐标系=right_arm_base_link帧
  坐标变换: ik_target_base = base_link_R^(-1) @ (ik_target_world - base_link_p)
  朝向变换: R_ee_base = base_link_R^(-1) @ R_ee_world (R_ee_world由重定向优化器FK自动求解)

================================================================================
RelaxedIK 求解器详细分析
================================================================================

1. 概述
   RelaxedIK 是一种基于优化的逆运动学求解器, 源自 Rust 实现 (librelaxed_ik_lib.so)。
   它同时优化末端执行器的位置和朝向, 通过容差参数(tolerance)控制位置/朝向的优先级。
   与传统解析IK不同, RelaxedIK使用数值优化, 支持关节限位和连续性约束。

2. 架构
   ┌──────────────────────────────────────────────────────────────┐
   │ Python层 (RelaxedIKSolver)                                   │
   │   solve_position_right(pos, quat_wxyz)                       │
   │     → _convert_wxyz_to_xyzw: [w,x,y,z] → [x,y,z,w]         │
   │     → RelaxedIKRust.solve_position(pos, quat_xyzw, tol)     │
   │                                                              │
   │ Rust层 (librelaxed_ik_lib.so)                                │
   │   solve_position(obj, pos_arr, pos_len,                      │
   │                   quat_arr, quat_len,                         │
   │                   tol_arr, tol_len) → Opt{data, length}      │
   │   内部: 梯度下降多目标优化 + 关节限位软约束 + 平滑性约束     │
   └──────────────────────────────────────────────────────────────┘

3. 输入参数
   3.1 位置 (positions)
     - 格式: 1D array, 长度 3*N (N=末端执行器数量, 本项目N=1)
     - 坐标系: right_arm_base_link 帧 (非世界坐标系!)
     - 变换: ik_target_base = base_link_R⁻¹ @ (ik_target_world - base_link_p)

   3.2 朝向 (orientations)
     - 格式: 1D array, 长度 4*N, 四元数 xyzw 格式
     - Python层输入 wxyz 格式, 内部自动转换为 xyzw
     - 坐标系: right_arm_base_link 帧
     - 变换: R_ee_base = base_link_R⁻¹ @ R_ee_world
     - R_ee_world 由重定向优化器FK自动求解:
       retargeting.retarget() 优化dummy free joints → set_qpos → FK → right_gripper_link朝向
     - 这与 visualize_hand_object.py 的方式完全一致

   3.3 容差 (tolerances)
     - 格式: 1D array, 长度 6*N
     - 每个末端6个值: [tx, ty, tz, rx, ry, rz]
     - 位置容差单位: 米 (越小越严格)
     - 旋转容差单位: 弧度 (越小越严格)
     - 当前设置: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
     - 含义: 位置误差<1mm, 旋转误差<0.001rad(~0.06°)
     - 注意: 容差过小可能导致无解或求解缓慢

4. 输出
   - 6个关节角度 (弧度), 对应 right_arm_joint1~6
   - 关节顺序与 r1_v2_1_0_floating_right.urdf 中的定义一致

5. 浮动URDF (r1_v2_1_0_floating_right.urdf)
   5.1 运动链结构:
     right_arm_base_link (根, 固定)
       → right_arm_joint1 (revolute, Z轴, ±2.88rad, ±165°)
         → right_arm_link1
           → right_arm_joint2 (revolute, Y轴, 0~3.14rad, 0~180°)
             → right_arm_link2
               → right_arm_joint3 (revolute, Y轴, -3.32~0rad, -190~0°)
                 → right_arm_link3
                   → right_arm_joint4 (revolute, Y轴, ±1.57rad, ±90°)
                     → right_arm_link4
                       → right_arm_joint5 (revolute, Z轴, ±1.57rad, ±90°)
                         → right_arm_link5
                           → right_arm_joint6 (revolute, X轴, ±2.88rad, ±165°)
                             → right_arm_link6
                               → right_gripper_joint (fixed)
                                 → right_gripper_link (末端执行器)

   5.2 关键尺寸:
     - base→joint1: Z偏移 0.086m
     - joint1→joint2: Y偏移 0.031m, Z偏移 0.049m
     - joint2→joint3: X偏移 -0.300m (上臂长度)
     - joint3→joint4: X偏移 0.175m, Z偏移 0.075m
     - joint4→joint5: X偏移 0.080m, Z偏移 0.041m
     - joint5→joint6: X偏移 0.023m, Z偏移 -0.041m
     - joint6→gripper: X偏移 0.082m
     - 总臂展约: 0.086+0.049+0.300+0.175+0.080+0.023+0.082 ≈ 0.795m

   5.3 Starting Config:
     [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
     对应度数: [-85.9°, 111.8°, -61.9°, -25.4°, 9.8°, 11.4°]
     这是一个自然弯曲姿态, 避免全零伸直状态

6. 坐标系变换详解
   6.1 世界坐标系 → base_link帧:
     位置: ik_target_base = base_link_R⁻¹ @ (ik_target_world - base_link_p)
     朝向: R_ee_base = base_link_R⁻¹ @ R_ee_world (R_ee_world由重定向优化器FK自动求解)

   6.2 base_link_R 的来源:
     R1机器人root绕Z轴旋转180° (四元数[0,0,0,1] wxyz格式)
     right_arm_base_link 继承root的旋转
     base_link_R = right_arm_base_link.get_entity_pose().to_transformation_matrix()[:3,:3]
     base_link_R⁻¹ = base_link_R.T (旋转矩阵的逆等于转置)

   6.3 夹爪朝向获取方式:
     不再手动计算 R_mano2world，而是利用重定向优化器自动求解：
     1. retargeting.retarget(ref_value, fixed_qpos) → sapien_qpos (含dummy free joints)
     2. r1_robot.set_qpos(sapien_qpos) → scene.step() → FK更新
     3. right_gripper_link.get_entity_pose() → ee_quat_world → R_ee_world
     4. r1_robot.set_qpos(saved_qpos) → 恢复
     这与 visualize_hand_object.py 的方式完全一致

7. 四元数格式说明
   - SAPIEN Pose.q: [w, x, y, z] 格式
   - RelaxedIKSolver 输入: [w, x, y, z] 格式
   - RelaxedIKRust (Rust层) 输入: [x, y, z, w] 格式
   - _convert_wxyz_to_xyzw 自动转换: [w,x,y,z] → [x,y,z,w]
   - pytransform3d: [w, x, y, z] 格式
   - 重要: 传入 [0,0,0,1] (wxyz) 会被转换为 [0,0,1,0] (xyzw),
     这是绕Z轴旋转180°, 不是单位四元数!
     单位四元数应为 [1,0,0,0] (wxyz) → [0,0,0,1] (xyzw)

8. 常见问题与调试
   Q: IK解的FK误差很大 (>5cm)?
   A: 检查以下几点:
      1. 四元数是否正确 — 传入手腕朝向而非单位四元数
      2. 坐标系变换 — 确保位置和朝向都在base_link帧
      3. 容差设置 — 旋转容差过小+错误朝向=位置严重偏离
      4. 目标是否可达 — 距离base_link超过0.713m则无解

   Q: IK求解失败 (抛出异常)?
   A: 目标可能超出工作空间, 检查:
      1. dist_to_base = ‖ik_target_base‖
      2. 如果 > ARM_MAX_REACH (0.713m), 目标不可达
      3. 尝试调整 mapping_offset 或 safety_offset

   Q: 运动不平滑?
   A: RelaxedIK本身有连续性约束(基于上一帧), 但可能不够:
      1. LPFilter (α=0.5) 对关节角做低通滤波
      2. TrajectorySmoother 做后处理平滑
      3. 检查 ee_pos_filter (α=0.6) 对IK目标的平滑
================================================================================
"""

import argparse           # 命令行参数解析
import sys                # 系统相关操作
import logging            # 日志记录
import tempfile           # 临时文件操作
from pathlib import Path  # 文件路径处理

import cv2                # OpenCV，用于视频处理
import numpy as np        # 数值计算
import sapien             # SAPIEN物理仿真引擎
from pytransform3d import rotations as pr  # 旋转矩阵和四元数处理

from dataset import DexYCBVideoDataset, YCB_CLASSES  # DexYCB数据集加载
from dex_retargeting import yourdfpy as urdf         # URDF文件处理
from dex_retargeting.constants import HandType, RobotName, OPERATOR2MANO_RIGHT, RetargetingType, get_default_config_path  # 重定向常量
from dex_retargeting.optimizer_utils import LPFilter   # 低通滤波器
from dex_retargeting.retargeting_config import RetargetingConfig  # 重定向配置
from hand_robot_viewer import RobotHandDatasetSAPIENViewer  # 机器人手部数据查看器

# 项目根目录（向上3级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# GalaxeaManipSim项目路径
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")

# 将GalaxeaManipSim加入Python搜索路径，以便导入其中的模块
sys.path.insert(0, str(GALAXEA_SIM_PATH))
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver  # 导入RelaxedIK求解器

# ==================== 全局常量定义 ====================
# RelaxedIK配置文件路径
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"

# 低通滤波器参数（alpha值越大，滤波强度越弱，越接近原始信号）
LP_ALPHA_EE = 0.6       # 末端执行器位置滤波系数
LP_ALPHA_JOINT = 0.5    # 关节角度滤波系数

# 机器人基座旋转：绕Z轴旋转180度，四元数格式 [w, x, y, z]
Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])

SAFETY_DISTANCE = 0.075  # 安全距离：IK目标点离机器人基座的最小距离（米）
WARMUP_FRAMES = 30       # Warmup过渡帧数：平滑地从初始姿态过渡到第一个IK目标

# 机器人臂基座在机器人本地坐标系中的偏移量
ARM_BASE_OFFSET_LOCAL = np.array([0.09193, -0.33649, 0.97171])
ARM_MAX_REACH = 0.713   # 机械臂最大可达距离（米）
COMFORTABLE_REACH = 0.40  # 舒适操作距离（米）

# 舒适目标点在机器人base_link局部坐标系中的位置：前方30cm，下方25cm
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])

# R1右臂关节限位（弧度）：[下限, 上限]
R1_RIGHT_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],   # 关节1
    [0.0, 3.1416],       # 关节2
    [-3.3161, 0.0],      # 关节3
    [-1.5708, 1.5708],   # 关节4
    [-1.5708, 1.5708],   # 关节5
    [-2.8798, 2.8798],   # 关节6
])

# 相机视角预设：不同视角对应的四元数 [w, x, y, z]
CAMERA_QUATS = {
    "behind": [0.0, 0.0, 1.0, 0.0],    # 机器人后方视角
    "front": [1.0, 0.0, 0.0, 0.0],     # 机器人前方视角
    "topdown": [0.7071, 0.0, 0.7071, 0.0],  # 俯视视角
}


class TrajectorySmoother:
    """
    轨迹后处理平滑器 — 五阶段管线 + 迭代收敛

    阶段A: 双向二阶Butterworth低通滤波 (零相位滞后, -40dB/dec衰减)
      比一阶指数滤波(α=0.3)衰减更快, 高频噪声抑制能力提升约2倍
      前向: s1[i] = s1[i-1] + α*(x[i]-s1[i-1]),  s2[i] = s2[i-1] + α*(s1[i]-s2[i-1])
      后向: s1'[i] = s1'[i+1] + α*(s2[i]-s1'[i+1]), s2'[i] = s2'[i+1] + α*(s1'[i]-s2'[i+1])

    阶段B: 速度限幅 (逐关节, 防止角速度超限)
      |Δq[i]/Δt| ≤ v_max  →  Δq[i] = clip(Δq[i], -v_max*dt, +v_max*dt)

    阶段C: 加速度限幅 (逐关节, 防止角加速度超限)
      |Δv[i]/Δt| ≤ a_max  →  Δv[i] = clip(Δv[i], -a_max*dt, +a_max*dt)

    阶段D: 加加速度限幅 (逐关节, 防止加速度突变 — 核心改进)
      |Δa[i]/Δt| ≤ j_max  →  Δa[i] = clip(Δa[i], -j_max*dt, +j_max*dt)
      → 消除运动中的"突然变化"感, 确保加速度连续变化

    阶段E: 迭代收敛 (B→C→D循环, 保证所有约束同时满足)
      单次限幅: 速度限幅可能违反加速度约束, 加速度限幅可能违反速度约束
      迭代限幅: 重复B→C→D直到轨迹变化量 < ε, 确保一致性

    平滑度指标 (含合格判定):
      - 最大关节角速度 (rad/s), 最大关节角加速度 (rad/s²), 最大关节角加加速度 (rad/s³)
      - 平滑度指数 SI = ∫jerk²dt (越小越平滑)
      - 各指标平滑前后改善率
      - 合格判定: 速度≤3.0, 加速度≤8.0, 加加速度≤50.0, SI改善≥50%
    """

    # 平滑度合格阈值
    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,           # 最大角速度 (rad/s)
        "max_acceleration": 8.0,       # 最大角加速度 (rad/s²)
        "max_jerk": 80.0,             # 最大角加加速度 (rad/s³)
        "si_improvement_min": 0.5,    # 最小平滑度改善率 (50%)
    }

    def __init__(
        self,
        fps=30,                          # 视频帧率
        max_velocity=1.5,                # 最大关节角速度 (rad/s)
        max_acceleration=4.0,            # 最大关节角加速度 (rad/s²)
        max_jerk=20.0,                   # 最大关节角加加速度 (rad/s³)
        lp_alpha=0.25,                   # 低通滤波系数
        butterworth_order=2,             # Butterworth滤波器阶数
        max_iterations=10,               # 最大迭代次数
        convergence_eps=1e-5,            # 收敛阈值
    ):
        """初始化轨迹平滑器"""
        self.dt = 1.0 / fps                    # 时间步长
        self.max_velocity = max_velocity       # 角速度限幅
        self.max_acceleration = max_acceleration  # 角加速度限幅
        self.max_jerk = max_jerk              # 角加加速度限幅
        self.lp_alpha = lp_alpha              # 滤波系数
        self.butterworth_order = butterworth_order  # 滤波器阶数
        self.max_iterations = max_iterations  # 最大迭代次数
        self.convergence_eps = convergence_eps  # 收敛阈值

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
        """
        平滑关节轨迹
        
        参数:
            qpos_sequence: 关节位置序列列表
            smooth_indices: 需要平滑的关节索引列表
            
        返回:
            smoothed_sequence: 平滑后的关节位置序列
            metrics: 平滑度指标字典
        """
        n_frames = len(qpos_sequence)
        n_joints = len(smooth_indices)

        # 提取需要平滑的关节数据
        trajectory = np.zeros((n_frames, n_joints))
        valid_mask = np.zeros(n_frames, dtype=bool)
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                trajectory[i] = qpos[smooth_indices]
                valid_mask[i] = True

        # 填充无效帧（用前一个有效帧或第一个有效帧）
        self._fill_invalid_frames(trajectory, valid_mask)
        trajectory_raw = trajectory.copy()  # 保存原始轨迹用于对比

        # 阶段A: 双向低通滤波
        trajectory = self._bidirectional_lowpass(trajectory)
        # 阶段B-E: 迭代速度/加速度/加加速度限幅
        trajectory = self._iterative_clamp(trajectory)

        # 将平滑后的轨迹合并回完整的关节位置序列
        smoothed_sequence = []
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                qpos_new = qpos.copy()
            else:
                # 找到前一个有效帧
                for j in range(i, -1, -1):
                    if qpos_sequence[j] is not None:
                        qpos_new = qpos_sequence[j].copy()
                        break
                else:
                    continue
            qpos_new[smooth_indices] = trajectory[i]

            smoothed_sequence.append(qpos_new)

        # 计算平滑度指标
        metrics = self._compute_metrics(trajectory, trajectory_raw)
        return smoothed_sequence, metrics

    def _fill_invalid_frames(self, trajectory, valid_mask):
        """填充无效帧的关节数据"""
        n_frames = len(trajectory)
        
        # 前向填充：用前一个有效帧填充后续无效帧
        last_valid = 0
        for i in range(n_frames):
            if valid_mask[i]:
                last_valid = i
            else:
                trajectory[i] = trajectory[last_valid]
        
        # 前向填充：用第一个有效帧填充前面的无效帧
        first_valid = np.argmax(valid_mask)
        for i in range(first_valid):
            trajectory[i] = trajectory[first_valid]

    def _bidirectional_lowpass(self, trajectory):
        """双向低通滤波（零相位滞后）"""
        if self.butterworth_order == 1:
            return self._bidirectional_lpf_order1(trajectory)
        elif self.butterworth_order == 2:
            return self._bidirectional_lpf_order2(trajectory)
        else:
            return self._bidirectional_lpf_order2(trajectory)  # 默认二阶

    def _bidirectional_lpf_order1(self, trajectory):
        """一阶双向低通滤波"""
        alpha = self.lp_alpha
        
        # 前向滤波
        forward = np.zeros_like(trajectory)
        forward[0] = trajectory[0]
        for i in range(1, len(trajectory)):
            forward[i] = forward[i - 1] + alpha * (trajectory[i] - forward[i - 1])
        
        # 后向滤波
        backward = np.zeros_like(trajectory)
        backward[-1] = forward[-1]
        for i in range(len(trajectory) - 2, -1, -1):
            backward[i] = backward[i + 1] + alpha * (forward[i] - backward[i + 1])
        
        return backward

    def _bidirectional_lpf_order2(self, trajectory):
        """二阶双向低通滤波（Butterworth近似）"""
        alpha = self.lp_alpha
        n_frames = len(trajectory)
        n_joints = trajectory.shape[1]

        # 前向滤波：两个状态变量
        s1_fwd = np.zeros_like(trajectory)
        s2_fwd = np.zeros_like(trajectory)
        s1_fwd[0] = trajectory[0]
        s2_fwd[0] = trajectory[0]
        for i in range(1, n_frames):
            s1_fwd[i] = s1_fwd[i - 1] + alpha * (trajectory[i] - s1_fwd[i - 1])
            s2_fwd[i] = s2_fwd[i - 1] + alpha * (s1_fwd[i] - s2_fwd[i - 1])

        # 后向滤波
        s1_bwd = np.zeros_like(trajectory)
        s2_bwd = np.zeros_like(trajectory)
        s1_bwd[-1] = s2_fwd[-1]
        s2_bwd[-1] = s2_fwd[-1]
        for i in range(n_frames - 2, -1, -1):
            s1_bwd[i] = s1_bwd[i + 1] + alpha * (s2_fwd[i] - s1_bwd[i + 1])
            s2_bwd[i] = s2_bwd[i + 1] + alpha * (s1_bwd[i] - s2_bwd[i + 1])

        return s2_bwd

    def _clamp_velocity(self, trajectory):
        """速度限幅：限制相邻帧之间的关节角速度"""
        max_delta = self.max_velocity * self.dt  # 每帧最大允许角度变化
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]  # 角度变化量
            clamped = np.clip(delta, -max_delta, max_delta)  # 限幅
            trajectory[i] = trajectory[i - 1] + clamped  # 更新
        return trajectory

    def _clamp_acceleration(self, trajectory):
        """加速度限幅：限制相邻帧之间的角加速度变化"""
        max_delta_v = self.max_acceleration * self.dt  # 每帧速度最大变化量
        for i in range(2, len(trajectory)):
            v_prev = trajectory[i - 1] - trajectory[i - 2]  # 前一帧速度
            v_curr = trajectory[i] - trajectory[i - 1]      # 当前帧速度
            delta_v = v_curr - v_prev                       # 速度变化量
            clamped_dv = np.clip(delta_v, -max_delta_v, max_delta_v)  # 限幅
            v_curr_clamped = v_prev + clamped_dv            # 修正后的当前速度
            trajectory[i] = trajectory[i - 1] + v_curr_clamped  # 更新位置
        return trajectory

    def _clamp_jerk(self, trajectory):
        """加加速度限幅：限制相邻帧之间的角加加速度变化（核心平滑功能）"""
        max_delta_a = self.max_jerk * self.dt  # 每帧加速度最大变化量
        for i in range(3, len(trajectory)):
            # 计算速度
            v_im2 = trajectory[i - 2] - trajectory[i - 3]  # i-2帧速度
            v_im1 = trajectory[i - 1] - trajectory[i - 2]  # i-1帧速度
            v_i = trajectory[i] - trajectory[i - 1]        # i帧速度
            
            # 计算加速度
            a_prev = v_im1 - v_im2                         # 前一帧加速度
            a_curr = v_i - v_im1                           # 当前帧加速度
            
            delta_a = a_curr - a_prev                      # 加速度变化量
            clamped_da = np.clip(delta_a, -max_delta_a, max_delta_a)  # 限幅
            a_curr_clamped = a_prev + clamped_da           # 修正后的当前加速度
            v_i_clamped = v_im1 + a_curr_clamped           # 修正后的当前速度
            trajectory[i] = trajectory[i - 1] + v_i_clamped  # 更新位置
        return trajectory

    def _iterative_clamp(self, trajectory):
        """迭代限幅：循环执行速度/加速度/加加速度限幅直到收敛"""
        for iteration in range(self.max_iterations):
            traj_before = trajectory.copy()  # 保存限幅前的轨迹
            
            # 依次执行三种限幅
            trajectory = self._clamp_velocity(trajectory)
            trajectory = self._clamp_acceleration(trajectory)
            trajectory = self._clamp_jerk(trajectory)
            
            # 检查是否收敛
            max_change = np.max(np.abs(trajectory - traj_before))
            if max_change < self.convergence_eps:
                break  # 收敛，退出迭代
        
        return trajectory

    def _compute_metrics(self, trajectory_smooth, trajectory_raw):
        """
        计算平滑度指标
        
        参数:
            trajectory_smooth: 平滑后的轨迹
            trajectory_raw: 原始轨迹
            
        返回:
            metrics: 包含各种平滑度指标的字典
        """
        dt = self.dt
        
        # 计算原始轨迹的导数
        vel_raw = np.diff(trajectory_raw, axis=0) / dt  # 速度
        acc_raw = np.diff(vel_raw, axis=0) / dt         # 加速度
        jerk_raw = np.diff(acc_raw, axis=0) / dt        # 加加速度
        
        # 计算平滑后轨迹的导数
        vel_smooth = np.diff(trajectory_smooth, axis=0) / dt
        acc_smooth = np.diff(vel_smooth, axis=0) / dt
        jerk_smooth = np.diff(acc_smooth, axis=0) / dt
        
        # 最大值指标
        raw_max_vel = float(np.max(np.abs(vel_raw)))
        raw_max_acc = float(np.max(np.abs(acc_raw))) if len(acc_raw) > 0 else 0.0
        raw_max_jerk = float(np.max(np.abs(jerk_raw))) if len(jerk_raw) > 0 else 0.0
        # 平滑度指数SI（Jerk平方积分）
        raw_si = float(np.sum(jerk_raw ** 2) * dt) if len(jerk_raw) > 0 else 0.0
        
        smooth_max_vel = float(np.max(np.abs(vel_smooth)))
        smooth_max_acc = float(np.max(np.abs(acc_smooth))) if len(acc_smooth) > 0 else 0.0
        smooth_max_jerk = float(np.max(np.abs(jerk_smooth))) if len(jerk_smooth) > 0 else 0.0
        smooth_si = float(np.sum(jerk_smooth ** 2) * dt) if len(jerk_smooth) > 0 else 0.0

        # 逐关节的最大值
        per_joint_max_vel = np.max(np.abs(vel_smooth), axis=0).tolist()
        per_joint_max_acc = np.max(np.abs(acc_smooth), axis=0).tolist() if len(acc_smooth) > 0 else []
        per_joint_max_jerk = np.max(np.abs(jerk_smooth), axis=0).tolist() if len(jerk_smooth) > 0 else []

        # 判断是否合格
        thresholds = self.SMOOTHNESS_THRESHOLDS
        pass_vel = smooth_max_vel <= thresholds["max_velocity"]
        pass_acc = smooth_max_acc <= thresholds["max_acceleration"]
        pass_jerk = smooth_max_jerk <= thresholds["max_jerk"]
        si_improvement = 1.0 - smooth_si / max(raw_si, 1e-12)  # SI改善率
        pass_si = si_improvement >= thresholds["si_improvement_min"]
        all_pass = pass_vel and pass_acc and pass_jerk and pass_si

        return {
            "raw_max_velocity": raw_max_vel,         # 原始最大速度
            "raw_max_acceleration": raw_max_acc,     # 原始最大加速度
            "raw_max_jerk": raw_max_jerk,             # 原始最大加加速度
            "raw_smoothness_index": raw_si,          # 原始平滑度指数
            "smooth_max_velocity": smooth_max_vel,    # 平滑后最大速度
            "smooth_max_acceleration": smooth_max_acc,  # 平滑后最大加速度
            "smooth_max_jerk": smooth_max_jerk,       # 平滑后最大加加速度
            "smooth_smoothness_index": smooth_si,     # 平滑后平滑度指数
            "velocity_reduction": 1.0 - smooth_max_vel / max(raw_max_vel, 1e-6),  # 速度降低率
            "acceleration_reduction": 1.0 - smooth_max_acc / max(raw_max_acc, 1e-6),  # 加速度降低率
            "jerk_reduction": 1.0 - smooth_si / max(raw_si, 1e-12),  # 加加速度降低率
            "per_joint_max_vel": per_joint_max_vel,  # 逐关节最大速度
            "per_joint_max_acc": per_joint_max_acc,  # 逐关节最大加速度
            "per_joint_max_jerk": per_joint_max_jerk,  # 逐关节最大加加速度
            "pass_velocity": pass_vel,               # 速度是否合格
            "pass_acceleration": pass_acc,           # 加速度是否合格
            "pass_jerk": pass_jerk,                 # 加加速度是否合格
            "pass_si_improvement": pass_si,          # SI改善是否合格
            "all_pass": all_pass,                   # 所有指标是否合格
            "si_improvement": si_improvement,       # SI改善率
            "thresholds": thresholds,               # 阈值
            "smoother_params": {                    # 平滑器参数
                "max_velocity": self.max_velocity,
                "max_acceleration": self.max_acceleration,
                "max_jerk": self.max_jerk,
                "lp_alpha": self.lp_alpha,
                "butterworth_order": self.butterworth_order,
                "max_iterations": self.max_iterations,
            },
        }


def _resolve_galaxea_sim_path():
    """
    解析GalaxeaManipSim仓库路径
    
    首先检查环境变量GALAXEA_SIM_PATH，
    然后尝试默认路径。
    
    Returns:
        Path: GalaxeaManipSim路径
    """
    import os
    env_path = os.environ.get("GALAXEA_SIM_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    for candidate in [
        Path("/home/an/robot_world_ws/src/GalaxeaManipSim"),
        Path.home() / "GalaxeaManipSim",
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "无法找到 GalaxeaManipSim 路径。"
        "请设置环境变量 GALAXEA_SIM_PATH 或确保仓库存在于默认位置。"
    )


def _joint_comfort_score(joint_values, joint_limits):
    """
    计算关节舒适度分数
    
    舒适度基于关节角度离中间位置的距离：
    - 1.0: 所有关节都在中间位置（最舒适）
    - 0.0: 所有关节都在极限位置（最不舒适）
    
    Args:
        joint_values: 关节角度数组
        joint_limits: 关节限位数组 [[min, max], ...]
    
    Returns:
        float: 舒适度分数 (0-1)
    """
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
    normalized_dist = np.abs(joint_values - mid) / half_range
    return float(1.0 - np.mean(normalized_dist))


class R1TrackingPipeline:
    """R1手部追踪视频生成管线主类"""

    def __init__(
        self,
        dexycb_dir: str,
        data_id: int = 0,
        output_video: str = "r1_tracking.mp4",
        fps: int = 30,
        view: str = "behind",
        logger: logging.Logger = None,
    ):
        """
        初始化管线
        
        Args:
            dexycb_dir: DexYCB数据集路径
            data_id: 数据ID
            output_video: 输出视频路径
            fps: 帧率
            view: 相机视角 ("behind", "front", "topdown")
            logger: 日志记录器
        """
        self.dexycb_dir = Path(dexycb_dir)
        self.data_id = data_id
        self.output_video = output_video
        self.fps = fps
        self.view = view
        self._galaxea_sim = _resolve_galaxea_sim_path()
        self.logger = logger or logging.getLogger("R1Tracking")

    def run(self, start_frame: int = 0, num_frames: int = 50):
        """
        运行完整的追踪视频生成管线
        
        Args:
            start_frame: 起始帧
            num_frames: 处理的帧数
        """
        self.logger.info("=" * 80)
        self.logger.info(f"DexYCB → Dex Retargeting → RelaxedIK (R1右臂) → 视频  [视角: {self.view}]")
        self.logger.info("=" * 80)

        # ── 阶段1: 数据加载 ──
        self.logger.info("\n[1/7] 加载 DexYCB ...")
        dataset = DexYCBVideoDataset(self.dexycb_dir, hand_type="right")
        sampled_data = dataset[self.data_id]
        hand_pose = sampled_data["hand_pose"]
        total_frames = hand_pose.shape[0]
        self.logger.info(f"  轨迹: {total_frames} 帧, 物体: {[YCB_CLASSES[yid] for yid in sampled_data['ycb_ids']]}")

        actual_frames = min(num_frames, total_frames - start_frame)
        if actual_frames <= 0:
            raise ValueError(f"帧范围无效: start={start_frame}, num={num_frames}, total={total_frames}")

        # ── 阶段2: Dex Retargeting 初始化 ──
        self.logger.info("\n[2/7] 初始化 Dex Retargeting ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        viewer = RobotHandDatasetSAPIENViewer(
            robot_names=[RobotName.r1_full],
            hand_type=HandType.right,
            headless=True,
            retargeting_overrides=dict(
                normal_delta=1e-5,
                huber_delta=0.01,
                target_link_names=[
                    "right_gripper_finger_link1",
                    "right_gripper_finger_link2",
                    "right_gripper_link",
                ],
                target_link_human_indices=np.array([4, 8, 0]),
            ),
        )
        viewer.load_object_hand(sampled_data)
        retargeting = viewer.retargetings[0]
        retarget2sapien = viewer.retarget2sapien[0]
        self.logger.info("  ✓ Dex Retargeting 就绪")

        # ── 阶段3: 手部轨迹分析 ──
        self.logger.info("\n[3/7] 分析手部轨迹 ...")
        wrist_positions, hand_stats = self._analyze_hand_trajectory(
            hand_pose, viewer, start_frame, actual_frames
        )
        self.logger.info(f"  有效帧数: {len(wrist_positions)}")
        self.logger.info(f"  手腕质心:   [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")
        self.logger.info(f"  手腕范围:   X[{hand_stats['range'][0]:.4f}], Y[{hand_stats['range'][1]:.4f}], Z[{hand_stats['range'][2]:.4f}]")

        # ── 阶段4: 机器人放置 ──
        self.logger.info("\n[4/7] 放置 R1 机器人（基座固定地面 Z=0） ...")
        scene = viewer.scene
        r1_robot = viewer.robots[0]

        # 180度Z旋转矩阵
        R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        # 将机器人局部坐标系下的臂基座偏移转换到世界坐标系
        arm_base_offset_world = R_180Z_mat @ ARM_BASE_OFFSET_LOCAL
        centroid = hand_stats["centroid"]

        # 计算机器人根位置，使其在Y方向上与手腕保持适当距离
        robot_root_y = centroid[1] - arm_base_offset_world[1] - 0.3
        robot_root_pos = np.array([0.0, robot_root_y, 0.0])

        # 设置机器人根姿态（绕Z轴旋转180度，背向操作者）
        r1_robot.set_root_pose(sapien.Pose(robot_root_pos.tolist(), Q_180Z.tolist()))
        self.logger.info(f"  机器人基座位置: [{robot_root_pos[0]:.4f}, {robot_root_pos[1]:.4f}, {robot_root_pos[2]:.4f}]")

        # 获取活动关节和索引
        active_joints = r1_robot.get_active_joints()
        joint_names = [j.get_name() for j in active_joints]
        right_arm_indices = [i for i, name in enumerate(joint_names) if "right_arm" in name]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        # 查找右臂末端执行器连杆
        right_ee_link = None
        for link in r1_robot.get_links():
            if "right_gripper_link" in link.get_name():
                right_ee_link = link
                break
        if right_ee_link is None:
            raise RuntimeError("无法找到 R1 右末端连杆 'right_gripper_link'")

        # 设置关节驱动属性（高刚度和阻尼用于位置控制）
        for joint in active_joints:
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        # 设置左臂为自然下垂姿态（简化）
        left_arm_indices = [i for i, name in enumerate(joint_names) if "left_arm" in name]
        initial_qpos = r1_robot.get_qpos().copy()
        left_arm_default = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
        for j, idx in enumerate(left_arm_indices):
            if j < len(left_arm_default):
                initial_qpos[idx] = left_arm_default[j]

        # 设置右臂为RelaxedIK starting_config（自然弯曲姿态，避免全零伸直）
        right_arm_starting = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
        for j, idx in enumerate(right_arm_indices):
            if j < len(right_arm_starting):
                initial_qpos[idx] = right_arm_starting[j]

        # 应用初始关节位置并更新场景
        r1_robot.set_qpos(initial_qpos)

        scene.step()
        scene.update_render()

        # 获取 right_arm_base_link 实际世界位姿
        right_arm_base_link = None
        for link in r1_robot.get_links():
            if "right_arm_base_link" in link.get_name():
                right_arm_base_link = link
                break
        if right_arm_base_link is None:
            raise RuntimeError("无法找到 R1 右臂基座连杆 'right_arm_base_link'")

        base_link_pose = right_arm_base_link.get_entity_pose()
        base_link_p = np.array(base_link_pose.p)
        base_link_q = np.array(base_link_pose.q)
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T

        self.logger.info(f"  right_arm_base_link 位置: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

        # ── 阶段5: 工作空间映射 ──
        self.logger.info("\n[5/7] 计算工作空间映射 ...")
        mapping_info = self._compute_workspace_mapping(
            hand_stats, base_link_p, base_link_R
        )
        self.logger.info(f"  映射偏移: [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")
        self.logger.info(f"  安全偏移: [{mapping_info['safety_offset'][0]:.4f}, {mapping_info['safety_offset'][1]:.4f}, {mapping_info['safety_offset'][2]:.4f}] (距离={SAFETY_DISTANCE}m)")
        self.logger.info(f"  映射后质心到base距离: {mapping_info['mapped_dist_to_base']:.4f}m / 臂展{ARM_MAX_REACH:.3f}m")

        # ── 阶段6: RelaxedIK 初始化 ──
        self.logger.info("\n[6/7] 初始化 RelaxedIK ...")
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        ik_solver.relaxed_ik_right.reset([-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985])
        self.logger.info("  ✓ RelaxedIK 就绪")

        # ── 阶段7+8: 预计算 (含warmup过渡) ──
        self.logger.info(f"\n[7/7] 预计算 {actual_frames} 帧 (含 {WARMUP_FRAMES} 帧warmup) ...")
        qpos_sequence, ik_targets_world, eval_pre = self._precompute(
            hand_pose=hand_pose,
            start_frame=start_frame,
            num_frames=actual_frames,
            viewer=viewer,
            retargeting=retargeting,
            retarget2sapien=retarget2sapien,
            ik_solver=ik_solver,
            r1_robot=r1_robot,
            right_arm_indices=right_arm_indices,
            base_link_p=base_link_p,
            base_link_R_inv=base_link_R_inv,
            mapping_info=mapping_info,
            scene=scene,
        )

        valid = sum(1 for x in qpos_sequence if x is not None)
        self.logger.info(f"  ✓ 预计算完成: {valid}/{actual_frames + WARMUP_FRAMES} 帧有效 (含warmup)")

        # ── 阶段8.5: 轨迹后处理平滑 (仅对数据帧, warmup帧保持原样) ──
        smooth_indices = list(right_arm_indices) + [gripper_idx1, gripper_idx2]
        smoother = TrajectorySmoother(
            fps=self.fps,
            max_velocity=1.5,
            max_acceleration=4.0,
            max_jerk=20.0,
            lp_alpha=0.25,
            butterworth_order=2,
            max_iterations=10,
            convergence_eps=1e-5,
        )
        self.logger.info("\n  轨迹后处理平滑 (仅数据帧, 双向二阶Butterworth LPF + 速度/加速度/加加速度迭代限幅) ...")

        warmup_qpos = qpos_sequence[:WARMUP_FRAMES]
        data_qpos = qpos_sequence[WARMUP_FRAMES:]
        data_smoothed, smooth_metrics = smoother.smooth_trajectory(data_qpos, smooth_indices)
        qpos_sequence = warmup_qpos + data_smoothed

        self.logger.info(f"  ✓ 平滑完成: 速度峰值 {smooth_metrics['smooth_max_velocity']:.2f} rad/s, "
                         f"加速度峰值 {smooth_metrics['smooth_max_acceleration']:.2f} rad/s², "
                         f"加加速度峰值 {smooth_metrics['smooth_max_jerk']:.1f} rad/s³")

        # ── 阶段9: 渲染视频 ──
        eval_render = self._render_video(
            scene, r1_robot, qpos_sequence, ik_targets_world,
            viewer, sampled_data, start_frame, right_ee_link,
            base_link_p, base_link_R_inv, mapping_info,
        )

        # ── 阶段10: 评估报告 ──
        self._output_evaluation(eval_pre, eval_render, hand_stats, mapping_info, valid, actual_frames, smooth_metrics)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("管线执行完成！")
        self.logger.info(f"视频: {self.output_video}")
        log_file_path = Path.cwd() / f"{Path(self.output_video).stem}.log"
        self.logger.info(f"日志: {log_file_path}")
        self.logger.info("=" * 80)

    # ──────────────────────────────────────────────────────────────────
    # 阶段3: 手部轨迹分析
    # 遍历所有帧, 通过MANO计算手腕位置, 统计质心/范围/标准差
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_gripper_orientation_from_hand(joints_world):
        wrist = joints_world[0, :3]
        thumb_tip = joints_world[4, :3]
        index_tip = joints_world[8, :3]
        index_mcp = joints_world[5, :3]
        middle_mcp = joints_world[9, :3]
        ring_mcp = joints_world[13, :3]

        y_axis = index_tip - thumb_tip
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            return None
        y_axis = y_axis / y_norm

        mcp_center = (index_mcp + middle_mcp + ring_mcp) / 3.0
        x_axis = mcp_center - wrist
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            return None
        x_axis = x_axis / x_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-8:
            return None
        z_axis = z_axis / z_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        return np.column_stack([x_axis, y_axis, z_axis])

    def _analyze_hand_trajectory(self, hand_pose, viewer, start_frame, num_frames):
        wrist_positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                continue
            # MANO正运动学: hand_pose → 21个3D关节点
            vertex, joints = viewer._compute_hand_geometry(hand_frame)
            if joints is not None:
                wrist_positions.append(joints[0, :3].copy())  # joint[0] = 手腕

        if not wrist_positions:
            raise RuntimeError("无法从数据中提取有效手腕位置")

        positions = np.array(wrist_positions)
        return wrist_positions, {
            "centroid": np.mean(positions, axis=0),
            "range": np.ptp(positions, axis=0),
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
            "std": np.std(positions, axis=0),
            "num_valid": len(wrist_positions),
        }

    # ──────────────────────────────────────────────────────────────────
    # 阶段5: 工作空间映射
    # 将手部轨迹从DexYCB坐标系平移到R1臂可达空间内
    # ──────────────────────────────────────────────────────────────────
    def _compute_workspace_mapping(self, hand_stats, base_link_p, base_link_R):
        centroid = hand_stats["centroid"]

        # 舒适目标在base_link帧中的位置: 前方30cm, 下方25cm
        comfort_target_world = base_link_R @ COMFORT_TARGET_IN_BASE + base_link_p

        # 映射偏移 = 舒适目标 - 手腕质心 (平移整个轨迹)
        mapping_offset = comfort_target_world - centroid

        # 安全距离: 沿接近方向(从目标→臂基座)偏移7.5cm
        approach_dir = base_link_p - comfort_target_world
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        safety_offset = approach_dir * SAFETY_DISTANCE

        # 验证映射后质心是否在臂展范围内
        mapped_centroid = centroid + mapping_offset + safety_offset
        mapped_in_base = base_link_R.T @ (mapped_centroid - base_link_p)
        mapped_dist = np.linalg.norm(mapped_in_base)

        return {
            "mapping_offset": mapping_offset,
            "safety_offset": safety_offset,
            "comfort_target_base": COMFORT_TARGET_IN_BASE.copy(),
            "comfort_target_world": comfort_target_world,
            "mapped_centroid": mapped_centroid,
            "mapped_dist_to_base": mapped_dist,
            "approach_dir": approach_dir,
        }

    # ──────────────────────────────────────────────────────────────────
    # 阶段7+8: 预计算 (含warmup过渡)
    #
    # Warmup: 从机器人当前关节角线性插值到第一帧IK结果
    # 防止启动时的大幅度跳变
    # ──────────────────────────────────────────────────────────────────
    def _precompute(
        self,
        hand_pose,
        start_frame,
        num_frames,
        viewer,
        retargeting,
        retarget2sapien,
        ik_solver,
        r1_robot,
        right_arm_indices,
        base_link_p,
        base_link_R_inv,
        mapping_info,
        scene,
    ):
        qpos_sequence = []
        ik_targets_world = []
        eval_data = {"ik_errors": [], "joint_values": [], "out_of_reach": 0}

        joint_names = [j.get_name() for j in r1_robot.get_active_joints()]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        # ── Dex Retargeting: 固定关节映射 ──
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        init_sapien_qpos = r1_robot.get_qpos().copy()
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_sapien_qpos[sapien2retarget[retarget_idx]]

        # ── Dex Retargeting: 参考关节索引 (人手→R1夹爪对应点) ──
        ref_indices = retargeting.optimizer.target_link_human_indices

        # ── 优化器内部机器人 + gripper_link索引 (用于FK提取位姿) ──
        internal_robot = retargeting.optimizer.robot
        gripper_link_idx = None
        for li, lname in enumerate(internal_robot.link_names):
            if lname == "right_gripper_link":
                gripper_link_idx = li
                break
        if gripper_link_idx is None:
            raise RuntimeError("优化器内部机器人中找不到 right_gripper_link")

        # ── Warm start: 用第一帧手腕位姿初始化优化器 ──
        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                continue
            vertex, joints = viewer._compute_hand_geometry(hand_frame)
            if joints is None:
                continue
            wrist_quat = rotations.quaternion_from_compact_axis_angle(hand_frame[0, 0:3])
            retargeting.warm_start(
                joints[0, :], wrist_quat,
                hand_type=HandType.right, is_mano_convention=True,
            )
            break

        # ── 保存当前机器人qpos，用于FK后恢复 ──
        saved_qpos = r1_robot.get_qpos().copy()

        mapping_offset = mapping_info["mapping_offset"]
        safety_offset = mapping_info["safety_offset"]

        # ── 先求解第一帧, 获取目标关节角 ──
        first_valid_frame = None
        first_ik_joints = None
        first_gripper1 = 0.0
        first_gripper2 = 0.0
        first_ik_target_world = None

        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                continue
            vertex, joints = viewer._compute_hand_geometry(hand_frame)
            if joints is None:
                continue

            # ② Dex Retargeting: 人手参考点 → R1夹爪关节角
            ref_value = joints[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            sapien_qpos = retarget_qpos[retarget2sapien]
            first_gripper1 = float(sapien_qpos[gripper_idx1])
            first_gripper2 = float(sapien_qpos[gripper_idx2])

            # ②.5 从优化器FK提取夹爪位姿
            # 添加第三约束点(wrist→gripper_link)后，优化器自然约束了朝向
            internal_robot.compute_forward_kinematics(retarget_qpos)
            gripper_pose_fk = internal_robot.get_link_pose(gripper_link_idx)
            gripper_pos_fk = gripper_pose_fk[:3, 3].copy()
            R_ee_world_fk = gripper_pose_fk[:3, :3].copy()

            # ③ 工作空间映射: 手腕→IK目标
            ik_target_world_raw = gripper_pos_fk + mapping_offset + safety_offset
            first_ik_target_world = ik_target_world_raw.copy()

            # ④ 变换到 base_link 帧 (IK求解器坐标系)
            ik_target_base = base_link_R_inv @ (ik_target_world_raw - base_link_p)
            R_ee_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
            try:
                first_ik_joints = np.array(
                    ik_solver.solve_position_right(
                        ik_target_base.tolist(), ee_quat_base.tolist()
                    )
                )
                first_ik_target_base = ik_target_base.copy()
                first_ee_quat_base = ee_quat_base.copy()
                first_valid_frame = probe_idx
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("无法求解任何有效帧的IK, 请检查数据和工作空间映射")

        # ── IK预热：对第一个目标点多次迭代，让RelaxedIK收敛 ──
        ik_warmup_iters = 200
        self.logger.info(f"  IK预热：对第一个目标点迭代 {ik_warmup_iters} 次 ...")
        for i in range(ik_warmup_iters):
            first_ik_joints = np.array(
                ik_solver.solve_position_right(
                    first_ik_target_base.tolist(), first_ee_quat_base.tolist()
                )
            )
        self.logger.info(f"  IK预热完成，关节角: {first_ik_joints}")

        # ── 阶段8: Warmup过渡 ──
        # 获取当前机器人右臂关节角
        current_right_joints = np.array([init_sapien_qpos[i] for i in right_arm_indices])

        self.logger.info(f"  Warmup: 从当前关节角过渡到第一帧IK结果 ({WARMUP_FRAMES}帧)")
        self.logger.info(f"    当前关节角: [{', '.join(f'{np.degrees(j):.1f}°' for j in current_right_joints)}]")
        self.logger.info(f"    目标关节角: [{', '.join(f'{np.degrees(j):.1f}°' for j in first_ik_joints)}]")

        # 初始化LPFilter, 从当前关节角开始
        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        ee_pos_filter.next(first_ik_target_world)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_right_joints)

        # 线性插值warmup帧
        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            # 使用smoothstep插值, 避免线性插值的速度突变
            t_smooth = t * t * (3 - 2 * t)
            interp_joints = current_right_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp_joints = joint_filter.next(interp_joints)

            r1_qpos = r1_robot.get_qpos().copy()
            for j, idx in enumerate(right_arm_indices):
                r1_qpos[idx] = interp_joints[j]
            # warmup期间夹爪保持打开
            r1_qpos[gripper_idx1] = 0.04
            r1_qpos[gripper_idx2] = -0.04
            qpos_sequence.append(r1_qpos)
            ik_targets_world.append(first_ik_target_world.copy())

        # ── 阶段7: 正式预计算所有帧 ──
        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx
            hand_frame = hand_pose[global_idx]

            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            # ① MANO正运动学: hand_pose → vertex, joints
            vertex, joints = viewer._compute_hand_geometry(hand_frame)
            if joints is None:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            # ② Dex Retargeting: 人手参考点 → R1夹爪关节角
            ref_value = joints[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            sapien_qpos = retarget_qpos[retarget2sapien]
            gripper1 = float(sapien_qpos[gripper_idx1])
            gripper2 = float(sapien_qpos[gripper_idx2])

            # ②.5 从优化器FK提取夹爪位姿
            internal_robot.compute_forward_kinematics(retarget_qpos)
            gripper_pose_fk = internal_robot.get_link_pose(gripper_link_idx)
            gripper_pos_fk = gripper_pose_fk[:3, 3].copy()
            R_ee_world_fk = gripper_pose_fk[:3, :3].copy()

            # ③ 工作空间映射 + LPFilter平滑
            ik_target_world_raw = gripper_pos_fk + mapping_offset + safety_offset
            ik_target_world = ee_pos_filter.next(ik_target_world_raw)
            ik_targets_world.append(ik_target_world.copy())

            # ④ 变换到 base_link 帧 (IK求解器坐标系)
            ik_target_base = base_link_R_inv @ (ik_target_world - base_link_p)
            dist_to_base = np.linalg.norm(ik_target_base)

            if dist_to_base > ARM_MAX_REACH:
                eval_data["out_of_reach"] += 1

            if local_idx == 0:
                self.logger.debug(f"\n  === IK 坐标变换调试 (帧 {global_idx}) ===")
                self.logger.debug(f"  FK夹爪位置(世界帧): [{gripper_pos_fk[0]:.4f}, {gripper_pos_fk[1]:.4f}, {gripper_pos_fk[2]:.4f}]")
                self.logger.debug(f"  IK目标(世界帧):     [{ik_target_world[0]:.4f}, {ik_target_world[1]:.4f}, {ik_target_world[2]:.4f}]")
                self.logger.debug(f"  IK目标(base帧):     [{ik_target_base[0]:.4f}, {ik_target_base[1]:.4f}, {ik_target_base[2]:.4f}]")
                ee_quat_debug = pr.quaternion_from_matrix(R_ee_world_fk)
                self.logger.debug(f"  FK夹爪朝向(世界帧): [{ee_quat_debug[0]:.4f}, {ee_quat_debug[1]:.4f}, {ee_quat_debug[2]:.4f}, {ee_quat_debug[3]:.4f}]")
                self.logger.debug(f"  距离base_link:      {dist_to_base:.4f}m {'✓' if dist_to_base < ARM_MAX_REACH else '✗ 超出'}")

            # ④ RelaxedIK: IK目标(base帧) + 末端朝向(base帧) → 6个关节角
            # 朝向来源：优化器FK (3约束点: thumb+index+wrist → 自然约束朝向)
            R_ee_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
            try:
                right_joints = np.array(
                    ik_solver.solve_position_right(
                        ik_target_base.tolist(), ee_quat_base.tolist()
                    )
                )
            except Exception as exc:
                self.logger.warning(f"  帧 {global_idx}: IK 失败 - {exc}")
                qpos_sequence.append(None)
                eval_data["ik_errors"].append(str(exc))
                continue

            # 关节角LPFilter平滑 (减少抖动)
            right_joints = joint_filter.next(right_joints)

            eval_data["joint_values"].append(right_joints.copy())

            # IK→FK验证: 用SAPIEN的FK检查IK解是否到达目标
            if local_idx == 0 or local_idx == num_frames - 1:
                self.logger.info(f"  帧{local_idx} IK目标(base帧): {ik_target_base}, 关节角: [{', '.join(f'{np.degrees(j):.1f}°' for j in right_joints)}]")

            # ⑤ 组装qpos: 右臂关节角 + 夹爪关节角
            r1_qpos = r1_robot.get_qpos().copy()
            if len(right_joints) == len(right_arm_indices):
                for j, idx in enumerate(right_arm_indices):
                    r1_qpos[idx] = right_joints[j]
            r1_qpos[gripper_idx1] = gripper1
            r1_qpos[gripper_idx2] = gripper2
            qpos_sequence.append(r1_qpos)

            if (local_idx + 1) % 10 == 0:
                self.logger.info(f"  已计算 {local_idx + 1}/{num_frames} 帧 ...")

        return qpos_sequence, ik_targets_world, eval_data

    # ──────────────────────────────────────────────────────────────────
    # 阶段9: 渲染视频
    # 逐帧: set_qpos → scene.step() → camera.take_picture() → 写入
    # ──────────────────────────────────────────────────────────────────
    def _render_video(
        self, scene, r1_robot, qpos_sequence, ik_targets_world,
        viewer, sampled_data, start_frame, right_ee_link,
        base_link_p, base_link_R_inv, mapping_info,
    ):
        self.logger.info(f"\n渲染视频 → {self.output_video}  [视角: {self.view}]")

        active_joints = r1_robot.get_active_joints()
        joint_names = [j.get_name() for j in active_joints]
        robot_root = np.array(r1_robot.get_root_pose().p)

        if self.view == "behind":
            camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
        elif self.view == "front":
            camera_pos = robot_root + np.array([-2.5, 0.0, 1.2])
        elif self.view == "topdown":
            camera_pos = robot_root + np.array([0.0, 0.0, 4.0])
        else:
            camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
        camera_quat = CAMERA_QUATS.get(self.view, CAMERA_QUATS["behind"])

        camera = scene.add_camera(
            name="main",
            width=1920,
            height=1080,
            fovy=np.deg2rad(60),
            near=0.01,
            far=200.0,
        )
        camera.set_local_pose(sapien.Pose(camera_pos.tolist(), camera_quat))
        self.logger.info(f"  相机位置: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")

        render_fps = int(self.fps * 0.75)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_video, fourcc, render_fps,
            (camera.get_width(), camera.get_height()),
        )
        self.logger.info(f"  视频帧率: {render_fps} (0.75x慢放, 原始{self.fps})")

        hand_pose = sampled_data["hand_pose"]
        object_pose = sampled_data["object_pose"]
        num_ycb = len(sampled_data["ycb_ids"])

        fk_errors = []
        fk_details = []
        comfort_scores = []

        ee_trajectory = []
        ik_target_trajectory = []
        wrist_trajectory = []

        mapping_offset = mapping_info["mapping_offset"]
        safety_offset = mapping_info["safety_offset"]

        coord_display_origin = robot_root + np.array([-0.8, 0.0, 0.0])

        for frame_idx, qpos in enumerate(qpos_sequence):
            is_warmup = frame_idx < WARMUP_FRAMES
            data_frame_idx = frame_idx - WARMUP_FRAMES
            global_idx = start_frame + max(data_frame_idx, 0)

            if qpos is not None:
                r1_robot.set_qpos(qpos)
                for joint in active_joints:
                    joint.set_drive_target(qpos[joint_names.index(joint.get_name())])

            if not is_warmup and data_frame_idx >= 0:
                hand_frame = hand_pose[global_idx]
            else:
                hand_frame = hand_pose[start_frame]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            vertex, joints = viewer._compute_hand_geometry(hand_frame)
            if vertex is not None:
                viewer._update_hand(vertex)

            if not is_warmup and data_frame_idx >= 0 and global_idx < object_pose.shape[0]:
                obj_frame = object_pose[global_idx]
            else:
                obj_frame = object_pose[start_frame]
            camera_pose = viewer.camera_pose
            for k in range(num_ycb):
                pos_quat = obj_frame[k]
                pose = camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                viewer.objects[k].set_pose(pose)

            scene.step()
            scene.update_render()

            ee_pos = None
            ee_quat_wxyz = None
            wrist_pos = None
            wrist_rot = None

            if not is_warmup and qpos is not None:
                ee_pose = right_ee_link.get_entity_pose()
                ee_pos = np.array(ee_pose.p)
                ee_quat_wxyz = np.array(ee_pose.q)
                ee_trajectory.append(ee_pos.copy())

                if joints is not None:
                    wrist_pos = joints[0, :3].copy()
                    wrist_rot = joints[0, 3:].reshape(3, 3).copy() if joints.shape[1] >= 12 else None
                    wrist_trajectory.append(wrist_pos.copy())

                if ik_targets_world[frame_idx] is not None:
                    ik_target_trajectory.append(ik_targets_world[frame_idx].copy())

                joint_names_list = [j.get_name() for j in r1_robot.get_active_joints()]
                right_arm_idx_list = [i for i, name in enumerate(joint_names_list) if "right_arm" in name]
                current_qpos = r1_robot.get_qpos()
                right_joints = np.array([current_qpos[i] for i in right_arm_idx_list])
                comfort = _joint_comfort_score(right_joints, R1_RIGHT_JOINT_LIMITS)
                comfort_scores.append(comfort)

                is_last_data_frame = (frame_idx == len(qpos_sequence) - 1)
                if is_last_data_frame and ik_targets_world[frame_idx] is not None:
                    target_pos = ik_targets_world[frame_idx]
                    error = np.linalg.norm(ee_pos - target_pos)
                    fk_errors.append(error)
                    fk_details.append({
                        "frame": frame_idx,
                        "ee_pos": ee_pos.copy(),
                        "target_pos": target_pos.copy(),
                        "error": error,
                        "right_joints": right_joints.copy(),
                        "comfort": comfort,
                    })

            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            bgr = np.ascontiguousarray(rgb[..., ::-1])

            bgr = self._draw_annotations(
                bgr, frame_idx, qpos_sequence, is_warmup,
                ee_trajectory, ik_target_trajectory, wrist_trajectory,
                robot_root, base_link_p, base_link_R_inv, mapping_info,
                camera, coord_display_origin,
                ee_pos, ee_quat_wxyz, wrist_pos, wrist_rot,
                hand_pose, start_frame, joints
            )

            writer.write(bgr)

            if (frame_idx + 1) % 10 == 0:
                self.logger.info(f"  已渲染 {frame_idx + 1}/{len(qpos_sequence)} 帧 ...")

        writer.release()
        self.logger.info(f"✓ 视频已保存: {self.output_video}")
        self.logger.info(f"  总帧数: {len(qpos_sequence)} (warmup={WARMUP_FRAMES} + 数据={len(qpos_sequence)-WARMUP_FRAMES}), 视角: {self.view}")

        return {
            "fk_errors": fk_errors,
            "fk_details": fk_details,
            "comfort_scores": comfort_scores,
        }

    def _draw_annotations(
        self, bgr, frame_idx, qpos_sequence, is_warmup,
        ee_trajectory, ik_target_trajectory, wrist_trajectory,
        robot_root, base_link_p, base_link_R_inv, mapping_info,
        camera, coord_display_origin,
        ee_pos, ee_quat_wxyz, wrist_pos, wrist_rot,
        hand_pose, start_frame, mano_joints=None
    ):
        """
        在视频帧上绘制可视化标注信息
        
        绘制内容包括：
        - Warmup/数据帧标签和进度条
        - 坐标系可视化（WORLD、ROBOT_BASE、ARM_BASE、EE、HAND）
        - 轨迹线（末端轨迹、IK目标轨迹、手腕轨迹）
        - 当前位置标记
        - 右侧数值面板（位置、误差、可达性）
        - 图例说明
        
        Args:
            bgr: 输入的BGR图像
            frame_idx: 当前帧索引
            qpos_sequence: 完整的关节位置序列
            is_warmup: 是否为warmup阶段
            ee_trajectory: 机器人末端轨迹列表
            ik_target_trajectory: IK目标轨迹列表
            wrist_trajectory: 手腕轨迹列表
            robot_root: 机器人根位置
            base_link_p: 右臂基座位置
            base_link_R_inv: 右臂基座旋转矩阵的逆
            mapping_info: 工作空间映射信息
            camera: SAPIEN相机对象
            coord_display_origin: 坐标系显示原点
            ee_pos: 当前末端位置
            ee_quat_wxyz: 当前末端姿态（wxyz格式）
            wrist_pos: 当前手腕位置
            wrist_rot: 当前手腕旋转矩阵
            
        Returns:
            标注后的BGR图像
        """
        h, w = bgr.shape[:2]
        data_frame_idx = frame_idx - WARMUP_FRAMES
        axis_len = 0.15

        # Warmup阶段绘制：显示Warmup进度条和文字
        if is_warmup:
            t = (frame_idx + 1) / WARMUP_FRAMES
            cv2.rectangle(bgr, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(bgr, f"Warmup {frame_idx+1}/{WARMUP_FRAMES} ({t*100:.0f}%)",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            progress_w = int(w * t)
            cv2.rectangle(bgr, (0, 50), (progress_w, 56), (0, 200, 255), -1)
        else:
            cv2.rectangle(bgr, (0, 0), (w, 80), (0, 0, 0), -1)

            if data_frame_idx == 0:
                cv2.rectangle(bgr, (0, 80), (w, 140), (0, 100, 255), -1)
                cv2.putText(bgr, ">>> MAPPING START <<<",
                            (w//2 - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.putText(bgr, f"Frame {data_frame_idx+1}  (global: {frame_idx})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if ee_trajectory and ik_target_trajectory:
                err_cm = np.linalg.norm(ee_trajectory[-1] - ik_target_trajectory[-1]) * 100
                err_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK: {err_cm:.1f}cm",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)

            view_label = {"behind": "Behind", "front": "Front", "topdown": "Top-down"}.get(self.view, self.view)
            cv2.putText(bgr, f"View: {view_label}",
                        (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(bgr, f"0.75x speed",
                        (w - 250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # ── 坐标系可视化 ──
        self._draw_axes(bgr, coord_display_origin, np.eye(3), axis_len, camera,
                        "WORLD", (200, 200, 200),
                        f"[{coord_display_origin[0]:.2f},{coord_display_origin[1]:.2f},{coord_display_origin[2]:.2f}]")

        robot_root_R = pr.matrix_from_quaternion([0.0, 0.0, 1.0, 0.0])
        self._draw_axes(bgr, robot_root, robot_root_R, axis_len * 0.8, camera,
                        "ROBOT_BASE", (200, 200, 0),
                        f"[{robot_root[0]:.2f},{robot_root[1]:.2f},{robot_root[2]:.2f}]")

        self._draw_axes(bgr, base_link_p, base_link_R_inv.T, axis_len * 0.8, camera,
                        "ARM_BASE", (0, 200, 200),
                        f"[{base_link_p[0]:.2f},{base_link_p[1]:.2f},{base_link_p[2]:.2f}]")

        if ee_pos is not None and ee_quat_wxyz is not None:
            ee_R = pr.matrix_from_quaternion(ee_quat_wxyz)
            self._draw_axes(bgr, ee_pos, ee_R, axis_len * 0.6, camera,
                            "EE", (0, 255, 0),
                            f"[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]")

        if wrist_pos is not None:
            wr = wrist_rot if wrist_rot is not None else np.eye(3)
            self._draw_axes(bgr, wrist_pos, wr, axis_len * 0.6, camera,
                            "HAND", (0, 165, 255),
                            f"[{wrist_pos[0]:.3f},{wrist_pos[1]:.3f},{wrist_pos[2]:.3f}]")

        # ── 轨迹线 ──
        if len(ee_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in ee_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                alpha_val = max(0.3, i / len(pts_2d))
                color = (0, int(255 * alpha_val), int(255 * (1 - alpha_val)))
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], color, 2)

        if len(ik_target_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in ik_target_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (255, 255, 0), 1)

        if len(wrist_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in wrist_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (0, 165, 255), 1)

        # ── 当前位置标记+数值 ──
        if ee_trajectory:
            ee_2d = self._project_point(ee_trajectory[-1], camera)
            if ee_2d is not None:
                cv2.circle(bgr, ee_2d, 6, (0, 255, 0), -1)
                cv2.circle(bgr, ee_2d, 6, (255, 255, 255), 1)

        if ik_target_trajectory:
            tgt_2d = self._project_point(ik_target_trajectory[-1], camera)
            if tgt_2d is not None:
                cv2.circle(bgr, tgt_2d, 6, (0, 255, 255), -1)
                cv2.circle(bgr, tgt_2d, 6, (255, 255, 255), 1)

        if wrist_trajectory:
            w_2d = self._project_point(wrist_trajectory[-1], camera)
            if w_2d is not None:
                cv2.circle(bgr, w_2d, 5, (0, 165, 255), -1)
                cv2.circle(bgr, w_2d, 5, (255, 255, 255), 1)

        # ── MANO参考点: 拇指尖(4)红色, 食指尖(8)蓝色 ──
        if mano_joints is not None and not is_warmup:
            thumb_tip_2d = self._project_point(mano_joints[4, :3], camera)
            index_tip_2d = self._project_point(mano_joints[8, :3], camera)
            if thumb_tip_2d is not None:
                cv2.circle(bgr, thumb_tip_2d, 7, (0, 0, 255), -1)
                cv2.circle(bgr, thumb_tip_2d, 7, (255, 255, 255), 1)
                cv2.putText(bgr, "4", (thumb_tip_2d[0] + 8, thumb_tip_2d[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if index_tip_2d is not None:
                cv2.circle(bgr, index_tip_2d, 7, (255, 0, 0), -1)
                cv2.circle(bgr, index_tip_2d, 7, (255, 255, 255), 1)
                cv2.putText(bgr, "8", (index_tip_2d[0] + 8, index_tip_2d[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # ── 右侧数值面板 ──
        if not is_warmup:
            panel_x = w - 380
            panel_y = 90
            cv2.rectangle(bgr, (panel_x, panel_y), (w, panel_y + 280), (0, 0, 0), -1)
            cv2.putText(bgr, "Position Data (m):", (panel_x + 5, panel_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            row = 0
            if ee_pos is not None:
                cv2.putText(bgr, f"EE:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
                cv2.putText(bgr, f"{ee_pos[0]:.3f} {ee_pos[1]:.3f} {ee_pos[2]:.3f}",
                            (panel_x + 40, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

            if wrist_pos is not None:
                cv2.putText(bgr, f"HAND:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
                cv2.putText(bgr, f"{wrist_pos[0]:.3f} {wrist_pos[1]:.3f} {wrist_pos[2]:.3f}",
                            (panel_x + 55, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

            if ik_target_trajectory:
                ik_p = ik_target_trajectory[-1]
                cv2.putText(bgr, f"IK:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                cv2.putText(bgr, f"{ik_p[0]:.3f} {ik_p[1]:.3f} {ik_p[2]:.3f}",
                            (panel_x + 40, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

            cv2.putText(bgr, f"BASE:", (panel_x + 5, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
            cv2.putText(bgr, f"{base_link_p[0]:.3f} {base_link_p[1]:.3f} {base_link_p[2]:.3f}",
                        (panel_x + 55, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            row += 1

            cv2.putText(bgr, f"ROOT:", (panel_x + 5, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
            cv2.putText(bgr, f"{robot_root[0]:.3f} {robot_root[1]:.3f} {robot_root[2]:.3f}",
                        (panel_x + 55, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            row += 1

            if ee_pos is not None and ik_target_trajectory:
                err = np.linalg.norm(ee_pos - ik_target_trajectory[-1])
                err_color = (0, 255, 0) if err < 0.02 else (0, 255, 255) if err < 0.05 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK err: {err*100:.2f}cm", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_color, 1)
                row += 1

            if ee_pos is not None and wrist_pos is not None:
                dist = np.linalg.norm(ee_pos - wrist_pos)
                cv2.putText(bgr, f"EE-HAND: {dist*100:.1f}cm", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

            if ee_pos is not None and base_link_p is not None:
                reach = np.linalg.norm(ee_pos - base_link_p)
                cv2.putText(bgr, f"Reach: {reach:.3f}m / {ARM_MAX_REACH:.3f}m",
                            (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

        # ── 图例 ──
        if not is_warmup:
            legend_y = h - 110
            cv2.rectangle(bgr, (10, legend_y - 5), (340, h - 5), (0, 0, 0), -1)
            cv2.circle(bgr, (25, legend_y + 10), 5, (0, 255, 0), -1)
            cv2.putText(bgr, "Robot EE", (40, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.circle(bgr, (25, legend_y + 30), 5, (0, 255, 255), -1)
            cv2.putText(bgr, "IK target", (40, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.circle(bgr, (25, legend_y + 50), 5, (0, 165, 255), -1)
            cv2.putText(bgr, "Hand wrist", (40, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(bgr, (170, legend_y + 10), (200, legend_y + 10), (0, 200, 128), 2)
            cv2.putText(bgr, "EE trail", (205, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(bgr, (170, legend_y + 30), (200, legend_y + 30), (255, 255, 0), 1)
            cv2.putText(bgr, "IK trail", (205, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(bgr, (170, legend_y + 50), (200, legend_y + 50), (0, 165, 255), 1)
            cv2.putText(bgr, "Hand trail", (205, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            coord_y = legend_y - 100
            cv2.rectangle(bgr, (10, coord_y - 5), (320, legend_y - 10), (0, 0, 0), -1)
            cv2.putText(bgr, "Coordinate Frames:", (20, coord_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(bgr, "WORLD (white)", (30, coord_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(bgr, "ROBOT_BASE (yellow)", (30, coord_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            cv2.putText(bgr, "ARM_BASE (cyan)", (30, coord_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
            cv2.putText(bgr, "EE (green)", (30, coord_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(bgr, "HAND (orange)", (30, coord_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        return bgr

    def _draw_axes(self, bgr, origin, R, length, camera, label, label_color, pos_text=""):
        """
        在图像上绘制3D坐标系的坐标轴
        
        坐标系颜色约定：
        - X轴：红色 (0, 0, 255)
        - Y轴：绿色 (0, 255, 0)  
        - Z轴：蓝色 (255, 0, 0)
        
        Args:
            bgr: 输入的BGR图像
            origin: 坐标系原点在3D空间中的位置
            R: 坐标系的旋转矩阵（3x3）
            length: 坐标轴的长度（米）
            camera: SAPIEN相机对象（用于投影）
            label: 坐标系标签文字
            label_color: 标签文字颜色 (B, G, R)
            pos_text: 可选的位置文本（显示在标签上方）
        """
        origin_2d = self._project_point(origin, camera)
        if origin_2d is None:
            return
        
        # 计算坐标轴终点在3D空间中的位置
        x_end = origin + R[:, 0] * length
        y_end = origin + R[:, 1] * length
        z_end = origin + R[:, 2] * length
        
        # 将3D点投影到2D图像平面
        x_2d = self._project_point(x_end, camera)
        y_2d = self._project_point(y_end, camera)
        z_2d = self._project_point(z_end, camera)
        
        lw = 2  # 线条宽度
        
        # 绘制X轴（红色）
        if x_2d is not None:
            cv2.line(bgr, origin_2d, x_2d, (0, 0, 255), lw)
            cv2.putText(bgr, "X", (x_2d[0]+3, x_2d[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 绘制Y轴（绿色）
        if y_2d is not None:
            cv2.line(bgr, origin_2d, y_2d, (0, 255, 0), lw)
            cv2.putText(bgr, "Y", (y_2d[0]+3, y_2d[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 绘制Z轴（蓝色）
        if z_2d is not None:
            cv2.line(bgr, origin_2d, z_2d, (255, 0, 0), lw)
            cv2.putText(bgr, "Z", (z_2d[0]+3, z_2d[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 绘制坐标系标签
        cv2.putText(bgr, label, (origin_2d[0]-15, origin_2d[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
        
        # 绘制位置文本（如果有）
        if pos_text:
            cv2.putText(bgr, pos_text, (origin_2d[0]-15, origin_2d[1]-22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)

    def _project_point(self, point_3d, camera):
        """
        将3D空间点投影到2D图像平面
        
        投影过程：
        1. 使用相机外参矩阵将3D点从世界坐标系变换到相机坐标系
        2. 使用相机内参矩阵将相机坐标系点投影到图像坐标系
        
        Args:
            point_3d: 3D空间点 [x, y, z]
            camera: SAPIEN相机对象
            
        Returns:
            投影后的2D图像坐标 (u, v)，如果点在相机后方或超出图像范围则返回None
        """
        try:
            # 获取相机外参矩阵（世界→相机变换）
            ext = camera.get_extrinsic_matrix().astype(np.float64)
            # 获取相机内参矩阵（相机→图像投影）
            int_mat = camera.get_intrinsic_matrix().astype(np.float64)
            
            R = ext[:3, :3]  # 旋转部分
            t = ext[:3, 3]   # 平移部分
            
            # 将3D点从世界坐标系变换到相机坐标系
            p_cam = R @ point_3d + t
            
            # 如果点在相机后方（z≤0），不投影
            if p_cam[2] <= 0.01:
                return None
            
            # 使用内参矩阵投影到图像平面
            uv = int_mat @ p_cam
            u = int(uv[0] / uv[2])
            v = int(uv[1] / uv[2])
            
            w_px = camera.get_width()
            h_px = camera.get_height()
            
            # 检查投影点是否在图像范围内
            if 0 <= u < w_px and 0 <= v < h_px:
                return (u, v)
            return None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # 阶段10: 评估报告
    # ──────────────────────────────────────────────────────────────────
    def _output_evaluation(self, eval_pre, eval_render, hand_stats, mapping_info, valid, total, smooth_metrics=None):
        """
        输出完整的评估报告
        
        报告包含以下7个部分：
        1. 手部轨迹分析：手腕质心、运动范围、标准差
        2. 工作空间映射：映射偏移、安全距离、可达性
        3. IK求解统计：有效帧数、失败帧数、超出臂展帧数
        4. FK验证：最后一帧位置误差
        5. 关节舒适度：平均和最低舒适度
        6. 运动平滑度：速度、加速度、加加速度的平滑前后对比
        7. 最后一帧详细验证
        
        Args:
            eval_pre: 预计算阶段的评估数据
            eval_render: 渲染阶段的评估数据
            hand_stats: 手部轨迹统计信息
            mapping_info: 工作空间映射信息
            valid: 有效帧数
            total: 总数据帧数（不含warmup）
            smooth_metrics: 轨迹平滑指标（可选）
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("评估报告")
        self.logger.info("=" * 80)

        # 第1部分：手部轨迹分析
        self.logger.info("\n── 1. 手部轨迹分析 ──")
        self.logger.info(f"  有效帧数:     {hand_stats['num_valid']}")
        self.logger.info(f"  手腕质心:     [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")
        self.logger.info(f"  运动范围:     X={hand_stats['range'][0]:.4f}m, Y={hand_stats['range'][1]:.4f}m, Z={hand_stats['range'][2]:.4f}m")
        self.logger.info(f"  运动标准差:   X={hand_stats['std'][0]:.4f}m, Y={hand_stats['std'][1]:.4f}m, Z={hand_stats['std'][2]:.4f}m")

        # 第2部分：工作空间映射
        self.logger.info("\n── 2. 工作空间映射 ──")
        self.logger.info(f"  映射偏移:     [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")
        self.logger.info(f"  安全距离:     {SAFETY_DISTANCE:.3f}m")
        self.logger.info(f"  映射后质心到base: {mapping_info['mapped_dist_to_base']:.4f}m / 臂展{ARM_MAX_REACH:.3f}m")

        # 第3部分：IK求解统计
        self.logger.info("\n── 3. IK 求解统计 ──")
        self.logger.info(f"  有效帧:       {valid}/{total + WARMUP_FRAMES} (含{WARMUP_FRAMES}帧warmup)")
        self.logger.info(f"  IK失败帧:     {len(eval_pre['ik_errors'])}")
        self.logger.info(f"  超出臂展帧:   {eval_pre['out_of_reach']}/{total}")

        fk_errors = eval_render["fk_errors"]
        fk_details = eval_render["fk_details"]
        if fk_errors and fk_details:
            self.logger.info("\n── 4. FK 验证（最后一帧位置误差） ──")
            d = fk_details[-1]
            self.logger.info(f"  末端位置:     [{d['ee_pos'][0]:.4f}, {d['ee_pos'][1]:.4f}, {d['ee_pos'][2]:.4f}]")
            self.logger.info(f"  IK目标位置:   [{d['target_pos'][0]:.4f}, {d['target_pos'][1]:.4f}, {d['target_pos'][2]:.4f}]")
            self.logger.info(f"  位置误差:     {d['error']*100:.2f} cm")

        comfort_scores = eval_render["comfort_scores"]
        if comfort_scores:
            self.logger.info("\n── 5. 关节舒适度 ──")
            self.logger.info(f"  平均舒适度:   {np.mean(comfort_scores):.4f} (1.0=最佳, 0.0=最差)")
            self.logger.info(f"  最低舒适度:   {np.min(comfort_scores):.4f}")

        if smooth_metrics:
            self.logger.info("\n── 6. 运动平滑度 ──")
            self.logger.info(f"  ┌─────────────────────┬──────────┬──────────┬──────────┬──────┐")
            self.logger.info(f"  │ 指标                │ 平滑前   │ 平滑后   │ 改善率   │ 判定 │")
            self.logger.info(f"  ├─────────────────────┼──────────┼──────────┼──────────┼──────┤")
            vel_pass = "✓" if smooth_metrics["pass_velocity"] else "✗"
            acc_pass = "✓" if smooth_metrics["pass_acceleration"] else "✗"
            jerk_pass = "✓" if smooth_metrics["pass_jerk"] else "✗"
            si_pass = "✓" if smooth_metrics["pass_si_improvement"] else "✗"
            self.logger.info(f"  │ 最大角速度(rad/s)   │ {smooth_metrics['raw_max_velocity']:8.2f} │ {smooth_metrics['smooth_max_velocity']:8.2f} │ {smooth_metrics['velocity_reduction']*100:6.1f}%  │  {vel_pass}   │")
            self.logger.info(f"  │ 最大角加速度(rad/s²)│ {smooth_metrics['raw_max_acceleration']:8.2f} │ {smooth_metrics['smooth_max_acceleration']:8.2f} │ {smooth_metrics['acceleration_reduction']*100:6.1f}%  │  {acc_pass}   │")
            self.logger.info(f"  │ 最大加加速度(rad/s³)│ {smooth_metrics['raw_max_jerk']:8.1f} │ {smooth_metrics['smooth_max_jerk']:8.1f} │ {(1.0-smooth_metrics['smooth_max_jerk']/max(smooth_metrics['raw_max_jerk'],1e-6))*100:6.1f}%  │  {jerk_pass}   │")
            self.logger.info(f"  │ 平滑度指数(SI)      │ {smooth_metrics['raw_smoothness_index']:8.1f} │ {smooth_metrics['smooth_smoothness_index']:8.1f} │ {smooth_metrics['jerk_reduction']*100:6.1f}%  │  {si_pass}   │")
            overall = "✓ PASS" if smooth_metrics["all_pass"] else "✗ FAIL"
            self.logger.info(f"  └─────────────────────┴──────────┴──────────┴──────────┴──────┘")
            self.logger.info(f"  综合判定: {overall}")

            thresholds = smooth_metrics["thresholds"]
            self.logger.info(f"\n  合格标准: 速度≤{thresholds['max_velocity']}, 加速度≤{thresholds['max_acceleration']}, "
                             f"加加速度≤{thresholds['max_jerk']}, SI改善≥{thresholds['si_improvement_min']*100:.0f}%")

            if smooth_metrics.get("per_joint_max_vel"):
                joint_labels = [f"J{i+1}" for i in range(len(smooth_metrics["per_joint_max_vel"]))]
                vel_str = "  ".join(f"{l}:{v:.2f}" for l, v in zip(joint_labels, smooth_metrics["per_joint_max_vel"]))
                self.logger.info(f"\n  逐关节最大角速度(rad/s): {vel_str}")
            if smooth_metrics.get("per_joint_max_acc"):
                acc_str = "  ".join(f"{l}:{v:.2f}" for l, v in zip(joint_labels, smooth_metrics["per_joint_max_acc"]))
                self.logger.info(f"  逐关节最大角加速度(rad/s²): {acc_str}")
            if smooth_metrics.get("per_joint_max_jerk"):
                jerk_str = "  ".join(f"{l}:{v:.1f}" for l, v in zip(joint_labels, smooth_metrics["per_joint_max_jerk"]))
                self.logger.info(f"  逐关节最大加加速度(rad/s³): {jerk_str}")

            params = smooth_metrics["smoother_params"]
            self.logger.info(f"\n  平滑参数: 二阶Butterworth LPF(α={params['lp_alpha']}) + "
                             f"速度限幅({params['max_velocity']}rad/s) + "
                             f"加速度限幅({params['max_acceleration']}rad/s²) + "
                             f"加加速度限幅({params['max_jerk']}rad/s³) + "
                             f"迭代收敛({params['max_iterations']}次)")

        fk_details = eval_render["fk_details"]
        if fk_details:
            self.logger.info("\n── 7. 最后一帧详细验证 ──")
            d = fk_details[-1]
            self.logger.info(f"  帧 {d['frame']}:")
            self.logger.info(f"    FK末端: [{d['ee_pos'][0]:.4f}, {d['ee_pos'][1]:.4f}, {d['ee_pos'][2]:.4f}]")
            self.logger.info(f"    IK目标: [{d['target_pos'][0]:.4f}, {d['target_pos'][1]:.4f}, {d['target_pos'][2]:.4f}]")
            self.logger.info(f"    误差: {d['error']*100:.2f}cm  关节: [{', '.join(f'{np.degrees(j):.1f}°' for j in d['right_joints'])}]  舒适度: {d['comfort']:.4f}")

        self.logger.info("\n" + "=" * 80)


def _setup_logger(output_video: str) -> logging.Logger:
    """
    配置日志系统，同时输出到控制台和文件
    
    日志配置：
    - 控制台输出：INFO级别，显示主要执行信息
    - 文件输出：DEBUG级别，保存所有详细信息
    - 日志格式：时间 [级别] 消息
    - 日志文件：与视频文件名对应，保存在当前工作目录
    
    Args:
        output_video: 输出视频文件路径，用于生成对应的日志文件名
        
    Returns:
        配置好的Logger对象
    """
    logger = logging.getLogger("R1Tracking")
    logger.setLevel(logging.DEBUG)
    
    # 清除已有handler，避免重复输出
    logger.handlers.clear()
    
    # 创建日志格式化器
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler (INFO级别，用于实时查看)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler (DEBUG级别，保存所有详细信息，用于调试)
    # 日志文件保存在当前工作目录，与视频文件名对应
    video_name = Path(output_video).stem
    log_file = Path.cwd() / f"{video_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def main():
    """
    主函数：命令行入口
    
    完整处理流程：
    1. 解析命令行参数
    2. 配置日志系统
    3. 创建R1TrackingPipeline实例
    4. 执行pipeline.run()生成视频
    
    命令行参数：
    --dexycb-dir: DexYCB数据集根目录（必需）
    --data-id: DexYCB数据索引（默认0）
    --start-frame: 起始帧（默认0）
    --num-frames: 帧数（默认50）
    --output-video: 输出视频路径（默认r1_tracking.mp4）
    --fps: 视频帧率（默认30）
    --view: 视角模式（behind/front/topdown，默认behind）
    """
    parser = argparse.ArgumentParser(
        description="DexYCB → Dex Retargeting (手部) → RelaxedIK (R1右臂) → 视频"
    )
    parser.add_argument("--dexycb-dir", type=str, required=True, help="DexYCB 数据集根目录")
    parser.add_argument("--data-id", type=int, default=0, help="DexYCB 数据索引")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=50, help="帧数")
    parser.add_argument("--output-video", type=str, default="r1_tracking.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument(
        "--view", type=str, default="behind", choices=["behind", "front", "topdown"],
        help="视角模式: behind=背向视角, front=前向视角, topdown=高空俯视"
    )

    args = parser.parse_args()
    
    # 初始化日志系统
    logger = _setup_logger(args.output_video)
    log_file_path = Path.cwd() / f"{Path(args.output_video).stem}.log"
    logger.info(f"日志文件已创建: {log_file_path}")

    # 创建并执行pipeline
    pipeline = R1TrackingPipeline(
        dexycb_dir=args.dexycb_dir,
        data_id=args.data_id,
        output_video=args.output_video,
        fps=args.fps,
        view=args.view,
        logger=logger,
    )
    pipeline.run(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()

"""共享物理参数与工具函数

参数来源:
  - SAPIEN PD驱动: 与 GalaxeaManipSim / 04_physics_simulation.py 一致 (stiffness=1000, damping=200)
  - PyBullet PD增益: 与 pybullet_pipeline.py 一致 (Kp=1000, Kd=200)
  - 坐标变换: 与 002_render_scene.py 一致 (RXWORLD_TO_SAPIEN)
  - 夹爪几何: 与 04_physics_simulation.py / hand_track 一致
"""

import os
from pathlib import Path
import numpy as np

# ============ 路径常量 ============
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # dex-retargeting/
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_ASSETS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1"
R1_MESH_DIR = R1_ASSETS / "meshes"
R1_URDF_DIR = R1_ASSETS / "configs" / "urdfs"

# 三形式 URDF 路径
FULL_ROBOT_URDF = R1_URDF_DIR / "r1_v2_1_0.urdf"
FLOATING_ARM_RIGHT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_right.urdf"
FLOATING_ARM_LEFT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_left.urdf"

# RelaxedIK 配置
R1_RIGHT_SETTINGS = R1_ASSETS / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = R1_ASSETS / "configs" / "settings_left.yaml"

# ============ 坐标变换 (与 002_render_scene.py 一致) ============
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 机器人参数 (与 02/04 一致) ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985]
ARM_MAX_REACH = 0.713

# R1 右臂关节限位 (rad) — 用于 IK 输出裁剪, 防止发散
# joint1: [-2.8798, 2.8798], joint2: [0, 3.2289], joint3: [-3.3161, 0],
# joint4: [-2.8798, 2.8798], joint5: [-1.6581, 1.6581], joint6: [-2.8798, 2.8798]
RIGHT_ARM_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],
    [0.0, 3.2289],
    [-3.3161, 0.0],
    [-2.8798, 2.8798],
    [-1.6581, 1.6581],
    [-2.8798, 2.8798],
], dtype=np.float64)
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
BASE_TRACKING_ALPHA = 0.15
SAFETY_DISTANCE = 0.05
WARMUP_FRAMES = 30

# ============ 物理参数 (与 GalaxeaManipSim / 04 一致) ============
# PD 驱动: 高刚度(100000)配合set_qpos会震荡, 纯PD驱动用合理刚度
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 1000.0
GRIPPER_DAMPING = 200.0
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
OBJECT_DENSITY = 1000.0
GROUND_HEIGHT = 0  # 对齐 GalaxeaManipSim: scene.add_ground(0)
GRAVITY = [0, 0, -9.81]

# PyBullet PD 增益 (匹配 SAPIEN)
PD_KP_ARM = 1000.0
PD_KD_ARM = 200.0
PD_KP_GRIPPER = 1000.0
PD_KD_GRIPPER = 200.0

# ============ 夹爪几何常数 (与 04/hand_track 一致) ============
_FINGER1_ORIGIN = np.array([0.03689, -0.020, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.020, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
GRIPPER_INIT_OPEN = 0.04
GRIPPER_MAX_OPEN = 0.05
# v6.6 (2026-08-31): 2.5 → 2.0 统一. 此前本文件 2.5 与 physics_env 本地 2.0 并存,
# 同一夹爪两套摩擦 (physics_env 建材质用 2.0, grasp_controller 从本文件取 2.5).
# 取 2.0 = v4.14f 实验最佳 (3.0 物体被卡住推不开 f1; 2.5 与 2.0 无明显差别).
# 2026-09-05: 环境旋钮 _GRIP_FRICTION (默认 2.0 不变) — 携带段滑脱对比实验用
# (历史: v4.14f 试过 3.0, 当时边降边闭方案下"物体被卡住推不开"; 新方案接近段手指不碰物体, 重试)
import os as _os_friction
GRIPPER_FRICTION = float(_os_friction.environ.get('_GRIP_FRICTION', '2.0'))   # 单一来源; physics_env 已改为从此处导入
GRIPPER_CLOSE_BIAS = 0.005  # 闭合偏移 5mm, 确保手指紧贴物体
# 注: 本文件曾定义 GRIPPER_FORCE = 10.0, 与 v6.0 的 FINGER_PD_FORCE_LIMIT = 100.0
# 差 10 倍且从未被任何模块使用 (drive 实际走 physics_env.GRIPPER_FORCE = 100N).
# 2026-08-31 删除该死值, 避免误导后续调参. 力限唯一来源 = FINGER_PD_FORCE_LIMIT.

# ============ v6.0 统一物理仿真参数 (全 stage 共用, 单一来源) ============
# 原则: 运行时只允许 set_drive_target + physics_step, 禁止 set_qpos / virtual_lock_targets
# (v4.52 的"drive 失效"实为 vlock 与 drive 冲突: vlock 每子步 set_qpos+qvel=0 钉死 qpos;
#  "36kg 下垂"实为误诊, URDF 整机仅 ~0.7kg, K=5000 下垂才 1.4mm)
VIRTUAL_PD_STIFFNESS = float(os.environ.get("_VIRTUAL_PD_K", 50000.0))
# 基座虚拟关节 PD 刚度. 默认 50000 是历史 "Stage1 纯 PD 成功配方".
# 2026-09-11: 改为可用环境变量 _VIRTUAL_PD_K / _VIRTUAL_PD_D 覆盖, 以便与
# GalaxeaManipSim 参考实现 (r1.py: joint_stiffness=1000, joint_damping=200,
# timestep=0.01, control_freq=15) 做 A/B 对比. 每次运行的实际取值会写入
# <run_root>/run_metadata.json, 便于跨运行追溯.
# v6.0b: 2500 → 600. 2500 是 ζ≈6.7 的重过阻尼 (2√(K·M)≈374, M≈0.7kg), 快速段
# 跟踪滞后实测达 9cm (warmup F50 vz_err), 纯 PD 下导致手指按帧号提前闭合、
# 骑在物体顶上 (Stage3 lift=0 的直接原因). 600 → ζ≈1.6, τ≈D/K=12ms, 无超调.
VIRTUAL_PD_DAMPING = float(os.environ.get("_VIRTUAL_PD_D", 600.0))
FINGER_PD_STIFFNESS = 1000.0     # = GRIPPER_STIFFNESS
FINGER_PD_DAMPING = 200.0        # = GRIPPER_DAMPING
FINGER_PD_FORCE_LIMIT = 100.0    # = GRASP_STRATEGIES['pd_then_lock'] (对齐 URDF effort)
CARRY_DEV_TOL_M = 0.005          # 统一物理验证: 夹持段手指中点-物体距离容差 5mm

# ============ 夹爪朝向约定 (v6.9, 2026-09-01) ============
# SAPIEN/GLB 世界系: Z 向上, 重力沿 -Z.
# 夹爪本体坐标系 (见 _FINGER1_ORIGIN / FINGER_FORWARD_NEUTRAL):
#   X 轴 = approach / 手指伸出方向 (手指在 base 前方 3.689cm)
#   Y 轴 = 开合方向 (_FINGER1_AXIS = [0,-1,0])
#   Z 轴 = X × Y
# 抓取桌面物体时 approach 必须朝下, 即 R 的 X 轴世界 z 分量 < 0.
GRAVITY_DIR = np.array([0.0, 0.0, -1.0])  # 单位重力方向 (SAPIEN/GLB)


def flip_mano_hand_down(joints, wrist_idx=0, gravity_dir=None):
    """v6.9b (2026-09-02): 若 MANO 手指向朝上, 绕**水平轴**翻转 180° 使其朝下。

    与 `align_gripper_approach_to_gravity` (下游, 修 R) 不同, 本函数在**上游**修 MANO
    关节本身 — 手姿态对了, 解算出的 R 自然朝下。

    实测 (test/diag_mano_R_orientation.py, 2026-09-02): 当前 MANO 手在该变换链下是
    **掌心朝天、指尖朝上**的 (掌面法向 z=+0.83, 指尖比手腕高 13.45cm), 而手位于
    物体上方 40cm — 姿态与"伸手抓桌面物体"矛盾, 属上游姿态翻转错误。
    翻转后: 掌法 z +0.846→-0.846, 指尖-腕 +14.31→-14.31cm, 指尖高度 0.574→0.288
    (逼近物体 0.039), 解算 R 的 dot(X,重力) 自然为 +0.326 → 无需下游补丁。

    翻转轴 h = normalize(cross(z_up, pointing)):
      - h 水平 (⊥ 重力) 且 ⊥ pointing 的水平投影
      - 绕 h 转 180°: v' = 2(v·h)h - v → **垂直分量取反**, 沿 h 分量保持
      - 效果: 手指朝下、掌心朝地, 而手的**水平朝向不变**
      - 这是**旋转** (det=+1), 非关于水平面的镜像, 不改变手性

    判据沿**重力方向**取点积 (用户要求): 仅当 `dot(pointing, g) < 0` (朝上) 时翻转,
    因此**幂等** — 手已朝下时原样返回, 可安全重复调用。

    Args:
        joints: (21, 3+) MANO 关节 (世界系); 非关节数组原样返回
        wrist_idx: 腕关节索引 (MANO j0)
        gravity_dir: (3,) 重力方向, 默认 -Z

    Returns:
        与 joints 同形状的数组 (必要时已翻转)
    """
    j = np.asarray(joints, dtype=np.float64)
    # 形状保护: 需要至少 j[0]/j[4]/j[8] (腕/拇指尖/食指尖)
    if j.ndim != 2 or j.shape[0] < 9 or j.shape[1] < 3:
        return joints

    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    g = g / max(np.linalg.norm(g), 1e-8)

    w = j[wrist_idx, :3]
    pointing = ((j[4, :3] + j[8, :3]) / 2.0) - w
    if float(np.dot(pointing, g)) >= 0.0:
        return joints  # 已朝下 (或水平), 幂等: 不翻转

    # 翻转轴: 水平, 且垂直于 pointing 的水平投影
    up = -g  # 世界上方向 (= -重力)
    h = np.cross(up, pointing)
    nl = float(np.linalg.norm(h))
    if nl < 1e-8:
        return joints  # pointing 已垂直, 无法定义水平翻转轴
    h = h / nl

    out = np.array(joints, dtype=np.float64, copy=True)
    rel = j[:, :3] - w
    proj = (rel @ h)[:, None] * h[None, :]
    out[:, :3] = w + (2.0 * proj - rel)
    return out


def align_gripper_approach_to_gravity(R, gravity_dir=None, up_tilt_deg=0.0):
    """确保夹爪 approach 轴 (gripper X = 手指伸出方向) 沿重力朝下。

    若 X 轴与重力反向 (即朝上), 绕**开合轴 (gripper Y)** 旋转 180°:
    X→-X, Z→-Z, Y 不变。这样开合平面保持不动, 手指从朝天翻为朝地。

    为什么绕 Y 而不是别的轴:
      绕 Y (开合轴) 翻转 = 夹爪"倒过来拿", 开合方向不变, 物理上唯一合理的翻转。
      绕 X 翻转只反向开合 (Y→-Y), 手指仍朝天, 解决不了朝上问题。

    判据沿**重力方向** (世界 -Z), 不依赖手掌平面法向/解剖叉乘 —
    后者只反映手掌自身朝向, 与"是否朝下"无必然联系。

    ⚠ 2026-09-11 回退说明 (用户定策):
      本函数位于 **Stage0 的 MANO→夹爪映射链路** (被 data_loader.
      compute_mano_based_gripper_pose / compute_analytical_gripper_pose 每帧调用),
      因此它的输出就是 Stage0 复现出来的夹爪姿态本身, Stage1/2/3 都应以它为准。
      本次会话一度把它改成"向上倾角 >10° 才翻"的死区判据 (up_tilt_deg=10), 那会
      **改变 Stage0 的映射结果**, 已按用户要求回退为原始判据 (up_tilt_deg=0.0,
      即 dot(X, g) < 0 就翻)。Stage0 已证明该映射正确, 不再改动。
      up_tilt_deg 形参仅作保留, 默认 0.0 = 原始行为.

    Args:
        R: (3,3) 夹爪旋转矩阵, 列为 [X, Y, Z] 三轴 (世界系表示)
        gravity_dir: (3,) 重力方向, 默认 -Z
        up_tilt_deg: 死区角 (度). 默认 0.0 = 原始行为 (朝上即翻).

    Returns:
        (3,3) 朝向修正后的旋转矩阵 (若已朝下则原样返回)
    """
    R = np.asarray(R, dtype=np.float64)
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    # dot(X, g) < 0 表示 X 与重力反向 = 手指朝上 → 需翻转
    if float(np.dot(R[:, 0], g)) < -float(np.sin(np.deg2rad(float(up_tilt_deg)))):
        R = R @ np.diag([-1.0, 1.0, -1.0])  # 绕 Y 轴 180°
    return R


def unflip_rotation_traj_continuity(R_traj, gravity_dir=None, direction_eps=0.15):
    """消除 align_gripper_approach_to_gravity 逐帧硬阈值造成的 180° 姿态台阶.

    背景 (2026-09-11 实测定位):
      `align_gripper_approach_to_gravity` 用 `dot(X, g) < 0` 这一**过零硬阈值**逐帧
      独立决策 (被 compute_mano_based_gripper_pose / compute_analytical_gripper_pose
      每帧调用一次). 当 MANO 的 approach 轴接近**水平**时, 该符号在相邻帧间来回跳,
      姿态序列因此出现真实的 180° 台阶. 7a/左手/scale0.6 实测:
        F7->F8 179.5° | F9->F10 179.1° | F13->F14 178.8° | F108->F109 179.3°
      而全序列平均帧间变化只有 7.63°; 该段 X.z = -0.085 … -0.003 (几乎水平).
      验证: 这几对 R 满足 R[i+1] ≈ R[i] @ diag(-1,1,-1) (残差 0.97/1.87/2.54°).

    危害: 1 个控制周期 (33ms) 内命令姿态翻转 180°, PD 物理上跟不住 →
      基座姿态滞后 11~48°、角速度 589°/s → 手指(伸出 7.74cm)方向扫偏厘米级 →
      接近段撞上目标物体 → 手指被顶到闭合极限 (实测 act=0.0000) → 全程接触.

    做法 (**只在模糊带内做连续性修正**):
      - `|dot(X, g)| > direction_eps` → approach 轴方向明确, **保持重力对齐结果不变**
        (否则会把整条轨迹换到另一个物理分支: 实测无限制版本翻掉 97/113 帧, 令
         R[F50] 被翻 → pre_translate 偏移错 → 整条轨迹偏离物体 12cm, 并在接近段
         产生 345° 的 slerp 大弧);
      - `|dot(X, g)| <= direction_eps` → approach 轴≈水平, 两个分支物理上等价,
        取与上一帧更近者, 从而消除台阶.

    Args:
        R_traj: (N, 3, 3) 姿态序列 (已做过重力对齐, 即每帧 X·g <= 0)
        gravity_dir: (3,) 重力方向, 默认 GRAVITY_DIR
        direction_eps: 模糊带半宽 (|X·g| 小于它才允许翻回)
    Returns:
        (R_fixed (N,3,3), n_fixed) — n_fixed 为被翻回的帧数
    """
    F = np.diag([-1.0, 1.0, -1.0])
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    g = g / max(float(np.linalg.norm(g)), 1e-8)
    R_arr = np.asarray(R_traj, dtype=np.float64)
    out = np.empty_like(R_arr)
    prev = None
    n_fixed = 0
    for i, R in enumerate(R_arr):
        x_dot_g = float(np.dot(R[:, 0], g))
        if prev is None:
            pick = R
        elif abs(x_dot_g) > direction_eps:
            pick = R  # 方向明确: 不改, 保住"approach 轴朝下"约定与轨迹基准
        else:
            _diff = R @ prev.T
            d_keep = float(np.arccos(np.clip((np.clip(np.trace(_diff), -1.0, 3.0) - 1.0) / 2.0, -1.0, 1.0)))
            _alt = R @ F
            _diff2 = _alt @ prev.T
            d_flip = float(np.arccos(np.clip((np.clip(np.trace(_diff2), -1.0, 3.0) - 1.0) / 2.0, -1.0, 1.0)))
            if d_flip < d_keep - 1e-9:
                pick = _alt
                n_fixed += 1
            else:
                pick = R
        out[i] = pick
        prev = pick
    return out, n_fixed


def compute_tracking_error(actual_pos_traj, target_pos_traj, start_frame=0, end_frame=None):
    """统一物理验证 ②: 跟随误差 — 仿真实际基座位置 vs 本 stage 的目标轨迹.

    与 validate_carry (夹爪-物体恒定距离) 并列为两个核心偏差指标:
      - 跟随误差 = 重定向/控制精度 (实际 vs 目标)
      - 恒定距离 = 是否真夹住并刚性跟随 (夹爪 vs 物体)

    重要: actual 与 target **必须来自同一 stage 的同一基准**.
    历史坑 (2026-08-31): View/vis_stage0_error_analysis.py 曾用
    Stage0 实际 qpos 比 Stage3 npz 的 mano_pos — 两者 MANO 平移状态不同
    (Stage0 neutral_offset 0.517m vs Stage3 pre_translate 0.6661m),
    得到恒定的 19.36cm "误差" 实为基准差, 不是跟踪误差.

    Args:
        actual_pos_traj: (N,3) 仿真实际基座位置 (每帧从 qpos + virtual_idx 提取)
        target_pos_traj: (N,3) 本 stage 的目标轨迹 (即送进 PD 的 target_pos)
        start_frame: 起始统计帧 (跳过初始就位段)
        end_frame: 结束帧 (None=末尾)

    Returns:
        dict: mean_mm, max_mm, final_mm, n_frames
    """
    a = np.asarray(actual_pos_traj, dtype=np.float64)
    t = np.asarray(target_pos_traj, dtype=np.float64)
    n = min(len(a), len(t))
    if end_frame is None:
        end_frame = n
    lo = max(0, min(start_frame, n - 1))
    hi = min(end_frame, n)
    if hi - lo < 1:
        return dict(mean_mm=float('nan'), max_mm=float('nan'),
                    final_mm=float('nan'), n_frames=0)
    d = np.linalg.norm(a[lo:hi] - t[lo:hi], axis=1)
    valid = np.isfinite(d)
    if not valid.any():
        return dict(mean_mm=float('nan'), max_mm=float('nan'),
                    final_mm=float('nan'), n_frames=0)
    d = d[valid]
    return dict(mean_mm=float(d.mean() * 1000),
                max_mm=float(d.max() * 1000),
                final_mm=float(d[-1] * 1000),
                n_frames=int(len(d)))


def validate_carry(fmid_traj, obj_traj, grasp_frame, release_frame=None,
                   tol_m=CARRY_DEV_TOL_M, one_sided=False):
    """统一物理验证: 夹持段 (抓取确认→释放) 手指中点与物体保持恒定距离 (刚性跟随).

    Args:
        fmid_traj: (N,3) 手指中点轨迹 (NaN 允许, 自动跳过)
        obj_traj: (N,3) 物体轨迹
        grasp_frame: 夹持起始帧 (该帧距离为基准 / F50)
        release_frame: 释放帧 (默认到末尾)
        tol_m: 恒定距离容差
        one_sided: True → 只罚 d>ref (物体相对手指分离=掉落风险), 不罚 d<ref;
                   F50 是 ref 的"最低/基准值", 不允许优化到低于它 (用户 2026-09-09).

    Returns:
        dict: ref_dist_mm, max_dev_mm, mean_dev_mm, final_dev_mm, passed
    """
    fmid_traj = np.asarray(fmid_traj, dtype=np.float64)
    obj_traj = np.asarray(obj_traj, dtype=np.float64)
    n = min(len(fmid_traj), len(obj_traj))
    end = n if release_frame is None else min(release_frame, n)
    lo = min(grasp_frame, n - 1)
    d = np.linalg.norm(fmid_traj[:end] - obj_traj[:end], axis=1)
    valid = ~np.isnan(d)
    valid[:lo] = False   # F38-F50 (F50) 之前不参与 (用户 2026-09-09: 接近段不计)
    if valid.sum() < 2:
        return dict(ref_dist_mm=float('nan'), max_dev_mm=float('nan'),
                    mean_dev_mm=float('nan'), final_dev_mm=float('nan'),
                    passed=False, n_valid=int(valid.sum()))
    _idx = np.where(valid)[0]
    ref = float(d[_idx[0]])          # 基准 = F50 (夹持起始帧) 距离 — 固定最低值
    _d = d[valid]
    if one_sided:
        # 只罚物体相对手指分离 (d>ref = 掉落), 不罚 d<ref (压紧/未分离)
        dev = np.maximum(0.0, _d - ref)
    else:
        dev = np.abs(_d - ref)
    return dict(ref_dist_mm=float(ref * 1000),
                max_dev_mm=float(dev.max() * 1000),
                mean_dev_mm=float(dev.mean() * 1000),
                final_dev_mm=float(dev[-1] * 1000),
                passed=bool((float(dev.max()) if dev.size else tol_m + 1) <= tol_m),
                n_valid=int(valid.sum()))

# ============ 平滑参数 ============
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
EMA_POS_ALPHA = 0.6
EMA_ORI_ALPHA = 0.6
SMOOTH_MAX_VELOCITY = 1.5
SMOOTH_MAX_ACCELERATION = 4.0
SMOOTH_MAX_JERK = 20.0
SMOOTH_LP_ALPHA = 0.25

# ============ IK 参数 ============
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20

# ============ 抓取检测 ============
GRASP_CONTACT_THRESHOLD = 2  # 至少2个接触点
GRASP_STABLE_FRAMES = 5      # 持续接触N帧判定稳定抓取
GRASP_OBJECT_VEL_THRESHOLD = 0.01  # 物体相对夹爪速度阈值

# ============ GLB 物体分类 ============
# 所有 GLB 物体设为 kinematic (静态), 避免物理不稳定导致物体飞走.
# 注: 之前小物体设为 dynamic, 但凸包碰撞不稳定, 物体飞到 x=-6.66, y=14.81.
# 设为 kinematic 后物体固定, 机器人仍可靠近做抓取姿态.
KINEMATIC_VOLUME_THRESHOLD = -1  # -1 表示所有物体都视为大型场景结构 (kinematic)
DYNAMIC_MASS_MAX = 5.0  # kg


def is_large_scene_object(bbox_min, bbox_max):
    """判断GLB物体是否为大型场景结构(桌面/地板), 应设为kinematic

    Args:
        bbox_min, bbox_max: 包围盒角点 (3,)

    Returns:
        bool: True表示大型场景结构
    """
    size = np.array(bbox_max) - np.array(bbox_min)
    volume = float(np.prod(size))
    return volume > KINEMATIC_VOLUME_THRESHOLD


def compute_object_mass(bbox_min, bbox_max, density=OBJECT_DENSITY):
    """根据包围盒体积和质量密度计算物体质量

    Args:
        bbox_min, bbox_max: 包围盒角点 (3,)
        density: 密度 kg/m^3

    Returns:
        float: 质量 kg
    """
    size = np.array(bbox_max) - np.array(bbox_min)
    volume = float(np.prod(size))
    return min(volume * density, DYNAMIC_MASS_MAX)

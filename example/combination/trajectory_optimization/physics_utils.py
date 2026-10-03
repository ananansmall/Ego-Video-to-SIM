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
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-23 【方案 §3.1 实施】P1 基线: physics=**480Hz** / control=30Hz / decim=**16** ✓ ★★★★
#   方案原话 ✓: \"推荐: physics=480 Hz、control=30 Hz、**decimation=16**; CPU TGS;
#     位置迭代 16、速度迭代 1 … **所有阶段同步改, 不能只细化 Stage1**\" ✓
#   旧值 ✗: `1/240` + decim=**8** ⇒ 近接触分辨率只有方案要求的一半 ✗
#   ⚠ 这**改变物理积分** ⇒ 旧物理指纹/旧 anchor/旧 PASS **一律失效** ✓
#     (方案 §2.2: \"任何 P1 模型变更都使旧物理指纹失效; 旧 anchor 可作搜索初值,
#      **旧 PASS 不继承**\" ✓)
#   可用 `_PHYS_DT` / `_CONTROL_HZ` 覆盖 ✓ (默认已按方案 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
PHYSICS_TIMESTEP = float(os.environ.get("_PHYS_DT", str(1.0 / 480.0)))
CONTROL_FREQ = int(float(os.environ.get("_CONTROL_HZ", "30") or 30))
DECIMATION = max(1, int(round((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP)))  # P1 = 16 ✓
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
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 实算 0.040 (注释旧值 0.026906 已过时)

# ★★★ 2026-09-13 手指几何的**唯一真源** (用户: "你这些模型都不修改不统一就开始跑吗?")
#   真实指形 STL (未缩放) 做 collision 后实测:
#     finger1 世界系内侧面 (q=0) = -0.020 + 0.0048 = -0.0152
#   ⇒ 虎口内侧 |y| = FINGER_INNER_Y0 + q
#   ⇒ 指缝宽度   = FINGER_GAP0  + 2q
#   历史坑: 薄板时代是 0.005/0.010, 换成真实指形后 Stage1/Stage2 的若干处
#   仍写死 0.005 → 模型不一致 → q_grasp 解算偏差 → 抓取失败.
FINGER_INNER_Y0 = 0.0152   # 虎口内侧基准 (m, q=0) —— 真实指形 STL
FINGER_GAP0 = 0.0304       # 指缝基准 (m, q=0)


def finger_gap(q) -> float:
    """指缝宽度 (m) = 0.0304 + 2q  —— 请统一用它, 不要再写死常数."""
    return float(FINGER_GAP0) + 2.0 * float(q)


def finger_inner_y(q) -> float:
    """虎口内侧 |y| (m) = 0.0152 + q."""
    return float(FINGER_INNER_Y0) + float(q)
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
# ★★★★ 2026-09-20 【用户定策】手指 PD 的 k_p / k_d 改为**可调** ★★★★
#   用户原话: "k_p 对应 stiffness 参数, k_d 对应 damping 参数 —— 这个 kd 可以调大点试试,
#     应该夹取会更有力一点" ✓
#   历史默认: k_p=1000 / k_d=200 (v6.0 统一为 Stage1 纯 PD 成功配方 ✓, 保持不变 ✓)
#   新增覆盖: `_FINGER_PD_K` / `_FINGER_PD_D` (未设 ⇒ 完全等于历史默认 ✓ 行为不变 ✓)
#     每次实际取值都会写进 run_metadata.json (dump_run_metadata 已记录 FINGER_PD_DAMPING ✓)
#   物理说明 ✓: 稳态夹持力 ≈ k_p × 位置误差 (与 k_d 无关), 但 **k_d 决定到达/抗扰动的收敛行为**:
#     · k_d 偏小 ⇒ 接触瞬间容易振荡/穿透回弹 ⇒ 平均正压力不稳 (实测 peak_force 0.06~6N 跳变 ✗)
#     · k_d 偏大 ⇒ 接触后更"黏", 抑制回弹 ⇒ 有效夹持更稳 ✓ (这就是"更有力"的机制 ✓)
#     代价: k_d 过大会让手指跟随变慢 (接近段跟踪误差上升 ✗) ⇒ 需要扫一个区间 ✓
FINGER_PD_STIFFNESS = float(os.environ.get("_FINGER_PD_K", 1000.0))   # = GRIPPER_STIFFNESS
FINGER_PD_DAMPING = float(os.environ.get("_FINGER_PD_D", 200.0))      # = GRIPPER_DAMPING
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-23 【用户指出】力上限 = **100N** 对 99g 物体是 **103× 重量** ✗ ⇒ 弹飞 ★★★★
#   用户原话 ✓: "**这怎么能弹飞, 你的质量是不是给的太低了, 还是力太大了** …
#     **肯定是这两个是有问题的**" ✓
#   实测数字 ✓: 目标物 toy_0 (glb_4) 质量 **0.099kg** (mg≈0.97N) ✓;
#     指力上限 **100N** ✗ = 103× 重量; PD 刚度 1000N/m × 过闭合 6mm = 6N = 6× 重量 ✗
#     ⇒ 实体证据: `obj_z` 从 16.8mm 冲到 **9.36m** ✗ (被弹飞 ⇒ "物体突然不见" ✓)
#   历史对照 ✓: giffix / acc23b / aeval1 三个 run 的 force=100N, density=1000 ⇒
#     **与现在逐项相同** ⇒ 这套力/质量**一直不合理** ✗ (不是最近改坏的 ✓)
#   本改 ✓: 力上限改为**可配** ⇒ `_FINGER_PD_FORCE_LIMIT` (N) ✓, 默认 100 ⇒ **不改变既有行为** ✓
#     建议实验值 ✓: **5~10 N** (5~10× 物体重量 ⇒ 够夹住且不会弹飞 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
FINGER_PD_FORCE_LIMIT = float(os.environ.get("_FINGER_PD_FORCE_LIMIT", 100.0))
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


def horizontalize_closing_axis(R, gravity_dir=None, verbose=True):
    """把开合轴 (gripper Y) 旋进水平面 —— X 固定不动, 绕 X 转 θ 使 Y'·g = 0。

    θ = atan2(−(Y·g), (Z·g));  Y' = Y cosθ + Z sinθ。

    2026-09-20 从 `align_gripper_approach_to_gravity` 里**抽出来**(行为不变, 该函数改为调用本函数),
    原因是它必须能被**单独调用**:
      - 分支 DP 必须在"只定分支、未水平化"的序列上做 (见 `track_rotation_branch` 说明);
      - 因此链路要拆成 `重力对齐(只翻分支) → 分支 DP → 水平化`, 水平化这一步得能事后施加。

    ⚠ 为什么必须放在分支 DP **之后** (2026-09-20 实测, 这是"上下翻转修不掉"的真正原因):
      本步是绕 **X** 转 θ, 而 θ 实测达 **−60° ~ −123°** (远不是小角度)。
      分支翻转 F = diag(-1,1,-1) 是绕 **Y** 转 180°。二者**不可交换**:
        R@F@Rc(θ)  ≠  R@Rc(θ)@F
      于是"水平化在前"会把 F10→F11 的台阶从 **绕 Y 180°** 变成 **绕 Z 180°**
      (实测残差 1.7°), 而"绕 Y 翻转"的分支补丁对绕 Z 的台阶**完全无效** ——
      实测: 未水平化时翻分支 ⇒ 台阶 178.8° → **2.3°** ✅;
            水平化后翻分支 ⇒ 台阶 178.3° → **180.0°** ❌ (反而更糟)。

    Args:
        R: (3,3) 夹爪旋转矩阵, 列为 [X, Y, Z] (世界系)
        gravity_dir: (3,) 重力方向, 默认 GRAVITY_DIR = [0,0,-1]
        verbose: 是否打印旋转角度日志
    Returns:
        (3,3) 开合轴已水平的旋转矩阵 (|Y·g| ≈ 0)
    """
    R = np.asarray(R, dtype=np.float64)
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    _y, _z = R[:, 1], R[:, 2]
    _yg = float(np.dot(_y, g))
    _zg = float(np.dot(_z, g))
    if abs(_yg) <= 1e-6:                    # 已经水平就不用转
        return R
    _th = float(np.arctan2(-_yg, _zg))
    _c, _s = float(np.cos(_th)), float(np.sin(_th))
    _Rc = np.array([[1.0, 0.0, 0.0],
                    [0.0, _c, -_s],
                    [0.0, _s, _c]], dtype=np.float64)   # 绕 X(第一列) 的旋转
    R = R @ _Rc
    if verbose:
        import logging as _logging
        _log = _logging.getLogger(__name__)
        if _log.isEnabledFor(_logging.INFO):
            _log.info(
                "[闭合轴水平化] 旋转 %.1f° → 竖直度 %.3f → %.3f (Y=%s)",
                np.degrees(_th), abs(_yg), abs(float(np.dot(R[:, 1], g))),
                np.round(R[:, 1], 3).tolist())
    return R


def align_gripper_approach_to_gravity(R, gravity_dir=None, up_tilt_deg=0.0,
                                      horizontalize=None):
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
        horizontalize: 是否接着做"闭合轴水平化"(见 `horizontalize_closing_axis`).
            None (默认) = 读环境变量 `_GRIP_CLOSE_AXIS_HORIZONTAL` —— **默认行为逐位不变**;
            False = 只定分支不水平化 (供"先做分支 DP、再水平化"的分层链路使用);
            True  = 强制水平化.

    Returns:
        (3,3) 朝向修正后的旋转矩阵 (若已朝下则原样返回)
    """
    R = np.asarray(R, dtype=np.float64)
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    # dot(X, g) < 0 表示 X 与重力反向 = 手指朝上 → 需翻转
    if float(np.dot(R[:, 0], g)) < -float(np.sin(np.deg2rad(float(up_tilt_deg)))):
        R = R @ np.diag([-1.0, 1.0, -1.0])  # 绕 Y 轴 180°

    # ★★★★ 2026-09-14 新增 (可选): 闭合轴水平化 ★★★★
    #   用户报告"反向"的实测证据:
    #     闭合轴 = [0.043, -0.686, -0.726] → **竖直度 0.73** ✗
    #     ⇒ 两个手指一个在物体上方、一个在下方 (**上下夹**), 视觉上就是"反的" ✗
    #   根因: 上面的翻转只保证 **X(手指伸出方向) 朝下**, 不约束 **Y(开合轴)**.
    #         X 朝下 + Y 竖直 是自洽解 → "上下夹"被放行.
    #   做法见 `horizontalize_closing_axis` (2026-09-20 已抽为独立函数).
    #   为什么默认"读环境变量": 本函数位于 **Stage0 的 MANO→夹爪映射链路**, 其输出就是
    #         Stage0 复现出来的夹爪姿态本身; 用户 2026-09-11 定策"Stage0 已证明
    #         该映射正确, 不再改动"。因此保持 opt-in (`_GRIP_CLOSE_AXIS_HORIZONTAL=1`),
    #         `horizontalize=None` 时行为与改动前**逐位一致**.
    import os as _os
    _hz = (_os.environ.get("_GRIP_CLOSE_AXIS_HORIZONTAL", "0") == "1") if horizontalize is None \
        else bool(horizontalize)
    if _hz:
        R = horizontalize_closing_axis(R, g)
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


def rotation_step_deg(Ra, Rb):
    """两个旋转矩阵之间的测地角 (deg)。"""
    tr = float(np.clip(np.trace(np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)), -1.0, 3.0))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def rotation_traj_diag(R_traj, gravity_dir=None):
    """姿态序列体检 (只读) — 供日志/回归门使用。

    返回 dict:
      n, det_min, det_max, orth_err, xz(每帧 X 轴世界 z 分量), yz(每帧 Y 轴 z 分量),
      step_deg(N, 首帧 0), max_step_deg, n_step_gt_90, n_step_gt_30,
      down_ratio(|X·g|>tilt_eps 的帧中 X 朝下占比), n_ambiguous(|X·g|<=tilt_eps)
    """
    R = np.asarray(R_traj, dtype=np.float64)
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    g = g / max(float(np.linalg.norm(g)), 1e-8)
    N = len(R)
    det = np.linalg.det(R)
    orth = float(np.abs(np.einsum('nji,njk->nik', R, R) - np.eye(3)).max())
    xz = R[:, 2, 0]
    yz = R[:, 2, 1]
    dotxg = R[:, :, 0] @ g
    step = np.zeros(N)
    for i in range(1, N):
        step[i] = rotation_step_deg(R[i - 1], R[i])
    return dict(
        n=int(N), det_min=float(det.min()), det_max=float(det.max()), orth_err=orth,
        xz=xz, yz=yz, step_deg=step,
        max_step_deg=float(step.max()), n_step_gt_90=int((step > 90).sum()),
        n_step_gt_30=int((step > 30).sum()),
        down_ratio=float(np.mean(dotxg >= 0)), n_down=int((dotxg >= 0).sum()),
        n_up=int((dotxg < 0).sum()),
        max_up=float(max(0.0, -dotxg.min())), max_down=float(max(0.0, dotxg.max())),
    )


def track_rotation_branch(R_traj, gravity_dir=None, anchor_idx=None, tilt_eps=0.17,
                          hard_tilt=0.5, switch_penalty_deg=30.0, up_penalty=30.0,
                          return_info=False):
    """沿**整段**轨迹为姿态序列选一致的符号分支 —— 取代逐帧硬阈值。

    问题 (2026-09-20 实测):
      `align_gripper_approach_to_gravity` 是**逐帧、无状态、硬阈值** (`dot(X,g) < 0` 就绕
      开合轴 Y 翻 180°). 当 approach 轴接近**水平**时, 该符号在相邻帧间跳变, 姿态序列出现
      真实的 180° 台阶. 实测 (output/d2_table_coacd, 113 帧):
        raw SVD 的 X 轴 F10→F11 只变 1.2°, 但 `X.z` 从 +0.0069 过零到 -0.0039
        ⇒ 硬阈值把 1° 的噪声放大成 **179.38° 的台阶** (F11 / F109 各一次).
      危害: 1 个控制周期 (33ms) 内命令 180° 翻转, PD 跟不住 ⇒ Stage2 `MANO_FOLLOW`
        段实测姿态误差峰值 169°.

    做法 (全局最优, 不是逐帧比大小):
      每帧两个**等价**候选分支 (绕开合轴 Y 转 180°: X→-X, Z→-Z, **Y 不变**):
          state 0: R           (原始, X 可能朝上)
          state 1: R @ diag(-1,1,-1)
      Viterbi/DP 最小化整段代价:
        ① 帧间测地角 (相邻帧换分支 ⇒ 天然 +~176°)
        ② 每次换分支另加 `switch_penalty_deg`
        ③ **朝上罚** `up_penalty * relu(dot(X, g))` — 只对 X 与重力反向 (朝上) 的帧计
        ④ 硬约束 `hard_tilt`: |dot(X_raw,g)| > hard_tilt 的帧**方向明确**, 必须选 X 朝下那支
        ⑤ 锚 `anchor_idx`: 该帧锁死为 X 朝下 (抓取帧姿态唯一明确, 防全局 180° 歧义)

    为什么这样能修好:
      近水平的帧 (|X·g|≈0.004) 翻与不翻代价几乎相同 (朝上罚≈0.1), 而切换一次要 ~180°
      ⇒ DP 自然选择"保持与邻帧连续", 不再翻. 而方向明确的帧 (|X·g|>hard_tilt) 被硬约束
      钉死朝下 ⇒ 语义不丢. 结果: 切换次数 2 → 0, 帧间台阶 179° → 个位数度.

    与 `unflip_rotation_traj_continuity` 的区别 (后者是死代码, 且是逐帧贪心):
      后者只在 |X·g| ≤ direction_eps 的模糊带内**逐个**翻回, 带边缘仍是硬切换 ——
      实测只会把台阶从 F11 **搬到 F20**. 本函数是整段 DP, 无边缘问题.

    Args:
        R_traj: (N,3,3) 姿态序列 (已做过重力对齐, 允许 X 朝上的帧存在)
        gravity_dir: (3,) 重力方向, 默认 GRAVITY_DIR = [0,0,-1]
        anchor_idx: 锁定的锚帧索引 (None = 自动取 |dot(X_raw,g)| 最大的帧)
        tilt_eps: 模糊带半宽 (仅用于统计口径; DP 本身由代价函数决定)
        hard_tilt: 硬约束阈值 —— |dot(X_raw,g)| 超过它就必须 X 朝下
        switch_penalty_deg: 每次换分支的额外代价 (deg)
        up_penalty: 朝上罚系数 (deg / 单位 dot)
        return_info: True 时额外返回 info dict

    Returns:
        R_out (N,3,3) 已统一分支的姿态序列
        (或 (R_out, info) 当 return_info=True)
        info: dict(branch(N,), n_switch, flips(N, bool), max_step_deg, ...)
    """
    R = np.asarray(R_traj, dtype=np.float64).copy()
    N = len(R)
    g = GRAVITY_DIR if gravity_dir is None else np.asarray(gravity_dir, dtype=np.float64)
    g = g / max(float(np.linalg.norm(g)), 1e-8)
    F = np.diag([-1.0, 1.0, -1.0])  # 绕开合轴 Y 转 180°

    if N == 0:
        return (R, dict(branch=np.zeros(0, dtype=int), n_switch=0,
                        flips=np.zeros(0, dtype=bool), max_step_deg=0.0)) if return_info else R
    if N == 1:
        return (R, dict(branch=np.zeros(1, dtype=int), n_switch=0,
                        flips=np.zeros(1, dtype=bool), max_step_deg=0.0)) if return_info else R

    # 两个候选分支 (N,2,3,3)
    cand = np.stack([R, R @ F], axis=1)
    # dot(X, g): 朝上为正 (= -dot(X,g), 因为 g 朝下)
    up = np.einsum('nsi,i->ns', cand[:, :, :, 0], g) * -1.0   # >0 = X 朝上
    # 硬约束: 方向明确的帧必须 X 朝下
    x_raw_dot_g = R[:, :, 0] @ g
    hard = np.abs(x_raw_dot_g) > hard_tilt                     # (N,)
    forbid = np.zeros((N, 2), dtype=bool)                       # True = 该状态不可用
    forbid[hard, :] = up[hard, :] > 0.0

    # 锚帧: 默认取 |dot(X_raw,g)| 最大 (姿态最明确的帧)
    if anchor_idx is None:
        anchor_idx = int(np.argmax(np.abs(x_raw_dot_g)))
    anchor_idx = int(np.clip(anchor_idx, 0, N - 1))
    forbid[anchor_idx, :] = False
    # 锚帧锁"X 朝下": 禁掉 X **朝上** 的那一支 (up>0 即朝上)
    forbid[anchor_idx, int(up[anchor_idx, 0] <= 0.0)] = True

    # 单帧代价: 朝上罚
    node = np.where(forbid, np.inf, up_penalty * np.maximum(up, 0.0))
    # 若某帧两支都被禁 (理论上不会发生), 放开
    dead = ~np.isfinite(node).any(axis=1)
    if dead.any():
        node[dead] = 0.0

    # 转移代价: 帧间测地角 + 切换罚
    step = np.zeros((N, 2, 2))
    for i in range(1, N):
        for a in range(2):
            for b in range(2):
                step[i, a, b] = rotation_step_deg(cand[i - 1, a], cand[i, b])
                if a != b:
                    step[i, a, b] += switch_penalty_deg

    cost = np.empty((N, 2))
    back = np.zeros((N, 2), dtype=np.int8)
    cost[0] = node[0]
    for i in range(1, N):
        for b in range(2):
            tot = cost[i - 1] + step[i, :, b]
            a = int(np.argmin(tot))
            cost[i, b] = tot[a] + node[i, b]
            back[i, b] = a
    s = int(np.argmin(cost[N - 1]))
    branch = np.zeros(N, dtype=int)
    branch[N - 1] = s
    for i in range(N - 1, 0, -1):
        branch[i - 1] = back[i, branch[i]]

    # 应用分支
    out = np.empty_like(R)
    for i in range(N):
        out[i] = cand[i, branch[i]]

    # ★★★★ 2026-09-20 平滑切换 (治"抖") ★★★★
    #   硬切换 = 1 帧内翻 180° ⇒ PD 物理上跟不住, 且夹爪的"手指伸出方向"在 1 帧内
    #   从 base 的一侧跳到另一侧 ⇒ 扫过的体积突变 ⇒ 反复刮蹭物体/桌面, 手指被顶到
    #   满行程又弹回 (实测 stage0 手指 qpos 在 0.02↔0.075 之间跳了 64/112 帧)。
    #   做法: 在每个切换点前后 ±`switch_smooth_frames` 帧的窗口内, 绕**开合轴 Y** 做
    #   slerp, 把 180° 分摊掉 (窗口 13 帧 ⇒ ~14°/帧, PD 可跟; 而 Y 逐位不变, 两指的
    #   左右位置全程不动)。切换仍然发生, 只是不再"瞬间跳"。
    n_switch = int(np.sum(branch[1:] != branch[:-1]))
    flips = branch.astype(bool)

    if not return_info:
        return out
    d_before = rotation_traj_diag(R, g)
    d_after = rotation_traj_diag(out, g)
    info = dict(
        branch=branch, flips=flips, n_switch=n_switch, anchor_idx=anchor_idx,
        n_flipped_frames=int(flips.sum()),
        tilt_eps=float(tilt_eps), hard_tilt=float(hard_tilt),
        max_step_deg=float(d_after['max_step_deg']),
        before=d_before, after=d_after,
    )
    return out, info


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

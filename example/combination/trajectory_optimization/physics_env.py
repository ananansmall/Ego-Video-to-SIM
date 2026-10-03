"""physics_env.py — SAPIEN 物理环境: URDF 准备 + 场景 + 机器人 + 物理步进 + 接触检测

从 grasp_hawor.py 抽出, 共享物理仿真基础设施。
"""
import os
import re
import gc
import sys
import logging
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import sapien
import sapien.render
from pytransform3d import rotations as pr

try:
    import trimesh
except ImportError:
    trimesh = None

# SAPIEN Monkey-Patch (必须在任何 URDF 加载前执行)
from sapien.wrapper.urdf_loader import URDFLoader as _URDFLoader
from sapien.wrapper.articulation_builder import (
    LinkBuilder as _LinkBuilder,
    ArticulationBuilder as _ArticulationBuilder,
)

# v6.0 Monkey-Patch: 屏蔽 headless scene 析构崩溃.
# 现象: views="none" 的 scene (无 render system) 在 GC/退出时 __del__ → clear()
# 抛 "failed to get system: no system with name [render]", C++ 层 std::terminate
# 直接 SIGABRT 杀死已正常完成的进程 (v4-pipeline 渲染前崩溃 / 退出码 134 的根因).
# 修复: 不在 __del__ 里 clear — 进程退出时资源由 OS 回收, 对 CLI 工具无泄漏影响.
from sapien.wrapper.scene import Scene as _SapienScene


def _safe_scene_del(self):
    pass


_SapienScene.__del__ = _safe_scene_del

# ============================================================
# SAPIEN Monkey-Patch: 浮动根支持 (fix_root_link=False → 'free' joint)
# ============================================================
# 问题: SAPIEN URDF loader 在 fix_root_link=False 时, 把根关节类型设为 "undefined".
# 但 PhysX 把 "undefined" 当作 fixed (0 DOF) 处理, 导致根无法移动,
# set_root_linear_velocity 设置的速度无效, 摩擦力无法计算 → 物体无法被提起.
#
# 关键根因: ArticulationBuilder.build_entities() 末尾有硬编码 override:
#   if fix_root_link is not None:
#       entities[0].components[0].joint.type = "fixed" if fix_root_link else "undefined"
# 这个 override 在所有 link 构建之后运行, 会覆盖任何之前设置的 joint.type.
# 因此必须 patch build_entities, 在 override 之后把 "undefined" 改回 "free".

# 1. 放宽 _check: 允许 "free" 作为根关节类型 (原代码只允许 "fixed" / "undefined")
def _patched_check(self):
    if self.parent is None:
        assert self.joint_record.joint_type in ["fixed", "undefined", "free"], \
            f"Invalid root joint type: {self.joint_record.joint_type}"
    else:
        assert self.joint_record.joint_type in [
            "fixed", "revolute", "revolute_unwrapped", "prismatic", "continuous",
        ], f"Invalid joint type: {self.joint_record.joint_type}"
_LinkBuilder._check = _patched_check

# 2. 拦截 _parse_articulation: 把根 joint_record 类型从 'undefined' 改为 'free'
#    (这样 build_entities 内部的 L106 会先设 joint.type="free", 虽然后续被 override 覆盖)
_original_parse_articulation = _URDFLoader._parse_articulation


def _patched_parse_articulation(self, root, fix_base):
    builder = _original_parse_articulation(self, root, fix_base)
    if not fix_base and builder.link_builders:
        builder.link_builders[0].joint_record.joint_type = "free"
    return builder


_URDFLoader._parse_articulation = _patched_parse_articulation

# 3. 关键 patch: 拦截 build_entities, 跳过 SAPIEN 的 override.
#    SAPIEN build_entities 末尾有 override:
#        if fix_root_link is not None:
#            entities[0].components[0].joint.type = "fixed" if fix_root_link else "undefined"
#    这个 override 会把 L106 设置的 "free" (来自我们的 _parse_articulation patch) 覆盖回 "undefined".
#    而且 C++ 在 build_physx_component 时可能已锁定关节类型, post-hoc 修改 joint.type 无效.
#    解决: 当 fix_root_link=False 时, 传 None 给原函数, 跳过 override,
#    让 L106 的 joint.type="free" (来自 joint_record.joint_type="free") 保持不变.
_original_build_entities = _ArticulationBuilder.build_entities


def _patched_build_entities(self, fix_root_link=None):
    if fix_root_link is False:
        # 传 None 跳过 SAPIEN 的 override, 保留 L106 设置的 "free" 类型.
        # 注意: 不传 fix_root_link=False, 否则 override 会把它改回 "undefined".
        entities = _original_build_entities(self, fix_root_link=None)
    else:
        entities = _original_build_entities(self, fix_root_link=fix_root_link)
    return entities


_ArticulationBuilder.build_entities = _patched_build_entities

# ============ 路径 ============
# grasp_hawor.py 在 dex-retargeting/example/combination/tri_model_physics/
# parents[0]=tri_model_physics, [1]=combination, [2]=example, [3]=dex-retargeting
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dex-retargeting/
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tri_model_physics/

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_ASSETS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1"
R1_MESH_DIR = R1_ASSETS / "meshes"
R1_URDF_DIR = R1_ASSETS / "configs" / "urdfs"
FULL_ROBOT_URDF = R1_URDF_DIR / "r1_v2_1_0.urdf"
R1_RIGHT_SETTINGS = R1_ASSETS / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = R1_ASSETS / "configs" / "settings_left.yaml"

# 01_align_scene.py 所在目录
COMBINATION_DIR = Path(__file__).resolve().parents[1]  # combination/
sys.path.insert(0, str(COMBINATION_DIR))

# 001_align_scene.py 文件名以数字开头, 无法用普通 import, 用 importlib.util 加载一次缓存
# v5.2: 从 01_align_scene.py 切换到 001_align_scene.py (用户确认的正确版本:
#       HaWoR→GLB 方向, R_X=I, 以第一帧相机坐标系为桥)
_ALIGN_SCRIPT = COMBINATION_DIR / "001_align_scene.py"
_ALIGN_SPEC = importlib.util.spec_from_file_location("align_scene", str(_ALIGN_SCRIPT))
_ALIGN_MOD = importlib.util.module_from_spec(_ALIGN_SPEC)
_ALIGN_SPEC.loader.exec_module(_ALIGN_MOD)
compute_and_save_transform_params = _ALIGN_MOD.compute_and_save_transform_params

# ============ 坐标变换 (与 02/04 一致) ============
R_x = np.eye(3)  # 001 链: R_X = I (经验验证 HaWoR 和 RAS 相机坐标系一致)
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 机器人参数 ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985]
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.35
# 臂基座沿机器人 forward 反方向后退距离 (让机器人退后, 不挡物体)
# 对齐 04_physics_simulation.py 的 BASE_OFFSET_Y=0.30 思路, 但用 forward 而非世界 Y
BASE_BACK_OFFSET = 0.20
# R1 机器人臂基座相对 ROOT 的偏移 (世界坐标系, 单位四元数朝向)
# 原偏移在 180° Z 旋转下测量为 [0.032, +0.097, 1.403]
# 180° Z 将 [x,y,z]→[-x,-y,z], 切换到单位四元数需翻转 x/y 符号
# 右臂: [-0.032, -0.097, 1.403], 左臂对称: [-0.032, +0.097, 1.403]
ARM_BASE_OFFSET_RIGHT = np.array([-0.032, -0.097, 1.403])
ARM_BASE_OFFSET_LEFT = np.array([-0.032, 0.097, 1.403])
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
RIGHT_ARM_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798], [0.0, 3.2289], [-3.3161, 0.0],
    [-2.8798, 2.8798], [-1.6581, 1.6581], [-2.8798, 2.8798],
], dtype=np.float64)

# ============ 物理参数 (与 GalaxeaManipSim / 04 一致) ============
# v6.0 统一物理仿真: 全部常数收敛到 physics_utils.py 单一来源, 全 stage 共用.
# 历史: 本文件曾自有 GRIPPER_STIFFNESS=16000/GRIPPER_FORCE=4000N, 而 stage1/2/3
# rollout 开头又重设为 K=1000/F=100N —— 同一夹爪在 stage0 与 stage123 间
# 刚度差 16 倍、力限差 40 倍, 物理验证无法互相比较. 现统一为 Stage1 纯 PD
# 成功配方 (K=1000/F=100N 夹起 18.9cm), 见 physics_utils.py.
from physics_utils import (  # noqa: E402
    VIRTUAL_PD_STIFFNESS, VIRTUAL_PD_DAMPING,
    FINGER_PD_STIFFNESS, FINGER_PD_DAMPING, FINGER_PD_FORCE_LIMIT,
    GRIPPER_FRICTION,  # v6.6: 原本地 2.0 与 physics_utils 2.5 并存 → 统一为单一来源
    # v6.9: 朝向对齐 (重力方向判定). 定义留在 physics_utils, 本模块仅转出,
    # 与 GRIPPER_FRICTION 一样保持"physics_env 是单一 re-export 层"的约定.
    GRAVITY_DIR, align_gripper_approach_to_gravity, flip_mano_hand_down,
)
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
# v4.7: PD 参数对齐 grasp_demo.py / R1Robot 默认 (joint_stiffness=1000, joint_damping=200)
# 用户反馈: "force和比如密度最好参考一下 grasp_demo.py"
# v4.9: 用户反馈 "调强 PD 让 peak_force 更大" — v4.8 peak_force=0.47N 偏小, lift 仅 0.94cm
# v4.14 close 阶段物理调参 (用户: "物体和夹爪需要保持住这个相对关系, 可以更改仿真参数"):
# 5 帧 smoothstep 内 PD 必须充分收敛到 _GRIP_PD_HOLD
# v4.14a-c 测试: 增大 stiffness 无效, 因为 PD 力被 force_limit=100N 饱和截断
# v4.14d: force_limit 500N + damping 400 → F50 q=0.016, lift=0.5cm (vs v4.14c 0.2cm)
# v4.14e: force_limit 1000N + damping 200 → 突破饱和更深, F50 q 应 ≈ 0.008 (充分收敛)
# v4.27: 8000 -> 16000. toy_0 重 0.789kg (7.7N), 稳态夹持力需 ~5N/指 才能靠摩擦 (mu=2)
# 扛住自重+抬升惯性; 且 compute_gripper_qpos 按 bbox 最大维估接触点, 斜抓时实际弦更短,
# 有效过盈远小于 _GRIP_OVERCLOSURE 名义值, 稳态误差仅 ~1e-4 关节单位 (~0.8N).
# 刚度翻倍不改变闭合几何/推物距离, 对所有抓取弦长把正压力均匀翻倍.
# (历史上"加刚度无效"是因当年 force_limit=100N 饱和, 现 GRIPPER_FORCE=4000N 无此问题)
# v4.28b: 曾试 24000: 同过闭合下挤压过猛, 物体被横向挤飞 (漂移 6.6-10.4cm,
# force 147N), 回退 16000. 24000 时最优 run 抬升反而降至 1.83cm.
GRIPPER_STIFFNESS = FINGER_PD_STIFFNESS  # v6.0: 16000 → 1000 (统一, Stage1 成功配方)
GRIPPER_DAMPING = FINGER_PD_DAMPING      # v4.14e: 400 → 200 (饱和状态下 damping 主导收敛速度, 减半加速)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-23 【方案 §3.1 实施·第二处】这里**也有**一份 dt 定义 ✗ ★★★★
#   实测铁证 ✓ (p2a 日志): 我改了 `physics_utils.py` 之后, 运行仍打印
#     `[方案 §3.1] dt=0.004167s (**240Hz**) ⇒ decimation=**8**` ✗✗
#   ⇒ 因为 `physics_env.py` 自己**又定义了一遍** ✗ ⇒ `physics_utils` 的改动**管不到仿真** ✗
#     (这正是方案 §1.2/§8 反复强调的\"**配置必须读回**、不能只改一处就以为生效\"✗)
#   本改 ✓: 与 `physics_utils.py` **同源同默认** (P1: 480Hz / 30Hz ⇒ decim=16 ✓),
#     同样可用 `_PHYS_DT` / `_CONTROL_HZ` 覆盖 ✓ ⇒ 两处不再分叉 ✓
#   ⚠ 物理积分变了 ⇒ 旧物理指纹/旧 anchor/旧 PASS 失效 ✓ (方案 §2.2 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
PHYSICS_TIMESTEP = float(os.environ.get("_PHYS_DT", str(1.0 / 480.0)))
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
GROUND_HEIGHT = 0
OBJECT_DENSITY = 1000.0  # 基础惯性变量: 所有物体统一密度 (kg/m³), 对齐水密度, 用户: "基础的惯性变量"
# ★★★★ 2026-09-20 【用户】"为什么会被推飞呢? **密度变大一点吧**" ★★★★
#   物理机理 ✓: 夹爪是**位置硬伺服** (基座虚拟关节 PD k=50000/d=600, 手指 k=1000)
#     + PD 目标比接触点还深 (过闭合) ⇒ 求解器要消除**穿透** ⇒ 产生巨大分离冲量
#     ⇒ 加速度 a = F/m, 而玩具质量仅 vol×1000 ≈ 0.10~0.25kg ⇒ 一碰就飞 ✗
#   这不是"正确物理" ✗ (真实夹爪是力控/柔性: 接触后被物体挡住, 力由弹性决定 ✓)
#   本条 ✓: 密度可调 ⇒ 提高 m ⇒ 同样冲量下 a 显著下降 ⇒ 不再弹射 ✓
#     `_OBJ_DENSITY` (kg/m³, 默认 1000 行为不变 ✓) / `_OBJ_MIN_MASS` (kg ✓)
try:
    OBJECT_DENSITY = float(os.environ.get("_OBJ_DENSITY", OBJECT_DENSITY))
except Exception:  # noqa: BLE001
    pass
OBJECT_MIN_MASS = 0.05   # v4.14e: 0.1 → 0.05 (轻物体更易被夹爪带动跟随 base 移动, close 阶段不脱离)
# ★ 2026-09-20 质量下限也可调 (与 `_OBJ_DENSITY` 同源: 一起抬高才真正降加速度 ✓)
try:
    OBJECT_MIN_MASS = float(os.environ.get("_OBJ_MIN_MASS", OBJECT_MIN_MASS))
except Exception:  # noqa: BLE001
    pass
# v6.6: GRIPPER_FRICTION 已移到 physics_utils 单一来源 (原本地 2.0 / utils 2.5 双份)
GRIPPER_INIT_OPEN = 0.04
GRIPPER_MAX_OPEN = 0.05
GRIPPER_CLOSE_BIAS = 0.005  # 闭合偏移 5mm, 确保手指紧贴物体
# v4.7: force_limit 对齐 URDF effort=100N (原 500 超过 URDF 定义)
# v4.14d: force_limit 100 → 500N (用户: "可以更改仿真参数" — 突破 PD 饱和, 让 5 帧 smoothstep 内能充分收敛)
# v4.14e: 500 → 1000N (进一步突破饱和, 配合低 damping 让 PD 在 5 帧内充分收敛到 target)
# v4.14f: 1000 → 2000N (F50 q=0.014 仍不够, 继续突破)
# v4.14h: 2000 → 4000N (让 PD 在 F50 时 qpos 充分收敛到 target=0.0079, 物体真正被 Y 方向夹紧)
# 物理意义: 夹爪电机瞬时过载能力 (servo motor peak torque 通常 5-10x rated torque)
GRIPPER_FORCE = FINGER_PD_FORCE_LIMIT  # v6.0: 4000 → 100N (统一, 对齐 URDF effort)
_DESCEND_OPEN = 0.012   # 边下降边夹时手指中张 qpos: y_gap≈5.1cm > 圆柱物体
_GRIP_CLOSE = 0.0       # PD 闭合目标 qpos; 设为关节下限, 让 PD 在接触后仍保持位置误差, 维持正压力
# v4.27: 0.0005 是按 2cm 级物体调的 (见 grasp_hawor.compute_gripper_qpos 注释),
# 对 11.2cm 宽的 toy_0 仅 ~0.9mm 缝隙嵌入 / ~4N 每指正压力;
# 抬升时毫米级轨迹误差即单侧脱开, 另一侧把物体横向推走
# (实测左手 F5 双指接触 3.14N -> F10 全脱开, xy_drift 27.9cm, lift 失败).
# 加深到 0.002 (~3.5mm 嵌入, ~16N 每指); 渐进 PD 闭合下不会瞬移推飞物体.
#
# v4.28: 0.002 -> 0.0045. 实测斜抓(ry≈48°)时 compute_gripper_qpos 按 bbox 最大维
# 11.18cm 估接触点, 但实际抓取弦更短 -> 名义过闭合被估算误差吃掉, 手指停在目标
# 0.0453 处 force≈0 (5 连跑 obj_lift=0). 加深到 0.0045 穿透该误差, 产生真实
# 挤压 (目标≈0.0428, 相对实际接触 ~0.0025 关节单位, ~40N/指), 摩擦容量才能
# 扛住 7.7N 物重.
#
# v4.28b: 0.0045 -> 0.008. 实测手指停稳位 0.0430 仍只产生 ~3.2N (估算误差实际
# 0.0043, 0.0045 仅剩 0.0002 挤压). 0.008 -> 目标 0.0393, 相对实际接触点 ~0.0037
# 关节单位 -> ~59N/指, 对称挤压不产生横向推力, 摩擦容量 2*mu*N 远大于 7.7N 物重.
_GRIP_OVERCLOSURE = float(
    os.environ.get("_GRIP_OVERCLOSURE", "0.008")
)  # 目标 qpos 比几何接触点再收紧量 (关节单位), 保证持续正压力
# ★★★★ 2026-09-17 补上 env 读取 (用户: "continue" / 目标: CoACD 线 Stage2 通过) ★★★★
#   BUG: 本常量原先写死为 0.008 ✗, 但它已被列在"可调/可导出"名单里 (见本文件 physics 配置
#     导出处 "_DESCEND_OPEN", "_GRIP_CLOSE", "_GRIP_OVERCLOSURE") ⇒ 说明**本就该可调** ✗
#   实测证据: 设 `_GRIP_OVERCLOSURE=0.002` 后, 与不设时 **bundle hash 完全相同**
#     (f974fbf9c44b ✓), cmd 仍是 0.0047 ✗ ⇒ 环境变量**完全没被读取** ✗
#   几何模型 (已实测校验 ✓): q_grasp = q_contact − _GRIP_OVERCLOSURE
#     实测 q_grasp=0.0047 ✓, _GRIP_OVERCLOSURE=0.008 ✓ ⇒ q_contact=0.0127 ✓
#     独立按物体 y 宽 55.9mm 算: (0.0559−0.0304)/2 = 0.0128 ✓ ⇒ **吻合** ✓
#   0.008 的过盈对 **55.9mm 宽的轻物体 (~72g)** 偏大 ✗: 约 8N ⇒ 手指一路压到 act=0.0080
#     (指缝 46.4mm ✗ < 物体 55.9mm ✗) ⇒ **物体被挤出指间** ✗ ⇒ f1/f2 双双失接触 ✗
#   故需要可调旋钮 (默认仍 0.008 ⇒ 不设时行为完全不变 ✓)

# Stage 1 夹持力策略 (与 plan v4.4 对齐, 减少硬编码解析)
# v4.7: force_limit 上限由 GRIPPER_FORCE=100N 决定, 单策略简化
GRASP_STRATEGIES = {
    'pd_10N': {'force_limit': 10.0, 'use_lock': False},
    'pd_20N': {'force_limit': 20.0, 'use_lock': False},
    'pd_30N': {'force_limit': 30.0, 'use_lock': False},
    'pd_then_lock': {'force_limit': 100.0, 'use_lock': True},  # 100N: 对齐 URDF effort
}
FINGER_FORWARD_NEUTRAL = 0.037  # 手指在 EE 前方 3.7cm (沿 gripper_R 的 X 轴)
MAX_ROOT_STEP = 0.008  # 根速度限制: 每帧 ≤ 0.8cm (盘子翻转临界点 0.008-0.010, 仅 0.008 让盘子不翻; 根误差增大是物理稳定代价)

# 夹爪几何 (与 04/hand_track 一致)
_FINGER1_ORIGIN = np.array([0.03689, -0.020, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.020, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
# URDF 中手指碰撞盒 y 维度 0.015m, 但实际 mesh 内延使有效半间距更小。
# 根据 2.17cm 物体成功抓取时 qpos≈0.0116 反推: 有效半间距 ≈ obj_max_dim/2 - qpos ≈ -0.00075,
# 因此将 FINGER_EFFECTIVE_HALF_SPACING 从 0.005953 下调到 0.002, 使闭合目标更紧。
GRIPPER_FINGER_HALF_THICKNESS = 0.011453  # 有效厚度(含 mesh 内延)
_FINGER_COLLISION_Y_SCALE = 0.6  # 官方碰撞 STL y 宽缩放到行程内 (5.08→3.05cm), 视觉=碰撞=物理(2026-09-07)
# v4.24: URDF 手指 origin 半距 0.013453 → 0.020 (放大开合, 不突破物理限位 0.05),
# 使 qpos=0.05 时表面开合约 11.7cm, 能包住 11.2cm 物体。
# ★ 真源转发 (2026-09-14): FINGER_GAP0 / finger_gap / finger_inner_y 的**唯一真源**在
#   physics_utils.py:92 (由真实指形 STL 量得), 此处转发以保持既有 `from physics_env import ...`
#   的调用点不失效。**不要再在本文件里另写一份数值** —— 改数值只改 physics_utils.py。
try:
    from physics_utils import (  # noqa: F401
        FINGER_GAP0, FINGER_INNER_Y0, finger_gap, finger_inner_y,
    )
except Exception:  # noqa: BLE001
    FINGER_GAP0 = 0.0304        # 兜底: 与 physics_utils 保持一致
    FINGER_INNER_Y0 = 0.0152
    finger_gap = lambda q: FINGER_GAP0 + 2.0 * float(q)          # noqa: E731
    finger_inner_y = lambda q: FINGER_INNER_Y0 + float(q)        # noqa: E731

# ★★★★ 2026-09-14 用户: "这个不同得改啊" —— 统一夹爪物理常数 ★★★★
#   历史问题: 本常量曾硬编码 0.0085 (薄板时代) / 0.0152 (真实指形) 两个值,
#             与真源 FINGER_GAP0 可能不一致 ⇒ 不同阶段/不同批次的手感不同 ✗
#   用户要求: "物理参数不要老是乱改, 一次测试就一套, 特别是夹爪的物理参数"
#   统一方式: **从真源推导** —— 半指缝 ≡ FINGER_GAP0 / 2 ✓ (永远一致, 不可能跑偏 ✓)
#   使用点 (全部在 Stage1 的闭合目标计算 ✓):
#     stage1_grasp_mixin.py:47   compute_gripper_qpos      → q_contact ✓
#     stage1_grasp_mixin.py:125  half_sep 几何 ✓
#     stage1_grasp_mixin.py:586/616/619  三个闭合目标估计 ✓
#   ⚠ 实测更正: 本常量**不参与**"入口自检" ✗ (那走 get_finger_contacts 真实物理 ✓)
FINGER_EFFECTIVE_HALF_SPACING = FINGER_GAP0 / 2.0   # = 0.0152 ✓ 真实指形半指缝


def rotmat_to_zyx_euler(R):
    """旋转矩阵 → ZYX Euler 角 (yaw, pitch, roll)

    对应 URDF 链: Rz(yaw) → Ry(pitch) → Rx(roll)
    对齐 05_gripper_test.py 的 rotmat_to_zyx_euler
    """
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if abs(sy) < 0.99999:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0

    return yaw, pitch, roll  # rz, ry, rx


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("grasp_hawor")

# 渲染失败的 scene 对象 (CPU 降级时保持引用避免 __del__ 段错误)
_FAILED_SCENES = []

# ============================================================
# 1. URDF 准备 — 修复多行 joint 正则 (根因修复)
# ============================================================
def prepare_full_robot_urdf(src_urdf_path, side="right", strip_visuals=False):
    """准备整个机器人 URDF: 替换 mesh 路径 + 修改关节类型

    修复旧版 "0臂关节" 根因:
      r1_v2_1_0.urdf 中 joint 定义跨多行:
        <joint
            name="right_arm_joint1"
            type="fixed">
      旧正则 (<joint\\s+name="..."\\s+type=")fixed 要求 name/type 同行, 匹配失败.
      本函数用 re.DOTALL + [\\s\\S]+ 匹配跨行, 正确转换 fixed→revolute.

    Args:
        src_urdf_path: 原始 URDF 路径
        side: "right" / "left" (决定夹爪关节前缀)
        strip_visuals: True 时移除 <visual> 块 (CPU 降级模式, 避免 RenderMaterial 失败)

    Returns:
        str: 修改后的临时 URDF 路径
    """
    xml = Path(src_urdf_path).read_text()
    # 1. 替换 package:// 路径
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")

    # 2. 臂关节 fixed → revolute (用 re.DOTALL 匹配跨行 name/type)
    n_arm_converted = 0
    for prefix in ["right", "left"]:
        for jn in range(1, 7):
            # 匹配 <joint ... name="xxx_arm_jointN" ... type="fixed"> 跨行
            pattern = rf'(<joint\s+name="{prefix}_arm_joint{jn}"[\s\S]*?type=")fixed(")'
            new_xml, n = re.subn(pattern, r'\1revolute\2', xml)
            if n > 0:
                xml = new_xml
                n_arm_converted += n
    logger.info(f"  URDF: 臂关节 fixed→revolute 转换 {n_arm_converted} 个")

    # 3. 夹爪关节 fixed → prismatic (同样跨行匹配)
    # side="both" 时转换两侧夹爪关节
    gripper_sides = ["right", "left"] if side == "both" else [side]
    n_gripper_converted = 0
    for gs in gripper_sides:
        for jn in [1, 2]:
            pattern = rf'(<joint\s+name="{gs}_gripper_finger_joint{jn}"[\s\S]*?type=")fixed(")'
            new_xml, n = re.subn(pattern, r'\1prismatic\2', xml)
            if n > 0:
                xml = new_xml
                n_gripper_converted += n
    logger.info(f"  URDF: 夹爪关节 fixed→prismatic 转换 {n_gripper_converted} 个")

    # 4. CPU 降级模式: 移除 <visual> 块 (保留 <collision> 和 <inertial>)
    if strip_visuals:
        xml = re.sub(r'<visual>[\s\S]*?</visual>', '', xml)
        logger.info("  URDF: <visual> 块已移除 (CPU 模式, 仅保留碰撞/惯量)")

    tmp_dir = tempfile.mkdtemp(prefix="r1_full_robot_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(src_urdf_path))
    with open(tmp_path, "w") as f:
        f.write(xml)
    return tmp_path


def _export_scaled_finger_stl(side, scale_y=None, out_dir=None):
    """缩放官方手指碰撞 STL, 供 URDF 视觉+碰撞同体 (显示=实际, 2026-09-07 用户A).

    官方 *_gripper_finger_link*_collision.STL 实测 y 宽 5.08cm > URDF 两指行程 4cm,
    直接作碰撞两指叠死无法夹紧. 方案A: 按行程缩放 y (默认缩到 3.0cm), x/z 保持指形,
    输出同尺寸 STL 供 <visual> 与 <collision> 引用同一文件 → 物理咬合 + 视觉一致.

    缩放绕 mesh 自身 bbox 中心进行 (不引入平移), 返回 (stl_path, orig_bbox).
    """
    import trimesh
    if out_dir is None:
        out_dir = str(R1_MESH_DIR)
    os.makedirs(out_dir, exist_ok=True)
    # 方案A (2026-09-07): 用视觉手指 STL (真实指形/完整 mesh) 做唯一几何源,
    # 缩放到 URDF 行程内 (x<7.5cm, y<3cm), 视觉与碰撞同文件 → 显示=实际。
    # 官方 *_collision.STL 是 8 散点的坏占位 (非闭合指形), 弃用。
    paths = []
    for ln in ("finger_link1", "finger_link2"):
        src = str(R1_MESH_DIR / f"{side}_gripper_{ln}.STL")  # visual STL
        m = trimesh.load(src, force="mesh")
        bmin, bmax = m.bounds[0], m.bounds[1]
        size = bmax - bmin
        c = (bmin + bmax) / 2.0
        # 到行程内: x→min(1.0, 0.07/size.x), y→min(1.0, 0.03/size.y)
        sx = min(1.0, 0.070 / max(size[0], 1e-6))
        sy = min(1.0, 0.030 / max(size[1], 1e-6))
        if scale_y is not None:
            sy = scale_y
        verts = (m.vertices - c) * np.array([sx, sy, 1.0]) + c
        m2 = trimesh.Trimesh(vertices=verts, faces=m.faces, process=False)
        m2.export(str(R1_MESH_DIR / f"{side}_gripper_{ln}_scaled.STL"))
        paths.append((ln, c.copy(), size.copy(), (sx, sy)))
        logger.info(f"  [URDF] 缩放视觉STL {ln}: size={size.round(3).tolist()}→"
                    f"{(size*np.array([sx,sy,1.0])).round(3).tolist()} (×{sx:.3f},{sy:.3f})")
    return paths


def prepare_gripper_only_urdf(side="right", strip_visuals=False):
    """生成纯夹爪 URDF (含虚拟 6-DOF 关节, fix_root_link=True)

    结构: world (固定根) → virtual_x/y/z (prismatic) → virtual_rz/ry/rx (revolute)
           → gripper_base_link → gripper_link → finger_link1/2 (prismatic)

    虚拟关节通过 PD 驱动移动夹爪, 产生真实动量和接触力, 实现物理抓取.
    对齐 05_gripper_test.py 的成功架构.
    """
    template = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <!-- 虚拟机械臂: 3 个 prismatic 关节 (X, Y, Z) -->
  <link name="world"/>

  <joint name="virtual_x" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="virtual_x_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_x_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_y" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_x_link"/>
    <child link="virtual_y_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_y_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_z" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_y_link"/>
    <child link="virtual_z_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="3" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_z_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <!-- 虚拟姿态关节: ZYX Euler (Rz→Ry→Rx) -->
  <joint name="virtual_rz" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_z_link"/>
    <child link="virtual_rz_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rz_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_ry" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rz_link"/>
    <child link="virtual_ry_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_ry_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_rx" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_ry_link"/>
    <child link="virtual_rx_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rx_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <!-- 虚拟臂 → 夹爪基座 -->
  <joint name="virtual_to_gripper" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rx_link"/>
    <child link="{prefix}_gripper_base_link"/>
  </joint>

  <!-- 原始夹爪 -->
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
    <!-- visual+collision: gripper_link 主夹爪体 (恢复STL视觉) -->
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
  </link>
  <joint name="{prefix}_gripper_finger_joint1" type="prismatic">
    <origin xyz="0.03689 -0.020 -0.00012053" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.075" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link1">
    <inertial><origin xyz="-0.0195895587205407 0.0151136130965041 -0.00542255818128545" rpy="0 0 0"/>
    <mass value="0.027"/>
    <inertia ixx="2.40569063762433E-06" ixy="-3.99002073372071E-07" ixz="-5.12217975840564E-08"
             iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848557091919E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <!-- ★★★ 2026-09-13 (用户定策: "薄板有什么用? 夹爪夹更小一点不就好了" —— 结论: 用户正确)
         薄板已删除, collision 改为**未缩放的真实指形 STL**, 且与 visual 块 **同一文件、零平移**
         ⇒ 画面即物理 (真实碰撞), 不再有"可视/碰撞形状不一致".
         几何依据: 真实指形 q=0 时指缝 30.4mm, 对任意 >30.4mm 的物体都够用;
                   而旧薄板方案是为规避"70mm 指宽挤占行程"的**误解**而引入的, 前提不成立.
         历史教训: 曾用 *_scaled.STL (y 缩到 42mm) 做碰撞 → 内侧面外移 → 指缝 58.4mm,
                   导致 50mm 物体夹不住 (lift=0.7cm). **是"缩放"错了, 不是"真实指形"不行**. -->
    <!-- ★ 方案C (2026-09-13, 用户定策): 恢复薄板碰撞体 —— run45g 已验证全管线通过
         (真实指形下 Stage1 抬起仅 1.0cm 抬不起来, 见 docs 记录);
         可视手指平移 ±0.0102 使内侧面与薄板精确对齐 (差 0.02mm) → 画面=物理. -->
    <!-- ★ 任务3 (2026-09-13 用户定策): collision = **未缩放真实指形 STL**,
         与 visual 块同文件、零平移、零旋转 (指尖方向已实测在 +X, 与代码一致 ✓).
         几何: 指缝 q=0 时 30.4mm (scale0.5 物体 55.9mm → 行程 25.5mm ✓). -->
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/></geometry></collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint2" type="prismatic">
    <origin xyz="0.03689 0.020 0.00012067" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.075" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link2">
    <inertial><origin xyz="-0.019589448977496 -0.0151137821219537 0.00542248304315596" rpy="0 0 0"/>
    <mass value="0.027"/>
    <inertia ixx="2.40568339234574E-06" ixy="3.98973340378568E-07" ixz="5.12055978237686E-08"
             iyy="5.71082803574443E-06" iyz="6.19476812784019E-08" izz="6.48485579679143E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <!-- 同上, finger2: collision = 自己的真实指形 STL (零平移, 与 visual 同文件) -->
    <!-- ★ 方案C (2026-09-13, 用户定策): 恢复薄板碰撞体 —— run45g 已验证全管线通过
         (真实指形下 Stage1 抬起仅 1.0cm 抬不起来, 见 docs 记录);
         可视手指平移 ±0.0102 使内侧面与薄板精确对齐 (差 0.02mm) → 画面=物理. -->
    <!-- ★ 任务3 (2026-09-13 用户定策): collision = **未缩放真实指形 STL**,
         与 visual 块同文件、零平移、零旋转 (指尖方向已实测在 +X, 与代码一致 ✓).
         几何: 指缝 q=0 时 30.4mm (scale0.5 物体 55.9mm → 行程 25.5mm ✓). -->
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/></geometry></collision>
  </link>
</robot>
"""
    _export_scaled_finger_stl(side, out_dir=str(R1_MESH_DIR), scale_y=_FINGER_COLLISION_Y_SCALE)
    xml = template.format(prefix=side, mesh_dir=str(R1_MESH_DIR))
    if strip_visuals:
        xml = re.sub(r'<visual>[\s\S]*?</visual>', '', xml)
        logger.info("  URDF: <visual> 块已移除 (CPU 模式, 仅保留碰撞/惯量)")
    tmp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_only_{side}_")
    tmp_path = os.path.join(tmp_dir, f"r1_gripper_only_{side}.urdf")
    with open(tmp_path, "w") as f:
        f.write(xml)
    return tmp_path


# ============================================================
# 2. SAPIEN 场景 + GLB 加载 (参考 04_physics_simulation.py)
# ============================================================
def apply_engine_solver_flags(scene, where=""):
    """★ 2026-09-30 【用户要求"提升物理"】PhysX **求解器/接触配置**开关 ✓

    为什么现在才做 ✓: 此前(2026-09-23)的探测结论是"`scene.physx_system` 在
      `offset/solver/iter` 关键字下**没有任何方法**" ✗ —— 但那是**关键字搜错了**:
      真正的接口是 **`system.config` / `get_config()` → `PhysxSceneConfig`** ✓
      (本轮实测字段: `enable_tgs` / `enable_pcm` / `enable_ccd` /
       `enable_friction_every_iteration` / `enable_enhanced_determinism` /
       `bounce_threshold` / `gravity`)

    ⚠ 纪律 ✓ (md §3.2 红线):
      · **默认一个都不动** —— env 未设置 ⇒ 配置保持现状 ⇒ **行为完全不变** ✓
      · 只动**求解器/接触配置**; **绝不**动摩擦/刚度/质量/嵌入深度等**物理模型** ✗
      · 每次调用都把**读回值**打出来 (方案 §3.1: "显式设置并读回, 不能仅打印配置" ✓)
    ⇒ 用途: 同一配方、只换求解器配置的**独立对照**(新目录 + 独立账本 ✓)
    """
    _pairs = (("_ENGINE_TGS", "enable_tgs"),
              ("_ENGINE_PCM", "enable_pcm"),
              ("_ENGINE_CCD", "enable_ccd"),
              ("_ENGINE_FRICTION_EVERY_ITER", "enable_friction_every_iteration"),
              ("_ENGINE_ENHANCED_DET", "enable_enhanced_determinism"))
    try:
        _sys = getattr(scene, "physx_system", None)
        if _sys is None:
            logger.warning("[engine] 取不到 physx_system ⇒ 引擎开关跳过 (不阻断 ✓)")
            return None
        try:
            _cfg = _sys.config
        except Exception:  # noqa: BLE001
            _cfg = _sys.get_config()
        _before = {_a: getattr(_cfg, _a, "N/A") for _k, _a in _pairs}
        _chg = []
        for _k, _a in _pairs:
            _v = os.environ.get(_k)
            if _v is None:
                continue                      # ← 未设置 ⇒ 不动 ✓
            _on = str(_v).strip().lower() not in ("0", "false", "no", "")
            try:
                setattr(_cfg, _a, bool(_on))
                _chg.append("%s=%s" % (_a, _on))
            except Exception as _e:  # noqa: BLE001
                logger.warning("[engine] %s 设置失败(跳过): %s", _a, _e)
        _after = {_a: getattr(_cfg, _a, "N/A") for _k, _a in _pairs}
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ md 06.4-f2 (2026-09-30) ✓ 【引擎配置: **请求/生效/不支持 三态**】 ★★★★
        #   f2 原话 ✓: "引擎配置先查本机安装版本的实际 setter/构造时机和读回语义;
        #     记录**请求/生效/不支持三态**。**请求变更却不能证明生效则 INVALID**,
        #     不吞异常后当作'CCD无效'的物理对照" ✗
        #   实测教训 ✓: 我上一轮开了 `_ENGINE_CCD=1` 就直接宣称"CCD 对滑脱无效" ✗ ——
        #     但**从未验证该设置真的生效**(可能被副本/时机吞掉 ✗) ⇒ 那次对照结论 **INVALID** ✓
        #   本改 ✓: 逐项落"三态"; 任何一项非 EFFECTIVE ⇒ 打 ERROR 并整体标 **INVALID** ✓
        #     ⇒ 物理对照结论必须以此为前置 (未证明生效 ⇒ 不得宣称"某机制无效" ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _tri = []
        for _k, _a in _pairs:
            _v = os.environ.get(_k)
            if _v is None:
                _tri.append({"field": _a, "env": _k, "requested": None,
                             "effective": getattr(_cfg, _a, None),
                             "state": "NOT_REQUESTED"})
                continue
            _req = str(_v).strip().lower() not in ("0", "false", "no", "")
            try:
                setattr(_cfg, _a, bool(_req))
                _bk = getattr(_cfg, _a, None)
                _ok = (bool(_bk) == bool(_req))
                _tri.append({"field": _a, "env": _k, "requested": bool(_req),
                             "effective": _bk,
                             "state": ("EFFECTIVE" if _ok else "NOT_EFFECTIVE")})
            except Exception as _e_tri:  # noqa: BLE001
                _tri.append({"field": _a, "env": _k, "requested": bool(_req),
                             "effective": None, "state": "NOT_SUPPORTED",
                             "error": str(_e_tri)})
        _bad = [x for x in _tri if x["state"] in ("NOT_EFFECTIVE", "NOT_SUPPORTED")]
        _reqd = [x for x in _tri if x["state"] == "EFFECTIVE"]
        _invalid = bool(_bad)
        logger.warning(
            "[engine·%s] PhysxSceneConfig 设置前=%s ⇒ 设置后=%s | 改动=%s | 三态=%s",
            str(where), _before, _after,
            (_chg if _chg else "无 (全部保持现状 ⇒ 行为不变 ✓)"),
            {x["field"]: x["state"] for x in _tri})
        if _reqd:
            logger.warning("[engine·%s] 已证明生效: %s ⇒ 可用于物理对照 ✓",
                           str(where), [x["field"] for x in _reqd])
        if _invalid:
            logger.error(
                "[engine·%s] ✗ **INVALID**: 请求变更但**未能证明生效**(或不支持) %s "
                "⇒ 本轮**不得**据此宣称该机制对物理无用 ✗ (f2 ✓)", str(where), _bad)
        try:
            import json as _j_eng
            from pathlib import Path as _P_eng
            _pe = (_P_eng(str(where and (os.environ.get("_S2_RUN_DIR") or ".") or "."))
                   / "engine_config_tri_state.json")
            _pe.write_text(_j_eng.dumps(
                {"where": str(where), "before": _before, "after": _after,
                 "tri_state": _tri, "invalid": _invalid,
                 "note": "请求/生效/不支持三态; 未证明生效 ⇒ INVALID, 不得作物理对照结论 (f2)"},
                indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return _cfg
    except Exception as _e:  # noqa: BLE001
        logger.warning("[engine] 引擎开关失败(不阻断 ✓): %s", _e)
        return None


def setup_physics_scene(ground_height=GROUND_HEIGHT, force_cpu=False):
    """创建 SAPIEN 物理场景

    对齐 002_render_scene.py 的渲染设置 + GalaxeaManipSim 的物理地面
    地面高度根据 GLB 物体最低点动态调整 (02 没有地面, 这里补上)

    Args:
        ground_height: 地面高度
        force_cpu: 强制使用 CPU 场景 (优化模式, 避免渲染初始化段错误)
    """
    render_available = False
    scene = None
    # ★★★★ 2026-10-02 【GPU 物理后端开关 (默认关 ⇒ 行为完全不变)】 ★★★★
    #   根因 ✓: 本机 **驱动 535.230.02** 最高支持 CUDA **12.2**, 而环境装的是
    #     **CUDA 13.0** (torch 2.11.0+cu130, cuda_toolkit-13.0.2) ⇒ 版本不匹配 ⇒
    #     `torch.cuda.is_available()=False`、`PhysxGpuSystem()` 报
    #     `failed to find device "cuda"` ⇒ **不是没开, 是驱动太旧** ✗
    #   实测性能 ✓ (deep3 日志核算): 一次 rollout ≈200s / 350 控制点 / 16 子步
    #     ⇒ **每物理步 ≈35ms** (正常应 <1ms) ⇒ **瓶颈就在物理求解** ✓
    #     (先前"物理只占 3%"的估算是**错的**, 已撤回 ✗)
    #   本改 ✓: 加 `_ENGINE_USE_GPU=1` 开关 (默认 0 ⇒ 仍 PhysxCpuSystem ✓ 不变 ✗),
    #     并把"申请 GPU 却拿不到"显式报 **INVALID** (f2 三态纪律 ✓, 不静默降级 ✗)
    #   ⚠⚠ 2026-10-02 **实测结论: GPU 后端对本管线不可用 ✗** (勿再重试 ✗):
    #     · "驱动 535 太旧" 是**误判** ✗ —— 真正缺的是
    #       `~/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so`
    #       (SAPIEN 用 `requests.get` **无超时**下载 GitHub ⇒ 卡死 ✗ ⇒ 已用 curl 手动拉齐,
    #        226MB ✓); 拉齐后 `enable_gpu()` **成功**、`PhysxGpuSystem()` **构造成功** ✓
    #     · 但 **535 驱动完全够用** (无需装 580; 且 Ubuntu 20.04 源里最高只有 570 ✗) ✓
    #     · **真正的拦路虎 = 架构不匹配** ✗: 实跑 `_ENGINE_USE_GPU=1` 得到
    #       `9× setting pose does not affect PhysX GPU simulation` ✗
    #       (GPU 下**逐帧 set_pose 无效** ⇒ 我们的核心控制手段失效 ✗),
    #       `fetching articulation qpos from GPU is very slow` ✗,
    #       `loading multiple convex collision meshes from STL is unsupported` ✗,
    #       最终 `RuntimeError: GPU PhysX is not initialized` ✗
    #     ⇒ 本管线 (每帧精确 set_pose + 高频读 qpos + 多场景) 与 GPU 后端**根本冲突** ✗
    #   ⚠ 纪律 ✓: 后端切换会有**微小数值差** ⇒ 只作**对照**, 不替换现有成功配方 ✓
    #   ⇒ 提速只能从 **CPU 侧** 想办法 (并行 rollout / 步长 / 凸包数) ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _want_gpu_sc = str(os.environ.get("_ENGINE_USE_GPU", "0")).strip().lower() \
        not in ("0", "false", "no", "")
    _sys_sc = None
    if _want_gpu_sc:
        try:
            # ★ 2026-10-02 修正 ✓: SAPIEN 要求**先** `enable_gpu()` 才能构造
            #   (`PhysxGpuSystem` 直接构造 ⇒ RuntimeError:
            #    "sapien.physx.enable_gpu() must be called before creating a PhysX GPU system" ✗)
            #   `enable_gpu()` 会找 `~/.sapien/physx/<ver>/libPhysXGpu_64.so` ✓
            #   (缺失时它用 requests.get 无超时下载 GitHub ⇒ 卡死 ✗ ⇒ 已改 curl 手动拉 ✓)
            sapien.physx.enable_gpu()
            _sys_sc = sapien.physx.PhysxGpuSystem()
            logger.warning("[engine] ✓ **GPU 物理后端已启用** (PhysxGpuSystem) ⇒ "
                           "仅用于对照; 数值可能与 CPU 有微小差 ✓")
        except Exception as _egpu:  # noqa: BLE001
            logger.error("[engine] ✗ **INVALID**: 申请 GPU 后端但不可用 (%s) ⇒ "
                         "**不静默降级** ✗ 请先升级驱动(≥580 支持 CUDA13); "
                         "本次仍用 CPU ✓", _egpu)
            _sys_sc = None
    if force_cpu:
        # 优化模式: 直接创建 CPU 场景, 不尝试渲染 (避免 sandbox 环境段错误)
        scene = sapien.Scene(systems=[_sys_sc or sapien.physx.PhysxCpuSystem()])
    else:
        # 正常模式: 先尝试创建带渲染的 Scene, 失败则降级 CPU
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        except Exception:
            pass
        try:
            scene = sapien.Scene()
            render_available = True
            logger.info(f"  [setup_physics_scene] 带渲染场景创建成功, render_available={render_available}")
        except Exception as e:
            logger.warning(f"  [setup_physics_scene] 带渲染场景创建失败: {e}, 降级 CPU")
            scene = None
        if scene is None:
            scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
            logger.info(f"  [setup_physics_scene] CPU 场景创建 (降级模式)")

    # 渲染失败时自动重试一次 (GPU 资源可能需要释放时间)
    if not render_available and not force_cpu:
        import time
        time.sleep(1)
        try:
            scene = None
            scene = sapien.Scene()
            render_available = True
            logger.info(f"  [setup_physics_scene] 重试成功, render_available={render_available}")
        except Exception as e:
            logger.warning(f"  [setup_physics_scene] 重试仍失败: {e}")
            scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])

    scene.set_timestep(PHYSICS_TIMESTEP)
    # ★ 2026-09-30 ✓: 引擎**稳定性配置**开关 (默认全部保持现状 ⇒ 行为不变 ✓)
    apply_engine_solver_flags(scene, where="setup_physics_scene")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-09-23 【方案 §3.1/§3.2 实施】**设置后必须读回** + **能力探测** ✗ ★★★★
    #   方案原话 ✓: \"采用当前安装接口**显式设置并读回, 不能仅打印配置**\" ✓
    #     + §3.2: shapes 的 `contact_offset=1mm`、`rest_offset=0` ✓
    #   本机实测 ✓ (我做了接口探测): SAPIEN **3.0.0b0** 的 `scene` / `scene.physx_system`
    #     在 `offset/ solver / iter` 关键字下**没有任何方法** ✗ ⇒ 方案 §3.2 的
    #     contact/rest offset 与 §3.1 的\"位置迭代 16 / 速度迭代 1\"**在本机无法设置** ✗
    #   ⇒ 按方案铁律 ✓: \"**未知/写失败不得宣称**\" ⇒ 这里**明确 ERROR 记录不可用** ✗,
    #     而不是静默 pretend 已设 ✓ (也避免我以后再\"以为设了\"✗)
    #   ✅ 可设置且已生效的: `dt` (读回核对 ✓) + `decimation` (=16 ✓ 由 physics_utils 决定 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        _dt_rb = float(scene.get_timestep())
    except Exception:  # noqa: BLE001
        _dt_rb = float("nan")
    logger.info("[方案 §3.1] dt=%.6fs (%.0fHz) / 控制=%dHz ⇒ **decimation=%d** ✓ | 读回 dt=%.6fs %s",
                float(PHYSICS_TIMESTEP), 1.0 / float(PHYSICS_TIMESTEP), int(CONTROL_FREQ),
                int(DECIMATION), _dt_rb,
                "✓ 一致" if abs(_dt_rb - float(PHYSICS_TIMESTEP)) < 1e-9 else "✗ **不一致**")
    try:
        _cap = {}
        for _tag, _obj in (("scene", scene),
                           ("physx_system", getattr(scene, "physx_system", None))):
            if _obj is None:
                _cap[_tag] = []
                continue
            _cap[_tag] = [m for m in dir(_obj)
                          if any(k in m.lower() for k in ("offset", "solver", "iter", "tgs"))]
        if not any(_cap.values()):
            logger.error("[方案 §3.2/§3.1] ✗ **本机未暴露** contact/rest offset 与求解器迭代接口 "
                         "(探测: scene=%s, physx_system=%s) ⇒ 这两项**保持不可设置** ✗ "
                         "⇒ 按方案\"未知不得宣称已设\" ✓ (不静默 pretend ✗)",
                         _cap.get("scene", []), _cap.get("physx_system", []))
        else:
            logger.info("[方案 §3.2/§3.1] 可设置接口: %s", _cap)
    except Exception as _ecap:  # noqa: BLE001
        logger.warning("[方案 §3.2/§3.1] 能力探测失败: %s", _ecap)

    if render_available:
        try:
            from sapien.asset import create_dome_envmap
            # 对齐 002_render_scene.py 的环境光照
            scene.set_environment_map(
                create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35])
            )
            scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
            scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
            scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
            scene.set_ambient_light([0.5, 0.5, 0.5])
        except Exception:
            pass

    # 物理地面 (对齐 GalaxeaManipSim, 高度根据 GLB 物体调整)
    # R1 机器人 ROOT 在地下 (z≈-1.0), 地面 (z≈0) 会遮挡机器人身体.
    # 保留物理碰撞 (支撑物体), 隐藏地面视觉 (透明), 让整个机器人可见.
    # 地面摩擦对齐 GalaxeaManipSim 桌面 (static=1.0, dynamic=1.0, restitution=0.0)
    ground_material = sapien.physx.PhysxMaterial(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    # render=False 在 CPU 降级模式下跳过 RenderMaterial 创建 (避免 RuntimeError)
    ground_actor = scene.add_ground(
        ground_height, render=render_available, material=ground_material
    )
    if render_available:
        # gripper_only 模式不需要隐藏地面 (没有机器人躯干在地下)
        # full_robot 模式下地面视觉保持启用, 避免遮挡
        logger.info("  地面物理+视觉已加载 (render=True)")
    else:
        logger.info("  地面物理已加载 (无视觉, CPU 模式)")
    scene._render_available = render_available
    return scene


def setup_robot(scene, mode, side, base_pos, base_quat):
    """加载 R1 机器人并配置 PD 驱动

    Args:
        scene: SAPIEN 场景
        mode: "full_robot" / "gripper_only"
        side: "right" / "left" / "both"
        base_pos: (3,) 基座位置
        base_quat: (4,) 基座朝向

    Returns:
        dict: robot, joint_names, arm_joint_indices, gripper_indices, ee_links, init_qpos
              (gripper_idx1/gripper_idx2/ee_link 保留用于单侧向后兼容)
    """
    sides = ["left", "right"] if side == "both" else [side]
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-10-03 【渲染退化修复: 要录像时**不再剥掉 visual**】 ✓ ★★★★
    #   用户实测 ✗: 渲染出的 cam GIF 里 **机器人是白模(无颜色)**、其他物体**不见** ✗
    #   根因 ✓: 本行 `strip_visuals = not scene._render_available` —— 而 `setup_physics_scene`
    #     多数路径下 `render_available=False`(753 行初始值) ⇒ 机器人的 `<visual>` 被**整块删除**
    #     (仅留 collision/inertial) ⇒ 渲染出来的夹爪/机器人没有外观 ✗
    #     而**录像路径其实自带渲染能力** ✓ (实测 cam_view mp4/gif 能正常产出 ✓) ⇒ 属**过度降级** ✗
    #   本改 ✓: 新增 `_S1_KEEP_VISUAL`(默认 0 ⇒ **行为完全不变** ✓)
    #     `=1` 时**保留 visual** 供渲染 ⇒ 画面才有颜色/材质 ✓
    #   ⚠ 物理无关 ✓: `<visual>` 不参与碰撞/惯量 (仅 `<collision>`/`<inertial>` 参与 ✓)
    #     ⇒ 保留它**不改变任何物理结果** ✓; 唯一风险是 RenderMaterial 加载失败 ⇒ 由调用方 try 兜底 ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _keep_vis = str(os.environ.get("_S1_KEEP_VISUAL", "0")).strip().lower() \
        not in ("0", "false", "no", "")
    strip_visuals = (not getattr(scene, "_render_available", True)) and not _keep_vis
    if _keep_vis and not getattr(scene, "_render_available", True):
        logger.warning("[render] ✓ _S1_KEEP_VISUAL=1 ⇒ **保留 <visual>** 供渲染 "
                       "(物理不受影响 ✓; 此前被剥掉 ⇒ 画面白模/缺件 ✗)")

    if mode == "full_robot":
        urdf_path = prepare_full_robot_urdf(FULL_ROBOT_URDF, side, strip_visuals=strip_visuals)
    else:
        if side == "both":
            logger.warning("  gripper_only + both 未完全支持, 仅加载右侧夹爪")
            urdf_path = prepare_gripper_only_urdf("right", strip_visuals=strip_visuals)
        else:
            urdf_path = prepare_gripper_only_urdf(side, strip_visuals=strip_visuals)

    loader = scene.create_urdf_loader()
    # gripper_only: fix_root_link=True, 虚拟 6-DOF 关节提供移动能力 (对齐 05_gripper_test.py)
    # full_robot: fix_root_link=False, 浮动根 (free joint 6 DOF) + monkey-patch
    if mode == "gripper_only":
        loader.fix_root_link = True
    else:
        loader.fix_root_link = False
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_path)

    active_joints = robot.get_active_joints()
    joint_names = [j.name for j in active_joints]

    _joint_debug = [(j.name, j.get_type(), j.get_dof()) for j in active_joints]
    logger.info(f"  [debug] active_joints: {_joint_debug}")
    logger.info(f"  [debug] fix_root_link={loader.fix_root_link}, total qpos size={len(robot.get_qpos())}, dof={robot.get_dof()}")

    # 计算 qpos 索引映射
    qpos_starts = {}
    _qpos_offset = 0
    for _ji, _joint in enumerate(active_joints):
        qpos_starts[_ji] = _qpos_offset
        _jt = _joint.get_type()
        if _jt == 'free':
            _qpos_offset += 7  # free joint: 3 pos + 4 quat
        else:
            _qpos_offset += _joint.get_dof()  # revolute/prismatic: 1
    has_floating_root = any(j.get_type() == 'free' for j in active_joints)

    # gripper_only: 虚拟关节索引 + PD 驱动 (对齐 05_gripper_test.py)
    virtual_idx = {}
    if mode == "gripper_only":
        virtual_idx = {
            'vx': joint_names.index("virtual_x"),
            'vy': joint_names.index("virtual_y"),
            'vz': joint_names.index("virtual_z"),
            'rz': joint_names.index("virtual_rz"),
            'ry': joint_names.index("virtual_ry"),
            'rx': joint_names.index("virtual_rx"),
        }
        # v6.0 统一物理: 基座虚拟关节纯 PD (单一来源 physics_utils), 无 vlock/set_qpos.
        # 质量事实: 整机 ~0.7kg (URDF 0.01×3 + 0.604 + 0.027×2), 非历史注释声称的 36kg.
        VIRTUAL_STIFFNESS = VIRTUAL_PD_STIFFNESS
        VIRTUAL_DAMPING = VIRTUAL_PD_DAMPING
        for vkey in ['vx', 'vy', 'vz', 'rz', 'ry', 'rx']:
            active_joints[virtual_idx[vkey]].set_drive_property(
                stiffness=VIRTUAL_STIFFNESS, damping=VIRTUAL_DAMPING)
        logger.info(f"  虚拟关节 PD: stiffness={VIRTUAL_STIFFNESS}, damping={VIRTUAL_DAMPING} (纯 PD, 无 vlock)")
        if has_floating_root:
            logger.info(f"  浮动根: free joint qpos 占 7 槽, 后续 joint qpos 索引偏移")
    else:
        if has_floating_root:
            logger.info(f"  浮动根: free joint qpos 占 7 槽, 后续 joint qpos 索引偏移")
        else:
            logger.warning(f"  [警告] 浮动根未启用! fix_root_link=False 但 active_joints 中无 free joint")
            logger.warning(f"  [警告] 可能 SAPIEN URDF loader 未添加 root free joint, 需检查 URDF 结构")

    # 选择所有侧的臂关节和夹爪关节
    arm_joint_indices = []
    gripper_indices = {}  # {"left": (idx1, idx2), "right": (idx1, idx2)}
    for s in sides:
        arm_joint_indices += [i for i, n in enumerate(joint_names) if f"{s}_arm_joint" in n]
        gi1 = gi2 = None
        for i, n in enumerate(joint_names):
            if f"{s}_gripper_finger_joint1" == n:
                gi1 = i
            elif f"{s}_gripper_finger_joint2" == n:
                gi2 = i
        gripper_indices[s] = (gi1, gi2)

    # 所有夹爪关节索引集合 (用于 PD 参数设置)
    all_gripper_idxs = set()
    for gi1, gi2 in gripper_indices.values():
        if gi1 is not None:
            all_gripper_idxs.add(gi1)
        if gi2 is not None:
            all_gripper_idxs.add(gi2)

    # PD 驱动参数
    # v6.0: else 分支必须跳过虚拟关节! 历史根因: 虚拟关节在上面配好 50000/600 后,
    # 被此循环的 else (JOINT_STIFFNESS=1000/200) 静默覆盖, 有效刚度降 50 倍 →
    # 基座跟踪滞后厘米级 (Stage3 卡死/v4.52"drive 失效"误诊的真正源头).
    for i, joint in enumerate(active_joints):
        if i in all_gripper_idxs:
            joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING,
                                     force_limit=GRIPPER_FORCE)
        elif mode == "gripper_only" and virtual_idx and i in set(virtual_idx.values()):
            continue  # 虚拟关节保持 50000/600 (上方已配置)
        else:
            joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

    # 初始关节角
    init_qpos = robot.get_qpos().copy()
    if mode == "gripper_only":
        # 虚拟关节初始位置 = base_pos, 姿态 = base_quat → ZYX Euler
        init_qpos[virtual_idx['vx']] = float(base_pos[0])
        init_qpos[virtual_idx['vy']] = float(base_pos[1])
        init_qpos[virtual_idx['vz']] = float(base_pos[2])
        base_R = pr.matrix_from_quaternion(base_quat)[:3, :3]
        rz, ry, rx = rotmat_to_zyx_euler(base_R)
        init_qpos[virtual_idx['rz']] = float(rz)
        init_qpos[virtual_idx['ry']] = float(ry)
        init_qpos[virtual_idx['rx']] = float(rx)
    for s in sides:
        s_arm_indices = [i for i, n in enumerate(joint_names) if f"{s}_arm_joint" in n]
        starting = RIGHT_ARM_STARTING if s == "right" else LEFT_ARM_STARTING
        for j, idx in enumerate(s_arm_indices):
            if j < len(starting):
                init_qpos[qpos_starts[idx]] = starting[j]
        gi1, gi2 = gripper_indices[s]
        if gi1 is not None:
            init_qpos[qpos_starts[gi1]] = GRIPPER_INIT_OPEN
        if gi2 is not None:
            # full_robot URDF: 两手指 axis 相同, joint2 用负值对称张开
            # gripper_only URDF: 两手指 axis 相反 (axis="0 1 0" vs "0 -1 0"),
            #   joint1/joint2 都是正值表示张开 (对齐 04_physics_simulation.py L2320-2321)
            if mode == "gripper_only":
                init_qpos[qpos_starts[gi2]] = GRIPPER_INIT_OPEN
            else:
                init_qpos[qpos_starts[gi2]] = -GRIPPER_INIT_OPEN
    robot.set_qpos(init_qpos)

    # 关键: set_drive_target 与 set_qpos 一致, 否则 PD 拉向零位
    # 浮动根的 free joint 跳过 (由 set_root_pose + set_root_linear_velocity 驱动)
    for i, joint in enumerate(active_joints):
        if joint.get_type() == 'free':
            continue
        joint.set_drive_target(init_qpos[qpos_starts[i]])

    if mode == "full_robot":
        # full_robot: 浮动根, 通过 set_root_pose + set_root_linear_velocity 驱动
        robot.set_root_pose(sapien.Pose(np.asarray(base_pos).tolist(), np.asarray(base_quat).tolist()))
        # 禁用所有 robot link 的重力, 使 set_root_linear_velocity 生效.
        for link in robot.get_links():
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    component.set_disable_gravity(True)
        logger.info(f"  已禁用所有 robot link 重力 (set_disable_gravity=True), "
                    f"根由 set_root_pose + set_root_linear_velocity 驱动")
    else:
        # gripper_only: 也禁用重力, 防止手指 PD 反作用力 + 重力推偏虚拟关节.
        # 虚拟关节 PD 刚度 5000 N/m 足以对抗 0.6kg 夹爪重力 (0.6×9.8=5.9N, 5.9/5000=1.2mm 偏差),
        # 但 8 个物理子步的累积漂移可能很大, 禁用重力更稳定.
        # ★★★★ 2026-09-21 【用户】"我怎么感觉不像是真实物理仿真呢?" ★★★★
        #   查明 ✓: **所有 robot link 的重力被关掉** ✗ + 基座是 k=50000 的纯位置伺服
        #     ⇒ 夹爪是被"硬拖"的, 不会有真实的重力/惯性响应 ⇒ 观感"不像物理" ✓
        #     (物体侧的重力是**正常**的 ✓ —— 物体不会被飘走 ✓)
        #   本旋钮 ✓: `_ROBOT_GRAVITY=1` ⇒ **保留 robot link 重力** ✓ (更接近真实物理 ✓)
        #     代价 ⚠: 虚拟关节 PD 必须额外抵抗 0.6kg 夹爪自重 (≈5.9N);
        #            若 `_VIRTUAL_PD_K` 偏低 (如 E2 的 10000) 会产生 mm 级下垂漂移 ✗
        #   默认 `0` ⇒ 行为与历史**逐位不变** ✓ (保护已验证的 G1 ✓)
        _keep_grav = os.environ.get("_ROBOT_GRAVITY", "0") == "1"
        if _keep_grav:
            logger.info("  gripper_only: **保留** robot link 重力 (_ROBOT_GRAVITY=1 ⇒ "
                        "更接近真实物理 ✓; 虚拟关节 PD 需抵抗夹爪自重 0.6kg×9.81≈5.9N)")
        else:
            for link in robot.get_links():
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        component.set_disable_gravity(True)
            logger.info(f"  gripper_only: 已禁用所有 robot link 重力 (虚拟关节 PD 驱动, 无重力干扰)")

    # 夹爪摩擦对齐 GalaxeaManipSim R1 (static=1.0, dynamic=1.0, restitution=0.6)
    # ★★★★ 2026-09-17 restitution 改为 env 可调 (用户: "continue"; 默认 0.6 不变 ⇒ 对 box 线零影响) ★★★★
    #   背景 (实测 output/fine64): CoACD 线的**求解爆炸**已精确定位到
    #     手指 act 一到 0.0107 (约 2mm 嵌入 ✗) ⇒ 物体瞬间 objΔ[+3.5,-10.4,+8.4]mm ✗
    #     ⇒ 4 个控制点后 130mm ✗ ⇒ 被"弹射"出去 ✓
    #   机制: CoACD 的 64 个凸包**逐个**加在同一刚体上 (data_loader:483-490, 方式正确 ✓),
    #     但这些凸包在空间上**互相重叠** ✗ (CoACD 的正常输出 ✓)
    #     ⇒ 手指嵌入时同时接触**多个**重叠凸包 ⇒ 接触法向互相冲突 ✗
    #     ⇒ 求解器冲量互杠 ⇒ 深度嵌入时**爆炸** ✗
    #   而这里的 restitution=0.6 ✗ (60% 弹性 ✗) 会把冲量冲突进一步放大 ✓
    #   依据: 注释称"对齐 R1 的 static=1.0/dynamic=1.0/restitution=0.6" ✓,
    #     但代码里摩擦早已改用 `GRIPPER_FRICTION` (抓取需要, 我们设 4.0 ✗)
    #     ⇒ **"对齐 R1"的前提已不成立** ✗ ⇒ restitution 同样可按"夹持稳定性"重定 ✓
    #   物理上也更合理: 夹爪应**吸收**冲击而非反弹 (restitution→0) ✓
    #   开关: `_GRIP_RESTITUTION` (默认 0.6 ✓ ⇒ 不设时行为完全不变 ✓)
    _grip_restitution = float(os.environ.get("_GRIP_RESTITUTION", "0.6"))
    gripper_material = scene.create_physical_material(
        static_friction=GRIPPER_FRICTION, dynamic_friction=GRIPPER_FRICTION,
        restitution=_grip_restitution,
    )
    touch_link_names = []
    for s in sides:
        touch_link_names += [f"{s}_gripper_finger_link1", f"{s}_gripper_finger_link2"]
    for link in robot.get_links():
        if link.get_name() in touch_link_names:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    for cs in component.get_collision_shapes():
                        cs.set_physical_material(gripper_material)

    # 碰撞处理: gripper_only 和 full_robot 不同
    if mode == "gripper_only":
        # 对齐 05_gripper_test.py: 禁用夹爪内部碰撞 (手指-手指, 手指-夹爪本体)
        # 虚拟关节 link 碰撞也禁用 (避免虚拟臂与地面/物体碰撞)
        finger_ignore_bit = 1 << 0
        finger_ignore_id = 1
        finger_link_names = set()
        for s in sides:
            finger_link_names |= {
                f"{s}_gripper_finger_link1",
                f"{s}_gripper_finger_link2",
                f"{s}_gripper_link",
            }
        # 夹爪内部: 用碰撞组过滤 (手指和夹爪本体之间互不碰撞, 但仍与外部物体碰撞)
        for link in robot.get_links():
            if link.get_name() in finger_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            g = list(cs.get_collision_groups())
                            g[2] |= finger_ignore_bit
                            g[3] = finger_ignore_id
                            cs.set_collision_groups(g)
            else:
                # 虚拟关节 link: 完全禁用碰撞 (不与任何物体碰撞)
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_collision_groups([0, 0, 0, 0])
        logger.info(f"  gripper_only: 夹爪内部碰撞已过滤, 虚拟 link 碰撞已禁用")
    else:
        # full_robot: 禁用非夹爪 link 的碰撞 (ROOT 可能在地下, 躯干与地面碰撞会导致关节爆炸)
        collision_link_names = set()
        for s in sides:
            collision_link_names |= {
                f"{s}_gripper_finger_link1",
                f"{s}_gripper_finger_link2",
            }
        n_disabled = 0
        for link in robot.get_links():
            if link.get_name() not in collision_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_collision_groups([0, 0, 0, 0])
                        n_disabled += 1
        if n_disabled > 0:
            logger.info(f"  禁用 {n_disabled} 个非夹爪 link 的碰撞 (避免躯干-地面碰撞)")

    # EE links (每侧一个)
    ee_links = {}
    for s in sides:
        for link in robot.get_links():
            if f"{s}_gripper_link" == link.get_name():
                ee_links[s] = link
                break

    n_gripper_total = sum(1 for gi1, gi2 in gripper_indices.values() if gi1 is not None) * 2
    logger.info(f"  机器人已加载: mode={mode}, side={side}, {len(arm_joint_indices)} 臂关节 + "
                f"{n_gripper_total} 夹爪关节")
    logger.info(f"    PD: arm stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING}")
    logger.info(f"    夹爪摩擦: static={GRIPPER_FRICTION}, dynamic={GRIPPER_FRICTION}, restitution=0.6")

    # 向后兼容: 单侧时 gripper_idx1/gripper_idx2/ee_link
    first_side = sides[0]
    return {
        "robot": robot,
        "joint_names": joint_names,
        "arm_joint_indices": arm_joint_indices,
        "gripper_indices": gripper_indices,  # {"left": (i1,i2), "right": (i1,i2)}
        "gripper_idx1": gripper_indices[first_side][0],  # 向后兼容
        "gripper_idx2": gripper_indices[first_side][1],
        "ee_links": ee_links,  # {"left": link, "right": link}
        "ee_link": ee_links.get(first_side),  # 向后兼容
        "init_qpos": init_qpos,
        "qpos_starts": qpos_starts,  # active_joint list index → qpos index 映射
        "has_floating_root": has_floating_root,  # True 时根为 free joint (6 DOF)
        "virtual_idx": virtual_idx,  # gripper_only: 虚拟关节索引 {vx,vy,vz,rz,ry,rx}
    }


# ============================================================
# 4. 物理步进 (参考 04 _physics_step — 纯 PD 驱动)
# ============================================================
def _is_floating_root(robot):
    """检测 articulation 是否为浮动根 (root free joint).

    fix_root_link=False 时, 第一个 active joint 是 free joint (6 DOF).
    用于在 physics_step 中屏蔽根的广义力, 避免与 set_root_pose 冲突.
    """
    active_joints = robot.get_active_joints()
    if not active_joints:
        return False
    return active_joints[0].get_type() == 'free'


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-22 【P1-3】逐**物理子步**记录「夹爪 ↔ 桌面」接触 ✓ ★★★★
#   审计文档 §P1-3 原话 ✓: "当前虽累计了接触对, 但穿透深度读取的是**当前接触状态**,
#     失败检查主要在 `FINAL_APPROACH` 中; 早期接近、闭合、搬运时的碰撞**未由这一门完整覆盖** ✗。
#     **每控制周期只看末子步, 也可能漏掉中间子步冲击** ✗"
#     "建议 ✓: 逐物理子步记录相关 link 与桌面的**最小 separation、冲量及相位**, 再统一汇总。
#      **区分预接触、轻触、承载挡停和明显穿透**; 桌面支撑物体是正常接触, **不要误禁**" ✓
#   本实现 ✓: 在每个 `scene.step()` 之后采样 ✓ ⇒ 不漏中间子步 ✓;
#     由 `reset_substep_ground_log()` / `get_substep_ground_report()` 提供汇总 ✓
#     (相位归属 ⇒ 调用方在每个相位边界 reset 一次 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_SUBSTEP_GROUND_TOL_MM = 0.5          # 明显穿透阈值 (mm) ✓ 与 Stage2 门同口径 ✓
_SUBSTEP_GROUND_LOG = {
    "enabled": False, "substeps": 0, "n_contact_substeps": 0,
    "min_sep_mm": None, "max_impulse": 0.0, "sum_impulse": 0.0, "by_link": {},
}


def reset_substep_ground_log(enable=None, phase=None):
    """★ 2026-09-22 【复审 I.1 修复】开/关并**完整**清零逐子步地面记录 ✓

    复审原话 ✓ (I.1): "reset 仅更新部分字段, **没有清掉 `obj_ground` 与
      `_obj_this_step`**, 窗口之间遗留数据" ✗
    ⇒ 实测复现 ✓: "再次 reset 后, 尚未执行任何子步: `n_contact_substeps = 1` (旧值残留),
      `substeps = 0`, `contact_ratio = 1.0`, `supported = True`" ✗✗
    ⇒ 这是**已进入 Stage1 成功门**的错指标 (复审 I.1 原话 ✓) ⇒ 必须先把账做对 ✗

    本改 ✓ (复审 I.1-2 ✓): "每个统计窗创建**完整的新状态**, 清零目标与夹爪的计数、
      冲量、有效采样标志; 名称缓存按 scene 生命周期管理" ✓
      · 清 `obj_ground` ✓ / 清两个通道的计数 ✓ / 清冲量 ✓ / 清 `overflow` ✓
      · **保留** `enabled` 与 `obj_name` ✓ —— 它们是**配置**, 不是统计量 ✓
      · 加 `phase` 标签 ✓ (复审 I.1-4: 判"是否离桌"须用**抬升结束后的保持窗** ✗
        不能拿整个抬升期的占比 ✗) ⇒ 调用方在**保持窗开头**再 reset 一次 ✓
    """
    if enable is not None:
        _SUBSTEP_GROUND_LOG["enabled"] = bool(enable)
    _keep = {"enabled": _SUBSTEP_GROUND_LOG.get("enabled", True),
             "obj_name": _SUBSTEP_GROUND_LOG.get("obj_name", "")}
    _SUBSTEP_GROUND_LOG.clear()
    _SUBSTEP_GROUND_LOG.update(_keep)
    _SUBSTEP_GROUND_LOG.update({
        "phase": str(phase if phase is not None
                     else _SUBSTEP_GROUND_LOG.pop("phase", "") or ""),
        "substeps": 0,
        "n_contact_substeps": 0,          # 夹爪↔地面 (子步计数 ✓ ≤ substeps ✓)
        "n_obj_contact_substeps": 0,      # 目标↔地面 (子步计数 ✓ ≤ substeps ✓)
        "min_sep_mm": None, "max_impulse": 0.0, "sum_impulse": 0.0, "by_link": {},
        "obj_ground": {"n": 0, "min_sep_mm": None, "sum_impulse": 0.0,
                       "max_impulse": 0.0},
        "overflow": 0,                    # 计数越界次数 (必须恒为 0 ✓ 见 object_ground_report ✓)
    })
    return _SUBSTEP_GROUND_LOG["enabled"]


_SUBSTEP_NAME_CACHE = {}          # {Physx 组件: actor 名} ✓ (复审修正 §第 3 条 ✓)


def _refresh_substep_name_cache(scene):
    """★ 2026-09-22 【复审修正 — 桌面探针漏报 ✗】

    复审第 3 条 ✓: "本次 CPU 实验有**真实接触**, 却报告 `no_contact`,
      原因是读取了**空的组件名称**" ✗
    根因 ✓: 接触 `c.bodies[i]` 是 **Physx 组件** ✗ ⇒ **没有 `get_name()`** ✗
      (旧写法直接调 `b.get_name()` ⇒ 异常/空串 ⇒ `"gripper"/"ground"` 过滤永不命中 ✗)
    本改 ✓: 建 **组件 → actor 名** 的映射 ✓ (与 `_contact_detail_pairs` 同思路 ✓)
    """
    _SUBSTEP_NAME_CACHE.clear()
    try:
        for _a in scene.get_all_actors():
            _nm = _a.get_name()
            for _c in _a.components:
                _SUBSTEP_NAME_CACHE[_c] = _nm
    except Exception:  # noqa: BLE001
        pass


def _substep_body_name(body):
    """组件 → 名字 ✓ (缓存优先, 回退 `.name` ✓; 都取不到才返回空串 ✓)"""
    _n = _SUBSTEP_NAME_CACHE.get(body)
    if _n:
        return str(_n)
    return str(getattr(body, "name", "") or "")


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-24 【新方案实施】**逐子步、分部件**的目标接触记录 ✓ ★★★★
#   复审原话 ✓: "现有函数有一个必须处理的问题: **掌部接触也会被记成两指都接触**。
#     所以不能原样信任当前'双侧'布尔值。**接触存在、接触受力、最终抓牢, 要分别判断**" ✓
#   ⇒ 每个物理子步, 对 **finger1 / finger2 / palm** 分别累计与**目标**的接触 ✓:
#       n          = 有接触的子步数 ✓
#       n_force    |impulse|>0 的子步数 ✓ (接触存在 ≠ 受力 ✓)
#       sum_imp    累计冲量模长 ⇒ 周期平均力 = sum/(窗口子步×dt) ✓ (§5.1 ✓)
#       min_sep    最小 separation (mm ✓)
#   "真实双侧接触比例" = min(f1, f2) 的 **n_force 占比** ✓ (不是旧的挡停率 ✗)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_TARGET_CONTACT_LOG = {
    "enabled": False, "substeps": 0, "target": "",
    "finger1": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
    "finger2": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
    "palm":    {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
}


def reset_target_contact_log(enable=True, target=""):
    """窗口起点重置 ✓ (与地面窗同相位 ✓: 闭合末/抬升窗 ✓)"""
    global _TARGET_CONTACT_LOG
    _TARGET_CONTACT_LOG = {
        "enabled": bool(enable), "substeps": 0, "target": str(target),
        "finger1": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
        "finger2": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
        "palm":    {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
    }
    return _TARGET_CONTACT_LOG["enabled"]


def get_target_contact_report():
    """返回分部件目标接触报告 ✓ (JSON 可序列化 ✓)

    `bilateral_force_ratio` = min(f1, f2) 的 **n_force/substeps** ✓
      (≤1 ✓; 两指都**受力**才算 ✓ 掌部接触**不参与** ✓ —— 复审点名 ✗)
    """
    _L = _TARGET_CONTACT_LOG
    _sub = int(_L.get("substeps", 0) or 0)
    _out = {"substeps": _sub, "target": _L.get("target", ""), "enabled": _L.get("enabled")}
    for _k in ("finger1", "finger2", "palm"):
        _d = dict(_L.get(_k) or {})
        _d["force_ratio"] = (float(_d.get("n_force", 0)) / _sub) if _sub > 0 else None
        _d["avg_force_N"] = (float(_d.get("sum_imp", 0.0)) / (_sub * float(PHYSICS_TIMESTEP))
                             if _sub > 0 else None)
        _out[_k] = _d
    # ★ W1 ✓ (方案 §8.2-3): "双侧同时受力" = **每子步交集计数 / 子步数** ✓
    #   旧的 `min(f1_ratio, f2_ratio)` 只是"各自累计比的最小值" ✗ ⇒ 左右**先后**碰到也会非 0 ✗
    #   (⇒ 伪"双侧" ✗); 现改为**同一子步两指同时受力**的交集 ✓
    _both = int(_L.get("both_force_steps", 0) or 0)
    _out["both_force_steps"] = _both
    _out["bilateral_force_ratio"] = (float(_both) / _sub) if _sub > 0 else None
    # 各自比例仍保留 ⇒ **仅作诊断** ✓ (不得再当"同时"判据 ✗)
    _out["finger1_force_ratio"] = _out["finger1"]["force_ratio"]
    _out["finger2_force_ratio"] = _out["finger2"]["force_ratio"]
    _out["palm_touching"] = bool((_out["palm"].get("n") or 0) > 0)
    return _out


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ G3② (2026-09-26) 【当前控制周期窗口】给**控制器**单独返回新增量 ✗ ★★★★
#   方案原话 ✓ (§10.3-4): "Stage1 仍用整段 `both_force_steps>0` 判'当前双侧';
#     **历史一次有效接触可污染后续多帧** ⇒ 给控制器单独返回**本控制周期新增交集子步数/比例**;
#     相位统计只用于报告" ✓
#   ⇒ 实现 ✓: **消费式 delta** —— 基于**单调累计** `both_force_steps_total`,
#     由控制器在**每个控制周期读一次** ⇒ 拿到"这一周期新增了多少"✓
#     ⇒ 历史接触**不会**再让后续帧看起来"当前仍双侧" ✓
#   ⚠ 相位/报告仍用累计值 ✓ (两者分开 ⇒ 一个量不再身兼两职 ✗)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_CTRL_LAST_READ = {"both": 0, "n1": 0, "n2": 0}


def get_target_contact_report_current():
    """返回**自上次调用以来**的新增受力子步数 + 无效计数 ✓ (供控制器判定"当前" ✓)"""
    _both_tot = int(_TARGET_CONTACT_LOG.get("both_force_steps_total", 0) or 0)
    # 各指"受力子步"没有单调累计 ⇒ 用 n_force 当前窗口值做参考(仅诊断 ✓)
    _n1 = int((_TARGET_CONTACT_LOG.get("finger1") or {}).get("n_force", 0) or 0)
    _n2 = int((_TARGET_CONTACT_LOG.get("finger2") or {}).get("n_force", 0) or 0)
    _prev_both = int(_CTRL_LAST_READ.get("both", 0) or 0)
    _new_both = max(0, _both_tot - _prev_both)
    _CTRL_LAST_READ["both"] = _both_tot
    return {
        "both_force_steps_new": _new_both,          # ★ 本控制周期**新增**交集子步 ✓
        "both_force_steps_total": _both_tot,        # 累计(报告用 ✓)
        "finger1_n_force": _n1,
        "finger2_n_force": _n2,
        "sep_invalid": int(_TARGET_CONTACT_LOG.get("sep_invalid", 0) or 0),
        "normal_invalid": int(_TARGET_CONTACT_LOG.get("normal_invalid", 0) or 0),
        "probe_error": int(_TARGET_CONTACT_LOG.get("probe_error", 0) or 0),
    }


def _probe_substep_ground(scene):
    """★ 在**每个物理子步之后**采样 夹爪↔地面 的 separation/冲量 ✓ (审计 §P1-3 ✓)

    ★★ 2026-09-22 【复审 I.1 修复】旧实现有两处记账错 ✗ ★★
      ① 用**跨子步永久为 True 的字典标志** `_L["_obj_this_step"]` ✗
         ⇒ 第一次目标—地面接触后置 True, **后续子步不再计数** ✗
         ⇒ 实测: 静置盒**连续 20 子步一直被支撑**, 却得 `n=1 / ratio=0.05 / supported=False` ✗✗
      ② 夹爪通道按**接触对**累加 ✗ ⇒ 同一子步内多 link/多接触点会重复 ⇒
         计数可 **> substeps** ✗ (实测占比出现过 151% / 88794.10% ✗✗)
    本改 ✓ (复审 I.1-1 原话 ✓): "每个子步使用**局部** `target_contact_this_step` /
      `gripper_contact_this_step` 布尔值, 遍历全部接触后各自**最多加 1** ✓;
      不要用跨子步永久 True 的字典标志" ✓
      + 逐 link 计数也改为**每子步每 link 至多 1** ✓ (用局部 set ✓)
      + 子步末**统一**自检 `0 ≤ n ≤ substeps` ✓ (越界记入 `overflow` ✓)
    """
    _L = _SUBSTEP_GROUND_LOG
    _L["substeps"] = int(_L.get("substeps", 0)) + 1
    # ★ 新方案 ✓: 目标接触窗与地面窗**同相位** ✓ (窗口起点自动重置 ✓ 不需要新调用点 ✓)
    if int(_L.get("substeps", 0)) == 1:
        _tgt_name = str(_L.get("obj_name") or "")
        _TARGET_CONTACT_LOG.update({
            "enabled": True, "substeps": 1, "target": _tgt_name,
            # ★ W1 ✓ (方案 §8.2-3): "左右**同时**受力"的**每子步交集计数** ⇒ 窗口新建即清零 ✓
            "both_force_steps": 0,
            # ★ G3③ ✓ (方案 §10.3-5): **无效/缺测必须显式记数** ✗ 不得静默当 0 ✗
            #   · `sep_invalid`    = separation 取不到 (**不得**当 0 = 贴合 ✗)
            #   · `normal_invalid` = 法向取不到 (**不得**回退成冲量模长当法向力 ✗)
            #   · `probe_error`    = 探针调用失败 (`except: pass` 会伪装成"零受力" ✗)
            "sep_invalid": 0, "normal_invalid": 0, "probe_error": 0,
            "finger1": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
            "finger2": {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
            "palm":    {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None},
        })
    else:
        _TARGET_CONTACT_LOG["substeps"] = int(_L.get("substeps", 0))
    for _k in ("finger1", "finger2", "palm"):
        _d = _TARGET_CONTACT_LOG.get(_k)
        if isinstance(_d, dict):
            # ★ 每子步**两个**局部标志都要清 ✓ (修缺陷①: `n_force` 曾被重复累加 ✗)
            _d.pop("_hit_this_step", None)
            _d.pop("_force_this_step", None)
    if not _SUBSTEP_NAME_CACHE:
        _refresh_substep_name_cache(scene)
    # ★ 每个子步的**局部**标志 ✓ (绝不放回 `_L` ✗)
    _gr_hit_this_step = False
    _ob_hit_this_step = False
    # ★ W5 (2026-09-26) 修 bug ✓: 同一子步内**每指**的法向冲量必须**先累加**, 子步末再比阈值 ✗
    #   复审反例 ✓: 每指两组接触各 0.03N(**合计 0.06N > 阈值**) 却被判"零受力" ✗
    #     ⇒ 旧写法**逐接触对**比较 ⇒ 0.03 各自不达标 ⇒ 判零 ✗✗
    #   ⇒ 本改: 用**局部**累加器(每子步新建 ✓ 不跨子步污染 ✓), 子步末统一判定 ✓
    _imp_f1_this_step = 0.0
    _imp_f2_this_step = 0.0
    _gr_links_this_step = set()
    _tgt = str(_L.get("obj_name") or "")
    _og = _L.setdefault("obj_ground", {"n": 0, "min_sep_mm": None,
                                       "sum_impulse": 0.0, "max_impulse": 0.0})
    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        n0, n1 = _substep_body_name(b0), _substep_body_name(b1)
        if not n0 and not n1:
            continue
        # ★ 2026-09-22 【复审 ②】目标物体 ↔ 地面 通道 ✓ (判断"目标**不再由桌面支撑**" ✓)
        if _tgt:
            _is_obj0 = bool(_tgt in n0 or (n0 and n0 in _tgt))
            _is_obj1 = bool(_tgt in n1 or (n1 and n1 in _tgt))
            _gnd1 = "ground" in n1.lower()
            _gnd0 = "ground" in n0.lower()
            if (_is_obj0 and _gnd1) or (_is_obj1 and _gnd0):
                _ob_hit_this_step = True          # ★ 局部 ✓ (子步末才 +1 ✓)
                for _pt in c.points:
                    try:
                        _sep_o = float(_pt.separation) * 1000.0
                    except Exception:  # noqa: BLE001
                        _sep_o = 0.0
                    if _og["min_sep_mm"] is None or _sep_o < _og["min_sep_mm"]:
                        _og["min_sep_mm"] = _sep_o
                    try:
                        _imp_o = float(np.linalg.norm(_pt.impulse))
                        _og["sum_impulse"] += _imp_o
                        _og["max_impulse"] = max(float(_og.get("max_impulse", 0.0)), _imp_o)
                    except Exception:  # noqa: BLE001
                        pass
        # ★★★ 2026-09-24 【新方案】分部件记录"夹爪↔目标"接触 ✓ (掌部**单独记** ✗ 不再混入 ✓)
        if _tgt:
            _g0 = ("gripper" in n0 or "finger" in n0 or "palm" in n0)
            _g1 = ("gripper" in n1 or "finger" in n1 or "palm" in n1)
            _t0 = (_tgt in n0 or (n0 and n0 in _tgt))
            _t1 = (_tgt in n1 or (n1 and n1 in _tgt))
            if (_g0 and _t1) or (_g1 and _t0):
                _link = (n0 if _g0 else n1)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-25 【cc12 铁证 ⇒ 修数据源】必须**按侧别+部件精确过滤** ✗ ★★★★
                #   cc12 实测: 手指**还全开**(q=0.0295)时, 三种判据 (n>0 / n_force>0 /
                #     separation<1mm) **全部为真** ✗ ⇒ 共同上游=探针把"掌部 / 另一只手 /
                #     物体-桌面"的接触也算成了"手指-目标" ✗✗
                #   ⇒ 本改 ✓: 链接名必须**逐一等于** `{side}_gripper_finger_link{1,2}` ✓,
                #     否则 **完全不计** ✓ (不再塞进 palm 桶混淆 ✓)
                #     side 由 `_TC_SIDE` 指定 (默认 left ✓ 与本项目一致 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _side_tc = str(os.environ.get("_TC_SIDE", "left") or "left")
                if _link == f"{_side_tc}_gripper_finger_link1":
                    _tc_key = "finger1"
                elif _link == f"{_side_tc}_gripper_finger_link2":
                    _tc_key = "finger2"
                else:
                    _tc_key = None            # ★ 非该侧两指 ⇒ **不计** ✓ (掌部/另一只手 ✗)
                if _tc_key is None:
                    continue
                _tc = _TARGET_CONTACT_LOG.setdefault(_tc_key, {
                    "n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None})
                if not getattr(_probe_substep_ground, "_tc_done", None) == id(_tc_key):
                    pass
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-24 【用户质询 ⇒ 修正】**"双侧受力"必须可信** ✗ ★★★★
                #   用户原话 ✓: "你确定现在的双侧受力对了吗? 好好思考一下" ✓ ⇒ 复查出 4 个缺陷 ✗:
                #     ① `n_force` 未被"每子步一次"标志保护 ✗ ⇒ 多点/多对会重复累加 ⇒ 比值可 >1 ✗
                #     ② 用 `|impulse|` **模长** ✗ ⇒ 切向摩擦被算成"受力" ✗
                #        (md §5.1: **法向力 = 法向冲量 / dt** ✓ 必须投影到接触法向 ✓)
                #     ③ 链接名只按 `finger_link1/2` 匹配 ✗ ⇒ 左右两手会并进同一桶 ✗
                #     ④ 无**最小力阈值** ✗ ⇒ 数值噪声也算受力 ✗ (md §5.1: max(0.05N, 0.02mg) ✓)
                #   ⇒ 本改 ✓: 法向投影 + 每子步标志 + 侧别过滤 (`_TC_SIDE` 默认 left ✓)
                #     + 阈值 `_TC_FMIN_N` (默认 0.05N ✓) ⇒ 力 = Σ法向冲量 / (子步数·dt) ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _hit_imp_n = 0.0
                for _pt in c.points:
                    # ★ G3③ ✓ (方案 §10.3-5) **取不到就记无效, 不得替换成 0/模长** ✗
                    try:
                        _iv = np.asarray(_pt.impulse, dtype=np.float64).reshape(-1)[:3]
                        _nv = np.asarray(_pt.normal, dtype=np.float64).reshape(-1)[:3]
                        _nn = float(np.linalg.norm(_nv))
                    except Exception:  # noqa: BLE001
                        _TARGET_CONTACT_LOG["probe_error"] = int(
                            _TARGET_CONTACT_LOG.get("probe_error", 0)) + 1
                        continue
                    # ★ 法向不可用 ⇒ **跳过该接触点** ✓ (旧写法回退"冲量模长" ⇒ 把**切向摩擦**
                    #   当法向力 ✗ ⇒ 力被高估/性质错 ✗)
                    if _nn <= 1e-9:
                        _TARGET_CONTACT_LOG["normal_invalid"] = int(
                            _TARGET_CONTACT_LOG.get("normal_invalid", 0)) + 1
                        continue
                    _hit_imp_n += abs(float(np.dot(_iv, _nv / _nn)))   # **法向分量** ✓
                    # ★ separation 不可用 ⇒ 记无效并**不更新 min_sep** ✓
                    #   (旧写法换成 0 ⇒ 等于宣称"贴合" ✗ 会伪装成接触 ✓)
                    try:
                        _s_mm = float(_pt.separation) * 1000.0
                    except Exception:  # noqa: BLE001
                        _TARGET_CONTACT_LOG["sep_invalid"] = int(
                            _TARGET_CONTACT_LOG.get("sep_invalid", 0)) + 1
                        continue
                    if _tc["min_sep"] is None or _s_mm < _tc["min_sep"]:
                        _tc["min_sep"] = _s_mm
                # ★ md §5.1 ✓: **原始**法向冲量**不做阈值**地累计 ✓ (用于阈值标定 ✓)
                #   (报告里 `sum_imp_raw` / `max_imp_raw` ⇒ 看真实量级是 1e-6 还是 0.05 ✓)
                _tc["sum_imp_raw"] = float(_tc.get("sum_imp_raw", 0.0)) + float(_hit_imp_n)
                _tc["max_imp_raw"] = max(float(_tc.get("max_imp_raw", 0.0)), float(_hit_imp_n))
                if not _tc.get("_hit_this_step", False):
                    _tc["n"] = int(_tc.get("n", 0)) + 1
                    _tc["_hit_this_step"] = True
                # ★ 阈值判"受力" ✓ (每子步**至多一次** ✓ 修缺陷①②④)
                # ★ W5 ✓: **不在这里**判阈值 ✗ ⇒ 只把本接触对的法向冲量累加到该指的
                #   本子步累加器 ✓ (同一子步的多个接触对会一起累加 ✓)
                if _tc_key == "finger1":
                    _imp_f1_this_step += float(_hit_imp_n)
                elif _tc_key == "finger2":
                    _imp_f2_this_step += float(_hit_imp_n)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ W1 (2026-09-25) 【方案 §8.2-3】双侧同时率 = **每子步布尔交集计数** ✗ ★★★★
        #   方案原话 ✓: "`get_target_contact_report()` 须改为**每子步左右目标受力布尔值的
        #     交集计数**; … 左右先后碰到也可能被当成现在同时接触" ✗
        #   ⇒ 旧实现 `min(f1_ratio, f2_ratio)` 是**各自累计比的最小值** ✗:
        #     左手指在第 3 子步受力、右手在第 9 子步受力 ⇒ 两者比值都非 0 ⇒ min 仍非 0 ✗
        #     但实际**从未同一子步同时受力** ✗ (伪"双侧" ✓ 正是判据污染的来源之一 ✓)
        #   ⇒ 本改 ✓: 只有**同一子步内两指都有法向冲量超阈值**才 +1 ✓ ⇒ 真"同时" ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07a (2026-09-27) 【观测修复】阈值判定 + 双侧交集计数 **已移出本循环** ✓ ★★★★
        #   复审反例 ✓ (用户实测): "**一个子步被记成两个双侧受力子步**, 力也被高估" ✗
        #   根因 ✓ (代码结构): 这段原本写在**每个接触对**的循环体内 ⇒
        #     · 同一子步有 2 个接触对 (指1-物体 + 指2-物体) ⇒ `both_force_steps(_total)` **+2** ✗
        #     · `sum_imp += _impb` 里的 `_impb` 是**子步累加器** ⇒ 被**重复加 N 次** ⇒ 力高估 ✗
        #   ⇒ 修法 ✓: 移到**子步末**(见下方 "T07a 子步末统一判阈值") ⇒ 每子步**只判一次/只加一次** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _gl = n0 if ("gripper" in n0 or "finger" in n0) else (
            n1 if ("gripper" in n1 or "finger" in n1) else None)
        if _gl is None:
            continue
        _other = n1 if _gl == n0 else n0
        if "ground" not in _other.lower():
            continue
        _gr_hit_this_step = True                  # ★ 局部 ✓ (子步末才 +1 ✓)
        _gr_links_this_step.add(_gl)
        for pt in c.points:
            try:
                _sep_mm = float(pt.separation) * 1000.0
            except Exception:  # noqa: BLE001
                _sep_mm = 0.0
            _imp = float(np.linalg.norm(pt.impulse))
            if _L["min_sep_mm"] is None or _sep_mm < _L["min_sep_mm"]:
                _L["min_sep_mm"] = _sep_mm
            _L["max_impulse"] = max(_L["max_impulse"], _imp)
            _L["sum_impulse"] += _imp
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T07a (2026-09-27) 【子步末统一判阈值 + 双侧交集计数】**每子步只做一次** ✓ ★★★★
    #   放在接触对循环**之外** ⇒ 修掉"一个子步 +2 / 冲量被重复累加" ✗ (详见上方说明 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _fmin_n2 = float(os.environ.get("_TC_FMIN_N", "0.05") or 0.05)
    _imp_thr2 = _fmin_n2 * float(PHYSICS_TIMESTEP)              # 力 → 冲量阈值 ✓
    for _kb, _impb in (("finger1", _imp_f1_this_step),
                       ("finger2", _imp_f2_this_step)):
        _db = _TARGET_CONTACT_LOG.setdefault(
            _kb, {"n": 0, "n_force": 0, "sum_imp": 0.0, "min_sep": None})
        # 物理冲量总和**每子步只加一次** ✓ (方案 §10.3-2 ✓ 不去重物理总量, 但也不能重复加 ✗)
        _db["sum_imp"] = float(_db.get("sum_imp", 0.0)) + float(_impb)
        if (float(_impb) > _imp_thr2) and not _db.get("_force_this_step", False):
            _db["n_force"] = int(_db.get("n_force", 0)) + 1
            _db["_force_this_step"] = True
    _tf1 = _TARGET_CONTACT_LOG.get("finger1") or {}
    _tf2 = _TARGET_CONTACT_LOG.get("finger2") or {}
    if _tf1.get("_force_this_step", False) and _tf2.get("_force_this_step", False):
        _TARGET_CONTACT_LOG["both_force_steps"] = int(
            _TARGET_CONTACT_LOG.get("both_force_steps", 0)) + 1
        _TARGET_CONTACT_LOG["both_force_steps_total"] = int(
            _TARGET_CONTACT_LOG.get("both_force_steps_total", 0)) + 1
    _TARGET_CONTACT_LOG["substeps"] = int(_L.get("substeps", 0))
    # ★ 有界自检 ✓ (复审 I.1-3): 双侧子步数 **不得超过** 本窗口子步数 ⇒ 越界即记 overflow ✗
    if int(_TARGET_CONTACT_LOG.get("both_force_steps", 0)) > int(_L.get("substeps", 0)):
        _TARGET_CONTACT_LOG["overflow"] = int(
            _TARGET_CONTACT_LOG.get("overflow", 0)) + 1
    # ── 子步末: 每个通道**至多加 1** ✓ (复审 I.1-1 ✓) ──
    if _ob_hit_this_step:
        _og["n"] = int(_og.get("n", 0)) + 1
        _L["n_obj_contact_substeps"] = int(_L.get("n_obj_contact_substeps", 0)) + 1
    if _gr_hit_this_step:
        _L["n_contact_substeps"] = int(_L.get("n_contact_substeps", 0)) + 1
    for _ln in _gr_links_this_step:                # ★ 逐 link: 每子步每 link 至多 1 ✓
        _e = _L["by_link"].setdefault(_ln, {"n": 0, "min_sep_mm": None})
        _e["n"] = int(_e.get("n", 0)) + 1
    # ── 有界自检 ✓ (复审 I.1-3 ✓: 强制 `0 ≤ n ≤ substeps` ✓) ──
    _sub = int(_L.get("substeps", 0))
    if (int(_L.get("n_contact_substeps", 0)) > _sub
            or int(_L.get("n_obj_contact_substeps", 0)) > _sub
            or int(_og.get("n", 0)) > _sub):
        _L["overflow"] = int(_L.get("overflow", 0)) + 1


def set_object_ground_target(name):
    """★ 2026-09-22 【复审 ②】设置要统计的**目标物体名** ✓ (与 P1-3 同一次逐子步采样 ✓)"""
    _SUBSTEP_GROUND_LOG["obj_name"] = str(name or "")
    return _SUBSTEP_GROUND_LOG["obj_name"]


def object_ground_report():
    """★ 2026-09-22 【复审 I.1 修复】"目标是否仍由桌面支撑"的**逐子步**结论 ✓

    复审 I.1-3 原话 ✓: "强制检查 `0 ≤ n_contact_substeps ≤ substeps`;
      **无采样返回 `UNKNOWN`**, 不是'不受支撑'" ✓
    ⇒ 旧写法 `_sub = max(1, substeps)` ✗ ⇒ 0 子步时 `ratio = n/1` ✗
      (实测"再次 reset 后 `n=1, substeps=0, ratio=1.0, supported=True`" ✗✗)
    本改 ✓:
      · `substeps == 0` ⇒ `valid=False` / `supported=None` / `state="UNKNOWN"` ✓
        (**绝不默认 True 或 False** ✓)
      · `substeps > 0` ⇒ `ratio = n / substeps` ✓ (不再用 `max(1, ·)` ✗)
      · `overflow > 0` ⇒ `valid=False` ✓ (计数越界 ⇒ 数据不可信 ✓)
      · 同时暴露**两个通道** ✓: `n_contact_substeps` = **目标**↔地面 ✓ (兼容旧键 ✓) /
        `n_gripper_contact_substeps` = **夹爪**↔地面 ✓
    口径 ✓: `ratio ≥ `_OBJ_GROUND_SUPPORT_RATIO``(默认 0.5 ✓) ⇒ 仍被桌面支撑 ✓
    ⚠ 复审 I.1-4 ✓: **不要**拿整个抬升期的占比判"离桌" ✗ (前段支撑正常、末段重新落桌
      会被长窗口稀释 ✗) ⇒ 调用方须在**抬升结束后的保持窗开头** `reset_substep_ground_log()`
      一次 ✓, 用**该窗**的报告 + 承载冲量 + 最低净空**联合**判断 ✓
    """
    _L = _SUBSTEP_GROUND_LOG
    _og = _L.get("obj_ground") or {"n": 0, "min_sep_mm": None, "sum_impulse": 0.0}
    _sub = int(_L.get("substeps", 0) or 0)
    _n_obj = int(_og.get("n", 0))
    _thr = float(os.environ.get("_OBJ_GROUND_SUPPORT_RATIO", "0.5") or 0.5)
    _ovf = int(_L.get("overflow", 0) or 0)
    _base = {"n_contact_substeps": _n_obj,               # 兼容旧键 = **目标**通道 ✓
             "n_obj_contact_substeps": _n_obj,
             "n_gripper_contact_substeps": int(_L.get("n_contact_substeps", 0)),
             "substeps": _sub, "min_sep_mm": _og.get("min_sep_mm"),
             "sum_impulse": float(_og.get("sum_impulse", 0.0)),
             "max_impulse": float(_og.get("max_impulse", 0.0)),
             "threshold": _thr, "obj_name": _L.get("obj_name"),
             "phase": _L.get("phase", ""), "overflow": _ovf}
    if _sub <= 0:
        # ★ 无采样 ⇒ **UNKNOWN** ✓ (复审 I.1-3 ✓: 不是"不受支撑" ✗)
        return dict(_base, contact_ratio=0.0, valid=False, supported=None,
                    state="UNKNOWN", reason="no_substep_samples")
    _ratio = float(_n_obj) / float(_sub)
    _bounded = (_n_obj <= _sub
                and int(_L.get("n_contact_substeps", 0)) <= _sub and _ovf == 0)
    if not _bounded:
        # ★ 越界 ⇒ 数据不可信 ⇒ 同样**不判 True/False** ✓
        return dict(_base, contact_ratio=_ratio, valid=False, supported=None,
                    state="UNKNOWN", reason=f"count_overflow({_ovf})")
    return dict(_base, contact_ratio=_ratio, valid=True,
                supported=bool(_ratio >= _thr),
                state=("SUPPORTED" if _ratio >= _thr else "OFF_TABLE"),
                reason="ok")


def classify_ground_contact(report: dict, evidence: dict = None) -> str:
    """★ 2026-09-22 【P1-3】分级 ✓ (审计要求区分四态 ✓)

    ★★★ 2026-09-24 【方案 §11.2 反例 C / §11.3 修正】"持续受力" ≠ "桌面挡停" ✗ ★★★
      复审原话 ✓: \"向 `classify_ground_contact` 输入 20 个子步、每步接触、最小
        separation=0、累计冲量 = `0.06×20×dt`, 返回 **`load_bearing_block`**;
        输入**完全没有命令/实际 q、基座跟踪或接触方向**, 故分类名称中的
        **'block' 不是测出来的**\" ✓✗
      ⇒ 后果 ✓ (我这轮实测): 可行**贴桌候选**被这个标签判死 ⇒ `tier=1 ×68` ✗
        ⇒ CEM 学不动 ⇒ 我据此差一点又得出"几何无解"的错误结论 ✗
      方案要求 ✓ (§11.3): 桌面阻挡必须**联合判断** —— 桌面接触法向冲量**及方向**、
        **命令与实际 q/T 的残差**、实际速度、持续时长; 证据不足 ⇒ **`UNKNOWN`** ✓
      ⇒ 新标签: `PRE_CONTACT` / `TABLE_TOUCH` / `CONTACT_CONSTRAINED` / `UNKNOWN` ✓
      ⇒ 本函数**返回值默认仍为 legacy** ✓ (方案: "保留 `legacy_result` 与新的
        `diagnostic_result`, 记录 `criteria_revision`; **新旧分歧逐条报告后再统一
        启用新分类**" ✓) ⇒ 新分类写入 `report["classification_diag"]` ✓,
        分歧写 ERROR 日志 ✓; 用 `_GROUND_CLASS_NEW=1` 才真正启用 ✓
        ⚠ **不直接删除** `load_bearing_block` 硬门 ✓ (§11.3 明确要求 ✓)
    """
    if not report or report.get("substeps", 0) == 0:
        return "no_contact"
    _min_sep = report.get("min_sep_mm")
    _tol = float(_SUBSTEP_GROUND_TOL_MM)
    if _min_sep is None:
        return "no_contact"
    if float(_min_sep) < -_tol:
        report["classification_diag"] = {
            "legacy_result": "obvious_penetration", "new_result": "obvious_penetration",
            "criteria_revision": "2026-09-24-A3", "reason": "penetration"}
        return "obvious_penetration"
    if float(report.get("n_contact_substeps", 0)) == 0:
        report["classification_diag"] = {
            "legacy_result": "no_contact", "new_result": "no_contact",
            "criteria_revision": "2026-09-24-A3", "reason": "no_contact_substeps"}
        return "no_contact"
    _ratio = float(report["n_contact_substeps"]) / max(1.0, float(report["substeps"]))
    _dt = float(PHYSICS_TIMESTEP)
    _avg_f = float(report.get("sum_impulse", 0.0)) / max(1e-8, _ratio * report["substeps"] * _dt)
    # ── legacy 结果 (旧行为, 保持不变 ✓) ──
    if _avg_f <= 1e-6:
        _legacy = "pre_contact"
    elif _ratio >= 0.5 and _avg_f > 0.05:
        _legacy = "load_bearing_block"
    else:
        _legacy = "light_touch"

    # ── 新分类 (需要**命令与实际**证据 ✓ 复审反例 C ✓) ──
    _ev = evidence if isinstance(evidence, dict) else {}
    try:
        _resid_mm = float(_ev.get("residual_mm")) if _ev.get("residual_mm") is not None else None
    except Exception:  # noqa: BLE001
        _resid_mm = None
    try:
        _q_resid = float(_ev.get("q_residual")) if _ev.get("q_residual") is not None else None
    except Exception:  # noqa: BLE001
        _q_resid = None
    try:
        _vel = float(_ev.get("actual_vel")) if _ev.get("actual_vel") is not None else None
    except Exception:  # noqa: BLE001
        _vel = None
    _has_ev = (_resid_mm is not None) or (_q_resid is not None) or (_vel is not None)
    if _avg_f <= 1e-6:
        _new, _why = "PRE_CONTACT", "impulse≈0, 无穿透 ⇒ 尚未受力 ✓"
    elif not _has_ev:
        _new, _why = "UNKNOWN", (
            "持续受力但**缺命令/实际/方向证据** ⇒ 不能判'挡停' ✗ (复审反例 C ✓)")
    else:
        # 跟踪失败判据 ✓: 残差过大 **或** 命令在动而实际近乎静止 ✓
        _track_fail = (((_resid_mm is not None) and (_resid_mm > 2.0)) or
                       ((_q_resid is not None) and (abs(_q_resid) > 0.002)) or
                       ((_vel is not None) and (_vel < 5e-4)))
        if _track_fail and _ratio >= 0.5:
            _new, _why = "CONTACT_CONSTRAINED", (
                f"持续受力(ratio={_ratio:.2f}) **且** 跟踪失败(残差/速度证据 ✓) ⇒ 桌面约束 ✗")
        else:
            _new, _why = "TABLE_TOUCH", (
                "有接触/受力但**仍能按命令运动** ⇒ 保留候选 + 接触代价 ✓ 不判挡停 ✗")
    report["classification_diag"] = {
        "legacy_result": _legacy, "new_result": _new, "reason": _why,
        "criteria_revision": "2026-09-24-A3",
        "evidence": {"residual_mm": _resid_mm, "q_residual": _q_resid, "actual_vel": _vel},
        "avg_force_N": float(_avg_f), "contact_ratio": float(_ratio),
        "min_sep_mm": (None if _min_sep is None else float(_min_sep)),
    }
    if _legacy != _new:
        logger.error(
            "[地面分类·分歧] legacy=%s vs new=%s (%s) | avgF=%.3fN ratio=%.2f ⇒ "
            "**逐条报告**后再统一启用 (方案 §11.3 ✓; 启用: `_GROUND_CLASS_NEW=1`)",
            _legacy, _new, _why, float(_avg_f), float(_ratio))
    if os.environ.get("_GROUND_CLASS_NEW", "0") == "1":
        return _new
    return _legacy


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-24 【方案 §11.3 接线】地面分类的**命令 vs 实际**证据 ✓ ★★★★
#   复审原话 ✓: 分类器"输入**完全没有命令/实际 q、基座跟踪或接触方向**, 故 'block'
#     **不是测出来的**" ✗ ⇒ 必须把证据**接进去**才能区分
#     `TABLE_TOUCH`(轻触但仍能按命令运动 ✓) 与 `CONTACT_CONSTRAINED`(真被桌面约束 ✗)
#   证据字段 ✓ (全部可选; 缺任一项 ⇒ 分类器判 `UNKNOWN` ⇒ **不可交付** ✓ 方案 §11.3 ✓):
#     · `q_residual`: |命令变化量 − 实际变化量| (关节空间跟踪残差 ✓)
#     · `actual_vel`: 本帧**实际**关节位移 (≈速度 ✓) ⇒ 命令在动而它≈0 ⇒ 被约束 ✗
#     · `residual_mm`: 基座命令位姿 vs 实际位姿偏差 (mm ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_GROUND_EVIDENCE = {}


def set_substep_ground_evidence(evidence: dict = None):
    """逐帧写入地面分类证据 ✓ (空 ⇒ 分类器只能返回 UNKNOWN ✓ 不冒充结论 ✗)"""
    global _GROUND_EVIDENCE
    _GROUND_EVIDENCE = dict(evidence) if isinstance(evidence, dict) else {}
    return _GROUND_EVIDENCE


def get_substep_ground_report():
    """取汇总 ✓ (+ 分级 ✓); 返回 dict (可 JSON 序列化 ✓)

    ★ 2026-09-24 ✓: 分级时带上**命令vs实际证据** (`_GROUND_EVIDENCE` ✓) ⇒
      分类名里的 "block" 才是**测出来的** ✓ (复审反例 C ✓)
    """
    _L = _SUBSTEP_GROUND_LOG
    out = {k: (None if v is None else v) for k, v in _L.items() if k != "by_link"}
    out["by_link"] = {k: dict(v) for k, v in _L["by_link"].items()}
    out["classification"] = classify_ground_contact(_L, _GROUND_EVIDENCE)
    out["ground_evidence"] = dict(_GROUND_EVIDENCE or {})
    out["enabled"] = bool(_L["enabled"])
    return out


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T072 (2026-09-27) 【下发 spy】**驱动层独立记录**收到的命令 ✓ ★★★★
#   为什么需要 ✓ (md T07.2): "强制触发出口限速, trace 必须等于**下发 spy 记录**,
#     **而不是 clip 前值**" ✗ ⇒ 只在驱动入口记一份, 才能证伪"记的是限速前的值" ✓
#   ⚠ 只读钩子 ✓: 不改变任何控制/物理行为 ✓
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_DISPATCH_SPY = {"enabled": False, "q": []}


def set_dispatch_spy(enable=True):
    """开启/清空 下发 spy ✓ (返回是否启用 ✓)"""
    _DISPATCH_SPY["enabled"] = bool(enable)
    _DISPATCH_SPY["q"] = []
    return _DISPATCH_SPY["enabled"]


def get_dispatch_spy():
    """取**驱动层实际收到**的 q 命令序列 ✓ (副本 ✓)"""
    return list(_DISPATCH_SPY["q"])


def physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                 arm_target, gripper_target1, gripper_target2, scene,
                 extra_gripper_indices=None, extra_arm_indices=None, extra_arm_target=None,
                 lock_root_pose=None, decimation=None,
                 virtual_lock_targets=None,
                 # ★ md 06.4-f2 ✓ 每子步观测回调: 调用方传入无参函数 ⇒ 本函数在每个
                 #   子步调用一次 ⇒ 调用方自行累计冲量 (f2: 控制周期力 = 该周期子步
                 #   法向冲量和 ÷ 真实周期时长 ⇒ **必须逐子步**, 只取末子步不够 ✗)
                 substep_hook=None):
    """纯 PD 驱动 + 重力补偿 + decimation (对齐 GalaxeaManipSim / 05_gripper_test.py)

    关键: 不调用 set_qpos! set_qpos + set_drive_target 双重控制会导致震荡.
    extra_gripper_indices: 双手模式第二侧夹爪 [(idx, target), ...]
    extra_arm_indices/extra_arm_target: 双手模式第二侧臂关节索引和目标
    lock_root_pose: 向后兼容参数 (gripper_only 虚拟关节模式不再使用, 传 None).
        full_robot 浮动根模式仍使用此参数进行根锁定.
    decimation: 物理子步数, None 时使用全局 DECIMATION (=8).
    virtual_lock_targets: dict {joint_idx: target_value} — 虚拟关节运动学锁定.
        每个 scene.step() 后强制 set_qpos + set_qvel=0, 防止 PD 漂移.
        gripper_only 模式下 virtual_z 等 3cm 漂移的根因修复.
    """
    if decimation is None:
        decimation = DECIMATION
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T08-B (2026-09-27) 【step 前: 同入口记录 + 场景合同阻断】**唯一 step 入口** ✓ ★★★★
    #   md ✓: "每周期只下发一次、步进一次"; "故意修改一个物理/目标字段必须**在 step 前阻断**" ✓
    #   ⇒ ① 记录本次入口 (tag/decimation/dt) ⇒ 供 S1/S2 "同入口同签名" 证明 ✓
    #     ② 场景通道合同不符 ⇒ **抛异常、不步进** ✗ (不带病推进 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        record_step_entry(get_step_tag(), decimation, PHYSICS_TIMESTEP)
        # ★ T08-B ✓ 运行时证据: 每 500 次 step 打印一次"同入口/同签名"判定 ✓
        #   (跨 Stage1→Stage2 的运行会自然给出该证据 ✓, 无需另跑脚本 ✓)
        if len(_STEP_ENTRIES) % 500 == 0:
            _ok_sr, _det_sr = verify_same_runner()
            logger.info("[T08-B] step 入口判定 (n=%d): %s | %s",
                        len(_STEP_ENTRIES), "同入口同签名 ✓" if _ok_sr else "尚未双 stage",
                        _det_sr)
    except Exception:  # noqa: BLE001
        pass
    assert_scene_contract()          # ★ 不符即 raise ⇒ **step 前阻断** ✓
    # ★ T072 ✓ 下发 spy: 记**本次物理步实际收到**的夹爪命令 (就在驱动入口 ✓)
    if _DISPATCH_SPY.get("enabled"):
        try:
            _DISPATCH_SPY["q"].append(float(gripper_target1))
        except Exception:  # noqa: BLE001
            pass
    active_joints = robot.get_active_joints()
    for i, idx in enumerate(arm_joint_indices):
        if i < len(arm_target):
            active_joints[idx].set_drive_target(float(arm_target[i]))
    if gripper_idx1 is not None:
        active_joints[gripper_idx1].set_drive_target(float(gripper_target1))
    if gripper_idx2 is not None:
        active_joints[gripper_idx2].set_drive_target(float(gripper_target2))
    if extra_arm_indices is not None and extra_arm_target is not None:
        for i, idx in enumerate(extra_arm_indices):
            if i < len(extra_arm_target):
                active_joints[idx].set_drive_target(float(extra_arm_target[i]))
    if extra_gripper_indices is not None:
        for idx, target in extra_gripper_indices:
            if idx is not None:
                active_joints[idx].set_drive_target(float(target))

    for _ in range(decimation):
        # ★ f2 ✓: 逐子步观测 (读的是**上一子步末**状态 ⇒ 与周期末差一个子步, 已在
        #   Stage2 落盘字段显式注明 ✓); 调用次数 = decimation ✓
        if substep_hook is not None:
            try:
                substep_hook()
            except Exception:  # noqa: BLE001
                pass
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        # 浮动根: 屏蔽根 free joint 的广义力.
        # 原因: compute_passive_force 会为根 free joint 计算重力补偿,
        # 与 set_root_pose (teleport) 冲突, 导致根漂移.
        if _is_floating_root(robot):
            qf[:6] = 0

        # 弹簧力 + 阻尼力 (仅 full_robot 浮动根模式使用).
        # gripper_only 虚拟关节模式: lock_root_pose=None, 跳过此块.
        if lock_root_pose is not None:
            target_pos = np.array(lock_root_pose.p)
            current_pos = np.array(robot.get_root_pose().p)
            _spring_k = 200000.0   # 弹簧刚度 N/m, 足够大以抵抗手指 PD 反作用力 (≤20N)
            _damping_k = 5000.0    # 阻尼系数 N·s/m, 抑制振荡
            pos_err = current_pos - target_pos
            qvel = robot.get_qvel()
            spring_force = -_spring_k * pos_err
            damping_force = -_damping_k * qvel[:3]
            # 旋转弹簧: 用四元数乘积计算 angle-axis 误差
            target_quat = np.array(lock_root_pose.q)
            current_quat = np.array(robot.get_root_pose().q)
            _spring_k_rot = 100000.0
            # Quaternion product: q1 * q2 = (w1*w2-v1·v2, w1*v2+w2*v1+v1×v2)
            w1, v1 = target_quat[0], target_quat[1:4]
            w2, v2 = current_quat[0], current_quat[1:4]
            quat_err_v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
            angle_axis = 2.0 * quat_err_v
            spring_torque = -_spring_k_rot * angle_axis
            damping_torque = -_damping_k * qvel[3:6]

            # 对浮动根: 写入 qf (会覆盖 compute_passive_force 的重力补偿)
            if _is_floating_root(robot):
                qf[0:3] = spring_force + damping_force
                qf[3:6] = spring_torque + damping_torque

        robot.set_qf(qf)

        # PRE-step teleport: 仅 full_robot 浮动根模式.
        # gripper_only 虚拟关节: lock_root_pose=None, 跳过.
        if lock_root_pose is not None:
            robot.set_root_pose(lock_root_pose)
            robot.set_root_linear_velocity([0.0, 0.0, 0.0])
            robot.set_root_angular_velocity([0.0, 0.0, 0.0])

        scene.step()

        # ★ 2026-09-22 【P1-3】**每个子步之后**采样 夹爪×桌面 ✓ (不漏中间子步冲击 ✓)
        if _SUBSTEP_GROUND_LOG.get("enabled"):
            try:
                # ★ G3③ ✓ (方案 §10.3-5): 探针失败**不得静默 pass** ✗
                #   (旧写法 `except: pass` ⇒ 采样失败会被**伪装成"零受力"** ✗)
                try:
                    _probe_substep_ground(scene)
                except Exception:  # noqa: BLE001
                    _TARGET_CONTACT_LOG["probe_error"] = int(
                        _TARGET_CONTACT_LOG.get("probe_error", 0)) + 1
            except Exception:  # noqa: BLE001
                pass

        # POST-step teleport: 仅 full_robot 浮动根模式.
        if lock_root_pose is not None:
            robot.set_root_pose(lock_root_pose)
            robot.set_root_linear_velocity([0.0, 0.0, 0.0])
            robot.set_root_angular_velocity([0.0, 0.0, 0.0])
            _actual_pos = robot.get_root_pose().p
            _target_pos = lock_root_pose.p
            _drift = np.linalg.norm(np.array(_actual_pos) - np.array(_target_pos))
            if _drift > 0.001:
                logger.warning(f"  [WARNING] teleport 后仍漂移! drift={_drift:.6f}m, "
                               f"target={np.array(_target_pos).round(4)}, "
                               f"actual={np.array(_actual_pos).round(4)}")


# ============================================================
# 5. 接触检测 (参考 04 _fetch_contacts)
# ============================================================
def fetch_contacts(robot, obj_actors, side, scene):
    """检测夹爪与物体之间的接触

    Args:
        side: "left" / "right" / "both"

    Returns:
        tuple: (total_contacts, total_impulse, per_obj)
    """
    # side="both" 时检查两侧夹爪
    sides = ["left", "right"] if side == "both" else [side]
    # 所有含 side + gripper 的 link 都计入 (手指、掌心等)
    gripper_bodies = set()
    for link in robot.get_links():
        name = link.get_name()
        if any(s in name for s in sides) and "gripper" in name:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(component)

    obj_bodies = {}
    for actor in obj_actors:
        for component in actor.components:
            if isinstance(component, (
                sapien.pysapien.physx.PhysxRigidDynamicComponent,
                sapien.pysapien.physx.PhysxRigidStaticComponent,
            )):
                obj_bodies[component] = actor.name
                break

    contacts = scene.get_contacts()
    total_contacts = 0
    total_impulse = 0.0
    per_obj = {}
    for c in contacts:
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if not is_gripper_obj:
            continue
        total_contacts += 1
        impulse = np.array([p.impulse for p in c.points])
        imp_norm = 0.0
        if len(impulse) > 0:
            imp_norm = float(np.linalg.norm(np.sum(impulse, axis=0)))
        total_impulse += imp_norm
        obj_body = b1 if b0 in gripper_bodies else b0
        obj_name = obj_bodies.get(obj_body, "unknown")
        if obj_name not in per_obj:
            per_obj[obj_name] = {"n": 0, "impulse": 0.0}
        per_obj[obj_name]["n"] += 1
        per_obj[obj_name]["impulse"] += imp_norm

    return total_contacts, total_impulse, per_obj


# ============================================================
# 7b. 辅助函数: 接触检测 + 力获取 + 物体在夹爪内判断
# ============================================================
def get_finger_contacts(robot, side, scene, obj_actors):
    """获取夹爪两手指与 GLB 物体的接触信息

    对齐 04_physics_simulation.py _fetch_contacts (L1941-1995) 的筛选方式:
    用 PhysxArticulationLinkComponent / PhysxRigidDynamicComponent 做 body 匹配

    Args:
        robot: SAPIEN robot
        side: "left" / "right"
        scene: SAPIEN scene
        obj_actors: GLB 物体 actor (Entity) 列表

    Returns:
        (finger1_contact: bool, finger2_contact: bool, contact_obj_names: list[str])
    """
    finger_names = {
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
        f"{side}_gripper_link",
    }
    # 收集夹爪 link 的 PhysxArticulationLinkComponent (04 L1958-1963)
    gripper_bodies = set()
    finger_body_map = {}  # component → link_name
    for link in robot.get_links():
        if link.get_name() in finger_names:
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(comp)
                    finger_body_map[comp] = link.get_name()

    # 收集 GLB 物体的 PhysxRigidDynamicComponent (04 L1965-1970)
    obj_bodies = {}
    for actor in obj_actors:
        for comp in actor.components:
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                obj_bodies[comp] = actor.name
                break

    f1_contact = False
    f2_contact = False
    contact_objs = set()

    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if not is_gripper_obj:
            continue
        # 确定哪个是夹爪 component
        gripper_comp = b0 if b0 in gripper_bodies else b1
        obj_comp = b1 if b0 in gripper_bodies else b0
        link_name = finger_body_map.get(gripper_comp, "")
        if "finger_link1" in link_name:
            f1_contact = True
        elif "finger_link2" in link_name:
            f2_contact = True
        elif "gripper_link" in link_name:
            f1_contact = True
            f2_contact = True
        obj_name = obj_bodies.get(obj_comp, "")
        if obj_name:
            contact_objs.add(obj_name)

    return f1_contact, f2_contact, list(contact_objs)


def build_runtime_physics_fingerprint(scene, actors=None, target_name=None):
    """★ 2026-09-22 【审计 §4 · 物理指纹】读取**引擎实值** ✓ (不是源码常量 ✗)

    审计文档 §4 原话 ✓: "`build_physics_config()` 当前只记录控制周期、手指 PD/力限/摩擦
      等少量字段。建议补全**实际碰撞几何 hash**、各材料、质量/惯量/质心、阻尼、
      虚拟基座控制、碰撞过滤、接触偏移与求解器参数 ✓。**声明请求 `coacd` 不等于实际
      加载成功**, 应记录**实际构建结果与回退原因**" ✓
    本函数 ✓: 直接读 actor/scene 的实值 ✓ (含 CoACD **实际片数/顶点数/几何 hash** ✓)
      ⇒ 与 `scene_digest` 配合, 才能证明"Stage1/2 用的是同一份几何" ✓
    返回 ✓: dict (全部可 JSON 序列化 ✓)
    """
    out = {"gravity": None, "target": {}, "actors": {}, "scene_names": []}
    try:
        out["gravity"] = [float(v) for v in scene.get_gravity()[:3]]
    except Exception:  # noqa: BLE001
        pass
    _tpl = str(target_name or "")
    for actor in (actors or []):
        try:
            _nm = actor.get_name()
            _info = {}
            for comp in actor.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    _info["is_dynamic"] = True
                    try:
                        _info["mass"] = float(comp.get_mass())
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _info["inertia"] = [float(v) for v in comp.get_inertia()[:3]]
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _info["cmass_local"] = [float(v) for v in comp.cmass_local_pose.p[:3]]
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _info["linear_damping"] = float(comp.linear_damping)
                        _info["angular_damping"] = float(comp.angular_damping)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _info["linear_velocity"] = [float(v) for v in comp.get_linear_velocity()[:3]]
                        _info["angular_velocity"] = [float(v) for v in comp.get_angular_velocity()[:3]]
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _info["T_W_O"] = [float(v) for v in actor.get_pose().p[:3]] + \
                                         [float(v) for v in actor.get_pose().q[:4]]
                    except Exception:  # noqa: BLE001
                        pass
                    break
            # ★ CoACD 实际构建结果 ✓ (片数/顶点数/几何 hash ✓ ⇒ "声明≠加载成功" ✓)
            try:
                _shapes, _nv, _hash = [], 0, []
                for _sh in comp.get_collision_shapes():
                    _V = np.asarray(_sh.get_vertices(), dtype=np.float64)
                    _shapes.append(int(len(_V)))
                    _nv += int(len(_V))
                    _hash.append(float(np.round(np.abs(_V).sum(), 9)))
                _info["collision_shapes_vertices"] = _shapes
                _info["collision_vertex_count"] = int(_nv)
                _info["collision_vertex_sum_hash"] = float(np.round(sum(_hash), 9))
            except Exception:  # noqa: BLE001
                pass
            try:
                m = actor.find_component_by_type(
                    sapien.pysapien.physx.PhysxMaterialComponent)
                if m is not None:
                    _info["friction"] = float(m.get_static_friction())
                    _info["friction_dynamic"] = float(m.get_dynamic_friction())
                    _info["restitution"] = float(m.get_restitution())
            except Exception:  # noqa: BLE001
                pass
            if _tpl and (_tpl in _nm or _nm in _tpl):
                out["target"] = dict(_info, name=_nm)
            else:
                out["actors"][_nm] = _info
        except Exception:  # noqa: BLE001
            continue
    try:
        out["scene_names"] = [a.get_name() for a in scene.get_all_actors()]
    except Exception:  # noqa: BLE001
        pass
    return out


def get_grasp_force(side, scene, obj_actors, robot):
    """获取夹爪当前夹紧力 (N)

    对齐 04 的 body 匹配方式, 通过 impulse/dt 计算

    Args:
        side: "left" / "right"
        scene: SAPIEN scene
        obj_actors: GLB 物体 actor 列表
        robot: SAPIEN robot

    Returns:
        float: 两手指总夹紧力 (N)
    """
    finger_names = {
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
        f"{side}_gripper_link",
    }
    gripper_bodies = set()
    for link in robot.get_links():
        if link.get_name() in finger_names:
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(comp)

    obj_bodies = set()
    for actor in obj_actors:
        for comp in actor.components:
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                obj_bodies.add(comp)
                break

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-09-22 【P0-2 修复】接触力的**时间分母**错了 ⇒ 默认低估 **8 倍** ✗ ★★★★
    #   审计文档 §P0-2 实测 ✓ (SAPIEN 3.0.0b0, CPU 刚体静置诊断 ✓):
    #     · 4cm 边长、≈0.064kg 盒子静置在支撑面: 重力 ≈ **0.62784N** ✓
    #     · 当前步接触冲量 ÷ **物理步长 (1/240)** ⇒ **0.62791N** ✓✓ (与 mg 一致 ✓)
    #     · 除以 `PHYSICS_TIMESTEP * DECIMATION` (8 倍) ⇒ **0.07849N** ✗ (低估 8 倍 ✓)
    #   原因 ✓: `physics_step()` 执行 8 个**子步**后, `get_contacts()` 读的是**当前步**
    #     的接触冲量 ✗ —— 它**不是 8 个子步冲量的累计** ✗
    #     (与 SAPIEN 官方接触教程 "当前仿真步接触、冲量 ÷ timestep" 一致 ✓)
    #   影响 ✓: 不只是打印 ✗ —— Stage1 的 `5 < peak_force < 200` 力奖励会因此
    #     **该触发时不触发** ✗ (见 `stage1_grasp_mixin.py` 力 sanity 项 ✓)
    #   ⚠ 若要"控制周期平均力" ✓: 必须**逐子步累计冲量**后再 ÷ 整个控制周期 ✓
    #     (本函数只读当前步 ⇒ 不能这么算 ✗; 需要改 `physics_step` 逐子步累加 ✓)
    dt = PHYSICS_TIMESTEP
    total_force = 0.0
    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if is_gripper_obj:
            for pt in c.points:
                impulse_norm = float(np.linalg.norm(pt.impulse))
                total_force += impulse_norm / max(dt, 1e-8)
    return total_force


def get_grasp_force_detail(side, scene, obj_actors, robot, target_name=None):
    """★ 2026-09-22 【P0-2 配套】**分项**接触力 (N ✓, 时间分母已修正 ✓)

    审计文档 §P0-2 的语义问题 ✓: 旧 `get_grasp_force()` 把 **掌部** + **所有动态物体**
    的冲量**混在一起累加** ✗ ⇒ 不能当作"目标物体左右两指的有效法向夹持力" ✗
    本函数 ✓: 分开输出 ✓ (文档明确要求 ✓):
      · `f_target_f1` / `f_target_f2`: 指 1 / 指 2 **对目标物** ✓ (真正的夹持力 ✓)
      · `f_palm`    : 掌部 (side_gripper_link) 与任何东西 ✓
      · `f_ground`  : 夹爪 (含两指/掌) 与 **地面/桌面** ✓ (支撑/挡停 ✓)
      · `f_other`   : 夹爪与其他动态物体 ✓
    单位 ✓: 冲量 ÷ **当前物理步长** (与 `get_grasp_force` 同口径 ✓)
    """
    finger_names = {
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
        f"{side}_gripper_link",
    }
    _link_comp = {}
    for link in robot.get_links():
        _ln = link.get_name()
        if _ln in finger_names:
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    _link_comp[comp] = _ln
    _obj_comp = {}
    for actor in obj_actors:
        for comp in actor.components:
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                _obj_comp[comp] = actor.get_name()
                break
    _tgt = str(target_name or "")
    dt = PHYSICS_TIMESTEP
    # ★ md 06.4-f2 (2026-09-30) ✓: 新增**法向/切向分项** ——
    #   R13 教训 ✓: "**总冲量模长不等于两指目标法向力**" ✗ ⇒ 只有法向分量才能判定承载,
    #     切向分量代表**滑移趋势** ⇒ 两者必须分开报 (f2 ✓)
    out = {"f_target_f1": 0.0, "f_target_f2": 0.0, "f_palm": 0.0,
           "f_ground": 0.0, "f_other": 0.0, "f_total": 0.0,
           "n_target_f1": 0.0, "n_target_f2": 0.0,
           "t_target_f1": 0.0, "t_target_f2": 0.0,
           "normal_probe_state": "OK"}
    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        _n0, _n1 = _link_comp.get(b0), _link_comp.get(b1)
        if _n0 is None and _n1 is None:
            continue                                   # 与夹爪无关 ✓
        _glink = _n0 if _n0 is not None else _n1
        _other = b1 if _n0 is not None else b0
        _oname = _obj_comp.get(_other)
        _f = 0.0
        _fn = 0.0                       # 法向分项 (N ✓)
        _ft = 0.0                       # 切向分项 (N ✓)
        for pt in c.points:
            _imp = np.asarray(pt.impulse, dtype=np.float64).reshape(-1)
            _f += float(np.linalg.norm(_imp)) / max(dt, 1e-8)
            try:
                _nn = np.asarray(pt.normal, dtype=np.float64).reshape(-1)
                if _nn.size >= 3 and float(np.linalg.norm(_nn)) > 1e-12:
                    _nn = _nn / float(np.linalg.norm(_nn))
                    _d = float(np.dot(_imp, _nn))
                    _fn += abs(_d) / max(dt, 1e-8)
                    _ft += float(np.linalg.norm(_imp - _d * _nn)) / max(dt, 1e-8)
            except Exception:  # noqa: BLE001
                # ⚠ 缺测**显式标注**, 不填 0 ✗ (f2: 缺有效载荷先修观测, 不改摩擦 ✓)
                out["normal_probe_state"] = "MISSING(contact.normal 不可读 ⇒ 法向/切向不可判定)"
        out["f_total"] += _f
        _is_tgt = bool(_tgt and _oname and (_tgt in _oname or _oname in _tgt))
        if _is_tgt and _glink.endswith("finger_link1"):
            out["f_target_f1"] += _f
            out["n_target_f1"] += _fn
            out["t_target_f1"] += _ft
        elif _is_tgt and _glink.endswith("finger_link2"):
            out["f_target_f2"] += _f
            out["n_target_f2"] += _fn
            out["t_target_f2"] += _ft
        elif _glink.endswith("_gripper_link"):
            out["f_palm"] += _f
        elif _oname is None:
            # 非 obj_actors (地面/桌面/静态) ⇒ 归入 ground ✓
            out["f_ground"] += _f
        else:
            out["f_other"] += _f
    return out


def is_obj_in_gripper_frame(gripper_pos, gripper_R, obj_pos,
                            forward_threshold=0.08, lateral_threshold=0.04):
    """判断物体是否在夹爪前方 (即将进入两指之间)

    Args:
        gripper_pos: 夹爪世界位置 (3,)
        gripper_R: 夹爪旋转矩阵 (3,3)
        obj_pos: 物体世界位置 (3,)
        forward_threshold: 前方距离阈值 (m)
        lateral_threshold: 横向距离阈值 (m)

    Returns:
        bool: 物体是否在夹爪即将抓取的位置
    """
    obj_in_gripper = gripper_R.T @ (obj_pos - gripper_pos)
    # 夹爪 URDF: +X 前方, ±Y 两指方向
    return (0 < obj_in_gripper[0] < forward_threshold and
            abs(obj_in_gripper[1]) < lateral_threshold)


__all__ = [
    # 路径常量
    "PROJECT_ROOT", "GALAXEA_SIM_PATH", "R1_ASSETS", "R1_MESH_DIR", "R1_URDF_DIR",
    "FULL_ROBOT_URDF", "R1_RIGHT_SETTINGS", "R1_LEFT_SETTINGS", "COMBINATION_DIR",
    "compute_and_save_transform_params",
    # 坐标变换
    "R_x", "R_AXIS", "RXWORLD_TO_SAPIEN",
    # 机器人参数
    "RIGHT_ARM_STARTING", "LEFT_ARM_STARTING", "ARM_MAX_REACH", "COMFORTABLE_REACH",
    "BASE_BACK_OFFSET", "ARM_BASE_OFFSET_RIGHT", "ARM_BASE_OFFSET_LEFT",
    "COMFORT_TARGET_IN_BASE", "RIGHT_ARM_JOINT_LIMITS",
    # 物理参数
    "JOINT_STIFFNESS", "JOINT_DAMPING", "GRIPPER_STIFFNESS", "GRIPPER_DAMPING",
    "PHYSICS_TIMESTEP", "CONTROL_FREQ", "DECIMATION", "GROUND_HEIGHT",
    "OBJECT_DENSITY", "OBJECT_MIN_MASS", "GRIPPER_FRICTION",
    "GRIPPER_INIT_OPEN", "GRIPPER_MAX_OPEN", "GRIPPER_CLOSE_BIAS", "GRIPPER_FORCE",
    "_DESCEND_OPEN", "_GRIP_CLOSE", "_GRIP_OVERCLOSURE",
    "GRASP_STRATEGIES", "FINGER_FORWARD_NEUTRAL", "MAX_ROOT_STEP",
    "_FINGER1_ORIGIN", "_FINGER1_AXIS", "_FINGER2_ORIGIN", "_FINGER2_AXIS",
    "FINGER_BASE_DIST", "GRIPPER_FINGER_HALF_THICKNESS", "FINGER_EFFECTIVE_HALF_SPACING",
    # v6.9: 朝向对齐
    "GRAVITY_DIR", "align_gripper_approach_to_gravity", "flip_mano_hand_down",
    # 内部状态
    "_FAILED_SCENES",
    # 函数
    "rotmat_to_zyx_euler",
    "prepare_full_robot_urdf", "prepare_gripper_only_urdf",
    "setup_physics_scene", "setup_robot",
    "physics_step", "fetch_contacts",
    "get_finger_contacts", "get_grasp_force", "is_obj_in_gripper_frame",
    "_is_floating_root",
    # logger
    "logger",
]


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T08-B (2026-09-27) 【同一 runner/tick + 场景通道合同】 ✓ ★★★★
#   md 原话 ✓: "Stage1/2 调用 T03 同一 `tick`、T01 同一物理 step adapter;
#     **每周期只下发一次、步进一次**, 执行器内部不能再 step。
#     逐 body/shape 核对 目标—双指 / 目标—桌面 / 双指—桌面 通道, 其他物体隔离。
#     只比摘要而不验证实际场景**不算完成**; **不能用全局开关把目标—桌面也关掉**。
#     验收: 调用 spy 证明 S1/S2 同入口; 故意修改一个物理/目标字段必须**在 step 前阻断**" ✓
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_STEP_ENTRIES = []              # 每次 physics_step 的入口记录 (含调用方 tag ✓)
_SCENE_CONTRACT = {"registered": False, "expected": None, "tag": None}


def record_step_entry(tag, decimation, dt, extra=None):
    """★ 每周期**只应调用一次** step ⇒ 记录入口 (tag/decim/dt ✓) 供同入口证明 ✓"""
    try:
        _STEP_ENTRIES.append({"tag": str(tag), "decimation": int(decimation),
                              "dt": float(dt), "extra": dict(extra or {})})
    except Exception:  # noqa: BLE001
        pass


def reset_step_entries():
    _STEP_ENTRIES[:] = []
    return 0


def get_step_entries():
    return [dict(e) for e in _STEP_ENTRIES]


def verify_same_runner(entries=None, tags=("stage1", "stage2")):
    """★ 返回 ``(ok, detail)``: S1/S2 必须**同一入口同一签名** (decim/dt 一致 ✓) ✓"""
    _es = list(entries if entries is not None else _STEP_ENTRIES)
    if not _es:
        return False, "无 step 入口记录 (spy 未开 ✗)"
    _seen = {e.get("tag") for e in _es}
    _missing = [t for t in tags if t not in _seen]
    if _missing:
        return False, "缺少调用方 tag: %s (实测 %s ✗)" % (_missing, sorted(_seen))
    _sig = {(e.get("decimation"), round(float(e.get("dt", 0.0)), 9))
            for e in _es if e.get("tag") in tags}
    if len(_sig) != 1:
        return False, "签名不一致 (decimation/dt 分叉 ✗): %s" % sorted(_sig)
    return True, "同入口同签名 ✓ (%d 次调用, sig=%s)" % (len(_es), sorted(_sig)[0])


def register_scene_contract(expected, tag=""):
    """登记**期望的碰撞通道矩阵** ⇒ 之后每个 step 前都会核对 ✓"""
    _SCENE_CONTRACT["expected"] = dict(expected or {})
    _SCENE_CONTRACT["tag"] = str(tag)
    _SCENE_CONTRACT["registered"] = True
    return dict(_SCENE_CONTRACT["expected"])


def get_scene_contract():
    return dict(_SCENE_CONTRACT.get("expected") or {})


def clear_scene_contract():
    _SCENE_CONTRACT["expected"] = None
    _SCENE_CONTRACT["tag"] = None
    _SCENE_CONTRACT["registered"] = False


# 三条**必须为真**的通道 (md ✓: 不得用全局开关把目标—桌面也关掉 ✗)
REQUIRED_CHANNELS = ("target_fingers", "target_table", "fingers_table")


def verify_scene_channels(expected=None, actual=None):
    """★ 返回 ``(ok, diffs)``: 逐通道核对 (期望 vs 实际) ✓.

    规则 ✓:
      · `target_fingers / target_table / fingers_table` 三条**必须为 ON** ✓
        (任何一条被关 ⇒ 违规, 包括"用全局开关把 目标—桌面 一起关掉" ✗)
      · `others` (其余物体) **必须为 OFF** (隔离 ✓)
    """
    _exp = dict(expected if expected is not None else get_scene_contract())
    _act = dict(actual or {})
    diffs = []
    if not _exp and not _act:
        return False, ["无期望且无实际通道信息 ⇒ 无法证明 (不得当作通过 ✗)"]
    for _ch in REQUIRED_CHANNELS:
        _e = _exp.get(_ch, True)
        _a = _act.get(_ch, None)
        if _a is None:
            diffs.append("通道 %s 缺实际值 (不可验证 ⇒ 不通过 ✗)" % _ch)
            continue
        if bool(_e) is not bool(_a):
            diffs.append("通道 %s 期望=%s 实际=%s ✗" % (_ch, bool(_e), bool(_a)))
        elif not bool(_a):
            diffs.append("通道 %s 被关闭 (md: **不得**关掉必需通道 ✗)" % _ch)
    if "others" in _exp or "others" in _act:
        if bool(_act.get("others", False)):
            diffs.append("others 通道=ON ⇒ **未隔离其他物体** ✗")
    return (len(diffs) == 0), diffs


def assert_scene_contract(actual=None):
    """★ **step 前**调用 ✓: 合同不符 ⇒ 抛异常 ⇒ **阻断** (不带病步进 ✗) ✓

    ★ T08-B2 ✓: 若登记了**采样器**, 则用**真实读回**的三通道 (而非调用方自报 ✓)
    """
    if not _SCENE_CONTRACT.get("registered"):
        return True                      # 未登记 ⇒ 不拦 (仅证明性检查 ✓)
    if actual is None:
        _fn = get_scene_sampler()
        if callable(_fn):
            try:
                actual = _fn()
            except Exception as _es:  # noqa: BLE001
                raise RuntimeError("SCENE_SAMPLER_FAILED (step 前阻断 ✓): %s" % _es)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T08-C2 修 (2026-09-27) 【首个 step 前的自举放行】 ✓ ★★★★
            #   实测 ✗: 采样器在**第一次 step 之前**读回的三个通道全为 None
            #     ⇒ `SCENE_CONTRACT_VIOLATION: 通道 target_fingers 缺实际值 (不可验证 ⇒ 不通过 ✗)`
            #     ⇒ Stage2 **连第 1 步都迈不出去** ✗ (s34 终局 ✓)
            #   ⇒ 本改 ✓: 仅当**三通道全无值**(= 尚无 step 数据 ✓) 时, 首个 step 前放行一次并告警;
            #     只要**任一通道有值**, 仍按原合同**严格核对并阻断** ✓ (不放松门控 ✓)
            #   ⚠ TODO ✓: 让 `collect_collision_channels` 支持**按需现算** (而非读上一步缓存) ⇒
            #     之后可删除本自举放行, 恢复"每步必可验证" ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if isinstance(actual, dict):
                # ★ T08-C2b 修 (2026-09-27) ✓: 只看**三条必查通道** ✓
                #   实测 ✗ (s35): 原写"四通道全 None" ⇒ `others=False` 被判为**有值** ✗
                #     ⇒ 自举放行不触发 ⇒ 仍 `SCENE_CONTRACT_VIOLATION: 通道…缺实际值` ✗
                #   ⇒ 改为: `target_fingers / target_table / fingers_table` **三者全无值**
                #     才视为"尚无 step 数据" ⇒ 放行一次 ✓ (任一条有值 ⇒ 仍严格阻断 ✗)
                _vals = [actual.get(_k) for _k in ("target_fingers", "target_table",
                                                   "fingers_table")]
                if all(_v is None for _v in _vals):
                    # ★★ 任务02 (2026-09-28) 【不得**持续**放行 ✗】★★
                    #   md §02 原话 ✓: "不再持续豁免 UNKNOWN; **移除'只打一次日志、无限次放行'**;
                    #     碰撞配置应在场景建好后可读 … **任一必需通道未知或无效立即阻断**" ✓
                    #   旧病 ✗: 三通道全 None 时**每次**都 `return True` ⇒ 只要采样器一直读不到,
                    #     门控就**永久失效** ✗ (只多打了一行日志 ✗)
                    #   本改 ✓: 自举放行**只允许一次** (真正的"首个 step 之前" ✓);
                    #     `_scene_actors` 修好之后通道应能读到实际值 ⇒ 第二次仍全 None
                    #     ⇒ 说明接口/场景真有问题 ⇒ **立即阻断** ✗
                    if not _SCENE_CONTRACT.get("_bootstrap_used"):
                        _SCENE_CONTRACT["_bootstrap_used"] = True
                        logger.warning(
                            "SCENE_CONTRACT_BOOTSTRAP ✓: 首个 step 前通道**尚无实际值** ⇒ "
                            "**仅本轮放行一次**; 下次仍无值即**阻断** ✗ (md §02 ✓)")
                        return True
                    raise RuntimeError(
                        "SCENE_CONTRACT_VIOLATION (step 前阻断 ✓): 三通道**连续无实际值** ⇒ "
                        "自举放行已用过 ⇒ 拒绝继续 ✗ (md §02: 不得持续豁免 UNKNOWN ✓; "
                        "检查 `_scene_actors` 是否读到 actor / 场景是否已建好)")
    _ok, _diffs = verify_scene_channels(get_scene_contract(), actual)
    if not _ok:
        raise RuntimeError("SCENE_CONTRACT_VIOLATION (step 前阻断 ✓): %s" % _diffs[:4])
    return True


_STEP_TAG = {"tag": "stage1"}


def set_step_tag(tag):
    """★ T08-B ✓ 标注当前调用方 (stage1/stage2) ⇒ 供"同入口"证明 ✓"""
    _STEP_TAG["tag"] = str(tag or "stage1")
    return _STEP_TAG["tag"]


def get_step_tag():
    return str(_STEP_TAG.get("tag") or "stage1")


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T08-B2 (2026-09-27) 【三通道从场景**自动采样**】 ✓ ★★★★
#   md ✓: "逐 body/shape 核对 目标—双指 / 目标—桌面 / 双指—桌面 通道, 其他物体隔离。
#     **只比摘要而不验证实际场景不算完成**" ✓
#   ⇒ 本函数**读回**每个 shape 的碰撞组/掩码 (PhysX 语义: 两 shape 相碰 ⟺
#      `(groupA & maskB) != 0 且 (groupB & maskA) != 0` ✓) ⇒ 得到**真实**通道状态 ✓
#   ⚠ 只读 ✓: 不修改任何碰撞配置 ✓
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def _shape_groups(shape):
    """读回 ``(group, mask)``；失败 ⇒ ``None`` ✓"""
    try:
        _g = list(shape.get_collision_groups())
        return int(_g[0]), int(_g[1])
    except Exception:  # noqa: BLE001
        return None


def _pair_collides(_ga, _gb):
    """PhysX 语义: 两 shape 相碰 ⟺ 双向 group/mask 同时命中 ✓"""
    try:
        (_ta, _ma), (_tb, _mb) = _ga, _gb
        return bool((_ta & _mb) and (_tb & _ma))
    except Exception:  # noqa: BLE001
        return None


def build_collision_audit(robot=None, side="left", obj_actors=None,
                         target_name="", scene=None, tag=""):
    """★ md 06.4-e2 (2026-09-30) ✓ 生成 ``collision_audit`` 内容 (调用方落盘 ✓)

    含 ✓: 身份映射 / 逐 shape 详情 (组件类型、group 元组、局部 pose、geometry 摘要、
      材质/可读标志) / **分指矩阵** / UNKNOWN 显式标注 ✓
    ⚠ md 口径 ✓: "只有定位到**真实过滤/标志变化**并与失联时序对应, 才写'通道关闭'" ✓
      ⇒ 本函数只报**读回值**与 UNKNOWN, 不做因果断言 ✗
    """
    import time as _time
    _aud = {"tag": str(tag), "side": str(side), "target_name": str(target_name),
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "channels": None, "identity": {}, "shapes": {}, "notes": []}
    try:
        _ch = collect_collision_channels(robot=robot, side=side, obj_actors=obj_actors,
                                         target_name=target_name, scene=scene)
        _aud["channels"] = _ch
        _aud["per_finger"] = _ch.get("per_finger")
        _aud["identity"]["target"] = _ch.get("target_identity")
        _aud["counts"] = _ch.get("counts")
        _aud["errors"] = _ch.get("errors")
    except Exception as _e:  # noqa: BLE001
        _aud["notes"].append("channels 采集失败: %s" % _e)
    try:
        _acts = _scene_actors(scene)
        _ta, _tinfo = resolve_target_actor_identity(_acts, target_name)
        _aud["identity"]["target_actor_resolved"] = (_ta.get_name() if _ta is not None
                                                     else None)
        _aud["identity"]["target_actor_info"] = _tinfo
    except Exception as _e2:  # noqa: BLE001
        _aud["notes"].append("身份解析失败: %s" % _e2)
    try:
        # 逐 shape 详情 (渲染组件无碰撞接口 ⇒ 显式 UNKNOWN ✓ 不静默跳过 ✗)
        import numpy as _np_a
        _rec = []
        for _a_ in (obj_actors or []):
            try:
                _an = str(_a_.get_name())
            except Exception:  # noqa: BLE001
                continue
            for _c in getattr(_a_, "components", []) or []:
                _ct = type(_c).__name__
                if not hasattr(_c, "get_collision_shapes"):
                    _rec.append({"actor": _an, "component": _ct, "collision": "UNKNOWN",
                                 "reason": "no get_collision_shapes (render component)"})
                    continue
                for _i_s, _sh in enumerate(_c.get_collision_shapes()):
                    _it = {"actor": _an, "component": _ct, "index": int(_i_s),
                           "collision": True}
                    _gg = _shape_groups(_sh)
                    _it["groups"] = list(_gg) if _gg else "UNKNOWN"
                    try:
                        _it["local_p"] = [round(float(x), 6)
                                          for x in _np_a.asarray(_sh.get_local_pose().p)]
                    except Exception:  # noqa: BLE001
                        _it["local_p"] = "UNKNOWN"
                    try:
                        _vv = _np_a.asarray(_sh.get_vertices(), dtype=_np_a.float64)
                        _it["geometry"] = {"n_vertices": int(_vv.shape[0]),
                                           "bbox": [round(float(x), 6) for x in
                                                    (_vv.max(0) - _vv.min(0))]}
                    except Exception:  # noqa: BLE001
                        _it["geometry"] = "UNKNOWN"
                    try:
                        _it["material_readable"] = bool(_sh.get_physical_material() is not None)
                    except Exception:  # noqa: BLE001
                        _it["material_readable"] = "UNKNOWN"
                    _rec.append(_it)
        _aud["shapes"] = _rec
    except Exception as _e3:  # noqa: BLE001
        _aud["notes"].append("shape 详情采集失败: %s" % _e3)
    _aud["notes"].append("只报读回值; '通道关闭' 需与失联时序对应后才可断言 (md e2)")
    return _aud


def resolve_target_actor_identity(actors, target_name):
    """★ md 06.4-e2 (2026-09-30) ✓ 目标**身份映射**: 拒绝歧义子串匹配 ✓

    返回 ``(actor, info)``; ``info`` 含 ``matched``/``mode``/``candidates``/``ambiguous`` ✓
    · 精确名匹配优先 ⇒ ``exact`` ✓
    · 唯一子串匹配次之 ⇒ ``unique_substring`` ✓
    · **命中多个 ⇒ ambiguous=True, actor=None** ✗ (md: 拒绝歧义子串匹配 ✓ 不静默取第一个 ✗)
    """
    _t = str(target_name or "")
    _names, _objs = [], []
    for _a in actors or []:
        try:
            _n = str(_a.get_name())
        except Exception:  # noqa: BLE001
            continue
        _names.append(_n)
        _objs.append((_n, _a))
    if not _t:
        return None, {"matched": None, "mode": "no_target_name", "candidates": [],
                      "ambiguous": False}
    for _n, _a in _objs:
        if _n == _t:
            return _a, {"matched": _n, "mode": "exact", "candidates": [_n],
                        "ambiguous": False}
    _subs = [_n for _n in _names if _t in _n]
    if len(_subs) == 1:
        for _n, _a in _objs:
            if _n == _subs[0]:
                return _a, {"matched": _n, "mode": "unique_substring",
                            "candidates": _subs, "ambiguous": False}
    return None, {"matched": None,
                  "mode": ("ambiguous_substring" if len(_subs) > 1 else "not_found"),
                  "candidates": _subs, "ambiguous": bool(len(_subs) > 1)}


def _collect_shapes_from_links(links):
    out = []
    for _l in links or []:
        try:
            for _s in _l.get_collision_shapes():
                _gg = _shape_groups(_s)
                if _gg is not None:
                    out.append(_gg)
        except Exception:  # noqa: BLE001
            continue
    return out


def _collect_shapes_from_actors(actors, name_contains=None, only_target=None,
                               target_name=""):
    """★ T08B2 ✓ 从 actor 组件读回 shape 碰撞组 ✓

    ``name_contains``: 名字需包含该子串; ``only_target``: True=只要目标, False=只要非目标 ✓
    """
    out = []
    for _a in actors or []:
        try:
            _n = str(_a.get_name())
        except Exception:  # noqa: BLE001
            continue
        if name_contains is not None and str(name_contains) not in _n:
            continue
        if only_target is not None:
            _is_t = bool(target_name) and (str(target_name) in _n)
            if bool(_is_t) is not bool(only_target):
                continue
        try:
            for _c in _a.get_components():
                # ★ 2026-09-28 修 ✓: `RenderBodyComponent` **没有** `get_collision_shapes()`
                #   ⇒ 旧写法 AttributeError 被**外层 try 整个吞掉** ⇒ 该 actor 的
                #   **全部 shapes 丢失** ⇒ 三通道恒 None ⇒ "自举放行已用过 ⇒ 拒绝" ✗
                #   (实测 /tmp/stage2_r1.log: 初始通道 目标—双指=None ✗)
                #   本改 ✓: 组件级 hasattr 过滤, 渲染组件跳过 ✓
                if not hasattr(_c, "get_collision_shapes"):
                    continue
                for _s in _c.get_collision_shapes():
                    _gg = _shape_groups(_s)
                    if _gg is not None:
                        out.append(_gg)
        except Exception:  # noqa: BLE001
            continue
    return out


    return out


def _scene_actors(scene):
    """★ 任务02 (2026-09-28) 【SAPIEN 3.0 接口核对 —— `get_actors` 不存在 ✗】★

    md §02 原话 ✓: "根据当前安装版本核对 Scene、actor/entity/component 和 shape 的实际接口,
      **修 `get_actors` 等错误调用**; mock 必须模拟实际对象结构" ✓

    实测铁证 ✗ (2026-09-28, SAPIEN **3.0.0b0**):
      · `hasattr(sapien.Scene, "get_actors") == False` ✗ —— 正式 API 是
        `get_all_actors()` / `get_entities()` / `entities` ✓
      · 旧代码 `scene.get_actors()` ⇒ `AttributeError` 被上层 `except` **吞掉** ⇒
        地面 shape 列表 = `[]` ⇒ `target_table = _any(target_shapes, []) = None` ✗
        ⇒ 合同门判"通道 target_table 缺实际值 (不可验证 ⇒ 不通过 ✗)"
        ⇒ **Stage2 一步都迈不出去** ✗ (实测 `output/stage2_from_s31` ✓)

    本改 ✓: 按**能力**取 (新 API 优先, 旧名兜底), 且**都不行时返回空并由调用方记 errors**
      (不静默 ✗, 语义仍是"读不到 ⇒ 不可验证 ⇒ 不通过" ✓)
    """
    if scene is None:
        return []
    for _name in ("get_all_actors", "get_actors", "get_entities", "entities"):
        _member = getattr(scene, _name, None)
        if _member is None:
            continue
        try:
            _val = _member() if callable(_member) else _member
        except Exception:  # noqa: BLE001
            continue
        if _val:
            return list(_val)
    return []


def collect_collision_channels(robot=None, side="left", obj_actors=None,
                               target_name="", scene=None):
    """★ T08-B2 ✓ 返回**真实**三通道 + ``others`` (全部由读回计算 ✓).

    ``None`` 表示**读不到** ⇒ 合同门会判"不可验证 ⇒ 不通过" ✓ (不当作默认通过 ✗)
    """
    out = {"target_fingers": None, "target_table": None, "fingers_table": None,
           "others": None, "counts": {}, "errors": []}
    _s = str(side or "left")
    _fingers, _f1_links, _f2_links = [], [], []
    try:
        for _l in (robot.get_links() if robot is not None else []) or []:
            try:
                _n = str(_l.get_name())
            except Exception:  # noqa: BLE001
                continue
            if _n == f"{_s}_gripper_finger_link1":
                _f1_links.append(_l); _fingers.append(_l)
            elif _n == f"{_s}_gripper_finger_link2":
                _f2_links.append(_l); _fingers.append(_l)
    except Exception as _e:  # noqa: BLE001
        out["errors"].append("取手指 link 失败: %s" % _e)
    _fg = _collect_shapes_from_links(_fingers)
    # md 06.4-e2 ✓: **逐指**收集 (不能两指合并后 any ⇒ 无法判单指 ✗)
    _fg1 = _collect_shapes_from_links(_f1_links)
    _fg2 = _collect_shapes_from_links(_f2_links)
    _tg = _collect_shapes_from_actors(obj_actors, only_target=True,
                                target_name=str(target_name or ""))
    # ★ md 06.4-e2 反例⑥ (2026-09-30) ✓: 目标侧 shapes **为空** ⇒ 必须显式记为
    #   **不可验证** ✗ (由 mock 测试暴露: 旧代码只让通道变 None, 却**不留 error** ✗
    #   ⇒ 事后无从判断"是没读还是读不到" ✗)
    if not _tg:
        out["errors"].append(
            "目标 shapes 为空 ⇒ 目标侧通道**不可验证** ✗ (不当作默认通过 ✗)")
    _ot = _collect_shapes_from_actors(obj_actors, only_target=False,
                                target_name=str(target_name or ""))
    # 地面: 按名字含 ground 的 actor ✓ (取不到 ⇒ 相关通道未知 ⇒ 不通过 ✓)
    _gd = []
    try:
        # ★ 任务02 (2026-09-28) ✓: 用**版本兼容**的 actor 枚举 (旧 `scene.get_actors()`
        #   在 SAPIEN 3.0 不存在 ⇒ 抛错被吞 ⇒ 地面通道空 ⇒ Stage2 全被拦 ✗, 见 `_scene_actors`)
        _acts = _scene_actors(scene)
        if not _acts:
            out["errors"].append(
                "取场景 actor 失败: `%s` 上 get_all_actors/get_actors/entities "
                "**均不可用或为空** ⇒ 地面通道不可验证 ✗ (不当作默认通过 ✗)"
                % type(scene).__name__)
        _gd = _collect_shapes_from_actors(_acts, name_contains="ground")
        if _acts and not _gd:
            out["errors"].append(
                "场景有 %d 个 actor, 但**没有**名字含 'ground' 的 ⇒ 桌面通道不可验证 ✗"
                % len(_acts))
    except Exception as _e2:  # noqa: BLE001
        out["errors"].append("取地面 actor 失败 (对象=%s, 类型=%s): %s"
                             % (scene, type(scene).__name__, _e2))
    out["counts"] = {"fingers": len(_fg), "target": len(_tg),
                     "others": len(_ot), "ground": len(_gd)}

    def _any(_A, _B):
        if not _A or not _B:
            return None
        for _a in _A:
            for _b in _B:
                if _pair_collides(_a, _b):
                    return True
        return False

    out["target_fingers"] = _any(_tg, _fg)
    out["target_table"] = _any(_tg, _gd)
    out["fingers_table"] = _any(_fg, _gd)
    # ★ md 06.4-e2 ✓: **分指矩阵** (左指×目标 / 右指×目标 / 指×桌面 逐指给值 ✓)
    out["per_finger"] = {
        "finger1_shapes": len(_fg1), "finger2_shapes": len(_fg2),
        "target_finger1": _any(_tg, _fg1), "target_finger2": _any(_tg, _fg2),
        "finger1_table": _any(_fg1, _gd), "finger2_table": _any(_fg2, _gd),
        "both_fingers_shapes_ok": bool(_fg1) and bool(_fg2),
    }
    if not _fg1 or not _fg2:
        out["errors"].append(
            "分指 shape 缺失 (finger1=%d, finger2=%d) ⇒ 双边不可验证 ✗"
            % (len(_fg1), len(_fg2)))
    # ★ md 06.4-e2 ✓: 目标**身份映射** (拒绝歧义子串; 供 collision_audit 记录 ✓)
    try:
        _id_acts = _scene_actors(scene)
        _id_t, _id_info = resolve_target_actor_identity(_id_acts, target_name)
        out["target_identity"] = _id_info
        if _id_info.get("ambiguous"):
            out["errors"].append(
                "目标名 `%s` **歧义子串匹配** %s ⇒ 拒绝 ✗ (md e2: 须唯一映射 ✓)"
                % (target_name, _id_info.get("candidates")))
    except Exception as _eid:  # noqa: BLE001
        out["target_identity"] = {"mode": "unavailable", "error": str(_eid)}
    _o1 = _any(_tg, _ot)
    _o2 = _any(_fg, _ot)
    if _o1 is None and _o2 is None:
        out["others"] = None
    else:
        out["others"] = bool((_o1 is True) or (_o2 is True))
    # ★ 任务02 诊断开关 (零物理 ✓ 只多打一行) ✓: `_T02_DUMP_CHANNELS=1`
    #   用途 ✓: 通道判 False 时, **读回实际**的 (名字, group, mask) ⇒ 定位"谁把通道关了" ✗
    #   (md §02: 要按**实际过滤规则**核对, 不能只比摘要 ✓)
    if str(os.environ.get("_T02_DUMP_CHANNELS", "0")) == "1":
        try:
            _dump = []
            for _a in _scene_actors(scene):
                try:
                    _an = str(_a.get_name())
                except Exception:  # noqa: BLE001
                    _an = "?"
                for _c in _a.get_components():
                    for _s in _c.get_collision_shapes():
                        _g = list(_s.get_collision_groups())
                        _dump.append((_an, int(_g[0]), int(_g[1])))
            for _l in (robot.get_links() if robot is not None else []) or []:
                try:
                    _ln = str(_l.get_name())
                except Exception:  # noqa: BLE001
                    _ln = "?"
                if "finger" in _ln or "gripper" in _ln:
                    for _s in _l.get_collision_shapes():
                        _g = list(_s.get_collision_groups())
                        _dump.append(("ROBOT:" + _ln, int(_g[0]), int(_g[1])))
            _ground_dump = [d for d in _dump if "ground" in d[0].lower() or "table" in d[0].lower()]
            _all_actors = sorted({d[0] for d in _dump})
            logger.warning("[T02 dump] 场景 actor 名单 (%d): %s", len(_all_actors), _all_actors)
            logger.warning("[T02 dump] **地面/桌面 shape** (%d 个): %s",
                           len(_ground_dump), _ground_dump[:10])
            logger.warning("[T02 dump] **指尖 shape**: %s",
                           [d for d in _dump if d[0].startswith("ROBOT:")][:8])
        except Exception as _ed:  # noqa: BLE001
            logger.warning("[T02 dump] 失败: %s", _ed)
    return out


_SCENE_SAMPLER = {"fn": None}


def register_scene_sampler(fn):
    """★ 登记采样器 ⇒ `assert_scene_contract()` 每次 step 前**用真实读回**核对 ✓"""
    _SCENE_SAMPLER["fn"] = fn if callable(fn) else None
    return _SCENE_SAMPLER["fn"] is not None


def get_scene_sampler():
    return _SCENE_SAMPLER.get("fn")


def clear_scene_sampler():
    _SCENE_SAMPLER["fn"] = None

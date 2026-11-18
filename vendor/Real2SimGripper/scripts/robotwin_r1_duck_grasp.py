#!/usr/bin/env python3
"""robotwin_r1_duck_grasp.py — 用真实人手轨迹驱动 Galaxea R1 抓取 GLB 中的粉色小鸭。

链路:
  1. 从 final_scene.glb 提取 toy_0(粉鸭) 网格 (Z-up 原始顶点, 重新居中) -> /tmp/pink_duck.glb
  2. RoboTwin(SAPIEN) 场景: 桌子 + R1; 粉鸭置于桌面
  3. 真实轨迹 joints_ras_world_left (Z-up, 米) 的抓取段, 映射到 RoboTwin:
       p_rb = duck_rb + (p_glb - duck_glb)          (两系同为 Z-up 米制, 仅平移)
  4. 逐帧 IK(连续性正则) 驱动 R1 左臂; 智能闭合夹爪
  5. 量化: 逐帧末端跟踪误差(mm/deg) + 粉鸭抬升 + 滑移
"""
import os
import sys
import types
import numpy as np
import xml.etree.ElementTree as ET

import sapien.core as sapien
import sapien.physx as sapienp
import sapien.render as sapienr
import imageio
import trimesh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- 资产 ----
R1_DIR = "/tmp/GalaxeaManipSim/galaxea_sim/assets/r1"
R1_URDF = os.path.join(R1_DIR, "robot.urdf")
GLB = "/path/to/workspace/traj_data_min/ras/7_vggt_omega/final_scene.glb"
NPZ = "/path/to/workspace/traj_data_min/hawor/7a/reconstruction/mano_ras_reconstruction_0_113.npz"
DUCK_DIR = "/tmp/duck_assets"                       # 压缩后的粉鸭资源 (减面+烘贴图)
DUCK_OBJ = os.path.join(DUCK_DIR, "pink_duck.obj")
DUCK_TEX = os.path.join(DUCK_DIR, "pink_duck_tex.png")   # 烘好的顶点色贴图 (SAPIEN 需显式加载)
COMPRESS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compress_duck.py")
SCENE_DIR = "/tmp/scene_assets"                     # 场景其余物体 (最终 GLB 复现)
SCENE_COMPRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compress_scene.py")
DUCK_GEOM = "geometry_4"                            # final_scene.glb 中粉鸭的几何名
OUT_DIR = os.path.join(REPO_ROOT, "demo_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- R1 常量 (与 robotwin_r1_display.py 一致) ----
# 基座 xy: 默认用「简单左右平移基线」——把基座沿 +x 平移即让左臂工作空间
# 覆盖人手搬运高点, 消除后段 IK 顶限位冻结 (follow_max 230mm -> 62mm)。
# 该值由 _scratch/scan_base.py 网格扫描得到 (左右甜区 x≈0.1~0.2);
# 进一步可用 agentic real2sim (agentic_place.py) 让 VLM 自主微调。
R1_BASE_P = [0.20, -0.42, 0.0]
R1_BASE_Q = [0.70710678, 0.0, 0.0, 0.70710678]      # 面朝 +y
# 工作机制臂: 真实视频是左手操作, 故用左臂。左臂与右臂互为关于矢状面(y=0)的镜像,
# 起始位姿由右臂镜像映射得到: left = [-r1, r2, r3, -r4, -r5, -r6]。
# (两臂关节限位一致; 夹爪局部系约定一致: 接近轴=局部 +x, 开合轴=局部 ±y)
R1_LEFT_START = np.array([1.5, 1.9508485794067383, -1.080855369567871,
                          0.4437669813632965, -0.17093953490257263, -0.19848759472370148])
R1_RIGHT_PARK = np.array([-1.0, 0.6, -0.4, -0.5, 0.0, 0.0])   # 右臂(非工作臂)收起位
IK_LIMITS = np.array([[-2.8798, 2.8798], [0.0, 3.2289], [-3.3161, 0.0],
                      [-2.8798, 2.8798], [-1.6581, 1.6581], [-2.8798, 2.8798]])

# ---- 工作机制臂命名 (左臂) ----
ARM = "left"
ARM_JOINTS = [f"{ARM}_arm_joint{i}" for i in range(1, 7)]
GRIP_LINK = f"{ARM}_gripper_link"
GRIP_FINGERS = (f"{ARM}_gripper_finger_joint1", f"{ARM}_gripper_finger_joint2")
PAD_FWD = 0.0369          # gripper_link -> pad 中心, 沿局部 +x
# PD 增益: 降到 1000/200 并配合逐物理步重力+科氏补偿 (compute_passive_force + set_qf),
# 重力下垂由补偿项抵消, 动作更柔顺 (原 4000/400 靠高刚度硬扛重力, 跟踪发硬)。
ARM_STIFF, ARM_DAMP = 1000.0, 200.0
GRIP_OPEN_Q = 0.045       # 内距 11.69cm
GRIP_HOLD_Q = 0.0302      # 夹住 9cm 粉鸭: 内距 = 0.017+2.22*q ≈ 8.4cm, 挤压 0.6cm
CLOSE_F = 48              # 闭合夹爪帧 (用户: ~F50 抓取; 数据: 指尖接触窗口末端)
OPEN_F = 92               # 张开夹爪帧 (用户: ~F90 放置; 数据: 掌心回落最低点)
TABLE_Z = 0.74
DUCK_RB_XY = np.array([0.0, -0.18])   # 粉鸭桌面 xy (R1 前侧可达)
CTRL_STEPS = 25           # 每个控制步的仿真步数 (dt=1/250 -> 控制频率 10Hz)
CTRL_FREQ = 10            # 控制频率 (Hz) = 250 / CTRL_STEPS

# 真实抓取窗口: 含接近段(F18-34 手伸向粉鸭) + 抓取后抬起搬运段(F34-108 掌心抬升13cm)
WIN = (18, 108)


def _install_lightweight_envs_package():
    envs_dir = os.path.join(REPO_ROOT, "envs")
    for name, path in (("envs", envs_dir), ("envs.utils", os.path.join(envs_dir, "utils")),
                       ("envs.robot", os.path.join(envs_dir, "robot")),
                       ("envs.camera", os.path.join(envs_dir, "camera"))):
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        pkg.__package__ = name
        sys.modules[name] = pkg
    stub = types.ModuleType("envs.robot.planner")
    stub.MplibPlanner = None
    stub.CuroboPlanner = None
    sys.modules["envs.robot.planner"] = stub


sys.path.insert(0, REPO_ROOT)
_install_lightweight_envs_package()
from envs.utils.create_actor import create_table, create_box  # noqa: E402


def make_r1_urdf_abs(urdf_path, out_path):
    tree = ET.parse(urdf_path)
    base = os.path.dirname(os.path.abspath(urdf_path))
    n = 0
    for mesh in tree.getroot().iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("./"):
            mesh.set("filename", os.path.join(base, fn[2:]))
            n += 1
    tree.write(out_path)
    return n


def setup_scene(engine):
    sapienr.set_global_config(max_num_materials=50000, max_num_textures=50000)
    scene = engine.create_scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
    # 光照 (颜色保真 + 稳定无闪烁):
    #  - SAPIEN 中 base_color/贴图按**线性**解释、输出经 sRGB 编码; 受光≈1.0 时颜色忠实饱和。
    #  - 采用**单主光(1 个平行光) + 高均匀环境光**, 去掉两个点光源:
    #    点光源的立方体阴影贴图随场景移动逐帧重算, 最易产生"同视角内忽明忽暗"的闪烁;
    #    单平行光只有一张阴影图, 配合高环境光(阴影很淡) => 光照稳定、各视角一致。
    #    实测: 粉鸭像素 (170,28,159) ≈ 顶点色 (170,19,157) (dist 9.6, 与三光源持平)。
    scene.set_ambient_light([0.90, 0.90, 0.90])
    scene.add_directional_light([0.3, 0.4, -1], [0.40, 0.40, 0.40], shadow=True)
    return scene


def look_at_pose(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """构造 SAPIEN 相机 pose。SAPIEN 相机约定: 局部 +x=forward, +y=left, +z=up。"""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    f = f / np.linalg.norm(f)
    l = np.cross(up, f)
    l = l / np.linalg.norm(l)
    u = np.cross(f, l)
    m = np.eye(4)
    m[:3, :3] = np.stack([f, l, u], axis=1)
    m[:3, 3] = eye
    return sapien.Pose(m)


# ---------- 粉鸭压缩资源 (减面 53.8x + 顶点色烘贴图) ----------
def ensure_duck_assets():
    """确保压缩后的粉鸭 OBJ+MTL+贴图存在; 返回 (原始GLB中粉鸭中心, 尺寸)。
    直接用原始 GLB 顶点色导入会渲染成白色 (SAPIEN 不读 GLB 顶点色), 故先用
    compress_duck.py 减面并把顶点色烘成贴图。"""
    import subprocess
    if not os.path.exists(DUCK_OBJ):
        print(f"生成压缩粉鸭资源: {COMPRESS_SCRIPT}")
        subprocess.run([sys.executable, COMPRESS_SCRIPT], check=True)
    meta = np.load(os.path.join(DUCK_DIR, "duck_meta.npz"))
    return meta["center"], meta["extent"]


def _textured_render_material(tex_path):
    """从烘好的 PNG 建带 base-color 贴图的渲染材质。

    关键 1: SAPIEN 的 add_visual_from_file **不会自动读 OBJ 的 MTL/map_Kd** (会渲染成白色),
    必须显式建 RenderMaterial 并 set_base_color_texture 再传入 (对齐 RoboTwin create_actor.py)。
    关键 2: 必须设 metallic/roughness —— 默认材质偏镜面, 点光源高光随视角漂移 (物体"变色")
    且整体发灰; 设为哑光 (metallic=0, roughness=0.85) 颜色才稳定、饱和。
    贴图缺失时返回 None (回退纯色, 至少不崩)。
    """
    if not os.path.exists(tex_path):
        return None
    tex = sapienr.RenderTexture2D(tex_path)
    mat = sapienr.RenderMaterial()
    mat.set_base_color_texture(tex)
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.metallic = 0.0            # 非金属: 避免镜面高光染色
    mat.roughness = 0.85          # 偏哑光: 漫反射为主, 颜色稳定不发灰
    mat.specular = 0.15
    return mat


def create_duck_actor(scene, pose):
    """用 SAPIEN 公共 API 建可抓实体 (OBJ 贴图视觉 + 凸碰撞)。"""
    b = scene.create_actor_builder()
    b.set_physx_body_type("dynamic")
    # 高摩擦材质 (friction=1.0): 与夹爪手指接触面提供足够摩擦, 抓取更稳
    grip_mat = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    b.add_multiple_convex_collisions_from_file(filename=DUCK_OBJ, scale=(1, 1, 1), material=grip_mat)
    vis_mat = _textured_render_material(DUCK_TEX)
    if vis_mat is not None:
        b.add_visual_from_file(filename=DUCK_OBJ, scale=(1, 1, 1), material=vis_mat)
    else:
        b.add_visual_from_file(filename=DUCK_OBJ, scale=(1, 1, 1))
    duck = b.build(name="pink_duck")
    duck.set_pose(pose)
    for comp in duck.get_components():
        if isinstance(comp, sapienp.PhysxRigidDynamicComponent):
            comp.mass = 0.04
            comp.angular_damping = 5.0
            comp.linear_damping = 0.5
    return duck


# ---------- 场景其余物体 (GLB 复现, 视觉贴图, 不参与碰撞) ----------
def ensure_scene_assets():
    """确保 final_scene.glb 全部物体已减面+烘贴图; 返回 meta (name, center, extent)。"""
    import subprocess
    meta_path = os.path.join(SCENE_DIR, "scene_meta.npz")
    if not os.path.exists(meta_path):
        print(f"生成场景物体资源: {SCENE_COMPRESS}")
        subprocess.run([sys.executable, SCENE_COMPRESS], check=True)
    return np.load(meta_path)


def create_prop(scene, name, obj_path, pose, collide=False):
    """静态道具。默认仅视觉(无碰撞), 复现摆设而不干扰抓取物理;
    collide=True 时额外挂静态凸碰撞 —— 用于放置目标盘子, 让物体真正落在盘上。"""
    b = scene.create_actor_builder()
    b.set_physx_body_type("static")
    vis_mat = _textured_render_material(obj_path.replace(".obj", "_tex.png"))
    if vis_mat is not None:
        b.add_visual_from_file(filename=obj_path, scale=(1, 1, 1), material=vis_mat)
    else:
        b.add_visual_from_file(filename=obj_path, scale=(1, 1, 1))
    if collide:
        b.add_multiple_convex_collisions_from_file(filename=obj_path, scale=(1, 1, 1))
    a = b.build(name=name)
    a.set_pose(pose)
    return a


def add_marker(scene, p, color, name, r=0.008):
    """纯视觉标记球 (无碰撞): 用于轨迹可视化, 不会与物体发生物理交互。"""
    b = scene.create_actor_builder()
    b.set_physx_body_type("static")
    b.add_sphere_visual(radius=r, material=tuple(color))
    a = b.build(name=name)
    a.set_pose(sapien.Pose(p=list(p)))
    return a


def step_sim(scene, robot, n=1):
    """每物理步先做重力+科氏/离心被动补偿再 step (对齐 RoboTwin BimanualManipulationEnv)。"""
    for _ in range(n):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()


def _vel_chain(y, dq_max, iters=6):
    """速度链式限幅 (稳定): 前向/后向 clip 到不动点。逐帧 |y[j]-y[j±1]| <= dq_max。"""
    y = np.asarray(y, float).copy()
    n = len(y)
    for _ in range(iters):
        prev = y.copy()
        for j in range(1, n):
            y[j] = np.clip(y[j], y[j - 1] - dq_max, y[j - 1] + dq_max)
        for j in range(n - 2, -1, -1):
            y[j] = np.clip(y[j], y[j + 1] - dq_max, y[j + 1] + dq_max)
        if np.allclose(y, prev, atol=1e-10):
            break
    return y


def rate_limit_traj(x, dq_max=0.12, ddq_max=0.04, dddq_max=0.015):
    """运动学限幅平滑器: 在零相位滑动平均之后, 对关节轨迹再施加
    速度 / 加速度 / jerk 三级上限 (前向-后向扫描, 零相位, 单位 rad/控制步)。

    dq_max   : 相邻控制步关节角差上限 (速度, 10Hz 下 0.12 ≈ 1.2 rad/s)
    ddq_max  : 二阶差分上限 (加速度)
    dddq_max : 三阶差分上限 (jerk)

    级联稳定设计 (实测: 多阶约束交织迭代会互相泵送能量导致发散, 禁止):
      1) 速度: 对轨迹本身做链式限幅 (收敛到不动点);
      2) 加速度: 对一阶差分序列做同款链式限幅后 cumsum 重建, 漂移过大则放弃本级;
      3) jerk: 对二阶差分同法; 每级后复检上一级约束。
    """
    x = np.asarray(x, float)
    y = _vel_chain(x, dq_max)                                   # ① 速度
    for _ in range(2):                                          # ② 加速度
        v = np.diff(y, axis=0)
        v_lim = _vel_chain(v, ddq_max)
        y2 = np.vstack([y[:1], y[:1] + np.cumsum(v_lim, axis=0)])
        if np.abs(y2 - y).max() > 0.3 or not np.isfinite(y2).all():
            break                                               # 重建漂移过大, 放弃本级
        y = _vel_chain(y2, dq_max)
    for _ in range(2):                                          # ③ jerk
        a = np.diff(y, 2, axis=0)
        a_lim = _vel_chain(a, dddq_max)
        # jerk 重建: 从二阶差分恢复一阶差分需要锚点 (首帧速度), 用当前轨迹的首段速度
        v0 = np.diff(y, axis=0)[:1]
        v_rec = np.vstack([v0, v0 + np.cumsum(a_lim, axis=0)])
        y2 = np.vstack([y[:1], y[:1] + np.cumsum(v_rec, axis=0)])
        if np.abs(y2 - y).max() > 0.3 or not np.isfinite(y2).all():
            break
        y = _vel_chain(y2, dq_max)
    return y


def smooth_traj(x, win=5, passes=2):
    """零相位对称滑动平均 (reflect 边界), 平滑轨迹抖动。

    win 为奇数窗口; passes 为级联次数(次数越多越平滑)。对称 FIR 无相位滞后,
    时序不变; 端点用 reflect 填充以避免边界塌陷。输入 (N, D)。
    """
    if win < 3:
        return np.asarray(x, float)
    pad = win // 2
    k = np.ones(win) / win
    y = np.asarray(x, float)
    for _ in range(passes):
        yp = np.pad(y, ((pad, pad), (0, 0)), mode="reflect")
        y = np.stack([np.convolve(yp[:, j], k, mode="valid") for j in range(y.shape[1])], axis=1)
    return y


# ---------- 人手轨迹 ----------
def hand_frames():
    """返回 (valid, palm_center(N,3), R_hand(N,3,3)) 均为 Z-up 米制。"""
    z = np.load(NPZ)
    J = z["joints_ras_world_left"]; valid = z["valid_left"]
    n = len(J)
    palm = J[:, [0, 5, 9, 13, 17]].mean(1)
    R = np.zeros((n, 3, 3))
    for i in range(n):
        j0, j5, j9, j17 = J[i, 0], J[i, 5], J[i, 9], J[i, 17]
        fwd = j9 - j0
        side = j17 - j5
        if np.linalg.norm(fwd) < 1e-6 or np.linalg.norm(side) < 1e-6:
            R[i] = np.eye(3); continue
        fwd /= np.linalg.norm(fwd)
        side = side - fwd * np.dot(side, fwd)
        side /= np.linalg.norm(side)
        nrm = np.cross(fwd, side)
        R[i] = np.stack([fwd, side, nrm], axis=1)   # 列: approach, finger-sep, normal
    return valid, palm, R, J[:, [4, 8, 12, 16, 20]], J[:, 0]


# ---------- IK: 位置 + 接近轴 + 开合轴 ----------
# 搜索可显式用 4/8 种子；验收默认保留原来的 16 种子。
# 减少种子会改变局部最优分支，不能据 PASS 或 follow 一致声称保真度不变。
IK_SEEDS = 16
IK_CONV_TOL = 1e-6
# 迭代上限。实测(2026-10-23, scripts/ik_curve.py)两个目标都跑满 1500 次:
#   ~90% 的 cost 进展在前 150 次内完成, 后 1350 次仅再改善 12%;
#   可达目标的位置误差在 it≈300 后反而从 0.28mm 恶化到 0.56mm。
# 所以"跑满 1500"既慢又更差。300 为经验值(可用 IK_MAX_ITERS 覆盖)。
IK_MAX_ITERS = 300
# 位置达标即停: pad 已进目标 2mm 内。再迭代主要磨姿态项, 而姿态项本身有地板。
IK_POS_TOL = 0.002


def validate_ik_seeds(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("ik_seeds must be a positive integer")
    return int(value)


def solve_ik(robot, scene, pos_t, app_dir, fin_dir, x0, note="", x_ref=None, reg=0.05,
             n_seeds=None, max_iters=None):
    g_link = robot.find_link_by_name(GRIP_LINK)
    rnames = ARM_JOINTS
    all_j = robot.get_active_joints()
    name2idx = {j.get_name(): k for k, j in enumerate(all_j)}
    sel = [name2idx[n] for n in rnames]
    q0 = robot.get_qpos().copy()

    def fk(x):
        q = q0.copy()
        for k, si in enumerate(sel):
            q[si] = x[k]
        robot.set_qpos(q)
        M = g_link.get_pose().to_transformation_matrix()
        return M[:3, 3] + M[:3, 0] * PAD_FWD, M[:3, :3]

    def cost_pad(x):
        """一次 FK 同时给出 cost 和 pad 位置（避免为拿位置误差再多一次 FK）。"""
        p, R = fk(x)
        a = R[:, 0]; f = R[:, 1]
        c = (3000.0 * np.sum((p - pos_t) ** 2)
             + 2.0 * (1.0 - a.dot(app_dir))
             + 1.0 * (1.0 - abs(f.dot(fin_dir))))
        if x_ref is not None:
            c += reg * float(np.sum((x - x_ref) ** 2))
        return c, p

    def cost(x):
        return cost_pad(x)[0]

    ns = validate_ik_seeds(n_seeds if n_seeds is not None else IK_SEEDS)
    mi = int(max_iters if max_iters is not None else IK_MAX_ITERS)
    rng = np.random.default_rng(0)
    lo, hi = IK_LIMITS[:, 0], IK_LIMITS[:, 1]
    if x_ref is not None:
        seeds = [x0.copy()] + [np.clip(x_ref + rng.normal(0, 0.3, 6), lo, hi)
                               for _ in range(max(0, ns - 1))]
    else:
        seeds = [x0.copy()] + [rng.uniform(lo, hi) for _ in range(max(0, ns - 1))]
    # 显式梯度下降: 步长 1e-4 的数值梯度。不能用 scipy L-BFGS-B —— 它默认梯度步长
    # 1e-8, 低于 SAPIEN set_qpos+FK 的浮点精度, 梯度被算成 0, 优化器立即退出。
    best, bx = None, x0.copy()
    for s in seeds:
        x = np.array(s, float)
        step = 0.4
        c = cost(x)
        if c < IK_CONV_TOL:          # 已收敛的种子不必再迭代
            if best is None or c < best:
                best, bx = c, x.copy()
            continue
        # 位置误差也由 cost 内部算一份, 避免重复 FK: 这里单独跟踪当前 pad
        for it in range(mi):
            if c < IK_CONV_TOL:      # 收敛早退(实测极少触发: 末 cost 常在 1~10 量级)
                break
            g = np.zeros(6)
            for k in range(6):
                dx = np.zeros(6); dx[k] = 1e-4
                g[k] = (cost(x + dx) - c) / 1e-4
            xn = np.clip(x - step * g, lo, hi)
            cn, pn = cost_pad(xn)
            if cn < c:
                x, c = xn, cn
                # 位置已达标就停: 再迭代主要在磨姿态项, 而姿态项本身有地板
                if np.linalg.norm(pn - pos_t) < IK_POS_TOL:
                    break
            else:
                step *= 0.5
                if step < 1e-5:
                    break
        if best is None or c < best:
            best, bx = c, x.copy()
    robot.set_qpos(q0)
    return bx, best


def ik_closed_loop(robot, scene, x, pos_t, app_dir, fin_dir, poke_q, iters=3, settle=300):
    """在物理里迭代: 用 pad 位置雅可比(阻尼最小二乘)修正关节, 把实际 pad 压到目标。"""
    g_link = robot.find_link_by_name(GRIP_LINK)
    rnames = ARM_JOINTS
    all_j = robot.get_active_joints()
    name2idx = {j.get_name(): k for k, j in enumerate(all_j)}
    sel = [name2idx[n] for n in rnames]
    jmap = {j.get_name(): j for j in all_j}
    lo, hi = IK_LIMITS[:, 0], IK_LIMITS[:, 1]

    def pad_pos(xx):
        jmap_ = all_j
        q = robot.get_qpos().copy()
        for k, si in enumerate(sel):
            q[si] = xx[k]
        robot.set_qpos(q)
        M = g_link.get_pose().to_transformation_matrix()
        return M[:3, 3] + M[:3, 0] * PAD_FWD

    xr = x.copy()
    for it in range(iters):
        for k, n in enumerate(rnames):
            jmap[n].set_drive_target(float(xr[k]))
        for nm in GRIP_FINGERS:
            if nm in jmap:
                jmap[nm].set_drive_target(poke_q)
        step_sim(scene, robot, settle)
        M = g_link.get_pose().to_transformation_matrix()
        p_act = M[:3, 3] + M[:3, 0] * PAD_FWD
        err = pos_t - p_act
        print(f"    闭环 it{it}: pad 实际={np.round(p_act,4).tolist()} 误差={np.linalg.norm(err)*100:.2f}cm")
        if np.linalg.norm(err) < 0.003:
            break
        # 位置雅可比 (3x6)
        q_save = robot.get_qpos().copy()
        J = np.zeros((3, 6))
        for k in range(6):
            dq = np.zeros(6); dq[k] = 1e-3
            p1 = pad_pos(np.clip(xr + dq, lo, hi))
            J[:, k] = (p1 - p_act) / 1e-3
        robot.set_qpos(q_save)
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        xr = np.clip(xr + np.clip(dq, -0.3, 0.3), lo, hi)
    return xr


# ---------- 成功判据 (仿真闭环 critic 的 verifier) ----------
def verify_grasp(duck_trace, duck_settled, pad_trace, k_g, k_close, k_open,
                 table_z=TABLE_Z, lift_thr=0.05, knock_thr=0.04,
                 slip_thr=0.07, land_thr=0.06, follow_max=None, follow_thr=0.10,
                 target=None):
    """判定一次抓取 episode 是否成功, 返回 {pass, checks, metrics, reasons}。

    判据 (均来自物理轨迹, 无需人工):
      1) no_knock_before_grasp: 抓取前粉鸭未被打飞 (相对落定位移 < knock_thr)
      2) lifted: 抬起高度 > lift_thr
      3) hold_integrity: 夹持期间 pad↔鸭 滑移 < slip_thr (真夹住, 非滑脱)
      4) released_on_table: 释放后物体静止在某个支撑面上
         - 无放置目标(target=None): 落回桌面 (末态 z <= 桌面 + land_thr)
         - 有放置目标: 落在目标上 **或** 落回桌面
         ⚠ 2026-10-19 修正: 旧版只认"落回桌面", 在"把物体放到盘子上"时是**假阴性**
           (盘面比桌面高 ~2.5cm, 物体静置在盘上 z≈0.805 > 阈值 0.80 → 必然失败)
      5) follow_integrity: 抓取后跟随段末端最大跟踪误差 < follow_thr
         (IK 不可达冻结时该值骤增, 用于触发基座位置等参数重搜)
      6) on_target: **仅当给了 target** —— 物体水平落在目标脚印内 且 高度≈目标静置高
         ⚠ 旧版无法区分"放到目标上" vs "错过目标掉桌上": 实测换成 8cm 小碟后
           verifier 判 PASS 但鸭子偏出目标 6cm, 且 released_on_table 通过的原因
           恰恰是掉在了桌面上 —— 旧判据**反向奖励了失败**。
    """
    d0 = duck_trace[k_g]
    lift = float(duck_trace[:, 2].max() - d0[2])
    knock = float(np.linalg.norm(duck_trace[k_close, :2] - duck_settled[:2]))
    slip = np.linalg.norm(pad_trace - duck_trace, axis=1)
    hold_slip = float(slip[k_close:k_open + 1].max()) if k_open > k_close else float(slip[k_close])
    landed_z = float(duck_trace[-1, 2])

    # ---- 目标相关: 水平在脚印内 且 高度≈目标静置高 ----
    on_target = None
    if target is not None:
        dx = float(duck_trace[-1, 0] - float(target["xy"][0]))
        dy = float(duck_trace[-1, 1] - float(target["xy"][1]))
        dz = abs(landed_z - float(target["rest_z"]))
        ex = float(target["ext"][0]) / 2.0
        ey = float(target["ext"][1]) / 2.0
        on_target = bool(abs(dx) <= ex and abs(dy) <= ey and dz <= land_thr)

    released_ok = (landed_z <= table_z + land_thr) if on_target is None \
        else (bool(on_target) or (landed_z <= table_z + land_thr))

    checks = {
        "no_knock_before_grasp": knock < knock_thr,
        "lifted": lift > lift_thr,
        "hold_integrity": hold_slip < slip_thr,
        "released_on_table": released_ok,
    }
    metrics = {"lift_m": lift, "knock_m": knock, "hold_slip_m": hold_slip, "landed_z": landed_z}
    if on_target is not None:
        checks["on_target"] = on_target
        metrics["on_target"] = on_target
        metrics["target_rest_z"] = float(target["rest_z"])
    if follow_max is not None:
        checks["follow_integrity"] = follow_max < follow_thr
        metrics["follow_max_m"] = float(follow_max)
    reasons = [k for k, v in checks.items() if not v]
    return {"pass": len(reasons) == 0, "checks": checks, "metrics": metrics, "reasons": reasons}


# 可被 build_episode(params=...) 覆盖的调参项 (仿真闭环 critic 用)。
# 默认值取自模块常量; autotune 脚本在这些键上做网格/随机搜索。
PARAM_DEFAULTS = {
    "ik_seeds": IK_SEEDS,           # fast=4 / balanced=8 / accurate=16
    "ik_max_iters": IK_MAX_ITERS,   # 每种子迭代上限; 300(实测 90% 进展在前 150 次)
    # R1 基座桌面系 xy。这是**唯一需要按场景调整的自由量**(单臂工作空间覆盖人手搬运高点),
    # 由 agent 搜索 (scan_base.py / agentic_place.py)。默认 [0.12,-0.42] = agent 对
    # "数据姿态统一跟随"搜出的可行值 (follow_max 80mm PASS); [0.20,-0.42] 是给旧"垂直"姿态调的。
    "base_xy": [0.12, -0.42],
    "duck_rb_xy": DUCK_RB_XY,        # 粉鸭桌面 xy 摆位
    "d_hover": 0.13,                 # 末端对中悬停距离 (沿接近轴退避)
    # 抓取姿态来源: "data"=复现人手抓取帧的真实接近轴(≈41°偏垂直)+指间对立轴;
    # "vertical"=旧启发式 (接近轴垂直, 开合轴对准物体水平窄边)。
    "grasp_orient": "data",
    "grasp_tilt_max": 90.0,          # 数据接近轴相对垂直的最大倾角(度); 超限则夹到该安全锥
    "grasp_fin": "data",             # 开合轴来源: "data"=数据指间对立轴, "narrow"=物体水平窄边
    "grasp_carry_blend": 0,          # 抓取后姿态过渡帧数; 0=整段恒定数据姿态(统一跟随), >0 由 agent 调
    "grip_open_q": GRIP_OPEN_Q,      # 夹爪张开指令
    "grip_hold_q": GRIP_HOLD_Q,      # 夹爪夹持指令 (越小越紧)
    "close_f": CLOSE_F,              # 闭合帧
    "open_f": OPEN_F,                # 张开/放置帧
    "ctrl_win": 5, "ctrl_passes": 2,   # 人手控制点平滑
    "joint_win": 7, "joint_passes": 1,  # 关节轨迹平滑
    "max_qvel": 0.12, "max_qacc": 0.04, "max_qjerk": 0.015,  # 运动学限幅 (rad/控制步)
    "time_scale": 1.0,               # 慢放/重定时因子 (>1 慢放: 参考轨迹重采样到更密帧, 逐帧增量更小)
    # ---- 放置目标修正 (默认关闭; 打开后最后一段不再跟随人手, 改为对中下放到盘子中心) ----
    # 动机: 单目人手重建末端有 ~13cm 漂移, 释放点落在盘子边缘而非中心 (视频里鸭子是在盘心的)。
    "plate_geom": None,              # 场景几何名 (如 "geometry_8"=蓝色圆盘); None=纯跟随人手
    "place_hover": 0.10,             # 放置对中悬停高度: 盘面之上多少米开始 xy 对中
    "place_t": 0.90,                 # 落点从盘心(0)朝机器人方向挪到盘内(1) —— 盘心常超出左臂可达范围
    "place_lift": 0.12,              # 释放后夹爪抬离高度
}


def build_episode(on_step=None, collect_frames=True, params=None):
    """构建并执行一次抓取 episode (搭建场景/粉鸭/真实轨迹/IK/闭环/执行)。

    on_step(ctx, k, frame_idx): 可选回调, 每执行完一个控制步后调用一次,
        供外部记录数据 (state/action/相机)。ctx 见下方构造。
    collect_frames: 是否渲染并收集视频帧 (数据生成时不需要, 设 False 加速)。
    params: dict, 覆盖 PARAM_DEFAULTS 中的调参项 (用于闭环重搜)。

    返回 dict, 含 sim 对象、轨迹、诊断量、视频帧、verdict (成功判据) 等。
    """
    P = dict(PARAM_DEFAULTS)
    if params:
        P.update(params)
    n_seeds = validate_ik_seeds(P["ik_seeds"])
    ik_max_iters = int(P.get("ik_max_iters", IK_MAX_ITERS))
    assert os.path.exists(R1_URDF), f"R1 URDF 不存在: {R1_URDF}"
    tmp_urdf = "/tmp/r1_duck_abs.urdf"
    make_r1_urdf_abs(R1_URDF, tmp_urdf)

    duck_center_glb, duck_ext = ensure_duck_assets()
    print(f"粉鸭(GLB) 原始中心={np.round(duck_center_glb,4).tolist()} 尺寸={np.round(duck_ext,4).tolist()}")
    duck_half = duck_ext[2] / 2

    engine = sapien.Engine()
    scene = setup_scene(engine)
    create_table(scene, sapien.Pose(p=[0, 0, TABLE_Z]), length=1.2, width=0.7,
                 height=TABLE_Z, thickness=0.05, is_static=True)
    create_box(scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5],
               color=(1, 0.9, 0.9), name="wall", is_static=True)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(tmp_urdf)
    robot.set_root_pose(sapien.Pose(p=[*P["base_xy"], 0.0], q=R1_BASE_Q))
    joints = robot.get_active_joints()
    jmap = {j.get_name(): j for j in joints}
    for j in joints:
        j.set_drive_property(stiffness=ARM_STIFF, damping=ARM_DAMP)
    # 夹爪手指高摩擦 (friction=1.0): PhysX 接触摩擦默认取两接触面平均, 仅设物体一侧会减半
    grip_mat = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    for link in robot.get_links():
        if link.get_name() in (f"{ARM}_gripper_finger_link1", f"{ARM}_gripper_finger_link2"):
            for comp in link.entity.components:
                if hasattr(comp, "physx_material"):
                    comp.physx_material = grip_mat

    init = {"torso_joint1": 0.0, "torso_joint2": 0.0, "torso_joint3": 0.0, "torso_joint4": 0.0}
    for i in range(6):
        init[f"left_arm_joint{i+1}"] = float(R1_LEFT_START[i])    # 工作臂 (左)
        init[f"right_arm_joint{i+1}"] = float(R1_RIGHT_PARK[i])   # 非工作臂 (右) 收起
    for nm, v in init.items():
        if nm in jmap:
            jmap[nm].set_drive_target(float(v))
    for nm in GRIP_FINGERS:
        if nm in jmap:
            jmap[nm].set_drive_target(P["grip_open_q"])
    step_sim(scene, robot, 150)

    # ---- 粉鸭 ----
    duck_rb_center = np.array([P["duck_rb_xy"][0], P["duck_rb_xy"][1], TABLE_Z + duck_half])
    duck = create_duck_actor(scene, sapien.Pose(p=duck_rb_center.tolist()))
    step_sim(scene, robot, 120)
    duck_settled = duck.get_pose().p.copy()
    print(f"粉鸭落定位置 = {np.round(duck_settled,4).tolist()}")

    # ---- 场景其余物体: 复现 final_scene.glb 摆设 (相对粉鸭的 RAS 位置映射) ----
    # 同系平移: p_rb = duck_rb + (p_glb - duck_glb)。全部为静态视觉道具(无碰撞),
    # 只复现场景外观, 不干扰抓取物理 (避免机械臂/被夹物体撞击其它道具而失稳)。
    sm = ensure_scene_assets()
    n_prop = 0
    for nm, c, ext in zip([str(x) for x in sm["name"]], sm["center"], sm["extent"]):
        if nm == DUCK_GEOM:                       # 粉鸭已单独加载 (动态可抓)
            continue
        p_rb = duck_rb_center + (np.asarray(c) - duck_center_glb)
        # 放置目标盘子要加静态凸碰撞, 物体才能真正落在盘上 (其余道具仍纯视觉, 不干扰物理)
        is_plate = P.get("plate_geom") is not None and nm == P["plate_geom"]
        create_prop(scene, nm, os.path.join(SCENE_DIR, f"{nm}.obj"),
                    sapien.Pose(p=p_rb.tolist()), collide=is_plate)
        n_prop += 1
    print(f"场景道具加载 = {n_prop} 个 (来自 final_scene.glb)")
    step_sim(scene, robot, 20)

    # ---- 真实轨迹窗口 ----
    valid, palm, Rhand, tips, wrist = hand_frames()
    seg = np.arange(WIN[0], WIN[1])
    seg = seg[valid[seg]]
    palm_seg = palm[seg]; R_seg = Rhand[seg]; tip_seg = tips[seg]
    print(f"轨迹窗口 F{WIN[0]}-{WIN[1]-1}: {len(seg)} 帧有效")

    # 抓取帧 = 指尖离粉鸭最近的那帧
    d_tip = np.linalg.norm(tip_seg - duck_center_glb, axis=2).min(1)
    k_g = int(np.argmin(d_tip))
    print(f"  抓起帧 = F{seg[k_g]} (指尖离鸭 {d_tip[k_g]*100:.2f}cm)")

    # 夹爪时序帧 (用户指定: F50 左右抓取, F90 左右放置)。数据佐证: 指尖接触窗口
    # F34-46 末端 = 闭合时机; 掌心回落最低点 F90-94 = 放置时机。
    def frame_idx(F):
        return int(np.argmin(np.abs(seg - F)))
    k_close = frame_idx(P["close_f"])       # 闭合夹爪 (抓住)
    k_open = frame_idx(P["open_f"])         # 张开夹爪 (释放/放置)
    k_open = max(k_open, k_close + 4)  # 保证至少闭合几帧后再张开
    print(f"  夹爪时序: 闭合@F{seg[k_close]}  张开@F{seg[k_open]}")

    # 关键: 手是"掌心中心"在鸭上方 ~20cm, 接触点在指尖。
    # 用抓取帧标定"物体在手坐标系内的固定偏移", 之后每帧把该偏移随手腕姿态带过去,
    # 得到真正贴合物体的控制点 (而不是掌心)。
    off_hand = R_seg[k_g].T @ (duck_center_glb - palm_seg[k_g])
    print(f"  物体在手坐标系内偏移 = {np.round(off_hand,4).tolist()} (模长 {np.linalg.norm(off_hand):.3f} m)")
    ctrl_glb = palm_seg + np.einsum("nij,nj->ni", R_seg, np.tile(off_hand, (len(seg), 1)))
    # 平滑: 原始人手重建逐帧有噪声, 先对控制点做零相位平滑, 消除逐帧抖动
    ctrl_raw = ctrl_glb.copy()
    ctrl_glb = smooth_traj(ctrl_glb, win=P["ctrl_win"], passes=P["ctrl_passes"])
    jit = np.linalg.norm(np.diff(ctrl_raw, axis=0) - np.diff(ctrl_glb, axis=0), axis=1)
    print(f"  控制点平滑: 逐帧速度抖动中位={np.median(jit)*1000:.2f}mm/帧 "
          f"(平滑前 {np.median(np.linalg.norm(np.diff(ctrl_raw,axis=0),axis=1))*1000:.2f}mm/帧)")

    # ---- 轨迹跟随: 位置直接跟随原轨迹(三轴); 姿态采用抓取方位角 ----
    # 位置: ctrl_glb(k) 是"手坐标系内固定偏移"随腕姿带过去的控制点, 在抓取帧恰好
    #       落在鸭中心。整段位移 (含 z) 直接平移到 RoboTwin: 三轴全部跟随原轨迹。
    N = len(seg)
    target_pad = duck_settled + (ctrl_glb - duck_center_glb)   # 位置: 三轴直接跟随
    mano_ref = target_pad.copy()   # 纯 mano 映射参考(对中插入修正前), 用于对比与可视化
    # ---- 姿态: 从人手抓取帧取真实接近轴/开合轴 (grasp_orient="data"); 或用旧垂直启发式 ----
    # 位置 pad 始终落在物体中心: 平行夹爪须两爪夹住两侧, 人手单指接触点 ≠ 两爪中点。
    # 姿态改为复现人手: 接近轴 = 抓取帧 wrist -> 最近接触指尖; 开合轴 = 拇指-食指对立轴
    # (对接近轴正交化)。实测人手接近轴偏垂直 ~41°, 旧版强制垂直 => "人手斜着捏、管线直下"。
    if P.get("grasp_orient", "data") == "data":
        dk = np.linalg.norm(tip_seg[k_g] - duck_center_glb, axis=1)
        jj = int(np.argmin(dk))                               # 抓取帧最近接触指尖
        app_axis = tip_seg[k_g, jj] - wrist[seg[k_g]]
        app_axis = app_axis / np.linalg.norm(app_axis)
        # 倾角安全锥: 相对垂直的倾角超过 grasp_tilt_max 则夹到该角度 (保留水平方位角)
        down = np.array([0.0, 0.0, -1.0])
        tilt = float(np.degrees(np.arccos(np.clip(app_axis @ down, -1, 1))))
        tmax = float(P.get("grasp_tilt_max", 90.0))
        if tilt > tmax:
            h = app_axis - (app_axis @ down) * down
            h = h / np.linalg.norm(h)
            t = np.deg2rad(tmax)
            app_axis = np.cos(t) * down + np.sin(t) * h
        if P.get("grasp_fin", "data") == "narrow":
            h_ax = np.array([duck_ext[0], duck_ext[1], 0.0])
            fin_axis = np.array([1.0, 0.0, 0.0]) if h_ax[0] <= h_ax[1] else np.array([0.0, 1.0, 0.0])
        else:
            fin_raw = tip_seg[k_g, 0] - tip_seg[k_g, 1]       # 拇指尖 - 食指尖 (对立轴)
            fin_axis = fin_raw - app_axis * float(np.dot(fin_raw, app_axis))
            fn = np.linalg.norm(fin_axis)
            fin_axis = fin_axis / fn if fn > 1e-6 else np.array([1.0, 0.0, 0.0])
        print(f"  姿态(数据): 接近轴={np.round(app_axis,3).tolist()} "
              f"(偏垂直 {np.degrees(np.arccos(abs(app_axis[2]))):.1f}°, 原始 {tilt:.1f}°), "
              f"开合轴={np.round(fin_axis,3).tolist()} (fin={P.get('grasp_fin','data')})")
    else:
        h_ax = np.array([duck_ext[0], duck_ext[1], 0.0])
        fin_axis = np.array([1.0, 0.0, 0.0]) if h_ax[0] <= h_ax[1] else np.array([0.0, 1.0, 0.0])
        app_axis = np.array([0.0, 0.0, -1.0])               # 接近轴: 垂直朝下
        print(f"  姿态: 接近轴=垂直, 开合轴={fin_axis.tolist()}(对准窄边 {min(h_ax[:2])*100:.1f}cm)")
    app_dirs = np.tile(app_axis, (N, 1))
    fin_dirs = np.tile(fin_axis, (N, 1))
    # 搬运段姿态过渡 (可选参数, 不是抓取几何的必需项): carry_blend>0 时把抓取后的接近轴
    # 在 carry_blend 帧内线性过渡回垂直(可达), 开合轴随之正交化; carry_blend<=0 时
    # **整段恒定**使用数据姿态(统一跟随, 不做任何阶段特例)。
    carry_blend = int(P.get("grasp_carry_blend", 0))
    down = np.array([0.0, 0.0, -1.0])
    if carry_blend > 0:
        for k in range(k_g, N):
            a = min(1.0, (k - k_g) / carry_blend)
            aa = app_axis * (1 - a) + down * a
            aa = aa / np.linalg.norm(aa)
            ff = fin_axis - aa * float(np.dot(fin_axis, aa))
            fn2 = np.linalg.norm(ff)
            app_dirs[k] = aa
            fin_dirs[k] = ff / fn2 if fn2 > 1e-6 else fin_dirs[k]
        print(f"  搬运段姿态过渡: 抓取后 {carry_blend} 帧内接近轴过渡到垂直")
    else:
        print("  姿态: 整段恒定数据姿态 (统一跟随, 无阶段特例)")

    # ---- 末端对中插入 (宽夹爪的几何约束, 非手工轨迹) ----
    # 实测: 11.7cm 宽的夹爪若沿人手的斜切入路径下探, 会在对准前用侧壁把 9cm 的小鸭
    # 推走 ~7cm (人手能斜切入是因为手指仅 ~2cm 宽)。故在物体上方安全高度处先完成
    # 对中(悬停), 再沿接近轴下探; 此前(接近段)完全跟随人手弧线, 抓取后(抬起搬运段)
    # 再次完全跟随人手轨迹。插入段的起点由人手轨迹自身决定。
    d_hover = P["d_hover"]                 # 悬停距离: 退避到物体之外再开始对中
    retreat = -app_axis                    # 从物体指向夹爪方向
    hover = duck_settled + retreat * d_hover
    s = np.linalg.norm(target_pad - duck_settled, axis=1)   # 到物体中心的距离
    k_p = 0
    for k in range(k_g, -1, -1):
        if s[k] > d_hover:
            k_p = k
            break
    k_align = min(k_p + 3, k_g)            # 3 帧内快速完成对中(高位, 安全)
    for k in range(k_p, k_align + 1):
        a = (k - k_p) / max(1, k_align - k_p)
        target_pad[k] = target_pad[k_p] * (1 - a) + hover * a          # 高位对中
    for k in range(k_align, k_g + 1):
        b = (k - k_align) / max(1, k_g - k_align)
        target_pad[k] = hover * (1 - b) + duck_settled * b             # 沿接近轴下探
    print(f"  末端对中插入: F{seg[k_p]}-F{seg[k_align]} 高位对中, F{seg[k_align]}-F{seg[k_g]} 沿接近轴下探"
          f" (此前/此后完全跟随人手)")

    # ---- 放置段修正 (plate_geom 打开时生效) ----
    # 人手重建末端有 ~13cm 漂移: 释放点落在盘子边缘而非中心(视频里鸭子是在盘心的)。
    # 故在释放帧之前把目标改成「盘上方悬停 → 纯垂直下放」, 让物体准确落在盘上;
    # 抓取前的接近/抓取/搬运段仍完全跟随人手。
    # 注: 盘心常超出左臂可达范围(离基座 0.63m, IK 顶肩限位 → follow_max 181mm),
    #     故 place_t 允许把落点从盘心朝机器人方向挪到盘内(仍在盘上), 换取 IK 可达。
    k_place = None
    if P.get("plate_geom"):
        pname = str(P["plate_geom"])
        names = [str(x) for x in sm["name"]]
        if pname not in names:
            print(f"  [放置修正] 场景中无 {pname} (可用: {names}) -> 忽略")
        else:
            pi = names.index(pname)
            plate_glb = np.asarray(sm["center"][pi], dtype=float)
            plate_ext = np.asarray(sm["extent"][pi], dtype=float)
            plate_rb = duck_rb_center + (plate_glb - duck_center_glb)   # 同系平移映射
            plate_top = plate_rb[2] + plate_ext[2] / 2.0                # 盘面高度
            rest_z = plate_top + duck_half                              # 鸭静置时 pad(中心)高度
            # 落点: place_t=0 盘心, =1 盘心沿朝机器人方向平移到盘边 (仍在盘上)
            place_t = float(np.clip(P.get("place_t", 0.0), 0.0, 1.0))
            tgt_xy = plate_rb[:2].copy()
            sgn = -1.0 if float(P["base_xy"][1]) < plate_rb[1] else 1.0
            tgt_xy[1] += sgn * place_t * (plate_ext[1] / 2.0)
            d_h = float(P["place_hover"])
            k_a = k_open
            for k in range(k_open, -1, -1):
                if target_pad[k, 2] > rest_z + d_h:
                    k_a = k
                    break
            k_a = min(k_a, k_open - 2) if k_open > 2 else k_a
            k_al = min(k_a + 3, k_open)                                 # 3 帧内 xy 对中到落点
            xy0 = target_pad[k_a, :2].copy()
            z_hold = target_pad[k_a, 2]
            for k in range(k_a, k_al + 1):
                a = (k - k_a) / max(1, k_al - k_a)
                target_pad[k, :2] = xy0 * (1 - a) + tgt_xy * a           # 高位对中
                target_pad[k, 2] = z_hold                                # 悬停
            for k in range(k_al, k_open + 1):
                b = (k - k_al) / max(1, k_open - k_al)
                target_pad[k, :2] = tgt_xy                                # 纯垂直
                target_pad[k, 2] = z_hold * (1 - b) + rest_z * b          # 下放到盘面
            # 释放后: 沿人手回撤方向抬起 (不回跟人手最低点 —— 人手此时按到桌面以下,
            # 跟随会把夹爪压进盘子; 但也不原地高位抬离, 那会因太远而 IK 顶限位)
            n_bl = 8
            for k in range(k_open, N):
                b = min(1.0, (k - k_open) / max(1, n_bl))
                target_pad[k, 0] = tgt_xy[0] * (1 - b) + mano_ref[k, 0] * b
                target_pad[k, 1] = tgt_xy[1] * (1 - b) + mano_ref[k, 1] * b
                target_pad[k, 2] = max(rest_z * (1 - b) + mano_ref[k, 2] * b, rest_z + 0.008)
            k_place = k_a
            print(f"  放置修正: {pname} 盘心={np.round(plate_rb,4).tolist()} "
                  f"落点={np.round(tgt_xy,4).tolist()} (place_t={place_t}) "
                  f"盘面z={plate_top:.4f} 静置pad z={rest_z:.4f} | "
                  f"对中F{seg[k_a]}-F{seg[k_al]} 下放F{seg[k_al]}-F{seg[k_open]} 释放后回撤")

    print(f"  位置: 三轴直接跟随原轨迹  位移范围(相对鸭)="
          f"{np.round((ctrl_glb-duck_center_glb).max(0)-(ctrl_glb-duck_center_glb).min(0),4).tolist()} m")
    print(f"  抓取点(鸭中心) = {np.round(duck_rb_center,4).tolist()}")

    # ---- 慢放/重定时 (time_scale > 1): 参考轨迹重采样到更密帧 ----
    # 目的: 让人手轨迹"慢放"执行 —— 相邻控制步的末端/关节增量更小, 运动更平滑、
    # 更易被 PD 驱动无滞后跟随。做法: 对已构建好的 target_pad/mano_ref/seg 沿旧帧号
    # 做线性重采样到 N*time_scale 帧, 并同步映射抓取/闭合/张开帧号。姿态(垂直/窄边)
    # 与对中插入几何不变(重采样发生在插入修正之后)。
    ts = float(P.get("time_scale", 1.0))
    if ts > 1.0 + 1e-6:
        N_old = N
        t_old = np.arange(N_old)
        N_new = int(round((N_old - 1) * ts)) + 1
        t_new = np.linspace(0, N_old - 1, N_new)

        def _resample(a):
            return np.stack([np.interp(t_new, t_old, a[:, j]) for j in range(a.shape[1])], axis=1)

        target_pad = _resample(target_pad)
        mano_ref = _resample(mano_ref)
        seg = np.round(np.interp(t_new, t_old, seg.astype(float))).astype(int)

        def _kmap(k):
            return int(np.argmin(np.abs(t_new - k)))

        k_g, k_close, k_open = _kmap(k_g), _kmap(k_close), _kmap(k_open)
        k_open = max(k_open, k_close + 4)
        N = N_new
        app_dirs = np.tile(app_axis, (N, 1))
        fin_dirs = np.tile(fin_axis, (N, 1))
        print(f"  慢放重定时 time_scale={ts:.2f}: 帧数 {N_old} -> {N}, "
              f"抓取@F{seg[k_g]} 闭合@F{seg[k_close]} 张开@F{seg[k_open]}")

    # ---- 逐帧 IK: 先解抓取帧, 再向两侧传播 ----
    # 人体控制点从高处下探, 早期帧若独立求解 6 自由度冗余臂会跳到"够不到低点"的分支;
    # 从抓取帧(可达)双向传播可保证整段落在同一可达分支。
    print("\n逐帧 IK (从抓取帧向两侧传播):")
    q_now = robot.get_qpos().copy()
    _jidx = {j.get_name(): k for k, j in enumerate(joints)}
    torso_n = [f"torso_joint{i}" for i in range(1, 5)]
    print("  [调试] IK 时 qpos: torso=%s 左臂=%s" % (
        np.round([q_now[_jidx[n]] for n in torso_n if n in _jidx], 3).tolist(),
        np.round([q_now[_jidx[n]] for n in ARM_JOINTS], 3).tolist()))
    xs = np.full((N, 6), np.nan)
    x_g, c_g = solve_ik(robot, scene, target_pad[k_g], app_dirs[k_g], fin_dirs[k_g],
                        R1_LEFT_START.copy(), note=f"F{seg[k_g]}", x_ref=None, n_seeds=n_seeds,
                        max_iters=ik_max_iters)
    xs[k_g] = x_g
    print(f"  抓取帧 F{seg[k_g]} 解={np.round(x_g,3).tolist()} cost={c_g:.6f}")
    x_prev = x_g.copy()
    for k in range(k_g + 1, N):
        x_new, c_new = solve_ik(robot, scene, target_pad[k], app_dirs[k], fin_dirs[k],
                                x_prev, x_ref=x_prev, n_seeds=n_seeds,
                                max_iters=ik_max_iters)
        if c_new > 1.0:
            # 发散保护: cost>1.0 意味着解到了限位角 (位置误差 >1.6cm),
            # 保持上一帧解, 避免个别坏帧污染整条轨迹 (平滑后仍会扩散)。
            x_new = x_prev.copy()
        xs[k] = x_new
        x_prev = x_new
    x_prev = x_g.copy()
    for k in range(k_g - 1, -1, -1):
        x_new, c_new = solve_ik(robot, scene, target_pad[k], app_dirs[k], fin_dirs[k],
                                x_prev, x_ref=x_prev, n_seeds=n_seeds,
                                max_iters=ik_max_iters)
        if c_new > 1.0:
            x_new = x_prev.copy()
        xs[k] = x_new
        x_prev = x_new
    for k in range(0, N, 4):
        print(f"  F{seg[k]:3d} 解={np.round(xs[k],3).tolist()}")

    # 关节轨迹平滑: 逐帧独立 IK 的解会有小幅抖动 (冗余臂同解邻域的数值噪声),
    # 对整段关节角做零相位滑动平均并夹回限位, 使驱动更顺滑。
    xs_raw = xs.copy()
    xs = smooth_traj(xs, win=P["joint_win"], passes=P["joint_passes"])
    xs = np.clip(xs, IK_LIMITS[:, 0], IK_LIMITS[:, 1])
    # 运动学限幅平滑器: 滑动平均后再施加 速度/加速度/jerk 三级限幅 (顺滑且不失真)
    xs = rate_limit_traj(xs, P["max_qvel"], P["max_qacc"], P["max_qjerk"])
    xs = np.clip(xs, IK_LIMITS[:, 0], IK_LIMITS[:, 1])
    xj = np.abs(np.diff(xs, axis=0) - np.diff(xs_raw, axis=0)).max()
    print(f"  关节轨迹平滑+限幅: 逐帧最大关节抖动变化 {np.degrees(xj):.2f}deg "
          f"(平滑前逐帧 |Δq|max={np.degrees(np.abs(np.diff(xs_raw,axis=0)).max()):.2f}deg/frame, "
          f"平滑后 {np.degrees(np.abs(np.diff(xs,axis=0)).max()):.2f}deg/frame)")
    # 限位饱和诊断: 逐帧统计贴限位 (<0.05 rad) 的关节数 —— 解释"最后一帧没达到"
    n_sat = np.sum(np.minimum(np.abs(xs - IK_LIMITS[:, 0]), np.abs(xs - IK_LIMITS[:, 1])) < 0.05, axis=1)
    # 平滑前饱和统计 (区分"IK 本身发散"与"平滑/限幅引入")
    n_sat0 = np.sum(np.minimum(np.abs(xs_raw - IK_LIMITS[:, 0]), np.abs(xs_raw - IK_LIMITS[:, 1])) < 0.05, axis=1)
    print(f"  [限位诊断] 平滑前贴限位帧数 = {int((n_sat0 > 0).sum())}/{N} (平滑后 {int((n_sat > 0).sum())}/{N})")
    if np.any(n_sat > 0):
        sat_frames = [(int(seg[k]), int(n_sat[k]), np.round(xs[k], 2).tolist())
                      for k in range(N) if n_sat[k] > 0]
        print(f"  [限位诊断] {int((n_sat > 0).sum())}/{N} 帧有关节贴限位 (<0.05rad):")
        for f, c, q in sat_frames[:12]:
            print(f"    F{f}: {c} 个关节贴限位  解={q}")
        if len(sat_frames) > 12:
            print(f"    ... 共 {len(sat_frames)} 帧")
    else:
        print("  [限位诊断] 无关节贴限位")

    # 抓取帧做闭环精修。
    # 注意: 闭环用 drive_target 驱动整条手臂从起始位扫到抓取位, 路径必然穿过粉鸭
    # (实测把鸭推走 10.5cm)。故闭环期间把粉鸭临时移到安全高度, 闭环后放回并重新稳定。
    print(f"\n抓取帧 F{seg[k_g]} 闭环精修 (期间隐藏粉鸭):")
    duck_park = (duck_rb_center + np.array([0.0, 0.0, 0.6])).tolist()
    duck.set_pose(sapien.Pose(p=duck_park))
    step_sim(scene, robot, 40)
    xs[k_g] = ik_closed_loop(robot, scene, xs[k_g], target_pad[k_g], app_dirs[k_g],
                             fin_dirs[k_g], P["grip_open_q"], iters=3, settle=300)

    all_j = robot.get_active_joints()
    name2idx = {j.get_name(): k for k, j in enumerate(all_j)}
    sel0 = [name2idx[n] for n in ARM_JOINTS]

    def set_arm(xk, settle=60):
        # 必须同时设 drive_target: 仅 set_qpos 会被高刚度驱动立刻拉回旧 target
        # (闭环精修留下的抓取位), 手臂并未真正移开, 放回粉鸭时会被顶飞。
        q = robot.get_qpos().copy()
        for k, si in enumerate(sel0):
            q[si] = xk[k]
        for jn, n in enumerate(ARM_JOINTS):
            jmap[n].set_drive_target(float(xk[jn]))
        robot.set_qpos(q)
        step_sim(scene, robot, settle)

    # 先把手臂移回轨迹首帧高位, 再把粉鸭放回桌面 ——
    # 否则闭环精修后停在抓取位的夹爪会把放回的粉鸭顶飞 (实测被推走 ~8cm)。
    set_arm(xs[0])
    duck.set_pose(sapien.Pose(p=duck_rb_center.tolist()))
    step_sim(scene, robot, 120)
    duck_settled = duck.get_pose().p.copy()
    print(f"  粉鸭归位(手臂已移开) = {np.round(duck_settled,4).tolist()}")

    # ---- 相机 (正面 / 侧面 / 胸部第一视角 / 原始FPV) 仅用于视频 ----
    frames = []
    if collect_frames:
        W, H = 480, 360
        front_cam = scene.add_camera(name="r1_front", width=W, height=H, fovy=np.deg2rad(45), near=0.05, far=100)
        front_cam.entity.set_pose(look_at_pose([1.05, -1.02, 1.19], [0.0, -0.30, 0.95]))   # 拉近(原[1.55,-1.35,1.30])
        side_cam = scene.add_camera(name="r1_side", width=W, height=H, fovy=np.deg2rad(45), near=0.05, far=100)
        side_cam.entity.set_pose(look_at_pose([-1.05, -0.34, 1.16], [0.0, -0.30, 0.95]))   # 拉近(原[-1.45,-0.35,1.25])
        # 胸部第一视角: 装在躯干前上方, 朝前(+y)俯视工作区, 近似机器人 ego 视角
        chest_cam = scene.add_camera(name="r1_chest", width=W, height=H, fovy=np.deg2rad(60), near=0.05, far=100)
        chest_cam.entity.set_pose(look_at_pose([0.0, -0.34, 1.42], [0.0, -0.05, 0.80]))

        # ---- 原始 FPV 视角: 用 ras extrinsics (逐帧 w2c) 复现真实 ego 相机轨迹 ----
        # 链路 (已验证: extrinsics 世界系与 joints_ras_world 逐帧回投残差 0.0mm):
        #   C_world(fi) = -Rᵀt (相机中心), 视线 = R 第三行, up = -R 第二行 (w2c, ras_world 系)
        #   -> 以鸭为锚纯平移映射到 RoboTwin (与场景道具同规则)。
        #   注意: 不需要 Kabsch (那是相机系->世界系的平均变换, 乘在世界系朝向上会转两次)。
        RAS_DIR = os.path.dirname(GLB)
        Kint = np.loadtxt(os.path.join(RAS_DIR, "intrinsic.txt"))
        fpv_fovy = 2 * np.arctan2(Kint[1, 2], Kint[1, 1])   # 2*atan(cy/fy)

        # FPV 眼前偏移: ego 相机位于人头中心 (≈R1 头部位置), R1 大头部会挡住画面。
        # 真实相机也在脸前方数厘米, 故沿视线前移 FPV_FWD 把相机挪到头部前方出画面。
        FPV_FWD = 0.25

        def fpv_pose(fi):
            E = np.loadtxt(os.path.join(RAS_DIR, "extrinsics", f"{int(fi)}.txt")).reshape(4, 4)
            R = E[:3, :3]                         # w2c 旋转 (ras_world 系)
            C = -R.T @ E[:3, 3]                   # 相机中心 (ras_world 系)
            f_ax = R[2, :]                        # 相机 +z 在世界系 = 视线方向
            up = -R[1, :]                         # OpenCV +y 朝下
            l_ax = np.cross(up, f_ax); l_ax /= np.linalg.norm(l_ax)
            u_ax = np.cross(f_ax, l_ax)
            Cp = duck_settled + (C - duck_center_glb)   # 鸭锚纯平移映射
            M = np.eye(4)
            M[:3, :3] = np.stack([f_ax, l_ax, u_ax], axis=1)   # SAPIEN: +x前 +y左 +z上
            M[:3, 3] = Cp + f_ax * FPV_FWD
            return sapien.Pose(M)

        fpv_cam = scene.add_camera(name="r1_fpv", width=W, height=H, fovy=fpv_fovy, near=0.05, far=100)
        fpv_cam.entity.set_pose(fpv_pose(seg[0]))
        # FPV 自检: 相机到鸭的距离与视线夹角 (应 ~1m 内且朝向工作区)
        p0 = fpv_cam.entity.get_pose().p
        v0 = fpv_cam.entity.get_pose().to_transformation_matrix()[:3, 0]
        d0 = duck_settled - p0
        print(f"  FPV 自检: 相机={np.round(p0,3).tolist()} 距鸭={np.linalg.norm(d0):.3f}m "
              f"视线·到鸭={float(v0@d0/np.linalg.norm(d0)):.3f}")

        quad = (front_cam, side_cam, fpv_cam, chest_cam)

        def render_pair(k=0):
            scene.update_render()
            if fpv_cam is not None:
                fpv_cam.entity.set_pose(fpv_pose(seg[k]))
            for c in quad:
                c.take_picture()
            imgs = [(np.clip(c.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8)
                    for c in quad]
            top = np.concatenate(imgs[:2], axis=1)      # 正面 | 侧面
            bot = np.concatenate(imgs[2:4], axis=1)     # 原始FPV | 胸部
            return np.concatenate([top, bot], axis=0)

        # mano 参考轨迹可视化 (绿色小球): 叠加在场景中, GIF 每帧可见, 与 R1 末端直接对比
        for i in range(0, N, 2):
            add_marker(scene, mano_ref[i], (0.1, 0.95, 0.1), f"mano_trail_{i}", r=0.007)
        print(f"  mano 参考轨迹标记点 = {len(range(0, N, 2))} 个 (绿色)")

    # ---- 控制步上下文 (供 on_step 回调记录数据) ----
    g_link = robot.find_link_by_name(GRIP_LINK)
    ctx = {
        "scene": scene, "robot": robot, "duck": duck, "jmap": jmap,
        "g_link": g_link, "seg": seg, "xs": xs, "k_g": k_g,
        "k_close": k_close, "k_open": k_open,
        "target_pad": target_pad, "N": N, "ctrl_steps": CTRL_STEPS,
        "grip_open_q": P["grip_open_q"], "grip_hold_q": P["grip_hold_q"],
    }

    # ---- 逐帧驱动 + 记录 ----
    print("\n执行轨迹跟踪:")
    traj_err_p, traj_err_o, duck_trace, pad_trace = [], [], [], []
    for k, fi in enumerate(seg):
        # 夹爪时序: 张开 -> (抓取帧)闭合夹住 -> (放置帧)张开释放
        if k < k_close:
            gfrac = 0.0
        elif k < k_open:
            gfrac = min(1.0, (k - k_close) / (3.0 * ts))
        else:
            gfrac = max(0.0, 1.0 - (k - k_open) / (2.0 * ts))      # 快速张开释放
        for jn, n in enumerate(ARM_JOINTS):
            jmap[n].set_drive_target(float(xs[k, jn]))
        q = P["grip_open_q"] * (1 - gfrac) + P["grip_hold_q"] * gfrac
        for nm in GRIP_FINGERS:
            if nm in jmap:
                jmap[nm].set_drive_target(q)
        step_sim(scene, robot, CTRL_STEPS)
        M = g_link.get_pose().to_transformation_matrix()
        p_act = M[:3, 3] + M[:3, 0] * PAD_FWD
        R_act = M[:3, :3]
        e_p = np.linalg.norm(p_act - target_pad[k])
        # 朝向误差: 接近轴夹角 + 开合轴夹角
        ang_a = np.degrees(np.arccos(np.clip(R_act[:, 0].dot(app_dirs[k]), -1, 1)))
        ang_f = np.degrees(np.arccos(np.clip(abs(R_act[:, 1].dot(fin_dirs[k])), -1, 1)))
        traj_err_p.append(e_p); traj_err_o.append((ang_a + ang_f) / 2)
        duck_trace.append(duck.get_pose().p.copy())
        pad_trace.append(p_act)
        if on_step is not None:
            on_step(ctx, k, int(fi))
        if collect_frames and (k % 3 == 0 or k >= k_g - 2):
            frames.append(render_pair(k))

    traj_err_p = np.array(traj_err_p); traj_err_o = np.array(traj_err_o)
    duck_trace = np.array(duck_trace); pad_trace = np.array(pad_trace)
    print("\n粉鸭 xy 漂移逐帧诊断 (落定 = %s):" % np.round(duck_settled, 4).tolist())
    for k in range(0, N, 2):
        drift = np.linalg.norm(duck_trace[k, :2] - duck_settled[:2])
        print(f"  k={k:2d} F{seg[k]:3d} duck={np.round(duck_trace[k],4).tolist()} "
              f"pad={np.round(pad_trace[k],4).tolist()} 漂移={drift*100:.2f}cm")

    # 抬升/滑移 (从抓取帧起算)
    d0 = duck_trace[k_g]
    # 参考轨迹复现段 = 进入末端对中插入之前的纯跟随段
    m_rep = np.arange(N) < k_p
    m_ins = (np.arange(N) >= k_p) & (np.arange(N) < k_g)
    m_grasp = np.arange(N) >= k_g
    print("\n" + "=" * 66)
    print("① 参考轨迹复现误差 (纯跟随段 F%d-F%d, 位置三轴跟踪真实人手轨迹):" % (seg[0], seg[k_p - 1]))
    if m_rep.sum():
        ep = np.linalg.norm(pad_trace[m_rep] - target_pad[m_rep], axis=1)
        ep_xy = np.linalg.norm(pad_trace[m_rep, :2] - target_pad[m_rep, :2], axis=1)
        print(f"  位置误差(xyz): 中位={np.median(ep)*1000:.1f}mm  max={ep.max()*1000:.1f}mm")
        print(f"  位置误差(xy) : 中位={np.median(ep_xy)*1000:.1f}mm  max={ep_xy.max()*1000:.1f}mm")
        print(f"  姿态误差: 中位={np.median(traj_err_o[m_rep]):.1f}deg  max={traj_err_o[m_rep].max():.1f}deg")
    print(f"\n② 末端对中插入段 (F{seg[k_p]}-F{seg[k_g]}): 高位对中+垂直下探 (宽夹爪几何必需)")
    print(f"  位置误差: 中位={np.median(traj_err_p[m_ins])*1000:.1f}mm  max={traj_err_p[m_ins].max()*1000:.1f}mm")
    print(f"\n③ 抓取后跟随段 (F{seg[k_g]}-F{seg[-1]}): 跟随人手抬起搬运")
    print(f"  位置误差: 中位={np.median(traj_err_p[m_grasp])*1000:.1f}mm  max={traj_err_p[m_grasp].max()*1000:.1f}mm")
    print(f"  姿态误差: 中位={np.median(traj_err_o[m_grasp]):.1f}deg")
    if k_place is not None:
        m_carry = (np.arange(N) >= k_g) & (np.arange(N) < k_place)
        m_place = np.arange(N) >= k_place
        print(f"\n④ 放置修正段 (F{seg[k_place]}-F{seg[-1]}): 对中下放到盘子中心 + 抬离")
        print(f"  搬运子段 F{seg[k_g]}-F{seg[k_place-1]}: "
              f"中位={np.median(traj_err_p[m_carry])*1000:.1f}mm  max={traj_err_p[m_carry].max()*1000:.1f}mm")
        print(f"  放置子段: 中位={np.median(traj_err_p[m_place])*1000:.1f}mm  "
              f"max={traj_err_p[m_place].max()*1000:.1f}mm  (目标已是修正后的对中下放)")
        print(f"  末点距盘心 = {np.linalg.norm(duck_trace[-1,:2]-plate_rb[:2])*1000:.1f} mm")
        print(f"  末点是否在盘内 = {'是' if np.linalg.norm(duck_trace[-1,:2]-plate_rb[:2]) < np.linalg.norm(plate_ext[:2])/2 else '否'}"
              f" (盘半宽 {np.linalg.norm(plate_ext[:2])/2*100:.1f}cm)")
    print("\n粉鸭抓取结果:")
    print(f"  抓取帧位置 {np.round(d0,4).tolist()}  末态 {np.round(duck_trace[-1],4).tolist()}")
    lifted = duck_trace[:, 2].max() - d0[2]
    print(f"  抬升(max) = {lifted*100:.2f} cm   放置帧高度 = {duck_trace[k_open,2]:.4f} "
          f"(放置时相对桌面 {duck_trace[k_open,2]-TABLE_Z:.4f} m)")
    print(f"  水平位移 = {np.linalg.norm(duck_trace[-1,:2]-d0[:2])*100:.2f} cm")
    slip = np.linalg.norm(pad_trace[:, :] - duck_trace, axis=1)
    print(f"  pad↔粉鸭 滑移: 闭合帧={slip[k_close]*100:.2f}cm  张开帧={slip[k_open]*100:.2f}cm  "
          f"末={slip[-1]*100:.2f}cm")
    on_table = duck_trace[-1, 2] > TABLE_Z + 0.05
    print(f"  释放后末态离桌(>5cm): {'是' if on_table else '否'}  末态高度={duck_trace[-1,2]:.4f}"
          f"  (放置并释放应落回桌面)")

    # ---- mano 参考轨迹 vs R1 实际末端 (retargeting 保真度) ----
    ep_mano = np.linalg.norm(pad_trace - mano_ref, axis=1)
    print("\nmano 参考轨迹 vs R1 实际末端 (三维距离):")
    if m_rep.sum():
        print(f"  纯跟随段 F{seg[0]}-F{seg[k_p-1]}: 中位={np.median(ep_mano[m_rep])*1000:.1f}mm  "
              f"max={ep_mano[m_rep].max()*1000:.1f}mm")
    else:
        print("  纯跟随段: 无 (该配置下悬停起点 k_p=0, 整段均为对中插入)")
    print(f"  对中插入段 F{seg[k_p]}-F{seg[k_g]}: 中位={np.median(ep_mano[m_ins])*1000:.1f}mm  "
          f"max={ep_mano[m_ins].max()*1000:.1f}mm  (保证夹爪罩住物体的主动偏移)")
    print(f"  全程: 中位={np.median(ep_mano)*1000:.1f}mm  max={ep_mano.max()*1000:.1f}mm")

    # ---- R1 实际轨迹可视化 (蓝色小球) + mano/R1 对比图 ----
    if collect_frames:
        for i in range(0, N, 2):
            add_marker(scene, pad_trace[i], (0.1, 0.3, 1.0), f"r1_trail_{i}", r=0.007)
        scene.update_render()
        front_cam.take_picture()
        cmp = (np.clip(front_cam.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8)
        from PIL import Image as _I
        cmp_path = os.path.join(OUT_DIR, "mano_vs_r1_trajectory.png")
        _I.fromarray(cmp).save(cmp_path)
        print(f"  轨迹对比图 (绿=mano参考, 蓝=R1实际) -> {cmp_path}")

    # ---- 平滑度指标 (供轨迹优化器/agent 使用) ----
    dq = np.abs(np.diff(xs, axis=0))
    jerk = np.abs(np.diff(xs, n=3, axis=0))
    ee_step = np.linalg.norm(np.diff(pad_trace, axis=0), axis=1)
    smooth = {
        "max_dq_deg": float(np.degrees(dq.max())),
        "rms_dq_deg": float(np.degrees(np.sqrt((dq ** 2).mean()))),
        "max_jerk": float(jerk.max()),
        "rms_jerk": float(np.sqrt((jerk ** 2).mean())),
        "ee_max_step_mm": float(ee_step.max() * 1000),
        "n_frames": int(N),
    }
    print("\n平滑度指标 (关节/末端逐控制步):")
    print(f"  关节 max|Δq|={smooth['max_dq_deg']:.3f}deg  RMS|Δq|={smooth['rms_dq_deg']:.3f}deg  "
          f"RMS|jerk|={smooth['rms_jerk']:.4f} (rad/步^3)")
    print(f"  末端 max 逐帧位移={smooth['ee_max_step_mm']:.2f}mm  帧数={N}")

    # ---- 成功判据 (verifier) ----
    # follow_max: 抓取后跟随段末端最大跟踪误差 (IK 不可达冻结时骤增, 触发重搜)
    follow_max = float(np.linalg.norm(pad_trace[k_g:] - target_pad[k_g:], axis=1).max())
    # 放置目标存在时, 把"目标脚印/静置高度"传给 verifier —— 否则它只会认"落回桌面",
    # 在"把物体放到盘子上"时必然假阴性 (盘面比桌面高 ~2.5cm)。
    _tgt = None
    if k_place is not None and plate_rb is not None:
        _tgt = {"xy": [float(plate_rb[0]), float(plate_rb[1])],
                "ext": [float(plate_ext[0]), float(plate_ext[1]), float(plate_ext[2])],
                "rest_z": float(rest_z)}
    verdict = verify_grasp(duck_trace, duck_settled, pad_trace, k_g, k_close, k_open,
                           follow_max=follow_max, target=_tgt)
    print("\n" + "-" * 66)
    print("成功判据 (verifier): %s" % ("PASS" if verdict["pass"] else "FAIL"))
    for name, ok in verdict["checks"].items():
        print(f"  [{'x' if ok else ' '}] {name}")
    m = verdict["metrics"]
    print(f"  抬升={m['lift_m']*100:.2f}cm  抓前位移={m['knock_m']*100:.2f}cm  "
          f"夹持滑移={m['hold_slip_m']*100:.2f}cm  末态z={m['landed_z']:.4f}")
    if verdict["reasons"]:
        print(f"  失败原因: {verdict['reasons']}")

    return {
        "scene": scene, "robot": robot, "duck": duck, "ctx": ctx,
        "seg": seg, "xs": xs, "k_g": k_g, "target_pad": target_pad, "N": N,
        "frames": frames, "traj_err_p": traj_err_p, "traj_err_o": traj_err_o,
        "duck_trace": duck_trace, "pad_trace": pad_trace, "mano_ref": mano_ref,
        "duck_settled": duck_settled, "duck_center_glb": duck_center_glb,
        "verdict": verdict, "params": P, "k_close": k_close, "k_open": k_open,
        "duck_rb_center": duck_rb_center, "smooth": smooth,
        "k_place": k_place,
        "plate_rb": (plate_rb if k_place is not None else None),
        "plate_ext": (plate_ext if k_place is not None else None),
        "rest_z": (rest_z if k_place is not None else None),
    }


def main():
    ep = build_episode(on_step=None, collect_frames=True)
    frames = ep["frames"]
    k_g = ep["k_g"]; duck_trace = ep["duck_trace"]
    pad_trace = ep["pad_trace"]
    traj_err_p = ep["traj_err_p"]

    np.savez(os.path.join(OUT_DIR, "duck_track.npz"),
             target=ep["target_pad"], pad=pad_trace, duck=duck_trace,
             err_p=traj_err_p, err_o=ep["traj_err_o"], k_g=k_g)
    print(f"\n轨迹数据 -> {os.path.join(OUT_DIR, 'duck_track.npz')}")

    # ---- 保存视频 ----
    gif_path = os.path.join(OUT_DIR, "robotwin_r1_duck_grasp.gif")
    mp4_path = os.path.join(OUT_DIR, "robotwin_r1_duck_grasp.mp4")
    from PIL import Image
    pal = [Image.fromarray(f).quantize(colors=64, method=Image.MEDIANCUT) for f in frames]
    pal[0].save(gif_path, save_all=True, append_images=pal[1:],
                duration=80, loop=0, optimize=True, disposal=2)
    try:
        imageio.mimsave(mp4_path, frames, fps=12, macro_block_size=1)
    except Exception as e:
        print(f"mp4 skipped: {e}")
    print(f"\nframes={len(frames)} size={frames[0].shape}")
    print(f"GIF -> {gif_path} ({os.path.getsize(gif_path)/1e6:.2f} MB)")
    print(f"MP4 -> {mp4_path} ({os.path.getsize(mp4_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

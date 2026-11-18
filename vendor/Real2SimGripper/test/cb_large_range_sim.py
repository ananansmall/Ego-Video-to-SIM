#!/usr/bin/env python3
"""cb_large_range_sim.py — chairbottle 右手大范围移动 → 完整 R1 物理仿真渲染 (A 方案)。

任务脚本 (thin): 只负责加载本任务的目标位姿序列并驱动渲染。
通用逻辑 (底盘跟随 / 限位感知 IK / 相机) 全部在 large_range_track.py;
场景/物理/相机原语复用主管线 robotwin_r1_duck_grasp.py (完整 R1 + 地面 + PD)。

重定向: chairbottle 无 dex_retargeting 依赖; 用 SVD (cb_svd_retarget.py) 从 MANO 三点
(wrist/拇指尖/食指尖) 解世界系夹爪位姿。底盘偏置/手高对齐由 large_range_track.comfort_frame
从机器人工作空间自动推导, 无按任务硬编码。
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import sapien.core as sapien
import sapien.physx as sapienp
import imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robotwin_r1_duck_grasp as rg      # noqa: E402  主管线: 场景/物理/相机原语
import bp14_dual_arm_sim as bp            # noqa: E402  原语: fk_pad / resolve / se2_pose / _limits
import large_range_track as lrt           # noqa: E402  通用内核 (泛化层)
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

CB_TARGETS = "/path/to/workspace/_scratch/chairbottle/cb_right_targets.npz"
CB_HAWOR = "/path/to/project_data/mano_ws/output/mano/chairbottle/hawor/reconstruction/hawor_results_0_592.npz"
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")
# EF3R 世界 (y-up) → sim 世界 (z-up) 的轴变换 (与 ef3r_to_targets 同一常量)
M_EF3R_TO_SIM = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
# 大范围后端的产物统一写进 */large_range/ 子目录 (与 demo_output/results 两个 README
# 的"按项目分子目录"约定一致, 此前写到根目录造成产物散落)。目录自动创建。
LR_RESULTS_DIR = os.path.join(RESULTS_DIR, "large_range")
LR_DEMO_DIR = os.path.join(os.path.dirname(HERE), "demo_output", "large_range")
BASE_SMOOTH_WIN = 9               # 底盘水平轨迹平滑窗口 (0=不平滑)
# ⚠ 以下 EGO_* 常量已停用 (2026-10-24): 第一视角改为"机器人机载 FPV"
#   (lrt.fpv_cam_pose, 见 update_cams), 因为数据集 ego 相机的原点 = 人的眼睛,
#   与机器人底盘位置重合 -> 画面被自己的头/手臂占满, 且 SAPIEN 3.0.0b1 没有
#   按相机隐藏 actor 的 API。保留定义仅为历史参考, 勿再使用。
# EGO_BACK, EGO_UP = 0.30, 0.20     # 沿光轴后拉 / 竖直抬高
# EGO_MIN_R = 0.75                  # 相机与底盘水平距离小于此值 -> 抬到头顶俯视
# EGO_HEAD_Z = 1.55                 # 机器人头顶高度 (实际用 cf["head_z"], 由真机标出)
# EGO_PITCH_DROP = 0.22             # 近距第一视角视线下俯量


def _build_scene():
    """完整 R1 + 地面 + 远墙 (无静止桌子: 大范围移动任务)。"""
    engine = sapien.Engine()
    scene = rg.setup_scene(engine)
    rg.create_box(scene, sapien.Pose(p=[0, -5.0, 1.5]), half_size=[8, 0.3, 1.5],
                  color=(1, 0.9, 0.9), name="wall", is_static=True)
    # Absolute mesh paths let independent runs keep private URDFs.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="r1-lr-") as tmp:
        tmp_urdf = os.path.join(tmp, "r1_abs.urdf")
        rg.make_r1_urdf_abs(bp.R1_URDF, tmp_urdf)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(tmp_urdf)
    robot.set_root_pose(sapien.Pose(p=rg.R1_BASE_P, q=rg.R1_BASE_Q))
    return scene, robot


def _dump_metrics(payload):
    """把本轮指标+配置落盘到 results/large_range/<task>.json, 供后续改动对比。

    此前指标只在终端打印、不留档 -> 每次改动无法直接对比, 只能重跑。"""
    os.makedirs(LR_RESULTS_DIR, exist_ok=True)
    path = os.path.join(LR_RESULTS_DIR, f"{payload.get('task', 'run')}.json")
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    print(f"  指标 -> {path}")
    return path


def _plot_follow(path, tgt_p, ik_p, phys_p, base_xy, pos_err, ori_err,
                 phys_err, phys_ori, transit, name="chairbottle"):
    """轨迹 vs 跟随 对比图: 3D 路径 / 俯视路径 / 逐帧位置误差 / 姿态+底盘速度。

    目的: 一眼看清「目标怎么走、IK 解怎么跟、物理执行又落后多少」,
    以及误差集中在哪些帧 (灰底 = 分支过渡窗口)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(tgt_p)
    f = np.arange(N)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot(tgt_p[:, 0], tgt_p[:, 1], tgt_p[:, 2], "-", lw=2.4, c="tab:blue", label="target (hand)")
    ax.plot(ik_p[:, 0], ik_p[:, 1], ik_p[:, 2], "-", lw=1.4, c="tab:orange", label="IK solution")
    ax.plot(phys_p[:, 0], phys_p[:, 1], phys_p[:, 2], "-", lw=1.2, c="tab:green", label="physics follow")
    ax.plot(base_xy[:, 0], base_xy[:, 1], np.zeros(N), "-", lw=1.0, c="0.6", label="base path")
    ax.scatter(tgt_p[0, 0], tgt_p[0, 1], tgt_p[0, 2], c="k", s=30, marker="o")
    ax.scatter(tgt_p[-1, 0], tgt_p[-1, 1], tgt_p[-1, 2], c="k", s=30, marker="X")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("3D end-effector trajectory")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(tgt_p[:, 0], tgt_p[:, 1], "-", lw=2.4, c="tab:blue", label="target")
    ax.plot(ik_p[:, 0], ik_p[:, 1], "-", lw=1.4, c="tab:orange", label="IK")
    ax.plot(phys_p[:, 0], phys_p[:, 1], "-", lw=1.2, c="tab:green", label="physics")
    ax.plot(base_xy[:, 0], base_xy[:, 1], "--", lw=1.0, c="0.6", label="base")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top view (x-y)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for i in np.where(transit)[0]:
        ax.axvspan(i - 0.5, i + 0.5, color="0.85", lw=0)
    ax.plot(f, pos_err, lw=1.3, c="tab:orange", label=f"IK  mean {pos_err.mean():.1f}mm")
    ax.plot(f, phys_err, lw=1.3, c="tab:green", label=f"physics mean {phys_err.mean():.1f}mm")
    ax.set_xlabel("frame"); ax.set_ylabel("position error [mm]")
    ax.set_title("Per-frame position error (grey = branch transition)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(f, ori_err, lw=1.2, c="tab:orange", label=f"IK ori {ori_err.mean():.2f}deg")
    ax.plot(f, phys_ori, lw=1.2, c="tab:green", label=f"physics ori {phys_ori.mean():.2f}deg")
    ax.set_xlabel("frame"); ax.set_ylabel("orientation error [deg]")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax2 = ax.twinx()
    bstep = np.r_[0.0, np.linalg.norm(np.diff(base_xy, axis=0), axis=1)] * 1000
    ax2.plot(f, bstep, lw=1.0, c="0.5", alpha=0.8, label="base step [mm/frame]")
    ax2.set_ylabel("base step [mm/frame]", color="0.4")
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Orientation error & base speed")

    fig.suptitle(f"{name} large-range: trajectory tracking ({N} frames)", fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  轨迹跟随对比图 -> {path}")
    return path


def _ef3r_sim_rotation(scene_cloud=None, object_npz=None, targets=None):
    """EF3R 帧 → sim 帧的旋转: 有 gravity_align 缓存/可估计则用 M'=M@R_g (重力对齐),
    否则回退旧 M_EF3R_TO_SIM (假定 EF3R y-up=竖直 —— 456483158 实测该假设差 ~66°)。
    ef3r 目录从 object/场景点云路径反推 (<ef3r-dir>/object_instance*.npz,
    <ef3r-dir>/pointcloud/scene_cloud.npz)。"""
    import gravity_align as _ga
    ef_dir = None
    if object_npz and os.path.exists(object_npz):
        ef_dir = os.path.dirname(object_npz)
    elif scene_cloud and os.path.exists(scene_cloud):
        ef_dir = os.path.dirname(os.path.dirname(scene_cloud))
    elif targets and os.path.exists(targets):
        ef_dir = os.path.dirname(targets)   # ef3r_right_targets.npz 就在 ef3r 目录里
    if ef_dir:
        try:
            M = _ga.sim_rotation(ef_dir)
            if M is not None:
                return M, ef_dir
        except Exception:
            pass
    return M_EF3R_TO_SIM.copy(), None


def main(stride=1, ik_only=False, base_smooth=None, wbc=True, torso_wbc=False,
         w_base=None, base_range=None, w_ori=None, grip_mode="data",
         object_npz=None, grip_grasp=None, object_physics=False, scene_cloud=None,
         use_yaw=True, smooth=False, w_smooth=2.0,
         post_win=11, post_base_step=None, post_arm_step=None, re_resolve=False,
         w_cont=3.0, w_bcont=None, posture_weight=0.0, joint_step_max=180.0,
         finger_force_limit=None, table_collision=True,
         torso_gain=1.0, wrist_gain=1.0, base_ref_source="wrist",
         depenetration_vel=None, base_interp=False, legacy_physics=True,
         obstacles=None, w_obs=0.02, env_boxes=None, obs_margin=0.02,
         base_step_max=None, spread_max=25.0, arm_beam=0,
         limit_weight=1.0, limit_margin=20.0, phys_only=False, phys_steps=40,
         clock_mode="legacy",
         stiff_scale=1.0, n_local=2, n_rand=1, max_nfev=150,
         targets=CB_TARGETS, name="chairbottle", torso_pitch_deg=None,
         joint_reserve_deg=0.0, base_lead_s=0.0, torso_pitch_guard_deg=0.0,
         handoff_freeze_frames=0, ring_place=False):
    if not np.isfinite([joint_reserve_deg, base_lead_s]).all() or min(joint_reserve_deg, base_lead_s) < 0:
        raise ValueError("joint reserve and base lead time must be finite and nonnegative")
    if torso_pitch_deg is not None:
        torso_pitch_deg = np.broadcast_to(np.asarray(torso_pitch_deg, float), (3,))
        if (not np.isfinite(torso_pitch_guard_deg) or torso_pitch_guard_deg < 0
                or not np.all((torso_pitch_deg > torso_pitch_guard_deg) & (torso_pitch_deg < 90))
                or not (wbc and torso_wbc and use_yaw)):
            raise ValueError("R1 torso pitch bound requires yaw-enabled torso WBC and 0 < pitch < 90 degrees")
        if arm_beam or smooth or re_resolve:
            raise ValueError("torso pitch bound requires the joint base/arm solve without legacy re-solves")
    elif joint_reserve_deg or base_lead_s or torso_pitch_guard_deg:
        raise ValueError("joint reserve and base lead require a torso pitch bound")
    base_smooth = BASE_SMOOTH_WIN if base_smooth is None else base_smooth
    # ---- WBC 参数: 默认 auto (稳健全局基准, 所有轨迹共用), 显式传入则覆盖 ----
    _manual = any(v is not None for v in (w_base, w_bcont, base_range))
    autop = lrt.auto_wbc_params(w_base=w_base, w_bcont=w_bcont, base_range=base_range)
    w_base, w_bcont, base_range = autop["w_base"], autop["w_bcont"], autop["base_range"]
    # w_ori: 姿态残差权重, 默认 10 (姿态保真优先)。调小换位置精度+关节裕度:
    # 实测 (2026-10-25, ef3r456/chairbottle): 1.0 时 IK mean 12.4→5.67 / 6.66→4.61mm,
    # <5° 裕度帧 115→30 / 85→4, 代价姿态 0.12→7.72° / 0.08→6.17° (远在平行夹爪
    # 抓取容差 ±10~15° 内)。权衡曲线见 docs/10_控制算法与跟随误差.md。
    w_ori = 10.0 if w_ori is None else float(w_ori)
    print(f"  WBC 参数 ({'显式覆盖' if _manual else 'auto 稳健基准'}): "
          f"w_base={w_base} w_bcont={w_bcont} base_range={base_range} w_ori={w_ori}")
    d = np.load(targets)
    for _k in ("pos", "rot", "grip"):
        if _k not in d.files:
            raise SystemExit(f"❌ targets npz 缺少必需字段 '{_k}': {targets}")
    # ---- EF3R 帧 → sim 帧: 重力对齐 M' (gravity_align; 无数据则回退旧 M) ----
    # 物体/台面/场景云的 geo 对齐与 targets npz 必须用同一 M' —— 转换器已用
    # M_sim 写 targets, 这里从 object/场景路径反推 ef3r 目录加载同一旋转。
    M_EF3R, _ga_dir = _ef3r_sim_rotation(scene_cloud=scene_cloud, object_npz=object_npz,
                                         targets=targets)
    if _ga_dir:
        print(f"  EF3R→sim 旋转: 重力对齐 M_sim (gravity_align: {os.path.basename(_ga_dir)})")
    else:
        print("  EF3R→sim 旋转: 旧 M_EF3R_TO_SIM (无 gravity 数据, 假定 EF3R y-up=竖直)")
    pos0, rot0, grip0 = d["pos"], d["rot"], d["grip"]
    Nfull = len(pos0)
    # 目标位姿清洗: 修复上游重定向/SVD 产生的离群跳变帧 (如 ~180° 姿态翻转),
    # 否则 IK 会忠实跟踪跳变 -> 机械臂分支翻转 -> 视觉上"不平滑"。根因修复。
    pos0, rot0, bad = lrt.clean_pose_sequence(pos0, rot0)
    print(f"{name} 右手大范围移动: 原始 {Nfull} 帧 (stride={stride}); "
          f"目标清洗修复离群帧 {len(bad)} 个 {bad[:10]}{'...' if len(bad) > 10 else ''}")

    scene, robot = _build_scene()
    joints = robot.get_active_joints()
    jmap = {j.get_name(): j for j in joints}
    # 腕部相机 link 免碰撞: 大物体 (实测 ee1 玻璃容量瓶 122mm) 出生在两指中点时
    # 后部会顶进 realsense 相机 link, 动力学接触把臂持续顶飞 (pad 误差累积至
    # 1.7m, 接触探针实锤 right_realsense_link+ef3r_object 每帧都在)。相机是
    # 传感器支架无碰撞意义; 指爪/物体/台面接触不受影响。
    if not legacy_physics:
        for _ln in ("left_realsense_link", "right_realsense_link"):
            _rl = robot.find_link_by_name(_ln)
            if _rl is not None:
                for _comp in _rl.entity.components:
                    for _sh in getattr(_comp, "get_collision_shapes", list)():
                        _sh.set_collision_groups([0, 0, 0, 0])
                print(f"  相机 link 免碰撞: {_ln}")
    nm2i = {j.get_name(): k for k, j in enumerate(joints)}
    Rsel = [nm2i[f"right_arm_joint{i+1}"] for i in range(6)]
    limR = bp._limits("right")
    # 躯干升降进 WBC (9→13 DOF): torso_joint1-4 (全 revolute) 加入决策变量,
    # 解决"任务手高跨度 > 臂工作空间"的饱和 (chairbottle 实测 1.82m > 1.64m,
    # WBC 277 帧饱和 300mm)。sel 顺序 = [臂×6, 躯干×4] —— 下游 arm[:,:6]
    # 消费端全部不变; 躯干 = arm[:,6:10]。posture 正则把躯干拉向限位中点
    # (≈直立), 只在臂不可达时让位 (软层级, 标准 WBC 实践)。
    _torso_joints = [f"torso_joint{i+1}" for i in range(4)]
    if torso_wbc:
        Rsel = Rsel + [nm2i[nm] for nm in _torso_joints]
        _tlim = np.array([jmap[nm].get_limits()[0] for nm in _torso_joints])
        limR = np.vstack([limR, _tlim])
        _q_pref_arm = 0.5 * limR[:6, 0] + 0.5 * limR[:6, 1]
        _q_neutral = np.concatenate([_q_pref_arm, np.zeros(4)])
        print(f"  躯干进 WBC: 决策变量 9→{3+len(Rsel)} DOF (torso 4 + 臂 6), "
              f"躯干中性=直立(0), posture={max(posture_weight, 1.0):.1f}")
    side = "right"
    reserve = np.radians(joint_reserve_deg)
    plan_lim = limR + [reserve, -reserve] if reserve else limR
    if np.any(plan_lim[:, 0] >= plan_lim[:, 1]):
        raise ValueError("joint reserve leaves an empty planning interval")

    # ---- 泛化: 从机器人工作空间自动推导底盘偏置 + 手高对齐中心 ----
    cf = lrt.comfort_frame(robot, side, Rsel, limR, bp._gripper_pad_fk)
    print(f"  舒适帧(自动): base_off={np.round(cf['base_off'],3)}  z_center={cf['z_center']:.3f}  "
          f"ws_z=({cf['ws_z'][0]:.2f},{cf['ws_z'][1]:.2f})")

    # 数据集真实 ego 相机 (SLAM 世界系, OpenCV 约定 z=前/y=下); 与手部同步做高度对齐平移,
    # 保持"相机-手"相对关系 -> 第一视角即人手在真实世界看到的画面。
    # 优先用 targets 里**已做 SLAM 段对齐**的相机外参 (与手部同一连续世界系); 无则回退原文件。
    if "R_c2w" in d.files and "t_c2w" in d.files:
        R_c2w, t_c2w = d["R_c2w"].copy(), d["t_c2w"].copy()
    else:
        _hz = np.load(CB_HAWOR)
        R_c2w, t_c2w = _hz["R_c2w"].copy(), _hz["t_c2w"].copy()
    base_ref = np.asarray(rg.R1_BASE_P, np.float64)[:2]

    # ---- 世界系整体旋转: 让作业点落在机器人正前方 ----
    # 问题: HaWoR/SLAM 世界系的朝向与机器人无关。若手恰好在机器人侧面 (实测 seq456:
    # 手在 +x 而机器人朝 +y), 腕部要扭 ~90° 才能捏住 -> j6 顶死限位, 第一视角也是
    # "侧脸看手"。把世界绕 z 旋转, 是纯坐标约定变换, 不改变任何几何关系。
    # 目标方位角取 comfort_frame 的机器人前方 (+y), 这样 cf 的 base_off 无需改动。
    _dxy = pos0[:, :2].mean(axis=0) - base_ref
    _az = np.arctan2(pos0[:, 1] - base_ref[1], pos0[:, 0] - base_ref[0])
    _az_span = float(np.degrees(np.ptp(np.unwrap(_az))))
    _yaw0 = (float(np.arctan2(_dxy[1], _dxy[0])) - np.pi / 2
             if np.linalg.norm(_dxy) > 0.05 else 0.0)
    # 只在「作业点方位角基本固定」时旋转 (原地/近直线任务)。绕圈任务 (chairbottle:
    # 方位角跨 200°+) 的平角均值没有意义, 旋转会把整段轨迹拧到不可达区 -> 误差爆炸。
    if _az_span > 30.0:
        _yaw0 = 0.0
    _Rz = np.eye(3)
    if abs(_yaw0) > np.radians(5.0):
        _c, _s = np.cos(-_yaw0), np.sin(-_yaw0)
        _Rz = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
        print(f"  世界系旋转 {-np.degrees(_yaw0):+.1f}° (作业点方位角归零, 机器人正对手)")
        pos0 = pos0 @ _Rz.T
        rot0 = np.einsum("ij,njk,kl->nil", _Rz, rot0, _Rz.T)
        R_c2w = np.einsum("ij,njk->nik", _Rz, R_c2w)
        t_c2w = t_c2w @ _Rz.T

    # 高度对齐 + 底盘跟随 均在**全分辨率**下完成, 之后再下采样
    # (否则先下采样再平滑会把底盘轨迹抹平, 导致手臂够不到)
    # 操作高度 = 标准桌面 0.75m + 抓取高度 0.2m = 0.95m (R1 往下抓, 与原 duck
    # 任务一致) —— 旧 z_center=1.409 把手举到 1.4m → 桌面浮到 1.2m+, 机器人
    # "往上抓" (用户实测: 场景悬浮、应往下抓)。重力对齐链 (有场景云做地板
    # 参照) 用 0.95; 无场景云的链保持 cf 标定 (向后兼容)。
    # body 参考源 (mocap 路径): 交互面对齐 0.85m (R1 肩 0.9m 的灵巧区) ——
    # cf["z_center"]=1.287 是鸭子时代标定, 对 mocap 行走任务会把腕心顶到
    # 肩上 0.4m (边界伸展, 雅可比退化)
    _z_center = (0.95 if (_ga_dir is not None and scene_cloud
                          and os.path.exists(scene_cloud)) else
                 (0.85 if base_ref_source == "body" else cf["z_center"]))
    pos_f, z_shift = lrt.fit_height(pos0, _z_center, cf["ws_z"])
    t_c2w[:, 2] += z_shift
    if abs(z_shift) > 0:
        print(f"  高度对齐: 操作高度 {_z_center:.2f}m (标准桌面 0.75+抓取 0.2), "
              f"平移 {z_shift:+.3f}m")

    idx = np.arange(0, Nfull, stride)

    # ---- 逐帧底盘朝向参考: 跟随手掌 forward 的水平方位角 ----
    # 旧逻辑是"全局单一 yaw"且方位角跨度 >30° 就整体弃用(_yaw0=0)。绕圈任务里
    # 人绕作业点走, 手掌朝向随身体转动 -> 机器人 yaw 不跟 -> 腕被拧死 (j2-j5 顶限位)。
    # 信号源为什么不能用「手相对底盘的方位角」: follow 模式下 base = hand - base_off
    # (固定世界偏移), 该方位角恒等于 atan2(base_off) —— 常数 (实测 chairbottle 跨度
    # 0.0°, 减均值后仅剩平滑残差 ~3°), 对求解零贡献。真正把腕拧死的是手掌朝向:
    # chairbottle 手掌 forward 方位角扫过 261° (逐帧 p95 4°, 可被平滑+限速跟住),
    # yaw 跟住它后腕只吸收残差。seq456 手掌仅扫 19° —— 腕部自己转得过来, 见下方门控。
    # 锚定: **不减任何常数** (既不减均值也不减首帧)。ref_yaw 必须是手掌 forward 的
    # 绝对方位角 - 90°: 目标姿态 rot0 就在世界系里, 夹爪 x 轴要对准它, 底盘 yaw 也必须
    # 在同一绝对基准上对准 (实测: 数据手掌 x 指向 ~157-209°, 而 yaw=0 前方是 +y=90°,
    # 减均值/减首帧都会给整条序列强加 ~65-88° 的恒定腕部扭转 -> j2-j5 全程顶死,
    # mean 274mm; 绝对对准后逐帧臂 IK pos ≤66mm / ori ≤0.8°, 无贴限位)。
    _pfwd = np.einsum("nij,j->ni", rot0, np.array([1.0, 0.0, 0.0]))
    _az_h = np.unwrap(np.arctan2(_pfwd[:, 1], _pfwd[:, 0]))   # 解卷绕保连续
    _palm_span = float(np.degrees(np.ptp(_az_h)))
    # 门控 (按手掌扫掠, 不是旧的"位置方位角跨度"): <60° 时腕部自己转得过来
    # (j6 ±165°), 走旧路径 ref_yaw≡0 —— 此时 track_base_se2(yaw=0) 精确退化为
    # 固定偏置跟随, 保留 seq456 已验证的 ~1mm 标定舒适位姿精度 (实测: 不门控时
    # seq456 mean 1.0→8.2mm 退化, 因为底盘转到 +30° 丢掉了标定位姿)。≥60°
    # (chairbottle 260°) 走逐帧朝向参考 + SE(2) 跟随。
    if _palm_span < 60.0:
        ref_yaw_full = np.zeros(Nfull)
        _mode = "固定朝向(旧路径, 手掌扫掠<60°腕部自转)"
    else:
        ref_yaw_full = _az_h - np.pi / 2.0   # yaw=0 时机器人前方 = 世界 +y (R1_BASE_Q=+90°)
        # 锚定: **不减任何常数**。ref_yaw 必须是手掌 forward 的绝对方位角 - 90°:
        # 目标姿态 rot0 在世界系里, 夹爪 x 轴对准它, 底盘 yaw 也必须同基准对准
        # (实测: 减均值/减首帧都给整条序列强加 ~65-88° 恒定腕扭 -> 全程顶死;
        #  绝对对准后逐帧臂 IK pos ≤66mm / ori ≤0.8°)。
        ref_yaw_full = rg.smooth_traj(ref_yaw_full.reshape(-1, 1),
                                      max(3, int(base_smooth)), 2).ravel()
        ref_yaw_full = lrt.clip_step(ref_yaw_full, np.radians(3.0))
        _mode = "逐帧手掌朝向 + SE(2) 跟随"
    ref_yaw = ref_yaw_full[idx]
    print(f"  朝向参考({_mode}): {np.degrees(ref_yaw).min():+.1f}deg ~ "
          f"{np.degrees(ref_yaw).max():+.1f}deg  幅度 {np.degrees(np.ptp(ref_yaw)):.1f}deg"
          f"  (手掌方位角跨度 {_palm_span:.0f}deg;"
          f" 位置方位角跨度 {_az_span:.0f}deg/follow模式下为常数)")

    # 底盘跟随: 偏置随朝向参考旋转 (SE(2))。若偏置仍是固定世界向量, 底盘转过 ~90°
    # 后手就到了机器人侧后方, 手臂不可达 (实测 mean 274mm); 旋转偏置让手始终在机体
    # 前方, 与人行走时手在身前的几何一致。track_base(固定偏置) 仅适用于朝向不变任务。
    # 底盘参考源: wrist = 手腕 xy (旧行为, 向后兼容); body = 身体锚点轨迹
    # (t_c2w: EF3R=相机头 / mocap=骨盆) —— 行走任务中手腕随步态摆动, 跟手腕
    # 会让底盘偏离身体锚点 0.4-0.7m → 手臂全程最大伸展 (雅可比退化, 关节噪声
    # 放大成百 mm pad 误差, 实测 ee5_mocap)。body 源 = 底盘走身体轨迹。
    _ref_src = pos_f[:, :2]
    if base_ref_source == "body" and np.abs(t_c2w[:, :2]).max() > 0.1:
        _ref_src = t_c2w[:, :2]
        print(f"  底盘参考源: body (身体锚点轨迹, 覆盖 "
              f"{np.linalg.norm(np.diff(_ref_src, axis=0), axis=1).sum():.1f}m)")
    base_full, _, _ = lrt.track_base_se2(_ref_src, ref_yaw_full, cf["base_off"],
                                         base_ref, smooth_win=base_smooth,
                                         smooth_fn=rg.smooth_traj)

    _camera_base = base_full.copy()
    _camera_pad = pos_f + rg.PAD_FWD * _pfwd
    if ring_place:
        if not object_npz or not object_physics or stride != 1:
            raise ValueError("ring place requires physical object and stride=1")
        with np.load(object_npz) as ring:
            if "ring_segment_count" not in ring or "ring_radius_m" not in ring:
                raise ValueError("ring place requires a calibrated ring contract")
            ring_surface = ring["surface_place_V"] @ M_EF3R.T @ _Rz.T
            ring_center = ring_surface[:, :2].mean(0)
            ring_top = float(ring_surface[:, 2].max() + z_shift)
            h = np.asarray(ring["pca_ext"], float)/2
            h[0] = 0.040 if h[0] > 0.050 else h[0]
            h[1] = 0.050 if h[1] > 0.052 else h[1]
            labels = d["grip_states"]
            kh, kp, kr = [int(np.flatnonzero(labels == i)[0]) for i in (3, 4, 5)]
            release_basis = np.column_stack((rot0[kr, :, 0],
                np.cross(rot0[kr, :, 1], rot0[kr, :, 0]), rot0[kr, :, 1]))
            half_z = float(np.abs(release_basis[2]) @ h)
            # The conservative radial envelope uses the same collider as the runtime.
            pad0 = pos_f + rg.PAD_FWD * _pfwd
            pos_f = lrt.ring_approach(pad0, kh, kp, kr, ring_center, ring_top,
                float(ring["ring_radius_m"]), [np.linalg.norm(h), 0, half_z],
                carry_half_z=float(np.linalg.norm(h))) - rg.PAD_FWD*_pfwd
            print(f"  Ring place plan: center {ring_center.round(4)}, top {ring_top:.4f}, "
                  f"PLACE F{kp}, release request F{kr}")

    if base_lead_s:
        states = d["grip_states"] if "grip_states" in d.files else None
        if states is None or states.shape != (Nfull,) or not np.any(states >= 2):
            raise ValueError("base lead requires a per-frame grasp state")
        lead_frames = max(1, round(base_lead_s * float(d.get("fps", 30.0))))
        reach_xy, _, _ = lrt.track_base_se2(
            pos_f[:, :2] + rg.PAD_FWD * _pfwd[:, :2], ref_yaw_full, cf["base_off"],
            base_ref, smooth_win=base_smooth, smooth_fn=rg.smooth_traj)
        base_full = lrt.lead_base_reference(base_full, reach_xy, states, lead_frames)

    pos, rot, grip = pos_f[idx], rot0[idx], grip0[idx]
    R_c2w, t_c2w = R_c2w[idx], t_c2w[idx]
    base_xy = base_full[idx]
    N = len(idx)
    dx = base_xy[:, 0] - base_ref[0]
    dy = base_xy[:, 1] - base_ref[1]
    byaw = np.zeros(N)
    print(f"  → 采样后 {N} 帧")
    print(f"  手高对齐: 平移 {z_shift:+.3f}m -> 手高 {pos[:,2].min():.2f}~{pos[:,2].max():.2f}m")
    for a, nm in enumerate("xyz"):
        print(f"  世界目标 {nm}: {pos[:,a].min():.2f}~{pos[:,a].max():.2f} m (幅度 {np.ptp(pos[:,a]):.2f})")
    step = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
    # 底盘逐帧步长上限: 由**数据本身速度**决定, 不手工按轨迹调参。
    # 参考轨迹经过平滑 (窗 9), 快速段必然落后于手; WBC 需要"追赶余量"把它补回来,
    # 否则手臂硬够 -> 贴限位换分支 -> 摊平出几百 mm 的过渡误差。实测 chairbottle:
    # 上限 104mm(p99×2) 时 mean 25.6mm, 115mm 时 11.5mm, 即余量 ≈ 参考步长的 2 倍
    # 才够 (平滑窗把多帧位移摊平, 峰值需 2~2.5×)。慢速序列 (seq456: 参考 max 8.6mm)
    # 由 50mm 下限兜住, 同时压掉 WBC 偶发的 72mm 单帧跳。
    if base_step_max is None:
        base_step_max = max(0.05, float(step.max()) * 2.5)
    print(f"  底盘步长上限 (自动): {base_step_max*1000:.0f}mm/帧 "
          f"(参考 p95 {np.percentile(step,95)*1000:.0f} / max {step.max()*1000:.0f}mm)")
    print(f"  底盘轨迹: x {dx.min():.2f}~{dx.max():.2f}  y {dy.min():.2f}~{dy.max():.2f} m "
          f"(行程 {step.sum():.2f} m, 平滑窗 {base_smooth})")
    print(f"  底盘步长 max {step.max()*1000:.1f}mm/帧  p95 {np.percentile(step,95)*1000:.1f}mm")
    dx0, dy0 = dx.copy(), dy.copy()          # 启发式跟随参考 (WBC 的正则中心)

    # 世界系 pad 目标
    robot.set_root_pose(sapien.Pose(p=rg.R1_BASE_P, q=rg.R1_BASE_Q))
    pad_p = pos + np.einsum("nij,j->ni", rot, np.array([rg.PAD_FWD, 0.0, 0.0]))
    # 地板守卫: 目标轨迹最低点不得低于地面 +3cm —— 地下目标 = 物理臂压进地面
    # 碰撞体被持续顶飞 (实测 ee1 最低 -0.31m → pad 误差累积 1.7m, follow 模式
    # 无此问题因 kinematic 臂不受地面力)。全链 (目标/物体/台面) 统一平移。
    _z_min = float(min(pad_p[:, 2].min(), pos_f[:, 2].min()))
    if _z_min < 0.03 and not legacy_physics:
        _dz = 0.03 - _z_min
        pad_p[:, 2] += _dz
        _camera_pad[:, 2] += _dz
        pos_f[:, 2] += _dz
        t_c2w[:, 2] += _dz
        print(f"  地板守卫: 腕/pad 最低 z {_z_min:.3f}m < 0.03, 全链平移 {_dz:+.3f}m")
    tgt = [(pad_p[k], rot[k]) for k in range(N)]

    # ---- 求解: 全身 IK (WBC, 底盘+臂联合) 或 解耦 (旧方案) ----
    _t_solve = time.time()
    if wbc:
        print("  全身 IK (WBC: 底盘 x,y,yaw + 臂 联合求解)...")
        base_xyz, arm, pos_err, ori_err = lrt.solve_wbc_track(
            robot, side, Rsel, plan_lim, dx, dy, tgt, pad_p, bp.se2_pose, bp._gripper_pad_fk,
            w_base=w_base, base_range=base_range, use_yaw=use_yaw,
            obstacles=obstacles, w_obs=w_obs, obs_margin=obs_margin,
            w_cont=w_cont, w_bcont=w_bcont,
            w_posture=(posture_weight if posture_weight > 0 else
                       (1.0 if torso_wbc else 0.0)), joint_step_max=joint_step_max,
            q_neutral=(_q_neutral if torso_wbc else None),
            base_step_max=base_step_max,
            w_limit=limit_weight, limit_margin_deg=limit_margin,
            w_ori=w_ori,
            n_local=n_local, n_rand=n_rand, max_nfev=max_nfev,
            ref_yaw=ref_yaw,
            torso_pitch=(np.radians(torso_pitch_deg - torso_pitch_guard_deg)
                         if torso_pitch_deg is not None else None))
        dx, dy, byaw = base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2]
        base_xy = base_ref + np.stack([dx, dy], axis=1)
        step0 = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
        j0 = bp.traj_metrics(arm)[1]
        print(f"  [WBC 逐帧] 底盘步长 max {step0.max()*1000:.0f}mm p95 {np.percentile(step0,95)*1000:.0f}mm "
              f"| 臂 jerk {j0:.2f} | 位置 mean {pos_err.mean():.1f}mm | 姿态 mean {ori_err.mean():.2f}deg")
        if smooth:
            print("  时序优化 (离线 MPC: 跟踪 + Laplacian 平滑)...")
            # smooth_wbc_track 的 yaw 列是「相对 ref_yaw 的修正量」: 传入前从绝对角
            # 减掉 ref_yaw; 内部残差用 ref_yaw 重建绝对角, 返回的 base[:,2] 已是绝对角。
            x_init = np.concatenate([base_xyz[:, :2] - np.stack([dx0, dy0], axis=1),
                                     base_xyz[:, 2:3] - ref_yaw[:, None], arm], axis=1)
            base_xyz, arm, pos_err, ori_err = lrt.smooth_wbc_track(
                robot, side, Rsel, limR, dx0, dy0, tgt, pad_p,
                bp.se2_pose, bp._gripper_pad_fk, x_init,
                w_base=w_base, w_smooth=w_smooth, base_range=base_range, use_yaw=use_yaw,
                ref_yaw=ref_yaw)
            dx, dy, byaw = base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2]
            base_xy = base_ref + np.stack([dx, dy], axis=1)
            step1 = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            j1 = bp.traj_metrics(arm)[1]
            print(f"  [时序平滑] 底盘步长 max {step1.max()*1000:.0f}mm p95 {np.percentile(step1,95)*1000:.0f}mm "
                  f"| 臂 jerk {j1:.2f} | 位置 mean {pos_err.mean():.1f}mm | 姿态 mean {ori_err.mean():.2f}deg")
        print(f"  WBC 底盘修正: 相对跟随参考 偏移 max "
              f"x {np.abs(dx - dx0).max()*1000:.0f}mm y {np.abs(dy - dy0).max()*1000:.0f}mm "
              f"| yaw {np.degrees(np.abs(byaw)).max():.1f}deg")
        # 诊断: 尾段(位置误差最大段) 底盘解 vs 参考, 判断是否被 base_range 卡住
        _w = int(np.argsort(pos_err)[::-1][0])
        print(f"  诊断 误差最大帧 idx={_w}: 底盘参考(dx0,dy0)=({dx0[_w]*1000:.0f},{dy0[_w]*1000:.0f})mm "
              f"WBC解=({base_xyz[_w,0]*1000:.0f},{base_xyz[_w,1]*1000:.0f})mm "
              f"偏移=({(base_xyz[_w,0]-dx0[_w])*1000:.0f},{(base_xyz[_w,1]-dy0[_w])*1000:.0f})mm "
              f"base_range={base_range}m")
        print(f"  诊断 该帧臂关节解={np.round(np.degrees(arm[_w]),1).tolist()}deg "
              f"限位={np.round(np.degrees(limR),1).tolist()}")
        # 定位跳变帧: 区分是底盘跳还是臂分支跳, 便于针对性处理
        _dq = np.degrees(np.abs(np.diff(arm, axis=0))).max(axis=1)
        _db = np.linalg.norm(np.diff(base_xyz[:, :2], axis=0), axis=1) * 1000
        _jump = np.where((_dq > 60) | (_db > 60))[0] + 1
        print(f"  诊断 跳变帧 (臂>60° 或 底盘>60mm): {_jump.tolist()}")
        _bad = np.where(pos_err > 100)[0]
        print(f"  诊断 位置误差>100mm 帧数 {len(_bad)}: {_bad[:20].tolist()}"
              f"{'...' if len(_bad) > 20 else ''}")
        for _jf in _jump[:6]:
            print(f"    F{_jf}: 臂Δq {_dq[_jf-1]:.0f}° 底盘Δ {_db[_jf-1]:.0f}mm "
                  f"pos_err {pos_err[_jf-1]:.0f}->{pos_err[_jf]:.0f}mm "
                  f"| j1..j6 {np.round(np.degrees(arm[_jf-1]),0).tolist()}"
                  f" -> {np.round(np.degrees(arm[_jf]),0).tolist()}")
    # ---- 臂多分支 beam search (全局一致分支): 底盘沿用 WBC 解, 只重解臂。
    # 单链条会贴限位后被逼单帧翻转 (实测 173°); beam 保留多条分支假设, 提前换到
    # 能走得更远的那条, 在硬步长约束下取得"精度 + 平滑"同时成立。
    if wbc and arm_beam > 0:
        print(f"  臂多分支 beam search (beam={arm_beam}, 硬步长 {spread_max:.0f}°/帧)...")
        arm, pos_err, ori_err = lrt.solve_arm_beam(
            robot, side, Rsel, limR, dx, dy, byaw, tgt, pad_p,
            bp.se2_pose, bp._gripper_pad_fk, bp.limit_aware_resolve,
            beam=arm_beam, step_max_deg=(spread_max if spread_max > 0 else 35.0))
        _dq_b = np.degrees(np.abs(np.diff(arm, axis=0))).max(axis=1)
        print(f"    -> beam 后 max|Δq| {_dq_b.max():.1f}° | 位置 mean {pos_err.mean():.1f}mm "
              f"p95 {np.percentile(pos_err,95):.1f} max {pos_err.max():.1f}mm")
    # ---- 分支切换摊平: 冗余臂在个别帧必须换分支 (否则顶死限位, 误差 260mm+),
    # 但放开后是单帧 100°+ 抽搐。做法: 先对关节轨迹做速率限幅得到平滑"种子",
    # 再用限位感知 IK 逐帧投影回目标 -> 换分支被摊成多帧过渡, 且末端仍贴目标。
    if wbc and spread_max > 0 and torso_pitch_deg is None:
        _dq0 = np.degrees(np.abs(np.diff(arm, axis=0))).max(axis=1) if N > 1 else np.zeros(1)
        _bad0 = np.where(_dq0 > spread_max)[0] + 1
        if len(_bad0):
            print(f"  分支过渡摊平: 原始 max|Δq| {_dq0.max():.1f}° -> 限速 {spread_max:.0f}°/帧 "
                  f"(跳变帧 {_bad0.tolist()})")
            for _j in _bad0:
                _ji = int(np.argmax(np.abs(arm[_j] - arm[_j - 1])))
                print(f"    F{_j}: 主导关节 j{_ji+1} "
                      f"{np.degrees(arm[_j-1,_ji]):.0f}°->{np.degrees(arm[_j,_ji]):.0f}° "
                      f"| 前 {np.round(np.degrees(arm[_j-1]),0).tolist()} "
                      f"后 {np.round(np.degrees(arm[_j]),0).tolist()}")
            seeds_full = lrt.rate_limit_joints(arm, np.radians(spread_max))
            # **开环过渡 + 末端落回目标**: 中间帧直接用限速种子 (真机换分支时本来就
            # 只能沿一条可达路径连续走过去, 期间末端短暂偏离目标是物理必然, 不是 bug);
            # 只把窗口最后一帧用 IK 落回目标, 避免整段重解又把过渡改回抽搐。
            for _j in _bad0:
                _R = int(np.ceil(_dq0[_j - 1] / spread_max)) + 2
                _lo = max(1, _j - 1)
                _hi = min(N, _j + _R)
                arm[_lo:_hi] = seeds_full[_lo:_hi]
                _last = _hi - 1
                _a, _pe, _oe = lrt.solve_arm_track(
                    robot, side, Rsel, limR, dx[_last:_hi], dy[_last:_hi], byaw[_last:_hi],
                    [tgt[i] for i in range(_last, _hi)], pad_p[_last:_hi],
                    bp.se2_pose, bp._gripper_pad_fk, bp.limit_aware_resolve,
                    n_local=10, n_rand=0, w_cont=8.0, seed_arm=seeds_full[_last:_hi])
                arm[_last] = _a[0]
                # 窗口内真实跟踪误差 (底盘已配合, 照实记录不掩盖)
                for i in range(_lo, _hi):
                    robot.set_root_pose(bp.se2_pose(dx[i], dy[i], byaw[i]))
                    _p, _Rr = bp._gripper_pad_fk(robot, side, Rsel, arm[i])
                    pos_err[i] = np.linalg.norm(_p - pad_p[i]) * 1000.0
                    _c = (np.trace(tgt[i][1].T @ _Rr) - 1.0) / 2.0
                    ori_err[i] = np.degrees(np.arccos(np.clip(_c, -1.0, 1.0)))
                _bshift = np.linalg.norm(
                    np.stack([dx[_lo + 1:_hi] - dx[_lo:_hi - 1],
                              dy[_lo + 1:_hi] - dy[_lo:_hi - 1]], axis=1), axis=1) * 1000
                print(f"      窗口 F{_lo}-{_last}: err {np.round(pos_err[_lo:_hi], 0).tolist()}mm "
                      f"| 底盘位移/帧 {np.round(_bshift, 0).tolist()}mm")
            base_xy = base_ref + np.stack([dx, dy], axis=1)   # 底盘已被过渡窗口修改
            _dq1 = np.degrees(np.abs(np.diff(arm, axis=0))).max(axis=1)
            print(f"    -> 摊平后 max|Δq| {_dq1.max():.1f}° | 位置 mean {pos_err.mean():.1f}mm "
                  f"max {pos_err.max():.1f}mm (开环过渡段误差已照实计入)")
    if not wbc:
        print("  限位感知 IK 跟踪 (解耦: 底盘固定跟随)...")
        arm, pos_err, ori_err = lrt.solve_arm_track(robot, side, Rsel, limR, dx, dy, byaw,
                                                    tgt, pad_p, bp.se2_pose, bp._gripper_pad_fk,
                                                    bp.limit_aware_resolve)

    # ---- 底盘平滑 (可选) + 臂强连续重解 (总是执行) ----
    # 病根: WBC 逐帧独立解出现「臂关节 180° 分支翻转」(wrist flip), 并被底盘跳变触发
    # (底盘跳 442mm -> 臂补偿 -> 翻转)。对策: ①平滑低维的底盘轨迹 (底盘本就该平滑);
    # ②臂在上一帧解的邻域内局部搜索重解 (solve_arm_track 改为仅首帧用全局起点, 从机制上
    # 杜绝翻转)。目标固定于世界 -> 末端跟踪精度恢复。
    if wbc and not smooth and re_resolve:
        wbc_arm = arm.copy()                 # WBC 的臂解 -> 作为重解种子 (保持分支)
        if post_win >= 3:
            bx, by, byaw_s, _ = lrt.post_smooth_traj(
                base_xy, byaw, arm, win=post_win, passes=2,
                base_step_max=post_base_step, smooth_fn=rg.smooth_traj)
            base_xy = np.stack([bx, by], axis=1)
            byaw = byaw_s
            dx = base_xy[:, 0] - base_ref[0]
            dy = base_xy[:, 1] - base_ref[1]
            # 底盘已被平滑 -> 臂在**新底盘**上重解; 以 WBC 解为种子且**关闭全局随机起点**
            # (n_rand=0): 纯局部 + 强连续, 保持臂分支不变 (否则会跳到 180° 反向等价解)。
            arm, pos_err, ori_err = lrt.solve_arm_track(
                robot, side, Rsel, limR, dx, dy, byaw, tgt, pad_p,
                bp.se2_pose, bp._gripper_pad_fk, bp.limit_aware_resolve,
                n_local=8, n_rand=0, w_cont=10.0, seed_arm=wbc_arm)
        bstep_p = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
        print(f"  [底盘平滑 win={post_win}+臂局部重解] 底盘步长 max {bstep_p.max()*1000:.0f}mm "
              f"p95 {np.percentile(bstep_p,95)*1000:.0f}mm")

    _dt_solve = time.time() - _t_solve
    print(f"  解算耗时 {_dt_solve:.1f}s ({_dt_solve / max(N, 1) * 1000:.0f} ms/帧, 含摊平)")

    m = bp.traj_metrics(arm)
    viol = int(np.sum(np.any((arm < limR[:, 0] - 1e-6) | (arm > limR[:, 1] + 1e-6), axis=1)))
    bstep = np.linalg.norm(np.diff(base_xy, axis=0), axis=1) if N > 1 else np.zeros(1)
    print(f"  右臂平滑度 max|Δq|={m[0]:.2f}deg  rms|jerk|={m[1]:.3f}deg")
    print(f"  右臂越限帧 {viol}/{N}")
    print(f"  底盘步长 max {bstep.max()*1000:.1f}mm/帧  p95 {np.percentile(bstep,95)*1000:.1f}mm")
    print(f"  IK 位置误差 mm : mean {pos_err.mean():.2f}  p95 {np.percentile(pos_err,95):.2f}  max {pos_err.max():.2f}")
    _worst = np.argsort(pos_err)[::-1][:8]
    print(f"  位置误差最大帧 idx={_worst.tolist()} err_mm={np.round(pos_err[_worst],0).tolist()}")
    print(f"  IK 姿态误差 deg: mean {ori_err.mean():.2f}  p95 {np.percentile(ori_err,95):.2f}  max {ori_err.max():.2f}")
    # 关节裕度: 贴限位 = 后续动作没余量 (真机会卡死/报警), 需提前暴露
    _qdeg = np.degrees(arm)
    _mgn = np.minimum(_qdeg - np.degrees(limR[:, 0]), np.degrees(limR[:, 1]) - _qdeg)
    _tight = np.where(_mgn.min(axis=1) < 5.0)[0]
    _worst_tight = int(np.argmin(_mgn.min(axis=1)))
    print(f"  关节裕度: 各关节最小裕度 {np.round(_mgn.min(axis=0), 1).tolist()} deg "
          f"(j1..j6) | 裕度<5° 帧 {len(_tight)}/{N}"
          + (f" 最紧帧 {_worst_tight}" if len(_tight) else ""))

    # ---- 误差归因: ①正常帧 vs 分支过渡窗口 ②底盘系三轴分解 ----
    _dq = np.degrees(np.abs(np.diff(arm, axis=0))).max(axis=1) if N > 1 else np.zeros(1)
    _transit = np.zeros(N, bool)
    for _i in np.where(_dq >= (spread_max - 0.1))[0] if spread_max > 0 else []:
        _transit[_i:_i + 2] = True              # 满速段 ≈ 开环过渡窗口
    _norm = ~_transit
    if _transit.any():
        print(f"  误差归因: 过渡帧 {int(_transit.sum())}/{N} ({(100*_transit.mean()):.0f}%) "
              f"占位置误差总量 {100*np.sum(pos_err[_transit])/max(pos_err.sum(),1e-9):.0f}% | "
              f"正常帧 mean {pos_err[_norm].mean():.1f}mm 过渡帧 mean {pos_err[_transit].mean():.1f}mm")
    # 底盘系三轴误差 (前后/左右/上下): 把世界系误差向量转到该帧底盘朝向
    _axes = np.zeros((N, 3))
    ik_p = np.zeros((N, 3))                  # IK 解对应的世界系 pad 位置 (用于轨迹对比图)
    for _k in range(N):
        robot.set_root_pose(bp.se2_pose(dx[_k], dy[_k], byaw[_k]))
        _pa, _Ra = bp._gripper_pad_fk(robot, side, Rsel, arm[_k])
        ik_p[_k] = _pa
        _e = pad_p[_k] - _pa
        _th = byaw[_k]
        _base_error_Rz = np.array([[np.cos(_th), -np.sin(_th), 0], [np.sin(_th), np.cos(_th), 0], [0, 0, 1]])
        _axes[_k] = _base_error_Rz.T @ _e
    print(f"  底盘系误差分解 (正常帧): 前后 {np.abs(_axes[_norm,0]).mean()*1000:.1f}mm  "
          f"左右 {np.abs(_axes[_norm,1]).mean()*1000:.1f}mm  "
          f"上下 {np.abs(_axes[_norm,2]).mean()*1000:.1f}mm")
    if _transit.any():                       # 无过渡窗口时避免 mean 空切片 -> nan
        print(f"  底盘系误差分解 (过渡帧): 前后 {np.abs(_axes[_transit,0]).mean()*1000:.1f}mm  "
              f"左右 {np.abs(_axes[_transit,1]).mean()*1000:.1f}mm  "
              f"上下 {np.abs(_axes[_transit,2]).mean()*1000:.1f}mm")
    # ---- 指标落盘 (ik_only 也保存, 便于参数扫描对比) ----
    _payload = {
        "task": f"{name}_large_range",
        "config": {"stride": stride, "n_frames": int(N), "wbc": wbc,
                   "w_base": w_base, "base_range": base_range, "use_yaw": use_yaw,
                   "smooth": smooth, "re_resolve": re_resolve,
                   "w_cont": w_cont, "w_bcont": w_bcont,
                   "base_smooth_win": base_smooth, "post_win": post_win},
        "collision": {"env_boxes_n": 0 if env_boxes is None else int(len(env_boxes)),
                      "obstacles_n": 0 if obstacles is None else int(len(obstacles)),
                      "w_obs": float(w_obs)},
        "target_clean": {"n_outlier_frames": int(len(bad))},
        "base": {"travel_m": float(np.linalg.norm(np.diff(base_xy, axis=0), axis=1).sum()),
                 "step_max_mm": float(bstep.max() * 1000),
                 "step_p95_mm": float(np.percentile(bstep, 95) * 1000)},
        "arm": {"max_dq_deg": float(m[0]), "rms_jerk_deg": float(m[1]),
                "limit_viol_frames": int(viol),
                "min_margin_deg_per_joint": [float(x) for x in _mgn.min(axis=0)],
                "tight_frames_lt5deg": int(len(_tight))},
        "ik_pos_err_mm": {"mean": float(pos_err.mean()),
                          "p95": float(np.percentile(pos_err, 95)),
                          "max": float(pos_err.max())},
        "ik_ori_err_deg": {"mean": float(ori_err.mean()),
                           "p95": float(np.percentile(ori_err, 95)),
                           "max": float(ori_err.max())},
    }
    if torso_pitch_deg is not None:
        pitch_deg = np.degrees(lrt.torso_link_pitch(arm[:, 6:9]))
        command_limits = torso_pitch_deg - torso_pitch_guard_deg
        if np.any(np.abs(pitch_deg) > command_limits + 1e-7):
            raise RuntimeError("bounded WBC emitted a torso link command outside its hard limit")
        _payload["config"].update(torso_pitch_deg=torso_pitch_deg.tolist(), joint_reserve_deg=joint_reserve_deg,
                                  base_lead_s=base_lead_s, torso_pitch_guard_deg=torso_pitch_guard_deg,
                                  ring_place=ring_place)
        _payload["torso"] = {
            "target_max_abs_pitch_deg": float(np.max(np.abs(pitch_deg[:, 2]))),
            "target_link_peak_deg": np.max(np.abs(pitch_deg), axis=0).tolist(),
            "target_link_limits_deg": command_limits.tolist(),
        }
    _dump_metrics(_payload)
    if ik_only:
        return None
    if torso_pitch_deg is None:
        arm = rg.smooth_traj(arm, 5, 2)   # legacy postfilter can overshoot the solver's bounds

    # ---- 可视化: 底盘地面路径 + 手部目标标记/路径 ----
    lrt.add_path_markers(scene, rg.create_box, base_xy, base_ref)
    marker = lrt.TargetMarker(scene, pad_p, rg.create_box)
    # 环境碰撞盒 (static): 拟合场景体素 → 物理碰撞 —— 机器人真实"撞上"环境
    # 而非穿过。env_boxes=(M,6) [cx,cy,cz,hx,hy,hz], 由 test 流程从场景点云
    # 路径附近体素化生成 (排除 pad 轨迹邻域防卡臂)。
    if env_boxes is not None and len(env_boxes):
        import sapien.render as _sre
        _emat = _sre.RenderMaterial()
        _emat.base_color = [0.55, 0.52, 0.48, 1.0]
        _emat.roughness = 0.9
        _n_env = 0
        for _bx in np.asarray(env_boxes, float):
            _bb = scene.create_actor_builder()
            _bb.set_physx_body_type("static")
            _bb.add_box_collision(sapien.Pose(p=[0, 0, 0]),
                                  half_size=[_bx[3], _bx[4], _bx[5]])
            _bb.add_box_visual(sapien.Pose(p=[0, 0, 0]),
                               half_size=[_bx[3], _bx[4], _bx[5]], material=_emat)
            _ab = _bb.build(name="env_collision_box")
            _ab.set_pose(sapien.Pose(p=_bx[:3]))
            _n_env += 1
        print(f"  环境碰撞盒: {_n_env} 个 static box 入仿真")

    # 负载自适应驱动增益 (门控参数, 默认 1.0=旧版): 躯干承担全臂+物体, 腕直接
    # 承受物体接触/惯性 —— mocap 移动复刻链传 6/3 (跟随 48→30mm), 456 旗舰
    # 依赖旧增益抽签故默认 1.0。阻尼按临界匹配 √k 缩放。
    # 指爪 link 去穿透限速 (与物体侧配对生效; 见下物体构建注释)。
    # depenetration_vel=None = 不设 (默认 1m/s 旧行为): 456 旗舰的接触抽签
    # 依赖旧动力学, 限速会变抓取结果 (实测 120→17mm) —— 仅 mocap/EE 链显式 0.05。
    if depenetration_vel is not None:
        for _ln in (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2",
                    f"{side}_gripper_link"):
            _fl = robot.find_link_by_name(_ln)
            if _fl is not None:
                for _comp in _fl.entity.components:
                    if hasattr(_comp, "set_max_depenetration_velocity"):
                        _comp.set_max_depenetration_velocity(float(depenetration_vel))

    for j in joints:
        _n = j.get_name()
        _g = stiff_scale
        if "torso_joint" in _n:
            _g *= torso_gain
        elif "arm_joint4" in _n or "arm_joint5" in _n or "arm_joint6" in _n:
            _g *= wrist_gain
        j.set_drive_property(stiffness=rg.ARM_STIFF * _g,
                             damping=rg.ARM_DAMP * np.sqrt(_g))
    if object_physics:
        # 物理抓取时指爪需要快速闭合到位: 默认 1000/200 从全开闭到目标要 ~60 帧,
        # 物体在闭合完成前就已滑落 (实测 F137 目标到位, F200 实际才闭合)。
        # 指爪是 prismatic 小行程关节, 提高刚度不会引起臂的抖动。
        for nm in (f"{side}_gripper_finger_joint1", f"{side}_gripper_finger_joint2"):
            if nm in jmap:
                if finger_force_limit is not None:
                    # 恒力夹持 (finger_force_limit 显式开启才生效, 默认 None=旧版行为):
                    # 接近段驱动封顶, 防闭合到位瞬间的驱动力尖峰经接触传入物体。
                    jmap[nm].set_drive_property(stiffness=8000.0, damping=400.0,
                                                force_limit=finger_force_limit)
                else:
                    jmap[nm].set_drive_property(stiffness=8000.0, damping=400.0)
        # A3: 指面摩擦材质**显式设置到碰撞 shape 级** —— 物体 grip_mat 摩擦 1.0,
        # 但指面此前用的是机器人默认材质 (未显式设置), PhysX 默认 combine(平均)
        # 后接触面摩擦不足, 交接重力后物体在两指间下滑 (实测 F142 交接 → F176 落地)。
        # 1.5/1.5 + 物体 1.0 → 组合 ~1.25; restitution 0 防弹跳。
        # (μ2.0 实验: chairbottle 蠕滑 128→125mm@F222, 无效 —— 蠕滑是运动学喂出
        # 而非摩擦极限, 见 docs/12 逃逸速度剖析)
        _fing_mat = scene.create_physical_material(static_friction=1.5, dynamic_friction=1.5,
                                                   restitution=0.0)
        for nm in (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2"):
            _link = robot.find_link_by_name(nm)
            _n_sh = 0
            for comp in _link.entity.components:
                for sh in getattr(comp, "get_collision_shapes", list)():
                    sh.set_physical_material(_fing_mat)
                    _n_sh += 1
            print(f"  指面摩擦: {nm} shapes={_n_sh} -> static=1.5 dynamic=1.5 "
                  f"(物体侧 1.0, PhysX 平均组合 ~1.25)")
    for t in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"):
        if t in jmap:
            jmap[t].set_drive_target(0.0)

    # ---- 场景点云视觉 (EgoFound3R VGGT 深度点云, 体素化小方块, 仅视觉) ----
    # 尺度对齐 (重力对齐模式): EF3R 手部自标定与 VGGT 场景度量存在竖直偏差
    # (实测 456483158: 场景桌面 z≈1.449 vs 手度量交互面 1.213, 差 236mm) ——
    # 不对齐则背景桌子悬浮在交互平面上方, "背景不像"。以手度量交互面 (接物
    # 桌) 为准, 场景云沿 sim z (重力) 平移 _dh_scene; 台面网格同 Shift。
    _dh_scene = 0.0
    registered_scene = False
    if scene_cloud and os.path.exists(scene_cloud):
        sc = np.load(scene_cloud)
        P_sc, C_sc = sc["points"], sc["colors"]
        registered_scene = bool(sc.get("mocap_world", np.array([0]))[0])
        if registered_scene:
            P_sc = P_sc @ _Rz.T
            P_sc[:, 2] += z_shift
        elif sc.get("already_sim", np.array([0]))[0] == 1:
            # 场景已在 sim 世界系 (mocap 路径: tracker 相机位姿直接定位, 无
            # M_sim/_Rz); 但 z_shift (fit_height 高度归一) 必须跟随 targets
            # —— 否则 targets 被重归一到工作高度而场景留在原地 (垂直脱节)
            P_sc[:, 2] += z_shift
            print(f"  场景点云: already_sim (mocap 系直通 + z_shift "
                  f"{z_shift:+.3f}, {len(P_sc)} 点)")
        else:
            # geo 对齐: M' (重力对齐轴变换) + 世界旋转 + z_shift —— 与手部目标同一链
            P_sc = P_sc @ M_EF3R.T
            if abs(_yaw0) > 1e-9:
                P_sc = P_sc @ _Rz.T
            P_sc[:, 2] += z_shift
        if not registered_scene:
            if _ga_dir is not None:
                _z_int = float(pad_p[:, 2].min()) - 0.06      # 交互面 = 接物桌顶 (手度量)
                _m_pick = ((d["grip_states"][idx] >= 1) & (d["grip_states"][idx] < 3)
                           if "grip_states" in d.files else np.ones(len(idx), bool))
                _xy0 = pad_p[_m_pick].mean(0)[:2] if _m_pick.any() else pad_p.mean(0)[:2]
                _nb = P_sc[(np.abs(P_sc[:, 0] - _xy0[0]) < 0.35)
                           & (np.abs(P_sc[:, 1] - _xy0[1]) < 0.35)
                           & (P_sc[:, 2] > _z_int - 0.05) & (P_sc[:, 2] < _z_int + 0.60)]
                if len(_nb) > 2000:
                    _hh, _ee = np.histogram(_nb[:, 2], bins=40)
                    _z_scene_tbl = float(_ee[int(np.argmax(_hh))] + 0.009)
                    _dh_scene = float(np.clip(_z_int - _z_scene_tbl, -0.5, 0.5))
                    P_sc[:, 2] += _dh_scene
                print(f"  背景尺度对齐: 场景桌面 z={_z_scene_tbl if len(_nb) > 2000 else float('nan'):.3f}"
                      f" -> 交互面 {_z_int:.3f} (沿重力平移 {_dh_scene*1000:+.0f}mm,"
                      f" 手度量 vs VGGT 场景度量偏差)")
            # 鬼影滤除: 手部最高点 +25cm 以上的背景/墙点, 以及轨迹工作区 1.2m 外的点
            _hand_top = pos[:, 2].max()
            keep = P_sc[:, 2] < _hand_top + 0.25
            P_sc, C_sc = P_sc[keep], C_sc[keep]
            keep = np.linalg.norm(P_sc[:, :2] - P_sc[:, :2].mean(0), axis=1) < 1.2
            P_sc, C_sc = P_sc[keep], C_sc[keep]
            # 背景桌面对齐: EF3R 手部自标定 vs VGGT 场景度量存在竖直偏差 (实测
            # ~142mm) —— 场景桌面高于交互平面, 画面里桌子悬空、机器人"往上够"。
            # 对齐: 抓取邻域的场景桌面向下平移到接物桌高度 (pad@PICK 最低 - 6cm),
            # 背景桌面与交互平面重合 (用户实测指认: 应往下抓, 桌面不该悬浮)。
            _m_pick_bg = ((d["grip_states"][idx] >= 1) & (d["grip_states"][idx] < 3)
                          if "grip_states" in d.files else np.ones(len(idx), bool))
            _xy_pick = pad_p[_m_pick_bg].mean(0)[:2] if _m_pick_bg.any() else pos_f.mean(0)[:2]
            _z_hand_tbl = float(pad_p[_m_pick_bg][:, 2].min()) - 0.06 if _m_pick_bg.any()             else float(pos_f[:, 2].min()) - 0.06
            _nbh = P_sc[(np.abs(P_sc[:, 0] - _xy_pick[0]) < 0.45)
                        & (np.abs(P_sc[:, 1] - _xy_pick[1]) < 0.45)
                        & (P_sc[:, 2] > _z_hand_tbl - 0.35) & (P_sc[:, 2] < _z_hand_tbl + 0.25)]
            if len(_nbh) > 2000:
                _hh2, _ee2 = np.histogram(_nbh[:, 2], bins=40)
                _z_tbl_scene = float(_ee2[int(np.argmax(_hh2))] + 0.01)
                _dh_scene = _z_hand_tbl - _z_tbl_scene
                if abs(_dh_scene) > 0.02:
                    P_sc[:, 2] += _dh_scene
                    print(f"  背景桌面对齐: 场景桌面 z={_z_tbl_scene:.3f} → 接物桌 "
                          f"{_z_hand_tbl:.3f} (平移 {_dh_scene*1000:+.0f}mm, 手度量 vs "
                          f"VGGT 场景度量偏差)")
                else:
                    print(f"  背景桌面对齐: 偏差 {_dh_scene*1000:.0f}mm, 无需平移")
            # 轨迹隧道镂空: 手部扫掠筒 (pad 轨迹 7cm 内) 的场景点移除 —— 交互区桌点
            # 密集会把夹爪/物体整个埋进体素里 (GIF 不可读); 挖出隧道后夹爪与物体可见,
            # 背景其余部分保留 (数据集可视化用, 不影响物理)。
            try:
                from scipy.spatial import cKDTree
                _tube = cKDTree(pad_p[::3])
                _n_tube = pad_p[::3].shape[0]
                _keep = _tube.query(P_sc, k=1, distance_upper_bound=0.07)[1] == _n_tube
                _n0 = len(P_sc)
                P_sc, C_sc = P_sc[_keep], C_sc[_keep]
                print(f"  轨迹隧道镂空: {_n0 - len(P_sc)} 点移除 ({_n0 - len(P_sc)} 点在扫掠筒内)")
            except Exception as _e:
                print(f"  轨迹隧道镂空跳过: {_e}")
        _vox = 0.018 if registered_scene else 0.04
        _key = np.floor(P_sc / _vox).astype(np.int64)
        _, _idx = np.unique(_key, axis=0, return_index=True)
        Pv, Cv = P_sc[_idx], C_sc[_idx]
        bb = scene.create_actor_builder()
        bb.set_physx_body_type("static")
        for p_, c_ in zip(Pv, Cv):
            bb.add_box_visual(sapien.Pose(p=list(p_)),
                              half_size=[_vox/2]*3,
                              material=(float(c_[0]), float(c_[1]), float(c_[2])))
        bb.build(name="ef3r_scene_cloud")
        print(f"  场景点云视觉: {len(Pv)} 体素 ({_vox*1000:.0f}mm, geo 对齐+鬼影滤除)")
    # ---- 被抓物体 (EgoFound3R 点云拟合的凸包实例, 可选) ----
    # geo 对齐三步: ①M 轴变换 (EF3R y-up → sim z-up, 与 ef3r_to_targets 同一变换)
    #              ②世界系旋转 _Rz (与手部目标同一处理) ③fit_height 的 z_shift。
    # 三步与手部目标完全同源 → 夹爪到目标位 = 到物体位。薄轴对准夹爪开合方向。
    obj_actor, obj_trace, obj_meta = None, None, {}
    _support_name = None
    _place_collider = None           # physics 交接时恢复碰撞组的接物桌形状
                                     # (chairbottle 物体无 surface_pick_V 时支撑块跳过,
                                     #  仍需已初始化, 交接块才能安全判 None)
    k_lift, _obj_comp, k_grav_on = 0, None, None
    k_release, _R_rel = 0, None
    if object_npz:
        od = np.load(object_npz)
        verts = od["verts"] @ M_EF3R.T          # ① y-up→z-up (方向向量用右乘转置)
        c_placed = M_EF3R @ od["center_placed"]
        if abs(_yaw0) > 1e-9:                          # ② 世界系旋转 (仅旋转任务存在)
            verts = verts @ _Rz.T
            c_placed = _Rz @ c_placed
        verts[:, 2] += z_shift                         # ③ 高度对齐 (与 pos_f 同步)
        c_placed[2] += z_shift
        # PCA 轴/尺寸**从实际网格自算**, 不读 od["pca_*"] 元数据。
        # 为什么: 实测 object_instance_agile.npz 的 pca_ext=[50.5,47.2,25.2] 与
        # verts 真实包围盒不一致 (对齐后仍出现 46.4mm 的"第三维") —— 元数据是上游
        # 在别的点集上算的。碰撞盒尺寸一旦与视觉网格错位, 两指就会插进物体内部
        # (内距 21.9mm 却按 47.2mm 建盒), 切 dynamic 后必掉 (agile456 F149→F160)。
        _v0 = verts - verts.mean(0)
        _w, _V = np.linalg.eigh(_v0.T @ _v0)          # 升序特征值
        _order = np.argsort(_w)[::-1]                  # 长 → 短
        axes_sim = _V[:, _order]                       # 列 = 三个主轴 (世界系)
        ext_actual = np.ptp(_v0 @ axes_sim, axis=0)    # 沿各主轴的真实尺寸
        f_lift = int(np.argmax(d["grip_states"] >= 2)) if "grip_states" in d.files else 0
        k_lift = int(np.argmax(d["grip_states"][idx] >= 2)) if "grip_states" in d.files else 0
        rot_grasp = rot0[np.searchsorted(idx, f_lift)] if "grip_states" in d.files else rot0[0]
        a_open = rot_grasp @ np.array([0.0, 1.0, 0.0])  # 夹爪开合方向 (gripper y)
        a_fwd = rot_grasp @ np.array([1.0, 0.0, 0.0])   # pad 前向 (gripper x)
        # ---- 抓取朝向: 物体局部系 = 自身 PCA 主轴系, actor 姿态 = 抓取基 ----
        # 旧实现有两处叠加错误 (实测: 物体沿开合轴跨度 65.6mm, 薄轴仅 25.2mm):
        #  ① R_al = B_tgt @ axes_sim.T 的三列全取自 axes_sim 自身, 交叉项互相抵消
        #     => 退化成 diag(±1,±1,±1), 物体**压根没被旋转**;
        #  ② verts_al = R_al·v0 被当成"局部顶点", 但 actor 的 pose 旋转又是 R_al
        #     => 实际做了两次旋转 (旧版因 R_al=±I 恰好看不出来)。
        # 正解: 局部坐标**就是 PCA 坐标** l = axes_sim^T·v0, actor 姿态 Rk = B_tgt。
        # 这样局部 x(长轴)/y(第三轴)/z(薄轴) 精确映射到 接近轴/第三轴/开合轴,
        # 开合方向的跨度恒为薄轴 25.2mm, 与碰撞盒 half_size=[长,第三,薄] 完全一致
        # (平行夹爪夹长方体必须夹最薄的一维; 指面 0.06×0.01×0.04, 长轴由 60mm 指长承托)。
        # 注意 PCA 前两轴 (50.6/47.2mm) 特征值接近 => 绕薄轴转角欠定, 只影响视觉
        # 朝向, 不再影响开合跨度。
        _a_third = np.cross(a_open, a_fwd)             # 右手系: cross(接近,第三)=开合
        B_tgt = np.column_stack([a_fwd, _a_third, a_open])   # det=+1
        _vlocal = _v0 @ axes_sim                        # PCA 局部坐标 (列 = 长/第三/薄)
        obj_center = pad_p[k_lift].copy()   # 夹爪抓取点 (pad 目标) —— 物体必须出生在两指之间
        verts_al = _vlocal + obj_center
        _R_rel = rot_grasp.T @ B_tgt         # 物体朝向相对 pad 系
        # 抓取跟随窗口: LIFT 起 → RELEASED 起 (刚体在手中); RELEASED 起切物理 (落支撑台)
        releases = np.flatnonzero(d["grip_states"][idx] >= 5) if "grip_states" in d.files else []
        k_release = int(releases[0]) if len(releases) else N
        ext = np.ptp(verts_al, axis=0)
        obj_meta = dict(ext_mm=np.round(ext*1000, 1).tolist(),
                        thin_mm=float(ext_actual[2]*1000),
                        pca_meta_ext_mm=np.round(od["pca_ext"]*1000, 1).tolist()
                        if "pca_ext" in od.files else None,
                        placed_xy=[float(x) for x in c_placed[:2]])
        obj_path = f"/tmp/ef3r_object_{name}.obj"
        _vc = od["visual_local_V"] if "visual_local_V" in od.files else verts_al - obj_center
        visual_faces = od["visual_local_F"] if "visual_local_F" in od.files else od["faces"]
        if "visual_local_V" in od.files:
            obj_meta["geometry_kind"] = str(od.get("geometry_kind", "user_visual"))
        _vn = np.zeros_like(_vc)
        for a, b, c3 in visual_faces:
            _n = np.cross(_vc[b] - _vc[a], _vc[c3] - _vc[a])
            _n /= max(np.linalg.norm(_n), 1e-12)
            _vn[[a, b, c3]] += _n
        _vn /= np.maximum(np.linalg.norm(_vn, axis=1, keepdims=True), 1e-12)
        with open(obj_path, "w") as f:
            f.write(f"# EF3R object hull, geo-aligned, centered\n")
            for v in _vc:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for n in _vn:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for a, b, c3 in visual_faces:
                f.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c3+1}//{c3+1}\n")
        b = scene.create_actor_builder()
        b.set_physx_body_type("dynamic")
        grip_mat = scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        # 碰撞体 = PCA 轴对齐的盒子 (简单拟合, 平面接触稳定, 避免粗糙凸包点接触弹飞);
        # 视觉 = 凸包 OBJ (保留有机外形)。对齐系: x=长轴, y=第三轴, z=薄轴(开合方向)
        # half_size 必须用**实际网格**尺寸 ext_actual (与视觉一致), 不用 pca_ext 元数据。
        # 抓握立方钳位 (仅大物体): 长/中半轴 >50mm 时钳到 40/50mm —— 指根
        # ~40mm 内不可有碰撞体, 否则物体出生即与 gripper link 深度互穿 → 解算器
        # 分离力把臂顶飞 (实测 ee1/ee5mocap pad 甩至 1.9m)。小物体 (鸭 43mm)
        # 保持原盒 —— 钳位会改其接触抽签 (实测 456 抓起✓→✗)。薄轴 (开合方向)
        # 保留全尺寸 = 指面接触不变; 视觉网格保留全形。
        _half = [float(ext_actual[0]) / 2, float(ext_actual[1]) / 2, float(ext_actual[2]) / 2]
        if _half[0] > 0.050:
            _half[0] = 0.040
        if _half[1] > 0.052:
            _half[1] = 0.050
        # [ee17] 锥形物体抓握覆盖: PCA 包围盒近立方 (漏斗杯口 96mm) 时, 薄轴盒
        # 挡住手指压入锥面 (交接 gap 96.5mm = 表面轻贴, 摩擦≈0 -> 滑落)。物体
        # npz 提供 grasp_half_m (实测锥面接触宽 56mm/2) 时覆盖薄轴, 让指压入
        # 锥面摩擦夹紧。squeeze 目标 (CB_SQUEEZE_NOMINAL) 需与之配套。
        if "grasp_half_m" in od.files:
            _gh = np.asarray(od["grasp_half_m"], float)
            if _gh.shape == (3,) and np.all((_gh > 0.008) & (_gh < 0.060)):
                _half = [float(x) for x in _gh]
                print(f"  抓握盒覆盖 (grasp_half_m): {_half}")
        obj_meta["box_half_m"] = [float(x) for x in _half]   # 供抓取诊断用 (_half 后面会被桌子覆盖)
        b.add_box_collision(sapien.Pose(p=[0, 0, 0]), half_size=_half, material=grip_mat)
        import sapien.render as _sr
        _vmat = _sr.RenderMaterial()
        # 物体可见性: 亮橙 (上一代硬编码 0.08 黑色是"黑色耳机"专用 —— 黑物体
        # 贴黑夹爪在 GIF 里完全不可见; add_visual_from_file 不读 OBJ 的 MTL 纹理)
        _vmat.base_color = ([0.86, 0.67, 0.30, 1.0] if "visual_local_V" in od.files
                            else [0.95, 0.45, 0.10, 1.0])
        _vmat.metallic = 0.0
        _vmat.roughness = 0.6
        b.add_visual_from_file(filename=obj_path, material=_vmat)   # 亮红视觉: 调试可见
        obj_actor = b.build(name="ef3r_object")
        # 质心对中: 模型顶点相对质心, actor 位姿 = 物体质心位姿 (轨迹语义正确)
        obj_actor.set_pose(sapien.Pose(p=list(obj_center)))
        for comp in obj_actor.get_components():
            if isinstance(comp, sapienp.PhysxRigidDynamicComponent):
                comp.mass = 0.12
                comp.angular_damping = 5.0
                comp.linear_damping = 0.5
                # 两种模式都 kinematic+无重力 出生 (悬置在抓取点, 免疫接触推动):
                #   follow  → 跟随轨迹至 RELEASED, 切 dynamic 落支撑台
                #   physics → 跟随至闭合到位, 切 dynamic 被两指夹住 (真接触抓取)
                if depenetration_vel is not None and \
                        hasattr(comp, "set_max_depenetration_velocity"):
                    # 去穿透限速 (门控参数 depenetration_vel, mocap/EE 链 0.05):
                    # 指爪咬入物体时解算器的分离冲量 ∝ 该速度 × 反射惯量 ——
                    # 1m/s 下每步 2-5 N·s 的"分离锤"持续打腕/臂 (跟随 36mm →
                    # 抓取 321mm 的根因)。限速后分离力降 ~20x; 456 旗舰依赖
                    # 旧动力学抽签, 故默认不设。
                    comp.set_max_depenetration_velocity(float(depenetration_vel))
                if hasattr(comp, "set_disable_gravity"):
                    comp.set_disable_gravity(True)
                    comp.set_kinematic(True)
                    _obj_comp = comp
        # follow 模式: 记录物体默认碰撞组并**出生即清零** —— kinematic 胶水盒与
        # 物理指爪的任何穿插都会每子步推手指 (抓取闭合期撞上 → phys 6.3→55mm,
        # 实测 2026-10-29)。视觉穿插由 grip 执行钳位 (开合≥薄轴+1.5mm) 避免。
        # RELEASED 恢复组 + 切 dynamic 落台 (与 PLACE 台面碰撞需要组回来)。
        obj_shape_groups = None
        if obj_actor is not None and not object_physics and _obj_comp is not None:
            obj_shape_groups = [list(sh.get_collision_groups())
                                for sh in _obj_comp.get_collision_shapes()]
            for sh in _obj_comp.get_collision_shapes():
                sh.set_collision_groups([0, 0, 0, 0])
        # 放置支撑面 fallback (无拟合面片时): 朴素轴对齐方块
        if "support_placed" in od.files and "surface_pick_V" not in od.files:
            sup = od["support_placed"]
            sup = M_EF3R @ sup
            if abs(_yaw0) > 1e-9:
                sup = _Rz @ sup
            sup[2] += z_shift
            _sup_h = 0.04
            sb = scene.create_actor_builder()
            sb.set_physx_body_type("static")
            sb.add_box_collision(sapien.Pose(p=[0, 0, -_sup_h / 2]), half_size=[0.3, 0.3, _sup_h / 2])
            sb.add_box_visual(sapien.Pose(p=[0, 0, -_sup_h / 2]), half_size=[0.3, 0.3, _sup_h / 2],
                              material=(0.75, 0.73, 0.70))
            sup_actor = sb.build(name="ef3r_support")
            _rel_xy = pad_p[k_release][:2] if k_release < N else sup[:2]
            sup_actor.set_pose(sapien.Pose(p=[float(_rel_xy[0]), float(_rel_xy[1]), float(sup[2])]))
            _support_name = "ef3r_support"
            obj_meta["support_z"] = float(sup[2])
            print(f"  放置支撑台: 顶面 z={sup[2]:.3f} (EF3R RELEASED 场景点拟合)")
        obj_trace = np.zeros((N, 3))
        obj_trace[0] = obj_actor.get_pose().p
        # 手指链接高摩擦 (PhysX 摩擦取两接触面平均, 只设物体侧会减半 —— 对齐 duck)
        for link in robot.get_links():
            if link.get_name() in (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2"):
                for comp in link.entity.components:
                    if hasattr(comp, "physx_material"):
                        comp.physx_material = grip_mat
        _half_z = float(ext_actual[2]) / 2
        # ---- 支撑台面 mesh (点云拟合面片, 带朝向): PICK 仅 follow 模式 / PLACE 两模式 ----
        def _surf_pose(R_ef, c_ef):
            R_s = M_EF3R @ R_ef
            c_s = M_EF3R @ c_ef
            if abs(_yaw0) > 1e-9:
                R_s = _Rz @ R_s
                c_s = _Rz @ c_s
            c_s[2] += z_shift + _dh_scene
            return c_s, Rot.from_matrix(R_s).as_quat()[[3, 0, 1, 2]]
        if "surface_pick_V" in od.files:
            # 台面 = 原点云真实表面的 Delaunay 网格 (geo 对齐后写 OBJ, 视觉+凸包碰撞)
            def _surf_mesh_actor(nm, V_ef, F_ef, collide):
                V_s = V_ef @ M_EF3R.T
                if abs(_yaw0) > 1e-9:
                    V_s = V_s @ _Rz.T
                V_s[:, 2] += z_shift + _dh_scene
                _cc = V_s.mean(0)
                pth = f"/tmp/ef3r_surface_{name}_{nm}.obj"
                _n = np.zeros_like(V_s)
                for a, b, c3 in F_ef:
                    _n[[a, b, c3]] += np.cross(V_s[b] - V_s[a], V_s[c3] - V_s[a])
                _n /= np.maximum(np.linalg.norm(_n, axis=1, keepdims=True), 1e-12)
                with open(pth, "w") as f:
                    for v, n in zip(V_s - _cc, _n):
                        f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\nvn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
                    for a, b, c3 in F_ef:
                        f.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c3+1}//{c3+1}\n")
                sb = scene.create_actor_builder()
                sb.set_physx_body_type("static")
                import sapien.render as _sr2
                _tmat = _sr2.RenderMaterial()
                _tmat.base_color = [0.80, 0.78, 0.74, 1.0]
                _tmat.roughness = 0.85
                if collide:
                    if "ring_segment_count" in od.files and nm == "ef3r_place_support":
                        count = int(od["ring_segment_count"])
                        if count < 8 or len(V_s) != 8*count:
                            raise ValueError("ring contract must contain at least eight ordered box segments")
                        for v in V_s.reshape(-1, 8, 3):
                            edges = v[[4, 2, 1]] - v[0]
                            size = np.linalg.norm(edges, axis=1)
                            if not np.all(size > 0):
                                raise ValueError("ring segment has a zero dimension")
                            basis = (edges / size[:, None]).T
                            q = Rot.from_matrix(basis).as_quat()[[3, 0, 1, 2]]
                            sb.add_box_collision(sapien.Pose(p=v.mean(0)-_cc, q=q),
                                                 half_size=size/2, material=grip_mat)
                        print(f"  Annular support: {count} convex pieces, center {_cc.round(4)}")
                    else:
                        sb.add_multiple_convex_collisions_from_file(filename=pth, material=grip_mat)
                sb.add_visual_from_file(filename=pth, material=_tmat)
                _sup = sb.build(name=nm)
                _sup.set_pose(sapien.Pose(p=list(_cc)))
                return float(_cc[2]), _sup
            # PICK 台面碰撞只在 object_physics 模式启用 (物体出生在其上, 需要真实支撑);
            # follow 模式物体是 kinematic 跟随, 台面碰撞只会挡住指爪下降 —— 物体出生点
            # 比 EF3R 台面拟合低 ~18mm (跨链米制残差), 指爪被台面顶住 → 物理误差 122mm
            # (实测 2026-10-29, 即 docs/11 §3.6 指爪几何限制在 follow 模式的表现)。
            # PICK 台面: 物体出生在**两指之间** (pad 抓取点), 从不在台面上休息 →
            # 台面碰撞对手指是纯干扰 (手下探时被顶开, 实测内距 26.4→43.1mm)。
            # 常关碰撞, 纯视觉; 落物由"重力对齐接物桌"负责。
            if "surface_pick_V" in od.files and len(od["surface_pick_V"]) > 0:
                _z1, _pick_actor = _surf_mesh_actor("ef3r_pick_support", od["surface_pick_V"],
                                                    od["surface_pick_F"], False)
            else:
                _pick_actor = None
            if "surface_place_V" in od.files and len(od["surface_place_V"]) > 0:
                _z2, _place_actor = _surf_mesh_actor("ef3r_place_support", od["surface_place_V"],
                                                     od["surface_place_F"], True)
                _support_name = "ef3r_place_support"
                obj_meta["support_z"] = (float(ring_surface[:, 2].max() + z_shift + _dh_scene)
                                         if ring_place else _z2)
            else:
                _place_actor = None
            # PLACE 台面碰撞延迟启用: 台面网格拟合自 AGILE FPose 漂移的放置参考,
            # z 范围 1.32~1.62m 悬在搬运走廊正前方 —— 手部轨迹 172 帧直接穿台, 物理臂
            # 全程被静态台面挡住 (实测 phys 6.3→59mm, 2026-10-29)。跟随期组清零,
            # RELEASE 时刻恢复 (正好接住切 dynamic 的物体)。
            _place_collider = None
            if _place_actor is not None:
                for comp in _place_actor.get_components():
                    if isinstance(comp, sapienp.PhysxRigidStaticComponent):
                        _place_collider = [(sh, list(sh.get_collision_groups()))
                                           for sh in comp.get_collision_shapes()]
                        for sh, _g in _place_collider:
                            sh.set_collision_groups([0, 0, 0, 0])
            # 重力对齐模式: 水平接物桌 (真实场景 pick/place 同桌)。
            # 高度用手度量系: pad@PICK 段最低点 - 2cm (EF3R 手部自标定与场景 VGGT
            # 之间存在 ~95mm 竖直偏差, 场景测的桌高会让上臂/相机永久穿模 —— 上臂在
            # 躯干高度 1.2-1.4m 实测 1231 帧接触)。碰撞只在 RELEASE 落物时启用
            # (构建即清零碰撞组, k_sw 恢复): APPROACH/PICK/LIFT/PLACE 全程臂不与桌
            # 接触 (RELEASED 撤手段数据里手掌下探到桌平面下 ~20cm, 真实人手是撤到
            # 桌沿外 —— 单一高度平面模型无法同时满足, 延迟碰撞是自洽近似);
            # Delaunay 台面网格转纯视觉。
            if _ga_dir is not None and _pick_actor is not None and "ring_segment_count" not in od.files:
                for comp in _pick_actor.get_components():
                    if isinstance(comp, sapienp.PhysxRigidStaticComponent):
                        for sh in comp.get_collision_shapes():
                            sh.set_collision_groups([0, 0, 0, 0])
                for comp in (_place_actor.get_components() if _place_actor else []):
                    if isinstance(comp, sapienp.PhysxRigidStaticComponent):
                        for sh in comp.get_collision_shapes():
                            sh.set_collision_groups([0, 0, 0, 0])
                # 顶面 = 全程 pad 最低点 - 6cm: RELEASED 撤手段手会下探 (数据里
                # 手掌低于抓取平面 ~2-17cm), 桌面必须给整条手部轨迹让路
                _z_table = float(pad_p[:, 2].min()) - 0.06
                obj_meta["catch_z"] = float(_z_table)   # sim 系接物桌顶 (验收落台判据用)
                _kr5 = d["grip_states"] >= 5 if "grip_states" in d.files else None
                _kr_xy = pad_p[min(int(np.argmax(_kr5[idx])), len(idx) - 1)][:2] \
                    if _kr5 is not None and _kr5.any() else pos_f.mean(0)[:2]
                _cx = np.asarray(_kr_xy, np.float64)
                _half = np.array([0.26, 0.26])
                bb_t = scene.create_actor_builder()
                bb_t.set_physx_body_type("static")
                bb_t.add_box_collision(sapien.Pose(p=[0, 0, 0]),
                                       half_size=[float(_half[0]), float(_half[1]), 0.02],
                                       material=grip_mat)
                import sapien.render as _sr3
                _tmat3 = _sr3.RenderMaterial()
                _tmat3.base_color = [0.80, 0.78, 0.74, 1.0]
                _tmat3.roughness = 0.85
                bb_t.add_box_visual(sapien.Pose(p=[0, 0, 0]),
                                    half_size=[float(_half[0]), float(_half[1]), 0.02],
                                    material=_tmat3)
                _table_actor = bb_t.build(name="ef3r_table_gravity")
                _table_actor.set_pose(sapien.Pose(p=[float(_cx[0]), float(_cx[1]),
                                                     float(_z_table - 0.02)]))
                # 延迟碰撞: 构建即清零, RELEASE (k_sw) 恢复接住落下的物体
                _place_collider = None
                for comp in _table_actor.get_components():
                    if isinstance(comp, sapienp.PhysxRigidStaticComponent):
                        _place_collider = [(sh, list(sh.get_collision_groups()))
                                           for sh in comp.get_collision_shapes()]
                        for sh, _g in _place_collider:
                            sh.set_collision_groups([0, 0, 0, 0])
                _support_name = "ef3r_table_gravity"
                obj_meta["support_z"] = _z_table
                obj_meta["gravity_table_half_m"] = [float(_half[0]), float(_half[1])]
                print(f"  重力对齐接物桌: 顶 z={_z_table:.3f} (全程pad最低-6cm, 手度量) "
                      f"中心 xy={np.round(_cx,2)} 半宽 {_half[0]:.2f}m, 碰撞@RELEASE 启用 "
                      f"(Delaunay 台面转纯视觉)")
            _np_pick = len(od["surface_pick_V"]) if "surface_pick_V" in od.files else 0
            _np_place = len(od["surface_place_V"]) if "surface_place_V" in od.files else 0
            print(f"  支撑台面 mesh: PICK {_np_pick}点 + PLACE {_np_place}点 "
                  f"(原点云真实表面 Delaunay 网格, geo 对齐)")
        print(f"  物体 actor: bbox {obj_meta['ext_mm']}mm (薄轴 {obj_meta['thin_mm']:.1f}mm 对准开合方向), 质量 0.12kg, "
              f"出生@pad抓取点 F{idx[k_lift] if k_lift < len(idx) else '?'}"
              f"{' (+PICK 台面碰撞)' if object_physics else ' (PICK 台面仅视觉)'}")

    # ---- 夹爪执行信号: data=逐帧跟人手张合(连续, 最忠实) | states=状态机离散控制 ----
    # states 模式: 闭合@LIFT 起 / 张开@RELEASED 起 (EgoFound3R state_labels),
    # 固定开合宽度 + 速率限幅 (~24mm/s), 更干净抗噪, 贴近真机"到状态才动作"的逻辑。
    grip_exec = grip
    if grip_mode == "states" and "grip_states" not in d.files:
        # 鲁棒回退: 旧版 targets npz 没有状态标签时, 不中断, 回退 data 模式
        print("  ⚠ grip_mode='states' 但 targets npz 无 grip_states 字段 → 回退 data 模式")
        grip_mode = "data"
    if grip_mode == "states":
        labels = d["grip_states"]
        _LIFT = int(np.argmax(labels >= 1))          # PICK 起: 人手在 PICK 段闭合 (臂速较慢, 物理臂跟得上)
        _REL = int(np.argmax(labels >= 5)) if (labels >= 5).any() else N
        _open = min(float(grip[:, 0].max()), 0.045)  # 全开宽 (clip 到关节上限, 对齐 duck)
        _grasp = grip_grasp if grip_grasp is not None else 0.015  # 抓握宽 (有物体时≈薄轴对齐的闭合量)
        grip_exec = np.empty((N, 2))
        tgt_w = np.where(np.arange(N) < _LIFT, _open, _grasp)
        tgt_w[_REL:] = _open
        rate = 0.0008                                # m/帧 ≈ 24mm/s @30fps
        cur = _open
        for k in range(N):
            cur += np.clip(tgt_w[k] - cur, -rate, rate)
            grip_exec[k] = [cur, cur]
        _k_settle = _LIFT + int(np.argmax(np.abs(grip_exec[_LIFT:, 0] - _grasp) < 1e-4))
        k_grav_on = _k_settle + 2                        # 闭合到位后再恢复重力 (指爪已接触)
        print(f"  夹爪控制 (states): 全开 {_open*1000:.0f}mm, 抓握 {_grasp*1000:.0f}mm, "
              f"闭合@F{_LIFT} (到位@F{_k_settle}), 张开@F{_REL} (速率限幅 {rate*1000:.1f}mm/帧)")

    # ---- 物理抓取: 开合由**物体几何**推导 + 确定性时序 (覆盖 data/states) ----
    # 为什么必须覆盖 data: 人手 data 信号在抓握段是 8.5~11.4mm (按 R1 标定换算内距
    # 约 14~20mm), 与 25.2mm 的物体几何无关, 两指会直接压在物体内部或悬空 ——
    # 都必须由物体薄轴反推。抓握宽度/内距一律走 large_range_track 的单一标定真源
    # (GRIP_INNER_A/K, 由 URDF 实测推导), 不在此处另写公式。
    if obj_actor is not None and object_physics:
        _thin = float(obj_meta.get("thin_mm", 20.0)) / 1000.0
        _q_grasp = lrt.grasp_width_for_object(_thin)
        _inner_grasp = lrt.GRIP_INNER_A + lrt.GRIP_INNER_K * _q_grasp
        _k_rel = k_release if k_release > 0 else N
        _open = min(float(grip[:, 0].max()), 0.045)
        # 闭合速率: 45mm/s @30fps。快于 states 的 24mm/s —— 必须在 LIFT 段内闭合
        # 完成并留出稳定帧, 否则还没夹住就切 dynamic。
        rate = 0.0015
        grip_exec = np.empty((N, 2))
        cur = _open
        for k in range(N):
            # 注意: 变量名不能叫 tgt —— 外层 tgt 是目标位姿列表 (pos,rot)
            _w = _open if (k < k_lift or k >= _k_rel) else _q_grasp
            cur += np.clip(_w - cur, -rate, rate)
            grip_exec[k] = [cur, cur]
        _k_settle = (k_lift + int(np.argmax(np.abs(grip_exec[k_lift:, 0] - _q_grasp) < 1e-4))
                     if k_lift < N else N - 1)
        k_grav_on = min(_k_settle + 2, N - 1)
        print(f"  夹爪控制 (物理抓取): 全开 {_open*1000:.0f}mm -> 理论抓握 {_q_grasp*1000:.1f}mm "
              f"(物体薄轴 {_thin*1000:.1f}mm, 内距 {_inner_grasp*1000:.1f}mm), "
              f"闭合@F{k_lift}, 张开@F{_k_rel}; 实际目标将在 kinematic 段按实测 w_eff 自适应 (A1)")
    # A1 状态: _grip_live[0] = set_targets 的实时开合目标 (None=用预生成曲线);
    # w_eff = kinematic 段实测内距的 plateau 最小值 (两指真实能夹到的宽度, 已含
    # 残余失配); 抓握目标 = w_eff - 3mm (总咬合 3mm, 两侧 1.5mm)。
    # 注意 _grip_live 必须无条件存在 —— set_targets 无条件引用它。
    _grip_live = [None]

    # ---- follow 模式 + 物体在抓: 执行开合钳位到"刚好接触物体" ----
    # data 信号可能闭到比物体薄轴更窄 (AGILE 链实测 2.3mm vs 物体 27.6mm), 指爪持续
    # 挤压 kinematic 物体产生无界反力拖偏整条臂 (实测 phys 6.3→64mm)。object_physics
    # 模式不钳位 (真实夹持力是抓取的前提); 钳位在 states 的 _k_settle 判定之后。
    if obj_actor is not None and not object_physics:
        _thin = float(obj_meta.get("thin_mm", 20.0)) / 1000.0
        # 内距 = 物体薄轴 + 1.5mm (只接触不挤压) => squeeze 取负值复用同一标定
        _q_touch = lrt.grasp_width_for_object(_thin, squeeze=-0.0015)
        _n_clamped = int((grip_exec < _q_touch).sum())
        grip_exec = np.maximum(grip_exec, _q_touch)
        print(f"  夹爪执行钳位 (follow+物体): 开合 ≥ {_q_touch*1000:.1f}mm "
              f"(物体薄轴 {_thin*1000:.1f}mm), 钳位 {_n_clamped}/{grip_exec.size} 个信号")

    # A1 状态 (无条件初始化): _grip_live[0]=None 时 set_targets 走预生成曲线;
    # 其余字段仅物理抓取使用。_mid_hist/_gstate_s/_hold_contact 见 A5。
    _grip_live = [None]
    _w_eff = [None]
    _w_prev = [None]
    _cur_q = [None]
    _mid_hist = np.full((N, 3), np.nan)     # 两指中点轨迹 (A5: HOLD 相对滑移分母)
    _gstate_s = (d["grip_states"][idx] if "grip_states" in d.files else None)
    _hold_contact = [0, 0]                  # [有接触采样数, 总采样数] (A5)

    _torso_on = torso_wbc and len(arm[0]) >= 10

    def set_targets(k):
        for i in range(6):
            jmap[f"right_arm_joint{i+1}"].set_drive_target(float(arm[k, i]))
        if _torso_on:
            for i in range(4):
                jmap[f"torso_joint{i+1}"].set_drive_target(float(arm[k, 6 + i]))
        # 物理抓取: 开合目标由主循环按**实测内距**在线更新 (A1, w_eff 自适应);
        # 其余模式仍走预生成的 grip_exec 曲线。
        _q1 = _grip_live[0] if _grip_live[0] is not None else float(grip_exec[k, 0])
        jmap["right_gripper_finger_joint1"].set_drive_target(float(_q1))
        jmap["right_gripper_finger_joint2"].set_drive_target(float(_q1))

    set_targets(0)
    robot.set_root_pose(bp.se2_pose(dx[0], dy[0], byaw[0]))
    # 初始位姿 = 第一帧 IK 解 (不从 home 起步): 切窗片段从任务中段开始, 首帧
    # 目标可能远在 1.5m 外 —— 物理臂从 home 追不上起点, 误差全程 >1.5m (实测
    # ee1 F0 pad_err=1529mm)。按第一帧解摆好臂/躯干, 200 子步只做稳定下沉。
    if not legacy_physics:
        # 初始位姿 = 第一帧 IK 解: 切窗片段从任务中段开始, 首帧目标可能远在
        # 1.5m 外 —— 物理臂从 home 追不上起点 (实测 ee1 F0 pad_err=1529mm)。
        # legacy=True 保持 home 起步 (旗舰链旧行为)。
        _q0 = np.asarray(robot.get_qpos()).copy()
        for _i in range(6):
            _q0[nm2i[f"right_arm_joint{_i+1}"]] = float(arm[0, _i])
        if _torso_on and len(arm[0]) >= 10:
            for _i in range(4):
                _q0[nm2i[f"torso_joint{_i+1}"]] = float(arm[0, 6 + _i])
        robot.set_qpos(_q0)
    rg.step_sim(scene, robot, 200)

    # ---- 相机: 自动布置固定机位 (正面|侧面) + 机器人机载第一视角 + 跟随视角 ----
    view_base = _camera_base[idx] if registered_scene else base_xy
    view_pad = _camera_pad[idx] if registered_scene else pad_p
    view_yaw = ref_yaw if registered_scene else byaw
    fixed, ctr = lrt.auto_cameras(scene, rg.look_at_pose, view_base, view_pad)
    ego = scene.add_camera(name="ego", width=480, height=360, fovy=np.deg2rad(75),
                           near=0.02, far=500)
    track = scene.add_camera(name="track", width=480, height=360, fovy=np.deg2rad(50),
                             near=0.05, far=500)
    cams = [fixed[0], fixed[1], ego, track]

    def update_cams(k):
        # 第一视角 = **机器人机载 FPV**（相机在躯干顶端前方，视线锁定当前手部目标）
        # 为什么不再用数据集 ego 相机：第一人称数据的相机原点 = 人的眼睛，而机器人
        # 底盘正好站在同一位置 -> 相机落到机器人头部正上方/正后方，画面被自己的头和
        # 手臂占满（实测 seq456/chairbottle 都是整屏手臂）。SAPIEN 3.0.0b1 没有
        # 按相机隐藏 actor 的 API，"保留人眼位姿又排除自身"做不到。
        # 机载 FPV 物理自洽：相机位于身体最前上方、随底盘运动、永远看向手；
        # 且这正是真机部署时机器人自己的第一视角，比"人眼位姿"更有意义。
        # fwd_off=0.30 越过肩线(0.14)防自遮挡; yaw 随底盘旋转, 相机始终在身体前方
        _eye, _look = lrt.fpv_cam_pose(view_base[k], cf["forward"], cf["head_z"],
                                       view_pad[k], fwd_off=0.30, yaw=view_yaw[k])
        ego.entity.set_pose(rg.look_at_pose(_eye, _look))
        # 跟随视角: 斜前上方俯视 —— 相机抬高 (2.9m) 加大俯角, 越过场景体素云
        # (桌面点云在臂高形成"墙", 平视被挡, 抓取过程不可见); 俯视桌面可见
        # 臂-物交互全程。
        b = view_base[k]
        track.entity.set_pose(rg.look_at_pose([b[0] - 1.2, b[1] + 1.6, 2.9],
                                              [b[0], b[1] + 0.30, 1.05]))

    def render4(k):
        update_cams(k)
        scene.update_render()
        for c in cams:
            c.take_picture()
        imgs = [(np.clip(c.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8) for c in cams]
        return np.concatenate([np.concatenate(imgs[:2], axis=1),
                               np.concatenate(imgs[2:4], axis=1)], axis=0)

    # ---- 物理时钟 (EE17 修复): SAPIEN 子步 250Hz 固定, 按源轨迹时间戳推进 ----
    # 源 30fps 帧间隔 33.33ms -> 每帧平均 250/30 = 8.333 子步, 累计时钟分配
    # 8/8/9 交替; 600 帧 -> 累计 5000 子步 = 20.000s (旧版固定 40 子步 = 4.8x)。
    # clock_mode="legacy" 保留旧固定 phys_steps 子步 (AGILE 456 旗舰接触抽签依赖)。
    _clock_source = (clock_mode == "source")
    _handoff_freeze_frames = int(handoff_freeze_frames)
    _fps_src = float(d.get("fps", 30.0))
    _dt_sub = 1.0 / 250.0
    _sub_per_frame = 250.0 / _fps_src
    _sub_acc = 0.0
    _sub_total = 0
    _src_time = np.arange(N, dtype=np.float64) / _fps_src
    _phys_time = np.zeros(N)
    _steps_frame = np.zeros(N, dtype=np.int64)
    STEPS = phys_steps
    frames = []
    # 物理跟随实测: IK 解是"运动学理想", PD 驱动能否跟上要看执行层。逐帧记录
    # 实际 pad 位姿 vs 目标, 与 IK 层误差对比即可区分"解不准"和"动力跟不上"。
    phys_err = np.zeros(N); phys_ori = np.zeros(N)
    phys_err_next = np.zeros(N)      # 与"下一帧目标"的距离 -> 判断是否为一拍滞后
    phys_p = np.zeros((N, 3))        # 物理执行后的真实 pad 位置 (轨迹对比图)
    _obj_grasped = False             # follow 模式: 物理 pad 追上目标后置 True
    _k_rel_dyn = N                   # physics 模式: 夹爪实测闭合到位、释放重力的帧
    # A2-后处理: 释放后手指张开期间**冻结臂/底盘** —— 手指未张净就收缩,
    # 指面会扫过刚落桌的物体把它碰飞 (实测 77mm 滑移/弹飞, 逐次随机 = 接触混沌);
    # 物理上等价于"人手先张净再撤"。帧数按重定时缩放 (N/75 ≈ 0.13s 墙钟)。
    _rel_hold = max(4, int(round(N / 75)))
    _phys_pad_prev = (pad_p[0], rot0[0])
    _t_phys = time.time()
    # 执行轨迹 (数据集核心产物): base [xyz qw qx qy qz] / 全关节 q / 物体 [xyz q]
    _qdim = len(np.asarray(robot.get_qpos()))
    _exec_q = np.zeros((N, _qdim))
    _exec_base = np.zeros((N, 7))
    _exec_obj = np.full((N, 7), np.nan)
    _exec_grip = np.full((N, 7), np.nan)   # 物理 gripper link 实际位姿 (FK 真值)
    _cmd_base = np.zeros((N, 4))           # WBC 底盘指令 (dx,dy,byaw)
    _contact_impulses = np.zeros((N, 4))   # summed over all physics steps
    _bilateral_impulses = np.zeros((N, 2))
    _bilateral_steps = np.zeros(N, dtype=int)
    _support_steps = np.zeros(N, dtype=int)
    _object_dynamic = np.zeros(N, dtype=bool)
    _grasp_gap = np.full(N, np.nan)
    _preload_frame = N
    _actual_release = k_release if object_physics and k_release < N and not ring_place else None
    _contact_names = [f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2",
                      _support_name, "ground"]
    _contact_columns = {name: i for i, name in enumerate(_contact_names)}
    # 有符号竖直冲量 (用户要求): 旧统计取模, 铁圈侧撞也记成支撑。逐接触点取
    # impulse 的世界系 z 分量带符号累加 —— 正=托住漏斗 (向上), 负=压/侧撞。
    _contact_impulses_signed = np.zeros((N, 4))

    def record_object_contacts(frame):
        step_impulses = np.zeros(4)
        step_signed = np.zeros(4)
        for contact in scene.get_contacts():
            first, second = (body.get_entity().get_name() for body in contact.bodies)
            other = second if first == "ef3r_object" else first if second == "ef3r_object" else None
            column = _contact_columns.get(other)
            if column is not None:
                step_impulses[column] += sum(np.linalg.norm(point.impulse) for point in contact.points)
                step_signed[column] += sum(float(np.asarray(point.impulse)[2]) for point in contact.points)
        _contact_impulses[frame] += step_impulses
        _contact_impulses_signed[frame] += step_signed
        if np.all(step_impulses[:2] > 1e-7):
            _bilateral_impulses[frame] += step_impulses[:2]
            _bilateral_steps[frame] += 1
        if step_impulses[2] > 1e-7:
            _support_steps[frame] += 1

    track_object_contacts = object_physics and _obj_comp is not None
    for k in range(N):
        if _clock_source:
            # 累计时钟: 按源帧时间戳分配子步 (30fps -> 8/8/9 交替)
            _sub_acc += _sub_per_frame
            steps_k = int(np.floor(_sub_acc))
            _sub_acc -= steps_k
            steps_k = max(steps_k, 1)
        else:
            steps_k = int(STEPS)
        _sub_total += steps_k
        _steps_frame[k] = steps_k
        _phys_time[k] = _sub_total * _dt_sub
        _ke = k
        if object_physics and k_release < N and k >= k_release:
            if ring_place and _actual_release is None and k >= 2:
                support = _support_steps[k-2:k] >= max(1, steps_k // 2)
                ground = _contact_impulses[k-2:k, 3] > 1e-7
                still = np.linalg.norm(_exec_obj[k-1, :3] - _exec_obj[k-2, :3]) <= 0.005
                if np.all(support & _object_dynamic[k-2:k] & ~ground) and still:
                    _actual_release = k
                    print(f"  Measured support at F{k}; releasing after dwell")
            if _actual_release is None or k < _actual_release + _rel_hold:
                _ke = k_release
            else:
                _ke = k - (_actual_release - k_release)
        # 底盘**子步连续插值** (替代单次瞬移): 瞬移 = 无限加速度, 动态物体惯性
        # 冲量每帧锤击腕关节 (实测 ee1 腕 j4/j5/j6 漂移至 34/49/58°)。插值后底盘
        # 以有界速度 (~0.75m/s, 真实轮式底盘量级) 连续移动 —— 复刻移动的正确物理。
        _pose1 = bp.se2_pose(dx[_ke], dy[_ke], byaw[_ke])
        _cmd_base[k] = [dx[_ke], dy[_ke], byaw[_ke], _ke]
        if not base_interp:
            # legacy: 传送在物体块**之前** (652d897 原序) —— 胶合/物体逻辑读
            # 新基底位姿; 顺序改变 = 每帧 1 帧基底偏移 → 接触抽签重排 (456 实测
            # 120mm→19mm, 二分定位至此)
            robot.set_root_pose(_pose1)
        marker.update(pad_p[k])
        set_targets(_ke)
        if obj_actor is not None:
            # 切 dynamic 的时刻: 按**实测内距**判定, 不按帧数。
            # 物体 kinematically 悬在两指之间的中点, 所以两指闭合时**不可能穿过它**:
            # 实测内距会停在 ≈ 薄轴 (25.2mm) 不再下降。于是"内距 ≤ 薄轴+1mm"正好在
            # 两指刚接触物体的那一帧触发 —— 此时交接重力才是"已夹住"的状态。
            # 早于此交接 (例如内距还有 1mm 间隙) 会让物体先自由落体: 每控制帧 40 个
            # 1/250s 子步 = 0.16s, 一帧就掉 12cm, 根本等不到指爪合拢 (agile456 实测)。
            # 晚于此交接 (内距已小于薄轴, 两指各嵌进物体) 则 depenetration 冲量把
            # 物体弹出 (旧 +2mm 判据实测 F137 触发时内距 23.7mm -> 16 帧落地)。
            if object_physics:
                _l1c = robot.find_link_by_name(f"{side}_gripper_finger_link1").get_entity_pose().p
                _l2c = robot.find_link_by_name(f"{side}_gripper_finger_link2").get_entity_pose().p
                _inner_now = float(np.linalg.norm(_l1c - _l2c)) - lrt.GRIP_INNER_OFF
                _grasp_gap[k] = _inner_now
                width = None
                # ---- A1: w_eff 自适应抓握宽度 (kinematic 段实测) ----
                # 物体 kinematically 悬在两指中点, 两指闭合**不可能穿过它**: 实测内距
                # 降到接触宽度后走平 —— 走平值 w_eff 就是"两指真实能夹到的宽度"(已含
                # 残余失配/盒子歪斜)。稳定 3 帧后把目标切到 w_eff - 3mm (总咬合 3mm,
                # 两侧 1.5mm, 在 2mm 安全线内), 让 PD 预紧建立后再交接重力。
                if k >= k_lift:
                    if _cur_q[0] is None:
                        _cur_q[0] = float(grip_exec[k, 0])      # 从预生成曲线接棒
                    if k < _k_rel_dyn:
                        if k >= k_lift + 3 and np.all(_bilateral_steps[k-3:k] >= min(2, steps_k)):
                            width = lrt.stable_grasp_width(
                                _grasp_gap[k-2:k+1], _contact_impulses[k-3:k, :2],
                                2 * obj_meta["box_half_m"][2])
                            if width is not None:
                                _w_eff[0] = width
                                if _w_prev[0] is None:
                                    _w_prev[0] = width
                                    _preload_frame = k
                                    print(f"  Bilateral preload F{k}: stable gap {width*1000:.1f}mm")
                        _rate = rate
                    else:
                        # 交接后: 快速预紧 (8mm/帧) —— 法向力 = 刚度×侵入量, 爬坡越快
                        # 物体在重力下的初始下沉越小 (实测 kinematic 交接帧间隙
                        # 0.25mm 内无力, 慢爬坡 1 帧滑 17.6mm)。
                        _rate = 0.008
                    # 单位!!! _w_eff 是**内距 (m)**, _cur_q 是**关节量**: 内距->q 必须
                    # 过标定换算, 否则 23.2mm 被当成 q=0.0232 (内距 43mm), 手指被自己
                    # 撑开 (实测 F160 内距 37.4 / F176 43.1 = 恰好是这个错值)。
                    # 释放段 (k>=_k_rel): 回到预生成开合曲线 (张开!) —— live 目标若
                    # 锁死在预紧值, 手指永不张开, 缩手时刮着物体走 (实测 slide 52mm)。
                    # 触桌即张指: 物体落到接物桌后手指继续压着会拖行 (实测 52mm);
                    # 张指让手干净撤出。
                    if _actual_release is not None and k >= _actual_release:
                        _tgt_q = _open
                    elif _w_prev[0] is not None:
                        squeeze = float(os.environ.get("CB_SQUEEZE_NOMINAL", 0.003))
                        _tgt_q = lrt.grasp_width_for_object(_w_eff[0], squeeze=squeeze)
                    else:
                        _tgt_q = _q_grasp
                    _cur_q[0] += float(np.clip(_tgt_q - _cur_q[0], -_rate, _rate))
                    _grip_live[0] = _cur_q[0]
                _closed = width is not None and abs(_inner_now - width) <= 0.003
                if k >= _preload_frame + 2 and k < min(_k_rel_dyn, k_release) and _closed:
                    _k_rel_dyn = k
                    print(f"  Dynamic handoff F{k}: bilateral impulse "
                          f"{_bilateral_impulses[k-1].round(5)} N·s over {_bilateral_steps[k-1]}/{steps_k} substeps; "
                          f"gap {_inner_now*1000:.1f}mm")
                # ---- A2: 交接后阻性自检, 杜绝"假夹住"静默滑移 ----
                if _k_rel_dyn < N and k >= _k_rel_dyn:
                    _n_post = k - _k_rel_dyn
                    _midc = (np.asarray(_l1c) + np.asarray(_l2c)) / 2.0
                    _o_now = obj_actor.get_pose().p
                    _d_obj = float(np.linalg.norm(_midc - np.asarray(_o_now)))
                    _hb2 = obj_meta.get("box_half_m", [0.025, 0.025, 0.0125])
                    _Ro2 = obj_actor.get_pose().to_transformation_matrix()[:3, :3]
                    _yax2 = _phys_pad_prev[1][:, 1]
                    _span_y = float(sum(abs(_Ro2[:, i] @ _yax2) * 2 * _hb2[i]
                                        for i in range(3))) * 1000
                    if _n_post < 5:
                        print(f"    [交接自检 +{_n_post}] 内距 {_inner_now*1000:.1f}mm | "
                              f"物体-指中点 {_d_obj*1000:.1f}mm | 沿开合轴跨度 {_span_y:.1f}mm")
                    if k % 4 == 0:
                        # 蠕滑方向探针: 物体相对 pad 的位移分解到三轴 —— 开合轴
                        # (两指面法向) / 指长 (指尖->基部) / 嘴向 (指条宽度方向,
                        # 开放无遮挡)。哪轴漂移即哪向逃逸。
                        _Rp = _phys_pad_prev[1]
                        _pp = np.asarray(_phys_pad_prev[0])
                        _v = np.asarray(_o_now) - _pp
                        _flen = _midc - _pp
                        _flen = _flen / (np.linalg.norm(_flen) + 1e-9)
                        _fmouth = np.cross(_Rp[:, 1], _flen)
                        print(f"    [pad系探针 F{k}] 开合轴 "
                              f"{float(_v @ _Rp[:, 1])*1000:+.1f}mm | 指长 "
                              f"{float(_v @ _flen)*1000:+.1f}mm | 嘴向 "
                              f"{float(_v @ _fmouth)*1000:+.1f}mm")
                    if not obj_meta.get("slipped") and _n_post <= 20 and \
                            _inner_now < _w_eff[0] - 0.001 and _d_obj > 0.020:
                        obj_meta["slipped"] = True
                        print(f"    ⚠ 物体已脱手 F{k}: 内距 {_inner_now*1000:.1f}mm 已低于 "
                              f"w_eff {_w_eff[0]*1000:.1f}mm (两指没被挡住) 且物体偏离 "
                              f"指中点 {_d_obj*1000:.0f}mm —— 抓取失败, 非静默滑移")
                _mid_hist[k] = (np.asarray(_l1c) + np.asarray(_l2c)) / 2.0
            _k_sw = (k_release if not object_physics else _k_rel_dyn)
            if _obj_comp is not None and not object_physics:
                # 抓取跟随, 物理时序 (2026-10-29): 物体静止在出生点; 物理 pad 追上
                # (|phys-IK| < 40mm) 才算"抓住" → 跟随**物理** pad (被真实手抓着),
                # RELEASED 切 dynamic 落支撑台。旧版 glued 到 IK pad: 物理手指滞后
                # ~60mm 时 kinematic 物体反而把手指推着走 (实测 phys 6.3→64mm)。
                if k == k_lift:
                    _obj_grasped = False
                    _phys_pad_prev = (pad_p[k_lift], rot0[k_lift])
                if k_lift <= k < _k_sw and not _obj_grasped and \
                        np.linalg.norm(_phys_pad_prev[0] - pad_p[k]) < 0.04:
                    _obj_grasped = True
                    print(f"  物体抓取 (物理 pad 追上目标): F{k}")
                if _obj_grasped and k < _k_sw:
                    Rk = _phys_pad_prev[1] @ _R_rel
                    qk = Rot.from_matrix(Rk).as_quat()[[3, 0, 1, 2]]
                    obj_actor.set_pose(sapien.Pose(p=list(_phys_pad_prev[0]), q=qk))
                if k == _k_sw and _obj_grasped:
                    if obj_shape_groups is not None:
                        for sh, g in zip(_obj_comp.get_collision_shapes(), obj_shape_groups):
                            sh.set_collision_groups(g)          # 恢复: 与台面碰撞才能落住
                    if _place_collider is not None and table_collision:
                        for sh, g in _place_collider:
                            sh.set_collision_groups(g)          # 恢复: 接住落下的物体
                    _obj_comp.set_kinematic(False)   # RELEASED: 落支撑台 (PLACE 台面有碰撞)
                    _obj_comp.set_disable_gravity(False)
            elif _obj_comp is not None and object_physics and k_lift <= k < min(_k_sw, k_release):
                # 物体位置 = **实际两指内表面中点**, 不是 pad 原点:
                #   pad 原点(gripper_link)到两指中点实测差 3~12mm (沿开合轴), 物体若
                #   按 pad 原点放, 会整体偏向一侧指面 —— 实测 F124 物体在指1局部系
                #   y=+58mm (内距 74mm), 只贴着 finger2 内表面, 切 dynamic 立刻掉。
                #   用两指中点定位 => 物体恒在开合轴正中, 两侧对称接触。
                _l1p = robot.find_link_by_name(f"{side}_gripper_finger_link1").get_entity_pose().p
                _l2p = robot.find_link_by_name(f"{side}_gripper_finger_link2").get_entity_pose().p
                _mid = (np.asarray(_l1p) + np.asarray(_l2p)) / 2.0
                Rk = _phys_pad_prev[1] @ _R_rel
                qk = Rot.from_matrix(Rk).as_quat()[[3, 0, 1, 2]]
                if k == k_lift:
                    print(f"  物体出生 -> 两指中点 {np.round(_mid, 3)} "
                          f"(pad 原点 {np.round(_phys_pad_prev[0], 3)}, 差 "
                          f"{np.linalg.norm(_mid - np.asarray(_phys_pad_prev[0]))*1000:.0f}mm)")
                obj_actor.set_pose(sapien.Pose(p=list(_mid), q=qk))
                obj_trace[k] = _mid
            # [ee17] 交接冻结: 切 dynamic 前先让手指在 kinematic 物体上压入
            # squeeze 过盈量, 接触力建立后再放开 (滑移就发生在切 dynamic 后几帧)。
            # handoff_freeze_frames=0 = 旧行为 (EE17 用 8 帧 ≈ 0.27s)。
            _k_sw_eff = _k_sw + (_handoff_freeze_frames if object_physics else 0)
            if _obj_comp is not None and _k_sw <= k < _k_sw_eff and object_physics:
                _mid_f = (np.asarray(robot.find_link_by_name(
                    f"{side}_gripper_finger_link1").get_entity_pose().p)
                    + np.asarray(robot.find_link_by_name(
                        f"{side}_gripper_finger_link2").get_entity_pose().p)) / 2.0
                obj_actor.set_pose(sapien.Pose(p=list(_mid_f), q=qk))
                obj_trace[k] = _mid_f
            if _obj_comp is not None and k == _k_sw_eff and object_physics:
                _obj_comp.set_kinematic(False)      # physics: 压入建立后切 dynamic
                _obj_comp.set_disable_gravity(False)
                # 交接瞬间提高指驱动刚度 (Plan B-2): 法向力 = 刚度×侵入量, 物体切
                # dynamic 时指-物间隙 ~0.25mm 无力, 力爬坡期间物体会下滑 (实测 1 帧
                # 17.6mm)。刚度 8000→30000 让力爬坡快 3.75x, 初始下沉压进 ≤10mm。
                for nm in (f"{side}_gripper_finger_joint1", f"{side}_gripper_finger_joint2"):
                    if nm in jmap:
                        if finger_force_limit is not None:
                            # 恒力夹持 (标准仿真抓取手法): 位置目标在物体内 (伺服咬合)
                            # + force_limit 封顶 = 驱动力 ≤cap/侧。关键收益是去穿透
                            # 弹射力同被封顶: 旧版 K=30000 无上限时压入误差×刚度可达
                            # ~100N 持续压力, 触发求解器 max_depenetration_velocity=
                            # 1m/s 弹射 (chairbottle F208 1 帧逃逸根因, 10+ 变体穷尽
                            # 无效) —— cap 后压入深度自限 (cap/K=1mm), 逃逸消失。
                            # 默认 None=旧版 30000/1200 无上限 (AGILE 456 旗舰依赖
                            # 该组合的接触抽签, 加 cap 反而回退 219→126mm)。
                            # 携带段软指爪: 底盘逐帧 kinematic 瞬移 = 无限加速度,
                            # 硬指爪 (30000) 把物体惯性冲量直传腕关节 —— 腕 j4/j5/j6
                            # 棘轮式漂移 (实测 ee1: 3→34/49/58°, pad 甩至 1.7m)。
                            # 4000 软指爪以形变吸收冲量; 30N cap 保夹持力上限。
                            jmap[nm].set_drive_property(stiffness=4000.0, damping=150.0,
                                                        force_limit=finger_force_limit)
                        else:
                            jmap[nm].set_drive_property(stiffness=30000.0, damping=1200.0)
                print(f"  指驱动交接后: {'30000/150 + force_limit=%.0fN (恒力夹持)' % finger_force_limit if finger_force_limit is not None else '30000/1200 (旧版无上限)'}")
                # 接物桌碰撞组在此恢复 (原来只在 follow 分支恢复 → 物体穿桌直接落地,
                # 实测末态 z 0.013 vs 台面 1.213)
                if _place_collider is not None and table_collision:
                    for sh, g in _place_collider:
                        sh.set_collision_groups(g)
        if base_interp:
            # 底盘子步连续插值 (mocap 移动复刻链): 瞬移 = 无限加速度, 动态物体
            # 惯性冲量锤击腕关节。插值后有界速度 (~0.75m/s 真实轮式量级)。
            # 默认 False = 旧版单次瞬移 (456 旗舰接触抽签依赖旧动力学)。
            _p0v = np.asarray(robot.get_root_pose().p, float)
            _q0v = np.asarray(robot.get_root_pose().q, float)
            _p1v = np.asarray(_pose1.p, float)
            _q1v = np.asarray(_pose1.q, float)
            for _si in range(steps_k):
                _a = (_si + 1) / steps_k
                _pq = _p0v + (_p1v - _p0v) * _a
                _qq = _q0v + (_q1v - _q0v) * _a
                _qq /= np.linalg.norm(_qq)
                robot.set_root_pose(sapien.Pose(p=_pq, q=_qq))
                rg.step_sim(scene, robot, 1)
                if track_object_contacts:
                    record_object_contacts(k)
        elif track_object_contacts:
            for _ in range(steps_k):
                rg.step_sim(scene, robot, 1)
                record_object_contacts(k)
        else:
            rg.step_sim(scene, robot, steps_k)
        if track_object_contacts:
            _object_dynamic[k] = not _obj_comp.get_kinematic()
        # 数据集产物: 每帧**实际执行状态** (此前只有 IK 目标, 执行轨迹被丢弃)
        _pq = robot.get_root_pose()
        _exec_base[k] = [*_pq.p, *_pq.q]
        _exec_q[k] = np.asarray(robot.get_qpos())
        _gl = robot.find_link_by_name(f"{side}_gripper_link")
        if _gl is not None:
            _gp = _gl.get_pose()
            _exec_grip[k] = [*_gp.p, *_gp.q]
        if obj_actor is not None:
            _op = obj_actor.get_pose()
            _exec_obj[k] = [*_op.p, *_op.q]
            obj_trace[k] = _op.p
            if k in (124, 140, 160, 176, 200):    # 抓取时刻诊断: PhysX 接触对 + 指爪位姿
                _l1 = robot.find_link_by_name(f"{side}_gripper_finger_link1").get_pose().p
                _l2 = robot.find_link_by_name(f"{side}_gripper_finger_link2").get_pose().p
                _o = obj_actor.get_pose().p
                _ct = []
                for c in scene.get_contacts():
                    try:
                        bs = [b.get_entity().get_name() if hasattr(b, "get_entity") else "?"
                              for b in c.bodies]
                    except Exception:
                        bs = ["?"]
                    if any("ef3r_object" in str(x) for x in bs):
                        _ct.append(f"{bs[0]}<->{bs[1]}")
                _L1 = robot.find_link_by_name(f"{side}_gripper_finger_link1")
                _p1 = _L1.get_pose()
                _rel = _p1.to_transformation_matrix().T[:3, 3] @ np.eye(3)  # noqa
                _rel = np.linalg.inv(_p1.to_transformation_matrix()[:3, :3]) @ (_o - _p1.p)
                # 夹爪实际内距必须**实测两指间距**再减指厚, 不能读 qpos:
                # 探针发现 get_dof() 对两指返回同一个值 (=1), 读到的是别的关节, 数字无意义。
                _gap = float(np.linalg.norm(_l1 - _l2))
                _inner = _gap - lrt.GRIP_INNER_OFF  # 减两指碰撞盒的 2*(0.01 偏置 + 0.005 半厚)
                _q_real = (_gap - lrt.GRIP_LINK_GAP0) / 2.0
                _tg = [float(_grip_live[0]) if _grip_live[0] is not None
                       else float(grip_exec[k, 0])]*2
                _pad2mid = float(np.linalg.norm(_phys_pad_prev[0] - (_l1 + _l2) / 2)) * 1000
                # 物体沿夹爪开合方向 (y) 的**真实**跨度: 用 actor 实际旋转 + 盒子全尺寸
                # (不能用 od["verts"]: 那是 EF3R y-up 世界系, 与 sim 世界系不同)
                _hb = obj_meta.get("box_half_m", [0.025, 0.025, 0.0125])
                _Ro = obj_actor.get_pose().to_transformation_matrix()[:3, :3]
                _yax = _phys_pad_prev[1][:, 1]
                _thin_on_y = float(sum(abs(_Ro[:, i] @ _yax) * 2 * _hb[i]
                                       for i in range(3))) * 1000
                print(f"    [夹爪 F{k}] q 目标={_tg[0]:.4f} 实测(由指距反算)={_q_real:.4f} | "
                      f"内距 {_inner*1000:.1f}mm | 物体沿开合轴跨度 {_thin_on_y:.1f}mm | "
                      f"盒子半尺寸mm {np.round(np.array(_hb)*1000,1)}")
                print(f"    [诊断 F{k}] 指中点 {np.round((_l1+_l2)/2,3)} | 物体 {np.round(_o,3)} | "
                      f"距 {np.linalg.norm((_l1+_l2)/2-_o)*1000:.0f}mm | 接触: {_ct or '无'} | "
                      f"物体在指1局部系 {np.round(_rel*1000,0)}mm (x沿指/y开合/z)")
        _p, _R = bp._gripper_pad_fk(robot, side, Rsel,
                                    robot.get_qpos()[np.array(Rsel)])
        if obj_actor is not None:
            # 供下一帧物体跟随**物理** pad (follow: 追上目标后; physics: 全程)
            _phys_pad_prev = (_p, _R)
        phys_p[k] = _p
        phys_err[k] = float(np.linalg.norm(_p - pad_p[k])) * 1000.0
        phys_err_next[k] = float(np.linalg.norm(_p - pad_p[min(k + 1, N - 1)])) * 1000.0
        _c = (np.trace(tgt[k][1].T @ _R) - 1.0) / 2.0
        phys_ori[k] = float(np.degrees(np.arccos(np.clip(_c, -1.0, 1.0))))
        frames.append(render4(k) if not phys_only else None)
        if (k + 1) % 20 == 0:
            print(f"  帧 {k+1}/{N}")

    print(f"\n  物理跟随 vs IK 理想 (mm): 物理 mean {phys_err.mean():.1f} p95 "
          f"{np.percentile(phys_err,95):.1f} max {phys_err.max():.1f}  |  "
          f"IK mean {pos_err.mean():.1f} p95 {np.percentile(pos_err,95):.1f} "
          f"max {pos_err.max():.1f}")
    print(f"  物理姿态 vs IK 理想 (deg): 物理 mean {phys_ori.mean():.2f} max {phys_ori.max():.2f}  |  "
          f"IK mean {ori_err.mean():.2f} max {ori_err.max():.2f}")
    if _transit.any():
        print(f"  物理跟随分解: 正常帧 mean {phys_err[_norm].mean():.1f}mm "
              f"p95 {np.percentile(phys_err[_norm], 95):.1f}mm | "
              f"过渡帧 mean {phys_err[_transit].mean():.1f}mm "
              f"(占物理误差总量 {100*np.sum(phys_err[_transit])/max(phys_err.sum(),1e-9):.0f}%)")
    _lag_hit = float(np.mean(phys_err_next < phys_err))
    print(f"  滞后检验: 实际末端更接近**下一帧**目标的比例 {_lag_hit*100:.0f}% "
          f"(≈50%=无滞后/随机, >70%=明显滞后一拍) | 对下一帧 mean {phys_err_next.mean():.1f}mm "
          f"vs 对当前帧 {phys_err.mean():.1f}mm")
    # 物理层指标补进指标 JSON (此前只在终端打印): 更新 L495 已写入的 payload 并重写同文件
    if obj_actor is not None:
        z0 = obj_trace[0, 2]
        lift = (obj_trace[:, 2].max() - z0) * 1000
        _lb = d["grip_states"][idx] if "grip_states" in d.files else None
        # HOLD = 标签 3 (APPROACH0/PICK1/LIFT2/HOLD3/PLACE4/RELEASED5): 手持搬运段
        # 滑移指标只算纯 HOLD: PLACE 段物体落桌、手继续走, 相对距离变大是物理
        # 正确的放置 (实测段末 274.7mm 全是它), 不是滑移。
        _hold_m = ((_lb == 3) if _lb is not None else np.zeros(N, bool))
        hold_slip = (np.linalg.norm(obj_trace[_hold_m, :2] - obj_trace[_hold_m, :2].mean(0), axis=1).max() * 1000
                     if _hold_m.any() else 0.0)
        placed_off = float(np.linalg.norm(obj_trace[-1, :2] - obj_meta.get("placed_xy", obj_trace[-1, :2])) * 1000)
        if object_physics:
            _avg_steps = max(1, int(round(_sub_total / max(N, 1))))
            bilateral = _bilateral_steps >= min(2, _avg_steps)
            airborne = _object_dynamic & bilateral & ~(_contact_impulses[:, 2:] > 1e-7).any(axis=1)
            handoff = np.flatnonzero(_object_dynamic)
            lift = (max(0.0, float(obj_trace[airborne, 2].max()
                    - obj_trace[max(0, handoff[0]-1), 2]) * 1000) if airborne.any() else 0.0)
            _hold_contact = [int(np.count_nonzero(_hold_m & _object_dynamic & bilateral)),
                             int(np.count_nonzero(_hold_m))]
            obj_meta.update(dynamic_handoff_frame=int(handoff[0]) if len(handoff) else None,
                            actual_release_frame=_actual_release,
                            requested_release_frame=k_release if k_release < N else None,
                            intended_support=_support_name,
                            ring_approach_enabled=ring_place)
        _grasp_ok = bool(lift > 30.0)
        obj_meta.update(lift_mm=float(lift), hold_slip_mm=float(hold_slip),
                        placed_off_mm=placed_off, grasped=_grasp_ok)
        print(f"  物体结果: 抓起 {'✓' if _grasp_ok else '✗'} (抬升 {lift:.0f}mm) | "
              f"HOLD 滑移 {hold_slip:.0f}mm | 放置偏差 {placed_off:.0f}mm (FPose 参考, 含其漂移)")
        _kr = (_actual_release if object_physics else
               (int(np.argmax(_lb >= 5)) if _lb is not None and (_lb >= 5).any() else None))
        _skid = None
        _drift = None
        # 期望落高按**实际落地姿态**的竖直半跨度算 (盒在重力向的半投影):
        # 释放时夹爪开合轴若倾斜, 物体以棱边立着落桌 (EF3R 链实测重心高 37mm
        # = 73×24 棱边支撑, 平躺假设会误判 +25mm)。触桌检测与验收共用。
        _bh = obj_meta.get("box_half_m", [0.025, 0.025, 0.0125])
        _half_vert = float(sum(abs(obj_actor.get_pose().to_transformation_matrix()[:3, :3][:, i]
                                   @ np.array([0, 0, 1.0])) * _bh[i]
                               for i in range(3))) if obj_actor is not None else _bh[2]
        if _kr is not None:
            _support_hits = np.flatnonzero((_contact_impulses[_kr:, 2] > 1e-7)
                                          & _object_dynamic[_kr:]
                                          & (_contact_impulses[_kr:, 3] <= 1e-7))
            if object_physics and len(_support_hits):
                _td = _kr + int(_support_hits[0])
                _drift = float(np.linalg.norm(obj_trace[_td, :2] - obj_trace[_kr, :2]) * 1000)
                _skid = float(np.linalg.norm(obj_trace[-1, :2] - obj_trace[_td, :2]) * 1000)
                obj_meta.update(slide_mm=_skid, inflight_drift_mm=_drift, touchdown_frame=_td)
            obj_meta["landed_z"] = float(obj_trace[-1, 2])
        # ---- A5: 统一验收指标 (物理抓取判据, 不以 FPose 参考为准) ----
        if object_physics:
            # HOLD 相对滑移: 物体相对两指中点的漂移 (真物理滑移; 旧 XY-spread 把
            # 手持搬运位移也算进去, 不可用)
            _rel = None
            _creep = None
            _held = _hold_m & _object_dynamic & bilateral
            if _held.any():
                _relv = np.linalg.norm(obj_trace[_held] - _mid_hist[_held], axis=1)
                _rel = float(np.nanmax(_relv) * 1000) if len(_relv) else None
                _creep = float((np.nanmax(_relv) - np.nanmin(_relv)) * 1000)
                _hi = int(np.nanargmax(_relv))
                _hold_frames = np.where(_held)[0]
                _prof = " ".join(f"{v*1000:.0f}" for v in _relv[::3])
                print(f"    HOLD 相对滑移: 段内漂移增量 {_creep:.1f}mm (段起 "
                      f"{np.round(_relv[0]*1000,1)}mm 段末 {np.round(_relv[-1]*1000,1)}mm "
                      f"max {_rel:.1f}mm @F{_hold_frames[_hi]}), 采样 {_hold_contact} "
                      f"—— 静置偏移是交接瞬态+楔入位置, 不是滑移")
                print(f"    HOLD 漂移剖面 (每3帧): {_prof}")
            _land_z = float(obj_trace[-1, 2])
            _z_exp = float(obj_meta.get("support_z", float("nan"))) + _half_vert
            _rest_ok = lrt.supported_rest(obj_trace, _object_dynamic, _contact_impulses, _kr)
            if _rest_ok:
                _avg_steps_rest = max(1, int(round(_sub_total / max(N, 1))))
                _rest_ok = bool(np.all(_support_steps[-15:] >= max(1, _avg_steps_rest // 2)))
            if _rest_ok:
                dots = np.abs(_exec_obj[-15:, 3:7] @ _exec_obj[-1, 3:7])
                _rest_ok = bool(np.min(dots) >= np.cos(np.radians(2.5)))
            # 数据参照抬升: 人手 LIFT 后 pad z 峰升 (实测 duck 83mm / 456 191mm /
            # chairbottle ~1740mm)。阈值 = 0.7×参照 + 30mm 下限 —— 容忍腕-物偏移
            # 随姿态的变化 (实测 duck 物体 61 vs 手 83); 真掉手仍不及格。旧版硬编码
            # 183mm 是 456 专属基线, 对低搬运视频误判。
            _k_g = int(np.argmax(_lb >= 2)) if _lb is not None and (_lb >= 2).any() else 0
            _ref_rise = (float(pad_p[_k_g:, 2].max() - pad_p[_k_g, 2]) * 1000.0
                         if _k_g < len(pad_p) else 300.0)
            _lift_thr = max(30.0, 0.7 * _ref_rise)
            _hold_start = np.flatnonzero(_hold_m)
            _handoff_deadline = int(_hold_start[0]) if len(_hold_start) else k_release
            if ring_place:
                center_error = np.linalg.norm(obj_trace[-1, :2] - ring_center)
                _rest_ok = bool(_rest_ok and center_error <= 0.015)
                obj_meta["ring_center_error_mm"] = float(center_error * 1000)
            _acc = {
                f"lift_ge_{_lift_thr:.0f}mm(数据参照0.7x{_ref_rise:.0f}mm)":
                    bool(lift >= _lift_thr),
                # 滑移 = HOLD 内的相对漂移增量 (物体在指间移动才是滑移; 静置偏移
                # 是交接瞬态+楔入位置, 单独报告 hold_offset)
                "hold_creep_le_10mm(段内漂移)": bool(_creep is not None and _creep <= 10.0),
                # 落台 = 末态收敛 (静止在支撑面) + 未掉地 (放置本就低于持物最低点,
                # 掉地则要跌全桌高 ~1.2m; 0.5m 容差区分两者)。support_z 拟合值与
                # 接触面的偏差 (网格体积中心 vs 台布面, 跨系标定差 ~95mm) 只做诊断。
                "supported_rest(末15帧动态支撑、无地面和夹爪接触)": _rest_ok,
                # _skid=None = 序列无 RELEASED 标签 (持物到结尾) —— 不适用, 记 False 并
                # 在 skid_mm 里落 null
                "skid_le_20mm(落台后)": bool(_skid is not None and _skid <= 20.0),
                "bilateral_dynamic_hold(>=60%帧)": bool(_hold_contact[1] > 0 and
                                                 _hold_contact[0] / _hold_contact[1] >= 0.6),
                "dynamic_handoff_during_lift": bool(len(handoff) and handoff[0] < _handoff_deadline),
                "no_ground_contact_during_hold_or_release": bool(
                    len(handoff) and not (_contact_impulses[handoff[0]:, 3] > 1e-7).any()),
                "not_slipped(A2自检)": bool(not obj_meta.get("slipped")),
            }
            obj_meta["acceptance"] = {"pass": all(_acc.values()), "criteria": _acc,
                                      "hold_slip_creep_mm": _creep, "hold_offset_mm": _rel,
                                      "landed_z": _land_z, "expected_z": _z_exp,
                                      "inflight_drift_mm": _drift, "skid_mm": _skid,
                                      "hold_contact_samples": _hold_contact}
            print(f"  验收 (物理抓取): {'✅ 全部通过' if all(_acc.values()) else '❌ 未通过'}")
            for _k2, _v2 in _acc.items():
                print(f"    {'✓' if _v2 else '✗'} {_k2}")
        if "grip_states" in d.files:
            for _nm, _lo in [("PICK起", 1), ("LIFT起", 2), ("HOLD起", 3),
                             ("PLACE起", 4), ("RELEASED起", 5)]:
                _frames = np.flatnonzero(_lb == _lo)
                if len(_frames):
                    _k = _frames[0]
                    print(f"    F{_k} {_nm}: 物体 z={obj_trace[_k,2]:.3f} "
                          f"xy=({obj_trace[_k,0]:.2f},{obj_trace[_k,1]:.2f})")
    _payload.update({
        "phys_pos_err_mm": {"mean": float(phys_err.mean()),
                            "p95": float(np.percentile(phys_err, 95)),
                            "max": float(phys_err.max())},
        "phys_ori_err_deg": {"mean": float(phys_ori.mean()),
                             "max": float(phys_ori.max())},
        "lag_next_frame_ratio": _lag_hit,
        "object": obj_meta if obj_actor is not None else None,
    })
    if torso_pitch_deg is not None:
        t1, t2, t3 = [nm2i[nm] for nm in _torso_joints[:3]]
        pitch_deg = np.degrees(lrt.torso_link_pitch(_exec_q[:, [t1, t2, t3]]))
        _payload["torso"].update(exec_max_abs_pitch_deg=float(np.max(np.abs(pitch_deg[:, 2]))),
            exec_over_limit_frames=int(np.sum(np.abs(pitch_deg[:, 2]) > torso_pitch_deg[2])),
            exec_link_peak_deg=np.max(np.abs(pitch_deg), axis=0).tolist(),
            exec_link_over_limit_frames=np.sum(np.abs(pitch_deg) > torso_pitch_deg, axis=0).tolist())
    _dump_metrics(_payload)
    # 执行轨迹落盘 (数据集核心产物): 机器人实际执行状态逐帧序列 —— 此前只存
    # IK 目标与指标, 执行轨迹在仿真结束即丢弃。字段: base[xyz+quat(w 前)],
    # q (全关节, 含臂/躯干/指, 维度见 qdim), obj[xyz+quat] (无物体=NaN),
    # q_target (当帧驱动目标, len(arm[0]) = 6臂+4躯干[躯干WBC时]), grip_target (指目标)。
    np.savez(os.path.join(LR_RESULTS_DIR, f"{name}_exec_trace.npz"),
             base=_exec_base, q=_exec_q, qdim=_qdim, obj=_exec_obj,
             grip=_exec_grip, cmd_base=_cmd_base, phys_err=phys_err,
             q_target=np.asarray(arm[:N]), grip_target=np.asarray(grip_exec[:N, 0]),
             frame=np.arange(N), fps=_fps_src, contact_impulses_Ns=_contact_impulses,
             contact_names=np.array(["finger1", "finger2", "support", "ground"]),
             bilateral_impulses_Ns=_bilateral_impulses, bilateral_substeps=_bilateral_steps,
             support_substeps=_support_steps, physics_substeps=_steps_frame,
             source_time=_src_time, phys_time=_phys_time,
             contact_impulses_signed_z=_contact_impulses_signed,
             contact_names_signed=np.array(["finger1_z", "finger2_z", "support_z", "ground_z"]),
             object_dynamic=_object_dynamic, measured_gap_m=_grasp_gap,
             release_frame=-1 if _actual_release is None else _actual_release)
    print(f"  执行轨迹 -> {os.path.join(LR_RESULTS_DIR, name + '_exec_trace.npz')} "
          f"(q dim {_qdim}, {N} 帧)")
    _avg_steps_p = max(1, int(round(_sub_total / max(N, 1))))
    print(f"  物理时钟: 累计 {_sub_total} 子步 = {_sub_total * _dt_sub:.3f}s "
          f"(源 {N / _fps_src:.3f}s, {_avg_steps_p} 子步/帧平均, clock_mode={clock_mode})")
    _tgt_step = np.linalg.norm(np.diff(pad_p, axis=0), axis=1) * 1000
    print(f"  目标本身帧间位移: mean {_tgt_step.mean():.1f}mm p95 {np.percentile(_tgt_step,95):.1f}mm "
          f"max {_tgt_step.max():.1f}mm")
    _dt_phys = time.time() - _t_phys
    print(f"  物理仿真耗时 {_dt_phys:.1f}s ({_dt_phys / max(N, 1) * 1000:.0f} ms/帧, {STEPS} 子步)")

    # ---- 轨迹 vs 跟随 对比图 (目标 / IK 解 / 物理执行 三线叠加) ----
    os.makedirs(LR_RESULTS_DIR, exist_ok=True)
    _plot_follow(os.path.join(LR_RESULTS_DIR, f"{name}_follow_compare.png"),
                 pad_p, ik_p, phys_p, base_xy, pos_err, ori_err,
                 phys_err, phys_ori, _transit, name=name)

    if not phys_only:
        os.makedirs(LR_DEMO_DIR, exist_ok=True)
        gif = os.path.join(LR_DEMO_DIR, f"{name}_large_range.gif")
        imageio.mimsave(gif, frames, fps=10, loop=0)
        print(f"\n✓ {name} 大范围移动 R1 仿真 GIF -> {gif} ({len(frames)} 帧)")
        return gif
    print("\n(phys-only: 跳过渲染与 GIF 保存)")
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="chairbottle 右手大范围移动 → R1 仿真渲染 (A 方案)")
    ap.add_argument("--stride", type=int, default=1, help="帧下采样步长 (越大越快)")
    ap.add_argument("--ik-only", action="store_true", help="只做 IK 质检, 不渲染")
    ap.add_argument("--base-smooth", type=int, default=None, help=f"底盘平滑窗口 (默认 {BASE_SMOOTH_WIN}, 0=不平滑)")
    ap.add_argument("--wbc", dest="wbc", action="store_true", default=True, help="全身 IK (底盘+臂联合, 默认)")
    ap.add_argument("--no-wbc", dest="wbc", action="store_false", help="关闭全身 IK, 用旧的解耦方案")
    ap.add_argument("--w-base", type=float, default=None,
                    help="WBC 底盘偏移正则权重; 不传=auto(标定稳健值 12.0)")
    ap.add_argument("--base-range", type=float, default=None,
                    help="WBC 底盘相对参考的最大偏移 (m); 不传=auto(标定稳健值 1.0)")
    ap.add_argument("--no-yaw", dest="use_yaw", action="store_false", default=True, help="WBC 关闭底盘 yaw 自由度")
    ap.add_argument("--smooth", dest="smooth", action="store_true", default=False, help="WBC 后做时序平滑 (离线 MPC, 默认关)")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false", help="关闭时序平滑 (默认)")
    ap.add_argument("--w-smooth", type=float, default=2.0, help="时序平滑权重 (越大越平滑)")
    ap.add_argument("--post-win", type=int, default=11, help="输出后处理平滑窗口 (0/1 关闭); 对底盘+臂做零相位滑动平均")
    ap.add_argument("--post-base-step", type=float, default=None, help="后处理底盘逐帧限速 (m/帧, 如 0.08)")
    ap.add_argument("--post-arm-step", type=float, default=None, help="后处理臂关节逐帧限速 (rad/帧)")
    ap.add_argument("--re-resolve", dest="re_resolve", action="store_true", default=False,
                    help="WBC 后对臂做局部重解 (默认关: WBC 直接出结果更准)")
    ap.add_argument("--w-cont", type=float, default=3.0, help="WBC 臂帧间连续性权重 (越大越平滑)")
    ap.add_argument("--posture-weight", type=float, default=0.0,
                    help="WBC 零空间姿态正则 (把关节拉向限位中点, 防贴限位后翻分支)")
    ap.add_argument("--joint-step-max", type=float, default=180.0,
                    help="WBC 臂关节帧间硬步长上界 (deg/帧)")
    ap.add_argument("--base-step-max", type=float, default=None,
                    help="WBC 底盘逐帧步长上界 (m/帧); 不传=按数据速度自动 (p95×1.1, 下限 0.05)")
    ap.add_argument("--spread-max", type=float, default=25.0,
                    help="分支切换摊平限速 (deg/帧, 0=关闭); 消除单帧手腕翻转")
    ap.add_argument("--arm-beam", type=int, default=0,
                    help="臂多分支 beam search 宽度 (0=关闭); 全局一致分支选择")
    ap.add_argument("--limit-weight", type=float, default=1.0,
                    help="WBC 关节限位裕度惩罚权重 (越大越早避开限位)")
    ap.add_argument("--limit-margin", type=float, default=20.0,
                    help="WBC 关节限位裕度角 (deg); 越大越远离限位")
    ap.add_argument("--w-ori", type=float, default=None,
                    help="WBC 姿态残差权重 (默认 10=姿态保真; 调小换位置精度+关节裕度, 见 docs/10)")
    ap.add_argument("--grip-mode", default="data", choices=["data", "states"],
                    help="夹爪执行: data=逐帧跟人手张合; states=状态机离散 (闭合@LIFT/张开@RELEASED)")
    ap.add_argument("--scene-cloud", default=None, help="场景点云 npz (ef3r_pointcloud.py 产出, 体素视觉)")
    ap.add_argument("--phys-only", action="store_true", default=False,
                    help="只跑物理跟随测量, 不渲染 (快速诊断执行层误差)")
    ap.add_argument("--phys-steps", type=int, default=40,
                    help="每帧物理子步数 (40=0.16s/帧); 加大可测'时间是否足够'")
    ap.add_argument("--stiff-scale", type=float, default=1.0,
                    help="臂 PD 刚度倍数; 加大可测'刚度是否不足'")
    ap.add_argument("--w-bcont", type=float, default=None,
                     help="WBC 底盘帧间连续性权重; 不传=auto(标定稳健值 5.0)")
    ap.add_argument("--n-local", type=int, default=2, help="WBC 每帧热启动邻域扰动起点数")
    ap.add_argument("--n-rand", type=int, default=1, help="WBC 每帧全局随机起点数")
    ap.add_argument("--max-nfev", type=int, default=150, help="每起点 least_squares 最大迭代")
    ap.add_argument("--targets", default=CB_TARGETS,
                    help="目标 npz (pos/rot/grip[/R_c2w/t_c2w]), 默认 chairbottle")
    ap.add_argument("--name", default="chairbottle", help="任务名 (决定输出文件名)")
    a = ap.parse_args()
    main(stride=a.stride, ik_only=a.ik_only, base_smooth=a.base_smooth,
         wbc=a.wbc, w_base=a.w_base, base_range=a.base_range, use_yaw=a.use_yaw,
         smooth=a.smooth, w_smooth=a.w_smooth,
         post_win=a.post_win, post_base_step=a.post_base_step, post_arm_step=a.post_arm_step,
         re_resolve=a.re_resolve, w_cont=a.w_cont, w_bcont=a.w_bcont,
         posture_weight=a.posture_weight, joint_step_max=a.joint_step_max,
         base_step_max=a.base_step_max, spread_max=a.spread_max, arm_beam=a.arm_beam,
         limit_weight=a.limit_weight, limit_margin=a.limit_margin, w_ori=a.w_ori,
         grip_mode=a.grip_mode,
         phys_only=a.phys_only, phys_steps=a.phys_steps, stiff_scale=a.stiff_scale,
         n_local=a.n_local, n_rand=a.n_rand, max_nfev=a.max_nfev,
         targets=a.targets, name=a.name, scene_cloud=a.scene_cloud)

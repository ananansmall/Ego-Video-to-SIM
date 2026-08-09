#!/usr/bin/env python3
"""vis_trajectory_optimize.py — 可视化 MANO 轨迹 vs 优化轨迹

独立脚本, 不依赖 grasp_hawor.py (避免 sapien 依赖).

展示:
  1. 3D: MANO 夹爪轨迹 + 优化后轨迹 + 物体位置 + 抓取窗口
  2. 折线: 每帧夹爪-物体距离 + 窗口标记

用法:
    python vis_trajectory_optimize.py \
        --hawor-dir /home/an/data/hawor/7 \
        --ras-dir /home/an/data/ras/my_7mp4_result \
        --opt-params output/gripper_only_right/opt_params.npy \
        [--save /tmp/vis_opt]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
try:
    import tkinter
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 注册 3D projection

# ─── 独立的函数和常量 (避免 import grasp_hawor 触发 sapien) ───
from trajectory_loader import compute_mano_joints

# ============================================================
# 坐标变换: GLB → SAPIEN (通过 HaWoR 中间空间)
# ============================================================
RXWORLD_TO_SAPIEN = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def load_transform_params(output_dir):
    """加载对齐变换参数: HaWoR → GLB transform.

    p_glb = scale * (R @ p_hawor + t)

    Returns (R, t, scale) 或 None.
    """
    for candidate in [
        Path(output_dir) / "alignment" / "transform_params.npz",
        Path(output_dir).parent / "alignment" / "transform_params.npz",
    ]:
        if candidate.exists():
            data = dict(np.load(candidate, allow_pickle=True))
            s = float(data.get("scale", 1.0))
            R_mat = np.array(data.get("R", np.eye(3)), dtype=np.float64)
            t_vec = np.array(data.get("t", np.zeros(3)), dtype=np.float64)
            return R_mat, t_vec, s
    return None


def glb_to_sapien(pos_glb, R_h2g, t_h2g, scale):
    """将 GLB 空间坐标转换为 SAPIEN 空间坐标.

    GLB → HaWoR: p_hawor = R_h2g^T @ (p_glb / scale - t_h2g)
    HaWoR → SAPIEN: p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
    """
    pos_np = np.asarray(pos_glb, dtype=np.float64)
    p_hawor = (R_h2g.T @ ((pos_np / scale) - t_h2g).T).T
    p_sapien = (RXWORLD_TO_SAPIEN @ p_hawor.T).T
    return p_sapien

# 夹爪 finger 参数 (与 physics_utils 一致)
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906


def compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2, prefix="right"):
    """加权 SVD 夹爪位姿 (与 grasp_hawor.py 一致)"""
    W_Y = 5.0
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    required_open_sum = finger_dist - FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    finger_mid = (mano_finger1 + mano_finger2) / 2
    pointing = finger_mid - mano_wrist
    pointing = pointing / max(np.linalg.norm(pointing), 1e-6)

    y_sign = 1.0 if prefix == "right" else -1.0
    opening = y_sign * v_finger / max(finger_dist, 1e-6)

    gripper_x = np.array([1.0, 0.0, 0.0])
    gripper_y = np.array([0.0, 1.0, 0.0])

    W = np.diag([1.0, W_Y])
    A = np.column_stack([gripper_x, gripper_y]) @ W
    B = np.column_stack([pointing, opening]) @ W
    H = A @ B.T
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    root_R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T

    finger1_in_gripper = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
    finger2_in_gripper = _FINGER2_ORIGIN + _FINGER2_AXIS * joint2
    finger_mid_in_gripper = (finger1_in_gripper + finger2_in_gripper) / 2
    root_pos = finger_mid - root_R @ finger_mid_in_gripper

    return root_pos, root_R, joint1, joint2


# 默认 MANO 模型路径
MANO_MODEL_CANDIDATES = [
    Path(__file__).parent / ".." / "position_retargeting" / "manopth" / "mano_v1_2" / "models",
    Path.home() / "robot_world_ws" / "src" / "dex-retargeting" / "example" / "position_retargeting" / "manopth" / "mano_v1_2" / "models",
]
MANO_PY_CANDIDATES = [
    Path(__file__).parent / ".." / "position_retargeting",
    Path.home() / "robot_world_ws" / "src" / "dex-retargeting" / "example" / "position_retargeting",
]


def find_mano_model():
    for c in MANO_MODEL_CANDIDATES:
        if list(c.glob("*.pkl")):
            return c
    return None


def create_mano_layer(side, betas):
    for mp in MANO_PY_CANDIDATES:
        sp = str(mp)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    from mano_layer import MANOLayer
    return MANOLayer(side, betas)


def load_glb_centroids(glb_path):
    """Load GLB and return list of (name, centroid) with real node names."""
    import trimesh
    scene = trimesh.load(glb_path)

    # Build geometry_key → node_name mapping
    geom_to_node = {}
    for node_name in scene.graph.nodes:
        children = scene.graph[node_name]
        if children[1] is not None:
            geom_to_node[children[1]] = node_name

    geoms = []
    if hasattr(scene, "geometry"):
        for geom_key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                centroid = geom.vertices.mean(axis=0)
                real_name = geom_to_node.get(geom_key, geom_key)
                geoms.append((real_name, centroid))
    else:
        geoms.append(("scene", scene.vertices.mean(axis=0)))
    return geoms


def load_sim_output(output_dir):
    """从仿真输出加载物体位姿 (SAPIEN 空间).

    优先读取 verify_first_frame.json 中的 obj_pos.
    若不存在, 读取 run_first_frame.json.
    """
    for fname in ["verify_first_frame.json", "run_first_frame.json"]:
        p = Path(output_dir) / fname
        if p.exists():
            import json
            with open(p) as f:
                data = json.load(f)
            obj_pos = np.array(data.get("obj_pos", [0, 0, 0]), dtype=np.float64)
            return obj_pos
    return None


def compute_gripper_trajectory(hawor_data, hand_idx, start_frame, num_frames):
    """Compute MANO gripper trajectory."""
    side = "left" if hand_idx == 0 else "right"
    betas = hawor_data["pred_betas"][hand_idx, start_frame].astype(np.float32)
    mano_layer = create_mano_layer(side, betas)

    traj = {"pos": [], "R": [], "j1": [], "j2": []}
    valid_count = 0
    for fi in range(start_frame, start_frame + num_frames):
        if fi >= hawor_data["pred_trans"].shape[1]:
            break
        if fi < hawor_data["pred_valid"].shape[1] and not hawor_data["pred_valid"][hand_idx, fi]:
            continue
        try:
            _, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][hand_idx, fi],
                hawor_data["pred_hand_pose"][hand_idx, fi],
                hawor_data["pred_trans"][hand_idx, fi],
            )
            joints_sapien = (RXWORLD_TO_SAPIEN @ joints.T).T
            wrist = joints_sapien[0, :3]
            f1 = joints_sapien[4, :3]  # index finger
            f2 = joints_sapien[8, :3]  # middle finger
            root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                wrist, f1, f2, prefix=side)
            traj["pos"].append(root_pos)
            traj["R"].append(root_R)
            traj["j1"].append(j1)
            traj["j2"].append(j2)
            valid_count += 1
        except Exception as _e:
            print(f"  [WARN] F{fi}: compute_gripper_trajectory failed: {_e}")
            continue

    traj["pos"] = np.array(traj["pos"], dtype=np.float64)
    traj["R"] = np.array(traj["R"], dtype=np.float64)
    return traj, side


def apply_optimized_offset(mano_pos, mano_R, offsets):
    """Apply keyframe offsets to trajectory."""
    from traj_optimize import apply_keyframe_offset
    opt_pos = mano_pos.copy()
    opt_R = mano_R.copy()
    for fi in range(len(mano_pos)):
        if fi < len(offsets):
            opt_pos[fi], opt_R[fi] = apply_keyframe_offset(
                mano_pos[fi], mano_R[fi], offsets[fi])
    return opt_pos, opt_R


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory optimization")
    parser.add_argument("--hawor-dir", default="/home/an/data/hawor/7")
    parser.add_argument("--ras-dir", default="/home/an/data/ras/my_7mp4_result")
    parser.add_argument("--opt-params", type=str, default=None,
                        help="path to opt_params.npy")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory (for alignment transform, defaults to opt_params parent)")
    parser.add_argument("--hand-idx", type=int, default=1,
                        help="hand index: 0=left, 1=right")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=113)
    parser.add_argument("--save", type=str, default=None,
                        help="save path prefix (e.g. /tmp/vis_opt)")
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir)
    ras_dir = Path(args.ras_dir)
    glb_path = ras_dir / "final_scene.glb"

    # ── 1. Load HAWOR data ──
    rec_dir = hawor_dir / "reconstruction"
    npz_files = sorted(rec_dir.glob("hawor_results_*.npz"))
    if not npz_files:
        print(f"[ERROR] No npz files in {rec_dir}")
        sys.exit(1)
    npz_path = str(npz_files[0])
    print(f"Loading HAWOR data: {npz_path}")
    hawor_data = dict(np.load(npz_path, allow_pickle=True))

    # ── 确定输出目录 ──
    out_dir_sim = args.output_dir
    if out_dir_sim is None and args.opt_params:
        out_dir_sim = str(Path(args.opt_params).parent)

    # ── 加载 GLB 物体并转换到 SAPIEN 空间 ──
    xform = load_transform_params(out_dir_sim) if out_dir_sim else None
    all_objects = []  # [(name, pos_sapien), ...]
    target_pos = None
    sim_target_pos = None

    if xform is not None:
        R_h2g, t_h2g, scale = xform
        print(f"Transform params loaded: scale={scale:.4f}")

        if glb_path.exists():
            raw_geoms = load_glb_centroids(str(glb_path))
            for name, c in raw_geoms:
                if np.linalg.norm(c) < 1e-6:
                    continue  # 跳过原点物体 (无效)
                p_sapien = glb_to_sapien(c, R_h2g, t_h2g, scale)
                all_objects.append((name, p_sapien))
            print(f"GLB objects in SAPIEN space: {len(all_objects)}")
            for name, p in all_objects:
                print(f"  {name}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
        else:
            print(f"[WARN] GLB not found: {glb_path}")

    # ── 加载仿真目标物体 (用于高亮显示) ──
    if out_dir_sim:
        sim_target_pos = load_sim_output(out_dir_sim)
        if sim_target_pos is not None:
            print(f"Sim target object pos (SAPIEN): ({sim_target_pos[0]:.4f}, {sim_target_pos[1]:.4f}, {sim_target_pos[2]:.4f})")

    # 确定目标物体: 仿真输出 > 最近 GLB 物体
    if sim_target_pos is not None:
        target_pos = sim_target_pos
    elif all_objects:
        wrist_traj = mano_traj["pos"]
        min_dist = float('inf')
        for name, p in all_objects:
            d = float(np.linalg.norm(wrist_traj - p, axis=1).min())
            if d < min_dist:
                min_dist = d
                target_pos = p
                target_geom = name
        print(f"Target object (closest to wrist): {target_geom} (dist={min_dist:.4f}m)")

    # ── 2. Compute MANO gripper trajectory ──
    print(f"Computing MANO gripper trajectory (hand_idx={args.hand_idx})...")
    mano_traj, side = compute_gripper_trajectory(
        hawor_data, args.hand_idx, args.start_frame, args.num_frames)
    n_frames = len(mano_traj["pos"])
    if n_frames == 0:
        print("[ERROR] No frames computed!")
        sys.exit(1)
    print(f"  {n_frames} frames, pos range: "
          f"X=[{mano_traj['pos'][:,0].min():.3f},{mano_traj['pos'][:,0].max():.3f}], "
          f"Y=[{mano_traj['pos'][:,1].min():.3f},{mano_traj['pos'][:,1].max():.3f}], "
          f"Z=[{mano_traj['pos'][:,2].min():.3f},{mano_traj['pos'][:,2].max():.3f}]")

    # ── 5. Compute optimized trajectory ──
    opt_pos = None
    opt_R = None
    if args.opt_params and Path(args.opt_params).exists():
        params = np.load(args.opt_params)
        print(f"Loaded opt_params: shape={params.shape}, ||params||={np.linalg.norm(params):.4f}")

        from scipy.spatial.transform import Rotation as R

        if len(params) == 3:
            # 第二十九轮: 3 维 XYZ 常量偏移
            from traj_optimize import apply_xyz_offset
            opt_pos = np.zeros_like(mano_traj["pos"])
            opt_R = np.zeros_like(mano_traj["R"])
            for fi in range(n_frames):
                opt_pos[fi], opt_R[fi] = apply_xyz_offset(
                    mano_traj["pos"][fi], mano_traj["R"][fi], params)
            print(f"  3D XYZ offset: [{params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f}]m")

        elif len(params) == 6:
            # 第二十九轮 Phase 2: 6 维 XYZ+RPY 偏移
            from traj_optimize import apply_xyz_offset
            xyz = params[0:3]
            rpy = params[3:6]
            r_corr = R.from_euler("xyz", rpy).as_matrix()
            opt_pos = np.zeros_like(mano_traj["pos"])
            opt_R = np.zeros_like(mano_traj["R"])
            for fi in range(n_frames):
                opt_pos[fi], opt_R_base = apply_xyz_offset(
                    mano_traj["pos"][fi], mano_traj["R"][fi], xyz)
                opt_R[fi] = r_corr @ opt_R_base if np.linalg.norm(rpy) > 1e-8 else opt_R_base
            print(f"  6D offset: xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}], "
                  f"rpy=[{rpy[0]:.4f},{rpy[1]:.4f},{rpy[2]:.4f}]")

        else:
            # 旧版: keyframe / 9-dim / 窗口参数
            params_path = Path(args.opt_params)
            window_path = params_path.parent / "window_frames.npy"
            if window_path.exists() and len(params) not in (9, 42):
                # 帧级窗口优化参数
                from traj_optimize import build_window_params, apply_window_offset
                grasp_window = np.load(window_path)
                window_params = build_window_params(grasp_window, n_frames)
                opt_pos = mano_traj["pos"].copy()
                opt_R = mano_traj["R"].copy()
                for fi in range(n_frames):
                    opt_pos[fi], opt_R[fi] = apply_window_offset(
                        mano_traj["pos"][fi], mano_traj["R"][fi], fi,
                        window_params, params, blend_frames=5)
                print(f"  Window params: {len(grasp_window)} frames "
                      f"[{grasp_window[0]}, {grasp_window[-1]}]")
            else:
                # 旧版 keyframe / 9-dim 参数
                from traj_optimize import interp_keyframes
                offsets = interp_keyframes(params, n_frames)
                print(f"  Offsets shape: {offsets.shape}")
                opt_pos, opt_R = apply_optimized_offset(
                    mano_traj["pos"], mano_traj["R"], offsets)
                z_offsets = offsets[:, 2]
                print(f"  Z offset: min={z_offsets.min():.4f}, max={z_offsets.max():.4f}, "
                      f"mean={z_offsets.mean():.4f}")
    else:
        print("No opt_params provided, showing MANO only")

    # ── 6. Compute distances per frame ──
    # MANO distance to target
    mano_dists = np.linalg.norm(mano_traj["pos"] - target_pos, axis=1) if target_pos is not None else None
    # Optimized distance to target
    opt_dists = np.linalg.norm(opt_pos - target_pos, axis=1) if opt_pos is not None and target_pos is not None else None

    # ── 7. Find grasp window (30% threshold) ──
    grasp_window = None
    if opt_dists is not None:
        threshold = np.percentile(opt_dists, 30)
        grasp_window = np.where(opt_dists <= threshold)[0]
        f_grasp = int(np.argmin(opt_dists))
        print(f"Grasp window: {len(grasp_window)} frames, "
              f"threshold(p30)={threshold:.4f}m, "
              f"closest frame F{f_grasp} (dist={opt_dists[f_grasp]:.4f}m)")

    # ── 8. Plot ──
    print("\nGenerating plots...")

    # 8a. 3D plot
    fig1 = plt.figure(figsize=(16, 12))
    ax = fig1.add_subplot(111, projection="3d")

    # MANO trajectory (blue)
    ax.plot(mano_traj["pos"][:, 0], mano_traj["pos"][:, 1], mano_traj["pos"][:, 2],
            "b-", alpha=0.6, linewidth=1.0, label=f"MANO traj ({n_frames} frames)")
    ax.scatter(mano_traj["pos"][0, 0], mano_traj["pos"][0, 1], mano_traj["pos"][0, 2],
               c="blue", s=60, marker="o", zorder=5)
    ax.text(mano_traj["pos"][0, 0], mano_traj["pos"][0, 1], mano_traj["pos"][0, 2],
            "  start", color="blue", fontsize=9)

    # Optimized trajectory (red)
    if opt_pos is not None:
        ax.plot(opt_pos[:, 0], opt_pos[:, 1], opt_pos[:, 2],
                "r-", alpha=0.6, linewidth=1.0, label=f"optimized traj")
        # Grasp window highlight
        if grasp_window is not None and len(grasp_window) > 0:
            ax.scatter(opt_pos[grasp_window, 0], opt_pos[grasp_window, 1],
                       opt_pos[grasp_window, 2], c="orange", s=15, alpha=0.5,
                       label=f"grasp window ({len(grasp_window)} frames)")
            fg = int(np.argmin(opt_dists))
            ax.scatter(opt_pos[fg, 0], opt_pos[fg, 1], opt_pos[fg, 2],
                       c="red", s=120, marker="*", zorder=6,
                       label=f"closest frame F{fg}")

    # All objects in SAPIEN space
    if all_objects:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_objects)))
        for i, (name, p) in enumerate(all_objects):
            ax.scatter(*p, color=colors[i], s=200, marker="o",
                       edgecolors="k", linewidths=1.0, zorder=5)
            obj_label = name.split('_')[-1] if '_' in name else name
            ax.text(p[0], p[1], p[2], f"  {obj_label}",
                    color=colors[i], fontsize=9, fontweight="bold")

    # Target object highlight
    if target_pos is not None:
        ax.scatter(*target_pos, c="green", s=300, marker="s",
                   edgecolors="k", linewidths=1.5, zorder=6, alpha=0.6,
                   label="target")

    # Legend
    ax.plot([], [], "b-", alpha=0.6, label="MANO trajectory")
    if opt_pos is not None:
        ax.plot([], [], "r-", alpha=0.6, label="optimized trajectory")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title(f"Gripper Trajectory: MANO vs Optimized ({side} hand)", fontsize=13)

    # Equal aspect
    all_pts = [mano_traj["pos"]]
    if opt_pos is not None:
        all_pts.append(opt_pos)
    if target_pos is not None:
        all_pts.append(np.array([target_pos]))
    all_pts = np.vstack(all_pts)
    mid = all_pts.mean(axis=0)
    max_range = max(all_pts.ptp(axis=0)) / 2 * 1.3
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    # 8b. Distance per frame
    if target_pos is not None:
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        frames = np.arange(len(mano_dists))

        ax2.plot(frames, mano_dists, "b-", alpha=0.5, linewidth=0.8,
                 label=f"MANO (min={mano_dists.min():.3f}m)")
        if opt_dists is not None:
            ax2.plot(frames, opt_dists, "r-", alpha=0.5, linewidth=0.8,
                     label=f"optimized (min={opt_dists.min():.3f}m)")

            # Grasp window
            if grasp_window is not None:
                ax2.axvspan(grasp_window[0], grasp_window[-1],
                            alpha=0.15, color="orange", label="grasp window")
                ax2.axhline(threshold, color="orange", linestyle="--", alpha=0.5,
                            label=f"p30 threshold={threshold:.3f}m")

        # f_grasp marker
        if opt_dists is not None:
            fg = int(np.argmin(opt_dists))
            ax2.axvline(fg, color="red", linestyle=":", alpha=0.7,
                         label=f"closest frame F{fg}")
        else:
            fg = int(np.argmin(mano_dists))
            ax2.axvline(fg, color="blue", linestyle=":", alpha=0.7,
                        label=f"closest frame F{fg}")

        ax2.set_xlabel("Frame index", fontsize=11)
        ax2.set_ylabel("Distance to target (m)", fontsize=11)
        ax2.set_title("Gripper-to-Object Distance per Frame", fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

    # ── 9. Save or show ──
    if args.save:
        fig1.savefig(f"{args.save}_3d.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {args.save}_3d.png")
        if target_pos is not None:
            fig2.savefig(f"{args.save}_dist.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: {args.save}_dist.png")
    else:
        plt.show()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
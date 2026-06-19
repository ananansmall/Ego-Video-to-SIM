lza326"""
三方对比: RAS (场景重建) + 7 (HaWoR SLAM) + 7_vggt-omega (HaWoR + VGGT-Omega)

生成:
  [原8张] 7 vs 7_vggt-omega 对比
  [新增6张] 三方对比 (RAS + 7 + 7_vggt)

用法:
  conda run -n dex python compare_hawor_dirs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

DIR_7 = "/home/an/data/hawor/7"
DIR_VGGT = "/home/an/data/hawor/7_vggt-omega"
DIR_RAS = "/home/an/data/ras/my_7mp4_result"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_rec(hawor_dir):
    rec_dir = Path(hawor_dir) / "reconstruction"
    rec_files = sorted(rec_dir.glob("hawor_results_*.npz"))
    if not rec_files:
        raise FileNotFoundError(f"No reconstruction in {hawor_dir}")
    return dict(np.load(str(rec_files[0]), allow_pickle=True))


def load_ras(ras_dir):
    """加载 RAS 数据: intrinsic, extrinsics (w2c), 转成 c2w"""
    ras_path = Path(ras_dir)

    # 读取 intrinsic
    with open(ras_path / "intrinsic.txt") as f:
        vals = [float(x) for x in f.read().strip().split()]
    K = np.array(vals).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 读取所有 extrinsics (w2c)
    ext_dir = ras_path / "extrinsics"
    ext_files = sorted(ext_dir.glob("*.txt"), key=lambda x: int(x.stem))

    R_c2w_list = []
    t_c2w_list = []
    for ef in ext_files:
        with open(ef) as f:
            vals = [float(x) for x in f.read().strip().split()]
        w2c = np.array(vals).reshape(4, 4)
        c2w = np.linalg.inv(w2c)
        R_c2w_list.append(c2w[:3, :3])
        t_c2w_list.append(c2w[:3, 3])

    return {
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal": (fx + fy) / 2,
        "R_c2w": np.array(R_c2w_list),
        "t_c2w": np.array(t_c2w_list),
        "n_frames": len(ext_files),
    }


def rotation_to_euler_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z]))


# ================== 原 8 张图 (7 vs 7_vggt) ==================

def plot_camera_trajectory_3d(rec_7, rec_vggt):
    fig = plt.figure(figsize=(18, 6))
    titles = ["7 (SLAM)", "7_vggt-omega", "Overlay"]
    datas = [
        (rec_7["t_c2w"], rec_7["R_c2w"]),
        (rec_vggt["t_c2w"], rec_vggt["R_c2w"]),
        None,
    ]
    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(titles[idx], fontsize=14, fontweight="bold")
        if idx < 2:
            t = datas[idx][0]
            R = datas[idx][1]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], "o-", markersize=2, linewidth=1)
            for i in range(0, len(t), max(1, len(t) // 8)):
                fwd = R[i] @ np.array([0, 0, -1]) * 0.02
                ax.quiver(t[i, 0], t[i, 1], t[i, 2], fwd[0], fwd[1], fwd[2],
                          color="red", arrow_length_ratio=0.3, linewidth=1.5)
            ax.scatter(*t[0], color="green", s=80, marker="^", label="start", zorder=5)
            ax.scatter(*t[-1], color="red", s=80, marker="v", label="end", zorder=5)
        else:
            t7 = rec_7["t_c2w"]
            tv = rec_vggt["t_c2w"]
            ax.plot(t7[:, 0], t7[:, 1], t7[:, 2], "b-o", markersize=2, linewidth=1, label="7 (SLAM)")
            ax.plot(tv[:, 0], tv[:, 1], tv[:, 2], "r-o", markersize=2, linewidth=1, label="7_vggt-omega")
            ax.legend(fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_camera_trajectory_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 01_camera_trajectory_3d.png")


def plot_hand_trajectory_3d(rec_7, rec_vggt, hand_idx=0):
    fig = plt.figure(figsize=(18, 6))
    t7 = rec_7["pred_trans"][hand_idx]
    tv = rec_vggt["pred_trans"][hand_idx]
    v7 = rec_7["pred_valid"][hand_idx]
    vv = rec_vggt["pred_valid"][hand_idx]
    titles = ["7 (SLAM)", "7_vggt-omega", "Overlay"]
    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(titles[idx], fontsize=14, fontweight="bold")
        if idx == 0:
            mask = v7 > 0.5
            ax.plot(t7[mask, 0], t7[mask, 1], t7[mask, 2], "b-", linewidth=1, alpha=0.5)
            ax.plot(t7[~mask, 0], t7[~mask, 1], t7[~mask, 2], "rx", markersize=4, label="invalid")
        elif idx == 1:
            mask = vv > 0.5
            ax.plot(tv[mask, 0], tv[mask, 1], tv[mask, 2], "r-", linewidth=1, alpha=0.5)
            ax.plot(tv[~mask, 0], tv[~mask, 1], tv[~mask, 2], "rx", markersize=4, label="invalid")
        else:
            ax.plot(t7[:, 0], t7[:, 1], t7[:, 2], "b-", linewidth=1, alpha=0.7, label="7 (SLAM)")
            ax.plot(tv[:, 0], tv[:, 1], tv[:, 2], "r-", linewidth=1, alpha=0.7, label="7_vggt-omega")
            ax.legend(fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_hand_trajectory_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 02_hand_trajectory_3d.png")


def plot_timeseries_comparison(rec_7, rec_vggt, hand_idx=0):
    n_frames = rec_7["pred_trans"].shape[1]
    frames = np.arange(n_frames)
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    t7 = rec_7["pred_trans"][hand_idx]
    tv = rec_vggt["pred_trans"][hand_idx]
    r7 = rec_7["pred_rot"][hand_idx]
    rv = rec_vggt["pred_rot"][hand_idx]
    tc7 = rec_7["t_c2w"]
    tcv = rec_vggt["t_c2w"]
    labels = ["X", "Y", "Z"]
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(frames, t7[:, i], "b-", linewidth=1, label="7", alpha=0.8)
        ax.plot(frames, tv[:, i], "r-", linewidth=1, label="7_vggt", alpha=0.8)
        ax.set_title(f"Hand trans {labels[i]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(frames, r7[:, i], "b-", linewidth=1, label="7", alpha=0.8)
        ax.plot(frames, rv[:, i], "r-", linewidth=1, label="7_vggt", alpha=0.8)
        ax.set_title(f"Hand rot {labels[i]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(frames, tc7[:, i], "b-", linewidth=1, label="7", alpha=0.8)
        ax.plot(frames, tcv[:, i], "r-", linewidth=1, label="7_vggt", alpha=0.8)
        ax.set_title(f"Cam trans {labels[i]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    euler_7 = np.array([rotation_to_euler_deg(rec_7["R_c2w"][i]) for i in range(n_frames)])
    euler_vggt = np.array([rotation_to_euler_deg(rec_vggt["R_c2w"][i]) for i in range(n_frames)])
    for i in range(3):
        ax = fig.add_subplot(gs[3, i])
        ax.plot(frames, euler_7[:, i], "b-", linewidth=1, label="7", alpha=0.8)
        ax.plot(frames, euler_vggt[:, i], "r-", linewidth=1, label="7_vggt", alpha=0.8)
        ax.set_title(f"Cam euler {labels[i]} (deg)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Time-series: 7 (blue) vs 7_vggt-omega (red)", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "03_timeseries_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 03_timeseries_comparison.png")


def plot_keyframe_hand_overlay(rec_7, rec_vggt, hand_idx=0):
    n_frames = rec_7["pred_trans"].shape[1]
    keyframes = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    fig, axes = plt.subplots(1, len(keyframes), figsize=(5 * len(keyframes), 5))
    t7 = rec_7["pred_trans"][hand_idx]
    tv = rec_vggt["pred_trans"][hand_idx]
    tc7 = rec_7["t_c2w"]
    tcv = rec_vggt["t_c2w"]
    for ax_idx, fi in enumerate(keyframes):
        ax = axes[ax_idx]
        ax.set_title(f"Frame {fi}", fontsize=12, fontweight="bold")
        ax.plot(t7[:, 0], t7[:, 2], "b-", linewidth=0.5, alpha=0.3)
        ax.plot(tv[:, 0], tv[:, 2], "r-", linewidth=0.5, alpha=0.3)
        ax.scatter(t7[fi, 0], t7[fi, 2], color="blue", s=100, marker="o", zorder=5, label="7 hand")
        ax.scatter(tv[fi, 0], tv[fi, 2], color="red", s=100, marker="o", zorder=5, label="7_vggt hand")
        ax.scatter(tc7[fi, 0], tc7[fi, 2], color="blue", s=60, marker="^", zorder=5, label="7 cam")
        ax.scatter(tcv[fi, 0], tcv[fi, 2], color="red", s=60, marker="^", zorder=5, label="7_vggt cam")
        fwd7 = rec_7["R_c2w"][fi] @ np.array([0, 0, -1]) * 0.03
        fwdv = rec_vggt["R_c2w"][fi] @ np.array([0, 0, -1]) * 0.03
        ax.annotate("", xy=(tc7[fi, 0] + fwd7[0], tc7[fi, 2] + fwd7[2]),
                     xytext=(tc7[fi, 0], tc7[fi, 2]),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
        ax.annotate("", xy=(tcv[fi, 0] + fwdv[0], tcv[fi, 2] + fwdv[2]),
                     xytext=(tcv[fi, 0], tcv[fi, 2]),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Keyframe (XZ plane, top-down)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "04_keyframe_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 04_keyframe_overlay.png")


def plot_diff_heatmap(rec_7, rec_vggt, hand_idx=0):
    n_frames = rec_7["pred_trans"].shape[1]
    frames = np.arange(n_frames)
    diff_trans = np.abs(rec_7["pred_trans"][hand_idx] - rec_vggt["pred_trans"][hand_idx])
    diff_rot = np.abs(rec_7["pred_rot"][hand_idx] - rec_vggt["pred_rot"][hand_idx])
    diff_cam_t = np.abs(rec_7["t_c2w"] - rec_vggt["t_c2w"])
    diff_cam_R = np.array([
        np.abs(rotation_to_euler_deg(rec_7["R_c2w"][i]) - rotation_to_euler_deg(rec_vggt["R_c2w"][i]))
        for i in range(n_frames)
    ])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    ax.stackplot(frames, diff_trans[:, 0], diff_trans[:, 1], diff_trans[:, 2], labels=["dx", "dy", "dz"], alpha=0.7)
    ax.set_title("Hand Translation Diff (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.stackplot(frames, diff_rot[:, 0], diff_rot[:, 1], diff_rot[:, 2], labels=["drx", "dry", "drz"], alpha=0.7)
    ax.set_title("Hand Rotation Diff (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.stackplot(frames, diff_cam_t[:, 0], diff_cam_t[:, 1], diff_cam_t[:, 2], labels=["dx", "dy", "dz"], alpha=0.7)
    ax.set_title("Camera Translation Diff (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.stackplot(frames, diff_cam_R[:, 0], diff_cam_R[:, 1], diff_cam_R[:, 2], labels=["drx", "dry", "drz"], alpha=0.7)
    ax.set_title("Camera Rotation Diff (deg)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Absolute Difference: |7 - 7_vggt-omega|", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "05_diff_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 05_diff_heatmap.png")


def plot_summary_table(rec_7, rec_vggt):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    rows = [["Property", "7 (SLAM)", "7_vggt-omega", "Diff"]]
    rows.append(["img_focal", f"{float(rec_7['img_focal']):.1f}", f"{float(rec_vggt['img_focal']):.1f}",
                  f"{abs(float(rec_7['img_focal']) - float(rec_vggt['img_focal'])):.1f}"])
    t7 = rec_7["pred_trans"][0]
    tv = rec_vggt["pred_trans"][0]
    tc7 = rec_7["t_c2w"]
    tcv = rec_vggt["t_c2w"]
    diff_t = np.abs(t7 - tv)
    diff_cam = np.abs(tc7 - tcv)
    rows.append(["hand_trans max_diff", f"{diff_t.max():.4f} m", "", ""])
    rows.append(["hand_trans mean_diff", f"{diff_t.mean():.4f} m", "", ""])
    rows.append(["cam_trans max_diff", f"{diff_cam.max():.4f} m", "", ""])
    rows.append(["cam_trans mean_diff", f"{diff_cam.mean():.4f} m", "", ""])
    vggt_keys = set(rec_vggt.keys()) - set(rec_7.keys())
    only7_keys = set(rec_7.keys()) - set(rec_vggt.keys())
    rows.append(["7_vggt extra keys", ", ".join(vggt_keys) if vggt_keys else "none", "", ""])
    rows.append(["7 only keys", ", ".join(only7_keys) if only7_keys else "none", "", ""])
    if "camera_source" in rec_vggt:
        rows.append(["camera_source", "N/A", str(rec_vggt["camera_source"]), ""])
    if "slam_scale" in rec_vggt:
        rows.append(["slam_scale", "N/A", f"{float(rec_vggt['slam_scale']):.4f}", ""])
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary: 7 vs 7_vggt-omega", fontsize=16, fontweight="bold", pad=20)
    fig.savefig(OUT_DIR / "06_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 06_summary_table.png")


def plot_hand_pose_pca_comparison(rec_7, rec_vggt, hand_idx=0):
    n_frames = rec_7["pred_trans"].shape[1]
    frames = np.arange(n_frames)
    hp7 = rec_7["pred_hand_pose"][hand_idx]
    hpv = rec_vggt["pred_hand_pose"][hand_idx]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i in range(min(9, hp7.shape[1])):
        ax = axes[i // 3, i % 3]
        ax.plot(frames, hp7[:, i], "b-", linewidth=1, label="7", alpha=0.8)
        ax.plot(frames, hpv[:, i], "r-", linewidth=1, label="7_vggt", alpha=0.8)
        ax.set_title(f"PCA coeff {i}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Hand Pose PCA Coefficients", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "07_hand_pose_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 07_hand_pose_pca.png")


def plot_camera_frustum_2d(rec_7, rec_vggt):
    n_frames = rec_7["pred_trans"].shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax_idx, (rec, title, color) in enumerate([
        (rec_7, "7 (SLAM)", "blue"),
        (rec_vggt, "7_vggt-omega", "red"),
    ]):
        ax = axes[ax_idx]
        t = rec["t_c2w"]
        R = rec["R_c2w"]
        focal = float(rec["img_focal"])
        ax.plot(t[:, 0], t[:, 2], "-", color=color, linewidth=1, alpha=0.5)
        step = max(1, n_frames // 12)
        for i in range(0, n_frames, step):
            fwd = R[i] @ np.array([0, 0, -1])
            right = R[i] @ np.array([1, 0, 0])
            up = R[i] @ np.array([0, 1, 0])
            fov_h = 2 * np.arctan2(960, focal)
            half_w = np.sin(fov_h / 2) * 0.03
            half_h = np.sin(fov_h / 2 * 540 / 960) * 0.03
            corners = [
                t[i] + fwd * 0.03 - right * half_w - up * half_h,
                t[i] + fwd * 0.03 + right * half_w - up * half_h,
                t[i] + fwd * 0.03 + right * half_w + up * half_h,
                t[i] + fwd * 0.03 - right * half_w + up * half_h,
            ]
            cx = [c[0] for c in corners]
            cz = [c[2] for c in corners]
            ax.plot(cx, cz, "-", color=color, linewidth=0.8, alpha=0.6)
        ax.scatter(t[0, 0], t[0, 2], color="green", s=80, marker="^", zorder=5, label="start")
        ax.scatter(t[-1, 0], t[-1, 2], color="orange", s=80, marker="v", zorder=5, label="end")
        ax.set_title(f"{title} (focal={focal:.0f}px)", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_camera_frustum_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 08_camera_frustum_2d.png")


# ================== 新增 6 张三方对比图 ==================

def plot_three_way_intrinsic(ras, rec_7, rec_vggt):
    """三方内参对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    rows = [
        ["Parameter", "RAS (RGBD)", "7 (SLAM)", "7_vggt-omega"],
        ["focal (px)", f"{ras['focal']:.2f}", f"{float(rec_7['img_focal']):.2f}", f"{float(rec_vggt['img_focal']):.2f}"],
        ["fx", f"{ras['fx']:.2f}", "N/A", "N/A"],
        ["fy", f"{ras['fy']:.2f}", "N/A", "N/A"],
        ["cx (principal)", f"{ras['cx']:.1f}", "N/A", "N/A"],
        ["cy (principal)", f"{ras['cy']:.1f}", "N/A", "N/A"],
        ["image size", "518x294 (RGBD)", "1920x1080 (HaWoR)", "1920x1080 (HaWoR)"],
        ["source", "RealSense intrinsics", "HaWoR (est_focal=600)", "VGGT-Omega"],
    ]

    if "img_center" in rec_vggt:
        rows.append(["img_center", "N/A", "N/A", f"{list(rec_vggt['img_center'])}"])

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    colors = ["#4472C4", "#70AD47", "#E84545"]
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(colors[col - 1])
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Three-way Intrinsic Comparison", fontsize=14, fontweight="bold", pad=20)
    fig.savefig(OUT_DIR / "09_three_way_intrinsic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 09_three_way_intrinsic.png")


def plot_three_way_camera_3d(ras, rec_7, rec_vggt):
    """三方相机轨迹3D对比（统一尺度下）"""
    fig = plt.figure(figsize=(20, 6))

    tc7 = rec_7["t_c2w"]
    tcv = rec_vggt["t_c2w"]
    tr = ras["t_c2w"]

    Rc7 = rec_7["R_c2w"]
    Rcv = rec_vggt["R_c2w"]
    Rr = ras["R_c2w"]

    titles = ["7 (SLAM)", "7_vggt-omega", "RAS (RGBD)"]

    for idx, (t, R, title, color) in enumerate([
        (tc7, Rc7, "7 (SLAM)", "blue"),
        (tcv, Rcv, "7_vggt-omega", "red"),
        (tr, Rr, "RAS (RGBD)", "green"),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.plot(t[:, 0], t[:, 1], t[:, 2], "o-", color=color, markersize=2, linewidth=1)
        step = max(1, len(t) // 8)
        for i in range(0, len(t), step):
            fwd = R[i] @ np.array([0, 0, -1]) * 0.02
            ax.quiver(t[i, 0], t[i, 1], t[i, 2], fwd[0], fwd[1], fwd[2],
                      color="red", arrow_length_ratio=0.3, linewidth=1.5)
        ax.scatter(*t[0], color="green", s=80, marker="^", zorder=5, label="start")
        ax.scatter(*t[-1], color="orange", s=80, marker="v", zorder=5, label="end")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "10_three_way_camera_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 10_three_way_camera_3d.png")


def plot_three_way_camera_2d(ras, rec_7, rec_vggt):
    """三方相机轨迹2D对比（XZ 俯视图，叠加显示）"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    tc7 = rec_7["t_c2w"]
    tcv = rec_vggt["t_c2w"]
    tr = ras["t_c2w"]

    Rc7 = rec_7["R_c2w"]
    Rcv = rec_vggt["R_c2w"]
    Rr = ras["R_c2w"]

    focals = [float(rec_7["img_focal"]), float(rec_vggt["img_focal"]), ras["focal"]]

    for ax_idx, view_name in enumerate(["XY plane (front view)", "XZ plane (top view)"]):
        ax = axes[ax_idx]
        if ax_idx == 0:
            x_i, y_i = 0, 1
            xlabel, ylabel = "X (m)", "Y (m)"
        else:
            x_i, y_i = 0, 2
            xlabel, ylabel = "X (m)", "Z (m)"

        for t, R, name, color in [
            (tc7, Rc7, "7 (SLAM)", "blue"),
            (tcv, Rcv, "7_vggt-omega", "red"),
            (tr, Rr, "RAS (RGBD)", "green"),
        ]:
            ax.plot(t[:, x_i], t[:, y_i], "-o", color=color, markersize=2,
                    linewidth=1, alpha=0.8, label=name)
            step = max(1, len(t) // 6)
            for i in range(0, len(t), step):
                fwd = R[i] @ np.array([0, 0, -1]) * 0.02
                ax.annotate("", xy=(t[i, x_i] + fwd[x_i], t[i, y_i] + fwd[y_i]),
                            xytext=(t[i, x_i], t[i, y_i]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

        ax.set_title(view_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Three-way Camera Trajectory (RAS=green, 7=blue, 7_vggt=red)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "11_three_way_camera_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 11_three_way_camera_2d.png")


def plot_three_way_frustum(ras, rec_7, rec_vggt):
    """三方相机视锥对比（用各自焦距计算FOV，统一尺度）"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    t_RAS = ras["t_c2w"]
    R_RAS = ras["R_c2w"]
    focal_RAS = ras["focal"]

    t_7 = rec_7["t_c2w"]
    R_7 = rec_7["R_c2w"]
    focal_7 = float(rec_7["img_focal"])

    t_v = rec_vggt["t_c2w"]
    R_v = rec_vggt["R_c2w"]
    focal_v = float(rec_vggt["img_focal"])

    for ax_idx, (t, R, focal, title, color, img_w, img_h) in enumerate([
        (t_RAS, R_RAS, focal_RAS, f"RAS (focal={focal_RAS:.0f}px, 518x294)", "green", 518, 294),
        (t_7, R_7, focal_7, f"7 (focal={focal_7:.0f}px, 1920x1080)", "blue", 1920, 1080),
        (t_v, R_v, focal_v, f"7_vggt (focal={focal_v:.0f}px, 1920x1080)", "red", 1920, 1080),
    ]):
        ax = axes[ax_idx]
        ax.plot(t[:, 0], t[:, 2], "-", color=color, linewidth=1, alpha=0.5)

        fov_h = 2 * np.arctan2(img_w / 2, focal)
        fov_v = 2 * np.arctan2(img_h / 2, focal)
        cone_len = 0.04
        half_w = np.tan(fov_h / 2) * cone_len
        half_h = np.tan(fov_v / 2) * cone_len

        step = max(1, len(t) // 8)
        for i in range(0, len(t), step):
            fwd = R[i] @ np.array([0, 0, -1])
            right = R[i] @ np.array([1, 0, 0])
            up = R[i] @ np.array([0, 1, 0])

            tip = t[i] + fwd * cone_len
            c1 = tip - right * half_w - up * half_h
            c2 = tip + right * half_w - up * half_h
            c3 = tip + right * half_w + up * half_h
            c4 = tip - right * half_w + up * half_h

            corners = np.array([c1, c2, c3, c4, c1])
            ax.plot(corners[:, 0], corners[:, 2], "-", color=color, linewidth=0.8, alpha=0.6)
            for j in range(4):
                ax.plot([t[i, 0], corners[j, 0]], [t[i, 2], corners[j, 2]],
                        "-", color=color, linewidth=0.4, alpha=0.3)

        ax.scatter(t[0, 0], t[0, 2], color="lime", s=80, marker="^", zorder=5, label="start")
        ax.scatter(t[-1, 0], t[-1, 2], color="orange", s=80, marker="v", zorder=5, label="end")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Three-way Camera Frustum (FOV scaled by focal & image size)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "12_three_way_frustum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 12_three_way_frustum.png")


def plot_three_way_fov():
    """三方FOV对比表"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    ras = load_ras(DIR_RAS)
    rec_7 = load_rec(DIR_7)
    rec_vggt = load_rec(DIR_VGGT)

    fov_h_ras = 2 * np.degrees(np.arctan2(ras["cx"], ras["fx"]))
    fov_h_7 = 2 * np.degrees(np.arctan2(1920 / 2, float(rec_7["img_focal"])))
    fov_h_v = 2 * np.degrees(np.arctan2(1920 / 2, float(rec_vggt["img_focal"])))

    fov_v_ras = 2 * np.degrees(np.arctan2(ras["cy"], ras["fy"]))
    fov_v_7 = 2 * np.degrees(np.arctan2(1080 / 2, float(rec_7["img_focal"])))
    fov_v_v = 2 * np.degrees(np.arctan2(1080 / 2, float(rec_vggt["img_focal"])))

    rows = [
        ["Metric", "RAS (RGBD)", "7 (SLAM)", "7_vggt-omega"],
        ["Image WxH", "518x294", "1920x1080", "1920x1080"],
        ["Focal (px)", f"{ras['focal']:.1f}", f"{float(rec_7['img_focal']):.1f}", f"{float(rec_vggt['img_focal']):.1f}"],
        ["FOV horizontal (deg)", f"{fov_h_ras:.1f}", f"{fov_h_7:.1f}", f"{fov_h_v:.1f}"],
        ["FOV vertical (deg)", f"{fov_v_ras:.1f}", f"{fov_v_7:.1f}", f"{fov_v_v:.1f}"],
        ["Camera source", "RealSense SLAM", "HaWoR SLAM", "VGGT-Omega"],
        ["n_frames", f"{ras['n_frames']}", f"{len(rec_7['t_c2w'])}", f"{len(rec_vggt['t_c2w'])}"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    colors = ["#70AD47", "#4472C4", "#E84545"]
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(colors[col - 1])
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Three-way FOV Comparison", fontsize=14, fontweight="bold", pad=20)
    fig.savefig(OUT_DIR / "13_three_way_fov.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 13_three_way_fov.png")


def plot_three_way_motion_range():
    """三方相机运动范围对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ras = load_ras(DIR_RAS)
    rec_7 = load_rec(DIR_7)
    rec_vggt = load_rec(DIR_VGGT)

    data = [
        ("RAS (RGBD)", ras["t_c2w"], "green"),
        ("7 (SLAM)", rec_7["t_c2w"], "blue"),
        ("7_vggt-omega", rec_vggt["t_c2w"], "red"),
    ]

    ax = axes[0]
    labels = ["X", "Y", "Z"]
    x = np.arange(len(labels))
    width = 0.25

    ranges = []
    for name, t, color in data:
        rng = [t[:, i].max() - t[:, i].min() for i in range(3)]
        ranges.append(rng)

    for i, (name, _, color) in enumerate(data):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, ranges[i], width, label=name, color=color, alpha=0.8)
        for bar, val in zip(bars, ranges[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{val:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Range (m)")
    ax.set_title("Camera Position Range (max - min)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    ax = axes[1]
    std_data = []
    for name, t, color in data:
        std = [t[:, i].std() for i in range(3)]
        std_data.append((name, std, color))

    for i, (name, std, color) in enumerate(std_data):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, std, width, label=name, color=color, alpha=0.8)
        for bar, val in zip(bars, std):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                    f"{val:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Std (m)")
    ax.set_title("Camera Position Std (motion magnitude)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    fig.suptitle("Three-way Camera Motion Magnitude", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "14_three_way_motion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 14_three_way_motion.png")


# ================== Main ==================

def main():
    print("=" * 60)
    print("  Three-way Comparison: RAS + 7 + 7_vggt-omega")
    print("=" * 60)

    print("\nLoading data...")
    rec_7 = load_rec(DIR_7)
    rec_vggt = load_rec(DIR_VGGT)
    ras = load_ras(DIR_RAS)

    print(f"  7:        {len(rec_7['t_c2w'])} frames, focal={float(rec_7['img_focal']):.1f}")
    print(f"  7_vggt:   {len(rec_vggt['t_c2w'])} frames, focal={float(rec_vggt['img_focal']):.1f}")
    print(f"  RAS:      {ras['n_frames']} frames, focal={ras['focal']:.1f} (fx={ras['fx']:.1f}, fy={ras['fy']:.1f})")

    print(f"\nGenerating images in {OUT_DIR}/ ...")

    print("\n[原 8 张] 7 vs 7_vggt-omega 对比 ...")
    plot_camera_trajectory_3d(rec_7, rec_vggt)
    plot_hand_trajectory_3d(rec_7, rec_vggt)
    plot_timeseries_comparison(rec_7, rec_vggt)
    plot_keyframe_hand_overlay(rec_7, rec_vggt)
    plot_diff_heatmap(rec_7, rec_vggt)
    plot_summary_table(rec_7, rec_vggt)
    plot_hand_pose_pca_comparison(rec_7, rec_vggt)
    plot_camera_frustum_2d(rec_7, rec_vggt)

    print("\n[新增 6 张] 三方对比 (RAS + 7 + 7_vggt) ...")
    plot_three_way_intrinsic(ras, rec_7, rec_vggt)
    plot_three_way_camera_3d(ras, rec_7, rec_vggt)
    plot_three_way_camera_2d(ras, rec_7, rec_vggt)
    plot_three_way_frustum(ras, rec_7, rec_vggt)
    plot_three_way_fov()
    plot_three_way_motion_range()

    print(f"\n{'=' * 60}")
    print(f"  Done! 14 images saved to {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

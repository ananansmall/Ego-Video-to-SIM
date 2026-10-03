"""Compute contact mask from HaWoR MANO data + RAS scene mesh.

=== 核心逻辑概述 ===

这个脚本解决一个问题：**从视频中重建的手部 MANO 参数中，自动提取"哪些手指在哪几帧接触到了物体"。**

背景：
  - HaWoR 输出了每帧的 MANO 参数 (trans/rot/hand_pose/betas)
  - RAS 输出了场景中的 3D mesh (包含被抓握的物体)
  - 但我们不知道"拇指/食指在哪几帧接触了物体"
  - 这个脚本通过计算"指尖到物体的距离"来自动判断

两种判断模式：

  1. OBJECT mode（物体模式）：
     指尖 → 物体 mesh 顶点的最近距离
     距离 < 5mm  → 接触 (contact_mask = 1.0)
     距离 < 15mm → 接近 (contact_mask = 0.5)
     距离 >= 15mm → 无接触 (contact_mask = 0.0)

  2. FINGER mode（手指模式）：
     两根手指（拇指+食指）的指尖距离
     距离 < 25mm → 抓握姿态 (grasp_mask = 1.0)
     距离 < 50mm → 接近抓握 (grasp_mask = 0.5)
     这个模式不依赖物体 mesh，只用 MANO 几何自身

输出用途：
  - contact_mask.npy  → 作为优化/奖励函数的 contact_rew 信号
  - contact_pos.npy  → 接触引导（contact_guidance）的目标位置
  - grasp_mask.npy   → 标记稳定抓握的时间段，用于扰动注入
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

# === 常量定义 ===
# MANO 关节索引：每个手指的 tip joint（指尖位置）
# 0=wrist, 1-4=thumb(MCP-IP-TIP), 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# tip 关节是每根手指的第 4 个（索引 = 4, 8, 12, 16, 20）
FINGER_TIP_JOINT_IDS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# OBJECT mode 的接触距离阈值（单位：米）
CONTACT_DIST_THRESH = 0.005   # 5mm: 小于此距离 → 判定为"接触"
CONTACT_DIST_WARNING = 0.015  # 15mm: 小于此距离 → 判定为"接近"（潜在接触）

# 物体 mesh 的最大顶点数（超过此数量则随机下采样，避免计算太慢）
MAX_OBJ_VERTS = 2048

# FINGER mode 的抓握距离阈值（拇指-食指距离，单位：米）
GRASP_DIST_THRESH = 0.025     # 25mm: 拇指和食指距离 < 25mm → 判定为抓握


def load_mano_layer(side="right"):
    """加载 MANO 正向运动学层（MANOLayer）。

    从多个候选路径中查找 mano_layer.py，找到后动态导入 MANOLayer 类。
    side: 'left' 或 'right'
    """
    candidates = [
        Path(__file__).parent.parent / "position_retargeting" / "mano_layer.py",
        Path("/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/mano_layer.py"),
        Path("/home/an/robot_world_ws/src/Ego-Video-to-SIM/libs/position_retargeting/mano_layer.py"),
    ]
    mano_layer_path = None
    for p in candidates:
        if p.exists():
            mano_layer_path = p
            break
    if mano_layer_path is None:
        raise FileNotFoundError("Cannot find mano_layer.py")
    print(f"[INFO] Loading MANO layer from: {mano_layer_path}")
    sys.path.insert(0, str(mano_layer_path.parent))
    from mano_layer import MANOLayer
    return MANOLayer


def compute_mango_fk(mano_layer_class, side, trans, rot, hand_pose, betas):
    """对每帧运行 MANO 正向运动学（FK），输出世界坐标系下的关节和网格顶点。

    Args:
        mano_layer_class: MANOLayer 类
        side: 'left' 或 'right'
        trans: (T, 3) 世界坐标系下的手部位置
        rot: (T, 3) 世界坐标系下的全局朝向（axis-angle）
        hand_pose: (T, 45) 手部姿态（15 关节 × 3 轴）
        betas: (T, 10) 或 (10,) 手型参数

    Returns:
        joints_world: (T, 21, 3) 21 个 MANO 关节的世界坐标
        verts_world: (T, 778, 3) 778 个 MANO 网格顶点的世界坐标
    """
    import torch
    T = trans.shape[0]
    if betas.ndim == 2:
        betas_mean = betas.mean(axis=0)
    else:
        betas_mean = betas
    mano_layer = mano_layer_class(side, betas_mean)
    pose = np.concatenate([rot, hand_pose], axis=1).astype(np.float32)  # (T, 48)
    verts_list, joints_list = [], []
    for t in range(T):
        p = torch.from_numpy(pose[t:t+1])        # (1, 48)
        t_tensor = torch.from_numpy(trans[t:t+1]) # (1, 3)
        v, j = mano_layer(p, t_tensor)           # MANO FK
        verts_list.append(v[0].cpu().numpy())     # (778, 3)
        joints_list.append(j[0].cpu().numpy())    # (21, 3)
    return np.array(joints_list), np.array(verts_list)


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 重建结果，提取 MANO 参数。

    HaWoR 输出两种格式：
      1. reconstruction/*.npz: pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid
      2. world_space_res.pth: 元组 (trans, rot, hand_pose, betas, valid)

    hand_idx=0 为左手，hand_idx=1 为右手。

    Returns:
        dict with keys: pred_trans (T,3), pred_rot (T,3), pred_hand_pose (T,45),
                       pred_betas (T,10), pred_valid (T,)
    """
    import joblib
    hawor_path = Path(hawor_dir)
    rec_dir = hawor_path / "reconstruction"
    rec_files = list(rec_dir.glob("*.npz"))
    rec_file = None
    for f in rec_files:
        try:
            d = np.load(f, allow_pickle=True)
            if "pred_trans" in d:
                rec_file = f
                break
        except Exception:
            continue
    if rec_file is None:
        ws_file = hawor_path / "world_space_res.pth"
        if ws_file.exists():
            ws = joblib.load(str(ws_file))
            return {
                "pred_trans": ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0]),
                "pred_rot": ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1]),
                "pred_hand_pose": ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2]),
                "pred_betas": ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3]),
                "pred_valid": ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4]),
            }
        raise FileNotFoundError(f"No reconstruction data found in {hawor_dir}")
    rec = np.load(str(rec_file), allow_pickle=True)
    print(f"[INFO] Loaded reconstruction from: {rec_file.name}")
    trans, rot, hand_pose, betas, valid = (
        rec["pred_trans"], rec["pred_rot"], rec["pred_hand_pose"],
        rec["pred_betas"], rec["pred_valid"],
    )
    if len(trans.shape) == 3:
        # 形状为 (2, T, ...)，选右手或左手
        trans, rot, hand_pose, betas, valid = (
            trans[hand_idx], rot[hand_idx], hand_pose[hand_idx],
            betas[hand_idx], valid[hand_idx],
        )
    result = {
        "pred_trans": trans, "pred_rot": rot,
        "pred_hand_pose": hand_pose, "pred_betas": betas, "pred_valid": valid,
    }
    _fill_nan_frames(result)
    return result


def _fill_nan_frames(data):
    """填充 HaWoR 数据中的 NaN 帧。

    HaWoR 的前几帧和后几帧可能全是 NaN（没有有效重建）。
    用 forward fill（前向填充）+ backward fill（后向填充）补齐，
    并将这些帧的 valid 标记设为 False。
    """
    T = data["pred_trans"].shape[0]
    float_keys = ["pred_trans", "pred_rot", "pred_hand_pose", "pred_betas"]
    nan_mask = np.zeros(T, dtype=bool)
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind == 'f':
            nan_mask |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
    if not nan_mask.any():
        return
    data["pred_valid"][nan_mask] = False
    # 前向填充：用上一个有效值填充 NaN
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind != 'f':
            continue
        last_valid = None
        for i in range(T):
            if not nan_mask[i]:
                last_valid = arr[i].copy()
            elif last_valid is not None:
                arr[i] = last_valid
    # 后向填充：用下一个有效值填充 NaN
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind != 'f':
            continue
        first_valid = None
        for i in range(T - 1, -1, -1):
            if not nan_mask[i]:
                first_valid = arr[i].copy()
            elif first_valid is not None:
                arr[i] = first_valid


def extract_object_mesh_from_ras(ras_dir, mesh_id=None):
    """从 RAS (VGGT-Omega) 场景 GLB 中提取物体 mesh。

    RAS 输出的 final_scene_stage52.glb 包含多个 mesh（bottle, bowl, box, cabinet,
    car, cup, door, scissor, table 等）。我们需要从中找到被抓握的物体。

    选择策略：
      1. 如果指定了 --mesh-id，直接使用 geometry_{mesh_id}
      2. 否则自动选择体积最小（< 10cm³）且顶点数 > 50 的 mesh（最可能是被握持的小物体）
      3. 如果没有小物体，回退到体积中位数的 mesh

    Args:
        ras_dir: RAS 输出目录
        mesh_id: 可选，直接指定 geometry_N 的 N

    Returns:
        verts_world: (V, 3) 物体顶点的世界坐标
    """
    ras_path = Path(ras_dir)
    candidates = [
        ras_path / "final_scene_stage52.glb",
        ras_path / "final_scene1.glb",
        ras_path / "final_scene_initial.glb",
        ras_path / "final_scene.glb",
    ]
    glbc_path = None
    for c in candidates:
        if c.exists():
            glbc_path = c
            break
    if glbc_path is None:
        raise FileNotFoundError(f"No scene GLB found in {ras_dir}")
    print(f"[INFO] Loading scene: {glbc_path}")
    scene = trimesh.load(str(glbc_path))

    # 遍历 scene graph 中的所有实例（instance），提取变换后的世界坐标顶点
    instances = []
    for node_name in list(scene.graph.nodes):
        if node_name in ("world", "grid_z0"):
            continue
        try:
            mat = scene.graph.get_transform(node_name)
            for geom_name, geom in scene.geometry.items():
                v = geom.vertices
                if len(v) < 30:
                    continue
                # 应用变换矩阵，得到世界坐标
                v_world = (mat[:3, :3] @ v.T + np.diag(mat[:3, 3])).T
                bbox = (v_world.min(axis=0), v_world.max(axis=0))
                size = bbox[1] - bbox[0]
                bbox_vol = np.prod(size)
                instances.append({
                    "name": node_name, "geom_name": geom_name, "mat": mat,
                    "verts_world": v_world, "bbox": bbox, "size": size,
                    "vol": bbox_vol, "center": (bbox[0] + bbox[1]) / 2,
                    "n_verts": len(v),
                })
                break
        except Exception:
            continue

    if not instances:
        raise RuntimeError("No instances found in RAS scene")
    print(f"[INFO] Found {len(instances)} instances:")
    print(f"{'name':<20s} {'geom':<15s} {'verts':>7s} {'size':<25s} {'vol':>10s}")
    print("-" * 77)
    for inst in sorted(instances, key=lambda x: x['n_verts']):
        print(f"{inst['name']:<20s} {inst['geom_name']:<15s} {inst['n_verts']:>7d} "
              f"{str(inst['size'].round(3)):<25s} {inst['vol']:>10.4f}")

    if mesh_id is not None:
        target_geom = f"geometry_{mesh_id}"
        for inst in instances:
            if inst["geom_name"] == target_geom:
                print(f"[INFO] Using specified mesh: {inst['name']} ({inst['geom_name']})")
                return inst["verts_world"]
        raise ValueError(f"geometry_{mesh_id} not found")

    # 自动选择：体积最小的小物体
    small = [i for i in instances if i["vol"] < 0.01 and i["n_verts"] > 50]
    if small:
        small.sort(key=lambda x: x["vol"])
        inst = small[0]
        print(f"[INFO] Auto-selected smallest object: {inst['name']} (vol={inst['vol']:.4f})")
        return inst["verts_world"]

    instances.sort(key=lambda x: x["vol"])
    mid = len(instances) // 2
    inst = instances[mid]
    print(f"[INFO] Fallback: selected median-volume instance: {inst['name']}")
    return inst["verts_world"]


def compute_contact_mask_from_object(joints_world, verts_world, obj_verts,
                                      dist_thresh=CONTACT_DIST_THRESH,
                                      dist_warning=CONTACT_DIST_WARNING,
                                      max_obj_verts=MAX_OBJ_VERTS):
    """OBJECT mode：计算每根手指指尖到物体 mesh 的距离，生成接触 mask。

    对每一帧、每一根手指的 tip joint：
      1. 找到离该 tip 最近的物体顶点
      2. 计算最近距离
      3. 距离 < 5mm → contact_mask = 1.0（接触）
         距离 < 15mm → contact_mask = 0.5（接近）
         其他 → 0.0

    Args:
        joints_world: (T, 21, 3) MANO 关节世界坐标
        verts_world: (T, 778, 3) MANO 网格顶点世界坐标（当前未使用）
        obj_verts: (V, 3) 物体顶点世界坐标

    Returns:
        contact_mask: (T, 5) 每根手指的接触状态（1.0/0.5/0.0）
        contact_dists: (T, 5) 每根手指到物体的最近距离
        contact_pos: (T, 5, 3) 每根手指对应的最近物体顶点位置
    """
    T = joints_world.shape[0]
    Nf = len(FINGER_TIP_JOINT_IDS)

    # 如果物体顶点太多，随机下采样以加速计算
    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(42)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts_sub = obj_verts[idx]
    else:
        obj_verts_sub = obj_verts

    contact_mask = np.zeros((T, Nf), dtype=np.float32)
    contact_dists = np.zeros((T, Nf), dtype=np.float32)
    contact_pos = np.zeros((T, Nf, 3), dtype=np.float32)

    for t in range(T):
        for fi, jid in enumerate(FINGER_TIP_JOINT_IDS):
            tip_pos = joints_world[t, jid]                      # 当前手指 tip 位置 (3,)
            dists = np.linalg.norm(obj_verts_sub - tip_pos, axis=1)  # tip 到所有物体顶点的距离
            min_dist = float(dists.min())                        # 最近距离
            min_idx = int(dists.argmin())                        # 最近顶点索引

            contact_dists[t, fi] = min_dist
            contact_pos[t, fi] = obj_verts_sub[min_idx]          # 记录最近物体顶点位置

            if min_dist < dist_thresh:
                contact_mask[t, fi] = 1.0
            elif min_dist < dist_warning:
                contact_mask[t, fi] = 0.5
            # else: 保持 0.0

    return contact_mask, contact_dists, contact_pos


def compute_contact_mask_from_fingers(joints_world, thresh=GRASP_DIST_THRESH):
    """FINGER mode：不依赖物体 mesh，仅用指尖之间的距离判断抓取。

    两个输出：
      1. grasp_mask: 拇指-食指距离 < 25mm → 抓握姿态
      2. contact_mask: 每根手指到最近其他手指的距离 < 5mm → 接触

    Args:
        joints_world: (T, 21, 3) MANO 关节世界坐标
        thresh: 拇指-食指距离阈值（默认 25mm）

    Returns:
        contact_mask: (T, 5) 每根手指的接触状态
        contact_dists: (T, 5) 每根手指到最近其他手指的距离
        contact_pos: (T, 5, 3) 每根手指对应的最近手指 tip 位置
        grasp_mask: (T,) 拇指-食指抓握状态（1.0/0.5/0.0）
        grasp_dist: (T,) 拇指-食指距离
    """
    T = joints_world.shape[0]
    Nf = len(FINGER_TIP_JOINT_IDS)
    tip_ids = FINGER_TIP_JOINT_IDS

    # 1. 抓握检测：拇指和食指的 tip 距离
    grasp_mask = np.zeros(T, dtype=np.float32)
    grasp_dist = np.zeros(T, dtype=np.float32)
    for t in range(T):
        thumb = joints_world[t, tip_ids[0]]
        index = joints_world[t, tip_ids[1]]
        d = np.linalg.norm(thumb - index)
        grasp_dist[t] = d
        if d < thresh:
            grasp_mask[t] = 1.0
        elif d < thresh * 2:
            grasp_mask[t] = 0.5

    # 2. 指尖相互接触检测：每根手指到最近其他手指的距离
    contact_mask = np.zeros((T, Nf), dtype=np.float32)
    contact_dists = np.zeros((T, Nf), dtype=np.float32)
    contact_pos = np.zeros((T, Nf, 3), dtype=np.float32)
    for t in range(T):
        tips = [joints_world[t, jid] for jid in tip_ids]
        for fi in range(Nf):
            other_dists = []
            for fj in range(Nf):
                if fj == fi:
                    continue
                d = np.linalg.norm(tips[fi] - tips[fj])
                other_dists.append(d)
            min_dist = float(min(other_dists))
            closest_idx = int(np.argmin(other_dists))
            contact_dists[t, fi] = min_dist
            contact_pos[t, fi] = tips[closest_idx]
            if min_dist < CONTACT_DIST_THRESH:
                contact_mask[t, fi] = 1.0
            elif min_dist < CONTACT_DIST_WARNING:
                contact_mask[t, fi] = 0.5

    return contact_mask, contact_dists, contact_pos, grasp_mask, grasp_dist


def write_text_report(output_dir, contact_mask, contact_dists, valid_mask,
                      hand_side, mode, extra=None):
    """将接触结果写入人类可读的文本报告。

    报告内容包括：
      - 每根手指的接触统计（接触帧数、比例、平均距离）
      - FINGER mode 下：抓握时间窗口（连续 >= 3 帧的抓握段）
      - 全局统计（总抓握帧数、平均距离）
    """
    path = Path(output_dir) / f"contact_{hand_side}_info.txt"
    T = contact_mask.shape[0]
    with open(path, "w") as f:
        f.write(f"Contact report for {hand_side} hand (mode={mode})\n")
        f.write(f"Total frames: {T}, Valid: {valid_mask.sum() if hasattr(valid_mask, 'sum') else T}\n")
        f.write(f"Fingers: {FINGER_NAMES}\n")
        f.write(f"Thresholds: <{CONTACT_DIST_THRESH:.3f}m = contact, <{CONTACT_DIST_WARNING:.3f}m = close\n\n")
        f.write("--- Per-finger contact statistics ---\n")
        for fi, name in enumerate(FINGER_NAMES):
            n_contact = int((contact_mask[:, fi] >= 0.5).sum())
            n_full = int((contact_mask[:, fi] >= 1.0).sum())
            pct = n_contact / T * 100
            avg_dist = float(contact_dists[:, fi].mean())
            f.write(f"{name:8s}: {n_contact:5d}/{T} close ({pct:.1f}%), "
                    f"{n_full:5d} full, avg_dist={avg_dist:.4f}m\n")
        if mode == "finger" and extra:
            grasp_mask, grasp_dist = extra
            f.write("\n--- Grasp windows (thumb-index < 25mm) ---\n")
            mask = grasp_mask >= 0.5
            windows = []
            start = None
            for t in range(T):
                if mask[t] and start is None:
                    start = t
                elif not mask[t] and start is not None:
                    if t - start >= 3:
                        windows.append((start, t - 1))
                    start = None
            if start is not None and T - start >= 3:
                windows.append((start, T - 1))
            if windows:
                for ws, we in windows:
                    avg_d = grasp_dist[ws:we+1].mean()
                    f.write(f"  frames {ws}-{we} ({we-ws+1} frames), "
                            f"avg_thumb_index_dist={avg_d:.4f}m\n")
            else:
                f.write("  No sustained grasp detected\n")
            n_grasp = int((grasp_mask >= 0.5).sum())
            f.write(f"\nTotal grasp frames: {n_grasp}/{T} ({n_grasp/T*100:.1f}%)\n")
            if n_grasp > 0:
                f.write(f"Avg thumb-index distance during grasp: "
                        f"{grasp_dist[grasp_mask >= 0.5].mean():.4f}m\n")
            f.write(f"Avg thumb-index distance (all frames): {grasp_dist.mean():.4f}m\n")
    print(f"[INFO] Report saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compute contact mask from HaWoR + RAS")
    parser.add_argument("--hawor-dir", required=True)
    parser.add_argument("--ras-dir", default=None)
    parser.add_argument("--hand-side", default="right", choices=["left", "right"])
    parser.add_argument("--output-dir", default="./contact_output")
    parser.add_argument("--dist-thresh", type=float, default=CONTACT_DIST_THRESH)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--mode", choices=["object", "finger", "both"], default="both")
    parser.add_argument("--mesh-id", type=int, default=None, help="RAS geometry index to use as object")
    parser.add_argument("--list-meshes", action="store_true", help="List RAS meshes and exit")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    hand_idx = 0 if args.hand_side == "left" else 1
    prefix = args.hand_side
    print("=" * 60)
    print(f"Contact mask computation: {args.hand_side} hand, mode={args.mode}")
    print("=" * 60)
    if args.list_meshes:
        assert args.ras_dir, "--ras-dir required for --list-meshes"
        extract_object_mesh_from_ras(args.ras_dir, mesh_id=None)
        return
    # Load HAWOR data
    hawor = load_hawor_data(args.hawor_dir, hand_idx=hand_idx)
    trans = hawor["pred_trans"].astype(np.float32)
    rot = hawor["pred_rot"].astype(np.float32)
    hand_pose = hawor["pred_hand_pose"].astype(np.float32)
    betas = hawor["pred_betas"].astype(np.float32)
    valid = hawor["pred_valid"].astype(bool)
    T_total = trans.shape[0]
    print(f"[INFO] HAWOR: {T_total} frames, valid={valid.sum()}")
    T = args.max_frames if args.max_frames > 0 else T_total
    trans, rot, hand_pose, betas = trans[:T], rot[:T], hand_pose[:T], betas[:T]
    valid = valid[:T] if valid.shape[0] >= T else np.ones(T, dtype=bool)
    # MANO FK
    print("[INFO] Computing MANO FK...")
    mano_layer_class = load_mano_layer(args.hand_side)
    joints_world, verts_world = compute_mango_fk(
        mano_layer_class, args.hand_side, trans, rot, hand_pose, betas
    )
    print(f"[INFO] MANO joints: {joints_world.shape}, verts: {verts_world.shape}")
    if np.isnan(joints_world).any():
        print(f"[WARN] MANO output contains NaN! NaN count: {np.isnan(joints_world).sum()}")
    # Compute contact masks
    results = {}
    if args.mode in ("finger", "both"):
        print("[INFO] Computing finger-proximity contact mask...")
        fmask, fdists, fpos, gmask, gdist = compute_contact_mask_from_fingers(
            joints_world, thresh=GRASP_DIST_THRESH
        )
        results["finger"] = (fmask, fdists, fpos, gmask, gdist)
        n_grasp = int((gmask >= 0.5).sum())
        print(f"[INFO] Grasp frames: {n_grasp}/{T} ({n_grasp/T*100:.1f}%)")
        if n_grasp > 0:
            print(f"[INFO] Avg thumb-index dist during grasp: {gdist[gmask >= 0.5].mean():.4f}m")
        print(f"[INFO] Min thumb-index dist: {np.nanmin(gdist):.4f}m")
        print(f"[INFO] Max thumb-index dist: {np.nanmax(gdist):.4f}m")
    if args.mode in ("object", "both"):
        assert args.ras_dir, "--ras-dir required for object mode"
        print("[INFO] Extracting object mesh...")
        obj_verts = extract_object_mesh_from_ras(args.ras_dir, mesh_id=args.mesh_id)
        print(f"[INFO] Object verts: {obj_verts.shape}")
        print("[INFO] Computing object-based contact mask...")
        omask, odists, opos = compute_contact_mask_from_object(
            joints_world, verts_world, obj_verts, dist_thresh=args.dist_thresh
        )
        results["object"] = (omask, odists, opos)
        n_full = int((omask >= 1.0).sum())
        n_close = int((omask >= 0.5).sum())
        print(f"[INFO] Full contact: {n_full}/{omask.size}, Close: {n_close}/{omask.size}")
    # Save
    if "finger" in results:
        fmask, fdists, fpos, gmask, gdist = results["finger"]
        np.save(f"{args.output_dir}/contact_{prefix}_finger.npy", fmask)
        np.save(f"{args.output_dir}/contact_dists_{prefix}_finger.npy", fdists)
        np.save(f"{args.output_dir}/grasp_mask_{prefix}.npy", gmask)
        np.save(f"{args.output_dir}/grasp_dist_{prefix}.npy", gdist)
    if "object" in results:
        omask, odists, opos = results["object"]
        np.save(f"{args.output_dir}/contact_{prefix}_object.npy", omask)
        np.save(f"{args.output_dir}/contact_dists_{prefix}_object.npy", odists)
        np.save(f"{args.output_dir}/contact_pos_{prefix}_object.npy", opos)
    np.save(f"{args.output_dir}/contact_valid_{prefix}.npy", valid)
    print(f"[INFO] Saved numpy files to {args.output_dir}/")
    # Report
    if "finger" in results:
        fmask, fdists, fpos, gmask, gdist = results["finger"]
        write_text_report(args.output_dir, fmask, fdists, valid, prefix, "finger", (gmask, gdist))
    elif "object" in results:
        omask, odists, opos = results["object"]
        write_text_report(args.output_dir, omask, odists, valid, prefix, "object")
    # Summary
    print("\n" + "=" * 60)
    print("CONTACT SUMMARY")
    print("=" * 60)
    if "finger" in results:
        fmask, fdists, fpos, gmask, gdist = results["finger"]
        print(f"\n[Mode: finger proximity]")
        print(f"Grasp (thumb-index close): {int((gmask >= 0.5).sum())}/{T} frames")
        for fi, name in enumerate(FINGER_NAMES):
            n = int((fmask[:, fi] >= 0.5).sum())
            print(f"  {name:8s}: {n:4d} frames in contact")
    if "object" in results:
        omask, odists, ypos = results["object"]
        print(f"\n[Mode: object mesh]")
        for fi, name in enumerate(FINGER_NAMES):
            n = int((omask[:, fi] >= 1.0).sum())
            nc = int((omask[:, fi] >= 0.5).sum())
            print(f"  {name:8s}: {n:4d} full, {nc:4d} close")


if __name__ == "__main__":
    main()

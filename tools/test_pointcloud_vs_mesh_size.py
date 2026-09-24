"""VGGT 点云实测尺寸 vs SAM3D mesh 尺寸对比测试。

验证内容:
1. 两把椅子的 VGGT 点云实测尺寸是否接近（用户观察：点云差不多）
2. mesh 经 T 变换后的实际尺寸与点云实测差多少（量化 SAM3D scale 漂移）
3. 若按点云校准，每个实例需要的 uniform rescale 因子

用法:
    python tools/test_pointcloud_vs_mesh_size.py [scene_dir]
    默认 scene_dir = output_v2/2chair-bottle_vggt_omega
"""
import os
import pickle
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(ROOT, "output_v2", "2chair-bottle_vggt_omega")

PCT_LO, PCT_HI = 0.5, 99.5  # 去掉分割边缘溢出的离群点


def load_data(scene_dir):
    with open(os.path.join(scene_dir, "_pre_dedup_masks.pkl"), "rb") as f:
        pre = pickle.load(f)
    with open(os.path.join(scene_dir, "all_instances.pkl"), "rb") as f:
        inst = pickle.load(f)
    return pre, inst


def match_prededup_instance(pre_chairs, frame_id):
    """在 pre-dedup 实例中找到在 frame_id 上 mask 面积最大的那个实例。"""
    best_idx, best_area = None, -1
    for idx, entries in enumerate(pre_chairs):
        for e in entries:
            if e["frame_id"] == frame_id:
                area = int(e["mask"].sum())
                if area > best_area:
                    best_area, best_idx = area, idx
    return best_idx, best_area


def pointcloud_extent(world_points, entries, frame_ids=None):
    """逐帧统计各轴 [PCT_LO, PCT_HI] 分位范围，再跨帧取 min/max。
    不能把所有帧的点合并后再取分位数——只有个别视角能看到的极值面
    （如椅腿底、椅背顶）占比极小，合并后会被裁掉，且各实例被裁程度不同。
    返回 (extent, n_points, used_frames)。"""
    los, his, n_pts, used = [], [], 0, []
    for e in entries:
        if frame_ids is not None and e["frame_id"] not in frame_ids:
            continue
        m = e["mask"] > 0.5
        if m.sum() < 500:
            continue
        pts = world_points[e["frame_id"]][m]
        los.append(np.percentile(pts, PCT_LO, axis=0))
        his.append(np.percentile(pts, PCT_HI, axis=0))
        n_pts += len(pts)
        used.append(e["frame_id"])
    if not los:
        return None, 0, []
    los, his = np.array(los), np.array(his)
    ext = his.max(0) - los.min(0)
    return ext, n_pts, used


def mesh_world_extent(mesh, T):
    v = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    vw = (T @ v.T).T[:, :3]
    return vw.max(0) - vw.min(0)


def floor_height(wp, sample_frames=(0, 25, 50, 75)):
    """全图点云的低分位作为地面高度（多帧取中位更稳）。"""
    vals = [np.percentile(wp[f][:, :, 2].reshape(-1), 2) for f in sample_frames]
    return float(np.median(vals))


def main():
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    pre, inst = load_data(scene_dir)
    wp = pre["world_points"]  # (N, H, W, 3)

    print(f"scene: {os.path.basename(scene_dir)}")
    fz = floor_height(wp)
    print(f"VGGT 地面高度 z ≈ {fz:.3f} m (全图点云 2% 分位)\n")

    # ── 遍历所有类别，mesh vs 点云 ──
    all_cats = inst["all_instances"]
    summary = []
    for cat, instances in all_cats.items():
        pre_cat = pre["all_masks"].get(cat, [])
        opt_frames = inst["all_optimal_frame_ids"].get(cat, [])
        for ii, c in enumerate(instances):
            mesh, T = c["original_mesh"], c["T"]
            opt_f = opt_frames[ii] if ii < len(opt_frames) else None
            pre_idx, area = match_prededup_instance(pre_cat, opt_f)
            ext_mesh = mesh_world_extent(mesh, T)
            if pre_idx is None:
                print(f"[warn] {cat}_{ii}: 找不到 frame {opt_f} 的 mask，跳过点云对比")
                summary.append((cat, ii, None, ext_mesh, None))
                continue
            entries = pre_cat[pre_idx]
            ext_pc, n_all, used = pointcloud_extent(wp, entries)
            ratio = ext_mesh / np.maximum(ext_pc, 1e-6)
            summary.append((cat, ii, ext_pc, ext_mesh, ratio))

            print(f"===== {cat}_{ii}  (optimal frame {opt_f}, "
                  f"点云 {len(used)} 帧) =====")
            print(f"  {'轴':<4}{'点云ext':>10}{'mesh ext':>12}{'mesh/点云':>10}")
            for a, name in enumerate("xyz"):
                print(f"  {name:<4}{ext_pc[a]:>10.3f}{ext_mesh[a]:>12.3f}"
                      f"{ratio[a]:>10.2f}")
            print(f"  → uniform_scale = median(ratio) = {float(np.median(ratio)):.3f}\n")

    # ── 物理先验校验: 判断 VGGT 坐标轴尺度是否可信 ──
    print("===== 物理先验校验（只用 VGGT 点云，与 SAM3D 无关）=====")
    checks = []
    for cat, ii, ext_pc, _, _ in summary:
        if ext_pc is None:
            continue
        if cat == "chair":
            checks.append((f"{cat}_{ii} 高度", float(ext_pc[2]), "椅子带靠背 0.85~1.05"))
        elif cat == "table":
            checks.append((f"{cat}_{ii} 高度", float(ext_pc[2]), "桌子 0.70~0.80"))
        elif cat == "bottle":
            checks.append((f"{cat}_{ii} 高度", float(ext_pc[2]), "水瓶 0.15~0.30"))
    for name, val, prior in checks:
        print(f"  点云实测 {name:<14} = {val:.3f} m   (常见范围: {prior})")

    # 桌面高度（离地）: table mask 点云的最高点 - 地面
    for cat, ii, ext_pc, _, _ in summary:
        if cat != "table" or ext_pc is None:
            continue
        pre_idx, _ = match_prededup_instance(
            pre["all_masks"]["table"],
            inst["all_optimal_frame_ids"]["table"][ii])
        entries = pre["all_masks"]["table"][pre_idx]
        top = max(np.percentile(wp[e["frame_id"]][e["mask"] > 0.5][:, 2], 99.5)
                  for e in entries if (e["mask"] > 0.5).sum() > 500)
        print(f"  点云实测 桌面离地高度     = {top - fz:.3f} m   (常见范围: 0.70~0.78)")

    print("\n结论解读:")
    print("  · 若上表点云实测值都落在常见范围 → VGGT 世界坐标尺度可信，问题在 SAM3D")
    print("  · 若所有物体 mesh/点云 都 >> 1 → SAM3D 系统性放大所有物体")


if __name__ == "__main__":
    main()

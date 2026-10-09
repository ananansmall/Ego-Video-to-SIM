"""方案 B: post-hoc 点云校准 mesh 尺寸 (docs/SAM3D_SCALE_DRIFT_REPORT.md §5.2)

对每个实例:
1. 逐帧 mask 点云质心 → KMeans(K=3) 聚类
   - 移动判定: 簇间质心最小间距 > 0.8×物体尺寸 (中位数逐帧 ext 范数) → 移动过
   - 移动过 (如被拿起的 bottle) → 取质心最紧凑的簇为静止段, 逐帧 ext 取中位数
     (偏离报告原文的"最大簇+min/max": 实测最大簇会选中运动段 -- bottle 运动
     帧数多于任一停留位; min/max 会被 VGGT 帧间漂移污染 -- bottle 在静止段内
     质心仍游走 0.3m, 被放大 6 倍; 单帧测距则不受漂移影响)
   - 没移动 (chair/table) → 全部帧跨帧 min/max
     (静止段视角可能从没见过桌腿, 只用静止段会把桌子缩 2.2 倍;
      min/max 能覆盖低频可见面, 且复现报告锚点 chair 1.53/2.11、table 1.62)
   - k 因子: 静止物体取三轴比例中位数; 移动物体只取 mesh 最长轴 (主轴)
     的比例 -- 点云水平向常被阴影/透明材质深度噪声污染 (bottle 7cm 宽
     被测成 22cm), 三轴中位数会被污染轴带偏
2. 逐帧 [0.5%, 99.5%] 分位 (不合并采样: 个别视角才能看到的极值面合并后会被裁掉),
   并丢弃 mask 像素 < 中位数 50% 的碎片帧 (chair_1 入画碎片段只看到一角)
3. uniform scale (k 因子取法见上), 只改 T 不改 mesh 顶点
4. T_new = Translate(pivot) @ Scale(k) @ Translate(-pivot) @ T, pivot=(cx, cy, min_z)
   (以底面中心为锚: 水平位置不变, 落地家具缩放后仍贴地, 动态物体位置不动)

输出 (写入场景目录):
- all_instances_rescaled.pkl   只改 T, 其余键原样保留
- rescale_check_before.glb / rescale_check_after.glb   预览对比用

用法:
    python tools/rescale_meshes_by_pointcloud.py [scene_dir]
    默认 scene_dir = output_v2/2chair-bottle_vggt_omega
"""
import os
import pickle
import sys

import numpy as np
import trimesh
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(ROOT, "output_v2", "2chair-bottle_vggt_omega")

MIN_MASK_PTS = 500        # 单帧 mask 少于该像素数不参与统计
KMEANS_K = 3
MIN_FRAMES_FOR_KMEANS = 6  # 帧数太少不做聚类, 全部当静止段
MOVED_RATIO = 0.8         # 簇间最小间距 > 该比例×物体尺寸 → 判定移动过
SLIVER_RATIO = 0.5        # mask 像素 < 该比例×中位数 → 视为只看到一角的碎片帧


def frame_stats(world_points, entries):
    """逐帧 (fid, 质心, px, los, his); 过滤 <MIN_MASK_PTS 的帧。"""
    stats = []
    for e in entries:
        m = e["mask"] > 0.5
        if m.sum() < MIN_MASK_PTS:
            continue
        pts = world_points[e["frame_id"]][m]
        stats.append((e["frame_id"], pts.mean(axis=0), int(m.sum()),
                      np.percentile(pts, 0.5, axis=0),
                      np.percentile(pts, 99.5, axis=0)))
    return stats


def aggregate(stats):
    """per-frame los/his → 点云尺寸。"""
    los = np.array([s[3] for s in stats])
    his = np.array([s[4] for s in stats])
    return his.max(0) - los.min(0)


def estimate_extent(world_points, entries):
    """返回 (ext_pc, moved, n_used, n_total, z_max)。

    moved: 簇间质心最小间距 > MOVED_RATIO×物体尺寸 (物体尺寸用逐帧 ext
    范数的中位数近似; 不能用 max -- bottle 的手持污染帧会把尺寸撑大,
    导致移动误判为静止)。移动过 → 静止段内取逐帧 ext 的中位数
    (单帧测距不受 VGGT 帧间漂移影响); 没移动 → 全部帧跨帧 min/max
    (覆盖低频可见面, 如桌腿)。

    z_max: 点云 z 方向最大值(跨帧), 供落地静止物体的「地面-顶面」高度计算。
    桌腿/椅腿是细结构, VGGT 深度不可靠 → 点云 min-max 会停在桌面底部而非
    地面; 但桌面面积大、分割可靠 → z_max 可信, 与已知地面(z≈0)合并即恢复
    真实高度。
    """
    stats = frame_stats(world_points, entries)
    n_total = len(stats)
    if n_total == 0:
        return None, False, 0, 0, None
    # 丢弃碎片帧
    px = np.array([s[2] for s in stats])
    stats = [s for s, p in zip(stats, px) if p >= SLIVER_RATIO * np.median(px)]
    if len(stats) < MIN_FRAMES_FOR_KMEANS:
        ext = aggregate(stats)
        z_max = float(np.array([s[4][2] for s in stats]).max())
        return ext, False, len(stats), n_total, z_max

    cents = np.array([s[1] for s in stats])
    k = min(KMEANS_K, len(stats))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(cents)
    means = np.array([cents[labels == lb].mean(0) for lb in range(k)])
    spreads = np.array([float((cents[labels == lb].max(0)
                               - cents[labels == lb].min(0)).sum())
                        for lb in range(k)])
    obj_size = float(np.median([np.linalg.norm(s[4] - s[3]) for s in stats]))
    sep = min(np.linalg.norm(means[i] - means[j])
              for i in range(k) for j in range(i + 1, k))
    moved = bool(sep > MOVED_RATIO * obj_size)
    if moved:
        # 静止段 = 质心最紧凑的簇; 逐帧 ext 取中位数 (漂移不敏感)
        best = int(np.argmin(spreads))
        sel = [s for s, lb in zip(stats, labels) if lb == best]
        ext = np.median(np.array([s[4] - s[3] for s in sel]), axis=0)
        z_max = float(np.array([s[4][2] for s in sel]).max())
        return ext, True, len(sel), n_total, z_max
    ext = aggregate(stats)
    z_max = float(np.array([s[4][2] for s in stats]).max())
    return ext, False, len(stats), n_total, z_max


def rescale_all_instances(output_path, all_instances, all_optimal_frame_ids):
    """对内存中的 all_instances 做点云尺寸校准 (供 mainv3 管线与独立脚本共用)。

    参数:
        output_path: 场景目录 (内含 Stage2 产出的 _pre_dedup_masks.pkl)
        all_instances: {cat: [instance_info, ...]}, instance_info 含 original_mesh/T
        all_optimal_frame_ids: {cat: [frame_id, ...]}
    返回:
        校准后的 all_instances (结构不变, 仅 T 被替换; 失败实例 k=1 原样保留)
    异常:
        _pre_dedup_masks.pkl 不存在等严重问题时抛出, 调用方自行兜底
    """
    with open(os.path.join(output_path, "_pre_dedup_masks.pkl"), "rb") as f:
        pre = pickle.load(f)
    wp = pre["world_points"]  # (N, H, W, 3)

    print(f"scene: {os.path.basename(output_path)}")
    hdr = (f"{'instance':<14}{'策略':<6}{'帧':>7}{'k':>7}"
           f"{'before(x/y/z)':>22}{'after(x/y/z)':>22}")
    print(hdr)
    print("-" * len(hdr))

    rescaled = {}
    all_after_ratios = []
    for cat, instances in all_instances.items():
        pre_cat = pre["all_masks"].get(cat, [])
        opt_frames = all_optimal_frame_ids.get(cat, [])
        rescaled[cat] = []
        for ii, c in enumerate(instances):
            mesh, T = c["original_mesh"], np.asarray(c["T"], dtype=np.float64)
            opt_f = opt_frames[ii] if ii < len(opt_frames) else None
            # 在 pre-dedup 片段中找 optimal frame 上 mask 最大的那个
            best_idx, best_area = None, -1
            for idx, es in enumerate(pre_cat):
                for e in es:
                    if e["frame_id"] == opt_f:
                        area = int((e["mask"] > 0.5).sum())
                        if area > best_area:
                            best_area, best_idx = area, idx
            if best_idx is None:
                print(f"[warn] {cat}_{ii}: 找不到 frame {opt_f} 的 mask, k=1 跳过")
                rescaled[cat].append({**c, "T": T.copy()})
                continue

            ext_pc, moved, n_used, n_total, z_max = estimate_extent(
                wp, pre_cat[best_idx])
            assert ext_pc is not None, f"{cat}_{ii}: 无有效帧"

            # 当前 mesh 世界尺寸 → k
            vw = (T @ np.hstack([mesh.vertices,
                                 np.ones((len(mesh.vertices), 1))]).T).T[:, :3]
            ext_mesh = vw.max(0) - vw.min(0)

            if moved:
                # 移动物体: 点云水平向常被阴影/透明材质深度噪声污染
                # (bottle mesh 7cm 宽被测成 22cm), 三轴中位数失效,
                # 只用 mesh 最长轴 (主轴, 通常为高度) 的比例
                a = int(np.argmax(ext_mesh))
                k_val = float(ext_pc[a] / max(ext_mesh[a], 1e-6))
                scale_x = scale_y = scale_z = k_val
                is_hybrid = False
            elif not moved and vw[:, 2].min() <= 0.30:
                # 落地静止物体 (如 table/chair): 混合策略
                # - 水平向 (x/y): 点云 min-max 比例 (桌面/椅面面积大, 分割可靠)
                # - 垂直向 (z): 根据点云 z_min 是否接近地面决定
                #   * 点云 z_min 接近地面 (如椅子, 椅背/座面大, 分割可靠): 用 min/max
                #   * 点云 z_min 远离地面 (如桌子, 桌腿细, VGGT 深度不可靠): 用顶面到地面的实际高度
                #     z_range = z_max - ground_z (桌面面积大、分割可靠 → z_max 可信;
                #     桌腿点云停在桌面底部而非地面, 必须减去 ground_z 才能恢复真实高度)
                # ground_z 用 2% 分位 (不是 min): min 会被埋地的椅子污染
                # (椅子埋地 z=-0.682, 拉低全场景 min; 2% 分位在地面附近, 更可靠)
                _all_z = np.concatenate([wp[i].reshape(-1, 3) for i in range(len(wp))])[:, 2]
                ground_z = float(np.percentile(_all_z, 2))
                z_min_pc = float(np.array([s[3][2] for s in frame_stats(wp, pre_cat[best_idx])]).min())
                z_range = float(z_max - ground_z)  # 地面到顶面的实际高度
                k_xy = float(np.median(ext_pc[:2] / np.maximum(ext_mesh[:2], 1e-6)))
                if z_min_pc - ground_z > 0.30:
                    # 点云 z_min 远离地面, 分两种情况:
                    if z_min_pc - ground_z >= 0.50:
                        # 抬高物体 (放在家具上, 如桌上的瓶子):
                        # z_max - ground_z 会把家具高度算进物体 → 严重放大
                        # (HOI4D 实测: 0.25m 瓶子被算成 1.15m → k_z=2.5 → 1m 高瓶)
                        # 此时点云自身 z 跨度 (z_max - z_min_pc) 就是物体真实高度
                        k_z = float((z_max - z_min_pc) / max(ext_mesh[2], 1e-6))
                    else:
                        # 落地家具但腿被遮挡 (gap 略超阈值): 顶面到地面 = 真实高度
                        k_z = float(z_range / max(ext_mesh[2], 1e-6))
                    is_hybrid = True
                else:
                    # 点云 z_min 接近地面 → 底部可见, 用 min/max
                    k_z = float(ext_pc[2] / max(ext_mesh[2], 1e-6))
                    is_hybrid = False
                scale_x, scale_y, scale_z = k_xy, k_xy, k_z
                k_val = (scale_x * scale_y * scale_z) ** (1/3)  # 仅用于打印
            else:
                # 非落地静止物体 (如桌上瓶子): 原始三轴比例
                k_val = float(np.median(ext_pc / np.maximum(ext_mesh, 1e-6)))
                scale_x = scale_y = scale_z = k_val
                is_hybrid = False

            # 以底面中心为锚缩放 (cx, cy, min_z): 水平位置不变, 垂直向上缩,
            # 落地家具 (min z≈0) 缩放后仍贴地 —— 绕质心缩放会让桌子悬浮
            pivot = np.array([vw[:, 0].mean(), vw[:, 1].mean(), vw[:, 2].min()])
            S = np.diag([scale_x, scale_y, scale_z, 1.0])
            off = np.eye(4)
            off[:3, 3] = -pivot
            off_inv = np.eye(4)
            off_inv[:3, 3] = pivot
            T_new = off_inv @ S @ off @ T

            # 校准后 ratio
            vw2 = (T_new @ np.hstack([mesh.vertices,
                                      np.ones((len(mesh.vertices), 1))]).T).T[:, :3]
            if is_hybrid:
                # 混合策略: z 用校准后高度与「点云自身 z 跨度」比 (应≈1.0)。
                # 必须与 k_z 的分子同源: k_z = pc_z_span/mesh_z, 所以校验也用
                # pc_z_span。早先这里用 z_range (=z_max-ground_z) 做参照, 而
                # ground_z 是全场景 2% 分位、会被埋地家具拉低 (HOI4D 实测 -0.757,
                # 真实地面远高于此), z_range 被撑大约 0.76m → after_z 一律 0.2~0.3,
                # 把测量本来正确的实例 (pc/mz≈1.0) 全判成垃圾而回退 k=1。
                # 抬高分支 k_z 用点云自身跨度, 正好与此处同源。
                pc_z_span = float(z_max - z_min_pc)   # k_z 的分子, 与之同源
                if pc_z_span <= 1e-6:
                    pc_z_span = float(ext_pc[2])
                z_after = vw2[:, 2].max() - vw2[:, 2].min()
                after_ratio = np.array([
                    (vw2[:, 0].max() - vw2[:, 0].min()) / max(ext_pc[0], 1e-6),
                    (vw2[:, 1].max() - vw2[:, 1].min()) / max(ext_pc[1], 1e-6),
                    z_after / max(pc_z_span, 1e-6),
                ])
            else:
                after_ratio = (vw2.max(0) - vw2.min(0)) / np.maximum(ext_pc, 1e-6)
            fmt = lambda r: "/".join(f"{x:.2f}" for x in r)
            tag = "混合" if is_hybrid else ("静止段" if moved else "全帧")

            # ── sanity guard: 校准自校验, 垃圾测量回退 k=1 ──
            # k 由点云测得, 校准后 mesh 应与点云吻合; 吻合不上说明测量本身
            # 不可信 (薄/平/透明物体如窗帘/电视/透明瓶的深度噪声),
            # 此时保留 SAM3D 原始尺寸 (k=1) 比套用垃圾缩放更安全。
            # 实测依据 (3chair 23 实例): 好校准 after≈1.0±0.3 且各轴一致;
            # 垃圾校准 after 出现 1.7~14.8 的偏离 (curtain_6 达 14.84)。
            _K_MIN, _K_MAX = 0.5, 2.0             # 移动物体单轴缩放比合法范围
            _AFTER_MIN, _AFTER_MAX = 0.70, 1.55   # 校准后各轴比值合法范围
            if moved:
                # 移动物体只信最长轴, after 在该轴恒=1 无信息量, 改查 k 本身
                guard_ok = _K_MIN <= k_val <= _K_MAX
                guard_why = f"移动分支 k={k_val:.2f} 超出 [{_K_MIN},{_K_MAX}]"
            else:
                _lo, _hi = float(after_ratio.min()), float(after_ratio.max())
                guard_ok = (_lo >= _AFTER_MIN and _hi <= _AFTER_MAX)
                guard_why = f"校准后比值 {fmt(after_ratio)} 超出 [{_AFTER_MIN},{_AFTER_MAX}]"
            if is_hybrid:
                # 混合分支 after_z ≡ 1.0 (k_z 由同一条公式构造), 自校验失明,
                # 补查 k_z 本身: 抬高物体 z 公式出错时 k_z 会到 2.5+
                _KZ_MIN, _KZ_MAX = 0.4, 2.0
                if not (_KZ_MIN <= k_z <= _KZ_MAX):
                    guard_ok = False
                    guard_why = f"混合分支 k_z={k_z:.2f} 超出 [{_KZ_MIN},{_KZ_MAX}]"
            if not guard_ok:
                print(f"{cat}_{ii:<9}[guard] 测量不可信, 保留原始尺寸: {guard_why}")
                rescaled[cat].append({**c, "T": T.copy()})
                continue

            all_after_ratios.append(after_ratio)
            print(f"{cat}_{ii:<9}{tag:<6}{n_used:>3}/{n_total:<3}"
                  f"{k_val:>7.3f}  {fmt(ext_mesh / np.maximum(ext_pc, 1e-6)):>20}"
                  f"{fmt(after_ratio):>20}")

            rescaled[cat].append({**c, "T": T_new})

    # 校准收敛性检查
    if all_after_ratios:
        arr = np.array(all_after_ratios)
        ok = bool(np.all(np.abs(arr - 1.0) < 0.10))
        print(f"\n校准后各轴 ratio 最大偏差: {np.abs(arr - 1.0).max():.3f} "
              f"→ {'✅ 收敛 (全部 <0.10)' if ok else '⚠️ 存在各向异性残差 >0.10'}")
    return rescaled


def main():
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    with open(os.path.join(scene_dir, "all_instances.pkl"), "rb") as f:
        inst = pickle.load(f)

    rescaled_all = rescale_all_instances(
        scene_dir, inst["all_instances"], inst["all_optimal_frame_ids"])

    rescaled = {"all_instances": rescaled_all,
                "all_optimal_frame_ids": inst["all_optimal_frame_ids"]}
    for key in inst:
        if key not in rescaled:
            rescaled[key] = inst[key]

    with open(os.path.join(scene_dir, "all_instances_rescaled.pkl"), "wb") as f:
        pickle.dump(rescaled, f)
    print(f"已写入 {os.path.join(scene_dir, 'all_instances_rescaled.pkl')}")

    # 前后 GLB (z-up → y-up, 与 mainv2.save_final_glb 一致, 供 render_glb_preview 对比)
    y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for name, src in (("rescale_check_before", inst["all_instances"]),
                      ("rescale_check_after", rescaled_all)):
        scene = trimesh.Scene()
        for cat, lst in src.items():
            for i, c in enumerate(lst):
                m = c["original_mesh"].copy()
                m.apply_transform(c["T"])
                scene.add_geometry(m, node_name=f"{cat}_{i}")
        scene.apply_transform(y_up)
        out = os.path.join(scene_dir, f"{name}.glb")
        scene.export(out)
        print(f"已写入 {out}")


if __name__ == "__main__":
    main()

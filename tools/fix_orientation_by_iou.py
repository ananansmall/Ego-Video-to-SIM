#!/usr/bin/env python3
"""用 IoU 目标函数选物体朝向 —— 取代任何关于"正立/倒置"的启发式。

原理 (已实测验证):
  把每个物体渲染到 VGGT 深度图上, 与 VGGT 深度比对得到 IoU。
  朝向错了 IoU 就会掉。所以不需要"椅子应该腿朝下"这类类别规则,
  只要在若干候选朝向里挑 IoU 最高的那个 —— 目标是可测量的,
  与场景/类别无关。

实测依据 (output_v3/weixin_v10, 帧 50~65):
  table_1 (0.66x0.65x0.28): 原始 IoU=0.0147 → 翻转 180° 后 0.5892 (好 40 倍)
  table_2 (1.01x0.98x1.22): 原始 0.0711 → 翻转后 0.0536  (原始已正确)
  table_0 (平板 1.05x0.97x0.05): 翻转前后 0.9766/0.9745 (平板上下对称, 不变 —— 正常)
  z 偏移 0.20m → IoU 0.0039; 绕 X 转 90° → 0.2162 (指标对位姿敏感)

为什么 Stage4 没修好:
  stage4/icp_optimization.py 的 ICP 是 position-only (帮助文本: "default:
  position-only rigid transform"), 只在位置上收敛, 不搜索朝向。所以倒置的
  物体即使 IoU 极低也保持倒置。

候选朝向: 6 个轴向 (±X/±Y/±Z 的 90° 组合中覆盖上方向翻转的 6 种) + 原始。
  这是"物体应该在 6 个正交朝向之一"的弱先验, 不涉及物体类别。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.eye(4)
    if axis == "x":
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    elif axis == "y":
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    else:
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def _about_center(R, center):
    B = np.eye(4); B[:3, 3] = center
    S = np.eye(4); S[:3, 3] = -center
    return B @ R @ S


def candidates():
    """候选朝向: 原始 + 绕三个轴各翻 180° + 绕 X/Y 转 ±90°。

    只覆盖"上方向"的 6 种可能 (Z±, Y±, X±), 这正是"物体立着/躺着/倒着"
    的全部离散情形, 不涉及类别知识。
    """
    yield ("original", np.eye(4))
    for ax in ("x", "y", "z"):
        yield (f"flip180_{ax}", _rot(ax, 180))
    for ax in ("x", "y"):
        yield (f"rot90_{ax}", _rot(ax, 90))
        yield (f"rot270_{ax}", _rot(ax, 270))


def compute_silhouette_iou(mesh, T, extrinsics, renderer, frames, masks):
    """渲染轮廓 vs **真实帧的 SAM3 mask** 的 IoU —— 这才是真值参照。

    为什么必须用真实 mask 而不是 VGGT 深度:
      深度 IoU 是拿"重建出的 mesh"去比"另一次重建出的深度", 两个都不准,
      会严重夸大问题。实测 weixin_v10 的 table_1:
        深度 IoU  0.062 → 0.821 (差 13 倍, 据此会把椅子翻成倾斜块状物)
        真实帧 IoU 0.654 → 0.712 (只差 6%, 朝向基本没问题)
      输入本来就是视频, 帧里的 mask 是一手观测, 拿它当参照才可信。
    """
    if not masks:
        return 0.0
    ious = []
    for f in frames:
        if f not in masks:
            continue
        try:
            _, _, m_ren = renderer.render_mesh(mesh, T, extrinsics[f])
        except Exception:
            continue
        m_ren = np.asarray(m_ren) > 0
        m_real = np.asarray(masks[f]) > 0
        if m_real.sum() == 0 or m_ren.sum() == 0:
            continue
        inter = (m_ren & m_real).sum()
        union = (m_ren | m_real).sum()
        if union > 0:
            ious.append(float(inter) / float(union))
    return float(np.mean(ious)) if ious else 0.0



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="场景目录 (需含 color/depth/extrinsics)")
    ap.add_argument("--output_glb", default=None)
    ap.add_argument("--min_gain", type=float, default=0.05,
                    help="IoU 提升超过该值才改朝向 (避免抖动)")
    ap.add_argument("--min_iou", type=float, default=0.30,
                    help="最佳 IoU 低于该值说明度量不可信, 一律不改 (见下方说明)")
    ap.add_argument("--max_frames", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from stage4.run_alignment import load_vggt_results, load_scene_instances
    from stage4.renderer import MeshRenderer
    from stage4.projection_alignment import compute_depth_iou

    D = args.scene
    # 兼容旧命名: 早期产物是 final_scene.glb / final_scene_stage5.glb,
    # 而 stage4.load_scene_instances 只认 final_scene_basic.glb。
    # 缺就顺链到一个存在的同名产物, 不改动原始文件。
    _basic = os.path.join(D, "final_scene_basic.glb")
    if not os.path.isfile(_basic):
        for alt in ("final_scene_stage5.glb", "final_scene.glb",
                    "final_scene_initial.glb"):
            p = os.path.join(D, alt)
            if os.path.isfile(p):
                try:
                    os.symlink(alt, _basic)
                    print(f"   (链 final_scene_basic.glb → {alt})")
                except Exception:
                    import shutil; shutil.copyfile(p, _basic)
                break
    v = load_vggt_results(D)
    K = np.asarray(v["intrinsic"], float)
    insts = load_scene_instances(D)
    H, W = np.asarray(v["depths"])[0].shape
    renderer = MeshRenderer(K, W, H)

    # 每个实例用它自己可见的帧 —— 用别的帧 IoU 恒为 0 (实测 table_1 在
    # 帧 0~30 不可见, IoU=0.0000, 会误判成"怎么转都不对")
    pre = os.path.join(D, "_pre_dedup_masks.pkl")
    vis = {}
    if os.path.isfile(pre):
        import pickle
        am = pickle.load(open(pre, "rb"))["all_masks"]
        for cat, trajs in am.items():
            for i, tr in enumerate(trajs):
                f = sorted(m["frame_id"] for m in tr)
                if f:
                    vis[f"{cat}_{i}"] = f
    n_frames = int(np.asarray(v["depths"]).shape[0])

    print(f"{'实例':18s} {'当前IoU':>8s} {'最佳':>8s} {'最佳朝向':>14s} {'增益':>7s}")
    print("-" * 62)
    changes = {}
    total_gain = 0.0
    for cat, lst in insts.items():
        for i, info in enumerate(lst):
            name = f"{cat}_{i}"
            mesh = info["original_mesh"]
            T = np.asarray(info["T"], float)
            fids = vis.get(name) or list(range(n_frames))
            if len(fids) > args.max_frames:
                idx = np.linspace(0, len(fids) - 1, args.max_frames).astype(int)
                fids = [fids[j] for j in idx]
            # 该实例在这些帧上的真实 SAM3 mask —— silhouette IoU 的参照
            fmask = {}
            for cat2, trajs2 in am.items():
                for i2, tr2 in enumerate(trajs2):
                    if f"{cat2}_{i2}" != name:
                        continue
                    for m in tr2:
                        fmask[m["frame_id"]] = m["mask"]
            center = np.asarray(mesh.bounds.mean(0), float)
            best_r, best_iou, best_tag = None, -1.0, "original"
            for tag, R in candidates():
                Tc = T @ _about_center(R, center)
                iou = compute_silhouette_iou(mesh, Tc, v["extrinsics"],
                                             renderer, fids, fmask)
                if iou > best_iou:
                    best_iou, best_tag, best_r = iou, tag, Tc
            cur = compute_silhouette_iou(mesh, T, v["extrinsics"],
                                         renderer, fids, fmask)
            gain = best_iou - cur
            mark = ""
            # 有效性门禁: 最佳 IoU 都低于阈值时, 说明这个场景的 mesh 与 VGGT
            # 深度根本不在同一尺度上, IoU 已退化成噪声 —— 此时任何"最高分"都是
            # 随机选择, 改朝向只会破坏本来正确的物体。
            # 实测教训: output_v2/2chair-bottle_rescaled 的 mesh 已被尺度标定,
            # 与 VGGT 原始深度(相对尺度)不匹配, 5 个实例 IoU 全部 <0.19
            # (weixin_v10 未缩放场景是 0.98/1.00)。工具当时把一把本来正立的
            # 椅子翻倒了。
            if best_iou < args.min_iou:
                mark = "  (IoU过低, 度量不可信, 不改)"
            elif gain > args.min_gain:
                changes[name] = best_r
                total_gain += gain
                mark = "  ← 改"
            print(f"{name:18s} {cur:8.4f} {best_iou:8.4f} {best_tag:>14s} "
                  f"{gain:+7.4f}{mark}")
            sys.stdout.flush()

    print("-" * 62)
    print(f"共 {len(changes)} 个实例改朝向, IoU 总增益 {total_gain:+.3f}")
    if args.dry_run or not changes:
        return

    for cat, lst in insts.items():
        for i, info in enumerate(lst):
            nm = f"{cat}_{i}"
            if nm in changes:
                info["T"] = changes[nm]
    import pickle
    with open(os.path.join(D, "all_instances_orientation_fixed.pkl"), "wb") as f:
        pickle.dump({"all_instances": insts}, f)
    print(f"已写 {D}/all_instances_orientation_fixed.pkl")

    try:
        from mainv3 import save_final_glb
        out = args.output_glb or "final_scene_orientation_fixed.glb"
        save_final_glb(insts, D, filename=out)
        print(f"已导出 {D}/{out}")
    except Exception as e:
        print(f"GLB 导出失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

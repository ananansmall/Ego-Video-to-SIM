#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 5.5 确定性 QC + 位姿调整 CLI（stage5.5 的工具集, 也是 opencode agent 的工具）

设计原则（LiteReality-Agent 方案A落地）:
  - 纯确定性几何, 可复现: check_support / check_glb 只读检测
  - 修改只作用于 "工作 GLB"（stage55_work.glb）, 绝不碰 stage5 产物
  - 坐标系: z-up, 单位米 (与 stage5 内部空间一致; GLB 文件本身仍是 y-up)

命令:
  inspect       --glb X                              列出所有实例 (名称/bbox/体积)
  check_support --glb X [--tol 0.005] [--cap 0.15]   支撑检测 (floating/embedded/ok + fix_dz)
  snap_support  --glb X [--passes 3]                 应用 Z 贴合修复 (root-first 链式, ≤cap)
  check_glb     --glb X [--min-depth 0.003]          真实三角形碰撞检测 + world_dxdy 修复向量
  nudge         --glb X [--rounds 4]                 确定性兜底: 沿碰撞向量推挤直到收敛
  set_xy        --glb X --obj NAME --dx F --dy F     XY 平移物体 (z-up 世界系)
  set_z         --glb X --obj NAME --dz F            Z 平移物体
  render        --glb X [--out PNG]                  四视图预览 PNG (复用 render_glb_preview)

每次修改 (set_xy/set_z/snap/nudge) 追加记录到 <glb 同目录>/stage55_moves.jsonl。
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Y_UP_TO_Z_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)
Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

# stage5 的 z-up 约定里重力沿 -z (地面 z=0, 物体在负 z)。
# stage5.5 统一转成常规 +z 朝上: X180 = 绕 X 轴 180° 旋转 (proper rotation, 无镜像)。
X180 = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

# 结构元素 (不可作为被推动对象): 类别名前缀
STRUCTURAL_PREFIXES = ("floor", "wall", "ceiling")


# ---------------------------------------------------------------- GLB I/O

def load_instances(glb_path):
    """加载 GLB → {name: mesh(常规 +z 朝上世界坐标, 已烘焙)}, node 名 {category}_{i}

    还原链: X180 @ Y_UP_TO_Z_UP @ node_transform, 与 save_instances 精确互逆。
    """
    import trimesh
    scene = trimesh.load(glb_path, process=False)
    insts = {}
    g = scene.graph
    for node_name in g.nodes_geometry:
        if node_name == 'world':
            continue
        parts = node_name.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        _, geom_name = g[node_name]
        if geom_name not in scene.geometry:
            continue
        mesh = scene.geometry[geom_name].copy()
        transform_to_node, _ = g.get(frame_to=node_name)
        mesh.apply_transform(transform_to_node)
        mesh.apply_transform(Y_UP_TO_Z_UP)
        mesh.apply_transform(X180)
        insts[node_name] = mesh
    return insts


def save_instances(insts, glb_path):
    """保存 {name: mesh(+z 朝上)} → GLB, 与 load_instances 精确互逆"""
    import trimesh
    scene = trimesh.Scene()
    for name, mesh in insts.items():
        scene.add_geometry(mesh, node_name=name)
    scene.apply_transform(X180 @ Z_UP_TO_Y_UP)
    os.makedirs(os.path.dirname(os.path.abspath(glb_path)), exist_ok=True)
    scene.export(glb_path)


def compare(glb_path, out=None, cap=3, max_width=512):
    """渲染真实相机视角并与实拍照片并排拼接 (每帧一行: 左=渲染 右=实拍)

    供 agent 视觉迭代对齐: 同机位直接对比布局差异。返回输出 PNG 路径。
    迭代用低配置 (3 帧/512宽) 保速度; 最终报告用 render/预览走高清。
    """
    import cv2
    import numpy as np
    from render_glb_preview import get_optimal_frame_ids, render_camera_views
    scene_dir = os.path.dirname(os.path.abspath(glb_path))
    fids = get_optimal_frame_ids(scene_dir, cap=cap) or [0]
    tiles, _ = render_camera_views(glb_path, scene_dir, fids, max_width=max_width)
    rows = []
    for fid, tile in zip(fids, tiles):
        photo = cv2.imread(os.path.join(scene_dir, "color", f"{fid}.jpg"))
        if photo is None:
            continue
        # render_camera_views 返回 RGB, cv2/imread 是 BGR: 统一转 BGR 再拼图,
        # 否则 R/B 互换 (黄碗变青碗)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        h = tile.shape[0]
        scale = h / photo.shape[0]
        photo = cv2.resize(photo, (max(1, int(photo.shape[1] * scale)), h))
        cv2.putText(tile, "RENDER", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.putText(photo, "REAL", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 150, 0), 3, cv2.LINE_AA)
        rows.append(np.hstack([tile, photo]))
    if not rows:
        raise RuntimeError("无可用的 实拍/渲染 帧对")
    out_png = out or os.path.splitext(glb_path)[0] + "_compare.png"
    cv2.imwrite(out_png, np.vstack(rows))
    return out_png


def gif_cmd(scene_dir, out=None, before=None, after=None,
            max_width=384, fps=6, max_frames=48):
    """抓取帧段逐帧渲染 GIF: 每帧三联 BEFORE(stage5)|AFTER(stage55)|REAL

    帧段来自 pick_place_frames.json 的 pick/place 窗口 (前后各扩3帧),
    段内逐帧不跳帧; 总帧数超过 max_frames 时自动加大步长。返回 GIF 路径。
    """
    import cv2
    import numpy as np
    from render_glb_preview import render_camera_views
    before = before or os.path.join(scene_dir, "final_scene_stage5.glb")
    after = after or os.path.join(scene_dir, "final_scene_stage55.glb")
    with open(os.path.join(scene_dir, "pick_place_frames.json")) as f:
        pf = json.load(f)
    spans = []
    for it in pf.get("interactions", []):
        pick = it.get("pick_frame") or it.get("pick_window", [0])[0]
        place_w = it.get("place_window") or [it.get("place_frame") or 0]
        place = it.get("place_frame") or place_w[-1]
        spans.append([max(0, min(pick, place_w[0]) - 3),
                      max(place, place_w[-1]) + 3])
    if not spans:
        spans = [[0, 0]]
    spans.sort()
    merged = [spans[0]]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    stride = 1
    n_total = sum(hi - lo + 1 for lo, hi in merged)
    while n_total // stride > max_frames:
        stride += 1
    fids = [f for lo, hi in merged for f in range(lo, hi + 1, stride)]
    if fids and fids[0] > 0:
        fids = [0] + fids  # 首帧也纳入, 便于对照视频开头
    print(f"gif: frames {fids[0]}..{fids[-1]} stride={stride} n={len(fids)}")
    tb, _ = render_camera_views(before, scene_dir, fids, max_width=max_width)
    ta, _ = render_camera_views(after, scene_dir, fids, max_width=max_width)
    frames = []
    for fid, rb, ra in zip(fids, tb, ta):
        real = cv2.imread(os.path.join(scene_dir, "color", f"{fid}.jpg"))
        if real is None:
            continue
        # imageio.mimsave 吃 RGB: 渲染本来就是 RGB 不动, 实拍 imread 是 BGR 需转
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        h = rb.shape[0]
        scale = h / real.shape[0]
        real = cv2.resize(real, (max(1, int(real.shape[1] * scale)), h))
        for img, tag, col in ((rb, "BEFORE", (255, 0, 0)),
                              (ra, "AFTER", (0, 180, 0)),
                              (real, "REAL", (0, 0, 255))):
            cv2.putText(img, tag, (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, tag, (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, col, 2, cv2.LINE_AA)
        w = min(rb.shape[1], ra.shape[1], real.shape[1])
        frames.append(np.hstack([i[:, :w] for i in (rb, ra, real)]))
    if not frames:
        raise RuntimeError("无可用帧")
    out_gif = out or os.path.join(scene_dir, "stage55_before_after_real.gif")
    import imageio
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    return out_gif


def _log_move(glb_path, entry):
    log_path = os.path.join(os.path.dirname(os.path.abspath(glb_path)),
                            "stage55_moves.jsonl")
    entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), **entry}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 基础几何

def _footprint(mesh, buffer=0.0):
    """mesh 的 xy 投影凸包 (shapely Polygon); 退化返回 None"""
    from shapely.geometry import Polygon
    pts = mesh.vertices[:, :2]
    if len(pts) < 3:
        return None
    poly = Polygon(pts).convex_hull
    if poly.area <= 1e-6:
        return None
    if buffer > 0:
        poly = poly.buffer(buffer)
    return poly


def _is_structural(name):
    return name.lower().startswith(STRUCTURAL_PREFIXES)


def _bbox_volume(mesh):
    ext = mesh.extents
    return float(np.prod(ext)) if np.all(ext > 0) else float("inf")


def _move(insts, name, d):
    insts[name].apply_translation(np.asarray(d, dtype=np.float64))


def _ray_support_top(smesh, fp, o_bottom, n_grid=4, z_margin=0.3, z_tol=0.002):
    """射线投射求支撑面: 从物体底面上方 z=o_bottom+z_margin 沿 -z 打射线到 smesh,
    取底面之下 (z <= o_bottom + z_tol) 的最高命中 z; 无命中返回 None

    网格点取自 footprint 多边形内部 (n_grid x n_grid), 规避顶点稀疏问题。
    """
    import shapely
    minx, miny, maxx, maxy = fp.bounds
    xs = np.linspace(minx, maxx, n_grid + 2)[1:-1]
    ys = np.linspace(miny, maxy, n_grid + 2)[1:-1]
    pts = [(x, y) for x in xs for y in ys
           if shapely.contains_xy(fp.buffer(-0.001), x, y)]
    if not pts:
        p = fp.representative_point()
        pts = [(p.x, p.y)]
    origins = np.array([(x, y, o_bottom + z_margin) for x, y in pts])
    dirs = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    try:
        locs, _, _ = smesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    except Exception:
        return None
    if len(locs) == 0:
        return None
    valid = locs[locs[:, 2] <= o_bottom + z_tol, 2]
    if len(valid) == 0:
        return None
    return float(valid.max())


def _candidate_top(smesh, fp, o_bottom, z_tol=0.002):
    """混合法求候选支撑者顶面: 先顶点法(快), 无命中再射线投射(密, 抗顶点稀疏)"""
    import shapely
    inside = smesh.vertices[:, :2]
    mask = shapely.contains_xy(fp, inside[:, 0], inside[:, 1])
    if mask.any():
        valid = smesh.vertices[mask, 2]
        valid = valid[valid <= o_bottom + z_tol]
        if len(valid):
            return float(valid.max())
    return _ray_support_top(smesh, fp, o_bottom)


# ---------------------------------------------------------------- check_support

def check_support(insts, tol=0.005, cap=0.15, fp_buffer=0.02, min_overlap=0.10):
    """支撑检测: 对每个物体找 footprint 下方最高顶面 / 地面, 量 gap

    支撑者需有"证据": 其顶点在物体 footprint 内、且距该顶面 5mm 内的顶点数 ≥3
    (防止剪刀尖等细长物体单顶点被误判为支撑面); 无证据者取 footprint 重叠最大者。
    gap = support_z - obj_bottom_z:
        |gap| <= tol  → ok
        gap >  tol    → embedded  (底面陷入支撑面, 需上抬)
        gap < -tol    → floating  (悬空, 需下落)
    fix_dz = clamp(gap, -cap, cap); 超出 cap 标记 large_gap=True
    """
    import shapely

    floor_z = min(float(m.bounds[0][2]) for m in insts.values())
    fps = {}
    for name, mesh in insts.items():
        fps[name] = _footprint(mesh, buffer=0.0)
    fp_areas = {n: (p.area if p is not None else 0.0) for n, p in fps.items()}

    results = {}
    for name, mesh in insts.items():
        if fp_areas[name] < 0.001:  # 退化 footprint (无法定义支撑) 跳过
            results[name] = {"status": "skipped", "reason": "degenerate_footprint"}
            continue
        fp = fps[name].buffer(fp_buffer)
        o_bottom = float(mesh.bounds[0][2])
        candidates = []  # (top, sname)
        n_overlap_any = 0  # footprint 有重叠的邻居数 (含无有效命中的)
        own_min, own_max = mesh.bounds[0][:2], mesh.bounds[1][:2]
        for sname, smesh in insts.items():
            if sname == name:
                continue
            smn, smx = smesh.bounds[0][:2], smesh.bounds[1][:2]
            if (own_min > smx).any() or (smn > own_max).any():
                continue  # AABB xy 不相交, 快速跳过
            sp = fps[sname]
            if sp is None:
                continue
            if not fp.intersects(sp):
                continue
            n_overlap_any += 1
            # 混合法求顶面 (顶点 → 射线); 命中面须在底面之下 (z_tol),
            # 同水平互嵌者 (表面在底面之上) 自动被拒绝
            top = _candidate_top(smesh, fp, o_bottom)
            if top is None:
                continue
            candidates.append((top, sname))
        if candidates:
            best = max(candidates, key=lambda c: c[0])
            best_top, best_supporter = best[0], best[1]
        elif n_overlap_any > 0:
            # 有水平重叠但无有效支撑命中 (同水平互穿簇): 不做 Z 回退
            # (防止幻影 floating 把物体压穿桌体), 交给 check_glb/agent 水平解决
            results[name] = {"status": "mutual_overlap", "fix_dz": 0.0,
                             "bottom_z": round(o_bottom, 4), "n_overlap": n_overlap_any}
            continue
        else:
            best_top, best_supporter = floor_z, "floor"
        gap = best_top - o_bottom
        if abs(gap) <= tol:
            status, fix = "ok", 0.0
        elif gap > tol:
            status, fix = "embedded", min(gap, cap)
        else:
            status, fix = "floating", max(gap, -cap)
        results[name] = {
            "status": status,
            "supporter": best_supporter,
            "support_z": round(best_top, 4),
            "bottom_z": round(o_bottom, 4),
            "gap_m": round(gap, 4),
            "fix_dz": round(fix, 4),
            "large_gap": abs(gap) > cap,
        }
    return results


def snap_support(glb_path, passes=3, tol=0.005, cap=0.15):
    """应用 Z 贴合修复: root-first (底面低者先), 逐 pass 重算, 最多 passes 轮"""
    insts = load_instances(glb_path)
    applied = []
    for p in range(passes):
        rep = check_support(insts, tol=tol, cap=cap)
        moved = 0
        # root-first: bottom_z 低的先修
        # large_gap (|gap|>cap) 不动: 大偏差属语义错误 (放错支撑物/悬空),
        # 盲目 Z snap 会落到错误的支撑面, 交给 agent 做 XY 语义修正
        order = sorted(
            (n for n, r in rep.items()
             if r.get("status") in ("embedded", "floating") and not r.get("large_gap")),
            key=lambda n: rep[n]["bottom_z"])
        for name in order:
            r = rep[name]
            dz = r["fix_dz"]
            if abs(dz) > 1e-4:
                _move(insts, name, (0, 0, dz))
                applied.append({"obj": name, "dz": round(dz, 4),
                                "status": r["status"], "pass": p})
                moved += 1
        if moved == 0:
            break
    save_instances(insts, glb_path)
    _log_move(glb_path, {"cmd": "snap_support", "src": "deterministic",
                         "applied": applied})
    return applied


# ---------------------------------------------------------------- check_glb

def check_glb(insts, min_depth=0.003, margin=0.002):
    """真实三角形碰撞检测 (FCL) + 每对附 world_dxdy 修复向量

    幅度/方向不直接用 fcl depth: 重建网格为薄壳, 表面交叉处局部 depth 恒 <1mm,
    无法反映真实重叠。改用 AABB 最小平移向量 (MTV):
      - 贴放 (resting): AABB 恰好面接触, 重叠 ≈0 → 不报
      - 真穿模: 最小重叠轴的穿透量即分离所需平移
    mover 选择: pair 中 bbox 体积较小且非结构元素的一方
    """
    from trimesh.collision import CollisionManager

    mgr = CollisionManager()
    for name, mesh in insts.items():
        mgr.add_object(name, mesh)
    collides, contacts = mgr.in_collision_internal(return_data=True)
    if not collides or not contacts:
        return []

    centers = {n: m.centroid for n, m in insts.items()}
    bounds = {n: m.bounds for n, m in insts.items()}
    vols = {n: _bbox_volume(m) for n, m in insts.items()}
    pairs = {}
    for c in contacts:
        if len(c.names) != 2:
            continue
        key = tuple(sorted(c.names))
        p = pairs.setdefault(key, {"n": 0, "fcl_depth": 0.0})
        p["n"] += 1
        p["fcl_depth"] = max(p["fcl_depth"], float(c.depth))

    clashes = []
    for (a, b), d in sorted(pairs.items()):
        ba, bb = bounds[a], bounds[b]
        overlap = np.minimum(ba[1], bb[1]) - np.maximum(ba[0], bb[0])
        if (overlap <= 0).any():  # AABB 不相交 → 仅表面擦碰, 忽略
            continue
        bottom_a, bottom_b = float(ba[0][2]), float(bb[0][2])
        same_layer = abs(bottom_a - bottom_b) < 0.008
        if same_layer:
            # 同层物体禁止 z 推挤 (防止推向/穿过支撑面, stage5 教训), 只水平分离
            axes = (0, 1)
        else:
            axes = (0, 1, 2)
        axis = int(np.argmin(overlap[list(axes)]))
        axis = axes[axis]
        sep = float(overlap[axis])
        if sep < min_depth:
            continue
        a_struct, b_struct = _is_structural(a), _is_structural(b)
        if axis == 2:
            # z 推挤只允许"上方物体向上抬" (嵌入叠置), 绝不向下
            mover = a if bottom_a > bottom_b else b
            vec = np.array([0.0, 0.0, sep + margin])
        else:
            if a_struct and b_struct:
                mover = a if vols[a] <= vols[b] else b
            elif a_struct:
                mover = b
            elif b_struct:
                mover = a
            else:
                mover = a if vols[a] <= vols[b] else b
            other = b if mover == a else a
            vec = np.zeros(3)
            vec[axis] = (1.0 if centers[mover][axis] >= centers[other][axis] else -1.0) \
                * (sep + margin)
        clashes.append({
            "a": a, "b": b,
            "sep_depth_m": round(sep, 4),
            "push_axis": "xyz"[axis],
            "n_contacts": d["n"],
            "fcl_depth_m": round(d["fcl_depth"], 5),
            "mover": mover,
            "world_dxdy": [round(float(v), 4) for v in vec],
        })
    clashes.sort(key=lambda c: -c["sep_depth_m"])
    return clashes


def nudge(glb_path, rounds=4, min_depth=0.003, max_step=0.15):
    """确定性兜底推挤: 每轮按 check_glb 的 world_dxdy 移动 mover, 直到无碰撞

    出口: 零碰撞 / 轮数上限 / 总深度不再下降 (卡死)
    """
    insts = load_instances(glb_path)
    history = []
    prev_total = None
    for r in range(rounds):
        clashes = check_glb(insts, min_depth=min_depth)
        total = sum(c["sep_depth_m"] for c in clashes)
        history.append({"round": r, "n_clashes": len(clashes),
                        "total_sep_m": round(total, 4)})
        if not clashes:
            break
        if prev_total is not None and total >= prev_total - 1e-4:
            history.append({"round": r, "note": "no_improvement_stop"})
            break
        prev_total = total
        for c in clashes:
            vec = np.array(c["world_dxdy"])
            step = np.linalg.norm(vec[:2]) if np.linalg.norm(vec[:2]) > 1e-6 \
                else abs(vec[2])
            if step > max_step:
                vec = vec / np.linalg.norm(vec) * max_step
            _move(insts, c["mover"], vec)
            _log_move(glb_path, {"cmd": "nudge", "src": "deterministic",
                                 "obj": c["mover"], "d": c["world_dxdy"],
                                 "round": r, "sep_depth_m": c["sep_depth_m"]})
    save_instances(insts, glb_path)
    final = check_glb(load_instances(glb_path), min_depth=min_depth)
    return {"history": history, "final_clashes": final}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Stage 5.5 QC CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_glb(p):
        p.add_argument("--glb", required=True, help="工作 GLB 路径")

    p = sub.add_parser("inspect"); add_glb(p)
    p = sub.add_parser("check_support"); add_glb(p)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--cap", type=float, default=0.15)
    p = sub.add_parser("snap_support"); add_glb(p)
    p.add_argument("--passes", type=int, default=3)
    p = sub.add_parser("check_glb"); add_glb(p)
    p.add_argument("--min-depth", type=float, default=0.001)
    p = sub.add_parser("nudge"); add_glb(p)
    p.add_argument("--rounds", type=int, default=4)
    p = sub.add_parser("set_xy"); add_glb(p)
    p.add_argument("--obj", required=True)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p = sub.add_parser("set_z"); add_glb(p)
    p.add_argument("--obj", required=True)
    p.add_argument("--dz", type=float, required=True)
    p = sub.add_parser("render"); add_glb(p)
    p.add_argument("--out", default=None, help="输出 PNG 路径 (默认覆盖到 --out)")
    p = sub.add_parser("compare"); add_glb(p)
    p.add_argument("--out", default=None, help="输出 PNG 路径")
    p = sub.add_parser("gif")
    p.add_argument("--scene-dir", required=True, help="场景输出目录")
    p.add_argument("--out", default=None, help="输出 GIF 路径")
    p.add_argument("--before", default=None, help="修改前 GLB (默认 final_scene_stage5.glb)")
    p.add_argument("--after", default=None, help="修改后 GLB (默认 final_scene_stage55.glb)")
    p.add_argument("--max-width", type=int, default=384)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=48)

    args = ap.parse_args()

    if args.cmd == "inspect":
        insts = load_instances(args.glb)
        for name, m in sorted(insts.items()):
            b = m.bounds
            print(f"{name:<20} bbox_min=({b[0][0]:+.3f},{b[0][1]:+.3f},{b[0][2]:+.3f}) "
                  f"bbox_max=({b[1][0]:+.3f},{b[1][1]:+.3f},{b[1][2]:+.3f}) "
                  f"vol={_bbox_volume(m):.4f}")
        print(f"total: {len(insts)} instances")

    elif args.cmd == "check_support":
        rep = check_support(load_instances(args.glb), tol=args.tol, cap=args.cap)
        print(json.dumps(rep, ensure_ascii=False, indent=2))

    elif args.cmd == "snap_support":
        applied = snap_support(args.glb, passes=args.passes)
        print(json.dumps({"applied": applied}, ensure_ascii=False, indent=2))

    elif args.cmd == "check_glb":
        clashes = check_glb(load_instances(args.glb), min_depth=args.min_depth)
        print(json.dumps({"n_clashes": len(clashes), "clashes": clashes},
                         ensure_ascii=False, indent=2))

    elif args.cmd == "nudge":
        res = nudge(args.glb, rounds=args.rounds)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd in ("set_xy", "set_z"):
        insts = load_instances(args.glb)
        if args.obj not in insts:
            print(f"ERROR: unknown object '{args.obj}'. 已知: {sorted(insts)}")
            sys.exit(2)
        d = {"set_xy": (getattr(args, "dx", 0.0), getattr(args, "dy", 0.0), 0.0),
             "set_z": (0.0, 0.0, getattr(args, "dz", 0.0))}[args.cmd]
        before = insts[args.obj].bounds[0].copy()
        _move(insts, args.obj, d)
        after = insts[args.obj].bounds[0]
        save_instances(insts, args.glb)
        _log_move(args.glb, {"cmd": args.cmd, "src": "agent", "obj": args.obj,
                             "d": list(d)})
        print(f"OK {args.obj} moved by ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) "
              f"bottom_z {before[2]:.3f} -> {after[2]:.3f}")

    elif args.cmd == "render":
        from render_glb_preview import render_preview_png
        out = render_preview_png(args.glb)
        if out is None:
            print("ERROR: render failed")
            sys.exit(1)
        if args.out and os.path.abspath(args.out) != os.path.abspath(out):
            shutil.copy(out, args.out)
            out = args.out
        print(out)

    elif args.cmd == "compare":
        out = compare(args.glb, out=args.out)
        print(out)

    elif args.cmd == "gif":
        out = gif_cmd(args.scene_dir, out=args.out, before=args.before,
                      after=args.after, max_width=args.max_width,
                      fps=args.fps, max_frames=args.max_frames)
        print(out)


if __name__ == "__main__":
    main()

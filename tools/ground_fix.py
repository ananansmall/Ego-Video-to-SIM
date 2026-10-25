"""重力可见化 + 退化桌面修复 (Stage 5 出口标准步骤)。

1. 桌腿合成: "table" 类实例若是薄板 (高<0.15m 且宽>0.5m) —— 手持俯拍视频
   里桌腿几乎不进画面, 点云只有桌面 (实测 wxD: is_planar_pointcloud 判平面
   后只剩 5cm 薄板贴地) —— 按 0.75m 类别先验合成 4 条腿并抬升桌面。
2. 支撑物跟随: relations 里 "supported by table_i" 的实例随桌面抬升 Δh,
   保持贴合。
3. 地板基准面: 全实例联合包围盒 + 2.4m 边距的薄板, 让重力参考可见
   (此前最终 GLB 只有物体实例, 无地板, 场景像漂在白空间)。

用法 (mainv3 Stage 5 出口自动调用):
    from tools.ground_fix import add_ground_and_legs
    all_instances, ground_report = add_ground_and_legs(all_instances, refined_relations)
"""
import numpy as np

LEG_HEIGHT = 0.75       # 桌类先验高度
LEG_THICK = 0.07
LEG_INSET = 0.07
SLAB_MAX_H = 0.15       # 薄板判定阈值
SLAB_MIN_W = 0.5
FLOOR_MARGIN = 2.4


def _wb(info):
    import trimesh
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], float))
    return m.bounds


def _world_mesh(info):
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], float))
    return m


def _synth_legs(info):
    """薄板桌面 → 加 4 腿并抬到 LEG_HEIGHT。返回 (新 info, 旧顶面, 新顶面) 或 None。"""
    import trimesh
    m = _world_mesh(info)
    b = m.bounds
    w, d, h = float(b[1][0] - b[0][0]), float(b[1][1] - b[0][1]), float(b[1][2] - b[0][2])
    if h >= SLAB_MAX_H or min(w, d) < SLAB_MIN_W:
        return None
    top_old = float(b[1][2])
    new_top = LEG_HEIGHT
    cx, cy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
    slab = m.copy()
    slab.apply_translation([0, 0, new_top - top_old])       # 抬升桌面
    parts = [slab]
    hx = w / 2 - LEG_INSET - LEG_THICK / 2
    hy = d / 2 - LEG_INSET - LEG_THICK / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = trimesh.creation.box(extents=[LEG_THICK, LEG_THICK, new_top - h])
            leg.apply_translation([cx + sx * hx, cy + sy * hy, (new_top - h) / 2])
            parts.append(leg)
    merged = trimesh.util.concatenate(parts)
    return {"original_mesh": merged, "T": np.eye(4)}, top_old, new_top


def add_ground_and_legs(all_instances, refined_relations, verbose=True):
    """Stage 5 出口调用。返回 (all_instances, report)。原地修改。"""
    report = {"legs": [], "reseated": [], "floor": None}

    # 1) 薄板桌面加腿
    raised = {}
    for cat, lst in list(all_instances.items()):
        if cat.lower() != "table":
            continue
        for i, info in enumerate(lst):
            name = f"{cat}_{i}"
            r = _synth_legs(info)
            if r is None:
                continue
            new_info, top_old, top_new = r
            lst[i] = new_info
            raised[name] = top_new - top_old
            b = _wb(new_info)
            report["legs"].append({"instance": name, "old_top": round(top_old, 3),
                                   "new_top": round(top_new, 3),
                                   "foot": [round(float(b[1][0]-b[0][0]), 2),
                                            round(float(b[1][1]-b[0][1]), 2)]})
            if verbose:
                print(f"   🦵 {name}: 薄板桌面 (高{top_old:.3f}m) 合成 4 桌腿 → 顶面 {top_new:.3f}m",
                      flush=True)

    # 2) 支撑物跟随抬升
    for name, rel in (refined_relations or {}).items():
        if not rel.startswith("supported by "):
            continue
        sup = rel[len("supported by "):].strip()
        if sup not in raised:
            continue
        dh = raised[sup]
        cat, idx = name.rsplit("_", 1)
        if cat in all_instances and int(idx) < len(all_instances[cat]):
            info = all_instances[cat][int(idx)]
            info["T"] = np.asarray(info["T"], float).copy()
            info["T"][2, 3] += dh
            report["reseated"].append({"instance": name, "raised_by": round(dh, 3)})
            if verbose:
                print(f"   ⬆️ {name}: 随 {sup} 抬升 {dh:.3f}m", flush=True)

    # 2.5) 沉地兜底: 底面 < -3cm 的实例无论什么关系都抬回地面 ——
    #      场景里不存在"地下"这种合法状态 (实测 151: 11 个碗沉地 0.04~0.64m,
    #      关系是 other objects, 落地补正只认 floor 关系所以没人管)。
    for cat, lst in all_instances.items():
        if cat.lower() == "floor":
            continue
        for i, info in enumerate(lst):
            b = _wb(info)
            if float(b[0][2]) < -0.03:
                info["T"] = np.asarray(info["T"], float).copy()
                info["T"][2, 3] -= float(b[0][2]) + 0.002
                report.setdefault("rescued_below_floor", []).append(
                    {"instance": f"{cat}_{i}", "was": round(float(b[0][2]), 3)})
                if verbose:
                    print(f"   ⛑️ {cat}_{i}: 沉地 {b[0][2]:.3f}m → 抬回地面", flush=True)

    # 3) 地板基准面
    bs = [_wb(x) for v in all_instances.values() for x in v]
    if bs:
        ux0 = min(b[0][0] for b in bs); ux1 = max(b[1][0] for b in bs)
        uy0 = min(b[0][1] for b in bs); uy1 = max(b[1][1] for b in bs)
        cx, cy = (ux0 + ux1) / 2, (uy0 + uy1) / 2
        sx = max(ux1 - ux0 + FLOOR_MARGIN, 3.0)
        sy = max(uy1 - uy0 + FLOOR_MARGIN, 3.0)
        import trimesh
        floor = trimesh.creation.box(extents=[sx, sy, 0.02])
        floor.apply_translation([cx, cy, -0.009])
        all_instances.setdefault("floor", []).append({"original_mesh": floor, "T": np.eye(4)})
        report["floor"] = {"size": [round(sx, 1), round(sy, 1)],
                           "center": [round(float(cx), 2), round(float(cy), 2)]}
        if verbose:
            print(f"   ⬜ 地板基准面: {sx:.1f}x{sy:.1f}m @ ({cx:.2f},{cy:.2f})", flush=True)
    return all_instances, report


def verify_scene_ground(all_instances, refined_relations, output_path):
    """出口自检: 沉地/悬空/倒置 计数, 落盘 ground_report.json。"""
    report = {"n_below_floor": 0, "n_floating": 0, "n_inverted": 0}
    for cat, lst in all_instances.items():
        for i, info in enumerate(lst):
            m = _world_mesh(info)
            b = m.bounds
            if float(b[0][2]) < -0.03:
                report["n_below_floor"] += 1
            rel = (refined_relations or {}).get(f"{cat}_{i}", "")
            if rel == "supported by floor" and float(b[0][2]) > 0.05:
                report["n_floating"] += 1
            up = float((np.asarray(info["T"])[:3, :3] @ np.array([0, 1, 0.]))[2])
            if up < -0.3:
                report["n_inverted"] += 1
    import json, os
    if output_path:
        with open(os.path.join(output_path, "ground_report.json"), "w") as f:
            json.dump(report, f, indent=1)
    return report

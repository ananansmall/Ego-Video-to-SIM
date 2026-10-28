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

verbose_flag = [True]   # 模块级 verbose 开关 (供 _scale_prior_fallback 打印)

LEG_HEIGHT = 0.75       # 桌类先验高度
LEG_THICK = 0.07
LEG_INSET = 0.07
SLAB_MAX_H = 0.15       # 薄板判定阈值
SLAB_MIN_W = 0.5
FLOOR_MARGIN = 2.4

# 小物体类别尺度先验 (最大边, 米)。仅当实例尺度偏离先验 >2.5 倍或 <0.4 倍时
# 才回退 —— 极端偏差才动 (rescale guard 拒绝后 k=1 原样保留, 实测 scissors
# 0.09m / pillow 0.16m), 正常尺度绝不碰。
SCALE_PRIOR_EXIT = {
    "plier": 0.22, "screwdriver": 0.22, "scissors": 0.20, "tray": 0.40,
    "pillow": 0.55, "knife": 0.25, "fork": 0.20, "spoon": 0.18,
    "cup": 0.12, "mug": 0.12, "bowl": 0.18, "plate": 0.26, "laptop": 0.40,
    "phone": 0.17, "cellphone": 0.17, "remote": 0.20, "remote_control": 0.20,
    "book": 0.25, "bottle": 0.25, "hammer": 0.33, "watch": 0.05,
    "glasses": 0.15, "earbud_case": 0.06, "mouse": 0.12,
    "keyboard": 0.45, "can": 0.12, "wallet": 0.12, "tape": 0.10, "marker": 0.15,
}
SCALE_DEV_HI, SCALE_DEV_LO = 2.5, 0.4


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


def _scale_prior_fallback(info, category, report):
    """尺度先验回退: 小物体尺度偏离先验 >2.5x 或 <0.4x 时缩放到先验。

    只覆盖 SCALE_PRIOR_EXIT 里的"尺寸稳定"小物体类; 家具类(桌/椅/床)尺寸
    变化大, 不做先验回退。"""
    prior = SCALE_PRIOR_EXIT.get(category.lower().replace(" ", "_"))
    if not prior:
        return False
    m = _world_mesh(info)
    mx = float(np.max(m.bounds[1] - m.bounds[0]))
    if mx <= 1e-6:
        return False
    if not (mx > SCALE_DEV_HI * prior or mx < SCALE_DEV_LO * prior):
        return False
    s = prior / mx
    T = np.asarray(info["T"], float).copy()
    T[:3, :3] *= s
    T[:3, 3] *= s                      # 缩放后仍以世界原点为基准, 再落地
    info["T"] = T
    b = _wb(info)
    info["T"][2, 3] -= float(b[0][2])  # 落地
    report.setdefault("scale_fallback", []).append(
        {"category": category, "was_max": round(mx, 3), "prior": prior, "scale": round(s, 3)})
    if verbose_flag[0]:
        print(f"   📏 {category}: 尺度 {mx:.3f}m 偏离先验 {prior}m → 回退 (x{s:.2f})", flush=True)
    return True


def _sup_capacity_fix(all_instances, refined_relations, report, verbose):
    """支撑容量检查: 被支撑物放不下时扩大支撑体足迹并网格化重排。"""
    import trimesh
    PACK = 0.55          # 桌面有效摆放系数 (圆形物体在方形面的排列效率)
    MAX_SCALE = 2.5
    # supporter name -> [被支撑实例名]
    sup_map = {}
    for name, rel in (refined_relations or {}).items():
        if rel.startswith("supported by "):
            sup = rel[len("supported by "):].strip()
            if sup and sup not in ("floor", "ground", "wall", "other objects") and "_" in sup:
                sup_map.setdefault(sup, []).append(name)

    def _get(name):
        cat, idx = name.rsplit("_", 1)
        lst = all_instances.get(cat, [])
        return (cat, int(idx), lst[int(idx)]) if int(idx) < len(lst) else None

    for sup, children in sup_map.items():
        got = _get(sup)
        if got is None:
            continue
        scat, sidx, sinfo = got
        if scat.lower() == "floor":
            continue
        sb = _wb(sinfo)
        s_area = float((sb[1][0]-sb[0][0]) * (sb[1][1]-sb[0][1]))
        need = 0.0
        kids = []
        for cname in children:
            g = _get(cname)
            if g is None:
                continue
            _, _, cinfo = g
            cb = _wb(cinfo)
            ca = float((cb[1][0]-cb[0][0]) * (cb[1][1]-cb[0][1]))
            need += ca
            kids.append((cname, cinfo, cb, ca))
        if not kids or s_area <= 1e-6:
            continue
        if need <= s_area * PACK:
            continue                                    # 放得下, 不动
        s_scale = min(float(np.sqrt(need / (s_area * PACK))), MAX_SCALE)
        if s_scale <= 1.01:
            continue
        # 扩大支撑体 x/y 足迹 (高度不变), 直接变换世界网格, T=identity
        sm = _world_mesh(sinfo)
        cx, cy = (sb[0][0]+sb[1][0])/2, (sb[0][1]+sb[1][1])/2
        sm.vertices[:, 0] = cx + s_scale * (sm.vertices[:, 0] - cx)
        sm.vertices[:, 1] = cy + s_scale * (sm.vertices[:, 1] - cy)
        sinfo["original_mesh"] = sm
        sinfo["T"] = np.eye(4)
        sb2 = _wb(sinfo)
        report.setdefault("capacity_enlarged", []).append(
            {"supporter": sup, "children": len(kids), "need_area": round(need, 3),
             "had_area": round(s_area, 3), "scale_xy": round(s_scale, 2)})
        if verbose:
            print(f"   📐 {sup}: 被支撑物需 {need:.3f}m² > 可用 {s_area*PACK:.3f}m² "
                  f"→ 足迹扩 x{s_scale:.2f}", flush=True)

        # 被支撑物网格化重排到新桌面上
        sb2 = _wb(sinfo)
        top = float(sb2[1][2])
        usable_x, usable_y = (sb2[1][0]-sb2[0][0]) - 0.12, (sb2[1][1]-sb2[0][1]) - 0.12
        cols = int(np.ceil(np.sqrt(len(kids))))
        for k, (cname, cinfo, cb, ca) in enumerate(kids):
            row, col = divmod(k, cols)
            tx = sb2[0][0] + 0.06 + usable_x * (col + 0.5) / cols
            ty = sb2[0][1] + 0.06 + usable_y * (row + 0.5) / max(1, int(np.ceil(len(kids)/cols)))
            bcx, bcy = (cb[0][0]+cb[1][0])/2, (cb[0][1]+cb[1][1])/2
            T = np.asarray(cinfo["T"], float).copy()
            T[0, 3] += tx - bcx; T[1, 3] += ty - bcy
            T[2, 3] += (top + 0.004) - float(cb[0][2])
            cinfo["T"] = T
        if verbose:
            print(f"   🔁 {len(kids)} 个被支撑物已在 {sup} 新桌面上重排", flush=True)


def _flimsy_shell_filter(all_instances, refined_relations, report, verbose):
    """薄壳碎片过滤: 同类别实体宿主附近的薄片 → 丢弃 (编号重排 + 关系同步)。

    判据: min(ext)/max(ext) < 0.08 且 max(ext) > 0.25m (大面积无厚度),
    且存在同类实体实例 (min/max > 0.2) 的 AABB 扩 0.4m 包含薄片中心。
    实测 112: 5 个 cabinet = 2 真柜体 + 3 平面薄片 (SAM3 把柜面单独成实例,
    薄片互相平行不重叠, 重叠去重天然抓不到)。"""
    infos = {}
    for cat, lst in all_instances.items():
        for i, x in enumerate(lst):
            b = _wb(x); e = b[1] - b[0]
            infos[f"{cat}_{i}"] = (cat, i, x, b, float(e.min()/max(e.max(), 1e-9)), float(e.max()))
    flimsy = {n: v for n, v in infos.items() if v[4] < 0.08 and v[5] > 0.25}
    if not flimsy:
        return
    drops = []
    for name, (cat, i, x, b, t, mx) in flimsy.items():
        c = (b[0] + b[1]) / 2
        for n2, (c2, i2, x2, b2, t2, mx2) in infos.items():
            if n2 == name or c2 != cat or t2 >= 0.2:
                continue
            if all((b2[0][k] - 0.4) <= c[k] <= (b2[1][k] + 0.4) for k in range(3)):
                drops.append((name, n2))
                break
    if not drops:
        return
    drop_names = {d[0] for d in drops}
    name_map = {}
    new_inst = {}
    for cat, lst in all_instances.items():
        kept = []
        for i, x in enumerate(lst):
            old = f"{cat}_{i}"
            if old in drop_names:
                continue
            name_map[old] = f"{cat}_{len(kept)}"
            kept.append(x)
        new_inst[cat] = kept
    all_instances.clear()
    all_instances.update(new_inst)
    for old in list(refined_relations.keys()):
        if old in drop_names:
            del refined_relations[old]
        elif old in name_map and name_map[old] != old:
            refined_relations[name_map[old]] = refined_relations.pop(old)
    report.setdefault("flimsy_shells_dropped", []).extend(
        [{"instance": d[0], "host": d[1]} for d in drops])
    if verbose:
        print(f"   🗑️ 薄壳碎片过滤: 丢弃 {len(drops)} 个 {sorted(d[0] for d in drops)}", flush=True)


def add_ground_and_legs(all_instances, refined_relations, verbose=True):
    """Stage 5 出口调用。返回 (all_instances, report)。原地修改。"""
    report = {"legs": [], "reseated": [], "floor": None}
    verbose_flag[0] = verbose

    # 0) 尺度先验回退 (在抬升/贴地之前修正尺度, 后续所有步骤用正确尺寸)
    for cat, lst in list(all_instances.items()):
        for i, info in enumerate(lst):
            _scale_prior_fallback(info, cat, report)

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

    # 2.6) 支撑容量检查 + 扩容重排 (用户洞察: 桌面太小是拥挤问题的根因 ——
    #      被支撑物的总足迹超过支撑面时, 物体必然被挤出/悬空/沉地)。
    #      对每个有具名被支撑物的支撑体: 所需面积 > 可用面积 × 0.55 时,
    #      按 sqrt 比例扩大支撑体 x/y 足迹 (高度不变), 然后把被支撑物
    #      在新桌面上网格化重排。上限 2.5x 防荒谬。
    _sup_capacity_fix(all_instances, refined_relations, report, verbose)

    # 2.65) 薄壳碎片过滤: 无厚度的大面积实例 (min/max<0.08 且 >0.25m) 且
    #       同类别实体实例在其附近时, 是"物体表面被单独分割"的皮 (SAM3 把
    #       柜面切成独立实例, 实测 112: 2 真柜体 + 3 平面薄片) —— 丢弃。
    #       没有实体宿主的薄物体 (笔记本/盘子/砧板) 合法, 保留。
    _flimsy_shell_filter(all_instances, refined_relations, report, verbose)

    # 2.7) 沉地兜底: 底面 < -3cm 的实例无论什么关系都抬回地面 ——
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

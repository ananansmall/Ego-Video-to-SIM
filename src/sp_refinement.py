import os
import argparse
import numpy as np
import trimesh
import json


def _get_wall_alignment_target(forward_vector, angle_tolerance=30.0):
    """
    Map object's forward direction to nearest horizontal cardinal axis.
    Returns (align_vector, target_axis) or (None, None) if outside tolerance.
    """
    yaw_deg = np.degrees(np.arctan2(forward_vector[1], forward_vector[0]))
    yaw_candidates = [
        (0.0, np.array([1.0, 0.0, 0.0]), 'x'),
        (90.0, np.array([0.0, 1.0, 0.0]), 'y'),
        (-90.0, np.array([0.0, -1.0, 0.0]), 'y'),
        (180.0, np.array([-1.0, 0.0, 0.0]), 'x'),
        (-180.0, np.array([-1.0, 0.0, 0.0]), 'x'),
    ]

    best = None
    best_dist = float("inf")
    for target_yaw, align_vec, axis in yaw_candidates:
        dist = abs(((yaw_deg - target_yaw + 180.0) % 360.0) - 180.0)
        if dist < best_dist:
            best_dist = dist
            best = (align_vec, axis)

    if best is None or best_dist > angle_tolerance:
        return None, None
    return best


def _select_closest_wall(walls_info, axis, center, span_margin=0.3):
    """
    Select nearest wall on a given axis. Prefer walls whose span covers object center.
    Returns (wall_info, distance) or (None, inf).
    """
    candidates = [w for w in walls_info if w.get('axis') == axis]
    if len(candidates) == 0:
        return None, float("inf")

    other_axis_idx = 1 if axis == 'x' else 0
    axis_idx = 0 if axis == 'x' else 1
    center_other = center[other_axis_idx]

    in_span = []
    for wall in candidates:
        span_start, span_end = wall.get('span', (-float("inf"), float("inf")))
        lo = min(span_start, span_end) - span_margin
        hi = max(span_start, span_end) + span_margin
        if lo <= center_other <= hi:
            in_span.append(wall)

    lookup = in_span if len(in_span) > 0 else candidates
    best_wall = min(lookup, key=lambda w: abs(w['position'] - center[axis_idx]))
    best_dist = abs(best_wall['position'] - center[axis_idx])
    return best_wall, best_dist

def ground_floating_objects(all_instances, refined_relations, verbose=True,
                            float_eps=0.02, post_tol=0.01, float_z_max=0.5):
    '''
    把关系为 "supported by floor" 但悬空的实例压到地面 (z=0)。

    为什么需要单独一步: 基础精修读的是 Stage1 的类别级关系, 而关系往往到 5.1
    才被推翻 (实测 egodex_4 cloth_0: Stage1 判 attached to wall → 基础精修走挂墙
    分支不落地; 5.1 改判 supported by floor 时位姿已定, 且 5.2 的穿模修复只做
    "分开"不做 "落地", 反而把悬空推得更高)。

    与 refine_supported_by_floor_object 的区别: 不扶正物体朝向。
    那个函数会把物体 y 轴对齐重力方向 (为落地家具设计), 对手持小物体会产生
    109~119° 的意外旋转 (实测 bracelet/stapler/screwdriver 等 5 个实例)。
    这里只改 T[2, 3], 尺寸/水平位置/形状均不变。

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 T
        refined_relations: {instance_name: relation_string} 实例级关系 (5.1 输出)
        float_eps: min_z 超过该值才算悬空 (避开浮点噪声与贴地家具的微小抬起)
        post_tol: 落地后 min_z 的允许残差
    Returns:
        (all_instances, n_grounded)
    '''
    n_grounded = 0
    for category, category_instances in all_instances.items():
        for idx, instance_info in enumerate(category_instances):
            name = f"{category}_{idx}"
            relation = refined_relations.get(name, "") or refined_relations.get(category, "")
            if "floor" not in relation.lower():
                continue

            T = np.asarray(instance_info["T"], dtype=float).copy()
            mesh = instance_info["original_mesh"].copy()
            mesh.apply_transform(T)
            z_min = float(mesh.bounds[0, 2])
            if z_min <= float_eps:
                continue
            # 与 refine_supported_by_floor_object 的_E_FLOOR_Z_MAX 一致:
            # 悬空超过 float_z_max 的不是落地偏差, 而是坐标系未对齐 (见 E 项修复)。
            # 不加此上界会把上游对齐错误强行压平, 抵消 E 的效果。
            if z_min > float_z_max:
                if verbose:
                    print(f"      [ground] {name}: 跳过 (z_min={z_min:+.4f}m > "
                          f"{float_z_max}m, 疑似坐标系未对齐而非悬空)", flush=True)
                continue

            # 只改z 平移量: 不动旋转、不动 xy、不动 mesh 顶点。
            # 先在副本上试算+ 校验, 通过后才写回, 避免半途失败留下脏 T。
            T_new = T.copy()
            T_new[2, 3] -= z_min
            check = instance_info["original_mesh"].copy()
            check.apply_transform(T_new)
            z_after = float(check.bounds[0, 2])
            if abs(z_after) > post_tol:
                # 理论上不可达 (纯平移必然精确), 万一失败则不写回
                if verbose:
                    print(f"      [ground] {name}: 落地校验失败 (z={z_after:+.4f}m), 保持原位",
                          flush=True)
                continue

            instance_info["T"] = T_new
            n_grounded += 1
            if verbose:
                print(f"      [ground] {name}: 悬空 z={z_min:+.4f}m → 落地 z={z_after:+.4f}m",
                      flush=True)
    return all_instances, n_grounded


def _world_bounds(instance_info):
    """物体在当前 T 下的世界坐标 AABB, shape (2, 3)。"""
    T = np.asarray(instance_info["T"], dtype=float)
    mesh = instance_info["original_mesh"].copy()
    mesh.apply_transform(T)
    return np.asarray(mesh.bounds, dtype=float)


def _footprint_area(b):
    """AABB 在 XY 平面的投影面积 (下限保护, 避免除零)。"""
    return max(1e-9, float(b[1, 0] - b[0, 0]) * float(b[1, 1] - b[0, 1]))


def _xy_overlap_ratio(b_obj, b_sup):
    """物体底面投影落在支撑物投影内的比例 (0~1)。"""
    ox = max(0.0, min(b_obj[1, 0], b_sup[1, 0]) - max(b_obj[0, 0], b_sup[0, 0]))
    oy = max(0.0, min(b_obj[1, 1], b_sup[1, 1]) - max(b_obj[0, 1], b_sup[0, 1]))
    return (ox * oy) / _footprint_area(b_obj)


def _gap_status(gap, sup_height, gap_tol=0.03):
    """gap → 摆放状态。

    陷入深度超过支撑物自身高度时判为 severe_penetration 而不是 sunken ——
    "装在容器里"不可能比容器还深, 这是坐标系/尺度错 (实测 egodex_15 整场景
    gap≈-1.1m), 不该由摆放补正去修, 修了只会把物体拖到同样错误的位置。
    """
    if abs(gap) <= gap_tol:
        return "ok"
    if gap > 0:
        return "floating"
    if -gap > max(1e-9, float(sup_height)):
        return "severe_penetration"
    return "sunken"


def _instance_index(all_instances):
    """{实例名: (category, idx)}"""
    index = {}
    for category, lst in all_instances.items():
        for i in range(len(lst or [])):
            index[f"{category}_{i}"] = (category, i)
    return index


def _parse_supporter(relation):
    """解析关系串 → (kind, supporter_name), kind ∈ floor/wall/unknown/named/None。"""
    r = (relation or "").strip()
    if not r.startswith("supported by "):
        return None, None
    sup = r[len("supported by "):].strip()
    if sup == "floor":
        return "floor", None
    if sup == "wall":
        return "wall", None
    if sup == "other objects":
        return "unknown", None
    return "named", sup


def _infer_supporter(name, b_obj, bounds, min_overlap=0.30, max_drop=0.60):
    """关系为 'supported by other objects' 且没解析出具体支撑物时的几何兜底。

    挑一个"顶面在物体底面之下、水平重叠够大、且顶面最高"的物体当支撑物。
    挑不到就返回 None (保持原位, 不乱猜)。

    Args:
        max_drop: 顶面低于物体底面超过该值就不算支撑物 (防止贴到远处的地面物体)
    """
    best, best_top = None, -float("inf")
    obj_bottom = float(b_obj[0, 2])
    for other, b in bounds.items():
        if other == name:
            continue
        top = float(b[1, 2])
        if top > obj_bottom + 1e-6:
            continue
        if obj_bottom - top > max_drop:
            continue
        if _xy_overlap_ratio(b_obj, b) < min_overlap:
            continue
        if top > best_top:
            best, best_top = other, top
    return best


def reseat_on_top_objects(all_instances, refined_relations, placement_strategies=None,
                          verbose=True, gap_tol=0.03, max_iter=4,
                          min_overlap=0.10, infer_min_overlap=0.30,
                          inside_ratio=0.5):
    '''
    把 "supported by X" (X 非 floor/wall) 的实例重新贴到支撑物 X 的顶面。

    为什么需要: 5.2 的 on_top 精修本来贴对了, 但随后的穿模修复会把物体推开 ——
    on_top 贴到顶面时接触面间隙为 0, 穿模检测把这种零间隙接触判成重叠并沿z 分离;
    物体掉到地面后底面共面, 又被判重叠, 形成自维持循环。
    实测 egodex_33: 23 次穿模修复 100% 是 z 轴分离且 100% 涉及 on_top 物体,
    toy_1 被推 9 次 x 0.0056m = 0.050m, 从 z=0.065 掉到地面 (plate_0 顶面仅 0.036m);
    全库 104 个 on_top/inside 关系只有 3 个贴合, 坏 101 个。

    这里只改 z 平移 (gap 修正量), 不动旋转/水平位置/尺寸。

    相对初版的三个修正:
    1. 按支撑链拓扑序处理 + 多轮迭代至收敛。初版是单次遍历 dict 顺序, 遇到
       A→B→C 的链时, 若 A 先处理会读到 B 的旧顶面, B 之后被抬到 C 上就会让
       A 悬空。实测 15 条多层链 (toy_0→lego_0→plate_0 等)。
    2. 环检测。实测 egodex_15 存在 sheet_0 ↔ sheet_1 二元环, 互相往上贴会无限
       发散; 这里断开投影面积最大者的出边, 让它当链的固定根。
    3. 'supported by other objects' 不再直接躺平。实测 52 个场景有 21 个这类
       物体, 之前两道补正都跳过它们, 位姿完全等于 VGGT 原始预测; 现在用
       _infer_supporter 做几何兜底挑支撑物。

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 T
        refined_relations: {instance_name: relation_string} 实例级关系 (5.1 输出)
        placement_strategies: {实例名: on_top/inside/against_side/...} (5.2 落盘)。
            有: 只重贴明确标为 on_top 的 (inside 的物体本就该在支撑物内部)。
            None: 退化为几何判据 —— 陷入深度超过自身半高的判为 inside 不动,
                 避免把碗里的面包拽到碗沿上。
        gap_tol: |gap| 在该范围内视为已贴合
        max_iter: 最大迭代轮数 (每轮都按拓扑序重算 AABB)
        min_overlap: 水平重叠低于该比例认为支撑物判错, 不硬拽
        infer_min_overlap: 几何推断支撑物时的重叠门槛 (比 min_overlap 严)
        inside_ratio: 无策略文件时, 陷入深度 / 物体高度 超过该值判为 inside
    Returns:
        (all_instances, n_reseated)
    '''
    index = _instance_index(all_instances)
    rel_of = {}
    for name, (category, _) in index.items():
        rel_of[name] = str(refined_relations.get(name, "")
                           or refined_relations.get(category, "") or "")

    sup_map, unknown = {}, set()
    for name in index:
        kind, sup = _parse_supporter(rel_of[name])
        if kind == "named" and sup in index and sup != name:
            sup_map[name] = sup
        elif kind == "unknown":
            unknown.add(name)

    def _bounds_now():
        return {n: _world_bounds(all_instances[c][i]) for n, (c, i) in index.items()}

    bounds0 = _bounds_now()
    foot = {n: _footprint_area(b) for n, b in bounds0.items()}

    # 环: 断开投影面积最大者的出边, 让它成为固定根
    severed = set()
    for start in list(sup_map):
        seen, cur = [], start
        while cur is not None and cur not in seen:
            seen.append(cur)
            cur = sup_map.get(cur)
        if cur is not None:
            cycle = seen[seen.index(cur):]
            root = max(cycle, key=lambda x: foot[x])
            severed.add(root)
            if verbose:
                print(f"      [reseat] ⚠️ 支撑环 {cycle} → 断开 {root} 的出边 (面积最大, 作根)",
                      flush=True)

    def _sup(n):
        return None if n in severed else sup_map.get(n)

    order, state = [], {}

    def _visit(n):
        if state.get(n) == 2:
            return
        state[n] = 1
        s = _sup(n)
        if s is not None:
            _visit(s)
        state[n] = 2
        order.append(n)

    for n in sorted(index):
        _visit(n)

    n_reseated = 0
    for _ in range(max_iter):
        moved = 0
        bounds = _bounds_now()
        for name in order:
            category, idx = index[name]
            instance_info = all_instances[category][idx]
            b = bounds[name]

            sup = _sup(name)
            inferred = False
            if sup is None:
                if name not in unknown:
                    continue
                sup = _infer_supporter(name, b, bounds, min_overlap=infer_min_overlap)
                if sup is None:
                    continue
                inferred = True

            sb = bounds[sup]
            overlap = _xy_overlap_ratio(b, sb)
            if overlap < min_overlap:
                if verbose:
                    print(f"      [reseat] {name}: 与支撑物 {sup} 水平重叠仅 "
                          f"{overlap*100:.0f}% (<{min_overlap*100:.0f}%), 判定支撑物有误, 跳过",
                          flush=True)
                continue

            if placement_strategies is not None:
                strategy = placement_strategies.get(name)
                if strategy is None:
                    strategy = placement_strategies.get(category.lower().strip())
                if strategy is not None and strategy != "on_top":
                    continue
            else:
                # 无策略文件: 用几何判据区分 "inside" 与 "严重穿透(坐标系数错)"
                h = max(1e-9, float(b[1, 2] - b[0, 2]))
                sup_h = max(1e-9, float(sb[1, 2]) - float(sb[0, 2]))
                sink = float(sb[1, 2]) - float(b[0, 2])
                if sink > sup_h:
                    # 陷入深度超过支撑物自身高度 → 不可能是"装在里面", 是坐标系/
                    # 尺度错了。实测 egodex_15 整场景 gap≈-1.1m (沉到地下 1 米多),
                    # 这类不该由 reseat 修 —— 修了只会把一堆物体拖到同样错误的位置。
                    if verbose:
                        print(f"      [reseat] {name}: ⚠️ 陷入 {sup} {sink:.4f}m > 支撑物高 "
                              f"{sup_h:.4f}m, 疑似坐标系未对齐 (非摆放错误), 跳过", flush=True)
                    continue
                if sink > inside_ratio * h:
                    # 装在容器里的深度不会超过自身半高太多 → inside (碗里的面包)
                    if verbose:
                        print(f"      [reseat] {name}: 陷入 {sup} 深度 {sink:.4f}m > "
                              f"{inside_ratio}×自身高 {h:.4f}m, 判为 inside, 跳过", flush=True)
                    continue

            sup_top = float(sb[1, 2])
            gap = float(b[0, 2]) - sup_top
            if abs(gap) <= gap_tol:
                continue

            T = np.asarray(instance_info["T"], dtype=float).copy()
            T_new = T.copy()
            T_new[2, 3] -= gap

            check = instance_info["original_mesh"].copy()
            check.apply_transform(T_new)
            new_gap = float(check.bounds[0, 2]) - sup_top
            if abs(new_gap) > gap_tol:
                if verbose:
                    print(f"      [reseat] {name}: 重贴校验失败 (gap={new_gap:+.4f}m), 保持原位",
                          flush=True)
                continue

            instance_info["T"] = T_new
            bounds[name] = _world_bounds(instance_info)  # 后续物体要看到新位置
            n_reseated += 1
            moved += 1
            if verbose:
                tag = "推断" if inferred else "关系"
                print(f"      [reseat] {name}: gap {gap:+.4f}m → {new_gap:+.4f}m "
                      f"({tag}支撑物 {sup} 顶面 {sup_top:.4f}m, 重叠 {overlap*100:.0f}%)", flush=True)

        if moved == 0:
            break

    return all_instances, n_reseated


def dump_placement_audit(all_instances, refined_relations, placement_strategies=None,
                         output_dir=None, gap_tol=0.03, verbose=True):
    '''
    落盘摆放审计: 每实例的世界 AABB / 关系 / 支撑物 / gap / 水平重叠 / 判定。

    为什么需要: 此前唯一能拿到最终位姿的地方是 stage61_physics_report.json,
    但它只记录 bottom_z (离地高度), 没有支撑物顶面, 算不出 gap —— 导致
    "桌上物体离地 0.9m" 被误判成悬空, 而真正陷进支撑物的物体反而看不出来
    (实测 egodex_0 rubik's cube_0 陷进 box_0 达 0.102m)。
    这里在 5.25 结束时把几何量一次性记全, 后面的修复才有验收依据。

    Args:
        output_dir: 非空时写 <output_dir>/placement_audit.json
    Returns:
        audit dict
    '''
    index = _instance_index(all_instances)
    bounds = {n: _world_bounds(all_instances[c][i]) for n, (c, i) in index.items()}

    objects = []
    for name in sorted(index):
        category, _ = index[name]
        relation = str(refined_relations.get(name, "")
                       or refined_relations.get(category, "") or "")
        kind, sup = _parse_supporter(relation)
        b = bounds[name]

        rec = {
            "name": name,
            "category": category,
            "relation": relation,
            "min_z": float(b[0, 2]),
            "max_z": float(b[1, 2]),
            "height": float(b[1, 2] - b[0, 2]),
            "center": [float(v) for v in (b[0] + b[1]) / 2.0],
            "supporter": None,
            "supporter_top": None,
            "gap": None,
            "xy_overlap": None,
            "status": "unsupported",
        }

        if kind in ("floor", "wall"):
            rec["supporter"] = kind
            rec["gap"] = rec["min_z"]
            rec["status"] = "ok" if abs(rec["min_z"]) <= gap_tol else (
                "floating" if rec["min_z"] > 0 else "sunken")
        elif sup is not None and sup in bounds:
            sb = bounds[sup]
            rec["supporter"] = sup
            rec["supporter_top"] = float(sb[1, 2])
            rec["gap"] = float(b[0, 2]) - float(sb[1, 2])
            rec["xy_overlap"] = _xy_overlap_ratio(b, sb)
            rec["supporter_height"] = float(sb[1, 2] - sb[0, 2])
            rec["status"] = _gap_status(rec["gap"], rec["supporter_height"], gap_tol)
        elif kind == "unknown":
            inferred = _infer_supporter(name, b, bounds)
            if inferred is not None:
                sb = bounds[inferred]
                rec["supporter"] = inferred + " (推断)"
                rec["supporter_top"] = float(sb[1, 2])
                rec["gap"] = float(b[0, 2]) - float(sb[1, 2])
                rec["xy_overlap"] = _xy_overlap_ratio(b, sb)
                rec["supporter_height"] = float(sb[1, 2] - sb[0, 2])
                rec["status"] = _gap_status(rec["gap"], rec["supporter_height"], gap_tol)
            else:
                rec["status"] = "no_supporter_found"

        if placement_strategies is not None:
            rec["strategy"] = placement_strategies.get(name) or \
                placement_strategies.get(category.lower().strip())
        objects.append(rec)

    n_bad = sum(1 for r in objects
                if r["status"] in ("floating", "sunken", "severe_penetration",
                                   "no_supporter_found"))
    audit = {
        "n_objects": len(objects),
        "n_problem": n_bad,
        "gap_tol": gap_tol,
        "n_by_status": {s: sum(1 for r in objects if r["status"] == s)
                        for s in sorted({r["status"] for r in objects})},
        "objects": objects,
    }

    if output_dir:
        try:
            path = os.path.join(output_dir, "placement_audit.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"   📋 摆放审计已落盘: {path} ({n_bad}/{len(objects)} 有问题)", flush=True)
        except Exception as e:
            print(f"   ⚠️ 摆放审计落盘失败: {e}", flush=True)
    return audit


def refine_supported_by_floor_object(object_info):
    '''
    This function refines the "supported_by_floor" relationship by align the axis and bottom of the object to the floor.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
    Returns: The same dict with a different "T" to align the object to the floor.
    '''
    old_pos = object_info["T"][:3, 3].copy()
    transform_matrix = object_info["T"]
    upper_real_vector = np.array([0, 0, 1])
    # Note that glb is y-up, but ours format is z-up.
    upper_transformed_vector = transform_matrix[:3,1] / np.linalg.norm(transform_matrix[:3,1])
    theta_gravity = np.arccos(np.clip(np.dot(upper_real_vector, upper_transformed_vector), -1.0, 1.0)) / np.pi * 180
    # if the upper_transformed_vector is close to the gravity direction, we consider it as supported_by_floor and align it to the floor. Otherwise, we keep the original transformation.
    if theta_gravity < 10.0 or theta_gravity > 170.0:
        if theta_gravity < 90.0:
            upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, upper_real_vector)
        else:
            upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, -upper_real_vector)
    else:
        upper_align_matrix = np.eye(4)
    # align the axis of the object to the floor, we only align the rotation part and keep the translation part unchanged to preserve the original position of the object.
    transform_matrix[:3, :3] = upper_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    # ── E: 门限与贴地自相矛盾 ──
    # 上面的门限只在 theta_gravity ∈ (10°,170°) 时"不扶正", 但下面无条件把物体
    # 贴到 z=0。二者矛盾: 物体被判定为"朝向不可信"却仍被强行摆到地面, 看起来
    # 落地了, 实际掩盖了上游坐标系对齐错误 (GeoCalib 重力方向错→ 整场景 z 轴错)。
    # 实测历史日志 290 条: theta_gravity ∈ [0.1°, 177.6°], 其中 170 条 (59%)
    # 落在 (10°,170°) 不扶正区间, z_min ∈ [-1.5146m, 1.7867m]仍全部贴到 z=0。
    #
    # 修法: 恢复 z_min 阈值。仅当物体确实接近地面时才落地, 否则保持原位并告警,
    # 让"坐标系可能没对齐"暴露出来, 而不是被贴地掩盖。
    # 阈值取 0.5m: 覆盖真实小偏差 (实测椅子 0.76-1.04m 属于对齐错误而非落地偏差),
    # 又不至于把明显错位的物体硬拽到地面。
    _FLOOR_Z_MAX = 0.5

    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]

    if abs(z_min) > _FLOOR_Z_MAX:
        print(f"      [floor] ⚠️ 跳过落地: theta_gravity={theta_gravity:.1f}°, "
              f"z_min={z_min:+.4f}m 超出 ±{_FLOOR_Z_MAX}m → 疑似坐标系未对齐"
              f"(非落地偏差), 保持原位不强行贴地", flush=True)
        object_info["T"] = transform_matrix
        return object_info

    translation_vector = np.array([0, 0, -z_min])
    translation_matrix = trimesh.transformations.translation_matrix(translation_vector)
    # align the bottom of the object to the floor
    transform_matrix = translation_matrix @ transform_matrix
    object_info["T"] = transform_matrix
    new_pos = object_info["T"][:3, 3].copy()
    _note = "" if theta_gravity < 10.0 or theta_gravity > 170.0 else " (未扶正, 朝向存疑)"
    print(f"      [floor] theta_gravity={theta_gravity:.1f}°, z_min={z_min:+.4f}m{_note}, "
          f"pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return object_info


def refine_embedded_in_wall_object(object_info, walls_info):
    '''
        This function refines the "embedded_in_wall" relationship by aligning the object horizontal axis
        to a wall direction and snapping the object center to the nearest valid wall plane.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
      walls_info: list returned by get_walls_info().
            camera_pos: optional camera position, kept for interface consistency with attached-to-wall refinement.
    Returns: The same dict with refined "T".
    '''
    old_pos = object_info["T"][:3, 3].copy()
    if not walls_info:
        return object_info

    transform_matrix = object_info["T"].copy()
    forward_vector = transform_matrix[:3, 2]
    forward_norm = np.linalg.norm(forward_vector)
    if forward_norm < 1e-8:
        return object_info
    forward_vector = forward_vector / forward_norm

    align_vector, wall_axis = _get_wall_alignment_target(forward_vector, angle_tolerance=30.0)
    if align_vector is None:
        return object_info

    # align object horizontal axis to wall axis; only update rotation and keep translation unchanged
    horizontal_align_matrix = trimesh.geometry.align_vectors(transform_matrix[:3, 2], align_vector)
    transform_matrix[:3, :3] = horizontal_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    center = transform_matrix[:3, 3]
    # snap the object center to the nearest wall plane if distance is small enough
    nearest_wall, min_dist = _select_closest_wall(walls_info, wall_axis, center, span_margin=0.3)
    if nearest_wall is not None and min_dist <= 0.3:
        axis_idx = 0 if wall_axis == 'x' else 1
        offset = nearest_wall['position'] - center[axis_idx]
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[axis_idx] = offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    # keep object above floor plane z=0
    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]
    if z_min < 0.0:
        transform_matrix = trimesh.transformations.translation_matrix(np.array([0.0, 0.0, -z_min])) @ transform_matrix

    object_info["T"] = transform_matrix
    new_pos = object_info["T"][:3, 3].copy()
    print(f"      [embedded_wall] wall_axis={wall_axis}, pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return object_info


def refine_attached_to_wall_object(object_info, walls_info, camera_pos=None):
    '''
        This function refines the "attached_to_wall" relationship by aligning the object horizontal axis,
        and snapping the back surface to the nearest wall plane.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
      walls_info: list returned by get_walls_info().
            camera_pos: optional camera position used to determine which side of the object should touch the wall.
    Returns: The same dict with refined "T".
    '''
    old_pos = object_info["T"][:3, 3].copy()
    if not walls_info:
        return object_info

    transform_matrix = object_info["T"].copy()
    forward_vector = transform_matrix[:3, 2]
    forward_norm = np.linalg.norm(forward_vector)
    if forward_norm < 1e-8:
        return object_info
    forward_vector = forward_vector / forward_norm

    align_vector, wall_axis = _get_wall_alignment_target(forward_vector, angle_tolerance=20.0)
    if align_vector is None:
        return object_info

    # align object horizontal axis to wall axis; only update rotation and keep translation unchanged
    horizontal_align_matrix = trimesh.geometry.align_vectors(transform_matrix[:3, 2], align_vector)
    transform_matrix[:3, :3] = horizontal_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    center = transform_matrix[:3, 3]
    nearest_wall, min_dist = _select_closest_wall(walls_info, wall_axis, center, span_margin=0.3)
    if nearest_wall is not None and min_dist <= 0.3:
        axis_idx = 0 if wall_axis == 'x' else 1

        # use back face as contact surface. camera_pos determines which side is considered as back side.
        transformed_vertices = trimesh.transformations.transform_points(
            object_info["original_mesh"].vertices.copy(),
            transform_matrix,
        )
        if camera_pos is not None:
            if camera_pos[axis_idx] > center[axis_idx]:
                contact_val = transformed_vertices[:, axis_idx].min()
            else:
                contact_val = transformed_vertices[:, axis_idx].max()
        else:
            # fallback to aligned direction when camera position is unavailable
            if align_vector[axis_idx] > 0:
                contact_val = transformed_vertices[:, axis_idx].min()
            else:
                contact_val = transformed_vertices[:, axis_idx].max()

        snap_offset = nearest_wall['position'] - contact_val
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[axis_idx] = snap_offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    # keep object above floor plane z=0
    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]
    if z_min < 0.0:
        transform_matrix = trimesh.transformations.translation_matrix(np.array([0.0, 0.0, -z_min])) @ transform_matrix

    object_info["T"] = transform_matrix
    new_pos = object_info["T"][:3, 3].copy()
    print(f"      [attached_wall] wall_axis={wall_axis}, pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return object_info

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
                            float_eps=0.02, post_tol=0.01, float_z_max=0.5,
                            ref_z=None, ref_tol=0.10):
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
        float_z_max: 悬空超过该值视为"疑似坐标系未对齐", 默认跳过不压平
        ref_z: {实例名: 参考 z_min}, 取自 Stage3 基础精修后的位置。
            **为什么需要**: float_z_max 的保护无法区分两种"高空悬停":
              (a) 坐标系未对齐 —— 物体从基础精修起就高 (参考值也高)
              (b) 被上游精修推高 —— 基础精修时贴地 (参考值≈0), 但 5.2 的
                  穿模修复把它推上去了
            实测 basic_pick_place_17 toy_1: 基础精修 z_min=0.0000, Stage5 后
            变成 0.8806m (>0.5 → 被跳过), toy_5 同理 0.6379m, 最终渲染时两个
            物体飘在空中。
            有了 ref_z 就能区分: 参考值≈0 且当前高出 ref_tol 以上 → 判为 (b),
            照常压回; 参考值本身就高 → 判为 (a), 保持跳过.
        ref_tol: 当前 z_min 超出参考值多少算"被推高"
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

            # 已贴地 (|z_min| 在浮点噪声内) → 无需处理
            if abs(z_min) <= float_eps:
                continue

            # 陷地 (z_min < 0): 一定是错的。与悬空不同, "半埋在地下"没有
            # "坐标系未对齐"这种歧义解释 —— 对齐错误通常表现为整体抬高/压低,
            # 而单个物体单独陷地只能是几何/关系误判。所以这里不做上界保护,
            # 直接压回 z=0。
            # 实测 basic_pick_place_46 card_0: 5.1 把 'supported by other objects'
            # 误判为 floor, Stage3 的 refine_supported_by_floor_object 因此没跑,
            # 物体保持 VGGT 原始位置 z_min=-0.049m, audit 标 sunken。
            if z_min < 0:
                T_new = T.copy()
                T_new[2, 3] -= z_min
                check = instance_info["original_mesh"].copy()
                check.apply_transform(T_new)
                z_after = float(check.bounds[0, 2])
                if abs(z_after) > post_tol:
                    if verbose:
                        print(f"      [ground] {name}: 陷地修正校验失败 (z={z_after:+.4f}m), "
                              f"保持原位", flush=True)
                    continue
                instance_info["T"] = T_new
                n_grounded += 1
                if verbose:
                    print(f"      [ground] {name}: 陷地 z={z_min:+.4f}m → 地面 z={z_after:+.4f}m",
                          flush=True)
                continue

            # 悬空: 上界保护 —— 不加会把上游对齐错误强行压平 (抵消 E 项效果)。
            # 但上界保护无法区分"坐标系未对齐"与"被上游精修推高", 用 ref_z 区分:
            #   参考值≈0 (基础精修时贴地) 且当前高出 ref_tol → 被推高, 照常压回
            #   参考值本身就高 → 坐标系未对齐, 保持跳过
            if z_min > float_z_max:
                _ref = (ref_z or {}).get(name)
                _pushed_up = (_ref is not None
                              and abs(_ref) <= float_eps
                              and (z_min - _ref) > ref_tol)
                if not _pushed_up:
                    if verbose:
                        print(f"      [ground] {name}: 跳过 (z_min={z_min:+.4f}m > "
                              f"{float_z_max}m, 疑似坐标系未对齐而非悬空"
                              + (f", 基础精修参考 z={_ref:+.4f}m" if _ref is not None else "")
                              + ")", flush=True)
                    continue
                if verbose:
                    print(f"      [ground] {name}: 高空悬空但基础精修时贴地 "
                          f"(ref z={_ref:+.4f}m) → 判为被上游推高, 压回地面",
                          flush=True)

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


def separate_object_penetrations(all_instances, refined_relations, verbose=True,
                                 pen_thresh=0.10, push_step=0.015, max_iter=40,
                                 gap_tol=0.005):
    """把互相穿模的物体沿 XY 推开 (只改水平位置, 不动 z / 旋转 / 尺寸)。

    为什么需要: 之前三层都没管"物体之间"的穿模 ——
      1. Stage1 的 prompt 要求 "If unsure, default to: supported by floor"
         (tools/generate_scene_json_stage1.py:225), 俯视桌面场景 SAM 看不到地面,
         VLM 按兜底把所有物体标成 floor
      2. 基础精修 refine_supported_by_floor_object 只做"单物体贴地/扶正"
      3. 5.2 的 resolve_penetrations 带 only_refine_other_objects=True,
         "supported by floor" 全部跳过
     实测 basic_pick_place_111 lego_0/lego_1 有 85% 顶点互相在对方 AABB 内,
     basic_pick_place_46 box_1/box_2 92%、card_0 埋进 box_1 91.6%,
     basic_pick_place_118 croissant_0 埋进 pillow_0 68% —— 全部没人管。

    做法: 迭代地把穿模对沿 XY 碰撞法向各推开一半, 直到无穿模或达到迭代上限。
    只改 xy 是为了不破坏已由 ground_floating_objects / reseat 修好的 z。

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 T
        refined_relations: {实例名: 关系} (仅用于跳过"本就是包含关系"的 inside)
        pen_thresh: 一方顶点落入对方 AABB 的比例超过该值即算穿模
        push_step: 单次推开步长 (米)
        max_iter: 最大迭代轮数
        gap_tol: 认为"已分开"的水平间隙
    Returns:
        (all_instances, n_separated)
    """
    import numpy as np

    def _items():
        out = []
        for cat, lst in all_instances.items():
            for idx, info in enumerate(lst):
                out.append((f"{cat}_{idx}", cat, idx, info))
        return out

    def _bounds(info):
        m = info["original_mesh"].copy()
        m.apply_transform(np.asarray(info["T"], dtype=float))
        return m.bounds


    def _penetration(ia, ib):
        """第 ia 个实例的顶点落入第 ib 个 AABB 的比例 (取双向最大值)。

        贴合 vs 真穿模的区分: 不收缩对方 AABB, 而是把**测试顶点沿自身中心
        缩 2%**。
          · 贴合面接触 → 顶点缩离对方 → 不算穿模 ✅
          · 真互穿     → 顶点仍在对方内部 → 算穿模 ✅
          · 盒子/薄片/任意网格都成立, 不依赖网格是否水密
        旧实现是把对方 AABB 每侧收缩 3mm, 对 box 型 mesh 致命: 盒子所有
        顶点都在边界上, 收缩后一个都进不去 → 穿模恒为 0 → 对盒子类物体
        完全失效 (实测两个 0.1m 立方体 100% 重合时 n=0)。
        """
        pa = _verts[ia]
        pb = _verts[ib]
        if len(pa) == 0 or len(pb) == 0:
            return 0.0
        # ── 与 dump_placement_audit 完全一致的判据 ──
        # 必须一致, 否则两边结论矛盾: 实测 002 里 audit 报 0 处穿模, 分离函数
        # 却推了 40 次 (它只用 AABB, 把"枕头压在被子上"也判穿模)。乱推会破坏
        # 本来正确的堆叠 —— 这是"越改越差"的另一个来源。
        # 判据: AABB 交叠 / 较小者体积 > 10%  且  mesh contains 比例 > 10%。
        _lo = np.maximum(_bb[ia][0], _bb[ib][0])
        _hi = np.minimum(_bb[ia][1], _bb[ib][1])
        _inter = float(np.prod(np.clip(_hi - _lo, 0.0, None)))
        if _inter <= 1e-7:
            return 0.0
        _va = float(np.prod(np.maximum(_bb[ia][1] - _bb[ia][0], 1e-9)))
        _vb = float(np.prod(np.maximum(_bb[ib][1] - _bb[ib][0], 1e-9)))
        if _inter / max(min(_va, _vb), 1e-9) <= 0.10:
            return 0.0
        _ERODE = 0.005
        # 再用 mesh 级 contains 验证, 与 dump_placement_audit 的判据保持一致。
        # 为什么必须一致: 分离函数原来只用 AABB, 而 audit 用 contains ——
        # 结果是分离函数对"枕头压在被子上"这类正常堆叠也判穿模, 疯狂推 40 次
        # 而 audit 的穿模对一个都没减少 (实测 002: 0 → 0 却推了 40 次;
        # 151: 2 → 2 也推了 40 次)。乱推会破坏本来正确的摆放。
        def _frac(_src_v, _src_c, _dst):
            _e = _src_v + (_src_c - _src_v) * _ERODE
            if len(_e) > 600:
                _e = _e[np.linspace(0, len(_e) - 1, 600).astype(int)]
            try:
                return float(np.mean(_dst.contains(_e)))
            except Exception:
                lo, hi = _dst.bounds[0], _dst.bounds[1]
                return float(np.all((_e >= lo) & (_e <= hi), axis=-1).mean())
        ra = _frac(pa, _ctr[ia], _meshes[ib])
        rb = _frac(pb, _ctr[ib], _meshes[ia])
        return max(ra, rb)

    items = _items()
    # ── 顶点缓存 ──
    # 为什么要缓存: 旧实现对每一对、每一轮迭代都 copy+transform 两个 mesh。
    # 25 个实例 = 300 对, 40 轮 = 12000 次 mesh 变换 → 实测超时 (>900s)。
    # 顶点只需在世界坐标系里算一次, 之后每次推开只做一次平移即可。
    _meshes = []         # 世界坐标 mesh (contains 用)
    _verts = []          # 世界坐标顶点 (np.ndarray)
    _bb = []             # 世界 AABB
    _ctr = []            # 顶点质心 (用于收缩)
    for _n, _c, _i, _info in items:
        _m = _info["original_mesh"].copy()
        _m.apply_transform(np.asarray(_info["T"], dtype=float))
        _v = np.asarray(_m.vertices, dtype=float)
        _verts.append(_v)
        _bb.append(_m.bounds)
        _meshes.append(_m)
        _ctr.append(_v.mean(axis=0) if len(_v) else np.zeros(3))

    # relations 里的具名堆叠对: "A supported by B" 且 B 是具体实例。
    # 这类对穿模时沿 z 抬升解决, 严禁 XY 推离 —— XY 推离会把贴合堆叠的
    # 物体推出支撑物足迹, 之后 reseat 因 overlap<min_overlap 拒绝拉回,
    # ground_floating 又因关系非 floor 不抬回, 物体就此落在地上。
    stacked = {}
    for _n, _rel in (refined_relations or {}).items():
        if isinstance(_rel, str) and _rel.startswith("supported by "):
            _sup = _rel[len("supported by "):].strip()
            if _sup and _sup not in ("floor", "wall") and "other objects" not in _sup and "_" in _sup:
                stacked[_n] = _sup

    def _xy_overlap_frac(ca, cb):
        """child 顶点足迹落入 parent 足迹 XY 范围的比例。"""
        pc = np.asarray(ca["original_mesh"].vertices)
        pc = (np.asarray(ca["T"], dtype=float) @ np.vstack(
            [pc.T, np.ones(len(pc))])).T[:, :2]
        pp = np.asarray(cb["original_mesh"].vertices)
        pp = (np.asarray(cb["T"], dtype=float) @ np.vstack(
            [pp.T, np.ones(len(pp))])).T[:, :2]
        lo, hi = pp.min(0), pp.max(0)
        inside = np.all((pc >= lo) & (pc <= hi), axis=-1).mean()
        return float(inside)

    skip_pairs = set()
    n_sep = 0
    for _it in range(max_iter):
        worst, worst_ratio = None, pen_thresh
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if (i, j) in skip_pairs:
                    continue
                na, ca, ia, xa = items[i]
                nb, cb, ib, xb = items[j]
                # 跳过 floor: 物体陷进地板/支撑物是**竖向**问题, 由
                # ground_floating_objects / reseat 修 (改 z)。本函数只改 XY,
                # 对竖向穿模无能为力 —— 不跳过会一直找不到解而白跑满 max_iter
                # (实测 151: cabinet_3 陷进 floor 13%, 推 40 次仍报 2 对)。
                if "floor" in f"{ca}".lower() or "floor" in f"{cb}".lower():
                    skip_pairs.add((i, j))
                    continue
                r = _penetration(i, j)
                if r > worst_ratio:
                    worst, worst_ratio = (i, j), r
        if worst is None:
            break
        i, j = worst
        na, ca, ia, xa = items[i]
        nb, cb, ib, xb = items[j]
        # 堆叠对: 沿 z 抬升 child (保持 xy), 不做水平推离
        if stacked.get(na) == nb or stacked.get(nb) == na:
            child, parent, cname = (xa, xb, na) if stacked.get(na) == nb else (xb, xa, nb)
            if _xy_overlap_frac(child, parent) < 0.3:
                # child 根本不在 parent 足迹上方: 几何上无法用抬升解决,
                # 语义归属留给上层 (5.15/5.2), 这里跳过该对防止空转
                skip_pairs.add((i, j))
                if verbose:
                    print(f"      [sep] {cname}: 堆叠关系但 xy 不在支撑物足迹内, 跳过", flush=True)
                continue
            mp = parent["original_mesh"].copy()
            mp.apply_transform(np.asarray(parent["T"], dtype=float))
            mc = child["original_mesh"].copy()
            mc.apply_transform(np.asarray(child["T"], dtype=float))
            lift = float(mp.bounds[1][2]) + gap_tol - float(mc.bounds[0][2])
            if lift > 0:
                Tc = np.asarray(child["T"], dtype=float).copy()
                Tc[2, 3] += lift
                child["T"] = Tc
                n_sep += 1
                if verbose and n_sep <= 12:
                    print(f"      [sep] {cname}: 与支撑物 {stacked.get(cname)} 穿模 "
                          f"→ 抬升 {lift:.3f}m (xy 不动)", flush=True)
            continue
        ba, bb = _bounds(xa), _bounds(xb)
        # 沿"两 AABB 中心连线"在 XY 平面上的方向推开, 各推一半
        cax, cay = (ba[0][0] + ba[1][0]) / 2, (ba[0][1] + ba[1][1]) / 2
        cbx, cby = (bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2
        dx, dy = cax - cbx, cay - cby
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            # 完全同心: 用索引决定方向, 保证能分开
            dx, dy, norm = 1.0, 0.0, 1.0
        dx, dy = dx / norm, dy / norm
        Ta = np.asarray(xa["T"], dtype=float).copy()
        Tb = np.asarray(xb["T"], dtype=float).copy()
        Ta[0, 3] += dx * push_step
        Ta[1, 3] += dy * push_step
        Tb[0, 3] -= dx * push_step
        Tb[1, 3] -= dy * push_step
        # 不做"夹回支撑面"的约束。曾试过 _clamp_into_support, 会让
        # basic_pick_place_141 彻底卡死: 9 个棋子被夹在棋盘内后无法互相让开,
        # 每轮迭代都推不动, 跑满 max_iter=40 仍无进展 (单场景 >6 分钟)。
        # 说明该场景按当前 mesh 尺度本来就在棋盘内放不下 —— 那是尺度问题,
        # 不该由分离环节用"把物体推出棋盘"来偿还。
        xa["T"] = Ta
        xb["T"] = Tb
        # 同步平移缓存, 后续迭代才能看到新的位置
        _shift = np.array([dx * push_step, dy * push_step, 0.0])
        _verts[i] = _verts[i] + _shift
        _ctr[i] = _ctr[i] + _shift
        _bb[i] = _bb[i] + _shift
        _verts[j] = _verts[j] - _shift
        _ctr[j] = _ctr[j] - _shift
        _bb[j] = _bb[j] - _shift
        _meshes[i].apply_translation(_shift)
        _meshes[j].apply_translation(-_shift)
        n_sep += 1
        if verbose and n_sep <= 12:
            print(f"      [sep] {na} ↔ {nb}: 穿模 {worst_ratio*100:.1f}% → 各推 "
                  f"({dx*push_step:+.4f},{dy*push_step:+.4f})m", flush=True)

    if verbose and n_sep:
        print(f"      [sep] 合计推开 {n_sep} 次", flush=True)
    elif verbose:
        print(f"      [sep] 无穿模 (阈值 {pen_thresh*100:.0f}%)", flush=True)
    return all_instances, n_sep


def recenter_edge_objects(all_instances, refined_relations, verbose=True,
                          min_overlap=0.5, skip_below=0.0, max_shift=0.35,
                          max_iter=3):
    """on_top 物体悬在支撑物边沿时, 沿 XY 平移到支撑物中心。

    为什么需要: reseat_on_top_objects 只修 z (贴合顶面), 不动水平位置, 所以
    "只搭上一个角"的物体照样贴着顶面悬浮在边上。
    实测 vertical_pick_place_115: bottle_0 与 box_0 的水平重叠只有 41%,
    即 59% 的瓶子悬在盒外 —— 视觉上就是"贴在盒边"。

    与 reseat 的关系: reseat 契约是"只改 z", 这里单独一步只改 XY, 互不干扰;
    顺序上必须放在 reseat 之后 (z 先落位, xy 再居中)。

    保守性:
      - 重叠 < skip_below (0.15) 时不处理 —— 那多半是支撑物判错, 硬拽更糟
        (reseat 自己的护栏用的是 0.10, 这里略高一些)
      - 单次平移量受 max_shift 限制, 最多迭代 max_iter 轮
      - 只处理 "supported by <非 floor/wall>" 的关系, floor 物体不动

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 T
        refined_relations: {实例名: 关系}
        min_overlap: 水平重叠达到该比例即认为已摆正, 不再移动
        skip_below: 重叠低于该比例视为支撑物判错, 跳过
        max_shift: 累计平移上限 (米), 防止甩飞
        max_iter: 最大迭代轮数
    Returns:
        (all_instances, n_recentered)
    """
    import numpy as _np

    index = _instance_index(all_instances)

    def _bounds(info):
        m = info["original_mesh"].copy()
        m.apply_transform(_np.asarray(info["T"], dtype=float))
        return m.bounds

    def _lookup(name):
        if name not in index:
            return None
        cat, idx = index[name]
        try:
            return all_instances[cat][idx]
        except Exception:
            return None

    n_moved = 0
    for it in range(max_iter):
        changed = 0
        for name, (cat, idx) in list(index.items()):
            rel = str(refined_relations.get(name, ""))
            if not rel.startswith("supported by "):
                continue
            sup_name = rel[len("supported by "):].strip()
            if sup_name.lower() in ("floor", "ground", "wall", ""):
                continue
            info = _lookup(name)
            sup = _lookup(sup_name)
            if info is None or sup is None or info is sup:
                continue
            b, sb = _bounds(info), _bounds(sup)
            ov = _xy_overlap_ratio(b, sb)
            if ov >= min_overlap:
                continue
            # 朝支撑物 xy 中心平移
            cx = (b[0][0] + b[1][0]) / 2
            cy = (b[0][1] + b[1][1]) / 2
            scx = (sb[0][0] + sb[1][0]) / 2
            scy = (sb[0][1] + sb[1][1]) / 2
            # 最小位移: 把物体中心 clamp 进"支撑面内缩物体半宽"后的范围, 让物体
            # 完整落在支撑面上, 而不是拖到正中心。
            # 拖到正中心位移往往过大 (实测 115 的 bottle_0 完全在 box_0 外面,
            # 距中心 0.40m), 既容易被"距离过大"的护栏误拦, 也让物体跑得比必要更远。
            # 只 clamp 中心本身的话物体仍有一半悬在边界上 (重叠 50%); 内缩半个
            # 物体尺寸后能到 100%。
            obj_hw = (b[1][0] - b[0][0]) / 2
            obj_hh = (b[1][1] - b[0][1]) / 2
            lo_x, hi_x = sb[0][0] + obj_hw, sb[1][0] - obj_hw
            lo_y, hi_y = sb[0][1] + obj_hh, sb[1][1] - obj_hh
            # 支撑物比物体还小 → 内缩后范围反转, 退化为对齐中心
            tx = min(max(cx, lo_x), hi_x) if lo_x <= hi_x else scx
            ty = min(max(cy, lo_y), hi_y) if lo_y <= hi_y else scy
            dx, dy = tx - cx, ty - cy
            dist = float(_np.hypot(dx, dy))
            if dist < 1e-6:
                continue
            # 兜底护栏: clamp 后仍超过上限, 说明支撑物范围异常, 不硬拽
            if dist > max_shift:
                if verbose:
                    print(f"      [recenter] {name} 需平移 {dist:.2f}m > {max_shift}m, "
                          f"疑支撑物范围异常, 跳过", flush=True)
                continue
            ux, uy = dx, dy
            T = _np.asarray(info["T"], dtype=float).copy()
            T[0, 3] += ux
            T[1, 3] += uy
            info["T"] = T
            changed += 1
            n_moved += 1
            if verbose and n_moved <= 10:
                print(f"      [recenter] {name} 在 {sup_name} 上仅 {ov*100:.0f}% 重叠"
                      f" → 平移 ({ux:+.4f},{uy:+.4f})m", flush=True)
        if not changed:
            break

    if verbose:
        print(f"      [recenter] 共居中 {n_moved} 次"
              f"{'' if n_moved else f' (无 <{min_overlap*100:.0f}% 重叠的边沿物体)'}",
              flush=True)
    return all_instances, n_moved


def repair_floor_relations(all_instances, refined_relations, verbose=True,
                           min_overlap=0.30):
    """floor 关系坐实检查 (独立入口, 供基础精修之前调用)。

    VLM 常把"桌上/托盘上"的物体判成 floor。物体底面明显不在地面 (>0.05m)、
    却完整落在某个具名实例足迹内且高度吻合 (|gap|≤0.06m) 时, 改挂该支撑物。

    为什么必须在基础精修**之前**: 基础精修对 floor 关系无条件贴地 —— 若它
    先跑, 物体已被砸到 z=0, 本检查看到"在地上"就不再改挂 (实测 egodex_32
    screwdriver 躺在 tray 上, 底 0.084m, 被砸到 0 后 5.15 放弃修正)。

    Returns: (refined_relations, n_fixed)
    """
    import numpy as _np

    def _bounds(info):
        m = info["original_mesh"].copy()
        m.apply_transform(_np.asarray(info["T"], dtype=float))
        return m.bounds

    def _score(child_b, sup_b):
        ox = max(0.0, min(child_b[1][0], sup_b[1][0]) - max(child_b[0][0], sup_b[0][0]))
        oy = max(0.0, min(child_b[1][1], sup_b[1][1]) - max(child_b[0][1], sup_b[0][1]))
        area = max((child_b[1][0] - child_b[0][0]) * (child_b[1][1] - child_b[0][1]), 1e-9)
        ov = ox * oy / area
        gap = abs(float(child_b[0][2] - sup_b[1][2]))
        return ov - gap * 0.5

    _idx = {}
    for _cat, _lst in all_instances.items():
        for _i, _info in enumerate(_lst):
            _idx[f"{_cat}_{_i}"] = (_cat, _info, _bounds(_info))

    n_fixed = 0
    for name, rel in list(refined_relations.items()):
        if not rel.startswith("supported by "):
            continue
        sup = rel[len("supported by "):].strip()
        if sup.lower() not in ("floor", "ground"):
            continue
        if name not in _idx:
            continue
        _, _, cb = _idx[name]
        if float(cb[0][2]) <= 0.05:
            continue                     # 真在地上, floor 没错
        _best_sup, _best_sc = None, None
        for _cand, (_ccat, _cinfo, _cbb) in _idx.items():
            if _cand == name:
                continue
            _ov = _score(cb, _cbb)
            _gap = abs(float(cb[0][2] - _cbb[1][2]))
            if _ov >= min_overlap and _gap <= 0.06:
                if _best_sc is None or _ov > _best_sc:
                    _best_sup, _best_sc = _cand, _ov
        if _best_sup is not None:
            refined_relations[name] = f"supported by {_best_sup}"
            n_fixed += 1
            if verbose:
                print(f"      [支撑物纠偏] {name}: 底面离地 {cb[0][2]:.3f}m 却被判 floor, "
                      f"xy 完整落在 {_best_sup} 足迹内 (重叠分 {_best_sc:.2f}) "
                      f"→ 改挂 {_best_sup}", flush=True)
    return refined_relations, n_fixed


def resolve_orphan_wall_relations(all_instances, refined_relations, verbose=True,
                                  flat_thre=0.20, drop_flat_fragments=True):
    """处理"关系是挂墙、但场景里没有墙"的孤儿实例。

    为什么需要: Stage1 的 VLM 常把**墙面本身**报成 curtain (3chair 场景报了 7 个
    curtain, 关系全是 attached to wall), 但 Stage3 并不会生成 wall 实例 —— 于是
    这些实例没有任何参照面, 既不会被落地补正 (它不属于 floor), 也不会被 reseat
    (它没有支撑物), 最终停在 VGGT 的原始位置: 有的悬在半空 (curtain_0 底 z=+0.317),
    有的陷进地下 (curtain_5 -0.139 / curtain_6 -0.108)。

    分治判据 (纯几何, 不依赖类别名):
      · 扁平度 = min(ext)/max(ext) < 0.20 → 极薄片, 是墙面/隔板被切成"物体"的
        碎片, 不是实体物件 → 直接剔除 (它本来就不该出现在场景里)
      · 其余 → 是块状实体, 只是关系标错 → 回退 supported by floor, 交给落地补正
        (这样它会贴到 z=0, 不再是悬空/陷地的白片)

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 (可能删元素)
        refined_relations: {实例名: 关系}, 原地修改
        flat_thre: 扁平度阈值
        drop_flat_fragments: 是否剔除极薄片 (False 则也回退 floor)
    Returns:
        (all_instances, refined_relations, n_dropped, n_refallback)
    """
    import numpy as _np

    # 场景里有没有墙? (类别名含 wall, 或者类别本身就叫 wall)
    has_wall = any("wall" in str(c).lower() for c in all_instances)
    # 统计有多少实例声明"挂墙"
    orphans = [n for n, r in refined_relations.items()
               if "attach" in str(r).lower() and "wall" in str(r).lower()]
    if not orphans:
        return all_instances, refined_relations, 0, 0
    if has_wall:
        return all_instances, refined_relations, 0, 0

    # 建 名→(cat,idx) 索引
    index = {}
    for _c, _lst in all_instances.items():
        for _i in range(len(_lst)):
            index[f"{_c}_{_i}"] = (_c, _i)

    drop_keys = []
    refallback = 0
    for name in orphans:
        if name not in index:
            continue
        _c, _i = index[name]
        _info = all_instances[_c][_i]
        try:
            _m = _info["original_mesh"].copy()
            _m.apply_transform(_np.asarray(_info["T"], dtype=float))
            _e = _m.bounds[1] - _m.bounds[0]
            _flat = float(min(_e) / max(max(_e), 1e-9))
        except Exception:
            _flat = 1.0
        if drop_flat_fragments and _flat < flat_thre:
            drop_keys.append(name)
            if verbose:
                print(f"      [挂墙孤儿] {name}: 极薄片 (扁平度 {_flat:.3f} < "
                      f"{flat_thre}) → 剔除 (墙面碎片, 非实物)", flush=True)
        else:
            refined_relations[name] = "supported by floor"
            refallback += 1
            if verbose:
                print(f"      [挂墙孤儿] {name}: 块状实物 (扁平度 {_flat:.3f}) → "
                      f"回退 supported by floor", flush=True)

    # ── 剔除 + 索引重映射 ──
    # 删掉同一类别中间的元素会让后面的序号前移 (curtain_3 → curtain_2),
    # 所以要先按"旧下标"分区、建立 旧→新 名字映射, 再重建 all_instances
    # 和 refined_relations, 避免关系名与实例错位。
    if drop_keys:
        _by_cat = {}
        for _n in drop_keys:
            _c, _i = index[_n]
            _by_cat.setdefault(_c, set()).add(_i)

        _new_instances = {}
        _old2new = {}          # (cat, old_idx) -> new_name
        for _c, _lst in all_instances.items():
            _keep = [j for j in range(len(_lst)) if j not in _by_cat.get(_c, set())]
            if not _keep:
                continue
            _new_instances[_c] = [_lst[j] for j in _keep]
            for _new_i, _old_i in enumerate(_keep):
                _old2new[(_c, _old_i)] = f"{_c}_{_new_i}"

        _new_rel = {}
        for _n, _r in refined_relations.items():
            _hit = None
            for (_c, _i), _new_name in _old2new.items():
                if _n == f"{_c}_{_i}":
                    _hit = _new_name
                    break
            if _hit is not None:
                _new_rel[_hit] = _r
            elif _n in _new_instances:      # 类别级关系 (键是类别名)
                _new_rel[_n] = _r
            # 其余 (被剔除的实例名) 丢弃

        all_instances = _new_instances
        refined_relations = _new_rel

    return all_instances, refined_relations, len(drop_keys), refallback


def fix_supporter_instance(all_instances, refined_relations, verbose=True,
                           min_overlap=0.30, z_tol=0.20, floor_fallback=True):
    """校验 5.1 选出的支撑物实例是否几何成立, 不成立则换成同类里最合适的。

    为什么需要: Stage 5.1 让 VLM 从候选实例列表里挑 parent。同类有多个实例时
    (实测 weixin_v8 有 3 个 table) VLM 会挑错 —— handle_0 几何上明明在
    table_1 上 (xy 完全落在 table_1 范围内, table_1 顶面 z=0.281, 物体底
    z=0.208), 但 VLM 选了 table_0 (顶面 z=0.050), 于是后面的 on_top 重贴
    按 table_0 去贴, 物体悬空 15.8cm。

    判据: 子物体底面投影与候选支撑物的重叠比例 + 两者 z 的贴合程度。
    只在"原选择明显不成立"且"同类里有明显更优"时才改, 否则保持 VLM 的选择
    (VLM 有语义信息, 几何只是校验)。

    Args:
        all_instances: {category: [instance_info, ...]}
        refined_relations: {实例名: 关系}, 原地修改
        min_overlap: 水平重叠低于此认为支撑关系不成立
        z_tol: |子物体底面 - 支撑物顶面| 超过此认为不成立
    Returns:
        (refined_relations, n_fixed)
    """
    import numpy as _np

    def _bounds(info):
        m = info["original_mesh"].copy()
        m.apply_transform(_np.asarray(info["T"], dtype=float))
        return m.bounds

    # 实例名 → (category, info, bounds)
    _idx = {}
    for _cat, _lst in all_instances.items():
        for _i, _info in enumerate(_lst):
            _idx[f"{_cat}_{_i}"] = (_cat, _info, _bounds(_info))

    def _score(child_b, sup_b):
        """几何匹配度: 水平重叠 (主) + z 贴合 (次)。"""
        ox = max(0.0, min(child_b[1][0], sup_b[1][0]) - max(child_b[0][0], sup_b[0][0]))
        oy = max(0.0, min(child_b[1][1], sup_b[1][1]) - max(child_b[0][1], sup_b[0][1]))
        area = max((child_b[1][0] - child_b[0][0]) * (child_b[1][1] - child_b[0][1]), 1e-9)
        ov = ox * oy / area
        gap = abs(float(child_b[0][2] - sup_b[1][2]))
        return ov - gap * 0.5     # z 差得越多越扣分

    n_fixed = 0
    for name, rel in list(refined_relations.items()):
        if not rel.startswith("supported by "):
            continue
        sup = rel[len("supported by "):].strip()
        if sup.lower() in ("floor", "ground"):
            # floor 坐实检查抽到 repair_floor_relations (供基础精修前独立调用)
            _r2, _n = repair_floor_relations(
                all_instances, {name: rel}, verbose=verbose, min_overlap=min_overlap)
            refined_relations.update(_r2)
            n_fixed += _n
            continue
        if sup.lower() in ("wall", "other objects", ""):
            continue
        if name not in _idx or sup not in _idx:
            continue
        _, _, cb = _idx[name]
        sup_cat, _, sb = _idx[sup]
        # 当前选择的几何分
        cur = _score(cb, sb)
        cur_ov = max(0.0, min(cb[1][0], sb[1][0]) - max(cb[0][0], sb[0][0])) * \
                 max(0.0, min(cb[1][1], sb[1][1]) - max(cb[0][1], sb[0][1])) / \
                 max((cb[1][0] - cb[0][0]) * (cb[1][1] - cb[0][1]), 1e-9)
        cur_gap = abs(float(cb[0][2] - sb[1][2]))
        if cur_ov >= min_overlap and cur_gap <= z_tol:
            continue                      # 原选择几何上成立, 尊重 VLM
        # 找同类里几何更优的实例
        best, best_score = None, cur
        for cand_name, (ccat, _, cbb) in _idx.items():
            if ccat != sup_cat or cand_name == sup:
                continue
            s = _score(cb, cbb)
            if s > best_score + 1e-6:
                best, best_score = cand_name, s
        if best is not None:
            refined_relations[name] = f"supported by {best}"
            n_fixed += 1
            if verbose:
                print(f"      [支撑物纠偏] {name}: {sup} (重叠{cur_ov*100:.0f}%, "
                      f"z差{cur_gap:.3f}m) → {best}", flush=True)
        elif floor_fallback:
            # 同类里也找不到几何成立的支撑物 → 考虑回退 floor。
            # 实测 vertical_pick_place_11: table_0 被重建到 z=[-1.17,-0.90]
            # (地下 1 米), 而 box_0/toy_0 在 z≈0, 两者水平重叠 0~1%。
            # 这种关系几何上不可能成立, 保留它会让 reseat/落地补正按错误基准
            # 计算 gap, 物体永远修不好。回退 floor 至少保证物体贴到 z=0 的
            # 一致基准上, 不会悬空或被拉到地下。
            #
            # ── 离地门限 ── 回退 floor 只对"本来就接近地面"的物体安全。
            # 悬在半空 (≥0.15m) 的物体一旦被翻成 floor, 落地补正/Phase1 会
            # 把它砸到 z=0 (实测 2chair: bag/bottle 在 0.37m 高被翻 floor 后
            # 落地, 失去与桌面的支撑语义)。远离地面的保留原关系并告警,
            # 交给 5.2/5.25 与 audit 暴露问题。
            if float(cb[0][2]) < 0.15:
                refined_relations[name] = "supported by floor"
                n_fixed += 1
                if verbose:
                    print(f"      [支撑物纠偏] {name}: {sup} 几何不成立 "
                          f"(重叠{cur_ov*100:.0f}%, z差{cur_gap:.3f}m) 且同类无替代 "
                          f"→ 回退 supported by floor (底面离地 {cb[0][2]:.3f}m)",
                          flush=True)
            elif verbose:
                print(f"      [支撑物纠偏] {name}: {sup} 几何不成立 "
                      f"(重叠{cur_ov*100:.0f}%, z差{cur_gap:.3f}m) 且同类无替代, "
                      f"但底面离地 {cb[0][2]:.3f}m ≥0.15m → 保留原关系 (防砸地)",
                      flush=True)
    return refined_relations, n_fixed


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
                # "装在容器里"只对容器成立。平板 (棋盘/桌面/托板) 顶面是承载面,
                # 不可能把东西装进去 —— 物体陷在平板里只说明它没被抬上来。
                # 判据: 支撑物高度 / 最大水平跨度 < 0.3 → 平板, 不做 inside 判定。
                # 实测 basic_pick_place_141: 棋盘 0.061m 高, 10 个棋子/遥控全在
                # z=0 (地上), 支撑物顶面 0.061 → sink=0.061 > 0.5×棋子高 0.045,
                # 全部被判 inside 跳过, 于是 10 个物体一直留在地面 (陷地 10/11)。
                _sup_horiz = max(1e-9, float(sb[1, 0] - sb[0, 0]),
                                 float(sb[1, 1] - sb[0, 1]))
                _is_flat_supporter = (sup_h / _sup_horiz) < 0.3
                if sink > inside_ratio * h and not _is_flat_supporter:
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
    # ── 两两 AABB 穿模检查 ──
    # 为什么必须补进 audit: 原 audit 只检查 z 向 gap, 完全不看 xy 重叠。
    # 实测 002_C1_Bowl-s4: audit 报 {'ok': 9} (ok 率 1.0), 但实际有 7 处真实穿模
    #   pillow_0/pillow_1 互相插入
    #   pillow_1/blanket_0、pillow_3/blanket_0 枕头插进被子
    #   blanket_0/hat_0 互相插入
    #   bed_0/bed_1、floor_0/floor_1 重复实例重叠
    # "ok 率 1.0" 因此是**不可信指标**, 会让下游 (以及我自己) 误以为场景正确。
    # 判据: 交叠体积 / 较小者体积 > 10% 且绝对交叠 > 1e-5 m³。
    # 注意不排除"物体 vs 它的支撑物": 正常贴合时 gap=0, 交叠体积为 0; 只有
    # 真陷进支撑物才会产生交叠 —— 那本来就该报出来。
    _names = [r["name"] for r in objects]
    _vols = {}
    for r in objects:
        _bb = bounds[r["name"]]
        _vols[r["name"]] = float(np.prod(np.maximum(_bb[1] - _bb[0], 1e-9)))

    # mesh 级互穿判定: AABB 重叠只是**候选**, 必须再验证顶点真的插进对方内部。
    # 为什么必须有这一层: 实测 002_C1_Bowl-s4 里 pillow_1 压在 blanket_0 上,
    # 两者 AABB 交叠 20% —— 但枕头坐在被子上时 AABB 必然重叠, mesh 却并不互穿。
    # 只用 AABB 会把"正常堆叠"误报成穿模 (7 对里有 4 对是这种假阳性)。
    # 判据: 一方顶点落在另一方内部的**体积占比** > 10%。
    def _mesh_of(_n):
        _c, _i = index[_n]
        _m = all_instances[_c][_i]["original_mesh"].copy()
        _m.apply_transform(np.asarray(all_instances[_c][_i]["T"], float))
        return _m

    def _inside_frac(_mA, _mB):
        """A 的顶点落在 B 内部的比例 (双向取大)。"""
        _best = 0.0
        for _src, _dst in ((_mA, _mB), (_mB, _mA)):
            try:
                _v = np.asarray(_src.vertices, float)
                if len(_v) == 0:
                    continue
                if len(_v) > 800:                      # 抽样, 控制耗时
                    _v = _v[np.linspace(0, len(_v) - 1, 800).astype(int)]
                # 顶点沿自身中心缩 0.5% 再判 contains:
                #   · 完全重合的盒子 → 顶点被推入对方内部, 能检出 ✅
                #   · 贴合面接触     → 顶点缩离对方, 不算穿模 ✅
                # 不收缩时 contains 对边界点返回 False, 会漏掉 100% 重合的情形。
                _c = np.asarray(_src.vertices, float).mean(axis=0)
                _v = _v + (_c - _v) * 0.005
                _ins = _dst.contains(_v)
                _best = max(_best, float(np.mean(_ins)))
            except Exception:
                continue
        return _best

    pen_pairs = []
    pen_of = {n: [] for n in _names}
    for _i in range(len(_names)):
        for _j in range(_i + 1, len(_names)):
            _a, _b = _names[_i], _names[_j]
            _ba, _bb2 = bounds[_a], bounds[_b]
            _lo = np.maximum(_ba[0], _bb2[0])
            _hi = np.minimum(_ba[1], _bb2[1])
            _inter = float(np.prod(np.clip(_hi - _lo, 0.0, None)))
            if _inter <= 1e-5:
                continue
            _frac = _inter / max(min(_vols[_a], _vols[_b]), 1e-9)
            if _frac <= 0.10:                          # AABB 都不重叠, 无需细查
                continue
            try:
                _infrac = _inside_frac(_mesh_of(_a), _mesh_of(_b))
            except Exception:
                _infrac = _frac                          # 退化: 用 AABB 比例
            if _infrac <= 0.10:
                continue                                 # AABB 重叠但 mesh 不互穿 → 正常堆叠
            pen_pairs.append({
                "a": _a, "b": _b,
                "overlap_frac": round(_frac, 3),
                "inside_frac": round(_infrac, 3),
                "overlap_vol": round(_inter, 6),
            })
            pen_of[_a].append(f"{_b}(内插{_infrac*100:.0f}%)")
            pen_of[_b].append(f"{_a}(内插{_infrac*100:.0f}%)")

    for r in objects:
        r["penetrations"] = pen_of.get(r["name"], [])
        # 只在原本 ok 的情况下降级, 不覆盖更严重的 floating/sunken
        if r["penetrations"] and r["status"] == "ok":
            r["status"] = "penetrating"
    n_bad = sum(1 for r in objects if r["status"] != "ok")

    audit = {
        "n_objects": len(objects),
        "n_problem": n_bad,
        "n_penetration_pairs": len(pen_pairs),
        "penetration_pairs": pen_pairs,
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
            # 网格约定 y-up (instance_generation 的 matrix_y2z): y 轴指向下方
            # (theta>90°) 就是物体倒置。必须翻回 +z; 旧实现 align 到 -z 是把
            # 倒置状态"对齐成完美倒置", 倒椅永远无人翻正 (实测 wxD chair_0)。
            upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, upper_real_vector)
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


def fix_orphan_instances(all_instances, refined_relations, verbose=True,
                         max_dist=1.0, gap_tol=0.05, z_ok_tol=0.05):
    """5.25 收敛循环之后: 孤儿实例处理 (VGGT 相对定位错的兜底, T5)。

    两类孤儿, 判据纯几何:

      ① "脱锚的贴附": 关系为 "supported by X" (X 具名实例), 竖向已贴合
         (|gap| < gap_tol, gap = 物体底面 z - 支撑物顶面 z), 但 xy 足迹与
         X 的重叠恰好为 0 —— 物体和支撑物在水平方向完全脱开。
         实测 002: pillow_2 在 x=[4.55,5.10], 而床在 x=[-0.94,1.82],
         水平重叠 0%, 相距 3~5m。
         → 沿 xy 夹回 X 的范围内 (与 recenter_edge_objects 同一套 clamp:
           物体中心 clamp 进"支撑物范围 内缩物体半宽"后的区间, 支撑物比
           物体还小时退化为对齐支撑物中心), z 保持不动。
         为什么不像 recenter 那样设 max_shift 护栏: recenter 的 0.35m 上限
         会把本函数要修的 3~5m 案例直接跳过 (它防的是"支撑物范围异常",
         而 |gap|<gap_tol + 重叠=0 两个条件已锁定"关系成立、位置错"),
         触发条件本身就是护栏。

      ② "离群的幽灵": 关系为 "supported by floor/ground", 底面 z 正常
         (|min_z| ≤ z_ok_tol, 即已正常落地), 但水平位置距场景中心超过
         max_dist —— VGGT 相对定位错的典型产物: 一个"实例"被丢到离整个
         场景几米外, 它不是摆放错, 是重建幻觉。
         → 标记 orphan (写进返回的 report, 不静默消失), 从 all_instances
           中剔除; 关系串里引用了被剔除实例的, 回退 "supported by floor"。
         场景中心用各实例 xy 中心的**中位数**而不是均值: 单个 5m 外的幽灵
         会把均值拉偏 5/N 米 (实例少时足以把真物体也误判成离群), 中位数对
         离群稳健。实例总数 < 4 时无法稳健地定义"离群", 全部保留。
         底面 z 不正常的远处实例不动 —— 那是坐标系未对齐, 该由落地补正/
         audit 暴露, 而不是当垃圾删掉。

    Args:
        all_instances: {category: [instance_info, ...]}, 原地修改 (① 改 T,
            ② 可能删元素并重建)
        refined_relations: {实例名: 关系}, 原地修改 (② 剔除后按旧下标重映射)
        max_dist: ② 判定 orphan 的水平距离阈值 (米)
        gap_tol: ① 认定"竖向已贴合"的 |gap| 上限 (米)
        z_ok_tol: ② 认定"底面 z 正常"的 |min_z| 上限 (米)
    Returns:
        (all_instances, refined_relations, report)
        report: {"clamped": [...], "orphans": [...], "repointed": [...],
                 "n_clamped": int, "n_removed": int, "scene_center": [x, y]}
    """
    import numpy as _np

    report = {"clamped": [], "orphans": [], "repointed": [],
              "n_clamped": 0, "n_removed": 0, "scene_center": None}
    index = _instance_index(all_instances)
    if not index:
        return all_instances, refined_relations, report

    def _bounds_of(info):
        m = info["original_mesh"].copy()
        m.apply_transform(_np.asarray(info["T"], dtype=float))
        return m.bounds

    def _rel_of(name, cat):
        return str(refined_relations.get(name, "")
                   or refined_relations.get(cat, "") or "")

    # ── ① 具名支撑物: 竖向贴合但 xy 足迹完全脱开 → 横向夹回支撑物范围 ──
    bounds = {n: _bounds_of(all_instances[c][i]) for n, (c, i) in index.items()}
    for name, (cat, idx) in index.items():
        kind, sup = _parse_supporter(_rel_of(name, cat))
        if kind != "named" or not sup or sup == name or sup not in index:
            continue
        info = all_instances[cat][idx]
        b, sb = bounds[name], bounds[sup]
        gap = float(b[0, 2]) - float(sb[1, 2])
        ov = _xy_overlap_ratio(b, sb)
        if abs(gap) >= gap_tol or ov > 0:
            continue
        cx = (b[0][0] + b[1][0]) / 2.0
        cy = (b[0][1] + b[1][1]) / 2.0
        obj_hw = (b[1][0] - b[0][0]) / 2.0
        obj_hh = (b[1][1] - b[0][1]) / 2.0
        lo_x, hi_x = sb[0][0] + obj_hw, sb[1][0] - obj_hw
        lo_y, hi_y = sb[0][1] + obj_hh, sb[1][1] - obj_hh
        scx = (sb[0][0] + sb[1][0]) / 2.0
        scy = (sb[0][1] + sb[1][1]) / 2.0
        tx = min(max(cx, lo_x), hi_x) if lo_x <= hi_x else scx
        ty = min(max(cy, lo_y), hi_y) if lo_y <= hi_y else scy
        dx, dy = tx - cx, ty - cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        T = _np.asarray(info["T"], dtype=float).copy()
        T[0, 3] += dx
        T[1, 3] += dy
        info["T"] = T
        bounds[name] = _bounds_of(info)   # 后续实例与 ② 必须看到新位置
        _dist = float(_np.hypot(dx, dy))
        report["n_clamped"] += 1
        report["clamped"].append({
            "name": name, "supporter": sup, "gap": round(gap, 4),
            "shift_xy": [round(dx, 4), round(dy, 4)], "shift": round(_dist, 4),
        })
        if verbose:
            print(f"      [orphan] {name}: 与支撑物 {sup} gap={gap:+.4f}m 但 xy 重叠 0% "
                  f"(相距 {_dist:.2f}m) → 横向夹回 (平移 ({dx:+.3f},{dy:+.3f})m, z 不动)",
                  flush=True)

    # ── ② floor/ground 且底面 z 正常但远离场景中心 → orphan, 剔除 ──
    bounds = {n: _bounds_of(all_instances[c][i]) for n, (c, i) in index.items()}
    centers = {n: ((b[0][0] + b[1][0]) / 2.0, (b[0][1] + b[1][1]) / 2.0)
               for n, b in bounds.items()}
    if len(centers) >= 4:
        _cen = _np.median(_np.asarray(list(centers.values()), dtype=float), axis=0)
        report["scene_center"] = [round(float(_cen[0]), 4), round(float(_cen[1]), 4)]
        # ── 尺度相对阈值 + 大物体保护 (P0-a) ──
        # 原实现用绝对 max_dist=1.0m 判"离群幽灵"。但 VGGT 是相对尺度, 而大型
        # 家具天然离"实例中心中位数"很远 —— 实测 002-namedrel: 真实的 bed_0
        # (2.7m 宽的床) 距中位数中心仅 1.17m, 就超过 1.0m 被判幽灵**删掉了**,
        # 直接导致 M4 ok 率 1.0 → 0.14。
        # 与 Stage4 ICP / MANIPULABLE_MAX_EXTENT / 交互位移 0.08m 是同一类错:
        # 绝对阈值用在相对坐标系里。
        #
        # 修正为"统计离群"判据 (尺度无关), 再叠一条大物体硬保护:
        #   ① 距离阈值 = max(绝对下限, 中位距离 + 3×1.4826×MAD)
        #      —— 只有真正孤零零甩在外面的极值才超线; 中位数/MAD 对单个极值
        #         本身不敏感 (床 1.17m 在众多数米内的正常分布中不算离群)。
        #   ② 物体自身水平跨度 > 全场最大实例跨度的 30% → 是家具, 硬保护不删。
        _cands = []
        for name in sorted(centers):
            cat, idx = index[name]
            rel = _rel_of(name, cat)
            if not rel.startswith("supported by "):
                continue
            if rel[len("supported by "):].strip().lower() not in ("floor", "ground"):
                continue
            b = bounds[name]
            if abs(float(b[0, 2])) > z_ok_tol:
                continue        # 底面 z 不正常 → 坐标系问题, 不删 (交给落地/audit)
            _cands.append((name, cat, b,
                           float(_np.hypot(centers[name][0] - _cen[0],
                                           centers[name][1] - _cen[1]))))
        _all_spans = [max(float(b[1][0] - b[0][0]), float(b[1][1] - b[0][1]))
                      for b in bounds.values()]
        _max_span = max(_all_spans) if _all_spans else 0.0
        _ds = _np.asarray([c[3] for c in _cands], dtype=float)
        if len(_ds) >= 3:
            _dmed = float(_np.median(_ds))
            _dmad = float(_np.median(_np.abs(_ds - _dmed)))
            _eff_dist = max(max_dist, _dmed + 3.0 * 1.4826 * max(_dmad, 1e-6))
        else:
            _dmed, _dmad = (float(_np.median(_ds)) if len(_ds) else 0.0), 0.0
            _eff_dist = max(max_dist, _dmed + 1.0)
        report["orphan_eff_dist"] = round(_eff_dist, 4)
        report["dist_median"] = round(_dmed, 4)
        report["dist_mad"] = round(_dmad, 4)
        report["max_instance_span"] = round(_max_span, 4)
        orphans = []
        for name, cat, b, d in _cands:
            # ② 大物体硬保护: 家具天然远离中心, 不能当幽灵删
            _own = max(float(b[1][0] - b[0][0]), float(b[1][1] - b[0][1]))
            if _max_span > 1e-6 and _own > 0.30 * _max_span:
                report.setdefault("kept_large", []).append(
                    {"name": name, "own_span": round(_own, 4),
                     "max_span": round(_max_span, 4)})
                if verbose:
                    print(f"      [orphan] {name}: 水平跨度 {_own:.2f}m > 全场最大 "
                          f"{_max_span:.2f}m 的 30% → 是家具, 不按幽灵剔除",
                          flush=True)
                continue
            if d <= _eff_dist:
                continue
            orphans.append({"name": name, "category": cat,
                            "min_z": round(float(b[0, 2]), 4),
                            "dist_to_center": round(d, 4),
                            "max_dist": _eff_dist})
            if verbose:
                print(f"      [orphan] {name}: floor 关系、底面 z={min_z:+.4f}m 正常, "
                      f"但距场景中心 {d:.2f}m > {max_dist:.2f}m → 标记 orphan 剔除",
                      flush=True)
        report["orphans"] = orphans

        if orphans:
            _removed = {o["name"] for o in orphans}
            # 先把引用了被剔除实例的关系回退 floor。必须在重映射**之前**做:
            # 删掉 pillow_1 后原 pillow_2 会前移成新 pillow_1, 旧名字会串号。
            for _n, _r in list(refined_relations.items()):
                if isinstance(_r, str) and _r.startswith("supported by "):
                    _sup = _r[len("supported by "):].strip()
                    if _sup in _removed:
                        refined_relations[_n] = "supported by floor"
                        report["repointed"].append(
                            {"name": _n, "old": _r, "new": "supported by floor"})
                        if verbose:
                            print(f"      [orphan] {_n}: 支撑物 {_sup} 已被剔除 "
                                  f"→ 关系回退 supported by floor", flush=True)
            # 剔除 + 索引重映射 (同 resolve_orphan_wall_relations: 删同一类别
            # 中间的元素会让后面的序号前移, 必须按旧下标分区后重建, 避免关系
            # 名与实例错位)。
            _by_cat = {}
            for _n in _removed:
                _c, _i = index[_n]
                _by_cat.setdefault(_c, set()).add(_i)
            _new_instances = {}
            _old2new = {}
            for _c, _lst in all_instances.items():
                _keep = [j for j in range(len(_lst))
                         if j not in _by_cat.get(_c, set())]
                if not _keep:
                    continue
                _new_instances[_c] = [_lst[j] for j in _keep]
                for _new_i, _old_i in enumerate(_keep):
                    _old2new[(_c, _old_i)] = f"{_c}_{_new_i}"
            _new_rel = {}
            for _n, _r in refined_relations.items():
                _hit = None
                for (_c, _i), _new_name in _old2new.items():
                    if _n == f"{_c}_{_i}":
                        _hit = _new_name
                        break
                if _hit is not None:
                    _new_rel[_hit] = _r
                elif _n in _new_instances:   # 类别级关系 (键是类别名)
                    _new_rel[_n] = _r
                # 其余 (被剔除的实例名) 丢弃

            # ── 支撑物引用重写 (P0-b) ──
            # 上面只重映射了关系的**键** (实例名), 但没有改**值**里的支撑物引用。
            # 删除同一类别的靠前元素会让后面的实例前移, 于是名字变了:
            #   bed_0 被删 → 原 bed_1 重编号为 bed_0
            #   但 pillow_0..3 / blanket_0 / hat_0 的关系里**仍写着 "bed_1"**
            #   → 指向一个已不存在的实例 → audit 全部判 unsupported。
            # 实测 002-namedrel: M4 ok 率从 1.0 掉到 0.14 (6 个 unsupported),
            # 而 docstring 承诺的"引用被剔除实例的回退 floor"只在删除前处理了
            # 直接指向被删者的情形, 覆盖不到"重编号"这一种。
            _name_remap = {f"{_c}_{_oi}": _nn
                           for (_c, _oi), _nn in _old2new.items()}
            _existing = {f"{_c}_{i}" for _c, _lst in _new_instances.items()
                         for i in range(len(_lst))}
            for _n, _r in list(_new_rel.items()):
                if not (isinstance(_r, str) and _r.startswith("supported by ")):
                    continue
                _sup = _r[len("supported by "):].strip()
                if _sup.lower() in ("floor", "ground", "wall") or not _sup:
                    continue
                if _sup in _existing:
                    continue                      # 引用仍有效, 不动
                _new_sup = _name_remap.get(_sup)
                if _new_sup and _new_sup in _existing:
                    _new_rel[_n] = f"supported by {_new_sup}"
                    report.setdefault("repointed", []).append(
                        {"name": _n, "old": _r, "new": _new_rel[_n]})
                    if verbose:
                        print(f"      [orphan] {_n}: 支撑物重编号 {_sup} → "
                              f"{_new_sup}", flush=True)
                else:
                    # 支撑物确实消失了 → 回退 floor, 不留悬空引用
                    _new_rel[_n] = "supported by floor"
                    report.setdefault("repointed", []).append(
                        {"name": _n, "old": _r, "new": "supported by floor"})
                    if verbose:
                        print(f"      [orphan] {_n}: 支撑物 {_sup} 已不存在 → "
                              f"关系回退 supported by floor", flush=True)

            all_instances = _new_instances
            refined_relations = _new_rel
            report["n_removed"] = len(orphans)
            if verbose:
                print(f"      [orphan] 共剔除 {len(orphans)} 个离群实例 "
                      f"({', '.join(sorted(_removed))})", flush=True)

    return all_instances, refined_relations, report

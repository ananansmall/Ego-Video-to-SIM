#!/usr/bin/env python3
"""管线结果量化评价 —— 输出 6 项可复现指标，取代"看渲染图猜"。

设计原则:
  1. 每项指标都能在不依赖人工判断的前提下算出
  2. 尽量以**真实视频帧的 mask** 作参照 (而非另一次重建)，因为那是唯一的一手观测
  3. 明确标注哪些指标是"有真值对照"、哪些只是"自洽性检查"

指标:
  M1 交互检出数        视频里真实有几处操作 vs 检出几个
                      HOI4D / Egodex 的 pick-place 视频期望 >=1
  M2 类别级 silhouette IoU
                     某类别所有 mesh 在该实例"最优帧"上渲染的并集
                     vs 该帧真实 SAM3 mask 并集。这是最接近真值的几何指标。
  M3 mesh/点云 尺寸比  该类别 mesh 并集 AABB / 点云 AABB。
                     显著偏离 1 说明 SAM3D 建大/建小了。
  M4 摆放到位率        audit 里 ok 的占比 + 各异常计数
  M5 穿模对数          两两 AABB 交叠超过小物体体积 10% 的对数
  M6 采样帧覆盖率      mask 覆盖帧数 / 该实例可见帧数。低 = 只被拍到几帧。

用法:
  python tools/eval_pipeline.py --scenes 112_C5_CellPhone-s4 151_C7_Plate-s4
  python tools/eval_pipeline.py --all --pattern "output_v3/*s4"
  python tools/eval_pipeline.py --scenes 112_C5_CellPhone-s4 --json out.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _resolve(name):
    """把 '112_C5_...' 解析成实际目录。"""
    if os.path.isdir(name):
        return name
    for base in ("output_v3", "output_v2"):
        p = os.path.join(REPO, base, name)
        if os.path.isdir(p):
            return p
    return None


def _world_bounds(info):
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], dtype=float))
    return m.bounds


def _penetration_pairs(inst, rel_names=None, vol_frac=0.10):
    """两两 AABB 交叠超过较小体积 10% 视为穿模。"""
    names, bounds, vols = [], [], []
    for c, lst in inst.items():
        for i, x in enumerate(lst):
            b = _world_bounds(x)
            names.append(f"{c}_{i}")
            bounds.append(b)
            vols.append(float(np.prod(np.maximum(b[1] - b[0], 1e-9))))
    n = 0
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo = np.maximum(bounds[i][0], bounds[j][0])
            hi = np.minimum(bounds[i][1], bounds[j][1])
            inter = float(np.prod(np.clip(hi - lo, 0, None)))
            if inter > vol_frac * min(vols[i], vols[j]):
                n += 1
                pairs.append(f"{names[i]}/{names[j]}")
    return n, pairs


def eval_scene(D, verbose=True):
    """返回该场景的指标 dict。任何一项算不出都记为 None 并写明原因。"""
    out = {"scene": os.path.basename(D), "dir": D}

    # ── 载入 ──
    pkl = os.path.join(D, "all_instances.pkl")
    if not os.path.isfile(pkl):
        out["error"] = "no all_instances.pkl"
        return out
    o = pickle.load(open(pkl, "rb"))
    inst = o.get("all_instances", o)
    opt_frames = o.get("all_optimal_frame_ids", {})

    rel = {}
    rp = os.path.join(D, "final_relations.json")
    if os.path.isfile(rp):
        rel = json.load(open(rp))

    out["n_instances"] = sum(len(v) for v in inst.values())
    out["categories"] = {k: len(v) for k, v in sorted(inst.items())}

    # ── M1 交互检出 ──
    pf = os.path.join(D, "pick_place_frames.json")
    if os.path.isfile(pf):
        try:
            j = json.load(open(pf))
            out["M1_n_interactions"] = len(j.get("interactions", []))
            out["M1_names"] = [it.get("instance_name") for it in j.get("interactions", [])]
        except Exception as e:
            out["M1_n_interactions"] = None
            out["M1_err"] = f"{type(e).__name__}"
    else:
        out["M1_n_interactions"] = None
        out["M1_err"] = "no pick_place_frames.json"

    # ── M4 摆放审计 ──
    ap = os.path.join(D, "placement_audit.json")
    if os.path.isfile(ap):
        try:
            a = json.load(open(ap))
            st = a.get("n_by_status", {})
            tot = sum(st.values()) or 1
            out["M4_by_status"] = st
            out["M4_ok_rate"] = round(st.get("ok", 0) / tot, 3)
        except Exception as e:
            out["M4_err"] = f"{type(e).__name__}"
    else:
        out["M4_err"] = "no placement_audit.json"

    # ── M5 穿模 ──
    # 必须复用 dump_placement_audit 的判据 (AABB 候选 + mesh contains 验证)。
    # 本脚本原来自己算 AABB 交叠, 会把"枕头压在被子上"这类正常堆叠误报成
    # 穿模 —— 实测 002 报 7 对、151 报 13 对、wxD 报 3 对, 而 audit 的
    # mesh 级结论全是 0。两套判据不一致会让指标失去可比性。
    try:
        from src.sp_refinement import dump_placement_audit
        _a = dump_placement_audit(inst, rel, output_dir=None, verbose=False)
        out["M5_n_penetration"] = int(_a.get("n_penetration_pairs", 0))
        out["M5_pairs"] = [f"{p['a']}/{p['b']}(内插{100*p.get('inside_frac',0):.0f}%)"
                           for p in (_a.get("penetration_pairs") or [])][:8]
        # 一致性守卫: 若 audit 与本地 AABB 口径差异过大, 明确标出来
        _n_abb, _ = _penetration_pairs(inst)
        if _n_abb != out["M5_n_penetration"]:
            out["M5_aabb_candidates"] = _n_abb
    except Exception as e:
        out["M5_err"] = f"{type(e).__name__}: {str(e)[:60]}"

    # ── M6 采样帧覆盖率（需要 _pre_dedup_masks.pkl）──
    pre = os.path.join(D, "_pre_dedup_masks.pkl")
    pre_am = None
    if os.path.isfile(pre):
        try:
            pre_am = pickle.load(open(pre, "rb"))["all_masks"]
        except Exception:
            pre_am = None
    if pre_am:
        cov = {}
        for c, lst in pre_am.items():
            for i, tr in enumerate(lst):
                if not tr:
                    continue
                f = sorted(m["frame_id"] for m in tr)
                cov[f"{c}_{i}"] = {"n_frames": len(f), "f0": f[0], "f_last": f[-1]}
        out["M6_coverage"] = cov
        if cov:
            out["M6_min_frames"] = min(v["n_frames"] for v in cov.values())
            out["M6_mean_frames"] = round(
                float(np.mean([v["n_frames"] for v in cov.values()])), 1)

    # ── M7 首帧正确率: 有 pick 的动态实例中, 初始帧 < pick 帧 的占比 ──
    tj_p = os.path.join(D, "dynamic_pointclouds", "trajectory_info.json")
    pp_p = os.path.join(D, "pick_place_frames.json")
    if os.path.isfile(tj_p) and os.path.isfile(pp_p):
        try:
            tj = json.load(open(tj_p))
            pp = json.load(open(pp_p))
            ratio = float(pp.get("frame_ratio") or 1.0) or 1.0
            n_have, n_ok = 0, 0
            per = []
            for key, tr in tj.items():
                evs = [e for e in pp.get("interactions", [])
                       if e.get("instance_name") == key]
                if not evs:
                    continue
                n_have += 1
                first_vis = min(tr["visible_frames"]) if tr.get("visible_frames") else None
                anchor = None
                # 与 mainv3 初始帧决策一致: 首次可见帧即锚点 (三级策略的兜底层)
                anchor = first_vis
                first_pick = min(e["pick_frame"] for e in evs) / ratio
                ok = anchor is not None and anchor < first_pick - 0.5
                n_ok += 1 if ok else 0
                per.append({"instance": key, "anchor": anchor,
                            "first_pick_slot": round(first_pick, 1), "pre_pick": ok})
            out["M7_first_frame"] = {"n_with_pick": n_have, "n_ok": n_ok,
                                     "rate": round(n_ok / max(n_have, 1), 2),
                                     "per_instance": per,
                                     "pass": bool(n_have == 0 or n_ok == n_have)}
        except Exception as e:
            out["M7_err"] = f"{type(e).__name__}: {e}"

    # world_points 只在 _pre_dedup_masks.pkl 里, M3/M2 分层诊断共用, 只载入一次
    wp = None
    if pre_am:
        try:
            wp = pickle.load(open(pre, "rb")).get("world_points")
        except Exception:
            wp = None

    # ── M3 mesh/点云 尺寸比（类别级）──
    if pre_am and wp is not None:
        ratios = {}
        for c, lst in inst.items():
            try:
                mb = np.array([_world_bounds(x) for x in lst])
                mext = np.max(mb[:, 1], axis=0) - np.min(mb[:, 0], axis=0)
                pts = []
                for tr in pre_am.get(c, []):
                    for m in tr:
                        pts.append(int(m["mask"].sum()))
                if wp is None:
                    continue
                P = []
                for tr in pre_am.get(c, []):
                    for m in tr:
                        p = wp[m["frame_id"]][m["mask"] > 0]
                        p = p[np.all(np.isfinite(p), axis=-1)]
                        if len(p):
                            P.append(p)
                if not P:
                    continue
                P = np.concatenate(P, axis=0)
                pext = P.max(0) - P.min(0)
                r = float(np.max(mext) / max(float(np.max(pext)), 1e-9))
                ratios[c] = round(r, 2)
            except Exception:
                continue
        if ratios:
            out["M3_size_ratio"] = ratios

    # ── M2 类别级 silhouette IoU ──
    try:
        out["M2_silhouette"], out["M2_layered"] = _silhouette_iou(
            D, inst, pre_am, opt_frames, wp=wp)
    except Exception as e:
        out["M2_err"] = f"{type(e).__name__}: {str(e)[:60]}"

    return out


def _iou(m_ren, mr):
    u = (m_ren | mr).sum()
    return float((m_ren & mr).sum() / u) if u > 0 else 0.0


def _attribution(iou_raw, iou_trans, iou_ts):
    """把 (1 - IoU_raw) 的总误差分解成 位置/尺度/形状 三份。

      位置误差 = 平移对齐带来的 IoU 增益
      尺度误差 = 在平移对齐之上再做尺度对齐带来的增益
      形状误差 = 尺度对齐后仍补不回来的残差
    三者之和恒等于总误差 (1 - IoU_raw), 是精确分解而非估计。
    """
    total = 1.0 - iou_raw
    if total <= 1e-6:
        return {"pos_pct": 0.0, "scale_pct": 0.0, "shape_pct": 0.0}
    return {
        "pos_pct": round(100.0 * (iou_trans - iou_raw) / total, 1),
        "scale_pct": round(100.0 * (iou_ts - iou_trans) / total, 1),
        "shape_pct": round(100.0 * (1.0 - iou_ts) / total, 1),
    }


def _load_cam(D):
    """从产物目录直接读相机参数, 不依赖任何外部模块。

    产物里已有: intrinsic.txt (3x3) / extrinsics/<frame>.txt (4x4) /
    color/<frame>.jpg。帧数取 extrinsics 与 color 的较小者。
    返回 (K, exts [N,4,4], H, W); 缺文件时返回 None。
    """
    import cv2
    ki = os.path.join(D, "intrinsic.txt")
    edir = os.path.join(D, "extrinsics")
    cdir = os.path.join(D, "color")
    if not (os.path.isfile(ki) and os.path.isdir(edir) and os.path.isdir(cdir)):
        return None
    try:
        K = np.loadtxt(ki, dtype=float)
        efs = sorted([f for f in os.listdir(edir) if f.endswith(".txt")],
                     key=lambda x: int(os.path.splitext(x)[0]))
        cfs = sorted([f for f in os.listdir(cdir) if f.endswith((".jpg", ".png"))],
                     key=lambda x: int(os.path.splitext(x)[0]))
        n = min(len(efs), len(cfs))
        if n == 0:
            return None
        exts = np.stack([np.loadtxt(os.path.join(edir, f), dtype=float)
                         for f in efs[:n]])
        im = cv2.imread(os.path.join(cdir, cfs[0]))
        if im is None:
            return None
        H, W = im.shape[:2]
        return K, exts, H, W
    except Exception:
        return None


def _silhouette_iou(D, inst, pre_am, opt_frames, wp=None):
    """类别级：该类所有 mesh 在最优帧渲染并集 vs 真实 mask 并集。

    为什么用类别级而不是实例级: 去重会合并实例（cabinet[1]+table[0]），
    实例与 pre-dedup mask 的对应关系不可靠；类别级没有这个歧义。

    T7 分层诊断: 在原有 raw IoU 之外, 对每个实例再算两种对齐后 IoU:
      ② 平移对齐   —— 把 mesh 世界系质心平移到候选 mask 点云质心 (消除位置误差)
      ③ 平移+尺度 —— 再按包围盒最大边之比均匀缩放 (消除尺度误差)
    对齐矩阵在世界系构造, 以 A @ T 传入原渲染路径 (render_mesh 先应用 T)。
    每个变体仍取"该帧同类别 mask 中 IoU 最大者", 与 raw 的 best-mask 口径一致。
    返回 (raw 结果 [与旧版完全同格式], 分层结果 dict)。
    """
    if not pre_am:
        return None, None
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # 相机参数直接读产物文件, 不再依赖旧的 stage4 模块 (Stage 4 已永久废弃)。
    from tools.silhouette_renderer import MeshRenderer

    _cam = _load_cam(D)
    if _cam is None:
        return None, None
    K, exts, H, W = _cam
    ren = MeshRenderer(K, W, H)

    res = {}
    layered = {"raw": {}, "trans": {}, "trans_scale": {}, "attribution": {}}
    for c, lst in inst.items():
        frames = opt_frames.get(c) or []
        if not frames:
            continue
        ious, ious_t, iou_s = [], [], []
        for i, x in enumerate(lst):
            # 每个实例用它自己的最优帧 (all_optimal_frame_ids 与实例顺序对齐)
            f = frames[i] if i < len(frames) else frames[0]
            if f >= len(exts):
                continue
            # 该帧同类别的候选真实 mask (与 raw best-mask 搜索范围一致)
            cands = []
            for tr in pre_am.get(c, []):
                for m in tr:
                    if m["frame_id"] != f:
                        continue
                    mr = np.asarray(m["mask"])
                    if mr.shape[:2] != (H, W):
                        continue
                    cands.append(mr > 0)
            try:
                T = np.asarray(x["T"], float)
                _, _, mk = ren.render_mesh(x["original_mesh"], T, exts[f])
            except Exception:
                continue
            m_ren = np.asarray(mk) > 0
            if m_ren.sum() == 0:
                continue
            # 取该帧同类别真实 mask 中与渲染最匹配的那个。
            # 用 max 而非并集: 去重会把多个 pre-dedup mask 合成一个实例,
            # 实例与 mask 的对应不可靠; "能否匹配到某块真实区域"是无歧义的下界,
            # 同时避免把该帧不可见的其它实例算进并集而稀释分数。
            best = max((_iou(m_ren, mr) for mr in cands), default=0.0)
            ious.append(best)
            ious_t.append(0.0)
            iou_s.append(0.0)
            if not cands:
                continue

            # ── 对齐变体: mesh 先落到世界系, 再按候选 mask 点云做平移/缩放 ──
            mw = x["original_mesh"].copy()
            mw.apply_transform(T)
            mesh_c = mw.vertices.mean(axis=0)
            mesh_ext = float(np.max(mw.bounds[1] - mw.bounds[0]))
            for use_scale, acc in ((False, ious_t), (True, iou_s)):
                best_l = 0.0
                for mr in cands:
                    try:
                        p = wp[f][mr]
                    except Exception:
                        continue
                    p = p[np.all(np.isfinite(p), axis=-1)]
                    if len(p) < 10:
                        continue
                    pc_c = p.mean(axis=0)
                    pc_ext = float(np.max(p.max(axis=0) - p.min(axis=0)))
                    A = np.eye(4)
                    if use_scale and mesh_ext > 1e-9 and pc_ext > 1e-9:
                        s = pc_ext / mesh_ext
                        A[:3, :3] = np.eye(3) * s
                        A[:3, 3] = pc_c - s * mesh_c
                    else:
                        A[:3, 3] = pc_c - mesh_c
                    try:
                        _, _, mk2 = ren.render_mesh(
                            x["original_mesh"], A @ T, exts[f])
                    except Exception:
                        continue
                    m2 = np.asarray(mk2) > 0
                    if m2.sum() == 0:
                        continue
                    best_l = max(best_l, _iou(m2, mr))
                acc[-1] = best_l
        if ious:
            res[c] = round(float(np.mean(ious)), 3)
            layered["raw"][c] = res[c]
            layered["trans"][c] = round(float(np.mean(ious_t)), 3)
            layered["trans_scale"][c] = round(float(np.mean(iou_s)), 3)
            layered["attribution"][c] = _attribution(
                float(np.mean(ious)), float(np.mean(ious_t)),
                float(np.mean(iou_s)))
    if not res:
        return None, None
    res["_mean"] = round(float(np.mean([x for x in res.values() if isinstance(x, float)])), 3)
    for k in ("raw", "trans", "trans_scale"):
        vals = [x for x in layered[k].values() if isinstance(x, float)]
        layered[k]["_mean"] = round(float(np.mean(vals)), 3) if vals else None
    att = layered["attribution"]
    layered["attribution"]["_mean"] = {
        k: round(float(np.mean([v[k] for v in att.values()])), 1)
        for k in ("pos_pct", "scale_pct", "shape_pct")} if att else {}
    return res, layered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--pattern", default="output_v3/*s4")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--t7-out", default=None,
                    help="T7 分层诊断落盘路径 (如 /tmp/t7_diagnosis.json)")
    args = ap.parse_args()

    dirs = []
    if args.scenes:
        for s in args.scenes:
            p = _resolve(s)
            if p:
                dirs.append(p)
            else:
                print(f"  ⚠️ 找不到 {s}")
    if args.all:
        dirs += sorted(glob.glob(os.path.join(REPO, args.pattern)))
        dirs += sorted(glob.glob(os.path.join(REPO, "output_v2/*rescaled")))
        dirs = sorted(set(dirs))

    rows = []
    for d in dirs:
        r = eval_scene(d)
        rows.append(r)
        print(f"\n=== {r['scene']}")
        if "error" in r:
            print(f"   ❌ {r['error']}")
            continue
        print(f"   实例 {r['n_instances']}: {r['categories']}")
        m1 = r.get("M1_n_interactions")
        print(f"   M1 交互检出: {m1 if m1 is not None else 'N/A ' + str(r.get('M1_err',''))}")
        m2 = r.get("M2_silhouette")
        if isinstance(m2, dict):
            mean = m2.pop("_mean", None)
            m2["MEAN"] = mean
            print(f"   M2 silhouette IoU: {m2}")
        else:
            print(f"   M2 silhouette IoU: N/A {r.get('M2_err','')}")
        lay = r.get("M2_layered")
        if isinstance(lay, dict) and lay.get("attribution"):
            print("   T7 分层诊断 (raw → +平移 → +平移+尺度 | 位置/尺度/形状占比):")
            att = lay["attribution"]
            for c in lay["raw"]:
                if c == "_mean" or c not in att:
                    continue
                a = att[c]
                print(f"      {c}: {lay['raw'][c]} → {lay['trans'][c]} → "
                      f"{lay['trans_scale'][c]}"
                      f"   (位置 {a['pos_pct']}% | 尺度 {a['scale_pct']}% | "
                      f"形状 {a['shape_pct']}%)")
            m = att.get("_mean", {})
            if m:
                print(f"      ⊕ 均值: raw={lay['raw'].get('_mean')} → "
                      f"+平移={lay['trans'].get('_mean')} → "
                      f"+尺度={lay['trans_scale'].get('_mean')}"
                      f"   (位置 {m.get('pos_pct')}% | 尺度 {m.get('scale_pct')}% | "
                      f"形状 {m.get('shape_pct')}%)")
        print(f"   M3 mesh/点云 尺寸比: {r.get('M3_size_ratio','N/A')}")
        print(f"   M4 摆放: {r.get('M4_by_status','N/A')} ok率={r.get('M4_ok_rate','N/A')}")
        print(f"   M5 穿模: {r.get('M5_n_penetration','N/A')} {r.get('M5_pairs','')}")
        print(f"   M6 采样帧: 最少 {r.get('M6_min_frames','N/A')} 帧, 平均 {r.get('M6_mean_frames','N/A')} 帧")
    m7 = r.get("M7_first_frame")
    if m7:
        print(f"   M7 首帧正确率: {m7['n_ok']}/{m7['n_with_pick']} (rate {m7['rate']})")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=2, default=str)
        print(f"\n已写出 {args.json}")

    if args.t7_out:
        # T7 分层诊断汇总: 逐场景逐类别三层 IoU + 归因, 末尾附跨场景均值
        t7 = {"scenes": [], "summary": None}
        agg = {"pos_pct": [], "scale_pct": [], "shape_pct": []}
        for r in rows:
            lay = r.get("M2_layered")
            if not isinstance(lay, dict) or not lay.get("attribution"):
                continue
            att = dict(lay["attribution"])
            att.pop("_mean", None)
            t7["scenes"].append({
                "scene": r["scene"],
                "raw": {k: v for k, v in lay["raw"].items() if k != "_mean"},
                "trans": {k: v for k, v in lay["trans"].items() if k != "_mean"},
                "trans_scale": {k: v for k, v in lay["trans_scale"].items() if k != "_mean"},
                "attribution": att,
            })
            for a in att.values():
                for k in agg:
                    agg[k].append(a[k])
        if t7["scenes"]:
            t7["summary"] = {k: round(float(np.mean(v)), 1)
                             for k, v in agg.items() if v}
        json.dump(t7, open(args.t7_out, "w"), ensure_ascii=False, indent=2)
        if t7["summary"]:
            s = t7["summary"]
            print(f"\n[T7 总体归因] 位置误差占 {s['pos_pct']}% | "
                  f"尺度误差占 {s['scale_pct']}% | 形状误差占 {s['shape_pct']}%"
                  f"  (共 {len(t7['scenes'])} 个场景)")
        print(f"已写出 {args.t7_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

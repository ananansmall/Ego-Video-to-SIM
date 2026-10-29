"""场景重建量化评价指标体系 (非循环: 全部对照观测数据, 不用场景自证)。

指标组:
  G 重力   G1 地面共面偏差  G2 竖直物对齐  G3 世界z轴 vs 真实帧竖直结构夹角
  P 位置   P1 回投深度偏差(米, 对照 VGGT 观测深度)  P2 多帧稳定性
  D 动态首帧 D1 锚点局部位移  D2 锚点在 pick 前?  D3 锚点高度离群
           D4 pick/place 时刻轨迹吻合
  R 关系   R1 贴合 gap  R2 足迹覆盖比
  S 尺度   S1 相对先验偏差

用法:
  python tools/eval_metrics.py --scene_dir output_v3/wxD-s4rel2 [--g3_frame N]
输出: <scene_dir>/eval_metrics.json + stdout 可读报告
"""
import argparse, json, os, pickle, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from tools.orientation_judge import SCALE_PRIOR


def _wb(info):
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], float))
    return m.bounds


def _up_z(info):
    return float((np.asarray(info["T"])[:3, :3] @ np.array([0, 1, 0.0]))[2])


# ─────────────────────── G 重力 ───────────────────────

def m_ground_plane(infos, relations):
    """G1: floor 关系实例的底面 z 的分布 (共面性)。"""
    bottoms = [float(_wb(x)[0][2]) for k, x in _iter(infos, relations, "floor")]
    if not bottoms:
        return {"metric": "G1_ground_coplanarity", "value": None, "pass": None,
                "note": "无 floor 关系实例"}
    arr = np.array(bottoms)
    dev = float(np.max(np.abs(arr - arr.mean())))
    return {"metric": "G1_ground_coplanarity", "value": round(dev, 4),
            "n": len(bottoms), "pass": bool(dev < 0.03),
            "note": f"floor 物体底面最大偏差 {dev:.3f}m (阈值 0.03)"}


def m_upright(infos, relations):
    """G2: floor 关系且非薄片实例的 up 轴与 +z 的对齐 (cos)。"""
    vals = []
    for k, x in _iter(infos, relations, "floor"):
        b = _wb(x)
        e = b[1] - b[0]
        if float(e.min()) / max(float(e.max()), 1e-9) < 0.15:
            continue                        # 薄片类跳过 (布/毯天然躺放)
        up = float((np.asarray(x["T"])[:3, :3] @ np.array([0, 1, 0.0]))[2])
        vals.append(abs(up))
    if not vals:
        return {"metric": "G2_upright_alignment", "value": None, "pass": None,
                "note": "无可用实例"}
    worst = float(min(vals))
    return {"metric": "G2_upright_alignment", "value": round(worst, 3),
            "n": len(vals), "pass": bool(worst > 0.9),
            "note": f"最差实例 |up·z| = {worst:.3f} (阈值 0.9)"}


def m_gravity_vs_frame(D, fid, infos, relations):
    """G3: 世界 z 轴投影方向 vs 真实帧竖直结构 (Hough) 的夹角。"""
    import cv2
    K = np.loadtxt(f"{D}/intrinsic.txt")
    w2c = np.loadtxt(f"{D}/extrinsics/{fid}.txt")
    frame = cv2.imread(f"{D}/color/{fid}.jpg")
    if frame is None:
        return {"metric": "G3_gravity_vs_frame", "value": None, "pass": None,
                "note": f"color/{fid}.jpg 不存在"}
    H, W = frame.shape[:2]
    # 真实帧竖直结构: Canny+Hough, 取接近竖直 (±35°内) 的线段中位角
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=H // 6, maxLineGap=8)
    real_ang = []
    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))  # 90=竖直
            if 55 <= ang <= 125:
                real_ang.append(ang)
    if not real_ang:
        return {"metric": "G3_gravity_vs_frame", "value": None, "pass": None,
                "note": "帧内无可信竖直结构"}
    real_med = float(np.median(real_ang))

    # 场景 z 轴投影: 取场景中心, 投影 z±0.4m 两点, 计算图像线段角
    allc = np.array([(d["bounds"][0] + d["bounds"][1]) / 2 for d in infos.values()])
    c = allc.mean(axis=0)

    def proj(P):
        Pc = w2c[:3, :3] @ P + w2c[:3, 3]
        if Pc[2] <= 0.01:
            return None
        uv = K @ Pc
        return (uv[0] / uv[2], uv[1] / uv[2])

    p1 = proj(c + np.array([0, 0, 0.4]))
    p2 = proj(c - np.array([0, 0, 0.4]))
    if not p1 or not p2:
        return {"metric": "G3_gravity_vs_frame", "value": None, "pass": None,
                "note": "z 轴投影不在画面内"}
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    scene_ang = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
    if dx < 0:                            # 方向归一 (上方向在画面里应向上)
        scene_ang = 180 - scene_ang if scene_ang < 90 else scene_ang
    diff = abs(scene_ang - real_med)
    diff = min(diff, 180 - diff)
    return {"metric": "G3_gravity_vs_frame", "value": round(diff, 1),
            "real_vertical_angle": round(real_med, 1),
            "scene_z_angle": round(scene_ang, 1),
            "n_real_lines": len(real_ang),
            "pass": bool(diff < 10),
            "note": f"场景z轴与真实竖直结构差 {diff:.1f}° (阈值 10°)"}


# ─────────────────────── P 位置 ───────────────────────

def m_reprojection_depth(D, fid_list, infos):
    """P1/P2: 真实位姿下场景渲染深度 vs VGGT 观测深度的中位偏差 (米)。"""
    import cv2
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import trimesh, pyrender
    K = np.loadtxt(f"{D}/intrinsic.txt")
    glb = None
    for g in ("final_scene_stage5.glb", "final_scene_fixed.glb"):
        if os.path.isfile(f"{D}/{g}"):
            glb = f"{D}/{g}"; break
    if glb is None:
        return {"metric": "P1_reprojection_depth", "value": None, "pass": None,
                "note": "无 GLB"}
    scene = trimesh.load(glb, force="scene")
    out = []
    for fid in fid_list:
        depth_png = f"{D}/depth/{fid}.png"
        w2c_f = f"{D}/extrinsics/{fid}.txt"
        if not (os.path.isfile(depth_png) and os.path.isfile(w2c_f)):
            continue
        depth_vggt = cv2.imread(depth_png, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        H, W = depth_vggt.shape
        w2c = np.loadtxt(w2c_f)
        prm = pyrender.Scene(bg_color=[255, 255, 255, 255])
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
                prm.add(pyrender.Mesh.from_trimesh(geom, smooth=False))
        cam_m = np.eye(4)
        cam_m[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
        c2w = np.linalg.inv(w2c)
        cam_m[:3, :3] = c2w[:3, :3] @ cam_m[:3, :3]
        cam_m[:3, 3] = c2w[:3, 3]
        prm.add(pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1],
                                          cx=K[0, 2], cy=K[1, 2],
                                          znear=0.01, zfar=100), pose=cam_m)
        r = pyrender.OffscreenRenderer(W, H)
        rend, depth_ren = r.render(prm)
        r.delete()
        zr = depth_ren.astype(np.float32)
        valid = (zr > 0.05) & (depth_vggt > 0.05)
        if valid.sum() < 200:
            out.append({"fid": fid, "n_px": int(valid.sum()),
                        "median_diff": None, "note": "重合像素不足"})
            continue
        diff = float(np.median(np.abs(zr[valid] - depth_vggt[valid])))
        out.append({"fid": fid, "n_px": int(valid.sum()),
                    "median_diff": round(diff, 4),
                    "pass": bool(diff < 0.05)})
    if not out:
        return {"metric": "P1_reprojection_depth", "value": None, "pass": None}
    good = [o["median_diff"] for o in out if o.get("median_diff") is not None]
    med = float(np.median(good)) if good else None
    return {"metric": "P1_reprojection_depth", "value": round(med, 4) if med else None,
            "per_frame": out, "pass": bool(med is not None and med < 0.05),
            "note": f"渲染深度 vs VGGT 观测深度中位差 {med:.3f}m (阈值 0.05)" if med else ""}


# ─────────────────────── D 动态首帧 ───────────────────────

def m_dynamic_first_frame(D, infos):
    """D1-D4: 动态物体初始帧选择的量化评价。"""
    tj_p = f"{D}/dynamic_pointclouds/trajectory_info.json"
    pp_p = f"{D}/pick_place_frames.json"
    if not os.path.isfile(tj_p):
        return {"metric": "D_dynamic_first_frame", "value": None, "pass": None,
                "note": "无动态物体"}
    tj = json.load(open(tj_p))
    pp = json.load(open(pp_p)) if os.path.isfile(pp_p) else {"interactions": []}
    ratio = float(pp.get("frame_ratio") or 1.0) or 1.0
    out = []
    for key, tr in tj.items():
        vis = tr["visible_frames"]
        cent = np.array(tr["centroids"], float)
        if len(vis) < 5:
            continue
        disp = np.linalg.norm(np.diff(cent, axis=0), axis=1)   # 帧间位移
        evs = [e for e in pp.get("interactions", [])
               if e["instance_name"] == key.replace("_inst", "_inst")]
        picks = [e["pick_frame"] / ratio for e in evs]         # 视频帧号→VGGT 槽位
        places = [e["place_frame"] / ratio for e in evs]
        anchor = vis[0]                                        # 管线的锚点 = 首次可见帧
        i_anchor = 0
        d1 = float(disp[0]) if len(disp) else 0.0              # 锚点局部位移
        # D2: 锚点是否在 pick 前
        pre_pick = bool(picks and anchor < min(picks) - 0.5)
        # D3: 锚点高度离群 — 锚点 z 与轨迹最低 25% 分位 z 差
        zq = float(np.quantile(cent[:, 2], 0.25))
        z_anchor = float(cent[0, 2])
        z_outlier = bool(z_anchor - zq > 0.10)                 # 锚点比低位段高 10cm → 手持嫌疑
        # D4: place 后轨迹是否稳定 (place 时刻后 3 帧位移)
        d4 = None
        if places:
            ip = min(range(len(vis)), key=lambda i: abs(vis[i] - places[-1]))
            if ip + 3 < len(disp):
                d4 = float(np.max(disp[ip:ip + 3]))
        out.append({
            "instance": key,
            "anchor_frame": anchor,
            "D1_anchor_disp_m": round(d1, 4),
            "D2_anchor_pre_pick": pre_pick,
            "D3_anchor_height_outlier": z_outlier,
            "D3_anchor_z": round(z_anchor, 3), "D3_low_q_z": round(zq, 3),
            "D4_post_place_disp": round(d4, 4) if d4 is not None else None,
            "n_picks": len(picks),
        })
    # 总评: 锚点不健康 = (有 pick 且锚点不在 pick 前) 或 高度离群 或 锚点位移大
    bad = [o for o in out if (not o["D2_anchor_pre_pick"] and o["n_picks"] > 0)
           or o["D3_anchor_height_outlier"] or o["D1_anchor_disp_m"] > 0.05]
    return {"metric": "D_dynamic_first_frame", "instances": out,
            "n_dynamic": len(out), "n_unhealthy": len(bad),
            "pass": bool(len(out) > 0 and len(bad) == 0),
            "note": "; ".join(f"{o['instance']}: 锚点位移{o['D1_anchor_disp_m']}m "
                              f"{'手持嫌疑' if o['D3_anchor_height_outlier'] else ''}"
                              for o in bad) or "全部健康"}


# ─────────────────────── R 关系 ───────────────────────

def m_relations(infos, relations):
    """R1 贴合 gap / R2 足迹覆盖比 (具名支撑关系)。"""
    out = []
    for name, rel in relations.items():
        if not rel.startswith("supported by "):
            continue
        sup = rel[len("supported by "):].strip()
        if sup in ("floor", "ground", "wall", "other objects", "") or sup not in infos:
            continue
        if name not in infos:
            continue
        b, sb = infos[name]["bounds"], infos[sup]["bounds"]
        gap = float(b[0][2] - sb[1][2])
        ox = min(b[1][0], sb[1][0]) - max(b[0][0], sb[0][0])
        oy = min(b[1][1], sb[1][1]) - max(b[0][1], sb[0][1])
        cover = 0.0
        ca = max((b[1][0]-b[0][0])*(b[1][1]-b[0][1]), 1e-9)
        if ox > 0 and oy > 0:
            cover = ox * oy / ca
        ok = abs(gap) <= 0.03 and cover >= 0.5
        out.append({"instance": name, "supporter": sup, "gap": round(gap, 3),
                    "footprint_cover": round(cover, 2), "pass": bool(ok)})
    bad = [o for o in out if not o["pass"]]
    return {"metric": "R_relations", "instances": out, "n": len(out),
            "pass": bool(out and len(bad) == 0),
            "note": "; ".join(f"{o['instance']}: gap{o['gap']} cover{o['footprint_cover']}"
                              for o in bad) or "全部贴合"}


# ─────────────────────── S 尺度 ───────────────────────

def m_scale(infos):
    """S1: 有先验类别相对先验的偏差。"""
    out = []
    for name, d in infos.items():
        cat = d["cat"].lower().replace(" ", "_")
        prior = SCALE_PRIOR.get(cat)
        if not prior:
            continue
        mx = float(np.max(d["bounds"][1] - d["bounds"][0]))
        ratio = mx / prior
        if ratio > 2.0 or ratio < 0.5:
            out.append({"instance": name, "max_ext": round(mx, 3),
                        "prior": prior, "ratio": round(ratio, 2)})
    return {"metric": "S1_scale_vs_prior", "outliers": out,
            "pass": bool(len(out) == 0),
            "note": "; ".join(f"{o['instance']}: {o['ratio']}×先验" for o in out) or "无离群"}


# ─────────────────────── 汇总 ───────────────────────

def _iter(infos, relations, want_sup):
    for name, d in infos.items():
        rel = relations.get(name, "")
        sup = rel[len("supported by "):].strip() if rel.startswith("supported by ") else ""
        if want_sup == "floor" and sup in ("floor", "ground"):
            yield name, x_from(infos, name)
        elif want_sup == "other" and sup == "other objects":
            yield name, x_from(infos, name)

def x_from(infos, name):
    # 兼容: infos 存的是 info dict
    return infos[name]["info"]


def eval_scene(D, g3_frame=None):
    blob = pickle.load(open(f"{D}/all_instances.pkl", "rb"))
    inst = blob["all_instances"] if isinstance(blob.get("all_instances"), dict) else blob
    rel_p = f"{D}/final_relations.json"
    relations = json.load(open(rel_p)) if os.path.isfile(rel_p) else \
        json.load(open(f"{D}/relations_scene_graph.json")).get("refined_relations", {})
    infos = {}
    for cat, lst in inst.items():
        for i, x in enumerate(lst):
            infos[f"{cat}_{i}"] = {"cat": cat, "info": x, "bounds": _wb(x),
                                   "up": _up_z(x)}

    M = {}
    M["G1"] = m_ground_plane(infos, relations)
    M["G2"] = m_upright(infos, relations)
    if g3_frame is not None:
        M["G3"] = m_gravity_vs_frame(D, g3_frame, infos, relations)
    fid_list = sorted(int(f.split(".")[0]) for f in os.listdir(f"{D}/depth")
                      if f.endswith(".png"))[::25][:4]
    M["P"] = m_reprojection_depth(D, fid_list, infos)
    M["D"] = m_dynamic_first_frame(D, infos)
    M["R"] = m_relations(infos, relations)
    M["S"] = m_scale(infos)

    passed = [k for k, v in M.items() if v.get("pass") is True]
    failed = [k for k, v in M.items() if v.get("pass") is False]
    na = [k for k, v in M.items() if v.get("pass") is None]
    summary = {"scene": D, "pass": passed, "fail": failed, "na": na,
               "score": f"{len(passed)}/{len(passed)+len(failed)}"}
    json.dump({"summary": summary, "metrics": M},
              open(f"{D}/eval_metrics.json", "w"), ensure_ascii=False, indent=1, default=str)
    return summary, M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, nargs="+")
    ap.add_argument("--g3_frame", type=int, default=None,
                    help="G3 使用的真实帧号 (默认自动取中间帧)")
    a = ap.parse_args()
    for D in a.scene_dir:
        gf = a.g3_frame
        if gf is None and os.path.isdir(f"{D}/color"):
            fs = sorted(int(f.split(".")[0]) for f in os.listdir(f"{D}/color") if f.endswith(".jpg"))
            gf = fs[len(fs)//2] if fs else None
        summary, M = eval_scene(D, g3_frame=gf)
        print(f"\n===== {D}")
        for k in ("G1", "G2", "G3", "P", "D", "R", "S"):
            m = M.get(k)
            if not m: continue
            mark = {"True": "✅", "False": "❌", "None": "➖"}[str(m.get("pass"))]
            print(f"  {mark} {k}: {m.get('note') or m.get('note','')}")
            for kk, vv in m.items():
                if kk not in ("metric", "note", "pass", "instances", "per_frame") and vv is not None:
                    print(f"      {kk}: {vv}")
            if m.get("instances"):
                for o in m["instances"]:
                    print(f"      - {o}")

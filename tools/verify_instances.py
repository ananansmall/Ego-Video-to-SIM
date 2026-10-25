"""逐实例体检: 朝向/落地/贴合/尺度/穿模 逐项打分, 按场景+类别汇总。

读 all_instances.pkl + final_relations.json, 输出:
  - 每实例一行: 状态位 + 关键数值
  - 场景级汇总 → <scene>/instance_report.json
  - 全库汇总 (打印)
用法: python verify_instances.py <scene_dir> [...] """
import sys, os, json, pickle, glob
import numpy as np

REPO = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene"
sys.path.insert(0, REPO)
from tools.orientation_judge import SCALE_PRIOR, UP_IRRELEVANT

GAP_TOL = 0.03      # 贴合容差
SUNK_TOL = -0.03    # 沉入容差
SCALE_LO, SCALE_HI = 0.5, 2.0   # 相对类别先验的倍数范围


def verify_scene(D):
    blob = pickle.load(open(f"{D}/all_instances.pkl", "rb"))
    inst = blob["all_instances"] if isinstance(blob.get("all_instances"), dict) else blob
    rp = f"{D}/final_relations.json"
    rel = json.load(open(rp)) if os.path.isfile(rp) else {}

    infos = {}
    for cat, lst in inst.items():
        for i, x in enumerate(lst):
            m = x["original_mesh"].copy()
            m.apply_transform(np.asarray(x["T"], float))
            infos[f"{cat}_{i}"] = {
                "cat": cat, "bounds": m.bounds,
                "up": float((np.asarray(x["T"])[:3, :3] @ np.array([0, 1, 0.]))[2]),
            }

    rows = []
    for name, d in infos.items():
        cat, b = d["cat"], d["bounds"]
        r = rel.get(name, "")
        sup = r[len("supported by "):].strip() if r.startswith("supported by ") else ""
        ext = b[1] - b[0]
        flags = []

        # 朝向
        cat_std = cat.lower().replace(" ", "_")
        if d["up"] < -0.3 and cat_std not in UP_IRRELEVANT and "attached to wall" not in r:
            flags.append("倒置")
        # 落地/贴合
        if sup in ("floor", "ground", ""):
            if float(b[0][2]) > GAP_TOL:
                flags.append(f"悬空{float(b[0][2]):.2f}m")
            elif float(b[0][2]) < SUNK_TOL:
                flags.append(f"沉地{float(b[0][2]):.2f}m")
        elif sup in infos:
            sb = infos[sup]["bounds"]
            gap = float(b[0][2] - sb[1][2])
            ox = min(b[1][0], sb[1][0]) - max(b[0][0], sb[0][0])
            oy = min(b[1][1], sb[1][1]) - max(b[0][1], sb[0][1])
            if ox > 0.02 and oy > 0.02:
                if gap > GAP_TOL:
                    flags.append(f"离支撑{gap:.2f}m")
                elif gap < -0.05:
                    flags.append(f"陷入支撑{gap:.2f}m")
            else:
                flags.append("不在支撑足迹内")
        # 尺度
        prior = SCALE_PRIOR.get(cat_std)
        mx = float(np.max(ext))
        if prior and not (SCALE_LO * prior <= mx <= SCALE_HI * prior):
            flags.append(f"尺度{mx:.2f}m(先验{prior})")
        # 穿模 (对, 两两 AABB 交)
        for name2, d2 in infos.items():
            if name2 <= name:
                continue
            b2 = d2["bounds"]
            oz = min(b[1][2], b2[1][2]) - max(b[0][2], b2[0][2])
            if oz <= 0.02:
                continue
            ox = min(b[1][0], b2[1][0]) - max(b[0][0], b2[0][0])
            oy = min(b[1][1], b2[1][1]) - max(b[0][1], b2[0][1])
            if ox > 0.02 and oy > 0.02:
                flags.append(f"与{name2}穿模")
        rows.append({"instance": name, "up": round(d["up"], 2),
                     "z": [round(float(b[0][2]), 3), round(float(b[1][2]), 3)],
                     "max_ext": round(mx, 2), "relation": r,
                     "ok": len(flags) == 0, "flags": flags})

    rep = {"scene": D, "n": len(rows),
           "n_ok": sum(1 for x in rows if x["ok"]),
           "instances": rows}
    json.dump(rep, open(f"{D}/instance_report.json", "w"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    scenes = sys.argv[1:]
    all_rows, tot, tot_ok = [], 0, 0
    cat_stat = {}
    for D in scenes:
        try:
            rep = verify_scene(D)
        except Exception as e:
            print(f"!! {D}: {type(e).__name__}: {e}")
            continue
        tot += rep["n"]; tot_ok += rep["n_ok"]
        bad = [x for x in rep["instances"] if not x["ok"]]
        print(f"{D.split('/')[-1]:36s} {rep['n_ok']}/{rep['n']} ok" +
              ("" if not bad else "  |  " + "; ".join(
                  f"{x['instance']}({','.join(x['flags'])})" for x in bad)))
        for x in rep["instances"]:
            c = x["instance"].rsplit("_", 1)[0]
            s = cat_stat.setdefault(c, {"n": 0, "ok": 0})
            s["n"] += 1; s["ok"] += x["ok"]
        all_rows += rep["instances"]
    print(f"\n=== 全库: {tot_ok}/{tot} 实例通过 ({tot_ok/max(tot,1)*100:.0f}%) ===")
    print("\n=== 按类别 ===")
    for c in sorted(cat_stat, key=lambda c: cat_stat[c]["ok"]/max(cat_stat[c]["n"],1)):
        s = cat_stat[c]
        print(f"  {c:22s} {s['ok']:3d}/{s['n']:<3d} ({s['ok']/max(s['n'],1)*100:.0f}%)")

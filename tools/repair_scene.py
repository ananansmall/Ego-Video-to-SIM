#!/usr/bin/env python3
"""场景修复: 把动态物体摆到最可能的支撑面上, 输出可直接看的 GLB。

针对 weixin_v8 暴露的问题:
  1. 同类多实例时 VLM 选错支撑物 (table_0 而物体几何在 table_1 上)
  2. 物体 z 与所有桌面顶面都不匹配 (重建本身空间不一致)
  3. inside_ratio 判据把"陷进盒状 table"误判成容器内物体而跳过

做法 (确定性, 不依赖 VLM):
  - 从 all_instances 里挑"最平的大表面"当支撑面 (扁平度 = 最薄维/最大维)
  - 把动态物体 (pick/place 交互涉及的实例) 的 xy 落到该面范围内,
    z 对齐到该面顶面
  - 其余实例保持不动
"""
import os
import pickle
import sys

import numpy as np

REPO = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene"
sys.path.insert(0, REPO)

SCENE = sys.argv[1] if len(sys.argv) > 1 else "weixin_v8"
D = os.path.join(REPO, "output_v3", SCENE)


def bounds(info):
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], dtype=float))
    return m.bounds


def flatness(b):
    e = b[1] - b[0]
    return float(min(e) / max(e))


def main():
    inst = pickle.load(open(os.path.join(D, "all_instances.pkl"), "rb"))["all_instances"]
    rel = {}
    rp = os.path.join(D, "final_relations.json")
    if os.path.isfile(rp):
        import json
        rel = json.load(open(rp))

    # 1) 动态实例 = 出现在交互里的
    # 注意命名不一致: 交互记录用 "{cat}_inst{i}", 而实例名是 "{cat}_{i}"
    # (实测 weixin_v8: interaction=table_inst1, 实例名=table_1)。
    # 另外实测 detector 会挑错对象 (把 table_inst1 当成被抓取的, 而真正被抓的
    # 是 handle_0 —— 它的 mask 起点 51 正好等于 pick=191/3.75≈51)。
    # 所以这里不盲信 detector: 用"mask 起点最接近 pick 帧"来复核。
    dyn = set()
    pp = os.path.join(D, "pick_place_frames.json")
    pick_rgb = None
    if os.path.isfile(pp):
        import json
        _pp = json.load(open(pp))
        _ratio = float(_pp.get("frame_ratio") or 1.0) or 1.0
        _picks = [it["pick_frame"] for it in _pp.get("interactions", []) if it.get("pick_frame")]
        if _picks:
            pick_rgb = min(_picks) / _ratio
        for it in _pp.get("interactions", []):
            nm = it.get("instance_name") or ""
            if nm:
                dyn.add(nm)
                if "_inst" in nm:
                    dyn.add(nm.replace("_inst", "_"))
    print(f"detector 报的动态实例: {sorted(dyn) or '(无)'}  pick_rgb={pick_rgb}")

    # 复核: 若 detector 报的对象其 mask 起点离 pick 很远, 而另有实例的 mask
    # 起点几乎等于 pick, 则改用后者 (它才是被抓的物体)。
    if pick_rgb is not None:
        _starts = {}
        for cat, lst in inst.items():
            for i, info in enumerate(lst):
                pass
        _pre = os.path.join(D, "_pre_dedup_masks.pkl")
        if os.path.isfile(_pre):
            _am = pickle.load(open(_pre, "rb"))["all_masks"]
            for cat, trajs in _am.items():
                for i, tr in enumerate(trajs):
                    fids = sorted(m["frame_id"] for m in tr)
                    if fids:
                        _starts[f"{cat}_{i}"] = fids[0]
        if _starts:
            _best = min(_starts.items(), key=lambda kv: abs(kv[1] - pick_rgb))
            _det_start = None
            for _n in dyn:
                if _n in _starts:
                    _det_start = _starts[_n]
                    break
            print(f"  各实例 mask 起点: {dict(sorted(_starts.items(), key=lambda x: x[1]))}")
            print(f"  detector 报的对象 mask 起点={_det_start}, "
                  f"最接近 pick({pick_rgb:.0f}) 的是 {_best[0]} (起点 {_best[1]})")
            if _best[0] not in dyn:
                # 复核认为 detector 挑错了 → 完全替换, 否则错对象(如 table_1)
                # 也会被一起搬动, 把场景弄乱
                dyn = {_best[0]}
                print(f"  → 替换动态物体为 {_best[0]} (detector 原报的有误)")

    # 2) 找最平的大表面 (候选支撑面)
    cands = []
    for cat, lst in inst.items():
        for i, info in enumerate(lst):
            b = bounds(info)
            e = b[1] - b[0]
            if max(e[0], e[1]) < 0.25:      # 太小不算"面"
                continue
            cands.append((f"{cat}_{i}", flatness(b), b))
    cands.sort(key=lambda x: x[1])
    print("候选支撑面 (按扁平度):")
    for n, f, b in cands[:4]:
        print(f"  {n:12s} 扁平度={f:.3f} 尺寸={(b[1]-b[0]).round(3).tolist()} 顶z={b[1][2]:+.3f}")
    if not cands:
        print("没有候选支撑面"); return
    surf_name, _, sb = cands[0]
    print(f"→ 选定支撑面: {surf_name}")

    # 3) 动态物体落到该面上
    out = {}
    for cat, lst in inst.items():
        new_lst = []
        for i, info in enumerate(lst):
            name = f"{cat}_{i}"
            b = bounds(info)
            if name in dyn:
                # xy: 物体中心夹到支撑面范围内 (留半个身位)
                hw = (b[1][0] - b[0][0]) / 2
                hh = (b[1][1] - b[0][1]) / 2
                cx = (b[0][0] + b[1][0]) / 2
                cy = (b[0][1] + b[1][1]) / 2
                tx = min(max(cx, sb[0][0] + hw), sb[1][0] - hw) if sb[0][0] + hw <= sb[1][0] - hw else (sb[0][0] + sb[1][0]) / 2
                ty = min(max(cy, sb[0][1] + hh), sb[1][1] - hh) if sb[0][1] + hh <= sb[1][1] - hh else (sb[0][1] + sb[1][1]) / 2
                T = np.asarray(info["T"], dtype=float).copy()
                T[0, 3] += tx - cx
                T[1, 3] += ty - cy
                T[2, 3] += float(sb[1][2] - b[0][2])     # 底面贴到支撑面顶面
                info["T"] = T
                nb = bounds(info)
                print(f"  {name}: xy 移到 ({tx:+.3f},{ty:+.3f}), "
                      f"z 底面 {b[0][2]:+.3f} → {nb[0][2]:+.3f} (贴到 {surf_name} 顶面)")
                if name in rel:
                    rel[name] = f"supported by {surf_name}"
            new_lst.append(info)
        out[cat] = new_lst

    # 4) 落盘
    with open(os.path.join(D, "all_instances.pkl"), "wb") as f:
        pickle.dump({"all_instances": out}, f)
    import json
    with open(os.path.join(D, "final_relations.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2, ensure_ascii=False)
    print(f"已写回 {D}/all_instances.pkl + final_relations.json")

    # 5) 导出 GLB
    try:
        from mainv3 import save_final_glb
        save_final_glb(out, D, filename="final_scene_repaired.glb")
        print("已导出 final_scene_repaired.glb")
    except Exception as e:
        print(f"GLB 导出失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

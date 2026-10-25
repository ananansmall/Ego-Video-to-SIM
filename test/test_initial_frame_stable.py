#!/usr/bin/env python3
"""初始帧选择: pick 前最长稳定段的中位数 (而非最后一帧) 回归测试。

背景: mainv3.py 选动态物体初始帧时, 老逻辑取 `max(fid for fid < pick_rgb)`
即 pick 前最后一帧。实测 basic_pick_place_17 的 toy_inst0 在 frame 21 有
单帧离群 (z 从 +0.06 跳到 +0.24, 下一帧立刻回落 +0.08), pick=29, 老逻辑
正好命中 frame 21 → 初始位置被放到错误高度。

修法: 在 pick 前找最长"稳定段" (相邻帧质心位移 <= 0.02m 视为静止),
取该段中位数帧。离群帧自成一格 (与前后位移都大), 不会被选为中位。

本测试锁住:
  1. 单帧离群不会被选为初始帧 (核心 case, 复刻 17 的 toy_inst0)
  2. 全程平稳时取稳定段中位 (不是第一帧也不是最后一帧)
  3. pick 前有多次抖动时, 选最长稳定段
  4. pick 前完全无稳定段时退回最后一帧
  5. 只有 1 帧时不崩
  6. pick 前为空 (pick 早于首帧) 返回 None
  7. 真实数据 basic_pick_place_17 验证

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_initial_frame_stable.py
"""
import os
import sys
import numpy as np

_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


def pick_initial_frame(centroid_data, pick_rgb, stable_disp=0.02, min_seg=3):
    """复刻 mainv3.py 的初始帧选择逻辑 (抽成纯函数便于测试)。"""
    before = [(fid, c) for fid, c in centroid_data if fid < pick_rgb]
    if not before:
        return None, "empty"
    best_seg, cur = [], [before[0]]
    for k in range(1, len(before)):
        d = float(np.linalg.norm(before[k][1] - before[k - 1][1]))
        if d <= stable_disp:
            cur.append(before[k])
        else:
            if len(cur) > len(best_seg):
                best_seg = cur
            cur = [before[k]]
    if len(cur) > len(best_seg):
        best_seg = cur
    if len(best_seg) >= min_seg:
        return best_seg[len(best_seg) // 2][0], "stable"
    return before[-1][0], "fallback"


def flat(n, z, start=0):
    return [(start + i, np.array([0.0, 0.0, z])) for i in range(n)]


print("\n[1] 单帧离群不被选中 (复刻 basic_pick_place_17 toy_inst0)")
cd = flat(21, 0.01) + [(21, np.array([0., 0., 0.24]))] + flat(7, 0.01, start=22)
f, how = pick_initial_frame(cd, 29)
check("选中稳定段而非离群帧", f != 21, f"选中 frame {f}")
check("落在平稳区 [0,20]", 0 <= f <= 20, f"frame {f}")
check("方式 = stable", how == "stable")

print("\n[2] 全程平稳取稳定段中位")
cd = flat(30, 0.05)
f, how = pick_initial_frame(cd, 30)
check("取中位 (14 或 15)", f in (14, 15), f"frame {f}")

print("\n[3] pick 前多次抖动, 选最长稳定段")
cd = (flat(5, 0.01)
      + [(5, np.array([0., 0., 0.3]))]
      + flat(3, 0.01, start=6)
      + [(9, np.array([0., 0., 0.4]))]
      + flat(20, 0.01, start=10))
f, how = pick_initial_frame(cd, 35)
check("落在最长段 [10,29]", 10 <= f <= 29, f"frame {f}")

print("\n[4] pick 前完全无稳定段 → 退回最后一帧")
cd = [(i, np.array([0., 0., 0.1 * i])) for i in range(10)]
f, how = pick_initial_frame(cd, 10)
check("退回最后一帧 (9)", f == 9, f"frame {f}")
check("方式 = fallback", how == "fallback")

print("\n[5] 只有 1 帧不崩")
f, how = pick_initial_frame([(0, np.array([0., 0., 0.02]))], 5)
check("返回 frame 0", f == 0, f"frame {f}")

print("\n[6] pick 前为空 (pick 早于首帧)")
f, how = pick_initial_frame(flat(5, 0.01, start=10), 3)
check("返回 None", f is None, f"frame {f}")

print("\n[7] 真实数据 basic_pick_place_17 的 toy_inst0")
p = ("/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/"
     "output_v3/basic_pick_place_17/_pre_dedup_masks.pkl")
if os.path.isfile(p):
    import pickle
    d = pickle.load(open(p, "rb"))
    masks, wp = d["all_masks"], d["world_points"]
    conf = d.get("world_points_conf")
    if conf is None:
        conf = np.ones_like(wp[..., :1])
    traj = []
    for im in sorted(masks["toy"][0], key=lambda x: x["frame_id"]):
        fid = im["frame_id"]
        m = im["mask"] > 0
        if not m.any():
            continue
        pts = wp[fid][m]
        c = np.asarray(conf[fid][m]).reshape(-1)
        fin = np.all(np.isfinite(pts), axis=-1)
        if not fin.any():
            continue
        pts, c = pts[fin], c[fin]
        th = np.percentile(c, 10)
        sel = c > th
        if not sel.any():
            sel = np.ones(len(c), bool)
        traj.append((fid, np.average(pts[sel], axis=0, weights=c[sel])))
    f, how = pick_initial_frame(traj, 29)
    z_sel = dict(traj).get(f)
    check("未选中离群帧 21", f != 21, f"选中 frame {f}, z={z_sel[2]:+.3f}")
    check("选中帧 z 接近首帧 z", abs(z_sel[2] - traj[0][1][2]) < 0.05,
          f"z={z_sel[2]:+.3f} vs 首帧 {traj[0][1][2]:+.3f}")
    print(f"      (轨迹 {len(traj)} 帧, 初始帧={f}, z={z_sel[2]:+.3f})")
else:
    print("    (pkl 不存在, 跳过)")

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)

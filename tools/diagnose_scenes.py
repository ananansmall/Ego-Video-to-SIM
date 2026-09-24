"""多场景批量诊断: mesh vs 点云尺寸 + 质心漂移检测。"""
import os
import pickle
import sys

import numpy as np

ROOT = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2"
PCT_LO, PCT_HI = 0.5, 99.5


def load(scene):
    with open(f"{ROOT}/{scene}/_pre_dedup_masks.pkl", "rb") as f:
        pre = pickle.load(f)
    with open(f"{ROOT}/{scene}/all_instances.pkl", "rb") as f:
        inst = pickle.load(f)
    return pre, inst


def match(pre_cat, opt_f):
    best_i, best_a = None, -1
    for i, entries in enumerate(pre_cat):
        for e in entries:
            if e["frame_id"] == opt_f:
                a = int(e["mask"].sum())
                if a > best_a:
                    best_a, best_i = a, i
    return best_i, best_a


def ext_entries(wp, entries, frames=None):
    los, his = [], []
    for e in entries:
        if frames and e["frame_id"] not in frames:
            continue
        m = e["mask"] > 0.5
        if m.sum() < 500:
            continue
        pts = wp[e["frame_id"]][m]
        los.append(np.percentile(pts, PCT_LO, 0))
        his.append(np.percentile(pts, PCT_HI, 0))
    if not los:
        return None
    return np.array(his).max(0) - np.array(los).min(0)


def mesh_world_ext(mesh, T):
    v = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    return (T @ v.T).T[:, :3].max(0) - (T @ v.T).T[:, :3].min(0)


def motion_check(wp, entries):
    """返回质心位移范围 (max_disp, num_frames)。"""
    cents = []
    for e in entries:
        m = e["mask"] > 0.5
        if m.sum() < 200:
            continue
        cents.append(wp[e["frame_id"]][m].mean(0))
    if len(cents) < 2:
        return 0.0, len(cents)
    c = np.array(cents)
    return float(c.max(0).max() - c.min(0).min()), len(cents)


def main():
    scenes = sys.argv[1:]
    for scene in scenes:
        print("=" * 78)
        print(f"  场景: {scene}")
        print("=" * 78)
        pre, inst = load(scene)
        wp = pre["world_points"]

        all_ok = True
        for cat, obj_insts in inst["all_instances"].items():
            pre_cat = pre["all_masks"].get(cat, [])
            opts = inst["all_optimal_frame_ids"].get(cat, [])
            for ii, c in enumerate(obj_insts):
                T, mesh = c["T"], c["original_mesh"]
                opt_f = opts[ii] if ii < len(opts) else None
                pi, _ = match(pre_cat, opt_f)
                mext = mesh_world_ext(mesh, T)
                entries = pre_cat[pi] if pi is not None else []
                pext = ext_entries(wp, entries) if entries else None
                disp, nframes = motion_check(wp, entries)
                is_moved = disp > 0.15

                ratio_med = float(np.median(mext / np.maximum(pext, 1e-6))) if pext is not None else None
                scale_factor = float(np.median([np.linalg.norm(T[:3, a]) for a in range(3)]))

                flag = ""
                if ratio_med and ratio_med > 1.4:
                    flag += " ⚠️MESH_BIG"
                    all_ok = False
                if ratio_med and ratio_med < 0.6:
                    flag += " ⚠️MESH_SMALL"
                    all_ok = False
                if is_moved:
                    flag += f" ⚠️MOVED{disp:.2f}m"
                    all_ok = False

                if pext is not None:
                    pc_str = tuple(round(v, 3) for v in pext)
                    r_str = f"{ratio_med:.2f}"
                else:
                    pc_str = "-"
                    r_str = "-"
                print(f"  {cat}_{ii} opt={opt_f} frames={nframes} "
                      f"mesh={tuple(round(v, 3) for v in mext)}  "
                      f"pc={pc_str}  ratio={r_str}  "
                      f"disp={disp:.3f}m  scale={scale_factor:.3f}{flag}")
        print()
        print("  判定: " + ("✅ 该场景无异常" if all_ok else "❌ 存在上述问题"))
        print()


if __name__ == "__main__":
    main()

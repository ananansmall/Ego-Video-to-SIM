"""★ 2026-09-24 【方案 §11.4】净空-可达性**几何扫描** (纯几何 ✓ 不步进物理 ✓)

目的 ✓ (复审 §11.4): 回答一个**有限的问题**:
    "在贴桌(净空 0~3mm)条件下, 是否存在让**两指同时触及目标侧壁**的抓取位姿?"
输出三选一 ✓ (§11.4 原文):
    FOUND_GEOMETRIC_CANDIDATE / NO_CANDIDATE_IN_SAMPLED_DOMAIN / PROVEN_INFEASIBLE

素材 (全部离线 ✓, 不依赖 SAPIEN ✓):
    · 目标 CoACD 凸包缓存: trimesh/agent_coacd/**/*.np* ✓
    · 手指碰撞几何: models/**/ *_gripper_finger_link{1,2}.STL ✓
    · 锚点位姿:      best_grasp.npz (pos / R / gripper_qpos ✓)

方法 ✓:
    · 距离用 **GJK** (与 Stage1 同款 ✓ 不是我另写一套近似 ✗)
    · 扫描: 锚点平移 {±5, ±2.5, 0}mm 各轴 × 开合 q {q_grasp, q_grasp+1, +2}mm ✓
    · 对每个净空档 {0, 0.5, 1, 2, 3}mm 报告最好的 `max(d_left, d_right)` ✓ (§11.4-3 ✓)

⚠ 已知近似 (诚实标注 ✓, 不得据此宣称"真实碰撞几何扫描" ✗):
    · 两指按"对称镜像 + 半间距 GAP0/2 + q"放置 ✓ (未解析 URDF 的 link origin ✗)
    · 指腹姿态取世界系下的**锚点姿态** ✓ (未做关节级正向运动学 ✗)
    ⇒ 结论只作**可达性初判** ✓; 命中候选仍须回动力学场景复核 ✓ (§11.4-5 ✓)
"""

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

GAP0 = 0.030          # 指缝基准 (m) — 与 physics_utils.FINGER_GAP0 同源口径 ✓
TABLE_Z = 0.0


def gjk_gap_mm(A, B, iters=64, eps=1e-12):
    """两个凸体 (顶点云) 的真实表面间隙 (mm) ⇒ 返回 (gap_mm, hit) ✓"""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.size == 0 or B.size == 0:
        return None, False

    def sup(P, d):
        return P[int(np.argmax(P @ d))]

    def closest(W):
        if len(W) == 1:
            return W[0], list(W)
        if len(W) == 2:
            a, b = W[0], W[1]
            ab = b - a
            t = float(np.dot(-a, ab) / max(float(np.dot(ab, ab)), eps))
            if t <= 0.0:
                return a, [a]
            if t >= 1.0:
                return b, [b]
            return a + t * ab, list(W)
        a, b, c = W[0], W[1], W[2]
        ab, ac = b - a, c - a
        n = np.cross(ab, ac)
        nn = float(np.dot(n, n))
        if nn > eps:
            k = float(np.dot(a, n)) / nn
            p = n * k
            v0 = np.cross(b - p, c - p)
            v1 = np.cross(c - p, a - p)
            v2 = np.cross(a - p, b - p)
            if (np.dot(v0, n) >= -eps and np.dot(v1, n) >= -eps
                    and np.dot(v2, n) >= -eps):
                return p, list(W)
        best, sub, bn = None, list(W), None
        for i, j in ((0, 1), (0, 2), (1, 2)):
            pp, ss = closest([W[i], W[j]])
            nn2 = float(np.dot(pp, pp))
            if bn is None or nn2 < bn:
                best, sub, bn = pp, ss, nn2
        return best, sub

    d = B.mean(0) - A.mean(0)
    nrm = float(np.linalg.norm(d))
    if not np.isfinite(nrm) or nrm < eps:
        d = np.array([1.0, 0.0, 0.0])
    d = d / max(nrm, eps)
    sim = [sup(A, d) - sup(B, -d)]
    d = -sim[0]
    if np.linalg.norm(d) < eps:
        return 0.0, True
    d = d / max(float(np.linalg.norm(d)), eps)
    for _ in range(iters):
        v = sup(A, d) - sup(B, -d)
        if float(np.dot(v, d)) < 0.0:
            p, _ = closest(sim)
            return float(np.linalg.norm(p)) * 1000.0, False
        sim = sim + [v]
        p, sub = closest(sim)
        if float(np.linalg.norm(p)) < eps:
            return 0.0, True
        sim = list(sub)
        d = -p
        if np.linalg.norm(d) < eps:
            return 0.0, True
        d = d / max(float(np.linalg.norm(d)), eps)
    p, _ = closest(sim)
    return float(np.linalg.norm(p)) * 1000.0, False


def _load_hulls():
    cands = sorted(glob.glob(os.path.join(_ROOT, "trimesh/agent_coacd/**/*.np*"),
                             recursive=True))
    if not cands:
        raise SystemExit("✗ 未找到 CoACD 凸包缓存 (trimesh/agent_coacd/**/*.np*) ✗")
    path = cands[-1]
    z = np.load(path, allow_pickle=False)
    hulls = []
    for k in z.files:
        a = np.asarray(z[k])
        if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] >= 4:
            hulls.append(a.astype(float))
    if not hulls:
        raise SystemExit(f"✗ 缓存 {path} 里没有顶点云 (keys={z.files}) ✗")
    print(f"[扫描] 凸包缓存: {os.path.relpath(path, _ROOT)} ⇒ {len(hulls)} 片 "
          f"(顶点 {sum(len(h) for h in hulls)}) ✓")
    return hulls


def _load_finger_meshes():
    pats = [os.path.join(_ROOT, "models/**/*finger_link1*.STL"),
            os.path.join(_ROOT, "models/**/*finger_link2*.STL")]
    out = []
    import trimesh  # 项目已装 (stage1 用到) ✓
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if not hits:
            raise SystemExit(f"✗ 未找到手指网格: {p} ✗")
        m = trimesh.load(hits[0], force="mesh")
        out.append(np.asarray(m.vertices, dtype=float))
    print(f"[扫描] 手指网格: link1 {len(out[0])} 顶点 / link2 {len(out[1])} 顶点 ✓")
    return out


def _anchor():
    for p in sorted(glob.glob(os.path.join(_ROOT, "**/best_grasp.npz"), recursive=True)):
        z = np.load(p, allow_pickle=False)
        if "pos" in z.files and "R" in z.files:
            print(f"[扫描] 锚点: {os.path.relpath(p, _ROOT)} ✓")
            return (np.asarray(z["pos"], float), np.asarray(z["R"], float),
                    float(z["gripper_qpos"]) if "gripper_qpos" in z.files else 0.0027)
    raise SystemExit("✗ 未找到锚点 best_grasp.npz (含 pos/R) ✗")


def main():
    hulls = _load_hulls()
    f1, f2 = _load_finger_meshes()
    pos, R, q_grasp = _anchor()
    obj_c = np.mean(np.concatenate(hulls, 0), 0)

    # 手指放在夹爪局部系: 开合轴 = 局部 y ✓ (stage1 口径 ✓), 半间距 = GAP0/2 + q ✓
    def finger_world(mesh_local, sign, q, pose_p, pose_R):
        half = GAP0 / 2.0 + float(q)
        off = np.array([0.0, sign * half, 0.0])
        return (mesh_local + off) @ pose_R.T + pose_p

    shifts = [-0.005, -0.0025, 0.0, 0.0025, 0.005]
    clears = [0.0, 0.5, 1.0, 2.0, 3.0]      # mm ✓ §11.4-3 对照切片 ✓
    rows = []
    gz = TABLE_Z
    for dz in shifts:
        for dy in shifts:
            for dx in shifts:
                for dq in (0.0, 0.001, 0.002):
                    p = pos + np.array([dx, dy, dz])
                    q = q_grasp + dq
                    W1 = finger_world(f1, +1.0, q, p, R)
                    W2 = finger_world(f2, -1.0, q, p, R)
                    lo_z = min(float(W1[:, 2].min()), float(W2[:, 2].min())) - gz
                    d_left, hit_l = gjk_gap_mm(W1, np.concatenate(hulls, 0))
                    d_right, hit_r = gjk_gap_mm(W2, np.concatenate(hulls, 0))
                    if d_left is None or d_right is None:
                        continue
                    rows.append({
                        "clear_mm": lo_z * 1000.0,
                        "d_left": d_left, "d_right": d_right,
                        "worst": max(d_left, d_right),
                        "hit": bool(hit_l or hit_r),
                        "pose": (p - pos).tolist(), "dq": dq,
                    })
    if not rows:
        raise SystemExit("✗ 没有采到有效样本 ✗")
    arr = rows
    print(f"[扫描] 采样 {len(arr)} 个 (平移 125 × q 3) ✓")
    print("\n=== 按净空档分组: 最好的 max(d_left,d_right) ===")
    verdict = "NO_CANDIDATE_IN_SAMPLED_DOMAIN"
    for c in clears:
        sel = [r for r in arr if abs(r["clear_mm"] - c) <= 0.25]
        if not sel:
            print(f"  净空≈{c:.1f}mm: 无样本")
            continue
        best = min(sel, key=lambda r: r["worst"])
        tag = "✓ 双侧可触及" if (best["worst"] <= 0.5 and not best["hit"]) else (
            "⚠ 相交" if best["hit"] else "✗ 够不到")
        print(f"  净空≈{c:.1f}mm: 最好 worst=max(dL,dR)={best['worst']:.2f}mm "
              f"(dL={best['d_left']:.2f}, dR={best['d_right']:.2f}) 平移={best['pose']} "
              f"dq={best['dq']*1000:.1f}mm ⇒ {tag}")
        if best["worst"] <= 0.5 and not best["hit"]:
            verdict = "FOUND_GEOMETRIC_CANDIDATE"
    print(f"\n=== 结论: **{verdict}** ===")
    print("⚠ 本扫描为**初判** ✓ (对称指放置 + 锚点姿态近似 ✗); 命中者须回动力学场景复核 ✓")
    out = os.path.join(_ROOT, "output", "access_scan_result.json")
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"verdict": verdict, "n_samples": len(arr),
                       "best": sorted(arr, key=lambda r: r["worst"])[:10]}, f, indent=1)
        print(f"[扫描] 结果已保存: {os.path.relpath(out, _ROOT)} ✓")
    except Exception as e:  # noqa: BLE001
        print("结果保存失败:", e)


if __name__ == "__main__":
    main()

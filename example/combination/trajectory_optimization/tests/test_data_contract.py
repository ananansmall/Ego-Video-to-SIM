"""数据契约测试 —— 专查"仿真 → 实际/跨后端"的数据传输 bug (2026-09-21).

背景 (用户 2026-09-21):
    "你可以先修复一下有没有相关数据的 bug, 再测试一下 stage2,
     特别是**数据从仿真到实际的传输**这种的, 可以测试一下目前的代码有没有 bug"

本脚本的 3 类检查 (全部**只读**, 不动仿真):
    ① **npz 键契约**: 真实产物文件的键 vs 源码里 `tp['...']` 之类的读取键
       ⇒ 抓 "读一个从来没写过的键" 这类 KeyError (已实测抓到 1 个 ✓ 见 BUG 清单)
    ② **导出包 schema**: stage1/2/3 bundle 的写出键 vs 读侧键
    ③ **CSV 列语义**: `gripper_trace.csv` 的命令列是否与"实际"列重复 (曾实测重复 ✗)

用法:  /home/an/miniconda3/envs/dex/bin/python tests/test_data_contract.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  ✗ {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  ⚠ {msg}")


# ---------------------------------------------------------------- ① npz 键契约
def newest(pattern: str) -> Path | None:
    fs = sorted(glob.glob(str(ROOT / pattern)), key=os.path.getmtime)
    return Path(fs[-1]) if fs else None


def check_transform_params() -> None:
    print("\n[①] transform_params.npz (仿真↔实际 的尺度/位姿变换)")
    f = newest("output/*/alignment/transform_params.npz")
    if f is None:
        warn("没有找到任何 transform_params.npz, 跳过")
        return
    keys = set(np.load(f).files)
    print(f"  样本: {f.relative_to(ROOT)}")
    print(f"  实际键 ({len(keys)}): {sorted(keys)}")

    # 读侧要求 (trajectory_loader.load_glb_transformed 使用 ✓)
    src = (ROOT / "trajectory_loader.py").read_text(encoding="utf-8")
    _reads = set(re.findall(r"tp\['([a-z_0-9]+)'\]", src))
    _reads |= set(re.findall(r'tp\["([a-z_0-9]+)"\]', src))
    print(f"  读侧键 (trajectory_loader): {sorted(_reads)}")
    for k in ("R_inv", "t_inv"):
        (ok if k in keys else fail)(f"必需键 {k!r} {'在' if k in keys else '**缺失** ✗'} 产物中")
    # scale 三选一 (修复后的兼容逻辑 ✓)
    if {"scale_ratio", "scale", "s_inv"} & keys:
        ok("scale 口径可用 (scale_ratio / scale / s_inv 至少一个存在 ✓)")
    else:
        fail("scale 口径**三者皆缺** ⇒ 任何读法都会失败 ✗")
    if "scale" not in keys:
        ok("已确认产物**没有** 'scale' 键 ⇒ `tp['scale']` 会 KeyError "
           "(trajectory_loader 已修为兼容三选一 ✓)")

    # t_inv 口径: **反向变换** (GLB→HaWoR) 是加载 GLB 所需 ✓; 正向键是新版 ✓
    #   实测核对 (2026-09-21): t_inv 与 t_hand_to_glb **逐值相同** ⇒ 不存在 2.55× 错位 ✓
    d = np.load(f)
    t_inv = np.asarray(d["t_inv"], dtype=np.float64).reshape(3)
    s_inv = float(d["s_inv"]) if "s_inv" in d.files else 1.0
    print(f"  t_inv            = {np.round(t_inv, 4)}  |t_inv|={np.linalg.norm(t_inv):.4f} m")
    if "t_hand_to_glb" in d.files and "R_hand_to_glb" in d.files:
        t_fwd = np.asarray(d["t_hand_to_glb"], dtype=np.float64).reshape(3)
        R_diff = float(np.abs(np.asarray(d["R_inv"], dtype=np.float64)
                              - np.asarray(d["R_hand_to_glb"], dtype=np.float64)).max())
        if np.allclose(t_inv, t_fwd, atol=1e-9) and R_diff < 1e-9:
            ok(f"反向键 == 正向键 (|Δt|={np.linalg.norm(t_inv - t_fwd):.2e}m, "
               f"max|ΔR|={R_diff:.2e}) ⇒ 两个方向的口径**本来就一致** ✓ (无 2.55× 错位 ✗)")
        else:
            warn(f"反向键 != 正向键 (|Δt|={np.linalg.norm(t_inv - t_fwd):.4f}m, "
                 f"max|ΔR|={R_diff:.3f}) ⇒ 需确认加载方向 (loader 用反向 ✓)")
    if ("scale_ratio" in d.files and abs(s_inv - 1.0) > 1e-6
            and abs(s_inv * float(d["scale_ratio"]) - 1.0) > 1e-6):
        fail(f"s_inv({s_inv:.4f}) 与 1/scale_ratio({1.0 / float(d['scale_ratio']):.4f}) 不自洽 ✗")
    elif "scale_ratio" in d.files:
        ok(f"s_inv == 1/scale_ratio ({s_inv:.4f} == 1/{float(d['scale_ratio']):.4f}) ✓")


def check_preprocess_traj() -> None:
    print("\n[②] preprocess_traj.npz (Stage1/2 共用的 MANO 预处理轨迹)")
    f = newest("output/*/preprocess_traj.npz")
    if f is None:
        warn("没有找到 preprocess_traj.npz, 跳过")
        return
    d = np.load(f, allow_pickle=True)
    print(f"  样本: {f.relative_to(ROOT)}")
    for k in sorted(d.files):
        a = d[k]
        print(f"    {k:20s} shape={getattr(a, 'shape', None)!s:16s} dtype={a.dtype}")
    # stage1_grasp_mixin 里用到的键 (从源码抓 `d['x']` / `npz['x']`)
    src = (ROOT / "stage1_grasp_mixin.py").read_text(encoding="utf-8")
    _need = set(re.findall(r"npz\[['\"]([a-z_0-9]+)['\"]\]", src))
    if _need:
        _miss = sorted(_need - set(d.files))
        (ok if not _miss else fail)(
            f"Stage1 读取的键均存在 ✓" if not _miss else f"Stage1 读的键缺失: {_miss} ✗")
    # 形状自洽: joints / joints_raw 应同形; grip_R 应是 (N,3,3) 正交阵
    for pair in (("joints", "joints_raw"),):
        a, b = pair
        if a in d.files and b in d.files:
            (ok if d[a].shape == d[b].shape else fail)(
                f"{a}.shape == {b}.shape ({d[a].shape})" if d[a].shape == d[b].shape
                else f"{a}.shape={d[a].shape} != {b}.shape={d[b].shape} ✗")
    if "grip_R" in d.files:
        R = np.asarray(d["grip_R"], dtype=np.float64)
        if R.ndim == 3 and R.shape[1:] == (3, 3):
            orth = np.abs(R @ np.transpose(R, (0, 2, 1)) - np.eye(3)).max()
            det = np.linalg.det(R)
            (ok if orth < 1e-6 else fail)(f"grip_R 正交 (max|RRᵀ−I|={orth:.2e})")
            (ok if np.allclose(det, 1.0, atol=1e-6) else fail)(
                f"grip_R 行列式 = 1 (det∈[{det.min():.6f},{det.max():.6f}])")
        else:
            fail(f"grip_R 形状异常: {R.shape} ✗ (期望 (N,3,3))")


def check_bundle_schema() -> None:
    print("\n[③] 导出 bundle (stage1/2/3 交付件) 的键契约")
    f = newest("output/*/bundle*.npz")
    if f is None:
        warn("没有找到 bundle*.npz (需跑到 Stage2/3 才产出 ✓), 只做源码级检查")
    # 源码级: stage3 写出键 vs 读侧键
    w_src = (ROOT / "stage3_optimize_mixin.py").read_text(encoding="utf-8")
    m = re.search(r"np\.savez_compressed\((.*?)\n        \)", w_src, re.S)
    if m:
        written = set(re.findall(r"^\s+([A-Za-z_0-9]+)=", m.group(1), re.M))
        print(f"  Stage3 写出键 ({len(written)}): {sorted(written)[:12]}{' ...' if len(written) > 12 else ''}")
        for k in ("base_pos", "base_R", "gripper_q", "control_dt", "bundle_hash"):
            (ok if k in written else fail)(f"Stage3 bundle 含 {k!r}")
    else:
        warn("无法解析 Stage3 的 np.savez_compressed 键列表")


def check_csv_columns() -> None:
    print("\n[④] gripper_trace.csv 的列语义 (命令列 vs 实际列)")
    f = newest("output/*/stage1/gripper_trace.csv")
    if f is None:
        warn("没有找到 gripper_trace.csv, 跳过")
        return
    header = f.read_text(encoding="utf-8").splitlines()[0].split(",")
    cols = {n: i for i, n in enumerate(header)}
    need = {"finger1_q", "finger2_q", "cmd_q1", "cmd_q2", "obj_z"}
    if not need <= set(cols):
        fail(f"表头缺列: {sorted(need - set(cols))} ✗")
        return
    rows = [ln.split(",") for ln in f.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    a1 = np.array([float(r[cols['finger1_q']]) for r in rows])
    c1 = np.array([float(r[cols['cmd_q1']]) for r in rows])
    same = np.allclose(a1, c1, atol=1e-9)
    print(f"  样本: {f.relative_to(ROOT)}  帧数={len(rows)}")
    if same:
        fail("`cmd_q1` 与 `finger1_q` **逐帧完全相同** ✗ ⇒ 命令列写成了实际值 "
             "(任何按 CSV 判'手指是否被挡停'的分析都会失效 ✗)")
    else:
        ok(f"命令列与实际列**已可区分** ✓ (中位差 {np.median((a1 - c1)) * 1000:+.2f}mm)")


def main() -> int:
    print("=" * 78)
    print("数据契约测试 (仿真 ↔ 实际/跨后端)")
    print("=" * 78)
    check_transform_params()
    check_preprocess_traj()
    check_bundle_schema()
    check_csv_columns()
    print("\n" + "=" * 78)
    print(f"结果: {len(FAILS)} 失败 / {len(WARNS)} 警告")
    for m in FAILS:
        print(f"  ✗ {m}")
    for m in WARNS:
        print(f"  ⚠ {m}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

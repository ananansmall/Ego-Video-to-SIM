#!/usr/bin/env python3
"""ee17_gravity_clock.py — EE17 时钟修复的独立验证 (用户交付要求第 1 步)。

两项检查:
  1. 同场景自由落体: 复用正式仿真的建场逻辑 (rg.setup_scene, 250Hz 子步),
     放一个静止动态刚体于地面上方 ~0.25m, 关闭该刚体阻尼, 首次接触前按
     物理子步记录 XYZ。拟合 z(t)=z0+v0 t+½ a t² 的 a → 竖直加速度应接近
     -9.81 m/s², x/y 线性拟合速度应接近 0。
  2. 600 帧时钟: 用与 cb_large_range_sim 相同的累计时钟 (30fps 源, 250Hz
     子步, 8/8/9 交替), 600 个帧间隔累计应为 5000 子步 = 20.000s。

输出: <out-dir>/gravity_clock.json + gravity_samples.npz (+终端摘要)。
用法:
    python test/ee17_gravity_clock.py --out-dir results/large_range/<TAG>
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import sapien.core as sapien                       # noqa: E402
import robotwin_r1_duck_grasp as rg                # noqa: E402  复用正式建场/步进


def free_fall(n_steps=250, z0=0.25, dt=1.0 / 250.0):
    """空场景 + 静止动态刚体自由落体, 按子步记录。首次接触即停。"""
    engine = sapien.Engine()
    scene = rg.setup_scene(engine)                 # 正式同款: 250Hz + ground + 光照
    # 地面 y? scene.add_ground(0) -> z=0。测试体: 8cm 立方, 悬在 z0 处
    half = 0.04
    bb = scene.create_actor_builder()
    bb.set_physx_body_type("dynamic")
    bb.add_box_collision(sapien.Pose(p=[0, 0, 0]), half_size=[half, half, half])
    bb.add_box_visual(sapien.Pose(p=[0, 0, 0]), half_size=[half, half, half],
                      material=[0.2, 0.6, 0.9])
    body = bb.build(name="gravity_probe")
    body.set_pose(sapien.Pose(p=[0.0, 0.0, z0]))
    # 关闭测试刚体阻尼 (用户要求): 不受任何驱动/速度耗散
    for comp in body.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            comp.linear_damping = 0.0
            comp.angular_damping = 0.0
    xyz = np.zeros((n_steps, 3))
    touched = n_steps
    ground_top = 0.0                               # add_ground 平面 z=0
    for s in range(n_steps):
        scene.step()                               # 纯物理: 无驱动 (自由落体)
        p = np.asarray(body.get_pose().p, float)
        xyz[s] = p
        if p[2] - half <= ground_top + 1e-6 and touched == n_steps:
            touched = s + 1                        # 首次接触步 (含该步即停记录)
            break
    n = touched if touched < n_steps else n_steps
    xyz = xyz[:n]
    t = np.arange(1, n + 1) * dt                   # 每子步后的时刻
    return t, xyz, {"n_steps_recorded": int(n),
                    "touched_ground_step": int(touched) if touched < n_steps else -1,
                    "z0": z0, "half": half, "dt": dt}


def fit_gravity(t, xyz, half=0.04, min_clear=0.02):
    """竖直: 二次拟合 z(t) -> a=2c; 水平: 线性拟合速度。
    触地/接触求解段会污染抛物线 (实测 62 步含触地帧 -> g 被拉到 -3.85),
    只保留离地 > min_clear 的纯自由落体样本。"""
    m = xyz[:, 2] - half > min_clear
    t_f, z_f, xy_f = t[m], xyz[m, 2], xyz[m, :2]
    c2, c1, c0 = np.polyfit(t_f, z_f, 2)
    a_z = 2.0 * c2
    vx = np.polyfit(t_f, xy_f[:, 0], 1)[0]
    vy = np.polyfit(t_f, xy_f[:, 1], 1)[0]
    resid_z = z_f - (c2 * t_f ** 2 + c1 * t_f + c0)
    return {"g_vertical_m_s2": float(a_z),
            "g_vertical_err_vs_981": float(a_z - (-9.81)),
            "horizontal_vx_m_s": float(vx),
            "horizontal_vy_m_s": float(vy),
            "horizontal_speed_m_s": float(np.hypot(vx, vy)),
            "z_fit_max_abs_resid_m": float(np.abs(resid_z).max()),
            "n_samples_free_fall": int(m.sum()),
            "n_samples_touched": int(len(t) - m.sum()),
            "t_span_s": float(t_f[-1] - t_f[0]) if len(t_f) > 1 else 0.0}


def clock_check(n_frames=600, fps=30.0, hz=250.0):
    """与 cb 相同的累计时钟: 600 帧间隔应累计 5000 子步 = 20.000s。"""
    sub_per_frame = hz / fps
    acc = 0.0
    total = 0
    steps = np.zeros(n_frames, dtype=np.int64)
    for k in range(n_frames):
        acc += sub_per_frame
        n = int(np.floor(acc))
        acc -= n
        n = max(n, 1)
        steps[k] = n
        total += n
    return {"n_frames": n_frames, "total_substeps": int(total),
            "phys_time_s": float(total / hz),
            "source_time_s": float(n_frames / fps),
            "aligned": bool(total == 5000 and abs(total / hz - n_frames / fps) < 1e-9),
            "steps_unique": sorted(set(steps.tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-steps", type=int, default=250)
    ap.add_argument("--z0", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    t, xyz, meta = free_fall(n_steps=a.n_steps, z0=a.z0)
    fit = fit_gravity(t, xyz, half=meta["half"])
    clk = clock_check()

    ok_g = abs(fit["g_vertical_err_vs_981"]) < 0.30
    ok_h = fit["horizontal_speed_m_s"] < 0.02
    ok_c = clk["aligned"]
    ok_resid = fit["z_fit_max_abs_resid_m"] < 5e-3
    all_ok = ok_g and ok_h and ok_c and ok_resid

    report = {"free_fall": meta, "fit": fit, "clock": clk,
              "gates": {"vertical_g_close_981": ok_g, "horizontal_near_zero": ok_h,
                        "clock_5000_substeps_20s": ok_c,
                        "z_parabola_resid_ok": ok_resid},
              "all_pass": bool(all_ok)}
    json.dump(report, open(os.path.join(a.out_dir, "gravity_clock.json"), "w"),
              indent=1, ensure_ascii=False)
    np.savez(os.path.join(a.out_dir, "gravity_samples.npz"),
             t=t, xyz=xyz, **{f"fit_{k}": v for k, v in fit.items()})
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(("PASS" if all_ok else "FAIL") + ": gravity_clock "
          f"(g={fit['g_vertical_m_s2']:.3f} m/s², |水平|={fit['horizontal_speed_m_s']:.4f} m/s, "
          f"clock={clk['total_substeps']} substeps/{clk['phys_time_s']:.3f}s)")


if __name__ == "__main__":
    main()

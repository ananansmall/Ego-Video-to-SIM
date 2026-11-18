#!/usr/bin/env python3
"""ee_mocap_targets.py — EgoExplorer mocap 轨迹 → R1 跟随目标 (绕开 EF3R 手部建模)。

用户定义的路径: EgoExplorer 自带**世界系+重力系** mocap (tracker_position,
y-up, tracker[2]=right_wrist, 全帧有效) —— 比 EF3R 的 ego 手部重建干净得多。
v0 = **不拟合手部**: 直接跟随 right_wrist 世界轨迹 (位置), 姿态固定下探
(平行爪可行几何), 夹爪全开 —— 纯"长距离跟随准确性"测试。

坐标系: EgoExplorer y-up (重力=-y, 与 EF3R 一致) → 管线 z-up:
    M = [[1,0,0],[0,0,1],[0,-1,0]]  (x,y,y-up) → (x,z,-y) z-up, 与 456 链同构

用法:
    python scripts/ee_mocap_targets.py --episode 5 [--t0 8 --t1 40]
输出: <out>/ee_mocap_targets.npz (与 ef3r_right_targets.npz 同契约, cb 直接消费)
"""
import argparse
import json
import os

import numpy as np

EE = "/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--t0", type=float, default=None, help="窗起点秒 (默认抓取前 3s)")
    ap.add_argument("--t1", type=float, default=None, help="窗终点秒 (默认放置段尾)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="时间伸展 >1=慢放 (腕速超机器人能力时合法慢速演示)")
    a = ap.parse_args()

    ep_dir = f"{EE}/episode_meta/episode_{a.episode:06d}"
    d = np.load(f"{ep_dir}/smplx/smplx_episode.npz")
    tr = d["tracker_position"]                       # (N,5,3) 世界系 y-up
    N = len(tr)
    wrist = tr[:, 2, :]                              # right_wrist 世界轨迹

    # 时间窗: segments 的抓取段/放置段 (无注解则全片)
    t0, t1 = 0.0, N / 30.0
    sg_p = f"{ep_dir}/annotations/segments.parquet"
    fps = 30.0
    if os.path.exists(sg_p):
        import subprocess
        out = subprocess.run(["/path/to/user_home/miniconda3/envs/agile-optim/bin/python",
                              "-c",
                              f"import pandas as pd;print(pd.read_parquet('{sg_p}').to_json(orient='records'))"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            segs = json.loads(out.stdout)
            _gs_all = [s for s in segs
                       if any(k in s.get("caption_zh", "") for k in
                              ("抓握", "握着", "拿起", "取出", "拾取"))]
            if a.t0 is not None:
                # 多窗口模式: 取**包含/最近**窗起点的抓取段
                grasp = min(_gs_all,
                            key=lambda s2: abs(float(s2["start"]) - a.t0),
                            default=None)
            else:
                grasp = _gs_all[0] if _gs_all else None
            if grasp:
                t0 = max(0.0, float(grasp["start"]) - 3.0)
            place = [s for s in segs
                     if any(k in s.get("caption_zh", "") for k in
                            ("放下", "放置", "放回", "平放"))
                     and float(s["start"]) >= float(grasp["end"]) - 1.0]
            if place:
                # 抓取后**第一个**放置段 + 35s 封顶 (多交互片段后续还有交互)
                t1 = min(float(place[0]["end"]) + 2.0, t0 + 35.0, N / fps)
            else:
                t1 = min(t1, t0 + 35.0)
    if a.t0 is not None:
        t0 = a.t0
    if a.t1 is not None:
        t1 = a.t1
    i0, i1 = int(t0 * fps), min(int(t1 * fps), N)
    if a.time_scale != 1.0:
        # 时间伸展: 窗内序列 (手/骨盆) 线性插值重采样 —— 帧数 ×scale,
        # 速度 ÷scale (合法慢速演示); 手/骨盆/关节时序同步伸展
        M_new = int((i1 - i0) * a.time_scale)
        _src = np.linspace(0, i1 - i0 - 1, M_new)
        _lo = np.floor(_src).astype(int)
        _fr = (_src - _lo)[:, None]
        _hi = np.minimum(_lo + 1, i1 - 1)
        _W = wrist[i0:i1]
        wrist = np.concatenate([wrist[:i0],
                                _W[_lo] * (1 - _fr) + _W[_hi] * _fr,
                                wrist[i1:]])
        i0, i1 = 0, M_new
        fps = fps / a.time_scale                      # 时序标注同步伸展
        print(f"  时间伸展 ×{a.time_scale}: 窗内 {M_new} 帧 (速度 ÷{a.time_scale:.2f})")
    print(f"  窗 {t0:.1f}-{t1:.1f}s ({i1-i0} 帧) | right_wrist 世界轨迹 (y-up)")

    W = wrist[i0:i1]                                  # (M,3) y-up 世界系 (手, 全局)
    body = d["transl"][i0:i1]                         # (M,3) SMPL-X 骨盆 (全局)
    # **移动复刻的正确分解** (与 EF3R 链同构): 底盘 = 身体轨迹 (人走 5m, 底盘跟
    # 5m), pad = 手**相对身体**的偏移 (始终在臂工作空间内) —— 全局手腕直接当
    # pad 目标会超出 WBC base_range (实测 5271mm 起步)。
    # y-up → z-up (重力对齐; 与 456 链 M_EF3R_TO_SIM 同构)
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # y-up→z-up 正确: z=+y 向上 (旧版 z=-y 方向反)
    Pw = W @ M.T                                      # 手 z-up 全局
    Pb = body @ M.T                                   # 骨盆 z-up 全局
    print(f"  身体轨迹范围 x {Pb[:,0].min():.2f}~{Pb[:,0].max():.2f} "
          f"y {Pb[:,1].min():.2f}~{Pb[:,1].max():.2f} (行走 "
          f"{np.linalg.norm(np.diff(Pb[:,:2],axis=0),axis=1).sum():.1f}m)")
    # 相对量: pad-骨盆 (基座系), 高度 = 手相对骨盆 + 操作高度 0.75
    P_rel = Pw - Pb                                   # (M,3) 手相对骨盆
    z_plane = float(np.median(P_rel[:, 2]))
    P_rel[:, 2] += 0.75 - z_plane                     # 交互面 → 0.75m
    print(f"  相对手-骨盆: x {P_rel[:,0].min():.2f}~{P_rel[:,0].max():.2f} "
          f"y {P_rel[:,1].min():.2f}~{P_rel[:,1].max():.2f} "
          f"z {P_rel[:,2].min():.2f}~{P_rel[:,2].max():.2f} (对齐后)")

    M_n = i1 - i0
    # 姿态: 固定下探 (接近轴 = -z, 开合轴 = x) —— 不拟合手部
    R_fix = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)  # z 接近, x 开合
    rot = np.tile(R_fix, (M_n, 1, 1))
    # pad = 腕前方 8cm (沿接近轴, 指尖位置)
    pad_fwd = np.array([0, 0, -0.08])
    # pad 目标 = **世界系手轨迹** (高度不对齐!): 高度归一全权交给 cb 的
    # fit_height (456 链验证的唯一权威) —— 脚本预对齐会与 cb 双重归一打架
    # (实测两次平移相抵后目标再钻地下)。底盘路径 = 骨盆 xy (t_c2w 契约)。
    _floor_z = float(np.min(d["transl"][:, 2]))
    pos = Pw + rot @ pad_fwd
    grip = np.full((M_n, 2), 0.045)                   # v0 全开 (纯跟随, 无抓取)
    valid = np.ones(M_n, bool)
    print(f"  手轨迹世界系 (y-up→z-up), 高度交 cb fit_height; 地面 z={_floor_z:.2f}")

    # 抓取时序 (用户: 一边抓取一边跟随): segments 抓取段→闭合(LIFT), 放置段→
    # 张开(RELEASED)。夹宽由 cb 物体几何推导 (物体薄轴), 不用手部拟合。
    gs = np.zeros(M_n, np.int64)                      # 0=APPROACH
    _c2w = np.zeros((M_n, 3))
    _c2w[:, :2] = Pb[:, :2]                           # 骨盆 xy = 底盘行走路径
    _c2w[:, 2] = Pw[:, 2]                             # 手高 (cb fit_height 统一平移)
    _grasp_t = _rel_t = None
    if os.path.exists(sg_p):
        import subprocess
        out = subprocess.run(["/path/to/user_home/miniconda3/envs/agile-optim/bin/python",
                              "-c",
                              f"import pandas as pd;print(pd.read_parquet('{sg_p}').to_json(orient='records'))"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            segs = json.loads(out.stdout)
            _g = next((s2 for s2 in segs
                       if any(k in s2.get("caption_zh", "") for k in
                              ("抓握", "握着", "拿起", "取出", "拾取"))), None)
            _r = [s2 for s2 in segs if any(k in s2.get("caption_zh", "") for k in
                                           ("放下", "放置", "放回", "平放"))]
            if _g and float(_g["start"]) >= t0:
                _grasp_t = float(_g["start"])
            if _r and float(_r[-1]["end"]) <= t1:
                _rel_t = float(_r[-1]["end"])
    if _grasp_t is not None:
        # 闭合帧 = 抓取段内**腕速度骤降**时刻 (握住物体的稳定点): 段起点只是
        # "开始去拿" —— 手还没到物体, 那时闭合会夹空 (实测 ep11/12 抓起✗)。
        # 规则: 段内首个 1s 窗中位速度 <0.05m/s 且其后 1s 仍 <0.08 (真稳定)。
        tr_w = wrist[i0:i1]                            # (M,3) mocap 世界系
        _v = np.linalg.norm(np.diff(tr_w, axis=0), axis=1) * fps
        _v = np.append(_v, _v[-1])
        _gs_i = max(0, int((_grasp_t - t0) * fps))
        _ge_i = min(int(((_grasp_t + 14) - t0) * fps), M_n - 2)   # 段尾+14s 内必稳定
        _ki = None
        for _a in range(_gs_i, min(_ge_i, M_n - 32)):
            _w1, _w2 = _v[_a:_a + 30], _v[_a + 30:_a + 60]
            if len(_w1) < 30 or len(_w2) < 30:
                break
            if float(np.median(_w1)) < 0.05 and float(np.median(_w2)) < 0.08:
                _ki = _a + 30
                break
        if _ki is None:
            _ki = _gs_i                                 # 回退: 段起点
        gs[_ki:] = 2                                   # LIFT 起: 物理抓取闭合
        grip[_ki:, 0] = 0.015                          # 抓握宽 (cb 物体几何覆盖)
        grip[_ki:, 1] = 0.015
        print(f"  抓取时序: 闭合@{t0 + _ki / fps:.1f}s (clip F{_ki}, "
              f"速度骤降检测)")
    if _rel_t is not None:
        _ri = max(0, min(int((_rel_t - t0) * fps), M_n - 1))
        gs[_ri:] = 5                                   # RELEASED 起: 张开
        grip[_ri:, 0] = 0.045
        grip[_ri:, 1] = 0.045
        print(f"  释放时序: 张开@{_rel_t:.1f}s (clip F{_ri})")

    out = a.out or f"/path/to/project_data/mano_ws/EgoFound3R/outputs/ee{a.episode}_mocap"
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "ef3r_right_targets.npz"),
             pos=pos, rot=rot, grip=grip, valid=valid, frame=np.arange(M_n),
             fps=fps, source="egoexplorer_mocap", episode=a.episode,
             window=[t0, t1],
             R_c2w=np.tile(np.eye(3), (M_n, 1, 1)),       # dummy 相机契约 (cb 必需)
             t_c2w=_c2w,
             grip_states=gs,
             grip_state_names=np.array(["APPROACH","PICK","LIFT","HOLD","PLACE","RELEASED"]),
             base_body_xy=Pb[:, :2],                 # 骨盆 xy (行走路径, 底盘跟随源)
             pelvis_z=Pb[:, 2])
    print(f"  ✓ {out}/ef3r_right_targets.npz ({M_n} 帧)")


if __name__ == "__main__":
    main()

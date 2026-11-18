#!/usr/bin/env python3
"""robot_config.py — 换机器人的配置层（**不改核心脚本**）。

思路
----
`robotwin_r1_duck_grasp.build_episode()` 在运行时读取一批模块级全局
（`R1_URDF` / `ARM_JOINTS` / `GRIP_LINK` / `GRIP_FINGERS` / `IK_LIMITS` /
`R1_LEFT_START` / `PAD_FWD` / `GRIP_OPEN_Q` / `GRIP_HOLD_Q` / `R1_BASE_P` ...）。
因此只要在调用前把这些全局替换掉，就能换机器人 —— **无需修改核心脚本**，
也就不会和并行编辑者冲突。

用法
----
    import robotwin_r1_duck_grasp as G
    from robot_config import apply_robot, inspect_robot
    rep = inspect_robot("r1_lite")      # 先看兼容性
    apply_robot(G, "r1_lite")           # 再打补丁
    ep = G.build_episode(params={...})

兼容性结论（实测 2026-10-15）
--------------------------
| 机器人   | 左臂DoF | 关节命名 | 结论 |
| r1       | 6       | 符合     | ✅ 当前在用 |
| r1_lite  | 6       | 符合     | ⚠️ 可加载，但限位更窄(joint4/5 仅 ±90°，r1 是 ±165°)，可达性更差 |
| r1_pro   | 7       | 符合     | ❌ 7-DoF，本仓 IK 种子是 6 维，**不能直接跑** |
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

GALAXEA_ASSETS = "/path/to/workspace/assets_galaxea/GalaxeaManipSim/galaxea_sim/assets"

# 每种机器人的默认配置。overrides 里没有的项会沿用核心脚本的常量。
ROBOTS = {
    "r1": {
        "dir": "r1",
        "arm": "left",
        # 以下为 r1 实测值(与核心脚本常量一致), 列出以便对照
        "pad_fwd": 0.0369,
        "grip_open_q": 0.045,
        "grip_hold_q": 0.0302,
        "base_p": [0.20, -0.42, 0.0],
        "base_q": [0.70710678, 0.0, 0.0, 0.70710678],
        # 右臂镜像得到左臂起始位: left = [-r1, r2, r3, -r4, -r5, -r6]
        "left_start": [1.5, 1.9508485794067383, -1.080855369567871,
                       0.4437669813632965, -0.17093953490257263, -0.19848759472370148],
        "right_park": [-1.0, 0.6, -0.4, -0.5, 0.0, 0.0],
    },
    "r1_lite": {
        "dir": "r1_lite",
        "arm": "left",
        # ⚠️ 以下三项**未标定**, 沿用 r1 值仅作起点; 换机器人后必须实测
        "pad_fwd": 0.0369,
        "grip_open_q": 0.045,
        "grip_hold_q": 0.0302,
        "base_p": [0.20, -0.42, 0.0],
        "base_q": [0.70710678, 0.0, 0.0, 0.70710678],
        "left_start": [1.5, 1.9508485794067383, -1.080855369567871,
                       0.4437669813632965, -0.17093953490257263, -0.19848759472370148],
        "right_park": [-1.0, 0.6, -0.4, -0.5, 0.0, 0.0],
    },
    "r1_pro": {
        "dir": "r1_pro",
        "arm": "left",
        "note": "7-DoF 臂, 本仓不支持",
    },
}


def urdf_path(robot):
    return os.path.join(GALAXEA_ASSETS, ROBOTS[robot]["dir"], "robot.urdf")


def arm_joint_limits(urdf, arm="left"):
    """直接从 URDF 解析 {arm}_arm_jointN 的限位(不加载仿真, 秒级返回)。"""
    root = ET.parse(urdf).getroot()
    rows = []
    for j in root.iter("joint"):
        nm = j.get("name") or ""
        if not nm.startswith(f"{arm}_arm_joint"):
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        rows.append((nm, float(lim.get("lower")), float(lim.get("upper"))))
    rows.sort(key=lambda r: int(r[0].rsplit("joint", 1)[-1]))
    return rows


def joint_names(urdf):
    root = ET.parse(urdf).getroot()
    return [j.get("name") for j in root.iter("joint") if j.get("name")]


def link_names(urdf):
    root = ET.parse(urdf).getroot()
    return [l.get("name") for l in root.iter("link") if l.get("name")]


def inspect_robot(robot):
    """返回兼容性报告(不加载仿真)。"""
    if robot not in ROBOTS:
        return {"ok": False, "reason": f"未配置的机器人: {robot}", "available": list(ROBOTS)}
    urdf = urdf_path(robot)
    if not os.path.exists(urdf):
        return {"ok": False, "reason": f"URDF 不存在: {urdf}"}
    arm = ROBOTS[robot].get("arm", "left")
    lims = arm_joint_limits(urdf, arm)
    names = joint_names(urdf)
    dof = len(lims)
    need_joints = [f"{arm}_arm_joint{i}" for i in range(1, 7)]
    need_links = [f"{arm}_gripper_link"]
    need_fingers = [f"{arm}_gripper_finger_joint1", f"{arm}_gripper_finger_joint2"]
    rep = {
        "ok": True, "robot": robot, "urdf": urdf, "arm": arm,
        "arm_dof": dof,
        "limits": [[n, lo, hi] for n, lo, hi in lims],
        "naming_ok": (all(n in names for n in need_joints + need_fingers)
                      and all(n in link_names(urdf) for n in need_links)),
        "missing": ([n for n in need_joints + need_fingers if n not in names]
                    + [n for n in need_links if n not in link_names(urdf)]),
        "dof_ok": dof == 6,
    }
    reasons = []
    if not rep["dof_ok"]:
        reasons.append(f"臂是 {dof}-DoF，本仓 IK 种子/限位矩阵按 6-DoF 写死 → 不能直接跑")
    if not rep["naming_ok"]:
        reasons.append(f"命名不符约定, 缺: {rep['missing']}")
    rep["reasons"] = reasons
    rep["runnable"] = rep["dof_ok"] and rep["naming_ok"]
    return rep


def absolutize_urdf(src, dst):
    """把 URDF 里所有相对路径的 mesh/texture 改成绝对路径后另存。

    为什么需要：核心脚本的 `make_r1_urdf_abs()` 只重写 `./` 开头的路径，
    而 r1_lite/r1_pro 的 URDF 用的是 `meshes/xxx.obj`（无 `./`），不会被处理，
    SAPIEN 直接报 `cannot make canonical path`。这里在配置层预先处理好。
    """
    base = os.path.dirname(os.path.abspath(src))
    tree = ET.parse(src)
    root = tree.getroot()
    n = 0
    for tag in ("mesh", "texture"):
        for el in root.iter(tag):
            fn = el.get("filename")
            if fn and not os.path.isabs(fn):
                el.set("filename", os.path.join(base, fn[2:] if fn.startswith("./") else fn))
                n += 1
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tree.write(dst)
    return dst, n


def apply_robot(G, robot, overrides=None):
    """把机器人配置打到核心脚本 G 的模块级全局上(猴子补丁, 不改文件)。

    overrides: dict, 可覆盖 pad_fwd / grip_open_q / grip_hold_q / base_xy /
               left_start / base_p / base_q / right_park / arm
    """
    rep = inspect_robot(robot)
    if not rep.get("runnable"):
        raise RuntimeError(f"机器人 {robot} 不可直接运行: {rep.get('reasons') or rep.get('reason')}")

    cfg = dict(ROBOTS[robot])
    cfg.update(overrides or {})
    arm = cfg.get("arm", "left")
    urdf = rep["urdf"]

    # ---- 资产 ----
    # 预 absolutize：兼容 `meshes/xxx.obj`(无 ./) 与 `./meshes/xxx` 两种写法
    abs_urdf, n_rewritten = absolutize_urdf(urdf, f"/tmp/robot_abs_{robot}.urdf")
    G.R1_DIR = os.path.dirname(urdf)
    G.R1_URDF = abs_urdf
    print(f"[robot_config] {robot}: URDF 已 absolutize（重写 {n_rewritten} 个资源路径）")
    # ---- 臂/夹爪命名 ----
    G.ARM = arm
    G.ARM_JOINTS = [f"{arm}_arm_joint{i}" for i in range(1, 7)]
    G.GRIP_LINK = f"{arm}_gripper_link"
    G.GRIP_FINGERS = (f"{arm}_gripper_finger_joint1", f"{arm}_gripper_finger_joint2")
    # ---- 关节限位(按 URDF 实测) ----
    G.IK_LIMITS = np.array([[lo, hi] for _, lo, hi in rep["limits"]], dtype=float)
    # ---- 基座 ----
    if cfg.get("base_p") is not None:
        G.R1_BASE_P = list(cfg["base_p"])
    if cfg.get("base_q") is not None:
        G.R1_BASE_Q = list(cfg["base_q"])
    # ---- 起始位 / 非工作臂 ----
    if cfg.get("left_start") is not None:
        G.R1_LEFT_START = np.asarray(cfg["left_start"], dtype=float)
    if cfg.get("right_park") is not None:
        G.R1_RIGHT_PARK = np.asarray(cfg["right_park"], dtype=float)
    # ---- 夹爪几何(换机器人必须重新标定) ----
    for k_src, k_dst in (("pad_fwd", "PAD_FWD"), ("grip_open_q", "GRIP_OPEN_Q"),
                         ("grip_hold_q", "GRIP_HOLD_Q")):
        if cfg.get(k_src) is not None:
            setattr(G, k_dst, float(cfg[k_src]))
    # ---- base_xy 覆盖: R1_BASE_P 与 PARAM_DEFAULTS["base_xy"] 必须同步 ----
    # `build_episode` 实际读的是 PARAM_DEFAULTS["base_xy"], 而 R1_BASE_P 只是初始
    # 位姿常量。只改其一会造成"看起来改了、其实没改"(实测踩坑)。两者一起写。
    if overrides and overrides.get("base_xy") is not None:
        bx = [float(overrides["base_xy"][0]), float(overrides["base_xy"][1])]
        bz = G.R1_BASE_P[2] if len(getattr(G, "R1_BASE_P", [])) > 2 else 0.0
        G.R1_BASE_P = [bx[0], bx[1], bz]
        if hasattr(G, "PARAM_DEFAULTS"):
            G.PARAM_DEFAULTS["base_xy"] = [bx[0], bx[1]]
    return rep


if __name__ == "__main__":
    import json
    for r in ROBOTS:
        print(json.dumps(inspect_robot(r), ensure_ascii=False, indent=1)[:900])
        print()

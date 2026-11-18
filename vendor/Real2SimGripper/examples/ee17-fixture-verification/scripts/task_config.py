#!/usr/bin/env python3
"""task_config.py —— 任务配置加载器（换任务只改 JSON，不改代码）

设计原则（与 robot_config / collision_config 一致）：
  **不修改核心脚本**，只在其模块级全局上打补丁，再交给 build_episode(params) 执行。

用法：
    import robotwin_r1_duck_grasp as G
    from task_config import load_task, apply_task
    cfg = load_task("configs/duck_to_plate.json")
    apply_task(G, cfg)                      # 打补丁
    ep = G.build_episode(params={...})      # 照常跑

配置文件字段见 configs/duck_to_plate.json（每个字段都带 _comment 说明）。

已知边界：
  - `contacts` 目前只取**第一段**（单接触段）；多接触段（pick→place→pick→place）
    需要在核心脚本里支持多次抓放时序，属后续工作。
  - 机器人标定量（pad_fwd / open_q / hold_q）换夹爪必须重新实测，JSON 只是集中管理。
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_task(path):
    """读任务 JSON。相对路径按仓库根解析。"""
    if not os.path.isabs(path):
        for base in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
            p = os.path.join(base, path)
            if os.path.exists(p):
                path = p
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"任务配置不存在: {path}")
    with open(path) as f:
        raw = json.load(f)
    cfg = _strip_comments(raw)          # 去掉所有 _ 开头的注释键
    cfg["_path"] = path
    return cfg


def _strip_comments(obj):
    """递归剥掉以 `_` 开头的注释键。

    schema 里所有说明都用 `_comment` / `_xxx_comment` 表达；若不剥掉，
    `cfg["tuning"]` 会混进字符串，`AO.clamp()` 转 float 时直接崩
    （实测: ValueError: could not convert string to float: '调参项...'）。
    保留 `_path` 以外的 `_` 键没有必要，故统一清理。
    """
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _num(v, default):
    try:
        return float(v)
    except Exception:
        return float(default)


def apply_task(G, cfg, verbose=True):
    """把任务配置打到核心脚本 G 的模块级全局上。返回 patch 摘要。"""
    applied = []

    def _set(name, val):
        if hasattr(G, name):
            setattr(G, name, val)
            applied.append(name)
        else:
            applied.append(f"{name}(核心脚本无此名, 跳过)")

    # ---- 路径 ----
    sc = cfg.get("scene", {})
    if sc.get("glb"):
        _set("GLB", sc["glb"])
    ht = cfg.get("hand_traj", {})
    if ht.get("mano_npz"):
        _set("NPZ", ht["mano_npz"])
    rb = cfg.get("robot", {})
    if rb.get("urdf"):
        _set("R1_URDF", rb["urdf"])
        _set("R1_DIR", os.path.dirname(rb["urdf"]))

    # ---- 物体 / 目标 ----
    ob = cfg.get("object", {})
    if ob.get("geom"):
        _set("DUCK_GEOM", str(ob["geom"]))
    if ob.get("spawn_xy"):
        _set("DUCK_RB_XY", np.asarray(ob["spawn_xy"], dtype=float))
    if sc.get("table_z") is not None:
        _set("TABLE_Z", _num(sc["table_z"], 0.74))
    if ht.get("window"):
        _set("WIN", tuple(int(x) for x in ht["window"]))

    # ---- 夹爪几何/驱动 ----
    gr = rb.get("gripper", {})
    if gr.get("pad_fwd") is not None:
        _set("PAD_FWD", _num(gr["pad_fwd"], 0.0369))
    if gr.get("stiffness") is not None and gr.get("damping") is not None:
        _set("ARM_STIFF", _num(gr["stiffness"], 1000.0))
        _set("ARM_DAMP", _num(gr["damping"], 200.0))

    # ---- 抓放帧: 取 contacts 第一段 ----
    contacts = ht.get("contacts") or []
    if contacts:
        c0 = contacts[0]
        if c0.get("close_f") is not None:
            _set("CLOSE_F", int(c0["close_f"]))
        if c0.get("open_f") is not None:
            _set("OPEN_F", int(c0["open_f"]))

    # ---- 合并进 PARAM_DEFAULTS（核心脚本以此为参数默认值）----
    P = dict(getattr(G, "PARAM_DEFAULTS", {}))
    P.setdefault("base_xy", list(rb.get("base_xy", [0.12, -0.40])))
    if rb.get("base_xy"):
        P["base_xy"] = list(rb["base_xy"])
    if gr.get("open_q") is not None:
        P["grip_open_q"] = _num(gr["open_q"], 0.05)
    if gr.get("hold_q") is not None:
        P["grip_hold_q"] = _num(gr["hold_q"], 0.0302)
    tg = cfg.get("target") or {}
    P["plate_geom"] = tg.get("geom") if tg else None
    if tg.get("reach_frac") is not None:
        P["place_t"] = float(np.clip(_num(tg["reach_frac"], 0.0), 0.0, 1.0))
    # ---- 调参项: 统一从顶层 tuning 段读(不要在 sim 里, 避免与物理参数混淆) ----
    tun = cfg.get("tuning", {})
    for k in ("time_scale", "place_hover", "place_lift", "d_hover",
              "ctrl_win", "ctrl_passes", "joint_win", "joint_passes", "ik_seeds"):
        if k in tun:
            P[k] = tun[k]
    for k in ("grasp_orient", "grasp_tilt_max", "grasp_fin", "grasp_carry_blend"):
        if k in ht:
            P[k] = ht[k]
    P["ik_seeds"] = G.validate_ik_seeds(tun.get("ik_seeds", G.IK_SEEDS))
    G.PARAM_DEFAULTS = P
    applied.append("PARAM_DEFAULTS")

    # ---- 道具碰撞: props_collision=true 时全部带碰撞 ----
    sim = cfg.get("sim", {})
    if sim.get("props_collision"):
        _orig = G.create_prop

        def _all_collide(scene, name, obj_path, pose, collide=False):
            return _orig(scene, name, obj_path, pose, collide=True)

        G.create_prop = _all_collide
        applied.append("create_prop→all-collide")

    # ---- 物体物理属性(质量/摩擦/阻尼) ----
    if ob.get("mass_kg") is not None or ob.get("friction") is not None:
        _duck0 = G.create_duck_actor

        def _duck_cfg(scene, pose):
            d = _duck0(scene, pose)
            try:
                for c in d.get_components():
                    if isinstance(c, G.sapienp.PhysxRigidDynamicComponent):
                        if ob.get("mass_kg") is not None:
                            c.mass = _num(ob["mass_kg"], 0.04)
                        if ob.get("linear_damping") is not None:
                            c.linear_damping = _num(ob["linear_damping"], 0.5)
                        if ob.get("angular_damping") is not None:
                            c.angular_damping = _num(ob["angular_damping"], 5.0)
            except Exception:
                pass
            return d

        G.create_duck_actor = _duck_cfg
        applied.append("create_duck_actor→物理属性")

    if verbose:
        print(f"[task_config] 已加载 {os.path.basename(cfg.get('_path', '?'))} "
              f"({cfg.get('task_id', '?')})")
        print(f"[task_config] 打补丁: {', '.join(applied)}")
    return applied


def summary(cfg):
    """人类可读的一页摘要，用于确认配置正确。"""
    ob, tg, ht, rb = cfg.get("object", {}), cfg.get("target") or {}, cfg.get("hand_traj", {}), cfg.get("robot", {})
    gr = rb.get("gripper", {})
    cs = ht.get("contacts") or [{}]
    lines = [
        f"任务        : {cfg.get('task_id')} — {cfg.get('description', '')}",
        f"场景 GLB    : {cfg.get('scene', {}).get('glb')}",
        f"抓取物体    : {ob.get('geom')} ({ob.get('name')}), 质量 {ob.get('mass_kg')} kg",
        f"放置目标    : {tg.get('geom', '无(纯跟随)')}  reach_frac={tg.get('reach_frac')}",
        f"人手轨迹    : {ht.get('mano_npz')}",
        f"  窗口      : {ht.get('window')}   臂: {ht.get('arm')}",
        f"  接触段    : {[(c.get('close_f'), c.get('open_f')) for c in cs]}"
        + (f"  (共 {len(cs)} 段)" if len(cs) > 1 else ""),
        f"机器人      : {rb.get('name')}  base_xy={rb.get('base_xy')}  URDF={rb.get('urdf')}",
        f"夹爪        : pad_fwd={gr.get('pad_fwd')}  open_q={gr.get('open_q')}  hold_q={gr.get('hold_q')}",
        f"碰撞        : mode={cfg.get('sim', {}).get('collision', {}).get('mode')} "
        f"threshold={cfg.get('sim', {}).get('collision', {}).get('threshold')}  "
        f"道具碰撞={cfg.get('sim', {}).get('props_collision')}",
        f"真值(GT)    : {cfg.get('demonstration_ground_truth', {}).get('normalised_offset')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    c = load_task(sys.argv[1] if len(sys.argv) > 1 else "configs/duck_to_plate.json")
    print(summary(c))

#!/usr/bin/env python3
"""collision_config.py — 碰撞几何配置：启用 COACD 凸分解（**不改核心脚本**）。

背景（实测 2026-10-16）
----------------------
`robotwin_r1_duck_grasp.py` 里用的是：
    add_multiple_convex_collisions_from_file(filename=..., decomposition="none")
                                                              ^^^^^^^^^^^^^^^^ 默认
`decomposition="none"` 意味着**不做凸分解**，实测粉鸭只得到 **1 个凸包**
= 整个鸭子外形的凸壳，所有凹陷（鸭嘴与身体之间、尾部）都被填平。

这也解释了为什么"减面 vs 不减面"对物理只有 1~4mm 影响 ——
**碰撞体是凸壳，网格细节根本没进物理**。

启用 COACD 后（同一份 12k 网格实测）：
    threshold=0.20 →  3 个凸包   26s
    threshold=0.10 →  3 个凸包   26s
    threshold=0.05 → 10 个凸包   43s   ← 推荐
    threshold=0.02 → 45 个凸包   60s

SAPIEN 自带缓存（`filename + ".coacd.ply"`，按 md5+参数校验），只算一次。

用法
----
    import robotwin_r1_duck_grasp as G
    from collision_config import enable_coacd
    enable_coacd(G, threshold=0.05)        # 猴子补丁替换 create_duck_actor / create_prop
    ep = G.build_episode(params={...})
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_state = {"enabled": False, "threshold": 0.05, "hulls": {}}


def _is_dynamic(comp, G):
    return isinstance(comp, G.sapienp.PhysxRigidDynamicComponent)


def make_create_duck_actor(G, threshold, max_depenetration_velocity=1.0):
    """返回启用 COACD 的 create_duck_actor 替换版。

    **必须同时限制 `max_depenetration_velocity`**（实测 2026-10-16）：
    SAPIEN 默认值是 `1e32`（等于无限）。COACD 的凸包比视觉网格向外膨胀约 2mm，
    夹爪按视觉宽度标定 → 多穿 0.74cm → PhysX 以无限去穿透速度把鸭子**弹飞到 29 米**。
    限到 1.0 m/s 后不再爆炸（抬升 11.57cm → 16.74cm）。
    """
    def create_duck_actor(scene, pose):
        b = scene.create_actor_builder()
        b.set_physx_body_type("dynamic")
        grip_mat = scene.create_physical_material(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        b.add_multiple_convex_collisions_from_file(
            filename=G.DUCK_OBJ, scale=(1, 1, 1), material=grip_mat,
            decomposition="coacd",
            decomposition_params=({"threshold": float(threshold)}
                                  if threshold is not None else None))
        vis_mat = G._textured_render_material(G.DUCK_TEX)
        if vis_mat is not None:
            b.add_visual_from_file(filename=G.DUCK_OBJ, scale=(1, 1, 1), material=vis_mat)
        else:
            b.add_visual_from_file(filename=G.DUCK_OBJ, scale=(1, 1, 1))
        duck = b.build(name="pink_duck")
        duck.set_pose(pose)
        for comp in duck.get_components():
            if _is_dynamic(comp, G):
                comp.mass = 0.04
                comp.angular_damping = 5.0
                comp.linear_damping = 0.5
                if max_depenetration_velocity is not None:
                    comp.set_max_depenetration_velocity(float(max_depenetration_velocity))
        try:
            dc = [c for c in duck.get_components() if _is_dynamic(c, G)][0]
            _state["hulls"]["duck"] = len(dc.get_collision_shapes())
            _state["duck_extent"] = None
        except Exception:
            pass
        return duck
    return create_duck_actor


def make_create_prop(G, threshold):
    """返回启用 COACD 的 create_prop 替换版（盘子是凹的，更需要凸分解）。"""
    def create_prop(scene, name, obj_path, pose, collide=False):
        b = scene.create_actor_builder()
        b.set_physx_body_type("static")
        vis_mat = G._textured_render_material(obj_path.replace(".obj", "_tex.png"))
        if vis_mat is not None:
            b.add_visual_from_file(filename=obj_path, scale=(1, 1, 1), material=vis_mat)
        else:
            b.add_visual_from_file(filename=obj_path, scale=(1, 1, 1))
        if collide:
            b.add_multiple_convex_collisions_from_file(
                filename=obj_path, scale=(1, 1, 1),
                decomposition="coacd",
                decomposition_params=({"threshold": float(threshold)}
                                      if threshold is not None else None))
        a = b.build(name=name)
        a.set_pose(pose)
        return a
    return create_prop


def enable_coacd(G, threshold=None, verbose=True, max_depenetration_velocity=1.0):
    """启用 COACD 凸分解。

    threshold 默认 **None = 不传, 用 SAPIEN 自己的默认(0.05)**。
    2026-10-22 核对：SAPIEN 会把 threshold 传给 COACD，不能宣称它普遍无效。
    同一输入 MD5 的历史缓存(threshold=0.20)有 3 个凸包，当前默认缓存有 8 个。
    缓存生成时的软件版本未知；这不是控制版本的阈值扫描，也不能单独归因 GT 差异。
    历史“0.05/0.10/0.20 都是 8 个”的观察不应外推，复现须记录参数及碰撞资产指纹。
    证据见 results/ik_agent_20261022/historical_collision_cache_audit.json。

    必须同时限制 `max_depenetration_velocity`(默认 1e32=无限): COACD 凸包比视觉网格
    外扩 ~2mm, 夹爪按视觉宽度标定 → 多穿 0.74cm → PhysX 以无限去穿透速度把鸭子弹飞到 29 米。
    """
    """把核心脚本的 create_duck_actor / create_prop 换成 COACD 版本。

    max_depenetration_velocity: 见 make_create_duck_actor 的说明，默认 1.0 m/s 防弹飞。
    """
    G.create_duck_actor = make_create_duck_actor(G, threshold, max_depenetration_velocity)
    G.create_prop = make_create_prop(G, threshold)
    _state["enabled"] = True
    _state["threshold"] = threshold          # None = 用 SAPIEN 默认 0.05
    _state["max_depenetration_velocity"] = max_depenetration_velocity
    if verbose:
        t = "SAPIEN 默认(0.05)" if threshold is None else f"{threshold}"
        print(f"[collision] COACD 已启用 threshold={t}, "
              f"max_depenetration_velocity={max_depenetration_velocity} "
              f"（碰撞体由 1 个凸壳 → 多个凸包；首次分解约 30~60s，之后走缓存）")
    return _state


def state():
    return dict(_state)


if __name__ == "__main__":
    import robotwin_r1_duck_grasp as G  # noqa: E402
    enable_coacd(G, 0.05)
    print(state())

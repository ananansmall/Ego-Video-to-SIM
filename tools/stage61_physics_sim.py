#!/usr/bin/env python3
"""Stage 6.1 — URDF 驱动的真实物理仿真 + GIF 输出.

修复 stage6 的四个问题:
  1. 质量/惯性: URDF 的 <inertial> 真正传给 SAPIEN (此前完全未读, SAPIEN 按包围盒
     默认密度自算, 导致 226 个 link 的 real_mass 全部作废)。若 URDF 的 inertia 是
     占位值 (ixx==iyy==izz 且无 xy/zx/yz 项), 则从 collision mesh 按质量重算。
  2. visual/collision 一致: 两者都用 collision mesh (凸包), 不再用包围盒色块。
     此前 visual 是包围盒、collision 是真mesh, 画面与物理形状不符。
  3. 不再掩盖穿地: 移除 z_offset 整体上抬, 改为如实报告初始穿透深度。
  4. 仿真必须真的跑: 显式 scene.step(), 并校验 actor 数量与初始穿透;
     若物体创建失败则显式报错, 而不是静默跳过。

坐标系: URDF 与 SAPIEN 3.x 均为 Z-up, 不做转换。

方案 B (--pyrender_render):
  部分场景 SAPIEN 渲染引擎内部与实体位姿同步脱钩 (物理正常但画面恒为纯灰,
  见 CHANGE_LOG "渲染确定性失效")。此模式**完全不使用 SAPIEN 渲染器**:
  SAPIEN 只跑物理并逐帧记录所有 actor 位姿, 由 pyrender 离线按真实 VGGT
  相机轨迹 (OpenCV 内参 -> OpenGL 离轴投影) 重建场景渲染 GIF。

Usage:
    python tools/stage61_physics_sim.py --output_dir output_v2/<scene>
    python tools/stage61_physics_sim.py --output_dir ... --n_frames 120 --trajectory
    python tools/stage61_physics_sim.py --output_dir ... --pyrender_render
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

# 必须在导入 pyrender/OpenGL 之前设置 (方案 B 用 EGL 离屏渲染)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

import sapien
import sapien.physx as physx

OBJ_COLORS = [
    [1.0, 0.25, 0.25, 1.0], [0.25, 0.8, 0.3, 1.0], [0.3, 0.35, 1.0, 1.0],
    [1.0, 0.8, 0.15, 1.0], [1.0, 0.45, 0.8, 1.0], [0.2, 0.95, 0.95, 1.0],
    [0.75, 0.5, 0.25, 1.0], [0.55, 0.55, 0.95, 1.0],
]


# ---------------------------------------------------------------- URDF 解析

def parse_urdf(urdf_path):
    """解析 scene.urdf -> (objects, base_link)

    每个 object 含: name, visual_mesh, collision_mesh, scale, world_pos, world_quat,
                    mass, inertia(6), mu1, mu2, inertia_is_placeholder
    """
    file_dir = os.path.dirname(os.path.abspath(urdf_path))
    root = ET.parse(urdf_path).getroot()

    joint_info = {}
    for joint in root.findall(".//joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        o = joint.find("origin")
        a = o.attrib if o is not None else {}
        joint_info[child] = (
            parent,
            list(map(float, a.get("xyz", "0 0 0").split())),
            list(map(float, a.get("rpy", "0 0 0").split())),
        )

    def quat_mul_wxyz(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    def world_transform(name, visited=None):
        """累积世界变换, 返回 (t, q_wxyz)。所有 link 都挂在 base 下, 用不到关节链。"""
        if visited is None:
            visited = set()
        if name in visited or name not in joint_info:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        visited.add(name)
        parent, xyz, rpy = joint_info[name]
        ct = np.array(xyz, dtype=np.float64)
        cq = R.from_euler("xyz", rpy, degrees=False).as_quat()
        cq_wxyz = np.array([cq[3], cq[0], cq[1], cq[2]])
        if parent != "base" and parent in joint_info and parent not in visited:
            pt, pq = world_transform(parent, visited)
            return pt + R.from_quat([pq[1], pq[2], pq[3], pq[0]]).as_matrix() @ ct, \
                quat_mul_wxyz(pq, cq_wxyz)
        return ct, cq_wxyz

    def resolve(rel):
        return os.path.normpath(os.path.join(file_dir, rel)) if rel else None

    objects = []
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue

        # collision mesh 优先 —— 它才是参与物理的形状, visual 用它保证一致
        coll = link.find(".//collision/geometry/mesh")
        coll_path = resolve(coll.get("filename")) if coll is not None else None
        vis = link.find(".//visual/geometry/mesh")
        vis_path = resolve(vis.get("filename")) if vis is not None else None
        # collision 缺失时退化用 visual, 并在报告里标注
        mesh_path = coll_path or vis_path
        if not mesh_path or not os.path.exists(mesh_path):
            continue

        scale_a = (coll if coll is not None else vis).get("scale", "1 1 1")
        scale = np.array(list(map(float, scale_a.split())))

        mu1, mu2 = 0.6, 0.5
        gaz = link.find(".//collision/gazebo")
        if gaz is not None:
            try:
                if gaz.find("mu1") is not None:
                    mu1 = float(gaz.find("mu1").text)
                if gaz.find("mu2") is not None:
                    mu2 = float(gaz.find("mu2").text)
            except (TypeError, ValueError):
                pass

        mass, inertia, placeholder = None, None, False
        inert = link.find(".//inertial")
        if inert is not None:
            m_el = inert.find("mass")
            if m_el is not None:
                mass = float(m_el.get("value"))
            i_el = inert.find("inertia")
            if i_el is not None:
                g = lambda k: float(i_el.get(k, "0.0"))  # noqa: E731
                inertia = np.array([g("ixx"), g("iyy"), g("izz"),
                                    g("ixy"), g("ixz"), g("iyz")])
                # EmbodiedGen 写的是 ixx=iyy=izz=1.0 且无惯性积 -> 占位值
                placeholder = bool(
                    abs(inertia[0] - 1.0) < 1e-9 and abs(inertia[1] - 1.0) < 1e-9
                    and abs(inertia[2] - 1.0) < 1e-9
                    and abs(inertia[3]) < 1e-12 and abs(inertia[4]) < 1e-12
                    and abs(inertia[5]) < 1e-12)
        if mass is None or mass <= 0:
            mass = 0.1

        wt, wq = world_transform(name)
        objects.append(dict(
            name=name, mesh_path=mesh_path, coll_path=coll_path,
            visual_mesh=(vis_path if vis_path and os.path.exists(vis_path) else None),
            used_visual_fallback=coll_path is None,
            scale=scale, world_pos=wt, world_quat=wq,
            mass=float(mass), inertia=inertia, inertia_is_placeholder=placeholder,
            mu1=mu1, mu2=mu2,
        ))

    return objects


# ------------------------------------------------------- 惯量 / 质量计算

def recompute_inertia(mesh_path, scale, mass):
    """从 mesh 按给定质量重算主惯量 (mesh 体积*密度=mass -> 密度=mass/volume)。

    URDF 里的 ixx=1.0 是占位值, 必须换掉, 否则转动惯量与质量不匹配,
    物体会以错误的角加速度翻滚。
    """
    m = trimesh.load(mesh_path, process=False)
    if not isinstance(m, trimesh.Trimesh) or m.volume <= 0 or not m.is_watertight:
        # 非封闭 mesh 体积不可靠 -> 用包围盒近似
        ext = np.ptp(m.vertices, axis=0) * scale
        box = trimesh.creation.box(extents=np.maximum(ext, 1e-4))
        m = box
    density = mass / m.volume
    # trimesh 的 moment_inertia 针对单位密度, 乘 density 得实际惯量
    I = np.asarray(m.moment_inertia, dtype=np.float64) * density
    return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


def mesh_volume(mesh_path, scale):
    """返回 (volume_m3, reliable)。非封闭 mesh 的体积不可信 -> reliable=False。

    用于把 Stage 6 的密度换算成质量: m = rho * V。URDF 里的 mass 是 VLM 对
    "这个物体大概多重"的猜测, 而 密度×真实体积 是物理量关系, 只要
    stage4 的尺度重投影没错, 后者更准。可靠性由调用方决定要不要用。
    """
    m = trimesh.load(mesh_path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        return None, False
    reliable = bool(m.is_watertight) and float(m.volume) > 0
    vol = float(m.volume) * float(np.prod(scale)) if reliable else None
    return vol, reliable


# ------------------------------------------- Stage 6 -> 6.1 对接 (关键)

def load_materials(out_dir):
    """读 Stage 6 的产物 `<scene>/materials.json`。

    边界: Stage 6 (tools/material_annotator.py) 输出的 key 形如
    `"bowl_inst0"`, 而 Stage 6.1 的 URDF link 名形如 `"bowl_0"` ——
    两边命名约定不一致, 这正是"Stage 6 标注结果无人消费"在代码层面
    的直接原因: 没有任何地方做过 `inst<N>` -> `_N` 的映射。返回值仍是
    原始 dict, 解析交给 resolve_material。
    """
    p = os.path.join(out_dir, "materials.json")
    if not os.path.exists(p):
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, p
    if not isinstance(data, dict) or not data:
        return None, p
    return data, p


def resolve_material(link_name, materials):
    """URDF link 名 -> Stage 6 的材质条目。返回 (entry, how)。

    how:
      exact             类别+实例编号精确命中 (bowl_0 -> bowl_inst0)
      category_uniform  该类别下所有实例的材质完全相同, 按类别取值 (安全)
      missing           无法对应 (Stage 6 没跑到这个物体)
    """
    if not materials:
        return None, "no_materials"
    if link_name in materials:
        return materials[link_name], "exact"
    if "_" not in link_name:
        return None, "missing"
    cat, idx = link_name.rsplit("_", 1)
    if not idx.isdigit():
        return None, "missing"
    key = f"{cat}_inst{int(idx)}"
    if key in materials:
        return materials[key], "exact"

    # 实例编号对不上 (Stage 6 与 Stage 6.1 的实例去重顺序可能不一致):
    # 只有该类别下材质完全一致时才按类别取值, 否则宁可放弃
    hits = [v for k, v in materials.items() if k.startswith(f"{cat}_inst")]
    if not hits:
        return None, "missing"
    sig = {(v.get("material"), v.get("density"), v.get("static_friction"),
            v.get("dynamic_friction"), v.get("restitution")) for v in hits}
    return (hits[0], "category_uniform") if len(sig) == 1 else (None, "missing")


def attach_materials(objects, materials, materials_path, mass_source="vlm"):
    """把 Stage 6 的材质挂到每个 URDF 物体上, 并统计覆盖率。

    做两件事:
      1. **摩擦 / 恢复系数** —— 无条件采用 Stage 6 的值 (URDF 里是 EmbodiedGen
         硬编码的 mu1=0.60/mu2=0.50/restitution=0.02, 与真实材质无关)。
      2. **质量** —— 按 mass_source 决定:
           vlm      (默认) 用 URDF 里 VLM 猜的质量
           density         用 Stage 6 密度 × mesh 真实体积
         默认不用密度法, 是因为 mesh 体积在这个管线里不可信: SAM3D 输出的
          absolute scale 有 1.5x 级漂移 (见 docs/SAM3D_SCALE_DRIFT_REPORT.md),
         实测 `newspaper_1` 会算出 10.9kg、`table_0` 35kg —— 比 VLM 猜的更离谱。

    两种质量都会写进报告, 且它们的比值本身是有用的诊断量:
    `ratio^(1/3)` ≈ mesh 的线性尺度误差倍数。
    """
    stats = dict(
        materials_path=materials_path,
        materials_available=bool(materials),
        n_materials_entries=len(materials) if materials else 0,
        n_objects=len(objects),
        n_matched=0, n_matched_exact=0, n_category_uniform=0, n_missing=0,
        n_mass_from_density=0, n_mass_from_vlm=0, n_volume_unreliable=0,
        mass_source=mass_source,
        materials=[],
    )
    ratios = []
    for obj in objects:
        entry, how = resolve_material(obj["name"], materials)
        obj["material"] = entry
        obj["material_source"] = how
        obj["mass_urdf"] = float(obj["mass"])   # VLM 猜的原始质量, 留作对照
        obj["mass_from_density"] = None
        obj["volume_m3"] = None
        obj["mass_source"] = "urdf_vlm"

        if entry:
            stats["n_matched"] += 1
            if how == "exact":
                stats["n_matched_exact"] += 1
            else:
                stats["n_category_uniform"] += 1
            rho = entry.get("density")
            vol, ok = mesh_volume(obj["mesh_path"], obj["scale"])
            if not ok:
                stats["n_volume_unreliable"] += 1
            elif rho and vol and vol > 0:
                obj["volume_m3"] = vol
                obj["mass_from_density"] = float(rho) * vol
                if obj["mass_urdf"] > 0:
                    ratios.append(obj["mass_from_density"] / obj["mass_urdf"])

        if mass_source == "density" and obj["mass_from_density"]:
            obj["mass"] = obj["mass_from_density"]
            obj["mass_source"] = "stage6_density"
            stats["n_mass_from_density"] += 1
        else:
            stats["n_mass_from_vlm"] += 1

        stats["materials"].append(dict(
            name=obj["name"],
            material=(entry or {}).get("material"),
            how=how,
            mass_final=round(float(obj["mass"]), 6),
            mass_source=obj["mass_source"],
            mass_urdf=round(obj["mass_urdf"], 6),
            mass_from_density=(None if obj["mass_from_density"] is None
                               else round(obj["mass_from_density"], 6)),
            volume_m3=(None if obj["volume_m3"] is None
                       else round(obj["volume_m3"], 8)),
            mu1=(entry or {}).get("static_friction"),
            mu2=(entry or {}).get("dynamic_friction"),
            restitution=(entry or {}).get("restitution"),
        ))

    if ratios:
        med = float(np.median(ratios))
        stats["density_vs_vlm"] = dict(
            n_compared=len(ratios),
            ratio_median=round(med, 3),
            implied_linear_scale_error=round(med ** (1.0 / 3.0), 3),
            note="ratio=mass_density/mass_vlm; 其立方根≈mesh 线性尺度误差。"
                 " SAM3D 已知有 ~1.5x 级漂移 (docs/SAM3D_SCALE_DRIFT_REPORT.md)",
        )
    return stats


def mesh_bottom_z(mesh_path, scale, world_pos, world_quat):
    """物体在世界系里的最低 z (用于报告初始穿透)。"""
    m = trimesh.load(mesh_path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        return None
    v = np.asarray(m.vertices) * scale
    Rm = R.from_quat([world_quat[1], world_quat[2], world_quat[3], world_quat[0]]).as_matrix()
    return float((Rm @ v.T).T[:, 2].min() + world_pos[2])


# ------------------------------------------------------------ 场景构建

def setup_scene(renderer=None, cpu_physx=False):
    # 实测: sapien.Scene() 的默认 PhysX 后端就是 PhysxCpuSystem (不是 GPU),
    # 所以 --cpu_physx 与默认等价, 仅为显式声明/排查保留。
    if cpu_physx:
        scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
    else:
        scene = sapien.Scene()   # 默认 PhysX 后端 = PhysxCpuSystem
    scene.set_ambient_light([0.45, 0.45, 0.45])
    scene.add_directional_light(direction=[0.3, 0.2, -1.0], color=[2.2, 2.15, 2.0],
                                shadow=True, shadow_map_size=2048)
    scene.add_directional_light(direction=[-1, -0.4, -0.6], color=[0.7, 0.7, 0.8])
    scene.add_directional_light(direction=[0.6, -0.8, 0.4], color=[0.6, 0.65, 0.7])
    return scene


def build_actors(scene, objects, renderer, verbose=True, force_box_visual=False):
    """把 URDF 物体装进 SAPIEN。返回 (loaded, failed) 两个列表。

    visual 与 collision 都用 collision mesh —— 保证画面与物理一致。
    force_box_visual: 调试用, 强制包围盒视觉, 二分 visual loader 问题。
    renderer=None: 只要物理不要渲染 (方案 B, 完全绕开 SAPIEN 渲染器);
                   此时不创建任何 visual 组件, 画面交给 pyrender 离线渲染。
    """
    loaded, failed = [], []
    for i, obj in enumerate(objects):
        # 摩擦/恢复系数: URDF 里是 EmbodiedGen 硬编码的 mu1=0.6/mu2=0.5、
        # restitution=0.02; 有 Stage 6 的 materials.json 就用 Stage 6 的值。
        entry = obj.get("material")
        mu1 = float(entry.get("static_friction", obj["mu1"])) if entry else obj["mu1"]
        mu2 = float(entry.get("dynamic_friction", obj["mu2"])) if entry else obj["mu2"]
        rest = float(entry.get("restitution", 0.02)) if entry else 0.02
        mu1 = float(np.clip(mu1, 0.1, 0.9))
        mu2 = float(np.clip(mu2, 0.1, 0.8))
        rest = float(np.clip(rest, 0.0, 0.9))
        obj["mu1_effective"], obj["mu2_effective"] = mu1, mu2
        obj["restitution_effective"] = rest
        mat = physx.PhysxMaterial(
            static_friction=mu1, dynamic_friction=mu2, restitution=rest,
        )
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        scale = obj["scale"]
        try:
            builder.add_multiple_convex_collisions_from_file(
                filename=obj["mesh_path"], material=mat, scale=scale)
        except Exception:
            try:
                builder.add_convex_collision_from_file(
                    filename=obj["mesh_path"], material=mat, scale=scale)
            except Exception as e:
                failed.append((obj["name"], f"collision 加载失败: {e}"))
                continue

        # inertia: 占位值 -> 按质量重算
        inertia = obj["inertia"]
        if obj["inertia_is_placeholder"] or inertia is None:
            try:
                inertia = recompute_inertia(obj["mesh_path"], scale, obj["mass"])
            except Exception as e:
                failed.append((obj["name"], f"惯量重算失败: {e}"))
                continue

        # visual 与 collision 都用同一个 mesh —— 保证画面里看到的形状
        # 就是参与物理碰撞的形状。实测(同相机同物体, 只换visual 类型):
        #   mesh visual -> 386 种颜色; box visual -> 仅 6 种(包围盒代理)
        # 所以视觉用真 mesh, 不再退回包围盒。
        if renderer is None:
            # 方案 B: 纯物理, 不建 visual (渲染由 pyrender 另做)
            obj["visual_is_mesh"] = False
        else:
            cm = renderer.create_material()
            cm.base_color = OBJ_COLORS[i % len(OBJ_COLORS)]
            cm.roughness, cm.metallic = 0.6, 0.0
            visual_is_mesh = True
            if force_box_visual:
                visual_is_mesh = False  # 调试: 强制包围盒, 二分 visual loader 问题
            if visual_is_mesh:
                try:
                    builder.add_visual_from_file(filename=obj["mesh_path"], scale=scale,
                                                 material=cm)
                except Exception:
                    visual_is_mesh = False
            if not visual_is_mesh:
                # 退回包围盒 (报告里以 n_visual_fallback 标注)
                half = np.maximum(np.ptp(
                    trimesh.load(obj["mesh_path"], process=False).vertices, axis=0
                ) * scale / 2.0, 0.02)
                builder.add_box_visual(half_size=half.tolist(), material=cm)
                obj["half_size"] = half
            obj["visual_is_mesh"] = visual_is_mesh

        try:
            actor = builder.build(name=obj["name"])
        except Exception as e:
            failed.append((obj["name"], f"build 失败: {e}"))
            continue

        # 关键: 质量与惯量真正生效。
        # SAPIEN 3.x 没有 set_auto_compute_mass, 但 set_mass/set_inertia 会覆盖
        # 自动计算值(实测: 不设时 cabinet_0 被算成 71.36kg, 设为 URDF 的 6.2114kg)。
        # set_inertia 只接受 3 个主对角项 (float32), 惯性积由 mesh 对称化忽略。
        try:
            rb = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if rb is None:
                failed.append((obj["name"], "取不到 PhysxRigidDynamicComponent"))
                continue
            rb.set_mass(float(obj["mass"]))
            rb.set_inertia(np.asarray(inertia[:3], dtype=np.float32))
            # 限制去穿速度: 初始穿模的物体接触时, PhysX 默认允许极大分离速度,
            # 会把轻物体(如 0.7g 的魔方)瞬间弹飞(实测 600 步飞出 61m)。
            # 限到 0.5m/s 让去穿平滑进行, 不破坏物理真实性。
            try:
                rb.set_max_depenetration_velocity(0.5)
            except Exception:
                pass
            # 回读校验, 确认真的写进去了 (而不是静默忽略)
            got_m = float(rb.get_mass())
            if abs(got_m - obj["mass"]) > max(1e-3, 0.01 * obj["mass"]):
                failed.append((obj["name"],
                               f"质量未生效: 期望 {obj['mass']:.4f} 实得 {got_m:.4f}"))
                continue
            obj["mass_effective"] = got_m
            obj["inertia_effective"] = np.asarray(rb.get_inertia(), dtype=float)
        except Exception as e:
            failed.append((obj["name"], f"质量/惯量设置失败: {e}"))
            continue

        actor.set_pose(sapien.Pose(p=obj["world_pos"].tolist(),
                                   q=obj["world_quat"].tolist()))
        loaded.append((obj, actor))

    if verbose:
        print(f"  装入 {len(loaded)} 个物体, 失败 {len(failed)} 个")
        for n, why in failed:
            print(f"    [FAIL] {n}: {why}")
    return loaded, failed


# ------------------------------------------------------------ 相机与输出

def look_at_pose(eye, target, world_up=np.array([0.0, 0.0, 1.0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    right = np.cross(f, world_up)
    if np.linalg.norm(right) < 1e-6:                 # 视线与 up 平行 -> 换一个 up
        right = np.cross(f, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, f)
    rot = np.column_stack([right, up, -f])
    xyzw = R.from_matrix(rot).as_quat()
    return sapien.Pose(p=eye.tolist(), q=[xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def load_camera_trajectory(out_dir):
    """读 VGGT 的 extrinsics 还原真实相机轨迹。

    extrinsics/{i}.txt 是 w2c (p_cam = R @ p_world + t), 相机世界位置 = -R.T @ t。
    场景 stage2 已用 R,t 把点云对齐到 z-up, 所以这里的相机位置与 URDF 同坐标系,
    可直接作为 SAPIEN 相机位姿 —— 这样 GIF 的视角与真实拍摄视频一致。

    注意: 不能用包围盒自动取景。151 的场景里物体到质心仅 0.02~0.20m, 而物体
    本身半米级, 自动取景会把相机塞进物体内部 (画面全灰)。真实相机在场景外面。
    """
    ext_dir = os.path.join(out_dir, "extrinsics")
    if not os.path.isdir(ext_dir):
        return None
    files = sorted((f for f in os.listdir(ext_dir) if f.endswith(".txt")),
                   key=lambda s: int(os.path.splitext(s)[0]))
    cams = []
    for f in files:
        e = np.loadtxt(os.path.join(ext_dir, f))
        R, t = e[:3, :3], e[:3, 3]
        # OpenCV 约定: 相机光轴在相机系是 +z
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        cams.append(dict(pos=-R.T @ t, fwd=fwd, w2c=e))
    if not cams:
        return None
    return cams


def load_intrinsic(out_dir):
    """读 VGGT intrinsic.txt (3x3, OpenCV 约定: 光轴 +z, x 右, y 下)。"""
    p = os.path.join(out_dir, "intrinsic.txt")
    if not os.path.exists(p):
        return None
    try:
        K = np.loadtxt(p)
    except Exception:
        return None
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        return None
    return K


def detect_image_size(out_dir, K, fallback):
    """真实图像尺寸: 优先读 color/0.*, 否则由主点反推 (2cx, 2cy)。"""
    cdir = os.path.join(out_dir, "color")
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                im = cv2.imread(os.path.join(cdir, f))
                if im is not None:
                    return im.shape[1], im.shape[0]
    if K is not None and K[0, 2] > 0 and K[1, 2] > 0:
        return int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
    return fallback


# ------------------------------------------------ 方案 B: pyrender 离线渲染

def _obj_c2w_gl(w2c):
    """OpenCV w2c (4x4) -> pyrender 相机节点矩阵 (OpenGL 相机系, camera->world)。

    相机系约定差一个 Y/Z 翻转: c_gl = diag(1,-1,-1) @ c_cv,
    故 cam_matrix = c2w_cv @ diag(1,-1,-1,1)。
    """
    Rm, t = w2c[:3, :3], w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = Rm.T
    c2w[:3, 3] = -Rm.T @ t
    return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])


def _look_at_c2w_gl(eye, target, up_world):
    """Z-up 世界里的 look-at -> pyrender 相机节点矩阵 (OpenCV 内参约定)。"""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f = f / np.linalg.norm(f)
    up_world = np.asarray(up_world, dtype=np.float64)
    x = np.cross(f, up_world)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(f, np.array([0.0, 1.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)          # CV 约定: y 向下
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, f
    c2w[:3, 3] = eye
    return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])


def _light_pose(direction):
    """DirLight 沿自身 -Z 发光; direction 为光线传播方向 (世界系)。"""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    z = -d
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    return T


def _load_render_mesh(obj, mesh_kind):
    """返回 (vertices(N,3) 已乘 scale, faces(M,3)) 或 None。"""
    path = obj["mesh_path"]
    if mesh_kind == "visual" and obj.get("visual_mesh"):
        path = obj["visual_mesh"]
    try:
        m = trimesh.load(path, process=False)
    except Exception:
        return None
    if isinstance(m, trimesh.Scene):
        geos = [g for g in m.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geos:
            return None
        m = trimesh.util.concatenate(geos)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        return None
    v = np.asarray(m.vertices, dtype=np.float64) * np.asarray(obj["scale"], dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    return v, f


def pyrender_render_frames(out_dir, loaded, poses_per_frame, real_cams, K,
                           img_w, img_h, mesh_kind="visual"):
    """用 pyrender 离线渲染每一帧。返回 (frames_rgb, cover_ratio_array)。"""
    import pyrender

    span_pts = np.array([o["world_pos"] for o, _ in loaded])
    center = span_pts.mean(axis=0)
    radius = max(float(np.linalg.norm(span_pts - center, axis=1).max()), 0.3)

    sc = pyrender.Scene(ambient_light=[0.45, 0.45, 0.45], bg_color=[0.97, 0.97, 0.97])

    node_idx, n_mesh_fail = [], []
    for i, (o, _) in enumerate(loaded):
        vf = _load_render_mesh(o, mesh_kind)
        if vf is None:
            node_idx.append(None)
            n_mesh_fail.append(o["name"])
            continue
        v, f = vf
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=OBJ_COLORS[i % len(OBJ_COLORS)],
            roughnessFactor=0.65, metallicFactor=0.0, doubleSided=True)
        mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)
        node_idx.append(sc.add(mesh, pose=np.eye(4)))

    # 地面: 这块 trimesh 版本没有 creation.plane, 用扁平 box 代替 (顶面 z=0)。
    # 边长要足够大, 否则会看到平面边缘 (看起来像"悬浮的地板")。
    L = max(radius * 40.0, 60.0)
    gbox = trimesh.creation.box(extents=[L, L, 0.02])
    gmat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.62, 0.62, 0.62, 1.0], roughnessFactor=0.95,
        metallicFactor=0.0)
    gpose = np.eye(4)
    gpose[0, 3], gpose[1, 3] = center[0], center[1]
    gpose[2, 3] = -0.01
    sc.add(pyrender.Mesh.from_trimesh(gbox, material=gmat, smooth=False), pose=gpose)

    sc.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.6),
           pose=_light_pose([-0.30, -0.20, -1.0]))
    sc.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.1),
           pose=_light_pose([1.0, 0.40, 0.60]))
    sc.add(pyrender.DirectionalLight(color=[0.90, 0.93, 1.0], intensity=0.7),
           pose=_light_pose([-0.60, 0.80, -0.40]))

    cam = make_intrinsic_camera(K, img_w, img_h, znear=0.01,
                               zfar=max(500.0, radius * 200))
    # 注意: Scene.add_node 不返回 node (只有 Scene.add 返回), 必须自己持引用
    cam_node = pyrender.Node(camera=cam, matrix=np.eye(4))
    sc.add_node(cam_node)

    n_frames = len(poses_per_frame)
    viewer = pyrender.OffscreenRenderer(img_w, img_h)
    frames, cover = [], []
    try:
        for i in range(n_frames):
            for j, node in enumerate(node_idx):
                if node is None:
                    continue
                p, q = poses_per_frame[i][j]
                T = np.eye(4)
                T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                T[:3, 3] = p
                sc.set_pose(node, T)
            if real_cams:
                ci = int(round(i * (len(real_cams) - 1) / max(n_frames - 1, 1)))
                sc.set_pose(cam_node, _obj_c2w_gl(real_cams[ci]["w2c"]))
            else:
                ang = np.radians(-125.0) + 2 * np.pi * i / n_frames
                elev = np.radians(32.0)
                d = np.array([np.cos(elev) * np.cos(ang),
                              np.cos(elev) * np.sin(ang), np.sin(elev)])
                sc.set_pose(cam_node,
                            _look_at_c2w_gl(center + d * radius * 3.0, center,
                                            np.array([0.0, 0.0, 1.0])))
            color, _ = viewer.render(sc)
            rgb = color[..., :3]
            frames.append(rgb.astype(np.uint8))
            a = rgb.astype(np.int16)
            cover.append(float(((a.max(axis=2) - a.min(axis=2)) > 25).mean()))
            if i % 20 == 0:
                print(f"  pyrender frame {i}/{n_frames}")
    finally:
        viewer.delete()

    return frames, np.array(cover), n_mesh_fail


class _IntrinsicCamera:
    """pyrender 相机: 直接用 VGGT 的 OpenCV 内参 (允许非对称主点)。

    pyrender 自带的 PerspectiveCamera 只能表达对称视锥, 而真实内参的
    (cx, cy) 不保证在画面正中, 故直接按 K 构造 OpenGL 投影矩阵。
    """

    def _make_cls():
        import pyrender

        class _C(pyrender.Camera):
            def __init__(self, K, width, height, znear=0.01, zfar=100.0, name=None):
                super().__init__(znear=znear, zfar=zfar, name=name)
                self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
                self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
                self.width, self.height = int(width), int(height)

            def get_projection_matrix(self, width=None, height=None):
                # 世界 -> OpenGL 相机系 -> NDC。GL 相机沿 -Z 看, 故与 CV 系差 Y/Z 翻转。
                w = self.width if width is None else width
                h = self.height if height is None else height
                P = np.zeros((4, 4), dtype=np.float64)
                # 注意: P[1,1] 必须为正 (与 pyrender 标准 1/tan(fov/2) 同号)。
                # 写成负会让画面绕主点上下翻转。
                P[0, 0] = 2.0 * self.fx / w
                P[1, 1] = 2.0 * self.fy / h
                P[0, 2] = 2.0 * self.cx / w - 1.0
                P[1, 2] = 2.0 * self.cy / h - 1.0
                n, f = self.znear, self.zfar
                P[2, 2] = -(f + n) / (f - n)
                P[2, 3] = -2.0 * f * n / (f - n)
                P[3, 2] = -1.0
                return P

        return _C


def make_intrinsic_camera(K, width, height, znear, zfar):
    return _IntrinsicCamera._make_cls()(K, width, height, znear=znear, zfar=zfar)


def fit_camera(loaded, margin=2.6):
    """loaded: build_actors 返回的 [(obj, actor), ...]"""
    pts = np.array([o["world_pos"] for o, _ in loaded])
    center = pts.mean(axis=0)
    radius = max(np.linalg.norm(pts - center, axis=1).max(), 0.3)
    dist = radius * margin
    # 侧上方俯视: 与 z 轴成 elev 角, 沿 +y 后退 (SAPIEN z-up)
    elev = np.radians(32.0)
    azim = np.radians(-125.0)
    d = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    return center + d * dist, center, radius


def main_pyrender(args, out_dir, objects, report):
    """方案 B 主流程: 纯物理仿真 + pyrender 离线渲染。

    为什么需要它: 3 个场景里 SAPIEN 渲染引擎与实体位姿同步脱钩 (物理与 mesh
    都正常, 画面恒为纯灰)。此路径完全不创建 SAPIEN 渲染器/渲染组件, 只保留
    PhysX 物理, 位姿交给 pyrender 重建, 从根本上避开该缺陷。
    """
    import imageio.v2 as imageio

    # ---- 1) 纯物理场景 (无渲染系统, 无 visual 组件) ----
    scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(1.0 / 240.0)
    scene.add_ground(0, render=False)
    print("\n[方案 B] 建立纯物理场景 (无 SAPIEN 渲染器, timestep=1/240)...")
    loaded, failed = build_actors(scene, objects, renderer=None)
    if not loaded:
        print("[ERROR] 没有任何物体成功装入, 拒绝输出 GIF")
        return 2
    if failed:
        print(f"  [WARN] {len(failed)} 个物体未能装入, 已在报告中记录")

    # ---- 2) 相机: 真实 VGGT 轨迹 + 真实内参 ----
    real_cams = None if args.no_vggt_cam else load_camera_trajectory(out_dir)
    K = load_intrinsic(out_dir)
    img_w, img_h = detect_image_size(out_dir, K, (args.img_w, args.img_h))
    if K is None:
        # 没有内参就退回对称视锥 (fovy 由命令行给)
        fy = (img_h / 2.0) / np.tan(np.deg2rad(args.fovy) / 2.0)
        K = np.array([[fy, 0.0, img_w / 2.0], [0.0, fy, img_h / 2.0], [0.0, 0.0, 1.0]])
        print("  [WARN] 无 intrinsic.txt, 用 fovy=%.1f° 对称内参" % args.fovy)
    else:
        print(f"  真实内参: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
              f"cx={K[0,2]:.1f} cy={K[1,2]:.1f} -> {img_w}x{img_h}")
    if real_cams:
        pts = np.array([c["pos"] for c in real_cams])
        print(f"  使用真实 VGGT 相机轨迹: {len(real_cams)} 帧, "
              f"位置范围 x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
              f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        cam_source = "vggt_extrinsics"
    else:
        print("  [WARN] 无 extrinsics, 退回包围盒环绕相机")
        cam_source = "bbox_orbit"

    # ---- 3) 仿真并逐帧记录所有 actor 位姿 ----
    n_frames = args.n_frames
    step_per_frame = max(args.sim_steps // n_frames, 1)
    print(f"\n仿真 {args.sim_steps} 步, 记录 {n_frames} 帧位姿 "
          f"(每帧 {step_per_frame} 步)...")
    z0 = {o["name"]: float(a.get_pose().p[2]) for o, a in loaded}
    poses_per_frame = []
    for i in range(n_frames):
        for _ in range(step_per_frame):
            scene.step()
        poses_per_frame.append([
            (np.asarray(a.get_pose().p, dtype=np.float64),
             np.asarray(a.get_pose().q, dtype=np.float64)) for _, a in loaded])
        if i % 20 == 0:
            print(f"  sim frame {i}/{n_frames}")

    zf = {o["name"]: float(poses_per_frame[-1][j][0][2])
          for j, (o, _) in enumerate(loaded)}
    moved = [n for n in zf if abs(zf[n] - z0[n]) > 0.005]
    report["render_backend"] = "pyrender_offscreen"
    report["sim"] = dict(
        n_loaded=len(loaded), n_failed=len(failed),
        failed=[{"name": n, "reason": w} for n, w in failed],
        sim_steps=args.sim_steps, n_frames=n_frames, camera_source=cam_source,
        n_real_cam_frames=len(real_cams) if real_cams else 0,
        z_before={k: round(v, 4) for k, v in z0.items()},
        z_after={k: round(v, 4) for k, v in zf.items()},
        n_moved=len(moved), moved=moved,
        physics_actually_ran=len(moved) > 0 or len(loaded) == 0,
        note="方案 B: 物理在 SAPIEN 中运行, 渲染由 pyrender 离线完成",
    )
    print(f"\n仿真结束: {len(moved)}/{len(loaded)} 个物体发生位移 (阈值 5mm)")

    # ---- 4) pyrender 离线渲染 ----
    print(f"\npyrender 渲染 {n_frames} 帧 ({img_w}x{img_h}, "
          f"mesh={args.render_mesh})...")
    frames, cover, mesh_fail = pyrender_render_frames(
        out_dir, loaded, poses_per_frame, real_cams, K, img_w, img_h,
        mesh_kind=args.render_mesh)

    # 渲染自检: 与主路径同一判据 (彩色像素占比 sat>25), 保证结论可比
    report["render_check"] = dict(
        colored_pixel_ratio_median=float(np.median(cover)),
        colored_pixel_ratio_min=float(cover.min()),
        n_frames_nearly_empty=int((cover < 0.005).sum()),
        empty_frame_indices=[int(i) for i in np.nonzero(cover < 0.005)[0]],
        mesh_load_failed=mesh_fail,
        note="彩色像素占比(sat>25)<0.5% 的帧视为空画面。"
             "空帧通常不是渲染失败, 而是该帧真实相机离被摄物极近 "
             "(如手持手机特写), 而 sim 里物体已落地/离场。",
    )
    if np.median(cover) < 0.005:
        print(f"\n[ERROR] 渲染自检失败: 彩色像素占比中位仅 {np.median(cover)*100:.2f}%")
        dbg = os.path.join(out_dir, "debug_render")
        os.makedirs(dbg, exist_ok=True)
        cv2.imwrite(os.path.join(dbg, "pyrender_frame0.png"),
                    cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        return 4
    print(f"渲染自检: 彩色像素占比中位 {np.median(cover)*100:.1f}%, "
          f"空画面帧 {int((cover<0.005).sum())}/{len(frames)}")

    # ---- 5) 输出 GIF / MP4 / 首帧 ----
    gif_path = os.path.join(out_dir, "stage61_physics.gif")
    stride = max(1, len(frames) // 120)
    gif_frames = frames[::stride]
    imageio.mimsave(gif_path, gif_frames, duration=1000.0 / args.fps, loop=0)
    mp4_path = os.path.join(out_dir, "stage61_physics.mp4")
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (img_w, img_h))
    for f in frames:
        # frames 是 RGB (渲染器输出), VideoWriter 期望 BGR —— 不转换会红蓝互换
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    png_path = os.path.join(out_dir, "stage61_physics_frame0.png")
    cv2.imwrite(png_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    print(f"\nGIF:  {gif_path}  ({len(gif_frames)} 帧 @ {args.fps}fps)")
    print(f"MP4:  {mp4_path}  ({len(frames)} 帧)")
    print(f"首帧: {png_path}")

    report["outputs"] = {"gif": gif_path, "mp4": mp4_path, "frame0": png_path,
                         "n_frames": n_frames, "fps": args.fps,
                         "img_w": img_w, "img_h": img_h,
                         "render_mesh": args.render_mesh}
    rp = os.path.join(out_dir, "stage61_physics_report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"报告: {rp}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sim_steps", type=int, default=600)
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--trajectory", action="store_true",
                    help="相机沿轨道环绕 (覆盖全部视角)")
    ap.add_argument("--report_only", action="store_true",
                    help="只做质量/惯量/穿地检查, 不跑仿真")
    ap.add_argument("--no_vggt_cam", action="store_true",
                    help="忽略 VGGT extrinsics, 用包围盒环绕相机")
    ap.add_argument("--force_box_visual", action="store_true",
                    help="调试: 强制包围盒视觉, 二分 visual loader 问题")
    ap.add_argument("--max_objects", type=int, default=0,
                    help="调试: 只装前 N 个物体 (0=全部), 二分毒化渲染的物体")
    ap.add_argument("--cpu_physx", action="store_true",
                    help="调试: 用 CPU PhysX (默认 GPU), 排查 GPU cooking 问题")
    ap.add_argument("--pyrender_render", action="store_true",
                    help="方案 B: SAPIEN 只跑物理记录位姿, pyrender 离线渲染 GIF "
                         "(绕开 SAPIEN 渲染引擎的确定性失效)")
    ap.add_argument("--render_mesh", choices=["visual", "collision"], default="visual",
                    help="方案 B: 渲染用 visual (真实外观) 还是 collision (物理形状)")
    ap.add_argument("--no_materials", action="store_true",
                    help="完全不读 Stage 6 的 materials.json (退回 URDF 的 VLM 质量 + "
                         "EmbodiedGen 硬编码 mu1=0.6/mu2=0.5/restitution=0.02)")
    ap.add_argument("--mass_source", choices=["vlm", "density"], default="vlm",
                    help="最终质量取哪个来源: vlm=URDF 里 VLM 猜的质量(默认, 稳); "
                         "density=Stage 6 密度×mesh 体积(物理正确但受尺度漂移污染)")
    ap.add_argument("--fovy", type=float, default=50.0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    urdf = os.path.join(out_dir, "urdf", "scene.urdf")
    if not os.path.exists(urdf):
        print(f"[ERROR] {urdf} 不存在")
        return 1

    print(f"解析 {urdf} ...")
    objects = parse_urdf(urdf)
    if args.max_objects > 0:
        objects = objects[:args.max_objects]
        print(f"  [调试] 只装前 {args.max_objects} 个物体")
    if not objects:
        print("[ERROR] URDF 里没解析到物体")
        return 1
    print(f"  {len(objects)} 个物体")

    # ---- Stage 6 -> 6.1 对接: 用 materials.json 覆盖质量 / 摩擦 / 恢复系数 ----
    mats, mats_path = (None, None) if args.no_materials else load_materials(out_dir)
    mat_stats = attach_materials(objects, mats, mats_path,
                                 mass_source=args.mass_source)
    if mat_stats["materials_available"]:
        dv = mat_stats.get("density_vs_vlm")
        print(f"  Stage 6 材质已接入: {os.path.basename(mats_path)}")
        print(f"    命中 {mat_stats['n_matched']}/{mat_stats['n_objects']} 个物体 "
              f"(精确 {mat_stats['n_matched_exact']}, 类别一致 {mat_stats['n_category_uniform']}), "
              f"未命中 {mat_stats['n_missing']}, 体积不可信 {mat_stats['n_volume_unreliable']}")
        print(f"    摩擦/恢复系数已按材质覆盖 (原为硬编码 0.60/0.50/0.02); "
              f"质量来源 {args.mass_source}")
        if dv:
            print(f"    [诊断] 密度质量/VLM 质量 中位 {dv['ratio_median']}x "
                  f"-> 隐含线性尺度误差 ≈ {dv['implied_linear_scale_error']}x "
                  f"({dv['n_compared']} 个物体可比)")
    else:
        print("  [WARN] 无 Stage 6 materials.json -> 质量仍用 URDF 的 VLM 值, "
              "摩擦仍用 EmbodiedGen 硬编码的 mu1=0.60/mu2=0.50/restitution=0.02 "
              "(先跑 mainv3 --enable_stage6 可补)")

    # ---- 事前检查: 质量/惯量/初始穿透 (不依赖仿真) ----
    precheck = []
    for o in objects:
        bz = mesh_bottom_z(o["mesh_path"], o["scale"], o["world_pos"], o["world_quat"])
        o["bottom_z"] = bz
        precheck.append(dict(
            name=o["name"], mass=o["mass"],
            inertia_placeholder=o["inertia_is_placeholder"],
            bottom_z=None if bz is None else round(bz, 4),
            penetration=None if bz is None else round(max(0.0, -bz), 4),
            collision_missing_used_visual=o["used_visual_fallback"],
        ))
    n_ph = sum(1 for p in precheck if p["penetration"] and p["penetration"] > 0.02)
    n_ph_mass = [p["mass"] for p in precheck]
    print(f"  质量范围 {min(n_ph_mass):.4f} ~ {max(n_ph_mass):.4f} kg "
          f"(跨度 {max(n_ph_mass)/max(min(n_ph_mass),1e-9):.0f} 倍)")
    print(f"  惯量为占位值(ixx=iyy=izz=1.0)的: "
          f"{sum(1 for p in precheck if p['inertia_placeholder'])}/{len(precheck)} -> 仿真时按质量重算")
    print(f"  初始穿透 >2cm 的: {n_ph}/{len(precheck)} (如实报告, 不再整体上抬掩盖)")

    report = dict(
        scene=os.path.basename(out_dir), n_objects=len(objects),
        n_inertia_placeholder=sum(1 for p in precheck if p["inertia_placeholder"]),
        n_penetrating=n_ph, penetration_objects=[p["name"] for p in precheck
                                                 if p["penetration"] and p["penetration"] > 0.02],
        objects=precheck, physics_backend=type(sapien.Scene()).__module__,
        materials_stage6=mat_stats,
    )
    if args.report_only:
        rp = os.path.join(out_dir, "stage61_physics_precheck.json")
        json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  已写出 {rp}")
        return 0

    # ---- 方案 B: SAPIEN 只跑物理, pyrender 离线渲染 ----
    if args.pyrender_render:
        return main_pyrender(args, out_dir, objects, report)

    # ---- 真实物理仿真 ----
    renderer = sapien.SapienRenderer()
    scene = setup_scene(renderer, cpu_physx=args.cpu_physx)
    scene.set_timestep(1.0 / 240.0)
    scene.add_ground(0, render=True)

    print(f"\n建立物理场景 (timestep=1/240)...")
    loaded, failed = build_actors(scene, objects, renderer,
                                  force_box_visual=args.force_box_visual)
    if not loaded:
        print("[ERROR] 没有任何物体成功装入, 拒绝输出 GIF "
              "(此前会静默跳过并宣称'success')")
        return 2
    if failed:
        print(f"  [WARN] {len(failed)} 个物体未能装入, 已在报告中记录")

    # 相机: 优先用真实 VGGT 轨迹 (覆盖全部相机帧), 否则退回包围盒环绕
    real_cams = None if args.no_vggt_cam else load_camera_trajectory(out_dir)
    eye, center, radius = fit_camera(loaded)
    if real_cams:
        pts = np.array([c["pos"] for c in real_cams])
        print(f"  使用真实 VGGT 相机轨迹: {len(real_cams)} 帧, "
              f"位置范围 x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
              f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        cam_source = "vggt_extrinsics"
    else:
        print("  [WARN] 无 extrinsics, 退回包围盒环绕相机")
        cam_source = "bbox_orbit"
    fovy = 50.0
    cam = scene.add_camera("cam", width=args.img_w, height=args.img_h,
                           fovy=np.deg2rad(fovy), near=0.005, far=max(60.0, radius * 20))
    if real_cams:
        c0 = real_cams[0]
        cam.set_entity_pose(look_at_pose(c0["pos"], c0["pos"] + c0["fwd"]))
    else:
        cam.set_entity_pose(look_at_pose(eye, center))
    print(f"  场景 center={np.round(center,3)} radius={radius:.2f}")

    z0 = {o["name"]: float(a.get_pose().p[2]) for o, a in loaded}

    # 渲染前置检查: 只报告相机与物体的距离, 不做拦截。
    # 曾有"相机落在 bbox 内 -> 画面全灰"的门控, 实测是错的: 3chair 的 120/120 帧
    # 相机都在 bbox 内, 但渲染完全正常(彩色占比 38.6%)。相机贴近物体是手持视频的
    # 正常情况, 不等于看不见物体 —— 画面里有没有东西最终由渲染自检判定。
    cam_check = {}
    if real_cams:
        pts = np.array([o["world_pos"] for o, _ in loaded])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        vpts = []
        for o, _ in loaded:
            try:
                mm = trimesh.load(o["mesh_path"], process=False)
                if isinstance(mm, trimesh.Trimesh):
                    vpts.append(np.asarray(mm.vertices) * o["scale"] + o["world_pos"])
            except Exception:
                pass
        surf_d = None
        if vpts:
            from scipy.spatial import cKDTree
            surf_d, _ = cKDTree(np.vstack(vpts)).query(
                np.array([c["pos"] for c in real_cams]))
        cam_check = dict(
            n_cams=len(real_cams),
            cam_inside_bbox=int(sum(1 for c in real_cams
                                    if np.all(c["pos"] >= lo) and np.all(c["pos"] <= hi))),
            scene_lo=lo.tolist(), scene_hi=hi.tolist(),
            cam_to_nearest_surface_m=(None if surf_d is None
                                      else float(np.median(surf_d))))
        if surf_d is not None:
            print(f"  相机到最近物体表面距离(中位): {np.median(surf_d):.3f}m")

    n_frames = args.n_frames
    step_per_frame = max(args.sim_steps // n_frames, 1)
    print(f"\n仿真 {args.sim_steps} 步, 渲染 {n_frames} 帧 "
          f"(每帧 {step_per_frame} 步)...")

    frames = []
    for i in range(n_frames):
        if real_cams:
            # 均匀采样覆盖全部真实相机帧 (用户要求: 相机都得全部渲染到)
            ci = int(round(i * (len(real_cams) - 1) / max(n_frames - 1, 1)))
            c = real_cams[ci]
            cam.set_entity_pose(look_at_pose(c["pos"], c["pos"] + c["fwd"]))
        elif args.trajectory:
            ang = np.radians(-125.0) + 2 * np.pi * i / n_frames
            elev = np.radians(32.0 + 10.0 * np.sin(2 * np.pi * i / n_frames))
            d = np.array([np.cos(elev) * np.cos(ang), np.cos(elev) * np.sin(ang),
                          np.sin(elev)])
            cam.set_entity_pose(look_at_pose(center + d * radius * 2.6, center))
        for _ in range(step_per_frame):
            scene.step()
        scene.update_render()
        cam.take_picture()
        img = cam.get_picture("Color")[..., :3]
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if i % 20 == 0:
            print(f"  frame {i}/{n_frames}")

    # ---- 仿真结果 ----
    zf, moved = {}, []
    for o, a in loaded:
        zf[o["name"]] = float(a.get_pose().p[2])
        if abs(zf[o["name"]] - z0[o["name"]]) > 0.005:
            moved.append(o["name"])
    report["sim"] = dict(
        n_loaded=len(loaded), n_failed=len(failed),
        failed=[{"name": n, "reason": w} for n, w in failed],
        sim_steps=args.sim_steps, n_frames=n_frames, camera_source=cam_source,
        camera_check=cam_check,
        n_real_cam_frames=len(real_cams) if real_cams else 0,
        z_before={k: round(v, 4) for k, v in z0.items()},
        z_after={k: round(v, 4) for k, v in zf.items()},
        n_moved=len(moved), moved=moved,
        physics_actually_ran=len(moved) > 0 or len(loaded) == 0,
    )
    n_vfall = sum(1 for o, _ in loaded if not o.get("visual_is_mesh", True))
    report["visual"] = dict(
        same_mesh_as_collision=True,
        n_visual_fallback=n_vfall,
        note="visual 与 collision 用同一 mesh; 数值仅统计 add_visual_from_file 失败而退回包围盒的物体")
    print(f"  visual: {len(loaded)-n_vfall}/{len(loaded)} 个物体用真 mesh 渲染"
          f" (与 collision 同一形状)")
    print(f"\n仿真结束: {len(moved)}/{len(loaded)} 个物体发生位移 "
          f"(阈值 5mm)")

    # ---- 渲染自检: 用"物体彩色"而非"与背景不同"来判定 ----
    # 教训: 地面和天空灰度也不同, 只按背景色差会把纯灰画面算成 47% 前景。
    # 物体用的是饱和色 (OBJ_COLORS), 故按"彩色像素"判定更可靠:
    #   max(RGB) - min(RGB) > 25  =>  像素不是灰的
    cover = []
    for f in frames:
        a = f.astype(np.int16)
        sat = a.max(axis=2) - a.min(axis=2)
        cover.append(float((sat > 25).mean()))
    cover = np.array(cover)
    report["render_check"] = dict(
        colored_pixel_ratio_median=float(np.median(cover)),
        colored_pixel_ratio_min=float(cover.min()),
        n_frames_nearly_empty=int((cover < 0.005).sum()),
        note="彩色像素占比(sat>25)<0.5% 的帧视为空画面")
    if np.median(cover) < 0.005:
        print(f"\n[ERROR] 渲染自检失败: 彩色像素占比中位仅 {np.median(cover)*100:.2f}%, "
              f"画面里没有物体(全灰)。拒绝输出无意义的 GIF。")
        print("        多数情况是上游 stage2 对齐错误 (如 R[2,2]>0 钝角, 重力方向错),")
        print("        导致相机与物体相对位置失真。")
        # ---- 调试二分: 俯视相机确认物体是否存在于渲染中 ----
        # 俯视能看到 -> 物体在渲染里, 是真实相机位姿与物体错位(坐标系问题);
        # 俯视也看不到 -> 物体根本没进渲染场景(visual 加载静默失败)。
        dbg_dir = os.path.join(out_dir, "debug_render")
        os.makedirs(dbg_dir, exist_ok=True)
        pts_now = np.array([a.get_pose().p for _, a in loaded])
        ctr_now = pts_now.mean(axis=0)
        # 组件级检查: actor 是否带渲染组件
        for o, a in loaded[:3]:
            comps = [type(c).__name__ for c in a.get_components()]
            print(f"        [调试] {o['name']} 组件: {comps}")
        # 终极二分: 加一个标准测试方块, 判断 scene 渲染是否整体失效
        probe = scene.create_actor_builder()
        probe.add_box_visual(half_size=[0.15, 0.15, 0.15])
        probe.add_box_collision(half_size=[0.15, 0.15, 0.15])
        probe_actor = probe.build(name="__probe__")
        probe_actor.set_pose(sapien.Pose(p=[ctr_now[0], ctr_now[1],
                                           pts_now[:, 2].max() + 0.3],
                                         q=[1, 0, 0, 0]))
        span = float(np.linalg.norm(pts_now.max(axis=0) - pts_now.min(axis=0)))
        dbg_cam = scene.add_camera(
            "dbg", width=args.img_w, height=args.img_h,
            fovy=np.deg2rad(60), near=0.005, far=max(80.0, span * 10))
        eye_dbg = ctr_now + np.array([0.0, 0.0, max(span * 1.5, 1.0)])
        want = look_at_pose(eye_dbg, ctr_now)
        dbg_cam.set_entity_pose(want)
        got = dbg_cam.get_pose()
        print(f"        [调试] dbg相机期望pose p={np.round(want.p,3)} q={np.round(want.q,4)}")
        print(f"        [调试] dbg相机实际pose p={np.round(got.p,3)} q={np.round(got.q,4)}")
        scene.update_render(); dbg_cam.take_picture()
        img_dbg = (np.clip(dbg_cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dbg_dir, "top_view.png"),
                    cv2.cvtColor(img_dbg, cv2.COLOR_RGB2BGR))
        # 用真实轨迹第一帧也存一张
        if real_cams:
            c0 = real_cams[0]
            dbg_cam.set_entity_pose(look_at_pose(c0["pos"], c0["pos"] + c0["fwd"]))
            scene.update_render(); dbg_cam.take_picture()
            img_c0 = (np.clip(dbg_cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dbg_dir, "realcam_frame0.png"),
                        cv2.cvtColor(img_c0, cv2.COLOR_RGB2BGR))
        a_dbg = img_dbg.astype(np.int16)
        sat_dbg = a_dbg.max(axis=2) - a_dbg.min(axis=2)
        top_ratio = float((sat_dbg > 25).mean())
        print(f"        [调试] 俯视相机彩色占比 {top_ratio*100:.1f}% "
              f"({'物体在渲染场景里, 是真实相机与物体错位' if top_ratio > 0.005 else '物体不在渲染场景, visual 加载静默失败'})")
        print(f"        [调试] 调试图已存: {dbg_dir}/top_view.png")
        print(f"        [调试] actor实际位置 bbox: "
              f"{np.round(pts_now.min(axis=0), 3)} -> {np.round(pts_now.max(axis=0), 3)}")
        # mesh 顶点的世界坐标 bbox: 判断物体是否其实在地下面/离相机视线太远
        from scipy.spatial import cKDTree
        vpts_dbg = []
        for o, a in loaded:
            try:
                mm = trimesh.load(o["mesh_path"], process=False)
                if isinstance(mm, trimesh.Trimesh):
                    pose = a.get_pose()
                    Rm = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
                    v = (Rm @ (np.asarray(mm.vertices) * o["scale"]).T).T + np.asarray(pose.p)
                    vpts_dbg.append(v)
            except Exception:
                pass
        if vpts_dbg:
            V = np.vstack(vpts_dbg)
            print(f"        [调试] mesh顶点世界bbox: "
                  f"{np.round(V.min(axis=0), 3)} -> {np.round(V.max(axis=0), 3)}"
                  f"  (地面 z=0, 顶点是否全部在地面下: {bool(V[:, 2].max() < 0)})")
            cam_pts = np.array([c["pos"] for c in real_cams]) if real_cams else pts_now
            d_surf, _ = cKDTree(V).query(cam_pts)
            print(f"        [调试] 相机到mesh顶点距离: min={d_surf.min():.3f} "
                  f"中位={np.median(d_surf):.3f} max={d_surf.max():.3f}")
        if real_cams:
            print(f"        [调试] 真实相机首帧位置: {np.round(real_cams[0]['pos'], 3)}")
        return 4
    print(f"\n渲染自检: 彩色像素占比中位 {np.median(cover)*100:.1f}%, "
          f"空画面帧 {int((cover<0.005).sum())}/{len(frames)}")

    # ---- 输出 GIF ----
    # 用 imageio 写真 GIF (此前用 mp4v 编解码器写 .gif, 容器与编码不匹配)
    import imageio.v2 as imageio
    gif_path = os.path.join(out_dir, "stage61_physics.gif")
    # GIF 调色板有限, 帧数多时按 duration 抽稀以控制体积
    stride = max(1, len(frames) // 120)
    gif_frames = frames[::stride]
    imageio.mimsave(gif_path, gif_frames, duration=1000.0 / args.fps, loop=0)
    print(f"\nGIF:  {gif_path}  ({len(gif_frames)} 帧 @ {args.fps}fps)")

    # 同时存一份 mp4 (无损全帧) 和首帧 png
    mp4_path = os.path.join(out_dir, "stage61_physics.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    for f in frames:
        # frames 是 RGB (SAPIEN 输出), VideoWriter 期望 BGR —— 不转换会红蓝互换
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    png_path = os.path.join(out_dir, "stage61_physics_frame0.png")
    cv2.imwrite(png_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    print(f"MP4:  {mp4_path}  ({len(frames)} 帧)")
    print(f"首帧: {png_path}")
    report["outputs"] = {"gif": gif_path, "mp4": mp4_path, "frame0": png_path,
                         "n_frames": n_frames, "fps": args.fps}
    rp = os.path.join(out_dir, "stage61_physics_report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\nGIF:  {gif_path}  ({n_frames} 帧 @ {args.fps}fps)")
    print(f"首帧: {png_path}")
    print(f"报告: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Stage 6.1: URDF 生成 + SAPIEN 物理仿真

从 Stage 5 的 final_scene_stage5.glb 拆解每个物体，
调用 EmbodiedGen 的 URDFGenerator（替换 VLM 为 Qwen），
为每个物体生成 URDF，合并为 scene.urdf，可选 SAPIEN 验证。

用法:
    python src/stage61_urdf_generator.py \\
        --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \\
        --output_dir output_v2/2chair-bottle_vggt_omega \\
        --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \\
        --validate

    # 跳过 SAPIEN 验证:
    python src/stage61_urdf_generator.py \\
        --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \\
        --output_dir output_v2/2chair-bottle_vggt_omega \\
        --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \\
        --no_validate
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

# ─── Step 0: 路径配置 ───────────────────────────────────────────────────────

# 添加 ReplicateAnyScene 根目录到 sys.path (用于 import tools/qwen_client)
_REPLICATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPLICATE_ROOT not in sys.path:
    sys.path.insert(0, _REPLICATE_ROOT)

# 添加 EmbodiedGen 根目录到 sys.path (用于 import embodied_gen.* )
_EMBODIED_GEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EmbodiedGen"))
if _EMBODIED_GEN_ROOT not in sys.path:
    sys.path.insert(0, _EMBODIED_GEN_ROOT)

# ─── 关键: Mock 外部依赖防止 import 崩溃 ────────────────────────────────────
# embodied_gen.validators.urdf_convertor 顶部的 import 链:
#   from embodied_gen.data.convex_decomposer import decompose_convex_mesh    ✅ 有 coacd
#   from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient         ❌ 缺 openai
#   from embodied_gen.utils.process_media import render_asset3d              ❌ 缺 nvdiffrast
#
# 我们在 import URDFGenerator 之前注入 MockModule。
class _MockProcessMedia:
    def __getattr__(self, name):
        return lambda *a, **k: []

class _MockGPTClient:
    """GPTclient 存根 — 不被使用（我们传自定义 QwenClient），但模块级 import 需要。"""
    def __init__(self, *a, **k):
        pass
    def query(self, *a, **k):
        return ""

class _MockGPTClientsModule:
    GPTclient = _MockGPTClient
    GPT_CLIENT = _MockGPTClient()  # 模块级单例存根
    def __getattr__(self, name):
        return lambda *a, **k: []

sys.modules.setdefault("embodied_gen.utils.gpt_clients", _MockGPTClientsModule())
sys.modules.setdefault("embodied_gen.utils.process_media", _MockProcessMedia())

# ─── 导入 EmbodiedGen 模块 ──────────────────────────────────────────────────
# load_actor_from_urdf 只在 SAPIEN 验证时用，延迟导入避免 mplib 依赖问题
from embodied_gen.validators.urdf_convertor import URDFGenerator

# ─── 导入本地模块 ────────────────────────────────────────────────────────────
from tools.qwen_client import QwenClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Open3D 渲染函数 ─────────────────────────────────────────────────────────

def open3d_render_mesh(mesh_path: str, output_dir: str, obj_name: str) -> list[str]:
    """用 Open3D 离屏渲染 4 个视角（前/右/后/左），返回渲染图片路径列表。"""
    import open3d as o3d
    from PIL import Image

    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()

    # 使用未光照 shader，黑底
    render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
    render.scene.set_background([0, 0, 0, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    render.scene.remove_geometry("mesh")
    render.scene.add_geometry("mesh", o3d_mesh, mat)

    # 计算 bbox 中心和对角线距离
    aabb = o3d_mesh.get_axis_aligned_bounding_box()
    max_bound = np.array(aabb.get_max_bound(), dtype=np.float64)
    min_bound = np.array(aabb.get_min_bound(), dtype=np.float64)
    center = (max_bound + min_bound) / 2
    diag = max_bound - min_bound
    dist = float(np.max(diag)) * 1.5

    # 4 个视角: 前/右/后/左 (绕 Y 轴旋转)
    camera_angles_deg = [0, 90, 180, -90]

    paths = []
    render_dir = os.path.join(output_dir, "urdf_renders")
    os.makedirs(render_dir, exist_ok=True)

    for i, angle_deg in enumerate(camera_angles_deg):
        angle_rad = np.radians(angle_deg)
        eye = (center + np.array([dist * np.cos(angle_rad), 0.0, dist * np.sin(angle_rad)])).astype(np.float32).reshape(3, 1)
        center_arr = center.astype(np.float32).reshape(3, 1)
        up_arr = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)
        render.setup_camera(
            vertical_field_of_view=50,
            center=center_arr,
            eye=eye,
            up=up_arr,
        )
        img = render.render_to_image()
        img_pil = Image.fromarray(np.array(img))
        out_path = os.path.join(render_dir, f"{obj_name}_{i}.png")
        img_pil.save(out_path)
        paths.append(out_path)
        logger.info(f"  Rendered view {i}: {out_path}")

    return paths


# ─── Step 1: GLB 拆解 ───────────────────────────────────────────────────────

def extract_objects_from_glb(glb_path: str, temp_dir: str) -> list[dict]:
    """从 GLB 拆解每个物体，返回 objects_info 列表。

    使用 scene.graph.geometry_nodes 获取几何名到节点名的映射，
    使用 scene.graph.transforms 获取世界变换矩阵。

    每个物体是一个 dict:
        {
            "name": "chair_0",
            "category": "chair",
            "obj_path": "/path/to/chair_0.obj",
            "position": [x, y, z],      # 世界坐标
            "matrix": np.array(4x4),    # 世界变换矩阵 (用于 URDF joint)
            "ground_z": float,           # 物体底部 z (相对原点)
        }
    """
    scene = trimesh.load(glb_path)
    objects_info = []

    # 从 graph 获取 geometry_name → [node_names] 映射
    # 例如: "geometry_1" -> ["bottle_0"]
    geom_to_nodes = scene.graph.geometry_nodes

    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            logger.info(f"  Skipping non-mesh: {geom_name}")
            continue

        # 获取节点名
        node_names = geom_to_nodes.get(geom_name, [geom_name])
        if not node_names:
            logger.warning(f"  {geom_name} has no nodes, skipping")
            continue

        node_name = node_names[0]  # 通常只有一个节点

        # 获取世界变换矩阵 (scene.graph.get 返回 (matrix, geom_name) tuple)
        graph_result = scene.graph.get(node_name)
        if graph_result is None:
            world_matrix = np.eye(4)
        else:
            if isinstance(graph_result, tuple):
                world_matrix = graph_result[0]
            else:
                world_matrix = graph_result
        world_matrix = np.asarray(world_matrix, dtype=np.float64)

        # 提取世界位置
        # 关键：GLB 中物体可能两种形式存在：
        #   1. 节点 transform 包含位置（如 2chair-bottle）→ world_matrix[:3,3] 正确
        #   2. 位置已烘焙到 mesh 顶点中（如 7_vggt_omega）→ world_matrix[:3,3]=[0,0,0]
        # 统一方案：用 world_transformed vertices 的质心作为世界位置
        # 手动做齐次变换
        verts_h = np.hstack([geom.vertices, np.ones((len(geom.vertices), 1))])
        world_vertices = (world_matrix @ verts_h.T).T[:, :3]
        world_centroid = world_vertices.mean(axis=0).tolist()

        # 同时检查 matrix 的 translation，若两者差异大，以顶点质心为准
        matrix_translation = world_matrix[:3, 3].tolist()
        if np.linalg.norm(np.array(world_centroid) - np.array(matrix_translation)) > 1e-6:
            position = world_centroid  # 顶点编码位置（7_vggt_omega 模式）
        else:
            position = matrix_translation  # 矩阵编码位置（2chair-bottle 模式）

        # 类别: "chair_0" -> "chair"
        if "_" in node_name:
            category = node_name.rsplit("_", 1)[0]
        else:
            category = node_name

        # 居中 mesh（供 URDFGenerator 使用）
        centered_mesh = geom.copy()
        centered_mesh.vertices -= centered_mesh.vertices.mean(axis=0)

        # 保存 OBJ
        obj_path = os.path.join(temp_dir, f"{node_name}.obj")
        os.makedirs(temp_dir, exist_ok=True)
        centered_mesh.export(obj_path)

        # 物体底部 z（局部坐标）
        bbox_min = geom.vertices.min(axis=0)
        bbox_max = geom.vertices.max(axis=0)
        local_ground_z = bbox_min[2]
        # 世界坐标底部 z：用 transform 后的顶点
        world_ground_z = float(world_vertices[:, 2].min())

        objects_info.append({
            "name": node_name,
            "category": category,
            "obj_path": obj_path,
            "position": position,
            "matrix": world_matrix,
            "ground_z": float(local_ground_z),
            "world_ground_z": world_ground_z,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        })

    logger.info(f"Extracted {len(objects_info)} objects from GLB")
    for obj in objects_info:
        logger.info(f"  {obj['name']} (category={obj['category']}, world_pos={obj['position']})")

    return objects_info


def _rewrite_urdf_inertia(urdf_path: str, mesh_folder: str) -> None:
    """mesh 旋转回 Z-up 后, 用旋转后的 collision mesh 重算惯量并写回 URDF。

    为什么必须做: EmbodiedGen 的 generate_urdf() 收到的输入 mesh 已被本文件上方
    旋转成 Y-up(Step 2d 的 Z→Y), 它是在那个**中间朝向**下计算 <inertial> 的; 而
    mesh 文件随后又被旋转回 Z-up, 于是 iyy/izz 互换。
    实测 chair_1(2chair-bottle_vggt_omega): URDF 写 izz=12.6284, 但 Z-up 真值是
    6.3896, 偏差 97.6%; chair_0 偏差 27.2%。这会让物理翻滚行为失真。

    口径与 tools/stage61_physics_sim.py::recompute_inertia 完全一致(仿真侧一旦发现
    URDF 惯性非占位值就直接采信, 不再重算): 水密网格用 trimesh.moment_inertia,
    非水密/体积异常退回均匀密度长方体近似。
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link = root.find("link")
    if link is None:
        return
    m_el = link.find("inertial/mass")
    i_el = link.find("inertial/inertia")
    if m_el is None or i_el is None:
        return
    mass = float(m_el.get("value"))
    cm = link.find(".//collision/geometry/mesh")
    if cm is None or not cm.get("filename"):
        return
    mp = os.path.normpath(os.path.join(os.path.dirname(urdf_path), cm.get("filename")))
    if not os.path.exists(mp):
        return
    m = trimesh.load(mp, force="mesh", process=False)
    if m.is_watertight and m.volume > 0:
        I = np.asarray(m.moment_inertia, dtype=np.float64) * (mass / float(m.volume))
        vals = dict(ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2],
                    ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2])
    else:
        ext = m.extents
        a, b, c = [float(max(v, 1e-4)) for v in ext]
        vals = dict(ixx=mass * (b * b + c * c) / 12.0,
                    iyy=mass * (a * a + c * c) / 12.0,
                    izz=mass * (a * a + b * b) / 12.0,
                    ixy=0.0, ixz=0.0, iyz=0.0)
    for k, v in vals.items():
        i_el.set(k, f"{v:.6f}")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"    惯量按 Z-up 重算写回: ixx={vals['ixx']:.6f} "
                f"iyy={vals['iyy']:.6f} izz={vals['izz']:.6f}")


# ─── Step 2: 逐个生成 URDF ──────────────────────────────────────────────────

def generate_urdf_for_object(
    obj_info: dict,
    output_dir: str,
    urdf_generator: URDFGenerator,
    qwen_client: QwenClient,
) -> str:
    """为单个物体生成 URDF。

    Step 2a: 渲染 4 视图
    Step 2b: Qwen VLM 推理
    Step 2c: 解析输出
    Step 2d: 生成 URDF

    Returns:
        生成的 URDF 文件路径
    """
    name = obj_info["name"]
    category = obj_info["category"]
    mesh_path = obj_info["obj_path"]
    obj_output_dir = os.path.join(output_dir, name)

    logger.info(f"\n=== Generating URDF for {name} (category={category}) ===")

    # Step 2a: 渲染 4 视图
    logger.info(f"  Step 2a: Rendering 4 views...")
    render_paths = open3d_render_mesh(mesh_path, obj_output_dir, name)

    # Step 2b: VLM 推理 — prompt 从 EmbodiedGen URDFGenerator.prompt_template 继承
    logger.info(f"  Step 2b: Qwen VLM inference...")
    text_prompt = urdf_generator.prompt_template.format(category=category.lower())
    response = qwen_client.query(text_prompt, render_paths)

    # Step 2c: 解析输出
    logger.info(f"  Step 2c: Parsing VLM response...")
    if response:
        try:
            attr_dict = urdf_generator.parse_response(response)
            logger.info(f"    Parsed: {attr_dict}")
        except Exception as e:
            logger.warning(f"    parse_response failed: {e}, using defaults")
            attr_dict = {
                "category": category.lower(),
                "description": category.lower(),
                "min_height": 0.1,
                "max_height": 0.2,
                "min_mass": 0.1,
                "max_mass": 0.3,
                "mu1": 0.6,
                "mu2": 0.5,
            }
    else:
        logger.warning("    VLM returned empty response, using defaults")
        attr_dict = {
            "category": category.lower(),
            "description": category.lower(),
            "min_height": 0.1,
            "max_height": 0.2,
            "min_mass": 0.1,
            "max_mass": 0.3,
            "mu1": 0.6,
            "mu2": 0.5,
        }

    # 关键: real_height 和 mass 从 GLB mesh 自身的真实尺寸/体积读取
    # GLB 已有真实物理尺度（由 extrinsics/intrinsic 确定），Qwen 只提供类别/描述/摩擦系数
    mesh_for_props = trimesh.load(mesh_path)
    z_extent = np.ptp(mesh_for_props.vertices, axis=0)[2]  # GLB 是 Z-up
    real_height = round(z_extent, 6)
    attr_dict["real_height"] = real_height

    # Mass 从 mesh 体积 × 密度计算
    # 关于非水密网格(实测 218 个物体里 42 个): 曾想改用包围盒体积, 但实测
    # 包围盒/trimesh 体积比中位 21.9x、最大 471x (如 newspaper_0 0.000225 -> 0.1059 m³,
    # 会把报纸算成 84.7kg), 明显更差。trimesh.volume 对这类"近似闭合"的网格反而给出
    # 合理值, 故保留它, 仅打 WARN 标记不确定性, 不再静默使用。
    volume = float(abs(mesh_for_props.volume))
    volume_reliable = bool(mesh_for_props.is_watertight)
    if not volume_reliable:
        logger.warning(
            f"    [质量] 网格非水密, trimesh.volume 有不确定性; "
            f"仍采用 {volume:.6f} m³ (包围盒近似会高估 ~22x, 不采用)"
        )
    # 按类别估算密度 (kg/m³)
    density_map = {"chair": 600, "table": 600, "bottle": 1000, "cup": 2500, "default": 800}
    density = density_map.get(category.lower(), density_map["default"])
    real_mass = round(volume * density, 4)
    logger.info(f"    mass source: volume={volume:.6f} m³ "
                f"(reliable={volume_reliable}) × density={density} "
                f"= {real_mass:.4f} kg")
    # 为 EmbodiedGen 的 attr_dict 提供兼容字段
    attr_dict["min_mass"] = real_mass
    attr_dict["max_mass"] = real_mass
    logger.info(f"    Mesh real height from GLB: {real_height:.4f} m, volume: {volume:.6f} m³, mass: {real_mass:.4f} kg")

    # Step 2d: 生成 URDF — 修复 Z-up → Y-up 坐标系问题
    logger.info(f"  Step 2d: Generating URDF...")

    # EmbodiedGen generate_urdf() 内部逻辑:
    #   raw_height = np.ptp(vertices)[1]  (用 Y 轴当高度)
    #   scale = real_height / raw_height
    #   然后保存 scaled mesh
    #
    # 但 GLB mesh 是 Z-up (高度沿 Z 轴)，导致 Y 轴不是真实高度。
    # 修复: 在传入前旋转 mesh (Z→Y)，让 EmbodiedGen 正确缩放；
    #       生成后再旋转回来 (Y→Z)，恢复 Z-up 坐标系。
    #
    # 同时 rotate_xyzw=None: 跳过 Blender→SAPIEN 的 90° X 旋转，
    # 因为 GLB 已经是 Z-up，不需要额外旋转。
    from scipy.spatial.transform import Rotation as R

    # 1. 旋转输入 mesh: Z→Y (绕 X 轴 -90°) — 放到独立临时目录
    temp_aligned_dir = os.path.join(obj_output_dir, "temp_aligned")
    os.makedirs(temp_aligned_dir, exist_ok=True)
    aligned_mesh_path = os.path.join(temp_aligned_dir, os.path.basename(mesh_path))

    mesh = trimesh.load(mesh_path)
    rot_z_to_y = R.from_euler("x", -90, degrees=True)
    T_z_to_y = np.eye(4)
    T_z_to_y[:3, :3] = rot_z_to_y.as_matrix()
    mesh.apply_transform(T_z_to_y)
    mesh.export(aligned_mesh_path)
    logger.info(f"    Rotated input mesh Z→Y: {aligned_mesh_path}")

    # 2. 调用 EmbodiedGen 生成 URDF (使用 rotate_xyzw=None 的 generator)
    old_rotate = urdf_generator.rotate_xyzw
    urdf_generator.rotate_xyzw = None
    try:
        urdf_path = urdf_generator.generate_urdf(
            input_mesh=aligned_mesh_path,
            output_dir=obj_output_dir,
            attr_dict=attr_dict,
            output_name=name,
        )
    finally:
        urdf_generator.rotate_xyzw = old_rotate

    # 3. 旋转输出 mesh 回 Z-up (绕 X 轴 +90°)
    rot_y_to_z = R.from_euler("x", 90, degrees=True)
    T_y_to_z = np.eye(4)
    T_y_to_z[:3, :3] = rot_y_to_z.as_matrix()

    mesh_folder = os.path.join(obj_output_dir, "mesh")
    for fname in os.listdir(mesh_folder):
        if fname.endswith(".obj"):
            fpath = os.path.join(mesh_folder, fname)
            m = trimesh.load(fpath)
            m.apply_transform(T_y_to_z)
            m.export(fpath)
            logger.info(f"    Rotated output mesh Y→Z: {fpath}")

    # 3.5 惯量修正: generate_urdf() 是在上面的 Y-up 中间朝向下算 <inertial> 的,
    #     mesh 此时已旋转回 Z-up, 必须用旋转后的 mesh 重算惯量并写回 URDF,
    #     否则 iyy/izz 互换 (实测 chair_1 偏差 97.6%, 会让翻滚行为失真)。
    _rewrite_urdf_inertia(urdf_path, mesh_folder)

    logger.info(f"    URDF saved to {urdf_path}")
    return urdf_path


# ─── Step 3: 合并 scene.urdf ────────────────────────────────────────────────

def merge_to_scene_urdf(
    objects_info: list[dict],
    output_dir: str,
) -> str:
    """合并所有物体 URDF 为一个 scene.urdf。

    每个物体通过 <joint type="fixed"> 连接到 base_link，
    joint 的 <origin> 保留 GLB 中的世界位姿。

    Returns:
        scene.urdf 路径
    """
    logger.info("\n=== Merging to scene.urdf ===")

    robot = ET.Element("robot", name="scene")
    base_link = ET.SubElement(robot, "link", name="base")
    ET.SubElement(base_link, "visual")
    ET.SubElement(base_link, "collision")

    # mesh 目录
    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    # 跟踪已复制的纹理文件
    copied_materials = {}  # {original_path: dest_path}

    for obj_info in objects_info:
        name = obj_info["name"]
        obj_urdf_dir = os.path.join(output_dir, name)
        obj_urdf_path = os.path.join(obj_urdf_dir, f"{name}.urdf")

        if not os.path.exists(obj_urdf_path):
            logger.warning(f"  URDF not found: {obj_urdf_path}, skipping")
            continue

        tree = ET.parse(obj_urdf_path)
        link = tree.getroot().find("link")
        if link is None:
            logger.warning(f"  No link in {obj_urdf_path}, skipping")
            continue

        # 复制 mesh 文件到合并目录
        obj_mesh_dir = os.path.join(obj_urdf_dir, "mesh")
        if os.path.isdir(obj_mesh_dir):
            for fname in os.listdir(obj_mesh_dir):
                src = os.path.join(obj_mesh_dir, fname)
                if os.path.isfile(src):
                    dst_name = f"{name}_{fname}"
                    dst = os.path.join(mesh_dir, dst_name)
                    shutil.copy2(src, dst)
                    logger.info(f"    Copied mesh: {fname} -> {dst_name}")

                    # 更新 link 中的 mesh 文件名
                    for mesh_tag in link.iter("mesh"):
                        old_filename = mesh_tag.get("filename")
                        if old_filename and fname in old_filename:
                            new_filename = f"mesh/{dst_name}"
                            mesh_tag.set("filename", new_filename)

        # 复制纹理文件（如果有）
        if os.path.isdir(obj_mesh_dir):
            for fname in os.listdir(obj_mesh_dir):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    src = os.path.join(obj_mesh_dir, fname)
                    if fname not in copied_materials:
                        dst_name = f"{name}_{fname}"
                        copied_materials[fname] = dst_name
                        shutil.copy2(src, os.path.join(mesh_dir, dst_name))
                        logger.info(f"    Copied texture: {fname} -> {dst_name}")

        # 更新 link name
        link.set("name", name)
        robot.append(link)

        # 创建 fixed joint
        joint = ET.SubElement(robot, "joint",
            name=f"joint_{name}", type="fixed")
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=name)
        pos = obj_info["position"]
        ET.SubElement(joint, "origin",
            xyz=f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}",
            rpy="0.0000 0.0000 0.0000")

    urdf_str = parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    scene_urdf_path = os.path.join(output_dir, "scene.urdf")
    with open(scene_urdf_path, "w") as f:
        f.write(urdf_str)

    logger.info(f"Scene URDF saved to {scene_urdf_path}")
    return scene_urdf_path


# ─── Step 4: SAPIEN 验证 ────────────────────────────────────────────────────

def _load_actor_from_urdf_simple(scene, urdf_path, pose):
    """Minimal SAPIEN URDF loader — 手动解析 URDF XML，不依赖 EmbodiedGen 的 mplib/mani_skill。
    
    复用了 EmbodiedGen embodied_gen.utils.simulation.load_actor_from_urdf 的核心逻辑。
    """
    import numpy as np
    import sapien.core as sapien
    import sapien.physx
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as R

    file_dir = os.path.dirname(urdf_path)

    def _get_local_pose(origin_tag):
        local_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if origin_tag is not None:
            xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
            qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
            local_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])
        return local_pose

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    node_name = root.get("name")

    visual_mesh = root.find(".//visual/geometry/mesh")
    visual_file = visual_mesh.get("filename")
    visual_scale = np.array([float(x) for x in visual_mesh.get("scale", "1.0 1.0 1.0").split()])
    visual_file = os.path.join(file_dir, visual_file)
    visual_pose = _get_local_pose(root.find(".//visual/origin"))

    collision_mesh = root.find(".//collision/geometry/mesh")
    collision_file = collision_mesh.get("filename")
    collision_scale = np.array([float(x) for x in collision_mesh.get("scale", "1.0 1.0 1.0").split()])
    collision_file = os.path.join(file_dir, collision_file)
    collision_pose = _get_local_pose(root.find(".//collision/origin"))

    static_fric = root.find(".//collision/gazebo/mu1").text
    dynamic_fric = root.find(".//collision/gazebo/mu2").text

    material = sapien.physx.PhysxMaterial(
        static_friction=float(static_fric),
        dynamic_friction=float(dynamic_fric),
        restitution=0.05,
    )

    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    builder.add_multiple_convex_collisions_from_file(
        collision_file,
        material=material,
        scale=collision_scale,
        pose=collision_pose,
    )
    builder.add_visual_from_file(
        visual_file,
        scale=visual_scale,
        pose=visual_pose,
    )
    if pose is None:
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
    builder.set_initial_pose(pose)

    actor = builder.build(name=node_name)
    return actor

def validate_with_sapien(
    urdf_dir: str,
    objects_info: list[dict],
    output_dir: str,
) -> dict:
    """用 SAPIEN 无头模式验证所有 URDF。"""
    import sapien.core as sapien
    import sapien.physx

    logger.info("\n=== SAPIEN Validation ===")
    scene = sapien.Scene()
    scene.set_timestep(1 / 200)
    scene.add_ground(0)

    errors = []
    actors = []

    for obj_info in objects_info:
        name = obj_info["name"]
        obj_urdf_path = os.path.join(urdf_dir, name, f"{name}.urdf")

        if not os.path.exists(obj_urdf_path):
            errors.append({"object": name, "error": f"URDF not found: {obj_urdf_path}"})
            continue

        try:
            pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=[1, 0, 0, 0])
            actor = _load_actor_from_urdf_simple(scene, obj_urdf_path, pose)
            actors.append((actor, name))
            logger.info(f"  Loaded: {name}")
        except Exception as e:
            errors.append({"object": name, "error": str(e)})
            logger.error(f"  FAILED: {name} - {e}")

    # 运行仿真
    logger.info(f"  Running {len(actors)} actors, {200} steps...")
    for _ in range(200):
        scene.step()

    # 检查结果
    results = []
    for actor, name in actors:
        z = float(actor.get_pose().p[2])
        results.append({
            "actor": name,
            "final_z": z,
            "stable": z > -0.01,
        })

    report = {
        "errors": errors,
        "results": results,
        "success": len(errors) == 0 and all(r["stable"] for r in results),
    }

    report_path = os.path.join(output_dir, "stage61_validation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  Validation report saved to {report_path}")
    logger.info(f"  Success: {report['success']}")
    if errors:
        logger.warning(f"  {len(errors)} errors: {errors}")

    return report


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def run_stage61(
    glb_path: str,
    output_dir: str,
    vlm_checkpoint: str,
    validate: bool = False,
):
    """Stage 6.1 完整流程。

    Args:
        glb_path: final_scene_stage5.glb 路径
        output_dir: 输出目录
        vlm_checkpoint: Qwen VLM 模型路径
        validate: 是否执行 SAPIEN 验证
    """
    logger.info("=" * 60)
    logger.info("Stage 6.1: URDF Generation + SAPIEN Validation")
    logger.info("=" * 60)

    start_time = time.time()
    report = {"objects": [], "errors": [], "timestamps": {}}

    # 输出目录 — 所有 stage61 产物放在 <output_dir>/urdf/ 子目录下
    base_dir = os.path.abspath(output_dir)
    urdf_dir = os.path.join(base_dir, "urdf")
    temp_dir = os.path.join(urdf_dir, "temp_stage61")

    # ── Step 0: 初始化 ──
    logger.info("\n=== Step 0: Initializing ===")
    qwen_client = QwenClient(vlm_checkpoint=vlm_checkpoint)
    urdf_generator = URDFGenerator(
        gpt_client=qwen_client,
        render_view_num=4,
        decompose_convex=True,
        rotate_xyzw=None,  # 不调用 __call__()，所以这个参数不用
    )

    # ── Step 1: GLB 拆解 ──
    logger.info("\n=== Step 1: Extracting objects from GLB ===")
    objects_info = extract_objects_from_glb(glb_path, temp_dir)
    report["timestamps"]["step1_done"] = time.time()

    # ── Step 2: 逐个生成 URDF ──
    logger.info("\n=== Step 2: Generating URDF for each object ===")
    generated_urdfs = []

    for i, obj_info in enumerate(objects_info):
        try:
            urdf_path = generate_urdf_for_object(obj_info, urdf_dir, urdf_generator, qwen_client)
            generated_urdfs.append(urdf_path)
            report["objects"].append({
                "name": obj_info["name"],
                "category": obj_info["category"],
                "urdf": urdf_path,
                "status": "success",
            })
        except Exception as e:
            logger.error(f"  FAILED for {obj_info['name']}: {e}")
            report["objects"].append({
                "name": obj_info["name"],
                "category": obj_info["category"],
                "urdf": None,
                "status": "failed",
                "error": str(e),
            })
            report["errors"].append({"object": obj_info["name"], "error": str(e)})

    report["timestamps"]["step2_done"] = time.time()

    # ── Step 3: 合并 scene.urdf ──
    logger.info("\n=== Step 3: Merging scene.urdf ===")
    try:
        scene_urdf_path = merge_to_scene_urdf(objects_info, urdf_dir)
        report["scene_urdf"] = scene_urdf_path
        report["scene_urdf_status"] = "success"
    except Exception as e:
        logger.error(f"  Failed to merge scene URDF: {e}")
        report["scene_urdf"] = None
        report["scene_urdf_status"] = "failed"
        report["errors"].append({"step": "merge", "error": str(e)})

    # ── Step 4: SAPIEN 验证 (可选) ──
    report["validation"] = {"enabled": validate}
    if validate:
        try:
            validation_report = validate_with_sapien(urdf_dir, objects_info, base_dir)
            report["validation"]["success"] = validation_report["success"]
            report["validation"]["details"] = validation_report
        except Exception as e:
            logger.error(f"  SAPIEN validation failed: {e}")
            report["validation"]["success"] = False
            report["validation"]["error"] = str(e)
            report["errors"].append({"step": "validation", "error": str(e)})
    else:
        report["validation"]["note"] = "Skipped (--no_validate)"

    # ── 保存报告 ──
    report["timestamps"]["completed"] = time.time()
    report["timestamps"]["duration_seconds"] = report["timestamps"]["completed"] - start_time

    report_path = os.path.join(urdf_dir, "stage61_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nStage 6.1 completed in {report['timestamps']['duration_seconds']:.1f}s")
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Total objects: {len(objects_info)}")
    logger.info(f"Generated URDFs: {len(generated_urdfs)}")
    logger.info(f"Errors: {len(report['errors'])}")

    return report


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6.1: URDF Generation + SAPIEN Validation")
    parser.add_argument("--glb", required=True, help="Path to final_scene_stage5.glb")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--vlm_checkpoint", default="/mnt/data/lza/models/Qwen3.5-9B",
                        help="Qwen VLM model path")
    parser.add_argument("--validate", action="store_true",
                        help="Run SAPIEN validation after URDF generation")
    parser.add_argument("--no_validate", action="store_false", dest="validate",
                        help="Skip SAPIEN validation (default)")
    args = parser.parse_args()

    run_stage61(
        glb_path=args.glb,
        output_dir=args.output_dir,
        vlm_checkpoint=args.vlm_checkpoint,
        validate=args.validate,
    )

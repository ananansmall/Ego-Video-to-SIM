"""提取 final_scene.glb 的全部物体 (除地面 grid): 减面 + 顶点色烘贴图, 导出
OBJ+MTL+PNG 供 RoboTwin/SAPIEN 使用, 并记录每个物体的 RAS 世界中心/尺寸/名称。

SAPIEN 不读 GLB 顶点色, 只认材质贴图, 故与粉鸭同法: quadric 减面 + UV atlas 烘焙。
"""
import os
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image

RAS = "/path/to/workspace/traj_data_min/ras/7_vggt_omega"
OUTDIR = "/tmp/scene_assets"
TEX_SIZE = 1024
TARGET_TRIS = 8000
SKIP = {"geometry_0"}          # 地面 grid

os.makedirs(OUTDIR, exist_ok=True)


def bake_export(name, geom, center):
    V = np.asarray(geom.vertices, dtype=float) - center
    F = np.asarray(geom.faces)
    C = np.asarray(geom.visual.vertex_colors)[:, :3].astype(float) / 255.0
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(F)
    m.vertex_colors = o3d.utility.Vector3dVector(C)
    m.compute_vertex_normals()
    if len(m.triangles) > TARGET_TRIS:
        m = m.simplify_quadric_decimation(target_number_of_triangles=TARGET_TRIS)
    m.remove_degenerate_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    C2 = np.asarray(m.vertex_colors)
    obj = os.path.join(OUTDIR, f"{name}.obj")
    tm = o3d.t.geometry.TriangleMesh.from_legacy(m)
    tm.vertex["albedo"] = o3d.core.Tensor(C2.astype(np.float32))
    try:
        tm.compute_uvatlas(size=TEX_SIZE)
        tm.material.set_default_properties()
        tex = tm.bake_vertex_attr_textures(TEX_SIZE, {"albedo"}, margin=4, update_material=True)
        a = tex["albedo"].numpy()
        o3d.t.io.write_triangle_mesh(obj, tm, write_vertex_colors=False, write_triangle_uvs=True)
        Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(OUTDIR, f"{name}_tex.png"))
        with open(os.path.join(OUTDIR, f"{name}.mtl"), "w") as fh:
            fh.write(f"newmtl {name}_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 10\n"
                     f"d 1\nillum 2\nmap_Kd {name}_tex.png\n")
        ntri = len(m.triangles); nv = len(m.vertices)
    except Exception as e:
        print(f"  [warn] {name} UV/烘焙失败({e}); 统一色")
        avg = C2.mean(0)
        m.paint_uniform_color(avg.tolist())
        o3d.io.write_triangle_mesh(obj, m, write_vertex_colors=True)
        ntri = len(m.triangles); nv = len(m.vertices)
    ext = np.asarray(m.vertices).max(0) - np.asarray(m.vertices).min(0)
    return center, ext, ntri, nv


def main():
    sc = trimesh.load(os.path.join(RAS, "final_scene.glb"), force="scene")
    meta = {"name": [], "center": [], "extent": []}
    print(f"场景几何数={len(sc.geometry)}")
    for node in sorted(sc.graph.nodes_geometry):
        T, gname = sc.graph[node]
        if gname in SKIP:
            continue
        g = sc.geometry[gname]
        V = np.asarray(g.vertices)
        center = (V.min(0) + V.max(0)) / 2          # RAS 原始顶点中心 (与粉鸭同系)
        center, ext, ntri, nv = bake_export(gname, g, center)
        meta["name"].append(gname); meta["center"].append(center); meta["extent"].append(ext)
        print(f"  {gname}: V {len(V)}->{nv}  tri->{ntri}  center={np.round(center,4).tolist()} "
              f"extent={np.round(ext,4).tolist()}")
    np.savez(os.path.join(OUTDIR, "scene_meta.npz"),
             name=np.array(meta["name"]),
             center=np.array(meta["center"]),
             extent=np.array(meta["extent"]))
    print(f"meta -> {OUTDIR}/scene_meta.npz  ({len(meta['name'])} 物体)")
    for f in sorted(os.listdir(OUTDIR)):
        print(f"  {f}  {os.path.getsize(os.path.join(OUTDIR,f))/1e3:.1f} KB")


if __name__ == "__main__":
    main()

"""压缩粉色小鸭: 322786 顶点 -> ~12k 三角面 (quadric 减面), 并把 GLB 顶点色烘成
贴图 (SAPIEN 不读 GLB 顶点色, 只认材质贴图), 导出 OBJ+MTL+PNG 供 RoboTwin/SAPIEN 使用。
"""
import os
import numpy as np
import trimesh
import open3d as o3d

RAS = "/path/to/workspace/traj_data_min/ras/7_vggt_omega"
OUTDIR = "/tmp/duck_assets"
os.makedirs(OUTDIR, exist_ok=True)
OBJ = os.path.join(OUTDIR, "pink_duck.obj")
TEX_SIZE = 1024
TARGET_TRIS = 12000


def main():
    sc = trimesh.load(os.path.join(RAS, "final_scene.glb"), force="scene")
    _, g = sc.graph.get("toy_0")
    geom = sc.geometry[g]
    V = np.asarray(geom.vertices, dtype=float)
    F = np.asarray(geom.faces)
    C = np.asarray(geom.visual.vertex_colors)[:, :3].astype(float) / 255.0
    center = (V.min(0) + V.max(0)) / 2
    V = V - center
    print(f"原始: V={len(V)} F={len(F)}  尺寸={np.round(V.max(0)-V.min(0),4).tolist()}")

    # ---- open3d legacy 网格 ----
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(F)
    m.vertex_colors = o3d.utility.Vector3dVector(C)
    m.compute_vertex_normals()
    print(f"减面前 三角={len(m.triangles)}")

    # ---- quadric 减面 (压缩) ----
    if len(m.triangles) > TARGET_TRIS:
        m = m.simplify_quadric_decimation(target_number_of_triangles=TARGET_TRIS)
    m.remove_degenerate_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    V2 = np.asarray(m.vertices); F2 = np.asarray(m.triangles); C2 = np.asarray(m.vertex_colors)
    print(f"减面后 三角={len(F2)} 顶点={len(V2)}  (压缩 {len(F)/len(F2):.1f}x)")

    # ---- tensor 网格: 顶点色 -> albedo, UV atlas, 烘焙 ----
    tm = o3d.t.geometry.TriangleMesh.from_legacy(m)
    tm.vertex["albedo"] = o3d.core.Tensor(C2.astype(np.float32))
    try:
        stretch, ncharts, npart = tm.compute_uvatlas(size=TEX_SIZE)
        print(f"UV atlas: charts={ncharts} stretch={stretch:.4f}")
        tm.material.set_default_properties()
        tex = tm.bake_vertex_attr_textures(TEX_SIZE, {"albedo"}, margin=4, update_material=True)
        a = tex["albedo"].numpy()
        print(f"烘焙贴图 albedo: {a.shape} range=[{a.min():.3f},{a.max():.3f}]")
        o3d.t.io.write_triangle_mesh(OBJ, tm, write_vertex_colors=False, write_triangle_uvs=True)
        # open3d 不写贴图文件, 手动存 PNG 并修 MTL 引用
        from PIL import Image
        Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUTDIR, "pink_duck_tex.png"))
        mtl = os.path.join(OUTDIR, "pink_duck.mtl")
        with open(mtl, "w") as fh:
            fh.write("newmtl pink_duck_0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 10\n"
                     "d 1\nillum 2\nmap_Kd pink_duck_tex.png\n")
    except Exception as e:
        print(f"[warn] UV/烘焙失败({e}); 退化为统一粉色材质")
        m.paint_uniform_color([0.667, 0.075, 0.612])   # 平均粉 (170,19,156)
        o3d.io.write_triangle_mesh(OBJ, m, write_vertex_colors=True)

    # 找导出的文件
    files = sorted(os.listdir(OUTDIR))
    print(f"导出目录 {OUTDIR}:")
    for f in files:
        print(f"  {f}  {os.path.getsize(os.path.join(OUTDIR,f))/1e3:.1f} KB")

    # 记录尺寸供 grasp 脚本使用
    np.savez(os.path.join(OUTDIR, "duck_meta.npz"), center=center,
             extent=(V2.max(0) - V2.min(0)),
             mean_color=np.asarray(geom.visual.vertex_colors)[:, :3].mean(0))
    print(f"meta center={np.round(center,4).tolist()} extent={np.round(V2.max(0)-V2.min(0),4).tolist()}")


if __name__ == "__main__":
    main()

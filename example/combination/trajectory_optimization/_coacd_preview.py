"""临时脚本: 对 7_vggt_omega GLB 中被抓取物体 (粉色东西, 对应 grasp_hawor 的 glb_4)
做 CoACD 凸分解, 渲染 "原始网格 + 凸分解碰撞体" 对比图, 输出 PNG。

不修改任何原始管线代码, 仅做可视化。
"""
import os
import numpy as np
import trimesh
import coacd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

GLB = "/home/an/data/ras/7_vggt_omega/final_scene.glb"
OUT = os.path.join(os.path.dirname(__file__), "coacd_preview.png")

scene = trimesh.load(GLB)

geom_to_node = {}
for node in scene.graph.nodes:
    if node == "world":
        continue
    try:
        d = scene.graph[node]
        if isinstance(d, tuple) and len(d) == 2 and d[1]:
            geom_to_node[d[1]] = node
    except Exception:
        pass

# 收集所有三角网格几何 (idx, real_name, vertices, faces, color)
geoms = []
for gi, (gn, gm) in enumerate(scene.geometry.items()):
    rn = geom_to_node.get(gn, gn)
    if not hasattr(gm, "faces") or not hasattr(gm, "vertices"):
        continue
    v = np.asarray(gm.vertices, dtype=float)
    f = np.asarray(gm.faces, dtype=np.int32)
    if v.shape[0] == 0 or f.shape[0] == 0:
        continue
    col = None
    try:
        vc = gm.visual.vertex_colors
        if vc is not None and len(vc) > 0:
            rgb = vc[:, :3].mean(0)
            col = (rgb / 255.0).tolist()
    except Exception:
        pass
    geoms.append((gi, rn, v, f, col))

# 选被抓取物体: 优先粉色物体 (r>0.4, g<0.35, b>g), 否则回退 glb_4 (idx==4)
target = None
for t in geoms:
    col = t[4]
    if col and col[0] > 0.4 and col[1] < 0.35 and col[2] > col[1]:
        target = t
        break
if target is None:
    for t in geoms:
        if t[0] == 4:
            target = t
            break
if target is None:
    target = geoms[0]

gi, rn, verts, faces, col = target
is_pink = bool(col) and (col[0] > 0.4 and col[1] < 0.35 and col[2] > col[1])
print(f"[target] glb_{gi}  real_name={rn}  verts={verts.shape[0]}  "
      f"faces={faces.shape[0]}  rgb={np.round(col[:3],3) if col else None}  pink={is_pink}")

# 居中到 bbox 中心 (local 坐标, 与管线一致)
center = verts.min(0) + (verts.max(0) - verts.min(0)) / 2
v_local = verts - center

# CoACD 凸分解
mesh = coacd.Mesh(np.ascontiguousarray(v_local, dtype=np.double),
                  np.ascontiguousarray(faces, dtype=np.int32))
parts = coacd.run_coacd(mesh, threshold=0.05, max_convex_hull=32, resolution=2000)
print(f"[coacd] 凸分解得到 {len(parts)} 个凸包")

# ---- 渲染 ----
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

# 原始网格: 抽样边线框 (轻量, 仅作形状参照)
n = faces.shape[0]
sample = np.random.RandomState(0).choice(n, size=min(n, 1500), replace=False)
edges = []
for fc in faces[sample]:
    p0, p1, p2 = v_local[fc[0]], v_local[fc[1]], v_local[fc[2]]
    edges.append([p0, p1]); edges.append([p1, p2]); edges.append([p2, p0])
ax.add_collection3d(Line3DCollection(edges, colors="lightgray", alpha=0.15, linewidths=0.2))

# 凸分解碰撞体: 各凸包不同半透明颜色
cmap = plt.get_cmap("tab10")
for i, (pv, pf) in enumerate(parts):
    tri = Poly3DCollection([pv[face] for face in pf], alpha=0.5)
    tri.set_facecolor(cmap(i % 10))
    tri.set_edgecolor("black")
    tri.set_linewidth(0.3)
    ax.add_collection3d(tri)

# 等比例显示 (matplotlib 3.1 无 set_box_aspect)
allv = np.vstack([v_local] + [pv for pv, _ in parts])
mn, mx = allv.min(0), allv.max(0)
cx = (mn + mx) / 2.0
half = float((mx - mn).max()) / 2.0 * 1.1
for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), cx):
    set_lim(c - half, c + half)

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"CoACD convex-decomposition collision bodies - glb_{gi} ({rn}, grasped/pink)\n"
             f"{len(parts)} convex hulls  (threshold=0.05, max_hull=32, resolution=2000)\n"
             f"orig mesh: {verts.shape[0]} verts / {faces.shape[0]} faces  |  local frame centered")
ax.view_init(elev=25, azim=45)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[done] 图片已保存: {OUT}")

#!/usr/bin/env python3
"""Diagnostic 3: 检查 URDF 夹爪的 visual 是否真正加载 (mesh 是否被解析)."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sapien
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gt5", os.path.join(os.path.dirname(__file__), "05_gripper_test.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

scene = gt.create_scene()

with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    robot, *_ = gt.load_gripper(scene, "right")
    for w in wlist:
        print("WARN:", w.message)

print("robot name:", robot.name)
for l in robot.get_links():
    if l.name == "world":
        continue
    vb = getattr(l, "get_visual_bodies", lambda: [])()
    nb = getattr(l, "get_collision_bodies", lambda: [])()
    print(f"link={l.name}  visual_bodies={len(vb)}  collision_bodies={len(nb)}")
    for b in vb:
        try:
            print("    visual:", b.get_visual_shapes() if hasattr(b, "get_visual_shapes") else b)
        except Exception as e:
            print("    visual err", e)

# 也测试: 直接 add_visual_from_file 在 gp 处, 并把相机放到原点看向 gripper
cam = gt.create_camera(scene)
gp = np.array([0.20, 0.0, 0.50])
gt.set_camera_pose(cam, np.array([0.45, 0.30, 0.9]), gp)  # 更远更高, 确保看到
mat = sapien.render.RenderMaterial(); mat.base_color = [0.95,0.35,0.05,1.0]
b = scene.create_actor_builder(); b.add_visual_from_file(str(gt.R1_MESH_DIR/"right_gripper_link.STL"), material=mat)
b.build_static(name="stl_at_gp")
for _ in range(2): scene.step()
scene.update_render(); cam.take_picture()
rgb = np.array(cam.get_picture("Color"))[...,:3]
o = (rgb[...,0]>0.5)&(rgb[...,1]<0.4)&(rgb[...,2]<0.15)
print(f"[STL@gp] 橙色像素={o.mean()*100:.2f}%  RGBmax={rgb.max(0)}")

# 相机放在原点看向 gp 测 URDF 夹爪 (它是 fix_root, 应在 gp)
gt.set_camera_pose(cam, np.array([0.45,0.30,0.9]), gp)
scene.update_render(); cam.take_picture()
rgb = np.array(cam.get_picture("Color"))[...,:3]
o = (rgb[...,0]>0.5)&(rgb[...,1]<0.4)&(rgb[...,2]<0.15)
print(f"[URDF@gp] 橙色像素={o.mean()*100:.2f}%  RGBmax={rgb.max(0)}")
print("诊断完成.")

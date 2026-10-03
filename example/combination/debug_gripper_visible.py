#!/usr/bin/env python3
"""诊断: 让夹爪在离屏画面中清晰可见.
方法:
  1) 相机大幅拉近, 使夹爪占据画面主体.
  2) 对每个夹爪 link 叠加一个橙色基础几何体(kinematic actor, 每帧同步 link 世界位姿),
     绕开 GLB 内嵌材质无法改色的问题, 保证夹爪为醒目橙色.
  3) 渲染单帧 PNG, 统计橙色像素占比, 用 read_file 直接查看确认.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sapien
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gt5", os.path.join(os.path.dirname(__file__), "05_gripper_test.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

PREFIX = "right"

# 橙色高亮叠加: 对每个 link 建一个 kinematic actor + 基础几何体(box), 颜色固定橙色.
# SAPIEN 基础几何体(add_box_visual)使用 RenderShape 的 base_color, 不受 GLB 内嵌材质影响.
def add_orange_overlays(scene, robot):
    links = {l.get_name(): l for l in robot.get_links()}
    # 各 link 近似包围盒半尺寸(米), 仅用于高亮显示, 不要求精确贴合.
    half = {
        f"{PREFIX}_gripper_link": np.array([0.035, 0.035, 0.025]),
        f"{PREFIX}_gripper_finger_link1": np.array([0.012, 0.012, 0.045]),
        f"{PREFIX}_gripper_finger_link2": np.array([0.012, 0.012, 0.045]),
    }
    overlays = []
    # 自发光橙色: 即使背景很亮、材质本身非金属, 也能保持饱和橙色.
    orange_mat = sapien.render.RenderMaterial(
        base_color=[0.6, 0.05, 0.0, 1.0],
        emission=[2.0, 0.05, 0.0, 1.0],
        metallic=0.0,
        roughness=0.9,
        specular=0.0,
    )
    for name, h in half.items():
        l = links.get(name)
        if l is None:
            print(f"  未找到 link: {name}")
            continue
        b = scene.create_actor_builder()
        b.add_box_visual(half_size=h, material=orange_mat)
        actor = b.build_kinematic(name=f"overlay_{name}")
        overlays.append((actor, l))
        print(f"  高亮叠加: {name}  half={h}")
    return overlays

def sync_overlays(overlays):
    for actor, l in overlays:
        actor.set_pose(l.get_pose())

def main():
    scene = gt.create_scene()
    robot, *_ = gt.load_gripper(scene, PREFIX)

    # 设一个代表性位姿: 夹爪悬在 z≈0.47, 手指微张.
    q = robot.get_qpos().copy()
    aj = robot.get_active_joints()
    n2i = {j.get_name(): i for i, j in enumerate(aj)}
    q[n2i["virtual_x"]] = 0.20
    q[n2i["virtual_y"]] = 0.00
    q[n2i["virtual_z"]] = 0.47
    q[n2i["virtual_rz"]] = 0.0
    q[n2i["virtual_ry"]] = 0.0
    q[n2i["virtual_rx"]] = 0.0
    q[n2i["right_gripper_finger_joint1"]] = 0.04
    q[n2i["right_gripper_finger_joint2"]] = 0.04
    robot.set_qpos(q)
    for _ in range(gt.DECIMATION):
        pf = robot.compute_passive_force(gravity=True)
        robot.set_qf(pf)
        scene.step()

    overlays = add_orange_overlays(scene, robot)
    sync_overlays(overlays)

    # 取夹爪本体世界坐标作为机位目标; 相机大幅拉近.
    gpos = None
    for l in robot.get_links():
        if "gripper_link" in l.get_name() and "finger" not in l.get_name():
            gpos = l.get_pose().p
    if gpos is None:
        gpos = robot.get_links()[1].get_pose().p
    print("夹爪本体世界坐标:", np.round(gpos, 3))

    cam = gt.create_camera(scene, width=640, height=480, fov_deg=40)
    # 拉近: 偏移量远小于之前的 [0.14,0.09,0.12]
    eye = gpos + np.array([0.06, 0.05, 0.07])
    gt.set_camera_pose(cam, eye, gpos)

    scene.step()
    scene.update_render()
    cam.take_picture()
    rgba = np.array(cam.get_picture("Color"))[..., :3]
    # 线性 -> sRGB (与 render_frame 一致)
    rgb_lin = np.clip(rgba, 0, 1)
    rgb = np.where(rgb_lin < 0.0031308, 12.92 * rgb_lin, 1.055 * rgb_lin ** (1/2.4) - 0.055)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    import imageio.v2 as imageio
    out = "/tmp/diag_visible.png"
    imageio.imwrite(out, rgb)
    print(f"已保存 {out}")

    r, g, b = rgb[..., 0]/255.0, rgb[..., 1]/255.0, rgb[..., 2]/255.0
    orange = (r > 0.55) & (g < 0.55) & (b < 0.35) & (r - b > 0.25)
    print(f"橙色像素占比: {orange.mean()*100:.2f}%  非背景像素占比: {(~((r>0.8)&(g>0.8)&(b>0.8))).mean()*100:.2f}%")
    print("诊断完成.")

if __name__ == "__main__":
    main()

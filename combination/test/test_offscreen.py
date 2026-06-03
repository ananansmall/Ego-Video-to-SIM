#!/usr/bin/env python3
import os
os.environ.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/nvidia_icd.json')
os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

import numpy as np
import trimesh
import sapien
import sapien.render
import cv2

GLB_PATH = os.path.join(os.path.dirname(__file__), 'output/alignment/scene_in_sapien.glb')
OUTPUT = os.path.join(os.path.dirname(__file__), 'output/test_offscreen.png')

def main():
    if not os.path.exists(GLB_PATH):
        print(f"✗ GLB 文件不存在: {GLB_PATH}")
        return

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 1, -1], [2.0, 2.0, 2.0])
    scene.add_directional_light([-1, -0.5, -1], [1.5, 1.4, 1.3])
    scene.set_ambient_light([0.3, 0.3, 0.3])

    tm_scene = trimesh.load(GLB_PATH, force='scene')
    all_v = np.vstack([g.vertices for g in tm_scene.geometry.values()])
    obj_center = all_v.mean(axis=0)
    obj_size = np.linalg.norm(all_v.max(axis=0) - all_v.min(axis=0))

    actors = []
    temp_files = []
    for geom_idx, (geom_name, geom) in enumerate(tm_scene.geometry.items()):
        avg_color = [0.5, 0.5, 0.5, 1.0]
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vc = geom.visual.vertex_colors
            if len(vc) > 0:
                avg_rgb = np.asarray(vc[:, :3], dtype=np.float64).mean(axis=0)
                avg_color = [float(avg_rgb[0])/255.0, float(avg_rgb[1])/255.0, float(avg_rgb[2])/255.0, 1.0]

        temp_ply = f'/tmp/test_offscreen_{os.getpid()}_{geom_idx}.ply'
        geom.export(temp_ply)
        temp_files.append(temp_ply)

        builder = scene.create_actor_builder()
        if avg_color != [0.5, 0.5, 0.5, 1.0]:
            material = sapien.render.RenderMaterial(
                base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3
            )
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)
        actor = builder.build_static(name=geom_name)
        actors.append(actor)
        print(f"  ✓ {geom_name}: color={avg_color}")

    print(f"\n物体中心: {obj_center}")
    print(f"物体大小: {obj_size:.3f}m")

    camera = scene.add_camera("test", 960, 540, np.deg2rad(77), 0.01, 100.0)

    cam_pos = obj_center + np.array([0, -obj_size * 1.5, obj_size * 0.5])
    forward = obj_center - cam_pos
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0, 0, 1.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    from pytransform3d import rotations as pr
    cam_R = np.eye(3)
    cam_R[:, 0] = -forward
    cam_R[:, 1] = up
    cam_R[:, 2] = right
    cam_quat = pr.quaternion_from_matrix(cam_R)
    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

    print(f"\n相机位置: {cam_pos}")
    print(f"相机朝向: {forward}")

    scene.update_render()
    camera.take_picture()
    rgb = camera.get_picture("Color")[..., :3]

    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(OUTPUT, bgr)

    mean_brightness = rgb.mean()
    non_black = (rgb > 0.01).sum()
    total = rgb.size
    print(f"\n渲染结果: {OUTPUT}")
    print(f"  平均亮度: {mean_brightness:.4f}")
    print(f"  非黑色像素: {non_black}/{total} ({non_black/total*100:.1f}%)")

    if mean_brightness < 0.05:
        print("\n✗ 图像几乎全黑 - 离屏相机无法看到物体!")
    elif mean_brightness > 0.05 and non_black / total < 0.3:
        print("\n⚠ 图像大部分为空 - 物体可能不在视野内")
    else:
        print("\n✓ 离屏相机可以看到物体!")

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    main()

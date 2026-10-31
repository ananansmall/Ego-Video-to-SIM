"""
Stage 4 对齐能力测试

测试方案:
1. 用 Replica GT mesh + GT depth + GT extrinsic 作为基准
2. 给 mesh 施加一个已知的扰动变换（平移+旋转）
3. 运行 Stage 4 对齐
4. 比较对齐结果与 GT 的差距

这样可以直接衡量 Stage 4 的对齐能力，不受物体拆分质量的影响。

Usage:
    python scripts/test_stage4_alignment.py \
        --replica_dir /mnt/data_8THDD/lza/dataset/Replica \
        --scene room0 \
        --output_dir /tmp/stage4_test \
        --max_frames 50 \
        --perturbation_t 0.3 \
        --perturbation_r 15
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import cv2
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_replica_trajectory(traj_path):
    traj = []
    with open(traj_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 16:
                traj.append(np.array(vals).reshape(4, 4))
    return np.array(traj)


def c2w_opengl_to_w2c_opencv_zup(c2w_opengl):
    """Convert Replica c2w (OpenGL, y-up) to w2c (OpenCV, z-up).

    Replica traj.txt stores camera-to-world in OpenGL convention:
      - y-up world, camera looks along -z (OpenGL standard)
    But the depth images were rendered with camera looking along +z
    (Habitat convention), so we need to flip the camera z-axis first.

    Conversion chain:
      c2w_gl_yup @ flip_z → c2w_gl_correct
      inv(c2w_gl_correct) → w2c_gl_yup
      gl2cv @ w2c_gl_yup → w2c_cv_yup
      y2z @ w2c_cv_yup @ inv(y2z) → w2c_cv_zup
    """
    y2z = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    gl2cv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    flip_z = np.diag([1.0, 1.0, -1.0, 1.0])
    c2w_corrected = c2w_opengl @ flip_z
    w2c_gl = np.linalg.inv(c2w_corrected)
    w2c_cv = gl2cv @ w2c_gl
    w2c_cv_zup = y2z @ w2c_cv @ np.linalg.inv(y2z)
    return w2c_cv_zup


def prepare_test_data(replica_dir, scene_name, output_dir, max_frames,
                      perturbation_t, perturbation_r):
    scene_dir = os.path.join(replica_dir, scene_name)
    results_dir = os.path.join(scene_dir, "results")
    traj_path = os.path.join(scene_dir, "traj.txt")
    cam_params_path = os.path.join(replica_dir, "cam_params.json")
    mesh_path = os.path.join(replica_dir, f"{scene_name}_mesh.ply")

    for p in [scene_dir, results_dir, traj_path, cam_params_path, mesh_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    with open(cam_params_path) as f:
        cam_params = json.load(f)["camera"]

    fx, fy = cam_params["fx"], cam_params["fy"]
    cx, cy = cam_params["cx"], cam_params["cy"]
    depth_scale = cam_params["scale"]

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    os.makedirs(output_dir, exist_ok=True)
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    ext_dir = os.path.join(output_dir, "extrinsics")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    total_frames = len(frame_files)
    print(f"Replica {scene_name}: {total_frames} 帧")

    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    traj = load_replica_trajectory(traj_path)

    for new_idx, orig_idx in enumerate(indices):
        shutil.copy2(
            os.path.join(results_dir, f"frame{orig_idx:06d}.jpg"),
            os.path.join(color_dir, f"{new_idx}.jpg"))

        depth_raw = cv2.imread(
            os.path.join(results_dir, f"depth{orig_idx:06d}.png"),
            cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / depth_scale
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{new_idx}.png"), depth_mm)

        src_idx = min(orig_idx, len(traj) - 1)
        w2c = c2w_opengl_to_w2c_opencv_zup(traj[src_idx])
        np.savetxt(os.path.join(ext_dir, f"{new_idx}.txt"), w2c, fmt='%.15e')

    np.savetxt(os.path.join(output_dir, "intrinsic.txt"), intrinsic, fmt='%.6f')

    # ── 加载 mesh 并转换为 z-up ──
    mesh_yup = trimesh.load(mesh_path)
    y2z = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    mesh_zup = mesh_yup.copy()
    mesh_zup.apply_transform(y2z)

    # ── 生成扰动变换 ──
    np.random.seed(42)
    angle_rad = np.deg2rad(perturbation_r)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    from scipy.spatial.transform import Rotation
    R_perturb = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    t_perturb = np.random.uniform(-perturbation_t, perturbation_t, size=3)

    T_perturb = np.eye(4)
    T_perturb[:3, :3] = R_perturb
    T_perturb[:3, 3] = t_perturb

    print(f"\n扰动变换:")
    print(f"  旋转: {perturbation_r}°, 轴: {axis}")
    print(f"  平移: {t_perturb}")
    print(f"  平移范数: {np.linalg.norm(t_perturb):.4f}m")

    # ── 创建 GLB: GT + 扰动版本 ──
    scene = trimesh.Scene()

    mesh_gt_zup = mesh_zup.copy()
    scene.add_geometry(mesh_gt_zup, node_name="GT_object", geom_name="GT_object")

    mesh_perturbed = mesh_zup.copy()
    mesh_perturbed.apply_transform(T_perturb)
    scene.add_geometry(mesh_perturbed, node_name="perturbed_object", geom_name="perturbed_object")

    # 转回 y-up 保存 GLB
    z2y = np.linalg.inv(y2z)
    scene_yup = trimesh.Scene()
    for name, geom in scene.geometry.items():
        geom_yup = geom.copy()
        geom_yup.apply_transform(z2y)
        T_node = np.eye(4)
        scene_yup.add_geometry(geom_yup, node_name=name, geom_name=name,
                               transform=T_node)

    glb_path = os.path.join(output_dir, "final_scene_basic.glb")
    scene_yup.export(glb_path)
    print(f"GLB已保存: {glb_path} (GT + perturbed)")

    # ── 保存测试元数据 ──
    meta = {
        "scene": scene_name,
        "num_frames": len(indices),
        "perturbation_rotation_deg": perturbation_r,
        "perturbation_translation_m": t_perturb.tolist(),
        "T_perturb": T_perturb.tolist(),
        "y2z": y2z.tolist(),
    }
    meta_path = os.path.join(output_dir, "test_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # ── 计算初始对齐质量 ──
    print(f"\n=== 初始对齐质量 ===")
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from archive.stage4.renderer import MeshRenderer
    from archive.stage4.projection_alignment import compute_depth_accuracy, compute_depth_iou

    renderer = MeshRenderer(intrinsic, depth_m.shape[1], depth_m.shape[0])

    depths = []
    for i in range(len(indices)):
        d = cv2.imread(os.path.join(depth_dir, f"{i}.png"), cv2.IMREAD_UNCHANGED)
        depths.append(d.astype(np.float32) / 1000.0)
    depths = np.array(depths)

    extrinsics = []
    for i in range(len(indices)):
        ext = np.loadtxt(os.path.join(ext_dir, f"{i}.txt"))
        extrinsics.append(ext)
    extrinsics = np.array(extrinsics)

    # mesh_gt_zup is already in z-up. render_mesh expects z-up mesh.
    T_identity = np.eye(4)
    sample_frames = list(range(0, len(indices), max(1, len(indices) // 10)))

    gt_acc = compute_depth_accuracy(mesh_gt_zup, T_identity, depths,
                                    extrinsics, renderer, sample_frames)
    gt_iou = compute_depth_iou(mesh_gt_zup, T_identity, depths,
                               extrinsics, renderer, sample_frames)
    print(f"  GT mesh: Acc@10% = {gt_acc:.4f}, IoU = {gt_iou:.4f}")

    perturbed_acc = compute_depth_accuracy(mesh_perturbed, T_identity, depths,
                                           extrinsics, renderer, sample_frames)
    perturbed_iou = compute_depth_iou(mesh_perturbed, T_identity, depths,
                                      extrinsics, renderer, sample_frames)
    print(f"  扰动 mesh: Acc@10% = {perturbed_acc:.4f}, IoU = {perturbed_iou:.4f}")
    print(f"  差距: Acc Δ = {gt_acc - perturbed_acc:.4f}, IoU Δ = {gt_iou - perturbed_iou:.4f}")

    print(f"\n✅ 测试数据准备完成!")
    print(f"输出目录: {output_dir}")
    print(f"\n运行 Stage 4 对齐:")
    print(f"  cd {REPO_ROOT}")
    print(f"  conda run -n ReplicateAnyScene python stage4/run_alignment.py \\")
    print(f"    --input_path {output_dir} \\")
    print(f"    --output_dir {output_dir}/aligned \\")
    print(f"    --num_iterations 8")

    return {
        "gt_acc": gt_acc,
        "gt_iou": gt_iou,
        "perturbed_acc": perturbed_acc,
        "perturbed_iou": perturbed_iou,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 4 alignment with Replica GT data")
    parser.add_argument("--replica_dir", type=str,
                        default="/mnt/data_8THDD/lza/dataset/Replica")
    parser.add_argument("--scene", type=str, default="room0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--perturbation_t", type=float, default=0.3,
                        help="扰动平移范围 (米)")
    parser.add_argument("--perturbation_r", type=float, default=15.0,
                        help="扰动旋转角度 (度)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/tmp/stage4_test_{args.scene}"

    prepare_test_data(args.replica_dir, args.scene, args.output_dir,
                      args.max_frames, args.perturbation_t, args.perturbation_r)

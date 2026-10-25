import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
import json
import argparse
import numpy as np
import cv2
import trimesh
import pyrender


def load_extrinsics(extrinsics_dir):
    ext_files = sorted([f for f in os.listdir(extrinsics_dir) if f.endswith('.txt')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    extrinsics = []
    for f in ext_files:
        ext = np.loadtxt(os.path.join(extrinsics_dir, f))
        extrinsics.append(ext)
    return extrinsics


def render_scene_at_view(scene_mesh, intrinsic, extrinsic, width, height):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    c2w_opencv = np.linalg.inv(extrinsic)
    opencv_to_opengl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
    c2w_opengl = c2w_opencv @ opencv_to_opengl
    zup_to_yup = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    cam_pose = zup_to_yup @ c2w_opengl

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    mesh_node = scene.add(scene_mesh)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth


def render_all_views(glb_path, intrinsic, extrinsics, color_dir, output_render_dir):
    os.makedirs(output_render_dir, exist_ok=True)

    print(f"📦 加载场景: {glb_path}")
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.dump())
    else:
        mesh = scene

    mesh_pr = pyrender.Mesh.from_trimesh(mesh)

    gt_files = sorted([f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))],
                      key=lambda x: int(os.path.splitext(x)[0]))
    if gt_files:
        sample_img = cv2.imread(os.path.join(color_dir, gt_files[0]))
        height, width = sample_img.shape[:2]
    else:
        height, width = 480, 640

    print(f"🖼️  渲染 {len(extrinsics)} 个视角 ({width}x{height})...")

    rendered_count = 0
    for i, ext in enumerate(extrinsics):
        try:
            color, depth = render_scene_at_view(mesh_pr, intrinsic, ext, width, height)
            out_path = os.path.join(output_render_dir, f"{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            rendered_count += 1
        except Exception as e:
            print(f"   ⚠️  帧{i}渲染失败: {e}")

    print(f"✅ 渲染完成: {rendered_count}/{len(extrinsics)} 帧")
    return output_render_dir


def run_visual_assessment(output_path, compute_lpips_flag=True, sample_count=None, glb_path=None):
    from assess.visual_metrics import evaluate_visual_quality

    color_dir = os.path.join(output_path, 'color')
    render_dir = os.path.join(output_path, 'rendered')

    if glb_path is None:
        glb_path = os.path.join(output_path, 'final_scene_stage5.glb')

    if not os.path.exists(render_dir) or len(os.listdir(render_dir)) == 0:
        intrinsic_path = os.path.join(output_path, 'intrinsic.txt')
        extrinsics_dir = os.path.join(output_path, 'extrinsics')

        if not os.path.exists(glb_path):
            print("⚠️  未找到 GLB 文件，跳过视觉评估")
            return {}
        if not os.path.exists(intrinsic_path) or not os.path.exists(extrinsics_dir):
            print("⚠️  未找到相机参数，跳过视觉评估")
            return {}

        intrinsic = np.loadtxt(intrinsic_path)
        extrinsics = load_extrinsics(extrinsics_dir)
        render_all_views(glb_path, intrinsic, extrinsics, color_dir, render_dir)

    print("\n📊 视觉质量评估 (PSNR / SSIM / LPIPS)...")
    results = evaluate_visual_quality(render_dir, color_dir, sample_count=sample_count,
                                       compute_lpips_flag=compute_lpips_flag)
    return results


def run_geometry_assessment(output_path, sample_count=None, glb_path=None):
    from assess.geometry_metrics import evaluate_spatial_accuracy

    print("\n📊 空间精度评估 (渲染深度 vs VGGT深度)...")
    results = evaluate_spatial_accuracy(output_path, sample_count=sample_count, glb_path=glb_path)
    return results


def run_mesh_geometry_assessment(generated_mesh_path, reference_mesh_path,
                                 num_sample_points=100000, f_thresholds=None,
                                 normalize=True):
    from assess.geometry_metrics import evaluate_mesh_geometry

    norm_str = "归一化" if normalize else "未归一化"
    print(f"\n📊 几何精度评估 (Mesh vs Mesh: CD / F-Score / NC / Accuracy / Completeness) [{norm_str}]...")
    results = evaluate_mesh_geometry(generated_mesh_path, reference_mesh_path,
                                     num_sample_points=num_sample_points,
                                     f_thresholds=f_thresholds,
                                     normalize=normalize)
    return results


def run_pointcloud_geometry_assessment(generated_ply_path, reference_ply_path,
                                       f_thresholds=None, voxel_size=0.005,
                                       normalize=True):
    from assess.geometry_metrics import evaluate_point_cloud_geometry

    norm_str = "归一化" if normalize else "未归一化"
    print(f"\n📊 几何精度评估 (点云 vs 点云: CD / F-Score / NC / Accuracy / Completeness) [{norm_str}]...")
    results = evaluate_point_cloud_geometry(generated_ply_path, reference_ply_path,
                                            f_thresholds=f_thresholds,
                                            voxel_size=voxel_size,
                                            normalize=normalize)
    return results


def run_textual_assessment(category_path, ground_truth_json=None, output_path=None):
    from assess.textual_metrics import evaluate_textual

    if not os.path.exists(category_path):
        print("⚠️  未找到类别JSON，跳过文本评估")
        return {}

    print("\n📊 文本完整性评估...")
    results = evaluate_textual(category_path, ground_truth_json, output_path)
    return results


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj


def print_summary(all_results):
    print("\n" + "=" * 70)
    print("📊 评估结果汇总")
    print("=" * 70)

    if 'visual' in all_results:
        v = all_results['visual']
        print("\n  [视觉质量 - 全图]")
        print(f"    PSNR:  {v.get('PSNR', 'N/A'):.2f} dB" if 'PSNR' in v else "    PSNR:  N/A")
        print(f"    SSIM:  {v.get('SSIM', 'N/A'):.4f}" if 'SSIM' in v else "    SSIM:  N/A")
        if 'LPIPS' in v:
            print(f"    LPIPS: {v['LPIPS']:.4f}")
        if 'render_coverage' in v:
            print(f"    渲染覆盖率: {v['render_coverage']:.2%}")
        if 'PSNR_masked' in v:
            print("\n  [视觉质量 - 仅物体区域 (masked)]")
            print(f"    PSNR_masked:  {v['PSNR_masked']:.2f} dB")
            print(f"    SSIM_masked:  {v['SSIM_masked']:.4f}")
        if 'LPIPS_masked' in v:
            print(f"    LPIPS_masked: {v['LPIPS_masked']:.4f}")

    if 'spatial' in all_results:
        s = all_results['spatial']
        print("\n  [空间精度 - 渲染深度 vs VGGT深度]")
        print(f"    MaskIoU:     {s.get('MaskIoU', 'N/A'):.4f}" if 'MaskIoU' in s else "    MaskIoU:     N/A")
        print(f"    DepthRMSE:   {s.get('DepthRMSE', 'N/A'):.4f} m" if 'DepthRMSE' in s else "    DepthRMSE:   N/A")
        print(f"    DepthAbsRel: {s.get('DepthAbsRel', 'N/A'):.4f}" if 'DepthAbsRel' in s else "    DepthAbsRel: N/A")

    if 'mesh_geometry' in all_results:
        g = all_results['mesh_geometry']
        norm_tag = " [归一化]" if g.get('normalized', False) else " [未归一化]"
        print(f"\n  [几何精度 - Mesh vs GT Mesh{norm_tag}]")
        if 'ChamferDistance_L2' in g:
            print(f"    CD-L2:  {g['ChamferDistance_L2']:.6f}")
        if 'ChamferDistance_L1' in g:
            print(f"    CD-L1:  {g['ChamferDistance_L1']:.6f}")
        if 'Accuracy' in g:
            print(f"    Accuracy:     {g['Accuracy']:.6f}")
        if 'Completeness' in g:
            print(f"    Completeness: {g['Completeness']:.6f}")
        if 'Overall' in g:
            print(f"    Overall:      {g['Overall']:.6f}")
        for key in sorted(g.keys()):
            if key.startswith('F-Score_'):
                thresh = key.replace('_', '.').replace('F-Score.', '')
                print(f"    F-Score@{thresh}: {g[key]:.4f}")
        if 'NormalConsistency' in g:
            print(f"    NormalConsistency: {g['NormalConsistency']:.4f}")
        if 'ChamferDistance_L2_raw' in g:
            print(f"    --- 原始空间 (未归一化) ---")
            print(f"    CD-L2_raw:  {g['ChamferDistance_L2_raw']:.6f}")
            print(f"    CD-L1_raw:  {g['ChamferDistance_L1_raw']:.6f}")
            print(f"    Accuracy_raw:     {g['Accuracy_raw']:.6f}")
            print(f"    Completeness_raw: {g['Completeness_raw']:.6f}")
            print(f"    Overall_raw:      {g['Overall_raw']:.6f}")

    if 'pointcloud_geometry' in all_results:
        g = all_results['pointcloud_geometry']
        norm_tag = " [归一化]" if g.get('normalized', False) else " [未归一化]"
        print(f"\n  [几何精度 - 点云 vs GT 点云{norm_tag}]")
        if 'ChamferDistance_L2' in g:
            print(f"    CD-L2:  {g['ChamferDistance_L2']:.6f}")
        if 'ChamferDistance_L1' in g:
            print(f"    CD-L1:  {g['ChamferDistance_L1']:.6f}")
        if 'Accuracy' in g:
            print(f"    Accuracy:     {g['Accuracy']:.6f}")
        if 'Completeness' in g:
            print(f"    Completeness: {g['Completeness']:.6f}")
        if 'Overall' in g:
            print(f"    Overall:      {g['Overall']:.6f}")
        for key in sorted(g.keys()):
            if key.startswith('F-Score_'):
                thresh = key.replace('_', '.').replace('F-Score.', '')
                print(f"    F-Score@{thresh}: {g[key]:.4f}")
        if 'NormalConsistency' in g:
            print(f"    NormalConsistency: {g['NormalConsistency']:.4f}")
        if 'ChamferDistance_L2_raw' in g:
            print(f"    --- 原始空间 (未归一化) ---")
            print(f"    CD-L2_raw:  {g['ChamferDistance_L2_raw']:.6f}")
            print(f"    CD-L1_raw:  {g['ChamferDistance_L1_raw']:.6f}")
            print(f"    Accuracy_raw:     {g['Accuracy_raw']:.6f}")
            print(f"    Completeness_raw: {g['Completeness_raw']:.6f}")
            print(f"    Overall_raw:      {g['Overall_raw']:.6f}")

    if 'textual' in all_results:
        t = all_results['textual']
        print("\n  [文本完整性]")
        if 'Recall' in t:
            print(f"    Recall:    {t['Recall']:.2%}")
        if 'Precision' in t:
            print(f"    Precision: {t['Precision']:.2%}")
        if 'F1' in t:
            print(f"    F1:        {t['F1']:.2%}")
        if 'num_predicted' in t:
            print(f"    检测类别数: {t['num_predicted']}")
        if 'num_generated' in t:
            print(f"    生成类别数: {t['num_generated']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ReplicateAnyScene Assessment")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory from main.py")
    parser.add_argument("--glb_path", type=str, default=None,
                        help="Custom GLB file path for assessment (default: output_path/final_scene_stage5.glb)")
    parser.add_argument("--category_path", type=str, default=None, help="Category JSON path")
    parser.add_argument("--ground_truth_json", type=str, default=None, help="Ground truth category JSON")
    parser.add_argument("--reference_mesh", type=str, default=None,
                        help="Ground truth reference mesh for geometry evaluation (GLB/OBJ/PLY)")
    parser.add_argument("--reference_ply", type=str, default=None,
                        help="Ground truth reference point cloud for geometry evaluation (PLY)")
    parser.add_argument("--skip_visual", action="store_true", help="Skip visual assessment")
    parser.add_argument("--skip_geometry", action="store_true", help="Skip spatial accuracy assessment")
    parser.add_argument("--skip_textual", action="store_true", help="Skip textual assessment")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS computation (faster)")
    parser.add_argument("--sample_count", type=int, default=None, help="Number of frames to sample for assessment")
    parser.add_argument("--num_sample_points", type=int, default=100000,
                        help="Number of points to sample for mesh geometry evaluation")
    parser.add_argument("--f_thresholds", type=float, nargs='+', default=[0.01, 0.02, 0.05],
                        help="Distance thresholds for F-Score computation")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable point cloud normalization (MonoSDF protocol uses normalization by default)")
    args = parser.parse_args()

    print("=" * 70)
    print("📊 ReplicateAnyScene 评估")
    print("=" * 70)
    print(f"   输出目录: {args.output_path}")
    if args.glb_path:
        print(f"   GLB文件: {args.glb_path}")

    all_results = {}

    glb_path = args.glb_path

    if not args.skip_visual:
        visual = run_visual_assessment(args.output_path, not args.skip_lpips, args.sample_count, glb_path=glb_path)
        if visual:
            all_results['visual'] = visual

    if not args.skip_geometry:
        geom = run_geometry_assessment(args.output_path, args.sample_count, glb_path=glb_path)
        if geom:
            all_results['spatial'] = geom

    eval_glb_path = glb_path if glb_path else os.path.join(args.output_path, 'final_scene_stage5.glb')
    normalize = not args.no_normalize
    if args.reference_mesh and os.path.exists(args.reference_mesh) and os.path.exists(eval_glb_path):
        mesh_geom = run_mesh_geometry_assessment(
            eval_glb_path, args.reference_mesh,
            num_sample_points=args.num_sample_points,
            f_thresholds=args.f_thresholds,
            normalize=normalize
        )
        if mesh_geom:
            all_results['mesh_geometry'] = mesh_geom

    ply_path = os.path.join(args.output_path, 'point_cloud.ply')
    if args.reference_ply and os.path.exists(args.reference_ply) and os.path.exists(ply_path):
        pc_geom = run_pointcloud_geometry_assessment(
            ply_path, args.reference_ply,
            f_thresholds=args.f_thresholds,
            normalize=normalize
        )
        if pc_geom:
            all_results['pointcloud_geometry'] = pc_geom

    if not args.skip_textual:
        cat_path = args.category_path
        if cat_path is None:
            for candidate in ['hallway.json', 'scene.json']:
                p = os.path.join(args.output_path, candidate)
                if os.path.exists(p):
                    cat_path = p
                    break
        if cat_path:
            text = run_textual_assessment(cat_path, args.ground_truth_json, args.output_path)
            if text:
                all_results['textual'] = text

    print_summary(all_results)

    results_path = os.path.join(args.output_path, 'assessment_results.json')
    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估结果已保存: {results_path}")

    return all_results


if __name__ == "__main__":
    main()

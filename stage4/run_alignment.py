"""
Stage 4 GLB Optimizer — Independent Pose Refinement Tool

Reads a scene GLB + VGGT output data from INPUT directory,
optimizes object poses via 2D-3D correspondence-based alignment (MASt3R-style),
and writes an improved GLB to OUTPUT directory.

This module does NOT need to run as part of the main pipeline.
It only requires the output directory produced by Stages 1-3.

Required input directory layout:
    <input_path>/
    ├── final_scene_basic.glb       # Scene GLB from Stage 3
    ├── intrinsic.txt         # (3,3) camera intrinsic matrix
    ├── color/                # RGB images   (0.jpg, 1.jpg, ...)
    ├── depth/                # Depth maps   (0.png, ...) in mm uint16
    └── extrinsics/           # Camera extrinsics (0.txt, ...) 4x4

Usage:
    # Basic: input and output are the same directory
    conda run -n ReplicateAnyScene python stage4/run_alignment.py \
        --input_path ./outputs/beizi

    # Separate input and output directories
    conda run -n ReplicateAnyScene python stage4/run_alignment.py \
        --input_path ./outputs/beizi \
        --output_dir ./outputs/beizi_aligned

    # With more iterations
    conda run -n ReplicateAnyScene python stage4/run_alignment.py \
        --input_path ./outputs/beizi \
        --num_iterations 12

Dependencies:
    - numpy, scipy, opencv-python, trimesh, pyrender
    - All available in the ReplicateAnyScene conda environment
    - No GPU required (pyrender uses EGL offscreen rendering)
"""

import os
import sys
import argparse
import shutil
import numpy as np
import cv2
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def load_vggt_results(input_path):
    """Load all VGGT prediction results from the input directory."""
    color_dir = os.path.join(input_path, 'color')
    depth_dir = os.path.join(input_path, 'depth')
    ext_dir = os.path.join(input_path, 'extrinsics')
    intrinsic_path = os.path.join(input_path, 'intrinsic.txt')

    for required in [color_dir, depth_dir, ext_dir, intrinsic_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required path not found: {required}")

    color_files = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    num_frames = len(color_files)
    print(f"  {num_frames} frames, ", end='')

    colors = []
    for fname in color_files:
        img = cv2.imread(os.path.join(color_dir, fname))
        colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    colors = np.array(colors)

    depths = []
    for i in range(num_frames):
        d = cv2.imread(os.path.join(depth_dir, f'{i}.png'), cv2.IMREAD_UNCHANGED)
        depths.append(d.astype(np.float32) / 1000.0)
    depths = np.array(depths)

    extrinsics = []
    for i in range(num_frames):
        ext = np.loadtxt(os.path.join(ext_dir, f'{i}.txt'))
        if ext.shape == (3, 4):
            full = np.eye(4)
            full[:3, :] = ext
            ext = full
        extrinsics.append(ext)
    extrinsics = np.array(extrinsics)

    intrinsic = np.loadtxt(intrinsic_path)

    print(f"image {colors.shape[1]}x{colors.shape[2]}, intrinsic loaded")
    return {
        'colors': colors,
        'depths': depths,
        'extrinsics': extrinsics,
        'intrinsic': intrinsic,
    }


def reconstruct_world_points(depths, extrinsics, intrinsic):
    """Reconstruct world_points from depth maps + camera params (VGGT convention)."""
    from stage4.projection_alignment import unproject_depth_to_world

    H, W = depths.shape[1], depths.shape[2]
    world_points = np.zeros((len(depths), H, W, 3), dtype=np.float32)

    for i, (d, ext) in enumerate(zip(depths, extrinsics)):
        world_points[i] = unproject_depth_to_world(d, ext, intrinsic)

    return world_points


def load_scene_instances(input_path):
    """Load the final_scene_basic.glb and extract per-object meshes."""
    glb_path = os.path.join(input_path, 'final_scene_basic.glb')
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"final_scene_basic.glb not found in {input_path}")

    scene = trimesh.load(glb_path)

    instances = {}
    if isinstance(scene, trimesh.Scene):
        node_to_geom = {}
        for node in scene.graph.nodes:
            T, geom_name = scene.graph.get(node)
            if geom_name is not None and geom_name in scene.geometry:
                node_to_geom[geom_name] = (node, T)

        for geom_name in scene.geometry:
            mesh = scene.geometry[geom_name]

            # Skip non-mesh geometries (e.g., Path3D for floor grids)
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"  ⏭️ Skipping non-mesh geometry: {geom_name} (type={type(mesh).__name__})")
                continue

            if geom_name in node_to_geom:
                node_name, T = node_to_geom[geom_name]
            else:
                T = np.eye(4)
                node_name = geom_name

            mesh.apply_transform(T)

            yup_to_zup = np.array([[1, 0, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]], dtype=np.float64)
            mesh.apply_transform(yup_to_zup)

            category = node_name.rsplit('_', 1)[0] if '_' in node_name else node_name
            if category not in instances:
                instances[category] = []
            instances[category].append({
                'original_mesh': mesh,
                'T': np.eye(4),
                'node_name': node_name,
            })
            print(f"  {node_name} ({geom_name}): {len(mesh.vertices)} verts, bounds={mesh.bounds[0].round(3)}~{mesh.bounds[1].round(3)}")

    return instances


def create_depth_based_masks(all_instances, depths, extrinsics, intrinsic, world_points):
    """Create per-instance masks by rendering + depth comparison."""
    from stage4.renderer import MeshRenderer
    H, W = depths.shape[1], depths.shape[2]
    num_frames = len(depths)
    renderer = MeshRenderer(intrinsic, W, H)

    all_masks = {}
    for category, inst_list in all_instances.items():
        cat_masks = []
        for inst in inst_list:
            mesh = inst['original_mesh']
            T = inst['T']
            im_list = []
            for fid in range(num_frames):
                _, dr, mr = renderer.render_mesh(mesh, T, extrinsics[fid])
                d_real = depths[fid]
                close = mr & (d_real > 0) & (dr > 0) & (np.abs(dr - d_real) < 0.15)
                if close.sum() < 20:
                    close = mr & (d_real > 0) & (dr > 0) & (np.abs(dr - d_real) < 0.45)
                if close.sum() > 20:
                    im_list.append({'frame_id': fid, 'mask': close})
            if im_list:
                cat_masks.append(im_list)
        all_masks[category] = cat_masks

    renderer.delete()
    return all_masks


def compute_optimal_frame_ids(all_masks, world_points):
    """Select the frame with largest surface area for each instance."""
    try:
        from src.geometry_utils import compute_surface_area_from_pointmap
        use_surface_area = True
    except ImportError:
        use_surface_area = False

    result = {}
    for cat, cat_masks in all_masks.items():
        ids = []
        for im_list in cat_masks:
            best_f, best_score = 0, 0
            for im in im_list:
                if use_surface_area:
                    score = compute_surface_area_from_pointmap(
                        world_points[im['frame_id']], im['mask'])
                else:
                    score = im['mask'].sum()
                if score > best_score:
                    best_score, best_f = score, im['frame_id']
            ids.append(best_f)
        result[cat] = ids
    return result


def save_aligned_glb(all_instances, output_dir, filename='aligned_scene.glb'):
    """Save the aligned scene as GLB (y-up convention, same as main.py)."""
    os.makedirs(output_dir, exist_ok=True)
    scene = trimesh.Scene()
    for category, inst_list in all_instances.items():
        for i, inst in enumerate(inst_list):
            m = inst['original_mesh'].copy()
            m.apply_transform(inst['T'])
            scene.add_geometry(m, node_name=f"{category}_{i}")
    zup_to_yup = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, -1, 0, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
    scene.apply_transform(zup_to_yup)
    path = os.path.join(output_dir, filename)
    scene.export(path)
    print(f"  Saved: {path}")


def copy_input_data_to_output(input_path, output_dir):
    """Copy necessary data files from input to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    subdirs = ['color', 'depth', 'extrinsics']
    files = ['intrinsic.txt']

    for subdir in subdirs:
        src = os.path.join(input_path, subdir)
        dst = os.path.join(output_dir, subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"  Copied: {subdir}/")

    for fname in files:
        src = os.path.join(input_path, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    src_glb = os.path.join(input_path, 'final_scene_basic.glb')
    dst_glb = os.path.join(output_dir, 'final_scene_basic.glb')
    if os.path.exists(src_glb) and not os.path.exists(dst_glb):
        shutil.copy2(src_glb, dst_glb)
        print(f"  Copied: final_scene_basic.glb")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 GLB Optimizer — 2D-3D Correspondence-Based Pose Refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Input and output in the same directory
  conda run -n ReplicateAnyScene python stage4/run_alignment.py \
      --input_path ./outputs/beizi

  # Separate input and output directories
  conda run -n ReplicateAnyScene python stage4/run_alignment.py \
      --input_path ./outputs/beizi \
      --output_dir ./outputs/beizi_aligned

  # With more iterations
  conda run -n ReplicateAnyScene python stage4/run_alignment.py \
      --input_path ./outputs/beizi \
      --output_dir ./outputs/beizi_aligned \
      --num_iterations 12

  # Allow scale adjustment (default: position-only rigid transform)
  conda run -n ReplicateAnyScene python stage4/run_alignment.py \
      --input_path ./outputs/beizi \
      --output_dir ./outputs/beizi_aligned \
      --with_scale
        """)
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input directory with final_scene_basic.glb + VGGT data (color/, depth/, extrinsics/, intrinsic.txt)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for aligned GLB (default: same as input_path)')
    parser.add_argument('--num_iterations', type=int, default=8,
                        help='Number of alignment iterations (default: 8)')
    parser.add_argument('--temporal_radius', type=int, default=5,
                        help='Temporal neighborhood radius (default: 5)')
    parser.add_argument('--with_scale', action='store_true',
                        help='Allow scale adjustment (default: position-only rigid transform)')
    parser.add_argument('--use_mast3r', action='store_true',
                        help='Use MASt3R for 2D matching (paper method, requires GPU)')
    parser.add_argument('--no_mast3r', action='store_true',
                        help='Disable MASt3R, use depth-based matching instead')
    parser.add_argument('--mast3r_device', type=str, default='cuda',
                        help='Device for MASt3R inference (default: cuda)')
    parser.add_argument('--output_filename', type=str, default='aligned_scene.glb',
                        help='Output GLB filename (default: aligned_scene.glb)')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_path

    os.chdir(REPO_ROOT)

    print("=" * 60)
    print("Stage 4 GLB Optimizer — 2D-3D Correspondence-Based Pose Refinement")
    print("=" * 60)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_dir}")

    # ── 1. Load data ──
    print("\n[1/5] Loading VGGT results...")
    vggt = load_vggt_results(input_path)

    print("[2/5] Reconstructing world points...")
    world_points = reconstruct_world_points(vggt['depths'], vggt['extrinsics'], vggt['intrinsic'])
    world_points_conf = np.ones_like(vggt['depths'], dtype=np.float32)
    vggt['world_points'] = world_points
    vggt['world_points_conf'] = world_points_conf

    print("[3/5] Loading scene GLB...")
    all_instances = load_scene_instances(input_path)

    print("[4/5] Creating instance masks...")
    all_masks = create_depth_based_masks(
        all_instances, vggt['depths'], vggt['extrinsics'],
        vggt['intrinsic'], world_points,
    )
    all_optimal_frame_ids = compute_optimal_frame_ids(all_masks, world_points)

    for cat in all_instances:
        print(f"  {cat}: {len(all_instances[cat])} instances, "
              f"optimal frames: {all_optimal_frame_ids.get(cat, [])}")

    # ── 2. Run alignment ──
    use_mast3r = args.use_mast3r and not args.no_mast3r
    print("\n[5/5] Running pose optimization...")
    if use_mast3r:
        print("  Mode: MASt3R (paper method, requires GPU)")
    else:
        print("  Mode: Depth-based matching (no MASt3R)")
    from stage4.combined_alignment import refine_single_instance_combined

    # 计算总实例数
    total_instances = 0
    for cat, insts in all_instances.items():
        total_instances += len(insts)
    current_instance = 0

    for category, cat_insts in all_instances.items():
        cat_masks = all_masks.get(category, [])
        cat_fids = all_optimal_frame_ids.get(category, [])

        if len(cat_insts) != len(cat_masks):
            print(f"  [Warning] {category}: instance/mask count mismatch, skipping")
            current_instance += len(cat_insts)
            continue

        for iid, (inst, masks) in enumerate(zip(cat_insts, cat_masks)):
            opt_fid = cat_fids[iid] if iid < len(cat_fids) else 0
            print(f"\n  >>> [{current_instance+1}/{total_instances}] {category} #{iid} (optimal frame: {opt_fid})")

            inst = refine_single_instance_combined(
                instance_info=inst,
                instance_masks=masks,
                optimal_frame_id=opt_fid,
                world_points=world_points,
                world_points_conf=world_points_conf,
                depths=vggt['depths'],
                extrinsics=vggt['extrinsics'],
                intrinsic=vggt['intrinsic'],
                colors=vggt['colors'],
                num_icp_iterations=args.num_iterations,
                temporal_radius=args.temporal_radius,
                instance_index=current_instance,
                total_instances=total_instances,
                instance_name=f"{category}_{iid}",
                use_mast3r=use_mast3r,
                mast3r_device=args.mast3r_device,
            )
            cat_insts[iid] = inst
            current_instance += 1

    # ── 3. Save ──
    print("\n" + "=" * 60)
    if output_dir != input_path:
        print("Copying input data to output directory...")
        copy_input_data_to_output(input_path, output_dir)

    print("Saving optimized GLB...")
    save_aligned_glb(all_instances, output_dir, args.output_filename)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

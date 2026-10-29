import numpy as np
import torch
import os
import sys
import json
import multiprocessing as mp
import importlib.util
import traceback
from queue import Empty
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

# ── T6 尺寸比门禁 ──────────────────────────────────────────────────────────
# ratio = SAM3D mesh 世界系包围盒最大边 / 该实例 mask 点云包围盒最大边。
# 点云与 SAM3D 输入同源 (world_points), 是一手观测, 尺度天然正确;
# ratio 显著偏离 1 说明 SAM3D 建大/建小了 (实测 002 pillow 3.82 / toy 0.18,
# 2chair bottle 0.10 / table 1.28 / chair 0.62, 121 mug 0.43, 151 stool 0.37)。
SCALE_GATE_LOW = 0.5
SCALE_GATE_HIGH = 2.0
# 点云少于该数时包围盒不可靠, 跳过门禁 (与 static_mesh.MIN_POINTS 对齐)
SCALE_GATE_MIN_POINTS = 60


def _mesh_world_max_extent(original_mesh, T):
    """mesh 应用 T 后的世界系 AABB 最大边长。"""
    m = original_mesh.copy()
    m.apply_transform(np.asarray(T, dtype=np.float64))
    return float(np.max(m.bounds[1] - m.bounds[0]))


def _scale_gate_instance(category, inst_idx, instance_result, mask,
                         world_points, frame_id):
    """T6: 对单个 SAM3D 实例做尺寸比门禁。

    ratio < 0.5 或 > 2.0 的实例 → 放弃 SAM3D mesh, 改用点云构网
    (src/static_mesh.py 的 build_mesh_from_pointcloud + build_static_instance,
    点云构网产物是世界坐标, T=单位阵)。正常实例原样返回, 不动。

    Returns:
        (instance_result, gate_entry): instance_result 可能被替换;
        gate_entry 为 None 表示点太少/出错无法评估 (不做门禁), 否则是
        一条报告记录 (action: pass / replaced_with_pointcloud /
        kept_sam3d_pointcloud_build_failed)。
    """
    name = f"{category}_inst{inst_idx}"
    try:
        mask = np.asarray(mask)
        p = np.asarray(world_points[frame_id])[mask > 0]
        p = p[np.all(np.isfinite(p), axis=-1)]
        if len(p) < SCALE_GATE_MIN_POINTS:
            return instance_result, None
        mesh_ext = _mesh_world_max_extent(
            instance_result["original_mesh"], instance_result["T"])
        pc_ext = float(np.max(p.max(axis=0) - p.min(axis=0)))
        if mesh_ext <= 1e-9 or pc_ext <= 1e-9:
            return instance_result, None
        ratio = mesh_ext / pc_ext
        entry = {
            "instance": name,
            "category": category,
            "frame_id": int(frame_id),
            "mesh_max_extent": round(mesh_ext, 4),
            "pointcloud_max_extent": round(pc_ext, 4),
            "ratio": round(ratio, 3),
        }
        if SCALE_GATE_LOW <= ratio <= SCALE_GATE_HIGH:
            entry["action"] = "pass"
            return instance_result, entry
        # 超限 → 尝试点云构网替换
        from src.static_mesh import build_mesh_from_pointcloud, build_static_instance
        tm = build_mesh_from_pointcloud(p)
        if tm is None:
            entry["action"] = "kept_sam3d_pointcloud_build_failed"
            print(f"[scale-gate] {name}: ratio={ratio:.2f} 超限但点云构网失败, "
                  f"保留 SAM3D mesh", flush=True)
            return instance_result, entry
        replacement = build_static_instance(tm, category, inst_idx)
        replacement["mesh_source"] = "pointcloud_scale_gate"  # 区分于常规静态路径
        replacement["scale_gate"] = entry                     # 标记随实例走, 供父进程落盘
        entry["action"] = "replaced_with_pointcloud"
        entry["ratio_after"] = round(
            float(np.max(tm.bounds[1] - tm.bounds[0])) / pc_ext, 3)
        print(f"[scale-gate] {name}: ratio={ratio:.2f} 超限 "
              f"(mesh {mesh_ext:.3f}m vs 点云 {pc_ext:.3f}m) → "
              f"替换为点云构网 mesh, ratio→{entry['ratio_after']:.2f}", flush=True)
        return replacement, entry
    except Exception as e:
        return instance_result, {
            "instance": name, "action": "gate_error",
            "error": f"{type(e).__name__}: {e}",
        }


def _resolve_output_dir_from_argv():
    """从命令行参数解析管线输出目录 (mainv2/mainv3 的 --output_path)。

    未显式给出时复刻其默认推导: output_v2/{视频名}_{vggt_model}。
    """
    argv = list(sys.argv)
    for i, a in enumerate(argv):
        if a == "--output_path" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--output_path="):
            return a.split("=", 1)[1]
    video = None
    for flag in ("--input_video", "--input_images"):
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv):
                video = argv[i + 1]
                break
            if a.startswith(flag + "="):
                video = a.split("=", 1)[1]
                break
        if video:
            break
    if video:
        model = "vggt"
        for i, a in enumerate(argv):
            if a == "--vggt_model" and i + 1 < len(argv):
                model = argv[i + 1]
                break
            if a.startswith("--vggt_model="):
                model = a.split("=", 1)[1]
                break
        stem = os.path.splitext(os.path.basename(video))[0]
        return os.path.join(".", "output_v2", f"{stem}_{model}")
    return None


def write_scale_gate_report(all_instances):
    """把被尺寸门禁替换的实例落盘为 <output>/scale_gate_report.json。

    标记在 _scale_gate_instance 里随实例走 (instance["scale_gate"]),
    这里在父进程收集写出 —— 父进程才有命令行, 能解析出输出目录。
    """
    try:
        entries = []
        for _cat, lst in (all_instances or {}).items():
            for x in lst:
                g = x.get("scale_gate") if isinstance(x, dict) else None
                if g:
                    entries.append(dict(g))
        if not entries:
            return
        report = {
            "gate": "scale_ratio",
            "low": SCALE_GATE_LOW,
            "high": SCALE_GATE_HIGH,
            "n_replaced": sum(1 for e in entries
                              if e.get("action") == "replaced_with_pointcloud"),
            "n_kept_sam3d": sum(1 for e in entries
                                if e.get("action") != "replaced_with_pointcloud"),
            "instances": entries,
        }
        out_dir = _resolve_output_dir_from_argv()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "scale_gate_report.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
            print(f"[scale-gate] 报告已写出: {path} "
                  f"(替换 {report['n_replaced']}, 保留 SAM3D {report['n_kept_sam3d']})",
                  flush=True)
        else:
            print(f"[scale-gate] ⚠️ 未能解析输出目录, 报告仅打印: "
                  f"{json.dumps(report, ensure_ascii=False)}", flush=True)
    except Exception as e:
        print(f"[scale-gate] ⚠️ 报告写出失败: {type(e).__name__}: {e}", flush=True)


def generate_3d_asset(image, mask, pointmap, extrinsic, inference):
    '''
    3D asset generation for a single object instance using sam3d. 
    Args:
        image: (H, W, 3) numpy array, RGB image of the scene
        mask: (H, W) numpy array, binary mask of the object instance
        pointmap: (H, W, 3) numpy array, world coordinates of each pixel in the image
        extrinsic: (4, 4) numpy array, camera extrinsic matrix for the image
        inference: sam3d inference object
    Returns:    A dictionary containing:
        - mesh: A trimesh object representing the reconstructed 3D mesh of the object instance in object coordinate system
        - T: the transformation matrix to align the mesh to the world coordinate system
    '''
    H, W = pointmap.shape[:2]
    points_world_flat = pointmap.reshape(-1, 3)  # shape: (H*W, 3)
    ones = np.ones((points_world_flat.shape[0], 1))
    points_world_hom = np.hstack([points_world_flat, ones])  # (N, 4)

    # vggt format to sam3d format
    points_cam_hom = (np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]) @ extrinsic @ points_world_hom.T).T  # (N, 4)
    points_cam_flat = points_cam_hom[:, :3]
    point_map_camera = torch.from_numpy(points_cam_flat).reshape(H, W, 3).to(torch.float32)
    point_map_camera = point_map_camera.contiguous()
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    original_mesh = output["glb"]

    # get the transformation matrix. The transformation is z-up.
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    matrix_l2c = l2c_transform.get_matrix()[0].transpose(0, 1).detach().cpu().numpy()
    matrix_y2z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    matrix_adjust = np.diag([-1, -1, 1, 1])
    matrix_ext_inv = np.linalg.inv(extrinsic)
    final_transform = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z

    return {
        "original_mesh": original_mesh,
        "T": final_transform
    }


def _generate_3d_asset_worker(queue, image, mask, pointmap, extrinsic, config_file, compile_model):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference

        inference = Inference(config_file=config_file, compile=compile_model)
        result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
        queue.put((True, result))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _generate_all_instances_worker(
    queue,
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file,
    compile_model,
):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference

        inference = Inference(config_file=config_file, compile=compile_model)
        all_instances = {}
        for category, category_masks in deduplicated_all_masks.items():
            all_instances[category] = []
            for inst_idx, (instance_masks, optimal_frame_id) in enumerate(
                    zip(category_masks, all_optimal_frame_ids[category])):
                image = colors[optimal_frame_id]
                mask = next(im["mask"] for im in instance_masks if im["frame_id"] == optimal_frame_id)
                pointmap = world_points[optimal_frame_id]
                extrinsic = extrinsics[optimal_frame_id]
                print(
                    f"[SAM3D subprocess] category={category}, frame={optimal_frame_id}, "
                    f"mask_sum={int(np.asarray(mask).sum())}"
                )
                instance_result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
                # T6 尺寸比门禁: SAM3D mesh 与 mask 点云尺度严重不符的实例
                # 改用点云构网 (点云与 SAM3D 输入同源, 尺度天然正确)。
                instance_result, _gate_entry = _scale_gate_instance(
                    category, inst_idx, instance_result, mask,
                    world_points, optimal_frame_id)
                all_instances[category].append(instance_result)
        print("[SAM3D subprocess] finished generating all instances, putting results in the queue. This may take a while...")
        queue.put((True, all_instances))
    except Exception:
        queue.put((False, traceback.format_exc()))
    finally:
        queue.close()
        queue.cancel_join_thread()


def generate_3d_asset_in_subprocess(
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file="./models/SAM3D/checkpoints/pipeline.yaml",
    compile_model=False,
    only_instances=None,
):
    """only_instances: {category: [inst_idx, ...]} 或 None。
        None = 全部跑 SAM3D (原行为)。
        非 None = 只对这些实例跑 SAM3D, 其余实例从 all_masks 中剔除, 由调用方
        用点云快速路径补上 (见 src/static_mesh.py)。这是"静态物体不烧 SAM3D"
        的提速开关 —— 实测只有 1~2 个实例被抓取的场景里, 绝大部分实例是静态的。
    """
    if only_instances is not None:
        deduplicated_all_masks = {
            cat: [m for k, m in enumerate(masks)
                  if k in set(only_instances.get(cat, []))]
            for cat, masks in deduplicated_all_masks.items()
        }
        all_optimal_frame_ids = {
            cat: [f for k, f in enumerate(ids)
                  if k in set(only_instances.get(cat, []))]
            for cat, ids in all_optimal_frame_ids.items()
        }
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_generate_all_instances_worker,
        args=(
            queue,
            deduplicated_all_masks,
            all_optimal_frame_ids,
            colors,
            world_points,
            extrinsics,
            config_file,
            compile_model,
        ),
    )
    process.start()
    try:
        try:
            # Read the payload before join(); joining first can deadlock on Queue finalizers.
            ok, payload = queue.get(timeout=7200)
        except Empty:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            raise RuntimeError("subprocess timed out waiting for result")

        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

        if ok:
            # T6: 被尺寸门禁替换的实例落盘 (需要父进程的命令行来定位输出目录)
            write_scale_gate_report(payload)
            return payload
        raise RuntimeError(payload)
    finally:
        queue.close()
        queue.cancel_join_thread()

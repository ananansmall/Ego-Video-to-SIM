import os
import io
import numpy as np
import torch
import trimesh
from PIL import Image

from src.vlm_logger import vlm_inference, set_vlm_log_dir

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor


VLM_CANDIDATES = [
    "/mnt/data/lza/models/Qwen3.5-9B",
    "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct",
]


def _find_vlm_checkpoint():
    for candidate in VLM_CANDIDATES:
        if not os.path.exists(candidate):
            continue
        if "snapshots" in candidate:
            snapshots_dir = os.path.join(candidate, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])
        else:
            return candidate
    return None


def _load_vlm(checkpoint_path):
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, processor


# VLM 推理函数已从 src/vlm_logger 导入，不再在此处定义


def _check_mesh_size(instance_info, category, world_points, instance_masks, min_ratio=0.05, max_ratio=5.0):
    """
    Vote 1: mesh size check.
    Compare the bounding box extents of the generated mesh against the 3D point cloud
    extents of the original mask region.

    Uses max-axis ratio (not volume ratio) because:
    - Volume ratio is extremely sensitive to occlusion (cubic amplification)
    - A partially visible object's point cloud bbox is much smaller than the full mesh
    - Max-axis ratio is more robust: it only compares the longest dimension
    - For a correctly sized mesh, max_axis_ratio should be close to 1.0
    - For a hallucinated mesh that fills the scene, max_axis_ratio >> 1.0
    """
    mesh = instance_info['original_mesh']
    T = instance_info['T']

    transformed = mesh.copy()
    transformed.apply_transform(T)
    mesh_extents = transformed.bounding_box.extents

    point_extents_list = []
    for im in instance_masks:
        fid = im['frame_id']
        mask = im['mask']
        if fid >= world_points.shape[0]:
            continue
        pts = world_points[fid]
        valid = mask > 0
        if not np.any(valid):
            continue
        p = pts[valid]
        finite = np.all(np.isfinite(p), axis=-1)
        if not np.any(finite):
            continue
        p = p[finite]
        ext = p.max(axis=0) - p.min(axis=0)
        point_extents_list.append(ext)

    if not point_extents_list:
        return None

    avg_point_extents = np.mean(point_extents_list, axis=0)
    if np.max(avg_point_extents) < 1e-8:
        return None

    # Max-axis ratio: mesh最长轴 vs 点云最长轴
    mesh_max_axis = float(np.max(mesh_extents))
    point_max_axis = float(np.max(avg_point_extents))
    max_axis_ratio = mesh_max_axis / max(point_max_axis, 1e-8)

    # Per-axis ratio: 检查每个轴的比例
    per_axis_ratios = mesh_extents / np.maximum(avg_point_extents, 1e-8)

    # 判定逻辑：
    # 1. max_axis_ratio > max_ratio: mesh的最长轴远大于点云最长轴 → 幻觉（mesh膨胀到整个场景）
    # 2. max_axis_ratio < min_ratio: mesh远小于点云 → mesh退化（几乎没有几何）
    # 3. 某个轴的比例 > max_ratio * 2: mesh在某个方向异常膨胀 → 幻觉
    # 4. 但如果 mesh 在遮挡方向（点云extent很小的轴）更大是正常的，不应判为幻觉

    # 找出点云中"遮挡方向"（extent最小的轴）
    min_point_axis = float(np.min(avg_point_extents))
    occluded_axes = avg_point_extents < (min_point_axis * 3)

    # 对非遮挡方向检查比例
    visible_axis_ratios = per_axis_ratios[~occluded_axes]
    if len(visible_axis_ratios) > 0:
        max_visible_ratio = float(np.max(visible_axis_ratios))
    else:
        max_visible_ratio = max_axis_ratio

    if max_axis_ratio < min_ratio:
        return False, f"mesh太小 (max_axis_ratio={max_axis_ratio:.4f} < {min_ratio})"
    if max_visible_ratio > max_ratio:
        return False, f"mesh太大 (max_visible_ratio={max_visible_ratio:.4f} > {max_ratio}, per_axis={per_axis_ratios})"
    return True, f"mesh大小合理 (max_axis_ratio={max_axis_ratio:.4f}, per_axis={per_axis_ratios})"


def _check_point_cloud_quality(instance_info, category, world_points, world_points_conf, instance_masks, min_conf=0.3, min_valid_ratio=0.3):
    """
    Vote 2: point cloud quality check.
    Check if the 3D point cloud in the mask region has enough confidence.
    Low confidence means VGGT couldn't reconstruct this region well,
    so the generated 3D asset is likely unreliable.
    """
    total_pixels = 0
    high_conf_pixels = 0

    for im in instance_masks:
        fid = im['frame_id']
        mask = im['mask']
        if fid >= len(world_points_conf):
            continue
        conf = world_points_conf[fid]
        valid = mask > 0
        total_pixels += int(np.sum(valid))
        if total_pixels == 0:
            continue
        high_conf_pixels += int(np.sum(conf[valid] > min_conf))

    if total_pixels == 0:
        return None

    valid_ratio = high_conf_pixels / total_pixels
    if valid_ratio < min_valid_ratio:
        return False, f"点云质量差 (high_conf_ratio={valid_ratio:.2f} < {min_valid_ratio})"
    return True, f"点云质量OK (high_conf_ratio={valid_ratio:.2f})"


def _check_vlm_render(instance_info, category, model, processor):
    """
    Vote 3: VLM render check.
    Render the mesh and ask VLM if it looks like the expected category.
    """
    mesh = instance_info['original_mesh']
    try:
        scene_tmp = trimesh.Scene()
        scene_tmp.add_geometry(mesh)
        png = scene_tmp.save_image(resolution=[256, 256])
        if png is None:
            return None
        pil_img = Image.open(io.BytesIO(png))
    except Exception:
        return None

    prompt = f'Is this a 3D model of a "{category}"? Answer yes or no.'
    try:
        output = vlm_inference(pil_img, model, processor, prompt, max_new_tokens=10, caller="stage3_asset_verify")
        lower = output.lower()
        if 'yes' in lower:
            return True, f"VLM确认是'{category}' (回复: {output})"
        else:
            return False, f"VLM认为不是'{category}' (回复: {output})"
    except Exception:
        return None


def verify_all_instances(all_instances, all_optimal_frame_ids, deduplicated_all_masks,
                         world_points, world_points_conf, min_votes=2):
    """
    Multi-vote verification of all generated 3D assets.

    Three voting dimensions:
    1. Mesh size vs point cloud size ratio
    2. Point cloud confidence quality
    3. VLM render verification

    An instance is removed only if >= min_votes dimensions vote against it.

    Args:
        all_instances: dict {category: [instance_info, ...]}
        all_optimal_frame_ids: dict {category: [frame_id, ...]}
        deduplicated_all_masks: dict {category: [instance_masks, ...]}
        world_points: (T, H, W, 3)
        world_points_conf: (T, H, W)
        min_votes: minimum number of negative votes to remove an instance (default 2)

    Returns:
        verified_instances: dict {category: [instance_info, ...]} with bad instances removed
    """
    vlm_checkpoint = _find_vlm_checkpoint()
    vlm_model = None
    vlm_proc = None

    if vlm_checkpoint:
        print(f"\n🔍 Stage 3.5: 多维度投票验证3D资产")
        print(f"   VLM: {vlm_checkpoint}")
        print(f"   投票维度: mesh大小 / 点云质量 / VLM渲染")
        print(f"   否决阈值: >= {min_votes} 票否决才删除")
        vlm_model, vlm_proc = _load_vlm(vlm_checkpoint)
    else:
        print("⚠️  未找到VLM模型，仅使用几何验证")

    verified_instances = {}
    for category, category_instances in all_instances.items():
        category_masks = deduplicated_all_masks.get(category, [])
        verified = []
        for inst_idx, instance_info in enumerate(category_instances):
            instance_masks = category_masks[inst_idx] if inst_idx < len(category_masks) else []

            votes_negative = 0
            vote_details = []
            vlm_passed = None

            r1 = _check_mesh_size(instance_info, category, world_points, instance_masks)
            if r1 is not None:
                passed, detail = r1
                vote_details.append(f"大小:{'✅' if passed else '❌'}({detail})")
                if not passed:
                    votes_negative += 1

            r2 = _check_point_cloud_quality(instance_info, category, world_points, world_points_conf, instance_masks)
            if r2 is not None:
                passed, detail = r2
                vote_details.append(f"点云:{'✅' if passed else '❌'}({detail})")
                if not passed:
                    votes_negative += 1

            if vlm_model is not None:
                r3 = _check_vlm_render(instance_info, category, vlm_model, vlm_proc)
                if r3 is not None:
                    passed, detail = r3
                    vote_details.append(f"VLM:{'✅' if passed else '❌'}({detail})")
                    vlm_passed = passed
                    if not passed:
                        votes_negative += 1

            # Special logic: If VLM explicitly said "yes", keep it regardless of other votes
            if vlm_passed is True:
                verified.append(instance_info)
                if votes_negative > 0:
                    print(f"   ⚠️  {category}_{inst_idx} 保留（VLM确认）: {' | '.join(vote_details)}")
                else:
                    print(f"   ✅ {category}_{inst_idx} 保留: {' | '.join(vote_details)}")
                continue

            # If VLM didn't say yes (or unavailable), use original min_votes logic
            if votes_negative >= min_votes:
                print(f"   🚫 {category}_{inst_idx} 被否决 ({votes_negative}票反对): {' | '.join(vote_details)}")
            else:
                verified.append(instance_info)
                if votes_negative > 0:
                    print(f"   ⚠️  {category}_{inst_idx} 保留: {' | '.join(vote_details)}")
                else:
                    print(f"   ✅ {category}_{inst_idx} 保留: {' | '.join(vote_details)}")

        if verified:
            verified_instances[category] = verified

    if vlm_model is not None:
        del vlm_model, vlm_proc
        torch.cuda.empty_cache()

    print(f"✅ 3D资产验证完成: 保留 {sum(len(v) for v in verified_instances.values())} 个实例")
    return verified_instances

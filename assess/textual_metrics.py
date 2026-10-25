import json
import os


def load_categories_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return set(data.keys())
    elif isinstance(data, list):
        categories = set()
        for item in data:
            if isinstance(item, dict) and 'category' in item:
                categories.add(item['category'])
            elif isinstance(item, str):
                categories.add(item)
        return categories
    return set()


def load_categories_from_glb_dir(output_path):
    glb_path = os.path.join(output_path, "final_scene_stage5.glb")
    if not os.path.exists(glb_path):
        return set()

    import trimesh
    scene = trimesh.load(glb_path)
    categories = set()
    if isinstance(scene, trimesh.Scene):
        for name in scene.geometry.keys():
            cat = name.rsplit('_', 1)[0] if '_' in name else name
            categories.add(cat)
    return categories


def compute_recall(predicted, ground_truth):
    if len(ground_truth) == 0:
        return 0.0
    return len(predicted & ground_truth) / len(ground_truth)


def compute_precision(predicted, ground_truth):
    if len(predicted) == 0:
        return 0.0
    return len(predicted & ground_truth) / len(predicted)


def compute_f1(predicted, ground_truth):
    prec = compute_precision(predicted, ground_truth)
    rec = compute_recall(predicted, ground_truth)
    if prec + rec < 1e-8:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_srr(predicted_categories, all_frame_detections):
    """
    Semantic Redundancy Rate: fraction of detected categories that are duplicates.
    SRR = (total_detections - unique_categories) / total_detections
    """
    total = sum(len(dets) for dets in all_frame_detections)
    unique = len(predicted_categories)
    if total == 0:
        return 0.0
    return max(0.0, (total - unique) / total)


def evaluate_textual(predicted_json_path, ground_truth_json_path=None, output_path=None):
    predicted_cats = load_categories_from_json(predicted_json_path)

    if output_path:
        generated_cats = load_categories_from_glb_dir(output_path)
    else:
        generated_cats = predicted_cats

    results = {
        'predicted_categories': sorted(predicted_cats),
        'num_predicted': len(predicted_cats),
        'generated_categories': sorted(generated_cats),
        'num_generated': len(generated_cats),
    }

    if ground_truth_json_path and os.path.exists(ground_truth_json_path):
        gt_cats = load_categories_from_json(ground_truth_json_path)
        results['ground_truth_categories'] = sorted(gt_cats)
        results['num_ground_truth'] = len(gt_cats)

        results['Recall'] = compute_recall(generated_cats, gt_cats)
        results['Precision'] = compute_precision(generated_cats, gt_cats)
        results['F1'] = compute_f1(generated_cats, gt_cats)

        missing = gt_cats - generated_cats
        extra = generated_cats - gt_cats
        results['missing_categories'] = sorted(missing)
        results['extra_categories'] = sorted(extra)
    else:
        print("⚠️  未提供GT JSON，仅报告检测到的类别")

    return results

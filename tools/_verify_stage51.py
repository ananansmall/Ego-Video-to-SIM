"""临时验证脚本: 直接在 3chair 已有数据上重跑 scene_graph 版 Stage 5.1,
验证 key 不匹配修复 + other objects 不降级逻辑。跑完即删。
"""
import os, sys, json
import numpy as np

REPO_ROOT = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene"
sys.path.insert(0, REPO_ROOT)

from tools.infer_relations_scene_graph import infer_relations_scene_graph

SCENE_DIR = os.path.join(REPO_ROOT, "output_v2", "3chair-bottle_vggt_omega")
STAGE1_JSON = os.path.join(SCENE_DIR, "scene_3chair-bottle_stage1.json")
VLM = "/mnt/data/lza/models/Qwen3.5-9B"

with open(STAGE1_JSON) as f:
    categories_and_relations = json.load(f)

# 加载 Stage 2 的 deduplicated masks (用于画标注框)
import pickle as _pk
with open(os.path.join(SCENE_DIR, "_pre_dedup_masks.pkl"), "rb") as f:
    _pkl = _pk.load(f)
instance_masks = _pkl.get("all_masks", {})

print("=" * 70)
print("Stage 1 原始关系:", json.dumps(categories_and_relations, ensure_ascii=False))
print("=" * 70)

# 直接调用修复后的 scene_graph 5.1, 复用已有 optimal_frames / keyframes / dynamic_pointclouds
refined, vlm = infer_relations_scene_graph(
    scene_dir=SCENE_DIR,
    vlm_checkpoint=VLM,
    categories_and_relations=categories_and_relations,
    instance_masks=instance_masks,
    optimal_frames_dir=os.path.join(SCENE_DIR, "optimal_frames"),
    keyframes_dir=os.path.join(SCENE_DIR, "keyframes"),
    dynamic_frames_dir=os.path.join(SCENE_DIR, "dynamic_pointclouds"),
    output_dir=SCENE_DIR,
)

print("\n" + "=" * 70)
print("修复后 5.1 关系:")
for k, v in refined.items():
    flag = "" if v == categories_and_relations.get(k, categories_and_relations.get(k.rsplit('_',1)[0], "")) else "  ⚠️ 改动"
    print(f"  {k}: {v}{flag}")
print("=" * 70)

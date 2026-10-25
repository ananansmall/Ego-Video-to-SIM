"""
诊断脚本: 在 dry_run 模式下运行 resolve_penetrations,
只打印诊断日志, 不修改任何实例数据。

用法:
    cd ReplicateAnyScene
    python3 tools/diagnose_penetration.py --scene_dir output_v2/121_C5_CellPhone_161deg_vggt_omega
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import trimesh

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["LIDRA_SKIP_INIT"] = "true"

from tools.refine_inter_object_placement import resolve_penetrations


Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def load_instances(scene_dir):
    """与 run_post_pipeline.py 一致的加载逻辑 (z-up 空间, T 独立)"""
    import pickle as _pkl

    pkl_candidates = [
        os.path.join(scene_dir, "all_instances.pkl"),
        os.path.join(scene_dir, "all_instances_stage4.pkl"),
    ]

    for pkl_path in pkl_candidates:
        if os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                pkl_data = _pkl.load(f)
            if isinstance(pkl_data, dict) and "all_instances" in pkl_data:
                return pkl_data["all_instances"], f"pkl: {os.path.basename(pkl_path)}"
            return pkl_data, f"pkl: {os.path.basename(pkl_path)}"

    # Fallback: GLB
    for glb_name in ["final_scene_stage4.glb", "final_scene_stage5.glb", "final_scene_basic.glb"]:
        glb_path = os.path.join(scene_dir, glb_name)
        if os.path.isfile(glb_path):
            scene = trimesh.load_scene(glb_path)
            all_instances = {}
            for node_name, geom in scene.geometry.items():
                mesh = geom.copy()
                mesh.apply_transform(Z_UP_TO_Y_UP)
                parts = node_name.rsplit("_", 1)
                if len(parts) == 2:
                    cat, idx_str = parts
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        cat, idx = node_name, 0
                else:
                    cat, idx = node_name, 0
                all_instances.setdefault(cat, [])
                if len(all_instances[cat]) <= idx:
                    while len(all_instances[cat]) <= idx:
                        all_instances[cat].append(None)
                all_instances[cat][idx] = {
                    "original_mesh": mesh,
                    "T": np.eye(4),
                    "index": idx,
                }
            return all_instances, f"glb: {glb_name}"

    return None, "未找到"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--no-small-mutual", action="store_true",
                        help="关闭同层小物体水平互退分离, 用于A/B对比")
    parser.add_argument("--with-sp", action="store_true",
                        help="先跑SP精修(无VLM→完整z轴贴合), 再穿模迭代——模拟真实管线链路")
    parser.add_argument("--pre-upright", action="store_true",
                        help="STEP0: 把 supported by floor 物体提前重力转正+落地(模拟Phase1前置)")
    args = parser.parse_args()

    print(f"📂 场景目录: {args.scene_dir}")
    print(f"🔀 小物体互退分离: {'关闭' if args.no_small_mutual else '开启(默认)'}")
    all_instances, source = load_instances(args.scene_dir)
    print(f"📦 加载方式: {source}")
    for cat, insts in sorted(all_instances.items()):
        print(f"    {cat}: {len(insts)} 个实例")

    # 加载关系 (与 run_post_pipeline.py 加载方式一致)
    refined_relations = {}
    sg_path = os.path.join(args.scene_dir, "relations_scene_graph.json")
    if os.path.isfile(sg_path):
        with open(sg_path, "r") as f:
            sg_data = json.load(f)
        # 管线直接写入的成品dict {"bottle_0": "supported by table_0", ...}
        if isinstance(sg_data.get("refined_relations"), dict):
            refined_relations = sg_data["refined_relations"]
        else:
            for obj in sg_data.get("scene_graph_objects", []):
                name = obj.get("instance") or f"{obj['category']}_{obj['inst_idx']}"
                parent = obj.get("parent")
                rel = obj.get("relationship") or (
                    f"{obj['relation']} by {parent}" if parent else obj.get("relation", "")
                )
                if rel:
                    refined_relations[name] = rel
        print(f"📋 场景图关系: {len(refined_relations)} 条")

    # 加载 Stage1 关系 (整个json就是 categories_and_relations, 顶层为类别键)
    categories_and_relations = {}
    stage1_json_path = os.path.join(args.scene_dir, "stage1_scene.json")
    if not os.path.isfile(stage1_json_path):
        stage1_json_path = os.path.join(args.scene_dir, "stage1_scene_output.json")
    if not os.path.isfile(stage1_json_path):
        for f in os.listdir(args.scene_dir):
            if "stage1" in f and f.endswith(".json"):
                stage1_json_path = os.path.join(args.scene_dir, f)
                break
    if os.path.isfile(stage1_json_path):
        with open(stage1_json_path, "r") as f:
            categories_and_relations = json.load(f)
        print(f"📋 Stage1 关系: {len(categories_and_relations)} 条")

    print("\n" + "=" * 70)
    print("🔬 [SIM] 穿模仿真诊断: 深拷贝上跑完整8轮, 不影响已保存输出")
    print("=" * 70)

    # 深拷贝, 避免修改加载的数据
    import copy
    all_instances_sim = copy.deepcopy(all_instances)

    # STEP0: 提前重力转正 (仅 supported by floor 的物体, 模拟Phase1前置)
    if args.pre_upright:
        import numpy as _np
        import trimesh as _tm
        from tools.refine_inter_object_placement import _get_transformed_mesh as _gtm
        print("\n" + "=" * 70)
        print("🔧 [STEP0] 提前重力转正 (--pre-upright): supported by floor 物体")
        print("=" * 70)
        # pkl结构为 {category: [info,...]}, 展开成 name→info
        flat_instances = {}
        for cat, insts in all_instances_sim.items():
            if isinstance(insts, list):
                for i, v in enumerate(insts):
                    flat_instances[f"{cat}_{i}"] = v
            elif isinstance(insts, dict) and "T" in insts:
                flat_instances[cat] = insts
        for name, info in flat_instances.items():
            rel = refined_relations.get(name, "")
            if "floor" not in rel:
                continue
            T = info["T"]
            up_world = T[:3, :3] @ _np.array([0.0, 1.0, 0.0])  # GLB y-up
            up_world = up_world / (_np.linalg.norm(up_world) + 1e-12)
            cos_t = _np.clip(_np.dot(up_world, [0.0, 0.0, 1.0]), -1.0, 1.0)
            theta = _np.degrees(_np.arccos(cos_t))
            if theta < 1.0:
                print(f"   {name}: theta={theta:.2f}° 已正立, 跳过")
                continue
            # 绕质心旋转: 本地up → 世界Z
            mesh_cur = _gtm(info)
            center = mesh_cur.centroid
            align4 = _tm.geometry.align_vectors(up_world, _np.array([0.0, 0.0, 1.0]))
            c_h = _np.array([center[0], center[1], center[2], 1.0])
            A = (_tm.transformations.translation_matrix(center)
                 @ align4
                 @ _tm.transformations.translation_matrix(-center))
            info["T"] = A @ T
            # 落地: bottom_z 钳到 0
            m2 = _gtm(info)
            dz = -float(m2.bounds[0, 2])
            info["T"] = _tm.transformations.translation_matrix([0.0, 0.0, dz]) @ info["T"]
            m3 = _gtm(info)
            print(f"   {name}: theta={theta:.2f}°→0°, z_min={dz + float(m2.bounds[0, 2]):.4f}→"
                  f"{float(m3.bounds[0, 2]):.4f}m")

    # STEP1: SP精修 (无VLM → refinement_degree=None → 完整z轴贴合,
    # 等价于几何校验覆写后的真实管线行为)
    if args.with_sp:
        from tools.refine_inter_object_placement import refine_inter_object_relations
        print("\n" + "=" * 70)
        print("🔧 [STEP1] SP精修链路 (无VLM, 纯内存, 等价改动A后行为)")
        print("=" * 70)
        all_instances_sim = refine_inter_object_relations(
            all_instances_sim,
            refined_relations=refined_relations or {},
            verbose=True,
            vlm_checkpoint=None,
            scene_dir=None,
            categories_and_relations=categories_and_relations or {},
        )

    print("\n" + "=" * 70)
    print(f"🔬 [STEP2] 穿模仿真诊断{'(SP后)' if args.with_sp else '(原始pkl)'}, 深拷贝上运行, 不影响已保存输出")
    print("=" * 70)

    resolve_penetrations(
        all_instances_sim,
        refined_relations=refined_relations if refined_relations else None,
        verbose=True,
        dry_run=False,
        max_iterations=8,
        scene_dir=args.scene_dir,
        shared_penetration=True,
        small_mutual_separation=not args.no_small_mutual,
        categories_and_relations=categories_and_relations if categories_and_relations else None,
    )

    print("\n✅ 仿真完成, 原始数据未修改")


if __name__ == "__main__":
    main()
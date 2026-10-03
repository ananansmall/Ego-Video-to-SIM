"""诊断脚本: 检查 MANO 手部轨迹与物体在 SAPIEN 坐标系中的对齐情况

用户反馈: "本身来说两个坐标系对应了是可以完成的, 但可能离目标物体还有些许差距"
但日志显示 MANO 手腕距目标物体 10cm, 这不是 "些许差距".
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from trajectory_loader import load_hawor_data, load_hawor_c2w
from physics_utils import RXWORLD_TO_SAPIEN
import grasp_hawor
load_glb_with_physics = grasp_hawor.load_glb_with_physics

HAWOR_DIR = "/home/an/data/hawor/7"
RAS_DIR = "/home/an/data/ras/my_7mp4_result"


def main():
    # 1. HaWoR 手部数据
    print("=" * 70)
    print("1. HaWoR 手部数据 (pred_trans)")
    hawor_data = load_hawor_data(HAWOR_DIR, hand_idx=0)
    pred_trans = np.asarray(hawor_data["pred_trans"])
    print(f"   shape: {pred_trans.shape}")
    print(f"   HaWoR range: x[{pred_trans[:,0].min():.3f},{pred_trans[:,0].max():.3f}] "
          f"y[{pred_trans[:,1].min():.3f},{pred_trans[:,1].max():.3f}] "
          f"z[{pred_trans[:,2].min():.3f},{pred_trans[:,2].max():.3f}]")

    # 2. 相机轨迹
    print("\n2. 相机轨迹 (t_c2w)")
    _, t_c2w_all = load_hawor_c2w(HAWOR_DIR)
    t_c2w = np.asarray(t_c2w_all)
    print(f"   shape: {t_c2w.shape}")
    print(f"   HaWoR range: x[{t_c2w[:,0].min():.3f},{t_c2w[:,0].max():.3f}] "
          f"y[{t_c2w[:,1].min():.3f},{t_c2w[:,1].max():.3f}] "
          f"z[{t_c2w[:,2].min():.3f},{t_c2w[:,2].max():.3f}]")

    # 3. 手腕 SAPIEN 坐标 (pred_trans ≈ wrist)
    print("\n3. 手腕 SAPIEN 坐标 (pred_trans 直接变换)")
    wrist_sapien = (RXWORLD_TO_SAPIEN @ pred_trans.T).T
    print(f"   SAPIEN range: x[{wrist_sapien[:,0].min():.3f},{wrist_sapien[:,0].max():.3f}] "
          f"y[{wrist_sapien[:,1].min():.3f},{wrist_sapien[:,1].max():.3f}] "
          f"z[{wrist_sapien[:,2].min():.3f},{wrist_sapien[:,2].max():.3f}]")

    # 4. 物体位置
    print("\n4. 物体位置 (GLB)")
    import sapien
    scene = sapien.Scene()
    glb_path = Path(RAS_DIR) / "final_scene.glb"
    transform_params_path = Path("/home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics/output/gripper_only_left/alignment/transform_params.npz")
    print(f"   GLB: {glb_path} (存在: {glb_path.exists()})")
    print(f"   transform_params: {transform_params_path} (存在: {transform_params_path.exists()})")

    obj_bbox_centers = {}
    ground_z = 0.0
    if glb_path.exists() and transform_params_path.exists():
        obj_actors, ground_z, obj_bbox_centers, obj_info = load_glb_with_physics(
            str(glb_path), str(transform_params_path), scene=scene, fast_collision=True
        )
        print(f"   ground_z: {ground_z:.4f}")
        print(f"   物体中心 (SAPIEN):")
        for name, center in obj_bbox_centers.items():
            print(f"     {name}: {np.array(center).round(4)}")

    # 5. 手腕 vs 物体距离
    print("\n5. 手腕 vs 物体距离 (SAPIEN)")
    for name, center in obj_bbox_centers.items():
        center = np.array(center)
        dists = np.linalg.norm(wrist_sapien - center, axis=1)
        f_min = int(np.argmin(dists))
        print(f"   {name}: min_dist={dists.min():.4f}m @ F{f_min}, "
              f"wrist={wrist_sapien[f_min].round(4)}, obj={center.round(4)}, "
              f"diff={(wrist_sapien[f_min]-center).round(4)}")

    # 6. transform_params 内容
    print("\n6. transform_params.npz")
    params = np.load(str(transform_params_path), allow_pickle=True)
    for k in params.files:
        v = params[k]
        try:
            if np.isscalar(v) or v.size <= 16:
                print(f"   {k}: shape={v.shape}, value={v}")
            else:
                print(f"   {k}: shape={v.shape}")
        except Exception:
            print(f"   {k}: shape={getattr(v,'shape','?')}")

    # 7. 关键: pred_trans vs t_c2w (手在相机前方多少)
    print("\n7. 手腕 vs 相机 (HaWoR, 检查深度对齐)")
    n = min(len(pred_trans), len(t_c2w))
    for fi in [0, n//4, n//2, 3*n//4, n-1]:
        d = np.linalg.norm(pred_trans[fi] - t_c2w[fi])
        print(f"   F{fi}: pred_trans={pred_trans[fi].round(4)}, "
              f"t_c2w={t_c2w[fi].round(4)}, dist={d:.4f}m")

    # 8. 物体在 HaWoR 坐标 (反变换 SAPIEN→HaWoR)
    print("\n8. 物体 HaWoR 坐标 (反变换, 检查与手/相机的关系)")
    # SAPIEN→HaWoR: RXWORLD_TO_SAPIEN^T (正交矩阵, 逆=转置)
    SAPIEN_TO_HAWOR = RXWORLD_TO_SAPIEN.T
    for name, center in obj_bbox_centers.items():
        center_sapien = np.array(center)
        center_hawor = SAPIEN_TO_HAWOR @ center_sapien
        print(f"   {name}: sapien={center_sapien.round(4)}, hawor={center_hawor.round(4)}")

    print("\n" + "=" * 70)
    print("诊断完成")


if __name__ == "__main__":
    main()

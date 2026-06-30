"""逐帧分析 HaWoR npz + RAS GLB 两个文件应该如何结合

用户反馈: "你先分析一下输入的两个文件, 应该是怎么样的, 我感觉你好像根本没有触碰到正确的物体,
老是在弄那个盘子, 我认为你对两个文件的结合还没有做到位, 你得逐帧分析一下, 这两个文件应该怎么结合."

正确加载方式 (与 grasp_hawor.py 完全一致):
1. HaWoR npz: trajectory_loader.load_hawor_data() → pred_trans (HaWoR SLAM, z-forward, y-down, 米)
2. RAS GLB: final_scene.glb, 多个 geometry (RAS y-up, 米)
3. 变换链 (与 load_glb_with_physics L496-500 一致):
     p_hawor = s_inv * (R_inv @ p_ras.T).T + t_inv     # RAS → HaWoR SLAM
     p_sapien = RXWORLD_TO_SAPIEN @ p_hawor.T           # HaWoR SLAM → SAPIEN
4. 物体 bbox 中心 = (bbox_min + bbox_max) / 2 (SAPIEN 世界坐标)

逐帧分析: 左手腕 pred_trans[f] (HaWoR SLAM) vs 各物体中心
- 关键: pred_trans 在 HaWoR SLAM, 物体中心在 SAPIEN, 不能直接比!
- 正确做法: 物体中心反变换回 HaWoR SLAM (用 SAPIEN_TO_HAWOR = RXWORLD_TO_SAPIEN 的逆)
"""
import sys
import numpy as np
from pathlib import Path

# 添加模块路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# RXWORLD_TO_SAPIEN = R_AXIS @ R_x, R_AXIS=[[1,0,0],[0,0,1],[0,-1,0]], R_x=diag(1,-1,-1)
# = [[1,0,0],[0,0,-1],[0,1,0]]  (注意 [2,1]=+1)
# SAPIEN_TO_HAWOR = 逆 = [[1,0,0],[0,0,1],[0,-1,0]]  (正交矩阵, 逆=转置)
RXWORLD_TO_SAPIEN = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
SAPIEN_TO_HAWOR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

HAWOR_DIR = Path("/home/an/data/hawor/7")
RAS_DIR = Path("/home/an/data/ras/my_7mp4_result")
# 使用已有的 alignment 输出 (gripper_only_left 和 full_robot_left 应该相同)
TRANSFORM_PARAMS = Path("/home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics/output/gripper_only_left/alignment/transform_params.npz")


def load_hawor(hawor_dir):
    """加载 HaWoR npz (与 trajectory_loader.load_hawor_data 一致)"""
    from trajectory_loader import load_hawor_data, load_hawor_c2w
    print(f"\n=== 加载 HaWoR 数据 ===")
    print(f"目录: {hawor_dir}")
    # 左手 (hand_idx=0)
    left = load_hawor_data(hawor_dir, hand_idx=0)
    print(f"左手 pred_trans: {left['pred_trans'].shape}")
    print(f"左手 pred_valid: {left['pred_valid'].shape}, valid count: {left['pred_valid'].sum()}")
    # 右手 (hand_idx=1)
    try:
        right = load_hawor_data(hawor_dir, hand_idx=1)
        print(f"右手 pred_trans: {right['pred_trans'].shape}")
    except Exception as e:
        print(f"右手加载失败: {e}")
        right = None
    R_c2w, t_c2w = load_hawor_c2w(hawor_dir)
    print(f"R_c2w: {R_c2w.shape}, t_c2w: {t_c2w.shape}")
    return left, right, R_c2w, t_c2w


def load_objects(ras_dir, transform_params_path):
    """加载 RAS GLB 物体, 计算每个物体的 bbox 中心 (HaWoR SLAM + SAPIEN 两套坐标系)

    与 load_glb_with_physics (grasp_hawor.py L464-583) 完全一致的变换链.
    """
    import trimesh
    print(f"\n=== 加载 RAS GLB + transform_params ===")
    print(f"GLB: {ras_dir / 'final_scene.glb'}")
    print(f"transform_params: {transform_params_path}")

    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    print(f"  s_inv={s_inv:.6f}, R_inv shape={R_inv.shape}, t_inv={t_inv}")

    glb_path = ras_dir / "final_scene.glb"
    trimesh_scene = trimesh.load(str(glb_path))
    if not isinstance(trimesh_scene, trimesh.Scene):
        print(f"  WARNING: GLB 不是 Scene, 是 {type(trimesh_scene)}")
        trimesh_scene = trimesh.Scene({"object": trimesh_scene})

    geometries = list(trimesh_scene.geometry.items())
    print(f"  geometry 数量: {len(geometries)}")

    objects = {}
    for geom_idx, (geom_name, geom) in enumerate(geometries):
        verts_ras = np.asarray(geom.vertices, dtype=np.float64)
        if len(verts_ras) == 0:
            continue
        # 变换链 (与 load_glb_with_physics L499-500 一致)
        verts_hawor = s_inv * (R_inv @ verts_ras.T).T + t_inv
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T

        bbox_min_s = verts_sapien.min(axis=0)
        bbox_max_s = verts_sapien.max(axis=0)
        bbox_center_s = (bbox_min_s + bbox_max_s) / 2
        bbox_min_h = verts_hawor.min(axis=0)
        bbox_max_h = verts_hawor.max(axis=0)
        bbox_center_h = (bbox_min_h + bbox_max_h) / 2

        obj_name = f"glb_{geom_idx}"
        objects[obj_name] = {
            "name": geom_name,
            "center_sapien": bbox_center_s,
            "center_hawor": bbox_center_h,
            "bbox_min_sapien": bbox_min_s,
            "bbox_max_sapien": bbox_max_s,
            "size_sapien": bbox_max_s - bbox_min_s,
            "n_vertices": len(verts_ras),
        }
    print(f"\n  物体汇总:")
    print(f"  {'name':<15} | {'center_sapien':>30} | {'center_hawor':>30} | {'size':>20} | verts")
    for name, obj in objects.items():
        cs = obj["center_sapien"]
        ch = obj["center_hawor"]
        sz = obj["size_sapien"]
        print(f"  {name:<15} | [{cs[0]:>7.3f},{cs[1]:>7.3f},{cs[2]:>7.3f}] | "
              f"[{ch[0]:>7.3f},{ch[1]:>7.3f},{ch[2]:>7.3f}] | "
              f"[{sz[0]:>5.2f},{sz[1]:>5.2f},{sz[2]:>5.2f}] | {obj['n_vertices']}")
    return objects


def frame_by_frame(hawor_data, objects, side_name="left"):
    """逐帧分析手腕 vs 物体 (在 HaWoR SLAM 坐标系比较, 因为 pred_trans 在 HaWoR SLAM)

    关键: pred_trans[f] 是 HaWoR SLAM 坐标系的手腕位置
          物体 center_hawor 也是 HaWoR SLAM 坐标系
          所以直接比 wrist vs center_hawor 即可!
    """
    pred_trans = hawor_data["pred_trans"]
    pred_valid = hawor_data["pred_valid"]
    n_frames = len(pred_trans)
    print(f"\n{'='*80}")
    print(f"逐帧分析: {side_name}手腕 (HaWoR SLAM) vs 物体中心 (HaWoR SLAM)")
    print(f"{'='*80}")
    print(f"帧数: {n_frames}, valid 帧数: {pred_valid.sum()}")
    print(f"手腕轨迹范围: x=[{pred_trans[:,0].min():.3f},{pred_trans[:,0].max():.3f}] "
          f"y=[{pred_trans[:,1].min():.3f},{pred_trans[:,1].max():.3f}] "
          f"z=[{pred_trans[:,2].min():.3f},{pred_trans[:,2].max():.3f}]")

    obj_names = list(objects.keys())
    obj_pos_h = {n: objects[n]["center_hawor"] for n in obj_names}

    print(f"\n物体中心 (HaWoR SLAM):")
    for n in obj_names:
        print(f"  {n}: {obj_pos_h[n].round(4)}")

    # 逐帧距离
    close_frames = {n: 0 for n in obj_names}    # < 5cm
    near_frames = {n: 0 for n in obj_names}     # < 10cm
    avg_dists = {n: [] for n in obj_names}
    nearest_per_frame = []
    min_dist_per_frame = []

    print(f"\n每帧最近物体 (只打印关键帧, 即每 10 帧或 <5cm):")
    print(f"{'F':>4} | {'wrist_hawor':>32} | {'nearest':>8} | {'d(cm)':>6} | {'2nd':>8} | {'d(cm)':>6} | valid")
    for f in range(n_frames):
        wrist = pred_trans[f]
        valid = bool(pred_valid[f])
        dists = [(n, float(np.linalg.norm(wrist - obj_pos_h[n]))) for n in obj_names]
        dists.sort(key=lambda x: x[1])
        nn, nd = dists[0]
        sn, sd = dists[1]
        for n, d in dists:
            avg_dists[n].append(d)
        nearest_per_frame.append(nn)
        min_dist_per_frame.append(nd)
        if nd < 0.05:
            close_frames[nn] += 1
        if nd < 0.10:
            near_frames[nn] += 1
        if f % 10 == 0 or nd < 0.05:
            print(f"{f:>4} | [{wrist[0]:>7.3f},{wrist[1]:>7.3f},{wrist[2]:>7.3f}] | "
                  f"{nn:>8} | {nd*100:>6.2f} | {sn:>8} | {sd*100:>6.2f} | {int(valid)}")

    # 统计
    print(f"\n{'='*80}")
    print(f"统计 ({side_name} 手腕)")
    print(f"{'='*80}")
    print(f"{'obj':<15} | {'close<5cm':>10} | {'near<10cm':>10} | {'avg(cm)':>8} | {'min(cm)':>8}")
    for n in obj_names:
        ds = avg_dists[n]
        print(f"{n:<15} | {close_frames[n]:>10} | {near_frames[n]:>10} | "
              f"{np.mean(ds)*100:>8.2f} | {np.min(ds)*100:>8.2f}")

    # 找真正要抓的物体 (停留时间最长的)
    print(f"\n判定:")
    if max(close_frames.values()) > 0:
        target = max(close_frames, key=close_frames.get)
        print(f"  真正要抓的物体 (close<5cm 帧数最多): {target} ({close_frames[target]} 帧)")
    else:
        target = min(avg_dists, key=lambda k: np.mean(avg_dists[k]))
        print(f"  无停留 (close<5cm=0), 退回平均距离最近: {target}")

    # 最接近瞬间
    best_frame = int(np.argmin(min_dist_per_frame))
    print(f"  最接近瞬间: F{best_frame}, 物体={nearest_per_frame[best_frame]}, "
          f"距离={min_dist_per_frame[best_frame]*100:.2f}cm")

    # 连续接近区段
    print(f"\n连续接近区段 (≥5 帧 <10cm):")
    segments = []
    cur_obj = None
    cur_start = 0
    for f in range(n_frames):
        no = nearest_per_frame[f]
        d = min_dist_per_frame[f]
        if d < 0.10:
            if cur_obj != no:
                if cur_obj is not None and (f - cur_start) >= 5:
                    segments.append((cur_obj, cur_start, f - 1))
                cur_obj = no
                cur_start = f
        else:
            if cur_obj is not None and (f - cur_start) >= 5:
                segments.append((cur_obj, cur_start, f - 1))
            cur_obj = None
    if cur_obj is not None and (n_frames - cur_start) >= 5:
        segments.append((cur_obj, cur_start, n_frames - 1))
    for obj, s, e in segments:
        print(f"  {obj}: F{s}-F{e} ({e-s+1} 帧)")

    # 找 F0 最近物体 (对比当前逻辑)
    print(f"\nF0 vs 真正要抓的物体对比 (验证之前的 bug):")
    f0_nearest = nearest_per_frame[0]
    f0_dist = min_dist_per_frame[0]
    print(f"  F0 最近: {f0_nearest} (距离 {f0_dist*100:.2f} cm)")
    print(f"  真正要抓: {target}")
    if f0_nearest != target:
        print(f"  ⚠ F0 误判! 这就是'老在弄那个盘子'的根因")
    else:
        print(f"  ✓ F0 与真正目标一致")

    return target, segments


def analyze_sapien_coords(hawor_data, objects, side_name="left"):
    """额外分析: 在 SAPIEN 坐标系比较 (验证 grasp_hawor.py 的逻辑)

    grasp_hawor.py 中:
    - obj_bbox_centers 是 SAPIEN 坐标
    - find_target_object_by_trajectory 把物体从 SAPIEN 反变换到 HaWoR SLAM 比较
    - 这里验证反变换是否正确
    """
    pred_trans = hawor_data["pred_trans"]
    print(f"\n{'='*80}")
    print(f"验证 SAPIEN ↔ HaWoR 反变换 ({side_name})")
    print(f"{'='*80}")
    # grasp_hawor.py L1268-1270:
    #   R_x_inv = diag(1, -1, -1)
    #   R_AXIS_inv = [[1,0,0],[0,0,-1],[0,1,0]]
    #   SAPIEN_TO_HAWOR = R_x_inv @ R_AXIS_inv
    R_x_inv = np.diag([1.0, -1.0, -1.0])
    R_AXIS_inv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    SAPIEN_TO_HAWOR_grasp = R_x_inv @ R_AXIS_inv
    print(f"grasp_hawor.py 用的 SAPIEN_TO_HAWOR:")
    print(f"  {SAPIEN_TO_HAWOR_grasp}")
    print(f"正确的 SAPIEN_TO_HAWOR (RXWORLD_TO_SAPIEN 的逆):")
    print(f"  {SAPIEN_TO_HAWOR}")
    if np.allclose(SAPIEN_TO_HAWOR_grasp, SAPIEN_TO_HAWOR):
        print(f"  ✓ 一致")
    else:
        print(f"  ✗ 不一致! 这可能是 grasp_hawor.py 找错物体的根因!")

    # 验证: 用 grasp_hawor.py 的反变换算物体 HaWoR 位置, 与正确的 center_hawor 对比
    print(f"\n物体中心反变换对比 (正确 vs grasp_hawor.py 用的):")
    print(f"{'obj':<15} | {'正确 center_hawor':>25} | {'grasp_hawor 反变换':>25} | 一致?")
    for name, obj in objects.items():
        cs = obj["center_sapien"]
        ch_correct = obj["center_hawor"]
        ch_grasp = SAPIEN_TO_HAWOR_grasp @ cs
        match = np.allclose(ch_correct, ch_grasp, atol=1e-4)
        print(f"{name:<15} | [{ch_correct[0]:>6.3f},{ch_correct[1]:>6.3f},{ch_correct[2]:>6.3f}] | "
              f"[{ch_grasp[0]:>6.3f},{ch_grasp[1]:>6.3f},{ch_grasp[2]:>6.3f}] | {'✓' if match else '✗'}")


def main():
    print("=" * 80)
    print("HaWoR npz + RAS GLB 逐帧结合分析 (正确加载)")
    print("=" * 80)
    left, right, R_c2w, t_c2w = load_hawor(HAWOR_DIR)
    objects = load_objects(RAS_DIR, TRANSFORM_PARAMS)
    print(f"\n加载了 {len(objects)} 个物体")

    # 验证坐标系反变换
    analyze_sapien_coords(left, objects, "left")

    # 左手逐帧分析
    target_left, segs_left = frame_by_frame(left, objects, "left")

    # 右手逐帧分析
    if right is not None:
        target_right, segs_right = frame_by_frame(right, objects, "right")

    print(f"\n{'='*80}")
    print(f"最终结论")
    print(f"{'='*80}")
    print(f"  左手真正要抓的物体: {target_left}")
    if right is not None:
        print(f"  右手真正要抓的物体: {target_right}")


if __name__ == "__main__":
    main()

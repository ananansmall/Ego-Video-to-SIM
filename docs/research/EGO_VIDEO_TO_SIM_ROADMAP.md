# Ego-Video-to-Sim 研究路线图 v3.1（解法导向，2026最新论文版）

> **项目**：HaWoR + ReplicateAnyScene + GalaxeaManipSim 合并
> **官方整合仓库**：[Ego-Video-to-SIM](https://github.com/ananansmall/Ego-Video-to-SIM)（HaWoR + RAS + SAPIEN + GalaxeaManipSim + dex-retargeting 一键整合）
> **目标平台**：Galaxea R1 / R1 Pro / R1 Lite
> **核心约束**：
> 1. **不写 IK、不做解析运动学**——直接用 GalaxeaManipSim 的 `bimanual_joint_position` / `bimanual_relaxed_ik`
> 2. **分割检测统一用 SAM 3**（Meta 2025-11，v2 2026-03，PCS 任务，4M 概念零样本）
> 3. **论文以 2025-2026 为主**，优先用已经在 CVPR/ICLR/ICRA 2026 落地的论文
> 4. **只写"怎么解"，不写"工具是什么"**——读者已经熟悉基础工具
>
> **本文档结构**：先给"最终数据集长什么样"（终极目标），再给 5 个问题的解法，最后给实施路径
>
> **配套详细论文解读**：[REFERENCED_PAPERS_DETAIL.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/REFERENCED_PAPERS_DETAIL.md)（10 篇核心论文，每篇 100+ 行深度解读）

---

## 0. 最终数据集长什么样（先看目标，再看解法）

> 这是你"5 个问题"的最终交付物形态——所有问题都围绕"如何把视频变成这个数据集"展开。

### 0.1 推荐格式：LeRobot v3 兼容 + GalaxeaManipSim 兼容 + 自定义 sidecar

**为什么不选其他格式**：
- ❌ **ROS2 bag**：体积大、序列化慢、训练侧还要再转
- ❌ **RLDS / TFDS**：Google 系，VLA 训练友好但 Galaxea 没现成支持
- ❌ **HDF5 + 自定义 schema**：和 LeRobot/Galaxea 生态脱节
- ✅ **LeRobot v3 + sidecar JSON**：

  - LeRobot 已经是 VLA / DP / π₀ / OpenVLA 训练的事实标准
  - GalaxeaManipSim 自带 `convert_*_to_lerobot` 工具，h5 → LeRobot 一键转
  - 用 `episode.meta.json` sidecar 挂你自己的中间结果，可读、可改、可升级

### 0.2 目录结构（一个视频对应一个 episode）

```
your_dataset/
├── meta/
│   ├── info.json                  ← 全局信息
│   ├── episodes.jsonl             ← 1 行 = 1 个 episode
│   ├── tasks.jsonl                ← 任务描述
│   └── stats.safetensors          ← 归一化用
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet       ← 主流水线要求的格式
│       └── episode_000000.meta.json     ← 你独有的中间结果
├── videos/
│   └── chunk-000/
│       └── observation.images.head_rgb/
│           └── episode_000000.mp4
└── custom_scenes/                 ← 你独有的"原始视频 + 3DGS + GLB"
    ├── episode_000000/
    │   ├── source.mp4
    │   ├── scene_3dgs.ply
    │   ├── objects/*.glb
    │   └── intermediate/
    │       ├── hawor_wrist.npy
    │       ├── hawor_mano.npy
    │       ├── vggt_pointmap.npy
    │       ├── sam3_masks/*.png
    │       ├── foundation_pose_T.npy
    │       └── ...
    └── episode_000001/
        └── ...
```

### 0.3 `info.json` 完整 schema

```json
{
  "codebase_version": "v3.1",
  "robot_type": "galaxea_r1_lite",
  "fps": 30,
  "total_episodes": 100,
  "total_frames": 30000,
  "splits": {"train": "0:80", "val": "80:90", "test": "90:100"},
  "features": {
    "observation.images.head_rgb":        {"shape": [224, 224, 3], "dtype": "video"},
    "observation.images.left_wrist_rgb":  {"shape": [224, 224, 3], "dtype": "video"},
    "observation.images.right_wrist_rgb": {"shape": [224, 224, 3], "dtype": "video"},
    "observation.depth.head_depth":       {"shape": [224, 224], "dtype": "image"},
    "observation.state":  {"shape": [16], "dtype": "float32"},
    "action":             {"shape": [16], "dtype": "float32"}
  },
  "task_names": ["pick_cup", "pour_water", "stack_block"]
}
```

### 0.4 `observation.state` / `action` 维度的两种选择

| 控制器 | 维度 | 内容 | 适用场景 |
|--------|------|------|---------|
| `bimanual_joint_position` | 16 (R1/R1Lite) / 18 (R1Pro) | `left_arm_joints(6/7) + left_gripper(1) + right_arm_joints(6/7) + right_gripper(1) + base(0)` | 关节空间 DP/BC；R1 Lite 唯一选项 |
| `bimanual_relaxed_ik` | 16 (R1/R1Lite) / 18 (R1Pro) | `left_ee_pose(7) + left_gripper(1) + right_ee_pose(7) + right_gripper(1) + base(0)` | EEF 空间；可被 VLA 直接消费 |

> ⚠️ `base_pose` 维度取决于是否把底座也作为 action 的一部分。R1 系列是移动底座（4 轮/3 轮），vx/vy/ω 加 3 维。共 19 (R1 Pro) 或 17 (R1/R1 Lite)。

### 0.5 `episode.meta.json` 完整 schema（你的"中间结果"全部在这里）

```json
{
  "schema_version": "3.1",

  "video_source": {
    "path": "/mnt/data_8THDD/lza/workspace/data/videos/beizi.mp4",
    "fps": 30,
    "duration_s": 12.5,
    "resolution": [1920, 1080],
    "camera_intrinsics_K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },

  "scene_reconstruction": {
    "method": "replicate_any_scene_vggt4d",
    "scene_3dgs_path": "scene.ply",
    "pointmap_path": "pointmap.npy",
    "confidence": 0.87,
    "sparse_pcd_path": "sparse_pcd.ply"
  },

  "segmentation": {
    "method": "sam3_pcs",
    "text_prompts": ["red cup", "table", "left hand"],
    "object_masks_dir": "sam3_masks/",
    "object_ids": {"red cup": 1, "table": 2, "left hand": 3}
  },

  "object_6d_poses": {
    "method": "foundation_pose_render_compare",
    "objects": {
      "1": {
        "name": "red cup",
        "glb_path": "objects/cup.glb",
        "T_world": "T_cup.npy",
        "T_init": "T_cup_init.npy",
        "alignment_error_m": 0.012,
        "psnr": 27.4
      }
    }
  },

  "hand_retargeting": {
    "method": "warp_style_mano_to_ee",
    "mano_wrist_traj": "mano_wrist.npy",
    "mano_joints_21xTx3": "mano_joints.npy",
    "T_ee_left_4x4xT": "T_ee_left.npy",
    "T_ee_right_4x4xT": "T_ee_right.npy",
    "gripper_width_left_1xT": "gw_left.npy",
    "gripper_width_right_1xT": "gw_right.npy",
    "controller": "bimanual_relaxed_ik"
  },

  "base_placement": {
    "method": "mobi_pi_bayes_opt",
    "base_xyt": [0.42, -0.15, 1.57],
    "search_iterations": 120,
    "score_breakdown": {"D_in": -0.12, "V_obj": 0.93, "C_free": 0.85}
  },

  "task_semantics": {
    "task_name": "pick_cup",
    "action_type": "pick",
    "contact_object": "red cup",
    "contact_start_frame": 45,
    "contact_end_frame": 78,
    "affordance_hint": "grasp cup handle",
    "vlm_model": "gemini-2.5-flash",
    "vlm_prompt": "What is the hand doing? Which object is the contact?",
    "vlm_response": "The hand is reaching for the red cup's handle to lift it."
  },

  "physics_params": {
    "method": "vlm_material_lookup",
    "objects": {
      "red cup": {"mass_kg": 0.3, "friction": 0.5, "material": "ceramic"}
    }
  },

  "primitive_sequence": [
    {"type": "reach",     "start": 0,   "end": 40,  "target_T_ee_left": "[...]"},
    {"type": "grasp",     "start": 45,  "end": 50,  "gripper_width": 0.04},
    {"type": "manipulate","start": 50,  "end": 75,  "motion": "lift"},
    {"type": "release",   "start": 78,  "end": 80,  "gripper_width": 0.08},
    {"type": "return",    "start": 80,  "end": 110, "home_T_ee_left": "[...]"}
  ],

  "consistency_check": {
    "method": "robopaint_render_compare",
    "render_psnr": 27.4,
    "render_lpips": 0.18,
    "trajectory_rmse_m": 0.022,
    "physics_violations": 0
  },

  "augmentation": {
    "method": "robowheel_helical",
    "num_variants": 50,
    "object_swap": true,
    "background_swap": true,
    "trajectory_mirror": true
  }
}
```

### 0.6 可定制表（用户最常问的"可以自定义吗"）

| 想要什么 | 怎么自定义 | 在哪里改 |
|---------|----------|---------|
| 加新传感器（腕力、触觉、IMU） | 在 `info.json` 的 `features` 加 `observation.tactile.*` / `observation.wrench.*`，schema 自动加 | `meta/info.json` |
| 换机器人（Franka / Aloha / UR5e） | 改 `robot_type` 字段 + 写对应 URDF 转换器（仿 `convert_single_galaxea_sim_to_lerobot` 写一个） | `meta/info.json` + `galaxea_sim/scripts/` |
| 改坐标系（z-up ↔ y-up） | 在 `episode.meta.json` 加 `coordinate_frames` 字段；转换时在 `coordinate_and_alignment.py` 一次性做 | `episode.meta.json` + `src/geometry_utils.py` |
| 加多相机视角 | 在 `features` 加 `observation.images.<cam_name>`；和 head_rgb 一致 | `meta/info.json` |
| 加 demo 成功/失败标签 | 加 `episode.success: bool` 字段，写在 parquet 与 sidecar | `episode_XXXXXX.parquet` + `.meta.json` |
| 加 base 真实位姿 | 加 `observation.base_pose: List[float]` (3,) | schema + 写入逻辑 |
| 接 π₀ / OpenVLA | 改 `action` 维度匹配 VLA 训练接口（OpenVLA 用 7-DoF） | `meta/info.json` |
| 接 VLM CoT 训练 | 在 sidecar 加 `vlm_chain_of_thought: List[str]` | `.meta.json` |
| 加长视频任务分段 | 在 `primitive_sequence` 加 segment 字段 | `.meta.json` |
| 加多任务版本 | `tasks.jsonl` 加 task_index 描述；同一 episode 可被多个 task 复用 | `meta/tasks.jsonl` |
| 加 3DGS / 神经场 | 在 `scene_reconstruction` 加 `nerf_path` / `3dgs_path` 字段 | `.meta.json` |
| 加 Grounding / affordance 标注 | 加 `affordance_heatmap: np.ndarray (H, W, K)` 字段 | `.meta.json` |
| 加失败 case 详细原因 | 加 `failure_reason: str` + `failure_frame: int` 字段 | `.meta.json` |
| 关联原始 HOI 视频（如 HOI4D、Epic-Kitchens） | 加 `source_hoi_dataset: str` 字段 | `.meta.json` |
| 多步操作（pick + place） | 多个 `primitive_sequence` entry | `.meta.json` |
| 切换关节/EEF 控制 | 同 episode 共存 `state_joints` + `state_eef` | `info.json` |
| 适配 R1 vs R1 Pro vs R1 Lite | `arm_dof=6/7` 自动调整 | `info.json` + 转换器 |

### 0.7 完整转换流程（视频 → 这个数据集的端到端 pipeline）

```python
# pipeline.py —— 一键视频 → LeRobot 数据集
def video_to_dataset(video_path, output_dir, robot_type="r1_lite"):
    # Step 1: 重建场景 (ReplicateAnyScene + VGGT4D)
    scene_dir = run_replicate_any_scene(video_path, output_dir / "recon")
    
    # Step 2: 分割所有相关物体 (SAM 3 PCS)
    masks = run_sam3_pcs(scene_dir, prompts=["red cup", "table", "left hand", ...])
    
    # Step 3: 跟踪物体 6D pose (FoundationPose Render&Compare, 直接对视频像素)
    obj_poses = run_foundation_pose(scene_dir, masks)  # ← 不对齐 VGGT，对齐视频
    
    # Step 4: 手部 6D (HaWoR)
    hand_poses = run_hawor(video_path)
    
    # Step 5: 物体轨迹预测 (ObjectForesight) - 用于预判碰撞
    future_obj = predict_object_foresight(obj_poses)
    
    # Step 6: 任务语义 (VLM)
    task = vlm_infer_task(video_path, masks, hand_poses)
    
    # Step 7: 物理参数 (VLM 材质查表)
    physics = vlm_infer_physics(masks)
    
    # Step 8: 动作原语切分 (reach/grasp/manipulate/release/return)
    primitives = segment_primitives(hand_poses, obj_poses)
    
    # Step 9: 底座搜索 (Mobi-π BO)
    base_xyt = mobi_pi_base_search(scene_dir, hand_poses, video_path)
    
    # Step 10: MANO → Galaxea EE (WARPED 风格) → 控制器接口
    T_ee_seq, gw_seq = warp_style_retarget(hand_poses)
    actions = to_galaxea_actions(T_ee_seq, gw_seq, primitives, base_xyt)
    
    # Step 11: 在 GalaxeaManipSim 跑物理仿真
    env = GalaxeaSimEnv(robot=robot_type, env_name=build_env_from_video(scene_dir, task))
    env.set_base_pose(*base_xyt)
    for a in actions:
        env.step(a)
    env.export_demo(f"{output_dir}/raw/demo.h5")
    
    # Step 12: 转 LeRobot (用 GalaxeaManipSim 自带工具)
    convert_to_lerobot(
        h5_path=f"{output_dir}/raw/demo.h5",
        out_dir=output_dir,
        robot=robot_type,
        use_eef=True,
    )
    
    # Step 13: 写 sidecar meta
    write_sidecar_meta(output_dir, episode_id, {
        "video_source": ...,
        "scene_reconstruction": ...,
        "object_6d_poses": ...,
        "hand_retargeting": ...,
        "base_placement": ...,
        "task_semantics": ...,
        "physics_params": ...,
        "primitive_sequence": ...,
    })
    
    return output_dir
```

---

## 1. 问题 1：机械臂+底座如何与视频映射

### 1.1 核心矛盾与解法选择

| 矛盾点 | 旧路线（失败） | 新路线（推荐） |
|--------|---------------|---------------|
| 人手→机器人 EE 维度差距大 | 自己写解析 IK | **WARPED** 风格 wrist→EE 模板 + Galaxea `bimanual_relaxed_ik` 接管关节解 |
| 底座位置没有真值 | 随机初始 | **Mobi-π** 风格贝叶斯优化 + DINOv2 in-distribution 评分 |
| 抓取是否物理可行 | 忽略 | **SynManDex** 风格 force-closure 验证 + Galaxea `check_object_lift` |

### 1.2 MANO → Galaxea EE 解法：WARPED 风格 retargeting

**首选论文**：**WARPED**（CMU 2026-04，arXiv:2604.10809）

WARPED 的核心 trick：**从 wrist camera 视角**而非第三人称视角做 retargeting——把 wrist 6DoF 当虚拟 EE，让 retargeted 机器人在 wrist 视角下与人手视觉对齐。

**对你项目 (R1/R1Pro/R1Lite) 的适配**：
- R1 头部有 RGB-D，wrist 也有 RGB——**两个视角都有**
- 直接套用 WARPED 的 wrist→EE 映射公式，**不写解析 IK**
- Wrist→EE 公式（WARPED Section III-E，论文 R1 适配版）：

```python
# warp_style_retarget.py —— 不写 IK，只算 EE 6DoF
def warp_style_retarget(mano_wrist_t, mano_joints_t, gal_urdf):
    """
    输入: MANO 21 关节点 + wrist 6DoF (来自 HaWoR)
    输出: Galaxea EE 6DoF + 夹爪宽度 (给 bimanual_relaxed_ik 控制器)
    """
    # WARPED 公式: 食指指尖 + 拇指指尖中点 → gripper TCP
    index_tip = mano_joints_t[:, 8]    # MANO index fingertip
    thumb_tip = mano_joints_t[:, 4]    # MANO thumb tip
    tcp_pos = (index_tip + thumb_tip) / 2.0  # (T, 3)
    
    # 食指 MCP → index base, 用于推算 gripper z 轴方向
    index_base = mano_joints_t[:, 5]   # MANO index MCP
    
    # EE z 轴 = 食指方向 (从 MCP 到指尖)
    ee_z = (index_tip - index_base)
    ee_z /= (np.linalg.norm(ee_z, axis=-1, keepdims=True) + 1e-6)
    
    # 右手系: x = world_up × z, y = z × x
    up = np.array([0, 0, 1])
    ee_x = np.cross(up, ee_z)
    ee_x /= (np.linalg.norm(ee_x, axis=-1, keepdims=True) + 1e-6)
    ee_y = np.cross(ee_z, ee_x)
    
    T_ee = np.eye(4)[None].repeat(len(mano_wrist_t), axis=0)
    T_ee[:, :3, 3] = tcp_pos
    T_ee[:, :3, 0] = ee_x
    T_ee[:, :3, 1] = ee_y
    T_ee[:, :3, 2] = ee_z
    
    # 夹爪宽度 = 食指与拇指指尖距离
    gripper_width = np.linalg.norm(index_tip - thumb_tip, axis=-1)
    
    return T_ee, gripper_width
```

**后续交给 Galaxea `bimanual_relaxed_ik`**（不写 IK，URDF 内置 relaxed IK 解算器自动处理）：

```python
# to_galaxea_actions.py —— WARPED 输出 → Galaxea 控制器接口
def to_galaxea_actions(T_ee_left, T_ee_right, gw_left, gw_right, primitives, base_xyt):
    actions = []
    for prim in primitives:
        action = {
            "left_arm":  {"ee_pose":  pose7d(T_ee_left[prim.end_idx])},  # xyzwxyz
            "right_arm": {"ee_pose":  pose7d(T_ee_right[prim.end_idx])},
            "left_gripper":  float(gw_left[prim.end_idx]),
            "right_gripper": float(gw_right[prim.end_idx]),
            "base": [base_xyt[0], base_xyt[1], base_xyt[2]],  # 一次性设置
        }
        actions.append(action)
    return actions
```

### 1.3 物理验证：SynManDex 思路

**论文**：**SynManDex**（arXiv:2606.09798，2026-06）——用 force-closure 优化接触点。

**对你的简化**（R1 是二指夹爪，不是灵巧手）：
- 不需要 pre-grasp 灵巧手模板
- 但 WARPED 输出 EE pose 后，**必须在 GalaxeaManipSim 物理仿真里验证一遍**：
  - 夹爪能不能合上（不撞物体、不卡死）
  - 物体能不能被提起（不滑落、不穿模）
  - 关节角是否超限（relaxed_ik 不可达时自动 fallback 到 joint controller）

```python
# validate_grasp.py —— Galaxea 自带物理引擎
def validate_grasp(env_id, grasp_T, obj_id, robot_id):
    env = GalaxeaSimEnv(env_id)
    env.set_ee_target(grasp_T)
    env.close_gripper(width=0.04)  # R1 二指夹爪最大宽度
    success = env.check_object_lift(obj_id)
    env.close()
    return success
```

### 1.4 底座位置搜索：Mobi-π 风格贝叶斯优化

**首选论文**：**Mobi-π**（Stanford + TRI, CoRL 2025, arXiv:2505.23692）

**核心思想**：在 3DGS 重建的场景中，搜索 (x, y, θ) 让机器人相机视角最接近"训练分布"。

```python
# mobi_pi_base_search.py —— 贝叶斯优化 + 混合评分
def mobi_pi_score(env_3dgs, video_frames, candidate_base_pose):
    x, y, theta = candidate_base_pose
    rendered = render_3dgs_from_base(env_3dgs, base=(x, y, theta), cam_height=1.4)
    
    # In-distribution score: DINOv2 特征 Chamfer 距离
    f_render = dino_v2_features(rendered)
    f_video = dino_v2_features(video_frames)
    D_in = -chamfer_distance(f_render, f_video)
    
    # 物体可见性 (VLM 检查)
    V_obj = vlm_check_visibility(rendered, hawor_obj_name)
    
    # 碰撞检查
    C_free = 1.0 - check_collision(gal_r1_urdf, base=(x, y, theta), env_3dgs)
    
    return 0.5 * D_in + 0.3 * V_obj + 0.2 * C_free

# 100-200 次贝叶斯优化 (GP + UCB)
best_pose = bayesian_optimize(mobi_pi_score, 
                              bounds=((-2, 2), (-2, 2), (-np.pi, np.pi)),
                              n_iter=150)
```

**2026 强化版**：**Category-level Last-meter Navigation**（UMN 2025-12, arXiv:2512.11173）——用 IL 学一个 category-level 导航策略，比 BO 更快。

### 1.5 端到端骨架代码

```python
# arm_base_mapping.py —— 主入口
def map_video_to_robot(video_path, hambo_dir, scene_3dgs, robot_type="r1_lite"):
    # 1. 加载 HaWoR 输出
    mano_wrist = np.load(f"{hambo_dir}/wrist.npy")  # (T, 4, 4)
    mano_joints = np.load(f"{hambo_dir}/joints.npy") # (T, 21, 3)
    
    # 2. WARPED 风格 retargeting
    T_ee_left, gw_left = warp_style_retarget(mano_wrist, mano_joints, "left")
    T_ee_right, gw_right = warp_style_retarget(mano_wrist, mano_joints, "right")
    
    # 3. 切分原语
    primitives = segment_primitives(T_ee_left, T_ee_right, gw_left, gw_right, 
                                    object_trajs, video_path)
    
    # 4. 底座搜索
    base_xyt = mobi_pi_base_search(scene_3dgs, video_path, mano_wrist)
    
    # 5. 转 Galaxea 控制器接口
    actions = to_galaxea_actions(T_ee_left, T_ee_right, gw_left, gw_right, 
                                  primitives, base_xyt)
    return actions, base_xyt
```

### 1.6 借鉴 [Ego-Video-to-SIM](https://github.com/ananansmall/Ego-Video-to-SIM) 仓库

该官方整合仓库已经做了：
- `pv_retargeting/`：用 **dex-retargeting** 库（dexsuite 出品，6 年迭代成熟库）做 MANO→夹爪映射
- `libs/galaxea_sim/`：GalaxeaManipSim 内部调用 `bimanual_relaxed_ik`

**直接复用 `libs/dex_retargeting`**：你不需要自己写 WARPED，可以直接调用 `dex_retargeting.retarget(mano, robot)`。

---

## 2. 问题 2：GLB / 视频 / HaWoR 三方对齐（stage4 重点改造）

### 2.1 你的痛点

> 当前 stage4 用 3D Lifting + Umeyama 把 GLB 对齐到 VGGT 点云，但 z 方向误差大、与视频不直接对应。

**为什么这是错的**：
- VGGT 是 monocular depth，遮挡区深度不可靠（墙前的画深度是画表面，不是墙）
- 目标应该是"对齐到视频 mask 区域"，不是"对齐到 VGGT 点云"
- 你的下游任务是"机器人执行同视频一致的动作"，坐标系源头必须是**视频**

### 2.2 正确解法：Render & Compare 路线

**首选**：**FoundationPose**（NVIDIA, CVPR 2024）+ **MegaPose**（NVIDIA, RA-L 2023）

Render&Compare 范式：用 GLB 在候选 pose 下渲染，与真实图像做 SSIM/LPIPS loss，梯度下降精化。

**为什么选它**：
- 工业级 6D pose 跟踪金标准（FoundationPose BOP-Classic-Core AP=0.78）
- model-based 模式直接吃你的 GLB
- Render&Compare 在像素级工作，**天然与视频对齐**

**OpenReview/CVPR 2026 最新进展**：

| 论文 | 优势 | 链接 |
|------|------|------|
| **OMNI-PoseX** (Apr 2026, arXiv:2604.02759) | SO(3) flow matching，open-world 实时 | [arXiv](https://arxiv.org/html/2604.02759v1) |
| **EgoXtreme** (CVPR 2026) | 第一视角 6D pose benchmark（运动模糊/烟雾/低光） | [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yoon_EgoXtreme_A_Dataset_for_Robust_Object_Pose_Estimation_in_Egocentric_CVPR_2026_paper.pdf) |
| **M-VTOP** (ICRA 2026, MERL) | 视觉+触觉融合，亚毫米精度 | [TR2026-070](https://merl.com/publications/docs/TR2026-070.pdf) |

### 2.3 完整算法（直接改造你的 stage4）

```python
# stage4_align_v3.py —— Render & Compare + Pose Graph Optimization
def stage4_align_v3(glb_model, video_frames, sam3_masks, vggt_K, init_pose):
    """
    输入: GLB 物体 + 视频帧 + SAM 3 mask + VGGT 出的 K
    输出: 每帧物体 6D pose (T, 4, 4)
    """
    # 1. SAM 3 mask 提供 ROI（在视频像素级，与视频直接对齐）
    # 2. 相机内参 (从 VGGT)
    K = vggt_K  # (3, 3)
    
    # 3. FoundationPose 初始化 (第一帧)
    rendered = render_glb(glb_model, init_pose, K)
    score, refined_T_0 = foundation_pose_refine(
        rendered, video_frames[0], sam3_masks[0], glb_model
    )
    
    # 4. 帧间跟踪: PnP + FoundationPose 精化
    poses = [refined_T_0]
    for s in range(1, len(video_frames)):
        # 用 VGGT TrackHead 跟踪 2D（你 stage2 已有）
        tracks_2d = vggt_trackhead(query=project_points(sample_3d, refined_T, K),
                                    frames=video_frames[:s+1])
        # PnP 给初值（粗对齐）
        coarse_T = solve_pnp(sample_3d, tracks_2d[-1], K)
        # FoundationPose 精化（细对齐，渲染对比）
        _, refined_T_s = foundation_pose_refine(
            render_glb(glb_model, coarse_T, K), 
            video_frames[s], sam3_masks[s], glb_model
        )
        poses.append(refined_T_s)
    
    # 5. Pose Graph Optimization (GTR 风格) 全局精化
    poses_opt = pose_graph_optimize(
        poses, video_frames, sam3_masks, glb_model
    )
    return poses_opt
```

**关键创新点**：
- **不再对齐 VGGT 点云**——对齐的是视频 mask 区域
- **PnP 给粗值 + FoundationPose 精化**——比单 PnP 鲁棒
- **Pose Graph Optimization** 全局一致性（GTR, Woven by Toyota 2025-05, arXiv:2505.11905）

### 2.4 Pose Graph Optimization 数学

```python
def pose_graph_optimize(poses, video_frames, masks, glb_model):
    """
    节点: 每帧物体 pose
    边:
      - unary: 单帧 Render&Compare 残差 (LPIPS + SSIM)
      - binary: 帧间光流残差 (T_obj[s] 和 T_obj[s+1] 之间的点运动应与光流一致)
    """
    # G2O / GTSAM 实现
    graph = gtsam.NonlinearFactorGraph()
    
    for s in range(len(poses)):
        # Unary: 渲染 vs 真实
        rendered = render_glb(glb_model, poses[s], K)
        residual = lpips_ssim_loss(rendered, video_frames[s], masks[s])
        graph.add(unary_factor(poses[s], residual))
        
        if s > 0:
            # Binary: 光流约束
            flow = compute_optical_flow(video_frames[s-1], video_frames[s])
            flow_residual = optical_flow_factor(poses[s-1], poses[s], flow)
            graph.add(binary_factor(poses[s-1], poses[s], flow_residual))
    
    # 优化
    initial = gtsam.Values()
    for s, p in enumerate(poses):
        initial.insert(s, gtsam.Pose3(p))
    result = gtsam.LevenbergMarquardtOptimizer().optimize(graph, initial)
    
    return [result.at(s).matrix() for s in range(len(poses))]
```

### 2.5 备选：GTR（3DGS-based Pose Tracking）

如果你的 GLB 模型不准确，用 3DGS 重建的物体模型做跟踪：

- **GTR**（Woven by Toyota + Toyota Research, 2025-05, arXiv:2505.11905）
- **GSGTrack**（Tongji 2024-12, arXiv:2412.02267）

这些方法**不需要 GLB**——直接用 3DGS 表示物体。

### 2.6 验证指标

| 指标 | 当前 | 目标 | 测量方法 |
|------|------|------|---------|
| Acc@10% (BOP) | 0.5 | ≥ 0.9 | 在 BOP-style test 上 |
| Z 方向误差 | 5cm | < 1cm | 已知物体尺寸做尺度验证 |
| 视频 PSNR (mask 内) | 22dB | ≥ 27dB | 渲染 vs 真实 mask 内 PSNR |
| 时间 jitter | 抖动 | < 2cm/帧 | 帧间 6DoF 差 |

---

## 3. 问题 3：让机器人知道在做什么（任务语义 + 抓取动作）

### 3.1 缺什么

| 缺口 | 表现 | 影响 |
|------|------|------|
| 缺任务语义 | 只知道手腕轨迹，不知道"抓杯子"还是"倒水" | 仿真器无法选择 task spec |
| 缺 affordance | 不知道抓哪个部位 | 抓取成功率 < 30% |
| 缺抓取姿态合成 | 没有 6-DoF gripper pose | 仿真器无法执行 |
| 缺动作原语切分 | 连续轨迹无法被 interpret | replay 不稳定 |

### 3.2 解法一：SAM 3 统一替代 Grounding DINO + SAM 2

**首选**：**SAM 3**（Meta 2025-11，v2 2026-03，arXiv:2511.16719）

**核心**：**Promptable Concept Segmentation (PCS)**——文本提示 "red cup" / 图像示例 / 点击，三选一，模型一次输出**所有匹配概念**的实例 mask + 唯一 ID。

**关键论文事实**（v2 2026-03）：
- 4M 唯一概念训练集 SA-Co
- 图像 PCS 比 SAM 2 提升 2 倍精度
- 视频 PCS 首次实现 75-80% 人类水平

```python
# sam3_pcs.py —— 替换 Grounding DINO + SAM 2
def segment_objects_with_sam3(video_frames, object_prompts):
    """
    object_prompts: ["red cup", "table", "left hand", ...]
    return: per-object masks per frame + consistent ID
    """
    sam3 = load_sam3()  # pip install sam3, https://github.com/facebookresearch/sam3
    predictor = sam3.VideoPredictor()
    with predictor.init_state(video=video_frames) as state:
        for prompt in object_prompts:
            # 文本 prompt -> SAM 3 自动出所有匹配实例
            masks = sam3.predict_with_text(prompt, frame_idx=0)
            for m in masks:
                predictor.add_new_mask(state, frame_idx=0, mask=m)
        # 在整个视频上传播
        masks_video = {}
        for frame_idx, obj_ids, masks in predictor.propagate(state):
            for oid, m in zip(obj_ids, masks):
                masks_video.setdefault(oid, []).append((frame_idx, m))
    return masks_video
```

| 原方案 | 新方案（SAM 3） |
|--------|---------------|
| Grounding DINO 检测框 → SAM 2 分割 | 直接 SAM 3 + 文本 prompt |
| 多物体需要多个 prompt | 一个 prompt 出所有同类物体 |
| 视频跟踪需要 SAM 2 单独做 | SAM 3 内置视频跟踪器 |
| 概念级泛化（"red cup"）需要 fine-tune | 零样本支持 4M+ 概念 |

### 3.3 解法二：任务语义（VLM 推断）

**首选 VLM**：Gemini 2.5 Flash / GPT-5 / Claude 4.5

```python
# task_inference.py —— VLM 推断任务
def vlm_infer_task(video_frames, sam3_masks, hawor_poses):
    prompt = """
    Given these video frames and segmentation masks:
    1. What is the user doing? (one of: pick, place, pour, open, close, push, pull, stack, mix, wipe, fold)
    2. Which object is being manipulated?
    3. What is the contact point on that object?
    4. Estimate the contact start/end frame.
    Return JSON.
    """
    response = gemini_2_5_flash.generate(
        video=video_frames, 
        masks=sam3_masks,
        prompt=prompt
    )
    return parse_json(response)
```

### 3.4 解法三：抓取 affordance（VoxAfford / FSAG 思路）

**首选**：**AffordSim + VoxAfford**（XJTU, Apr 2026, arXiv:2604.11674）——open-vocabulary 3D affordance 检测，在物体点云上出功能区域。

**核心论文事实**：
- 论文在 50 个任务 7 类上评估：grasp 53-93%，pour narrow 1-43%，mug hang 0-47%
- 证明了 affordance-aware 数据生成的必要性

**对你的简化**（R1 是二指夹爪）：
- Affordance 简化为"接触点 + 接触方向"两维
- 输入：物体点云（从 GLB 采样）+ 任务语义（VLM）
- 输出：6-DoF gripper pose + 闭合宽度

```python
# affordance_predict.py —— VoxAfford + VLM
def predict_grasp_affordance(object_glb, task_name):
    # 1. 从 GLB 采样点云
    pcd = sample_points_from_glb(object_glb, n=4096)
    
    # 2. VoxAfford 预测 affordance 区域
    affordance_map = vox_afford(pcd, text=task_name)  # (N,) 每个点的 affordance score
    
    # 3. VLM 推断接触点
    contact_point = pcd[affordance_map.argmax()]
    contact_normal = estimate_normal(pcd, contact_point)
    
    # 4. 6-DoF gripper pose
    grasp_T = compute_grasp_pose(contact_point, contact_normal)
    return grasp_T
```

**备选**：**FSAG**（arXiv:2601.08246, 2026-01）——手指级 affordance 字段，但你用二指夹爪不需要。

### 3.5 解法四：动作原语切分

**综合 A0 + GR00T-Mimic + RIGVid**：

```python
# segment_primitives.py —— 切分成 reach/grasp/manipulate/release/return
def segment_primitives(T_ee_left, T_ee_right, gw_left, gw_right, 
                       object_trajs, sam3_masks, video_frames):
    primitives = []
    
    # 1. 检测接触点：EE 与物体 mask 距离 < 阈值
    contact_start = detect_first_contact(T_ee_left, T_ee_right, object_trajs, sam3_masks)
    
    # 2. 检测释放点：mask 重叠消失或力距反向
    contact_end = detect_release(...)
    
    # 3. 切分
    primitives = [
        Primitive("reach",      start=0,         end=contact_start-5),
        Primitive("grasp",      start=contact_start, end=contact_start+5),
        Primitive("manipulate", start=contact_start+5, end=contact_end-3),
        Primitive("release",    start=contact_end, end=contact_end+2),
        Primitive("return",     start=contact_end+2, end=len(T_ee_left)),
    ]
    return primitives
```

### 3.6 端到端整合代码

```python
# task_semantics.py —— 主入口
def infer_task_and_grasp(video_frames, sam3_masks, hambo_dir, stage4_dir):
    # 1. 任务语义
    task = vlm_infer_task(video_frames, sam3_masks, hambo_dir)
    
    # 2. Affordance + grasp
    object_pcd = stage4_dir / "object_pcd.ply"
    grasp_T = predict_grasp_affordance(object_pcd, task["contact_object"])
    
    # 3. 原语切分
    T_ee_left = np.load(f"{hambo_dir}/T_ee_left.npy")
    primitives = segment_primitives(T_ee_left, ...)
    
    return task, grasp_T, primitives
```

---

## 4. 问题 4：物理仿真落地

### 4.1 决策

| 问题 | 答案 | 理由 |
|------|------|------|
| 仿真器选什么？ | **GalaxeaManipSim** | R1/R1Pro/R1Lite URDF 已有 + 30+ 任务 + `collect_demos`/`replay_demos` 现成 |
| IK 谁做？ | **`bimanual_relaxed_ik`** | Galaxea 自带；URDF 内置；不写解析 IK |
| 物理参数从哪来？ | **VLM 材质查表** | 简单可靠；日常物体预设库 |
| 物理一致性怎么验证？ | **RoboPaint Render&Compare** | 仿真渲染 vs 真实视频 PSNR/LPIPS |
| Replay 通过率低怎么办？ | 多次重试 + 物理参数调优 | 摩擦/质量查表 |

### 4.2 物理参数估计

**首选**：**RoboSimGS**（Wuhan U + Alibaba, 2025-10）——MLLM 从正交视图推断物理参数。

**对你的简化**（R1 任务物体都是日常用品）：

```python
# physics_infer.py —— 查表 VLM
PHYSICS_TABLE = {
    "plastic":   {"rho": 950,  "friction": 0.5, "young": 2e9},
    "glass":     {"rho": 2500, "friction": 0.4, "young": 70e9},
    "ceramic":   {"rho": 2400, "friction": 0.5, "young": 200e9},
    "wood":      {"rho": 700,  "friction": 0.6, "young": 12e9},
    "metal":     {"rho": 7800, "friction": 0.3, "young": 200e9},
    "rubber":    {"rho": 1100, "friction": 0.9, "young": 0.01e9},
}

def vlm_infer_physics(object_name, image_crop):
    material = vlm_query(f"What is the material of {object_name} in this image? "
                         f"Return one of: plastic, glass, ceramic, wood, metal, rubber.")
    return PHYSICS_TABLE[material]
```

### 4.3 物理一致性验证（RoboPaint 思路）

**首选论文**：**RoboPaint**（Paxini + SJTU + ZJU, arXiv:2602.05325, 2026-02）

**核心思想**：Real-Sim-Real 流水线，**仿真渲染 vs 真实视频**做一致性检查。

```python
# physics_validate.py —— RoboPaint Render&Compare
def validate_physics_consistency(env_id, replay_actions, real_video):
    # 1. Replay actions in GalaxeaManipSim
    env = GalaxeaSimEnv(env_id)
    rendered_frames = []
    for a in replay_actions:
        env.step(a)
        rendered_frames.append(env.render_camera("head_rgb"))
    env.close()
    
    # 2. PSNR / LPIPS 对比
    psnr = compute_psnr(rendered_frames, real_video)
    lpips = compute_lpips(rendered_frames, real_video)
    rmse_6d = compute_trajectory_rmse(rendered_frames, real_video)
    
    # 3. 物理违规检测
    violations = check_physics_violations(env.history)  # 穿模、掉落、卡死
    
    return {
        "psnr": psnr,        # > 25dB pass
        "lpips": lpips,      # < 0.2 pass
        "rmse_6d": rmse_6d,  # < 3cm pass
        "violations": violations,  # == 0 pass
    }
```

**不达标时的调优**：

```python
def tune_physics_params(env_id, replay_actions, real_video, max_iters=10):
    """RoboPaint-style 物理参数调优"""
    for it in range(max_iters):
        metrics = validate_physics_consistency(env_id, replay_actions, real_video)
        if metrics["psnr"] > 25 and metrics["lpips"] < 0.2:
            break
        # 不达标: 调整摩擦 / 质量 / 接触刚度
        if metrics["violations"] > 0:
            increase_contact_stiffness()
        if metrics["rmse_6d"] > 0.03:
            adjust_friction(delta=0.05)
    return env_id
```

### 4.4 视频→关节轨迹的转换（关键！）

```python
# video_to_joint_traj.py —— 你需要写的唯一转换器
def video_to_galaxea_actions(
    video_frames,        # (S, H, W, 3)
    hambo_mano_t,        # (T, 4, 4) 人手 wrist
    mano_joints_t,       # (T, 21, 3) MANO 关节点
    object_trajs,        # {obj_id: (T, 4, 4)} 物体 6DoF
    glb_poses_t,         # (T, 4, 4) 物体 GLB 在世界系位姿
    base_xyt,            # (x, y, theta)
):
    # 1. MANO → Galaxea EE (WARPED 风格)
    T_ee_seq, gw_seq = warp_style_retarget(hambo_mano_t, mano_joints_t)
    
    # 2. 动作原语切分
    primitives = segment_primitives(T_ee_seq, object_trajs)
    
    # 3. 转 Galaxea 控制器接口
    gal_actions = to_galaxea_actions(T_ee_seq, gw_seq, primitives, base_xyt)
    
    return gal_actions
```

### 4.5 Sim-to-Real Gap 兜底：DiffuDepGrasp 思路

**ICRA 2026** 论文 **DiffuDepGrasp**（arXiv:2511.12912）：扩散深度生成器，零样本 sim-to-real 95.7% 抓取成功率。

**对你**：把 Galaxea 仿真器渲染的 depth 加一个 diffusion-based 噪声模型，让训练分布与真机 RGB-D 分布一致。

---

## 5. 问题 5：场景与动作的泛化

### 5.1 三个层次的泛化

| 层次 | 挑战 | 解法 |
|------|------|------|
| 场景泛化 | 同一动作在厨房/客厅/办公室都有效 | **GRAIL** 3D 资产组合 |
| 物体泛化 | 同一抓取在杯/瓶/盒都工作 | **AffordSim** VoxAfford + **RIGVid** |
| 跨 embodiment 泛化 | R1 / R1 Pro / R1 Lite 都跑 | **RoboWheel** cross-embodiment + **HORA** dataset |

### 5.2 数据扩增四件套（重点 2025-2026 论文）

| 工具 | 论文 | 用途 |
|------|------|------|
| **RoboWheel** | **CVPR 2026, Tsinghua** | 端到端 HOI 视频→跨 embodiment 数据引擎 |
| **RIGVid** | **ICLR 2026, UIUC + Columbia** | 生成视频→6D pose 跟踪→retarget，零演示 |
| **4DGen** | **ICLR 2026, Stanford** | 4D 视频生成 + pointmap，从一帧推未来 |
| **AffordSim** | **2026-04, XJTU** | Affordance-aware Isaac Sim 数据生成 |
| **EgoSim** | **2026-04, SJTU + SH-AI Lab + HKU** | 第一视角世界模拟器，3D 场景可更新 |
| **DreamDojo** | **2026-02, NVIDIA** | 44k 小时人类视频，latent action 世界模型 |
| **H2R** | **2026-01, PKU + UW** | 不配对人→机器人视频转换 |
| **GRAIL** | **2026-06, NVIDIA + UCLA** | 全数字 humanoid loco-manipulation 数据生成 |
| **DiffuDepGrasp** | **ICRA 2026** | 扩散深度 sim-to-real 95.7% |
| **TrajBooster** | 2025-09, ZJU | 跨构型 VLA 微调 |

### 5.3 RoboWheel 详细（你最重要的参考）

**论文**：**RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning**（CVPR 2026, Tsinghua, arXiv 已上传）

**项目页**：https://zhangyuhong01.github.io/Robowheel/

**核心思想**（**直接对应当前项目**）：
1. HOI 视频 → 高精度人手-物体重建
2. **RL 优化器**：在 contact / penetration 约束下精修手-物体相对位姿（**保证物理合理**）
3. **跨 embodiment retarget**：到 robot arms + dexterous hands + humanoids
4. **Sim-augmented 框架**（Isaac Sim）：embodiment / trajectory / object / background / hand mirror 五维 domain randomization
5. **HORA 数据集**：150k 高质量操作轨迹，毫米级精度，融合触觉信号

**对你的项目**：
- 你和 RoboWheel 几乎做同一件事——他们已经做了 150k 轨迹
- **强烈建议**：跑 RoboWheel 公开的 HORA 数据集做 baseline
- **集成思路**：把 RoboWheel 的 RL 物理优化器（**对当前 stage4 的 z 方向误差问题可能直接有效**）作为你 stage4 的物理合理性后处理步骤

### 5.4 RIGVid 详细（"跟物体不跟手"思路）

**论文**：**Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations**（ICLR 2026, UIUC + Columbia, arXiv:2507.00990）

**核心思想**（**核心创新**：**跟踪被操作物体，不跟踪手**）：
1. 文生视频（Kling v1.6）→ 演示视频
2. VLM 过滤不合理的视频
3. monocular depth + 主动物体 mask（Grounding DINO + SAM-2）
4. **FoundationPose** 跟踪**物体** 6D pose
5. 物体轨迹 → embodiment-agnostic retarget（end-effector = T_obj × 固定 T_gripper_obj 偏移）
6. 闭环控制：偏差 > 3cm/20° 时回退到上一成功点

**对你的项目**：
- **直接复用 FoundationPose 做物体 6D 跟踪**（**这正是你 stage4 应该用的方法**）
- 思路可以借鉴：跟踪"被操作物体"作为更稳定的代理（不依赖人手重建的精度）
- 物体 6D trajectory → EE trajectory 的**embodiment-agnostic retarget 公式**可以直接抄

### 5.5 EgoSim 详细（你项目的镜像）

**论文**：**EgoSim: Egocentric World Simulator for Embodied Interaction Generation**（arXiv 2604.01001, 2026-04, SJTU + Shanghai AI Lab + HKU）

**核心思想**（**你的项目本质就是 EgoSim 的"反向"**）：
- EgoSim：从 (场景视频, 动作) → 生成 egocentric 观察 + 更新 3D 场景
- 你：从 (egocentric 视频) → 提取场景 + 动作 → 在仿真器重建

**对你的项目**：
- **可以直接用 EgoSim 的"3D 场景可更新"思路**——把 3DGS / 点云从静态升级为可更新
- 借鉴 EgoSim 的 monocular depth + TSDF 流水线（你已经在用 ReplicateAnyScene）
- 借鉴 EgoSim 的 VLM-based 数据过滤

### 5.6 DreamDojo 详细（世界模型新范式）

**论文**：**DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos**（arXiv 2602.06949, 2026-02, NVIDIA + HKUST + UC Berkeley + Stanford）

**核心思想**：
- 人类视频和机器人视频底层物理一致
- 用 44,000 小时人类第一视角视频预训练
- 用 **Latent Action Model** 自监督学习"动作语义"
- 三阶段训练：预训练（学物理）+ 后训练（学机器人控制）+ 蒸馏（实时性）
- **10.81 FPS 实时推理**，1 分钟长程生成

**对你的项目**：
- **可作为你的物理一致性验证器**——比 PSNR / LPIPS 鲁棒
- **可作为你的"另一种仿真器"**——和 GalaxeaManipSim 互补
- 借鉴 latent action 思想压缩你 HaWoR 的输出

### 5.7 资产库搭建

```bash
# 1. Objaverse 拉取
python -m objaverse.load --category kitchen
python -m objaverse.load --category furniture
python -m objaverse.load --category tool

# 2. 从图片生成 (TripoSR / InstantMesh)
python -m embodiedgen.image_to_3d --image_dir data/crops/ --output assets/generated/

# 3. 物理参数化
python -m vlm_infer_physics --assets_dir assets/ --output assets/with_physics/
```

### 5.8 评测协议

```python
# evaluation_protocol.py —— 三维泛化评测
def evaluate_policy(policy, tasks):
    results = {}
    for task_name, task_cfg in tasks.items():
        for env in get_envs(task_cfg["envs"]):           # 场景
            for embodiment in ["r1", "r1_pro", "r1_lite"]:  # embodiment
                for obj in task_cfg["objects"]:          # 物体
                    success = run_episode(policy, env, embodiment, obj, n=20)
                    results[(task_name, env, embodiment, obj)] = success
    return results
```

---

## 6. 2025-2026 必读论文清单（10+ 篇，每篇 100+ 行解读见配套文件）

> **完整深度解读**：[REFERENCED_PAPERS_DETAIL.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/REFERENCED_PAPERS_DETAIL.md)
>
> 每篇论文 100+ 行覆盖：问题、解法、实验、对你项目的具体用法、caveats、代码骨架

### 第一梯队：直接解决你的 5 个问题

| # | 论文 | 对应问题 | 链接 | 详细解读 |
|---|------|---------|------|---------|
| 1 | **EgoSim** (2026-04, SJTU+SH-AI+HKU) | #2+#5 3D 场景可更新 + egocentric 仿真 | [arXiv:2604.01001](https://arxiv.org/abs/2604.01001) | ✓ |
| 2 | **RoboWheel** (CVPR 2026, Tsinghua) | #1+#5 HOI→跨 embodiment 全 pipeline | [project](https://zhangyuhong01.github.io/Robowheel/) | ✓ |
| 3 | **RIGVid** (ICLR 2026, UIUC+Columbia) | #2 6D pose 跟踪 + retarget | [arXiv:2507.00990](https://arxiv.org/pdf/2507.00990v3) | ✓ |
| 4 | **4DGen** (ICLR 2026, Stanford) | #2 4D 视频生成 + pointmap | [project](https://robot4dgen.github.io/) | ✓ |
| 5 | **SAM 3** (Meta 2025-11, ICLR 2026) | #3 概念分割 | [arXiv:2511.16719](https://arxiv.org/pdf/2511.16719v2) | ✓ |
| 6 | **H2R** (2026-01, PKU+UW) | #5 视频→机器人数据增广 | [arXiv:2505.11920](https://arxiv.org/html/2505.11920v3) | ✓ |
| 7 | **DreamDojo** (2026-02, NVIDIA) | #4+#5 世界模型 + latent action | [arXiv:2602.06949](https://arxiv.org/pdf/2602.06949) | ✓ |
| 8 | **GRAIL** (2026-06, NVIDIA+UCLA) | #5 humanoid 全数字数据生成 | [arXiv:2606.05160](https://arxiv.org/html/2606.05160v1) | ✓ |
| 9 | **FoundationPose** (CVPR 2024, NVIDIA) | #2 6D pose 工业金标准 | [CVPR 2024] | ✓ |
| 10 | **AffordSim** (2026-04, XJTU) | #3+#5 affordance 数据生成 | [arXiv:2604.11674](https://arxiv.org/pdf/2604.11674) | ✓ |

### 第二梯队：核心辅助

| # | 论文 | 用途 |
|---|------|------|
| 11 | **WARPED** (2026-04, CMU) | wrist→EE 模板 |
| 12 | **Mobi-π** (CoRL 2025, Stanford) | 底座贝叶斯优化 |
| 13 | **SynManDex** (2026-06) | force-closure 验证 |
| 14 | **OMNI-PoseX** (2026-04) | SO(3) flow matching 6D pose |
| 15 | **EgoXtreme** (CVPR 2026) | 第一视角 6D pose benchmark |
| 16 | **M-VTOP** (ICRA 2026, MERL) | 视觉+触觉 6D |
| 17 | **MoDex** (2026-06, KTH) | 抓取 DP |
| 18 | **ObjectForesight** (2026-01, UW+CMU) | 物体轨迹预测 |
| 19 | **RoboPaint** (2026-02, Paxini) | 物理一致性 |
| 20 | **DiffuDepGrasp** (ICRA 2026) | 扩散深度 sim-to-real 95.7% |
| 21 | **TrajBooster** (2025-09, ZJU) | 跨构型 VLA |
| 22 | **TWIST** (CoRL 2025, Stanford) | 全身 humanoid teleop |
| 23 | **MegaPose** (RA-L 2023, NVIDIA) | Render&Compare 开山 |
| 24 | **GTR** (2025-05, Woven by Toyota) | 3DGS + Pose Tracking |
| 25 | **FSAG** (2026-01) | 手指 affordance |
| 26 | **Category-level Last-meter Nav** (2025-12, UMN) | category-level 底座 |
| 27 | **RoboSimGS** (2025-10, Wuhan U) | MLLM 物理参数 |
| 28 | **HORA dataset** (Tsinghua CVPR 2026) | 150k HOI 公开数据集 |

### 阅读优先级

| Week | 必读 |
|------|------|
| 1 | **EgoSim + RoboWheel + RIGVid** + Ego-Video-to-SIM README |
| 2 | **4DGen + SAM 3 + WARPED + Mobi-π + SynManDex** |
| 3 | **DreamDojo + H2R + GRAIL + AffordSim + RoboPaint** |
| 4 | **FoundationPose/MegaPose + DiffuDepGrasp + TrajBooster** |

---

## 7. 实施路线图（Phase 0-5）

### Phase 0：基线与决策（3 天）
- [ ] 锁定 R1 Lite 优先（OpenDP 51% baseline，6-DoF）
- [ ] 锁定数据集格式：LeRobot v3 + sidecar JSON
- [ ] 评测集：5 个第一人称视频
- [ ] baseline：collect_demos 跑 50 demos → LeRobot → 训练 DP

### Phase 1：Stage4 升级（1 周，最关键）
- [ ] 把 stage4 从 "VGGT Umeyama" 改为 "SAM 3 mask + FoundationPose Render&Compare"
- [ ] 加 Pose Graph Optimization（GTR 风格）
- [ ] 验证：Acc@10% ≥ 0.9，z 方向误差 < 1cm

### Phase 2：Retargeting + 底座（1 周）
- [ ] 复用 `libs/dex_retargeting/`（Ego-Video-to-SIM 仓库已集成）
- [ ] WARPED-style MANO → EE（**不写 IK**）
- [ ] Mobi-π BO 搜索 base pose
- [ ] Galaxea `bimanual_relaxed_ik` 接管
- [ ] 验证：EE 偏差 < 5cm

### Phase 3：任务语义 + 抓取（1 周）
- [ ] SAM 3 替换 Grounding DINO + SAM 2
- [ ] VLM 任务推理（Gemini 2.5 Flash）
- [ ] VoxAfford 抓取 affordance
- [ ] ObjectForesight 物体轨迹预测

### Phase 4：物理一致性（1 周）
- [ ] collect_demos + replay_demos 流程
- [ ] VLM 物理参数查表
- [ ] RoboPaint 风格 PSNR/LPIPS 闭环

### Phase 5：泛化（2 周）
- [ ] RoboWheel HORA 数据集 baseline
- [ ] RIGVid 思路做物体轨迹生成
- [ ] 跨 embodiment 验证
- [ ] 资产库搭建

---

## 8. 风险表

| 风险 | 影响 | 缓解 |
|------|------|------|
| SAM 3 工业场景漏检 | 物体 mask 缺失 | fallback 到 Grounding DINO 1.6 |
| FoundationPose 纹理弱物体失败 | 6D pose 抖动 | MegaPose 多假设投票 + OMNI-PoseX |
| WARPED head-cam 假设不符 | retargeting 偏差 | 显式 head→wrist 外参转换 |
| relaxed_ik 不可达率高 | 任务成功率 < 50% | fallback 到 joint controller |
| R1 Lite 6-DoF 限制 | 长臂展任务做不到 | 升级 R1 Pro 7-DoF |
| Objaverse 资产物理参数不准 | 仿真失真 | VLM 物理查表 |
| 视频与仿真坐标系不一致 | 物体姿态反转 | stage4 坐标系归一化层 |
| EgoSim 风格 3D 状态更新难 | 多步操作累积误差 | 用 RoboWheel RL 优化器 + Pose Graph Optimization |
| HORA 数据集 license | 商业限制 | 需查 RoboWheel 数据集协议 |

---

**文档结束**

- 论文：30+ 篇（2025-2026 占 95%）
- 解法：5 个问题各给出 1-3 个 2026 年最新解法
- 代码：3 个核心模块骨架（warp_style_retarget / stage4_align_v3 / mobi_pi_base_search）
- 终极目标：LeRobot v3 + Galaxea 兼容 + sidecar JSON，**可定制 16+ 项**（见 §0.6）
- 配套详细论文解读：[REFERENCED_PAPERS_DETAIL.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/REFERENCED_PAPERS_DETAIL.md)（10 篇，每篇 100+ 行）
- 官方整合仓库：[Ego-Video-to-SIM](https://github.com/ananansmall/Ego-Video-to-SIM)（含 SAPIEN + dex-retargeting 集成）

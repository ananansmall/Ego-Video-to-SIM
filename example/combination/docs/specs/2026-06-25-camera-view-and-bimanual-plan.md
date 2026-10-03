# SAPIEN 04 相机视角修复 + 双手逻辑 实现计划

**Goal:** 修复 SAPIEN 04 相机视角被机械臂挡住的问题，并添加双手同时驱动两个机械臂的能力。

**Architecture:** 任务1 通过移动机械臂基座到相机后方 (+Y 0.30m, Z 降到 0.55m) 解决视野遮挡；任务2 通过改造数据加载/机器人设置/追踪流程支持双手模式。两个任务共享基座位置逻辑。

**Tech Stack:** Python 3.10, SAPIEN, NumPy, pytransform3d, DexRetargeting, RelaxedIK, HaWoR, MANO

---

## 文件结构

只修改一个文件：
- `/home/an/robot_world_ws/src/dex-retargeting/example/combination/04_physics_simulation.py` (3452 行)

  职责：SAPIEN 物理仿真主文件，包含场景设置、机器人加载、retargeting、IK、渲染。

  改动点：
  - 常量 (line 106-108): COMFORTABLE_REACH, 新增 BASE_OFFSET_Y, COMFORT_TARGET_IN_BASE
  - `_detect_hand_idx` (line 463-491): 返回 list 而非 int
  - `load_hawor_data` (line 494-552): hand_idx 接受 int 或 list
  - `_compute_optimal_fixed_base` (line 1574-1575): 添加 Y 偏移
  - `_compute_fixed_base_clusters` (line 1647-1648, 1671-1672, 1684-1685): 3 处添加 Y 偏移
  - `PhysicsSimulator.__init__` (line 1371-1420): hand_idx 接受 int 或 list
  - `_setup_robot` (line 1745-1826): 重构为 `_setup_robots` 支持双手
  - 新增 `run_bimanual_tracking` 方法: 双手追踪主流程
  - `main` (line 3300-3301, 3360-3368, 3434-3448): --hand-idx 改为 str

---

## 任务1：移动机械臂基座到相机视野外

### 步骤 1.1: 修改常量定义

**文件**: `04_physics_simulation.py`
**位置**: line 106-108

```python
# 改前
COMFORTABLE_REACH = 0.70  # 基座高度: 提高到 0.70m, 让机械臂垂直抓取, 不靠近桌面
COMFORT_TARGET_IN_BASE = np.array([0.25, 0.0, -0.55])  # 舒适目标点: 更低更近, 机械臂垂直下垂
BASE_TRACKING_RANGE = 0.0  # 固定底座: 不跟踪手腕

# 改后
COMFORTABLE_REACH = 0.55  # 基座高度: 降到 0.55m, 让基座在相机后方时仍在臂展内
BASE_OFFSET_Y = 0.30      # 基座 Y 偏移到相机后方 (相机看 -Y, 基座在 +Y, 不挡视野)
COMFORT_TARGET_IN_BASE = np.array([0.25, 0.0, -0.50])  # 舒适目标点 (Z 跟随 COMFORTABLE_REACH 调整)
BASE_TRACKING_RANGE = 0.0  # 固定底座: 不跟踪手腕
```

**说明**: COMFORT_TARGET_IN_BASE 是基座坐标系下的舒适目标点。基座 180° Z yaw，基座坐标系 Y 反向。原本 [0.25, 0, -0.55] 在世界坐标 = base + R_z180 @ [0.25, 0, -0.55] = base + [-0.25, 0, -0.55]。新方案 [0.25, 0, -0.50] 在世界坐标 = base + [-0.25, 0, -0.50]。基座 Z 从 +0.70 降到 +0.55，目标 Z 也从 -0.55 调到 -0.50，保持目标点在世界坐标的 Z 大致一致（手腕附近）。

### 步骤 1.2: 修改 `_compute_optimal_fixed_base`

**文件**: `04_physics_simulation.py`
**位置**: line 1574-1575

```python
# 改前
arm_base_pos = centroid.copy()
arm_base_pos[2] += COMFORTABLE_REACH

# 改后
arm_base_pos = centroid.copy()
arm_base_pos[1] += BASE_OFFSET_Y  # 相机后方, 避免机械臂挡住视野
arm_base_pos[2] += COMFORTABLE_REACH
```

### 步骤 1.3: 修改 `_compute_fixed_base_clusters` (3 处)

**文件**: `04_physics_simulation.py`
**位置**: line 1647-1648 (帧数太少分支)

```python
# 改前
base_pos = centroid.copy()
base_pos[2] += COMFORTABLE_REACH

# 改后
base_pos = centroid.copy()
base_pos[1] += BASE_OFFSET_Y
base_pos[2] += COMFORTABLE_REACH
```

**位置**: line 1671-1672 (正常分支)

```python
# 改前
base_pos = centroid.copy()
base_pos[2] += COMFORTABLE_REACH

# 改后
base_pos = centroid.copy()
base_pos[1] += BASE_OFFSET_Y
base_pos[2] += COMFORTABLE_REACH
```

**位置**: line 1684-1685 (超出臂展回退分支)

```python
# 改前
base_pos = wrist_centroid.copy()
base_pos[2] += COMFORTABLE_REACH

# 改后
base_pos = wrist_centroid.copy()
base_pos[1] += BASE_OFFSET_Y
base_pos[2] += COMFORTABLE_REACH
```

### 步骤 1.4: 验证任务1

**命令**:
```bash
source /home/an/miniconda3/etc/profile.d/conda.sh && conda activate dex
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python 04_physics_simulation.py \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --fast-collision --num-frames 10
```

**预期输出**:
- 渲染成功 10/10 帧
- 日志显示 `最优固定基座位置: [-0.014, +0.288, +0.559]` (Y 偏移 +0.30, Z=0.55)
- 日志显示 `基座到最远手腕距离: 0.644m` (< 0.713m 臂展)

**验证脚本** (保存到 /tmp/verify_task1.py):
```python
import cv2
import numpy as np

cap = cv2.VideoCapture("/home/an/robot_world_ws/src/dex-retargeting/example/combination/physics_pipeline/output/physics_sim_physics_tracking.mp4")
ok, frame = cap.read()
cap.release()

# 桌面颜色 [0.47, 0.13, 0.32] (RGB) = [82, 33, 120] (BGR)
TABLE_BGR = np.array([82, 33, 120])
dist = np.linalg.norm(frame.astype(float) - TABLE_BGR.reshape(1, 1, 3), axis=2)
table_pct = 100 * (dist < 30).sum() / (frame.shape[0] * frame.shape[1])

# 银灰色机械臂
gray_mask = (np.abs(frame.astype(float) - np.array([200, 200, 200])).sum(axis=2) < 60) & \
            (frame.astype(float).std(axis=2) < 30)
arm_pct = 100 * gray_mask.sum() / (frame.shape[0] * frame.shape[1])

print(f"桌面颜色像素: {table_pct:.2f}% (目标 > 5%)")
print(f"机械臂像素: {arm_pct:.2f}% (目标 < 50%)")
assert table_pct > 5, f"桌面像素 {table_pct:.2f}% 未达目标 5%"
assert arm_pct < 50, f"机械臂像素 {arm_pct:.2f}% 未达目标 50%"
print("✓ 任务1 验证通过")
```

---

## 任务2：双手同时驱动两个机械臂

### 步骤 2.1: 修改 `_detect_hand_idx` 返回 list

**文件**: `04_physics_simulation.py`
**位置**: line 463-491

```python
# 改前
def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中哪只手是活跃的

    通过检查 cam_space/ 目录下的子目录来判断:
    - 如果只有 0/ 目录 → 左手活跃
    - 如果只有 1/ 目录 → 右手活跃
    - 如果两者都有 → 默认左手 (idx=0)

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        int: 手部索引 (0=左手, 1=右手)，或 None(无法检测)
    """
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if 0 in detected and 1 not in detected:
            return 0
        if 1 in detected and 0 not in detected:
            return 1
        if 0 in detected:
            return 0
        if 1 in detected:
            return 1
    return None

# 改后
def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中活跃的手

    通过检查 cam_space/ 目录下的子目录来判断:
    - [0]: 仅左手活跃
    - [1]: 仅右手活跃
    - [0, 1]: 双手都活跃
    - []: 无法检测

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        list[int]: 活跃手的索引列表 (空/单/双)
    """
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        return sorted(detected)
    return []
```

### 步骤 2.2: 修改 `load_hawor_data` 支持 list

**文件**: `04_physics_simulation.py`
**位置**: line 494-552

修改函数签名和返回值:

```python
# 改前
def load_hawor_data(hawor_dir, hand_idx=0):
    """... (现有文档字符串, hand_idx: 手部索引 (0=左手, 1=右手))"""
    # ... 加载原始数据 ...
    return {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
        "img_focal": img_focal,
    }

# 改后
def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    支持两种数据格式:
    1. reconstruction/hawor_results_*.npz (推荐, 含相机轨迹和焦距)
    2. world_space_res.pth (旧格式, 无相机轨迹)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: int (单手 0/1) 或 list[int] (双手 [0, 1])

    Returns:
        单手 (hand_idx=int): dict, 包含 pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, img_focal
        双手 (hand_idx=list): {0: dict, 1: dict}, 每只手一个 dict
    """
    # ... 现有加载逻辑 (rec_file/ws_file/img_focal) 不变 ...
    hawor_path = Path(hawor_dir)
    rec_file = _find_reconstruction_file(hawor_path)
    ws_file = hawor_path / "world_space_res.pth"
    img_focal = None
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        pred_trans = rec['pred_trans']
        pred_rot = rec['pred_rot']
        pred_hand_pose = rec['pred_hand_pose']
        pred_betas = rec['pred_betas']
        pred_valid = rec['pred_valid']
        if 'img_focal' in rec:
            img_focal = float(rec['img_focal'])
    elif ws_file.exists():
        ws = joblib.load(str(ws_file))
        pred_trans = ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0])
        pred_rot = ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1])
        pred_hand_pose = ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2])
        pred_betas = ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3])
        pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
    else:
        raise FileNotFoundError(/* 现有错误信息不变 */)

    est_focal_file = hawor_path / "est_focal.txt"
    if img_focal is None and est_focal_file.exists():
        try:
            img_focal = float(est_focal_file.read_text().strip())
        except Exception:
            pass

    def _pack(idx):
        return {
            "pred_trans": pred_trans[idx],
            "pred_rot": pred_rot[idx],
            "pred_hand_pose": pred_hand_pose[idx],
            "pred_betas": pred_betas[idx],
            "pred_valid": pred_valid[idx],
            "img_focal": img_focal,
        }

    if isinstance(hand_idx, list):
        return {idx: _pack(idx) for idx in hand_idx}
    return _pack(hand_idx)
```

### 步骤 2.3: 修改 `PhysicsSimulator.__init__` 支持 list

**文件**: `04_physics_simulation.py`
**位置**: line 1371-1420

修改 `hand_idx` 参数处理:

```python
# 改前 (line 1408)
self.hand_idx = hand_idx

# 改后
self.hand_idx = hand_idx if isinstance(hand_idx, list) else [hand_idx]
self.bimanual = len(self.hand_idx) == 2
```

### 步骤 2.4: 重构 `_setup_robot` 为 `_setup_robots` 支持双手

**文件**: `04_physics_simulation.py`
**位置**: line 1745-1826

将 `_setup_robot` 重命名为 `_setup_robots`，参数改为接受 hand_indices 列表:

```python
def _setup_robots(self, scene, hand_indices, arm_base_positions, arm_base_quats):
    """创建并配置 R1 臂机器人 (支持单/双手)

    Args:
        scene: SAPIEN 场景
        hand_indices: list[int], [0] 左手 / [1] 右手 / [0, 1] 双手
        arm_base_positions: {hand_idx: (3,)} 每只手的基座位置
        arm_base_quats: {hand_idx: (4,)} 每只手的基座朝向

    Returns:
        {hand_idx: dict}: 每只手的机器人信息
            dict 包含: robot, joint_names, arm_joint_indices, gripper_idx1, gripper_idx2, ee_link
    """
    robots = {}
    for hand_idx in hand_indices:
        prefix = "left" if hand_idx == 0 else "right"
        urdf_path_src = FLOATING_LEFT_URDF if hand_idx == 0 else FLOATING_RIGHT_URDF
        starting = LEFT_ARM_STARTING if hand_idx == 0 else RIGHT_ARM_STARTING

        arm_urdf_path = _prepare_arm_urdf(urdf_path_src, prefix)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(arm_urdf_path)

        active_joints = robot.get_active_joints()
        joint_names = [j.name for j in active_joints]
        arm_joint_indices = [i for i, n in enumerate(joint_names) if f"{prefix}_arm_joint" in n]
        gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
        gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

        for i, joint in enumerate(active_joints):
            if i in arm_joint_indices:
                joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)
            elif i in [gripper_idx1, gripper_idx2]:
                joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
            else:
                joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

        init_qpos = robot.get_qpos().copy()
        for j, idx in enumerate(arm_joint_indices):
            if j < len(starting):
                init_qpos[idx] = starting[j]
        init_qpos[gripper_idx1] = 0.04
        init_qpos[gripper_idx2] = 0.04
        robot.set_qpos(init_qpos)

        active_joints = robot.get_active_joints()
        for i, joint in enumerate(active_joints):
            joint.set_drive_target(init_qpos[i])

        robot.set_root_pose(sapien.Pose(arm_base_positions[hand_idx].tolist(),
                                        arm_base_quats[hand_idx].tolist()))

        touch_link_names = [
            f"{prefix}_gripper_finger_link1",
            f"{prefix}_gripper_finger_link2",
        ]
        for link in robot.get_links():
            if link.get_name() in touch_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_physical_material(
                                scene.create_physical_material(
                                    static_friction=1.0,
                                    dynamic_friction=1.0,
                                    restitution=0.6,
                                )
                            )

        ee_link = None
        for link in robot.get_links():
            if f"{prefix}_gripper_link" in link.get_name():
                ee_link = link
                break

        self.logger.info(f"  ✓ {prefix} 臂已加载: {len(arm_joint_indices)} 臂关节 + 2 夹爪关节")

        robots[hand_idx] = {
            "robot": robot,
            "joint_names": joint_names,
            "arm_joint_indices": arm_joint_indices,
            "gripper_idx1": gripper_idx1,
            "gripper_idx2": gripper_idx2,
            "ee_link": ee_link,
        }
    return robots
```

**保留 `_setup_robot` 作为单手兼容方法** (调用 `_setup_robots`):

```python
def _setup_robot(self, scene, arm_base_pos, arm_base_q):
    """单手模式: 加载单个 R1 臂 (兼容现有调用)"""
    robots = self._setup_robots(scene, [self.hand_idx[0]],
                                 {self.hand_idx[0]: arm_base_pos},
                                 {self.hand_idx[0]: arm_base_q})
    r = robots[self.hand_idx[0]]
    return (r["robot"], r["joint_names"], r["arm_joint_indices"],
            r["gripper_idx1"], r["gripper_idx2"], r["ee_link"])
```

### 步骤 2.5: 新增 `run_bimanual_tracking` 方法

**文件**: `04_physics_simulation.py`
**位置**: 在 `run_physics_tracking` 方法之后 (约 line 3200 附近，run_single_gripper_tracking 之前)

由于 run_physics_tracking 很长 (700+ 行)，双手版本的核心逻辑相同，只是对两只手分别处理。**实际实现时复用 run_physics_tracking 的代码结构，关键差异**:
1. 加载双手数据: `hawor_data_multi = load_hawor_data(self.hawor_dir, hand_idx=[0, 1])`
2. 加载双 URDF: `self._setup_robots(scene, [0, 1], {0: left_pos, 1: right_pos}, ...)`
3. 双 retargeting: 左手 `HandType.left`, 右手 `HandType.right`
4. 双 RelaxedIK: `ik_solver.relaxed_ik_left` 和 `ik_solver.relaxed_ik_right`
5. 每帧对两只手分别 retargeting + IK + set_qpos

**实现策略** (最小化代码重复):
- 提取 `run_physics_tracking` 的核心循环为 `_tracking_loop(robot_info, hawor_data, hand_idx, ...)`
- `run_physics_tracking` 调用 `_tracking_loop` 一次
- `run_bimanual_tracking` 调用 `_tracking_loop` 两次（左右手各一次）

**注意**: 完整代码很长，实现时按 run_physics_tracking 的结构复制并修改。关键修改点:
- `mano_side = "left" if hand_idx == 0 else "right"`
- `config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.left if hand_idx == 0 else HandType.right)`
- `target_link_names` 用对应 prefix
- `ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)` 或 `ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)`

### 步骤 2.6: 修改 main 命令行参数

**文件**: `04_physics_simulation.py`
**位置**: line 3300-3301

```python
# 改前
parser.add_argument("--hand-idx", type=int, default=-1,
                    help="手的索引: 0=左手, 1=右手, -1=自动检测")

# 改后
parser.add_argument("--hand-idx", type=str, default="-1",
                    help="手: 0=左手, 1=右手, both=双手, -1=自动检测")
```

**位置**: line 3348-3349 (输出文件名)

```python
# 改前
if args.hand_idx >= 0:
    name_parts.append(f"h{args.hand_idx}")

# 改后
if args.hand_idx not in ["-1", "both"]:
    name_parts.append(f"h{args.hand_idx}")
elif args.hand_idx == "both":
    name_parts.append("bimanual")
```

**位置**: line 3360-3368 (自动检测)

```python
# 改前
if args.hand_idx < 0:
    detected = _detect_hand_idx(Path(args.hawor_dir))
    if detected is not None:
        args.hand_idx = detected
        hand_label = "左手" if detected == 0 else "右手"
        print(f"自动检测到手: {hand_label} (idx={detected})")
    else:
        args.hand_idx = 0
        print(f"无法自动检测手, 默认使用左手 (idx=0)")

# 改后
if args.hand_idx == "-1":
    detected = _detect_hand_idx(Path(args.hawor_dir))
    if detected:
        args.hand_idx = "both" if len(detected) == 2 else str(detected[0])
        labels = ["左手", "右手"]
        print(f"自动检测到手: {[labels[i] for i in detected]} (idx={detected})")
    else:
        args.hand_idx = "0"
        print(f"无法自动检测手, 默认使用左手 (idx=0)")

# 解析 hand_idx 为 list
if args.hand_idx == "both":
    hand_indices = [0, 1]
else:
    hand_indices = [int(args.hand_idx)]
```

**位置**: line 3434-3448 (创建 simulator 并调用)

```python
# 改前
sim = PhysicsSimulator(hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                       transform_params_path=args.transform_params,
                       output=args.output, fps=args.fps, hand_idx=args.hand_idx,
                       ...)
sim.run_physics_tracking(start_frame=args.start_frame, num_frames=args.num_frames)

# 改后
sim = PhysicsSimulator(hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                       transform_params_path=args.transform_params,
                       output=args.output, fps=args.fps, hand_idx=hand_indices,
                       ...)
if sim.bimanual:
    sim.run_bimanual_tracking(start_frame=args.start_frame, num_frames=args.num_frames)
else:
    sim.run_physics_tracking(start_frame=args.start_frame, num_frames=args.num_frames)
```

### 步骤 2.7: 验证任务2

**单手回归测试**:
```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python 04_physics_simulation.py \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --fast-collision --num-frames 10 --hand-idx 1
```
预期: 渲染 10/10 帧，输出 `physics_sim_physics_tracking_h1.mp4`

**双手模式测试**:
```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python 04_physics_simulation.py \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --fast-collision --num-frames 10 --hand-idx both
```
预期: 渲染 10/10 帧，输出 `physics_sim_physics_tracking_bimanual.mp4`，视频中可见左右两个机械臂

**自动检测测试**:
```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python 04_physics_simulation.py \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --fast-collision --num-frames 10
```
预期: 自动检测到双手，进入双手模式

---

## Self-Review

### Spec 覆盖检查
- ✓ 任务1 基座位置: 步骤 1.1-1.4 覆盖常量 + 3 处函数 + 验证
- ✓ 任务2 双手逻辑: 步骤 2.1-2.7 覆盖检测/加载/设置/追踪/命令行/验证
- ⚠ 步骤 2.5 (run_bimanual_tracking) 代码较长，实现时需参考 run_physics_tracking 结构

### Placeholder 检查
- ⚠ 步骤 2.5 标注"完整代码很长，实现时按 run_physics_tracking 的结构复制并修改"——这是合理的，因为完整复制 700 行代码到计划文档不现实。实现时遵循设计文档的 [1/8]-[8/8] 结构。

### 类型一致性
- ✓ `_detect_hand_idx` 返回 `list[int]` (改后)
- ✓ `load_hawor_data` 接受 `int | list[int]`，返回 `dict | {int: dict}`
- ✓ `PhysicsSimulator.hand_idx` 内部统一为 `list[int]`
- ✓ `_setup_robots` 接受 `list[int]` 和 `{int: np.ndarray}`
- ✓ main 中 `hand_indices` 是 `list[int]`

---

## Execution Handoff

实现计划已完成。建议执行方式:

**推荐: 分阶段执行**
1. 先执行任务1 (步骤 1.1-1.4): 改 4 处常量 + 函数，运行验证
2. 任务1 验证通过后，执行任务2 (步骤 2.1-2.7): 双手逻辑改造
3. 任务2 验证通过后，运行集成测试 (双手 + 新基座)

**备选: 一次性执行所有步骤**
- 风险: 如果任务1 验证失败，任务2 的改动也需要回退
- 优势: 一次完成

请选择执行方式。

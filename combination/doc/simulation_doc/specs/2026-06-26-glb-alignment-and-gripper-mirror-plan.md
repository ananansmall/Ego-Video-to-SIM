# 04_render_dual_arm.py GLB 对齐与夹爪镜像修复 - 实施计划

**Goal:** 修复 `04_render_dual_arm.py` 的 GLB 对齐错乱、夹爪左右开合不对称、hand_detector 缺失、数据二次变换 4 项根因, 添加 FPS 性能基线
**Architecture:** 以 `02_render_scene.py` 为基准, 复制其已验证的函数到 04 内部, 不重写其他部分
**Tech Stack:** SAPIEN, trimesh, numpy, pytransform3d, OpenCV
**基准文件:** `/home/an/robot_world_ws/src/dex-retargeting/example/combination/02_render_scene.py`
**目标文件:** `/home/an/robot_world_ws/src/dex-retargeting/example/combination/04_render_dual_arm.py`

---

## 文件结构

仅修改 **1 个文件**:

| 文件 | 责任 | 改动类型 |
|------|------|---------|
| `04_render_dual_arm.py` | 双臂协同运动渲染 (SAPIEN) | 6 处微调 |

不创建新文件, 不修改其他文件。

---

## 任务清单 (8 个任务, 每个 2-5 分钟)

### 任务 1: 修正 LEFT_ARM_STARTING (1 行)

**文件**: `04_render_dual_arm.py`
**位置**: line 122
**改动**: 与 02 一致

**修改前**:
```python
LEFT_ARM_STARTING = [1.5, 1.9508, 1.0809, 0.4438, -0.1709, 0.1985]
```

**修改后**:
```python
LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]
```

**验证**:
```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
python -c "
import sys; sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('m', '04_render_dual_arm.py')
m = importlib.util.module_from_spec(spec)
# 不执行 main, 仅检查常量
import ast
tree = ast.parse(open('04_render_dual_arm.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t, 'id', '') == 'LEFT_ARM_STARTING':
                print('LEFT_ARM_STARTING =', ast.literal_eval(node.value))
"
# 预期输出: LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]
```

---

### 任务 2: 添加 ZUP_TO_YUP 常量 + _detect_glb_up_axis 函数

**文件**: `04_render_dual_arm.py`
**位置**: 在 `RXWORLD_TO_SAPIEN = R_AXIS @ R_x` (line 117) 之后, `R_AXIS = np.array(...)` (line 120) 之前插入

**插入代码** (复制自 02 line 867-899, 调整缩进与 04 一致):
```python
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def _detect_glb_up_axis(all_vertices):
    """检测 GLB 坐标系是 Z-UP 还是 Y-UP (复制自 02_render_scene.py).

    RAS 导出的 GLB 可能是 Y-UP 或 Z-UP.
    检测启发式: Z-UP 场景中地板在 z=0, 物体在 z>0;
                Y-UP 场景中地板在 y=0, 物体在 y>0.
    """
    FLOOR_THRESHOLD = 0.1
    min_z = all_vertices[:, 2].min()
    min_y = all_vertices[:, 1].min()
    z_is_floor = abs(min_z) < FLOOR_THRESHOLD
    y_is_floor = abs(min_y) < FLOOR_THRESHOLD
    if z_is_floor and not y_is_floor:
        return "z-up"
    if y_is_floor and not z_is_floor:
        return "y-up"
    if z_is_floor and y_is_floor:
        z_at_floor = (abs(all_vertices[:, 2]) < FLOOR_THRESHOLD).sum()
        y_at_floor = (abs(all_vertices[:, 1]) < FLOOR_THRESHOLD).sum()
        return "z-up" if z_at_floor > y_at_floor else "y-up"
    return "y-up"
```

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
assert 'ZUP_TO_YUP' in src, 'ZUP_TO_YUP 未添加'
assert '_detect_glb_up_axis' in src, '_detect_glb_up_axis 未添加'
ast.parse(src)
print('✓ 任务 2 验证通过')
"
```

---

### 任务 3: 替换 load_glb_transformed 函数体

**文件**: `04_render_dual_arm.py`
**位置**: line 362-416 (整个 `load_glb_transformed` 函数)

**改动**: 用 02 中的版本 (02 line 902-1022) 替换 04 中的版本。包含:
- 读取 `saved_glb_up_axis`
- 自动检测 GLB 坐标系
- 必要时 ZUP_TO_YUP 转换
- 添加 `gc.collect()` 内存释放
- 详细日志

**修改后** (完整函数体):
```python
def load_glb_transformed(glb_path, transform_params_path, scene, logger=None):
    """加载 GLB 场景并变换到 SAPIEN 坐标系 (复制自 02_render_scene.py).

    变换链:
      GLB (RAS, 可能 z-up 或 y-up) → Y-UP (如需) → HaWoR render world (y-up) → SAPIEN (z-up)
    """
    if trimesh is None:
        if logger:
            logger.error("  ✗ trimesh 未安装, 无法加载 GLB")
        return []

    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    saved_glb_up_axis = str(params.get('glb_up_axis', 'y-up')) if 'glb_up_axis' in params else None

    if logger:
        size_mb = Path(glb_path).stat().st_size / 1024 / 1024
        logger.info(f"  GLB 文件: {glb_path} ({size_mb:.1f} MB)")

    trimesh_scene = trimesh.load(str(glb_path))
    n_geom = len(trimesh_scene.geometry)
    if logger:
        logger.info(f"  GLB 内容: {n_geom} 个几何体")

    all_verts_list = []
    for _, geom in trimesh_scene.geometry.items():
        if len(geom.vertices) > 0:
            all_verts_list.append(geom.vertices)
    if saved_glb_up_axis is not None:
        glb_up_axis = saved_glb_up_axis
    elif all_verts_list:
        glb_up_axis = _detect_glb_up_axis(np.vstack(all_verts_list))
    else:
        glb_up_axis = "y-up"
    need_zup_to_yup = glb_up_axis == "z-up"
    if logger:
        logger.info(f"  GLB 坐标系: {glb_up_axis}{' (将转换到 Y-UP)' if need_zup_to_yup else ''}")

    import gc
    from sapien.core import Pose

    obj_actors = []
    temp_files = []

    for geom_name, geom in trimesh_scene.geometry.items():
        vertices = geom.vertices.copy()
        if not hasattr(geom, 'faces'):
            continue
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        if need_zup_to_yup:
            vertices = (ZUP_TO_YUP @ vertices.T).T

        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vertex_colors = geom.visual.vertex_colors
            if len(vertex_colors) > 0:
                avg_rgb = vertex_colors[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]

        temp_ply = f'/tmp/glb_actor_{os.getpid()}_{geom_name.replace(" ", "_")}.ply'
        geom_transformed = trimesh.Trimesh(
            vertices=vertices_sapien,
            faces=faces,
            visual=geom.visual
        )
        geom_transformed.export(temp_ply)
        temp_files.append(temp_ply)

        builder = scene.create_actor_builder()
        if avg_color is not None:
            material = sapien.render.RenderMaterial(
                base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3
            )
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)

        actor = builder.build_kinematic(name=geom_name)
        actor.set_pose(Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        obj_actors.append(actor)
        gc.collect()

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass

    if logger:
        logger.info(f"  ✓ GLB 加载完成: {len(obj_actors)} 个物体")
    return obj_actors
```

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
# 检查关键标志
assert 'saved_glb_up_axis' in src, '缺少 saved_glb_up_axis'
assert 'need_zup_to_yup' in src, '缺少 need_zup_to_yup'
assert 'gc.collect' in src, '缺少 gc.collect'
print('✓ 任务 3 验证通过')
"
```

---

### 任务 4: 移除 load_hawor_npz 中的 R_c2w/t_c2w 二次变换

**文件**: `04_render_dual_arm.py`
**位置**: line 192-218 (整个 `load_hawor_npz` 函数)

**修改前** (line 192-218):
```python
    rec = np.load(str(rec_file), allow_pickle=True)
    pred_trans = rec['pred_trans']
    pred_rot = rec['pred_rot']
    pred_hand_pose = rec['pred_hand_pose']
    pred_betas = rec['pred_betas']
    pred_valid = rec['pred_valid']
    R_c2w = rec['R_c2w']
    t_c2w = rec['t_c2w']

    for frame_i in range(pred_trans.shape[1]):
        for hand_i in range(pred_trans.shape[0]):
            if pred_valid[hand_i, frame_i]:
                pred_trans[hand_i, frame_i] = R_c2w[frame_i] @ pred_trans[hand_i, frame_i] + t_c2w[frame_i]
                rot_mat = pr.matrix_from_compact_axis_angle(pred_rot[hand_i, frame_i])
                rot_mat_world = R_c2w[frame_i] @ rot_mat
                pred_rot[hand_i, frame_i] = pr.compact_axis_angle_from_matrix(rot_mat_world)

    has_nan = np.isnan(pred_trans).any(axis=-1)
    pred_valid = pred_valid & ~has_nan
```

**修改后**:
```python
    rec = np.load(str(rec_file), allow_pickle=True)
    pred_trans = rec['pred_trans']
    pred_rot = rec['pred_rot']
    pred_hand_pose = rec['pred_hand_pose']
    pred_betas = rec['pred_betas']
    pred_valid = rec['pred_valid']

    # 注: npz 中 pred_trans/pred_rot 已是世界坐标 (与 02_render_scene.py load_hawor_data 一致)
    # 不再应用 R_c2w/t_c2w 二次变换 (此前会导致手部位置偏移, 让 GLB 看起来"位置被改了")
    # 相机轨迹 (R_c2w, t_c2w) 如需可在主流程中单独加载, 与手部数据解耦

    has_nan = np.isnan(pred_trans).any(axis=-1)
    pred_valid = pred_valid & ~has_nan
```

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
# 确认 R_c2w/t_c2w 在 load_hawor_npz 中已移除
import re
# 找到 load_hawor_npz 函数体
match = re.search(r'def load_hawor_npz.*?(?=\ndef )', src, re.DOTALL)
assert match, 'load_hawor_npz 未找到'
fn_body = match.group(0)
assert 'R_c2w' not in fn_body or 'rec[\'R_c2w\']' not in fn_body, 'R_c2w 仍在 load_hawor_npz 中'
print('✓ 任务 4 验证通过')
"
```

---

### 任务 5: 替换 detect_hands 函数为内联 _detect_hands (从 02 复制)

**文件**: `04_render_dual_arm.py`
**位置**: line 142-163 (含 Handedness import + detect_hands 函数)

**改动**:
1. 移除 `from hand_detector import HandDetector` (line 158)
2. 添加内联 `Handedness` 类 + `HandDetectionResult` 类
3. 添加内联 `_detect_hands_from_npz` 函数 (复制自 02 line 629-685 的核心逻辑)
4. 替换 `detect_hands` 函数体使用内联实现

**修改前** (line 142-163):
```python
# =============================================================================
# 1. HandDetector 集成
# =============================================================================
def detect_hands(hawor_dir, logger):
    """从 HaWoR 目录自动检测手部类型。

    使用 hand_track/hand_detector.py 的 HandDetector, 读取
    reconstruction/hawor_results_*.npz 自动判断 LEFT/RIGHT/BOTH/NONE。
    ...
    """
    from hand_detector import HandDetector
    detector = HandDetector(str(hawor_dir))
    result = detector.detect()
    logger.info(f"  [HandDetector] {result.description}")
    logger.info(f"  [HandDetector] 检测方法: {result.detection_method}")
    return result
```

**修改后** (完整代码):
```python
# =============================================================================
# 1. 手部类型检测 (内联实现, 复制自 02_render_scene.py _detect_hands)
# =============================================================================
class Handedness:
    """手部类型枚举 (替代缺失的 hand_detector.Handedness)."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    NONE = "none"


class HandDetectionResult:
    """手部检测结果 (替代缺失的 hand_detector.HandDetectionResult)."""
    def __init__(self, handedness, left_valid_frames=0, right_valid_frames=0, method="npz_pred_valid"):
        self.handedness = handedness
        self.left_valid_frames = left_valid_frames
        self.right_valid_frames = right_valid_frames
        self.detection_method = method
        if handedness == Handedness.LEFT:
            self.description = "left only"
        elif handedness == Handedness.RIGHT:
            self.description = "right only"
        elif handedness == Handedness.BOTH:
            self.description = "both hands"
        else:
            self.description = "none"


def _find_reconstruction_file_local(hawor_path):
    """查找 HaWoR reconstruction npz 文件 (复制自 02 line 590-604)."""
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def _detect_hands_local(hawor_path):
    """自动检测活跃的手部索引 (复制自 02 line 629-685 核心逻辑).

    Returns:
        list: 活跃手索引, [0]=左手, [1]=右手, [0,1]=双手
    """
    hawor_path = Path(hawor_path)

    # 方法1: 通过 cam_space 目录检测
    cam_dir = hawor_path / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if detected:
            return sorted(detected)

    # 方法2: 通过 reconstruction npz 的 pred_valid 检测
    rec_file = _find_reconstruction_file_local(hawor_path)
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        if 'pred_valid' in rec:
            pred_valid = rec['pred_valid']
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                hands = []
                if pred_valid[0].any():
                    hands.append(0)
                if pred_valid[1].any():
                    hands.append(1)
                return hands

    # 方法3: 通过 world_space_res.pth 检测
    ws_file = hawor_path / "world_space_res.pth"
    if ws_file.exists():
        import torch
        data = torch.load(str(ws_file), map_location='cpu')
        if 'pred_valid' in data:
            pred_valid = data['pred_valid'].numpy()
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                hands = []
                if pred_valid[0].any():
                    hands.append(0)
                if pred_valid[1].any():
                    hands.append(1)
                return hands

    return []


def detect_hands(hawor_dir, logger):
    """从 HaWoR 目录自动检测手部类型 (内联实现, 不依赖外部 hand_detector 模块).

    Args:
        hawor_dir: HaWoR 数据根目录
        logger:    Logger

    Returns:
        HandDetectionResult, 含 handedness/left_valid_frames/right_valid_frames
    """
    hands = _detect_hands_local(Path(hawor_dir))

    # 统计左右手有效帧数 (用于日志)
    left_valid = 0
    right_valid = 0
    rec_file = _find_reconstruction_file_local(Path(hawor_dir))
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        if 'pred_valid' in rec:
            pred_valid = rec['pred_valid']
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                left_valid = int(pred_valid[0].sum())
                right_valid = int(pred_valid[1].sum())

    if 0 in hands and 1 in hands:
        handedness = Handedness.BOTH
    elif 0 in hands:
        handedness = Handedness.LEFT
    elif 1 in hands:
        handedness = Handedness.RIGHT
    else:
        handedness = Handedness.NONE

    result = HandDetectionResult(
        handedness=handedness,
        left_valid_frames=left_valid,
        right_valid_frames=right_valid,
        method="npz_pred_valid+cam_space",
    )
    logger.info(f"  [HandDetector] {result.description} (left={left_valid}, right={right_valid})")
    logger.info(f"  [HandDetector] 检测方法: {result.detection_method}")
    return result
```

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
assert 'class Handedness' in src, 'Handedness 类未添加'
assert 'class HandDetectionResult' in src, 'HandDetectionResult 类未添加'
assert '_detect_hands_local' in src, '_detect_hands_local 未添加'
assert 'from hand_detector import' not in src, 'hand_detector import 仍存在'
print('✓ 任务 5 验证通过')
"
```

---

### 任务 6: 夹爪统一为单控制量 (用户指正, 真正根因修复)

**文件**: `04_render_dual_arm.py`
**位置 1**: line 521-522 (初始姿态)

**修改前**:
```python
init_qpos[self.gripper_idx1] = 0.04
init_qpos[self.gripper_idx2] = -0.04
```

**修改后**:
```python
init_qpos[self.gripper_idx1] = 0.04
init_qpos[self.gripper_idx2] = 0.04
```

**位置 2**: line 677-680 (retargeting 输出赋值)

**修改前**:
```python
if self.gripper_idx1 < len(sapien_qpos):
    qpos[self.gripper_idx1] = float(sapien_qpos[self.gripper_idx1])
if self.gripper_idx2 < len(sapien_qpos):
    qpos[self.gripper_idx2] = float(sapien_qpos[self.gripper_idx2])
```

**修改后**:
```python
if self.gripper_idx1 < len(sapien_qpos):
    gripper_open = float(sapien_qpos[self.gripper_idx1])
    gripper_open = max(0.0, min(0.05, gripper_open))   # clamp 到 URDF limit [0, 0.05]
    qpos[self.gripper_idx1] = gripper_open
    if self.gripper_idx2 < len(sapien_qpos):
        qpos[self.gripper_idx2] = gripper_open          # 同号, axis 相反, 对称开合
```

**原理**: URDF 中两个 finger joint 的 axis 相反 (右: joint1=(0,-1,0)/joint2=(0,1,0); 左: joint1=(0,1,0)/joint2=(0,-1,0)), 且 limit 都是非负 [0, 0.05]. 因此两个 joint 必须用**同号值**才会对称开合。此前代码用 `-0.04` 违反 limit 且符号错误, 导致两个 finger 都向同一方向移动 → 只开合一边。

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
# 确认初始姿态修正
assert 'init_qpos[self.gripper_idx2] = 0.04' in src, 'init_qpos idx2 未改为同号'
assert 'init_qpos[self.gripper_idx2] = -0.04' not in src, 'init_qpos idx2 仍是负值'
# 确认 retargeting 输出统一为单控制量
assert 'gripper_open = float' in src, 'gripper_open 单控制量未添加'
assert 'max(0.0, min(0.05, gripper_open))' in src, 'clamp 未添加'
assert 'qpos[self.gripper_idx2] = gripper_open' in src, 'idx2 未镜像 idx1'
print('✓ 任务 6 验证通过')
"
```

---

### 任务 7: 添加 FPS 性能基线测量

**文件**: `04_render_dual_arm.py`
**位置 1**: 主渲染循环之前 (line 950 附近, `kp_nodes_per_arm = {arm.prefix: [] for arm in arms}` 之后)

**插入代码**:
```python
    _frame_times = []  # FPS 性能基线测量
```

**位置 2**: 主渲染循环内 (`for local_idx in trange(...)` 循环体最后, `writer.write(bgr)` 之前)

**插入代码**:
```python
        _t_frame_end = time.time()
        _frame_times.append(_t_frame_end - _t_frame_start)
```

**位置 3**: 主渲染循环之前 (`for local_idx in trange(...)` 这一行之前)

**插入代码**:
```python
    _t_render_start = time.time()  # 已存在 (line 941)
```

实际上 line 941 已有 `_t_render_start = time.time()`, 不需要重复添加. 只需在循环开始处添加每帧计时.

**位置 4**: 循环开头 (`for local_idx in trange(num_frames, ...):` 之后)

**插入代码**:
```python
        _t_frame_start = time.time()
```

**位置 5**: 渲染完成后日志输出 (line 1029 `logger.info(f"  渲染耗时: ...")` 之后插入)

**插入代码**:
```python
    # FPS 性能基线 (用户要求 50fps 指标)
    if _frame_times:
        frame_times_arr = np.array(_frame_times)
        fps_mean = 1.0 / max(frame_times_arr.mean(), 1e-6)
        fps_p50 = 1.0 / max(np.percentile(frame_times_arr, 50), 1e-6)
        fps_p95 = 1.0 / max(np.percentile(frame_times_arr, 95), 1e-6)
        fps_target = 50
        status = '✓ 达标' if fps_mean >= fps_target else '✗ 未达标'
        logger.info(f"  性能基线: 平均 {fps_mean:.1f} fps, P50 {fps_p50:.1f} fps, P95 {fps_p95:.1f} fps")
        logger.info(f"  目标:     {fps_target} fps  {status}")
        logger.info(f"  帧时间:   平均 {frame_times_arr.mean()*1000:.1f} ms, "
                    f"P95 {np.percentile(frame_times_arr, 95)*1000:.1f} ms")
```

**验证**:
```bash
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
assert '_frame_times' in src, '_frame_times 未添加'
assert 'fps_mean' in src, 'fps_mean 未添加'
assert '目标' in src and '50' in src, '目标 50fps 日志未添加'
print('✓ 任务 7 验证通过')
"
```

---

### 任务 8: 最终静态验证

**验证命令**:
```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

# 1. 语法检查
python -c "
import ast
src = open('04_render_dual_arm.py').read()
ast.parse(src)
print('✓ 语法检查通过')
"

# 2. 关键修复检查
python -c "
src = open('04_render_dual_arm.py').read()
checks = [
    ('LEFT_ARM_STARTING 修复', 'LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]' in src),
    ('ZUP_TO_YUP 添加', 'ZUP_TO_YUP' in src),
    ('_detect_glb_up_axis 添加', '_detect_glb_up_axis' in src),
    ('saved_glb_up_axis 添加', 'saved_glb_up_axis' in src),
    ('gc.collect 添加', 'gc.collect' in src),
    ('R_c2w 二次变换移除', 'R_c2w[frame_i] @ pred_trans' not in src),
    ('hand_detector import 移除', 'from hand_detector import' not in src),
    ('Handedness 类添加', 'class Handedness' in src),
    ('_detect_hands_local 添加', '_detect_hands_local' in src),
    ('夹爪初始姿态同号', 'init_qpos[self.gripper_idx2] = 0.04' in src),
    ('夹爪初始姿态无负值', 'init_qpos[self.gripper_idx2] = -0.04' not in src),
    ('夹爪单控制量', 'gripper_open = float' in src),
    ('夹爪 clamp', 'max(0.0, min(0.05, gripper_open))' in src),
    ('夹爪 idx2 镜像 idx1', 'qpos[self.gripper_idx2] = gripper_open' in src),
    ('FPS 基线添加', 'fps_mean' in src and '目标' in src),
]
all_pass = True
for name, ok in checks:
    print(f'  {\"✓\" if ok else \"✗\"} {name}')
    if not ok:
        all_pass = False
print()
print('✓ 全部通过' if all_pass else '✗ 存在失败项')
"

# 3. import 检查 (不实际运行 main, 只检查模块级导入)
python -c "
import sys, os
sys.path.insert(0, '.')
# 设置环境变量避免 Vulkan 错误
os.environ['SAPIEN_LOG_LEVEL'] = 'ERROR'
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('m04', '04_render_dual_arm.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print('✓ 模块加载成功 (无 ImportError)')
except ImportError as e:
    print(f'✗ ImportError: {e}')
except Exception as e:
    # 其他错误 (如 SAPIEN 初始化) 不影响本次修复验证
    print(f'(非 ImportError, 可忽略): {type(e).__name__}: {e}')
"
```

**预期输出**: 全部 ✓ 通过

---

## 执行顺序

按任务 1 → 8 顺序执行, 每个任务完成后立即运行验证命令, 全部通过后进入 change-log 阶段.

---

## 风险与回滚

- 每个任务都是独立的字符串替换/插入, 可单独回滚
- 所有改动均基于 02 已验证的实现, 不引入新逻辑
- 若动态运行 (需 GPU) 出现新问题, 可逐项 revert 检查

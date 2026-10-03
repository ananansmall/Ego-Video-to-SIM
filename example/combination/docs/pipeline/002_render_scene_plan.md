# 002_render_scene.py 实施方案

**目标：** 创建与 `001_align_scene.py` 配套的场景渲染脚本，继承 `02_render_scene.py` 全部功能，集成自动手部检测（单/双手），并统一纳入 `00_run_pipeline.py` 新模式。

**坐标哲学：** 仿真坐标系 = GLB 原始 RAS 坐标系。去掉所有 ZUP_TO_YUP、R_AXIS、RXWORLD_TO_SAPIEN 变换。HaWoR 数据通过 `001_align_scene.py` 产出的 `transform_params.npz`（`scale_ratio`, `R_hand_to_glb`, `t_hand_to_glb`）映射到 GLB 坐标系。

**技术栈：** Python 3.10+, SAPIEN, trimesh, OpenCV, MANO, FFmpeg, NumPy

---

## 涉及文件

| 文件 | 操作 | 大致行数 | 职责 |
|------|------|----------|------|
| `hand_track/common.py` | **追加** 3 个函数 | +40 | 提供新坐标下的 GLB 加载、手部关节变换、相机变换工具 |
| `002_render_scene.py` | **新建** | ~850 | 主线渲染脚本，继承 02 全部功能 + 自动检测 + 新坐标 |
| `00_run_pipeline.py` | **修改** | +50 | 添加 `--mode align-render`，串联 001 → 002 |

**不变的文件：** `001_align_scene.py`, `01_align_scene.py`, `02_render_scene.py`, `hand_track/render_auto.py`, `hand_track/render_gripper_only.py`, `gripper_config.py`, `align_strategy.py`, 所有 `test/` 文件。

---

## 任务分解

### Task 1: 追加 `hand_track/common.py` 的 3 个新函数

**位置：** `hand_track/common.py`，在文件末尾追加（不改旧函数）。

#### 1.1 `load_glb_direct(glb_path, scene)`

将 GLB 顶点原样加载到 SAPIEN，不应用任何坐标变换。

```python
def load_glb_direct(glb_path, scene):
    glb_path = Path(glb_path)
    mesh = trimesh.load(str(glb_path))
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    render_mesh = sapien.render.RenderMesh(vertices, faces)
    render_body = scene.add_render_body()
    render_body.add_mesh(render_mesh, sapien.render.RenderMaterial())

    collision_mesh = sapien.collision.CollisionMesh(vertices, faces)
    builder = scene.create_actor_builder()
    builder.add_collision_from_mesh(collision_mesh)
    builder.add_visual_from_mesh(render_mesh)
    actor = builder.build_static("scene")
    return actor
```

**差异对照：** 旧 `load_glb_transformed` 的顶点变换链为 `ZUP_TO_YUP → s_inv(R_inv@v + t_inv) → RXWORLD_TO_SAPIEN`，这里全部删除。

#### 1.2 `hand_to_glb(pts, scale_ratio, R_hand_to_glb, t_hand_to_glb)`

```python
def hand_to_glb(pts, scale_ratio, R_hand_to_glb, t_hand_to_glb):
    pts = np.asarray(pts, dtype=np.float64)
    return scale_ratio * (R_hand_to_glb @ pts.T).T + t_hand_to_glb
```

**替换关系：** 替代 `_render_to_sapien(pts)` 的 `(RXWORLD_TO_SAPIEN @ pts.T).T`。

#### 1.3 `hawor_cam_to_glb_pose(R_c2w, t_c2w, scale_ratio, R_hand_to_glb, t_hand_to_glb)`

```python
def hawor_cam_to_glb_pose(R_c2w, t_c2w, scale_ratio, R_hand_to_glb, t_hand_to_glb):
    cam_pos = scale_ratio * (R_hand_to_glb @ t_c2w) + t_hand_to_glb
    cam_R = R_hand_to_glb @ R_c2w

    forward = cam_R[:, 2]
    left = -cam_R[:, 0]
    up = -cam_R[:, 1]
    sapien_R = np.column_stack([forward, left, up])
    U, _, Vh = np.linalg.svd(sapien_R)
    sapien_R = U @ Vh

    return cam_pos, pr.quaternion_from_matrix(sapien_R)
```

**替换关系：** 替代 `hawor_cam_to_sapien_pose` 的 `R_AXIS @ t / R_AXIS @ R` + OpenCV→SAPIEN 重排。

**命名约定说明：**
- `cam_R` 在 OpenCV 约定中：col0=右矢量, col1=下矢量, col2=前矢量（视线方向）
- SAPIEN 相机约定：X=前, Y=左, Z=上
- 从 OpenCV col2（前）→ SAPIEN X（前），OpenCV -col0（-右=左）→ SAPIEN Y（左），OpenCV -col1（-下=上）→ SAPIEN Z（上）

---

### Task 2: 创建 `002_render_scene.py`

**位置：** `robot_world_ws/src/dex-retargeting/example/combination/002_render_scene.py`

#### 2.1 删除的常量

从旧 `02_render_scene.py` 中删除以下常量定义（约 L70-80）：

| 常量 | 值 | 用途（旧） | 删除原因 |
|------|-----|-----------|---------|
| `ZUP_TO_YUP` | `np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])` | GLB ZUP→SAPIEN YUP | GLB 原样加载 |
| `R_AXIS` | `np.array([[1,0,0],[0,-1,0],[0,0,-1]])` | 轴对齐 | 不再需要 |
| `R_x` | `np.array([[1,0,0],[0,-1,0],[0,0,-1]])` | HaWoR render world 约定 | 由 R_hand_to_glb 替代 |
| `RXWORLD_TO_SAPIEN` | `np.array([[-1,0,0],[0,0,1],[0,1,0]])` | 渲染世界→SAPIEN | 由 hand_to_glb 替代 |

#### 2.2 导入

```python
import sys, os, json, time, subprocess as sp
from pathlib import Path
import numpy as np
import trimesh
import cv2
import sapien
import sapien.render
from tqdm import trange
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")

# 内部模块
from hand_track.common import (
    load_hawor_data, load_hawor_c2w, load_glb_data,
    compute_mano_joints, detect_hands,
    load_glb_direct, hand_to_glb, hawor_cam_to_glb_pose,
    _render_keypoints, IK_SOLVE_PER_FRAME, WARMUP_FRAMES,
    make_look_at_camera,
)
from hand_track.gripper_config import get_gripper_urdf_path
```

#### 2.3 函数清单

| 函数 | 来源 | 签名 | 说明 |
|------|------|------|------|
| `setup_scene()` | 从 02 照搬 | `→ (scene, renderer)` | 创建 SAPIEN 场景和渲染器 |
| `load_hawor_data()` | 从 02 照搬 | `(hawor_dir, ...) → dict` | 加载 HaWoR 重建数据 |
| `_render_keypoints()` | 从 common 导入 | `(joints, ctx, scene, nodes, ref_indices) → nodes` | 渲染关键点球体 |
| `_combine_videos_side_by_side()` | 从 `render_auto.py` 照搬 | `(left_path, right_path, output_path)` | 左右视频并排合成 |
| `load_glb_direct()` | 从 common **import** | `(glb_path, scene) → actor` | 原样加载 GLB |
| `hand_to_glb()` | 从 common **import** | `(pts, s, R, t) → ndarray` | 手部关节→GLB |
| `hawor_cam_to_glb_pose()` | 从 common **import** | `(R, t, s, R_h2g, t_h2g) → (pos, quat)` | 相机→GLB+SAPIEN |
| `detect_hands()` | 从 common **import** | `(hawor_dir) → list[int]` | 自动检测手部 |
| `render_robot_video()` | **重写** | `(hawor_dir, glb_path, transform_params, ...) → Path` | 主渲染逻辑 |
| `render_gripper_video()` | 从 common 照搬并改新坐标 | `(hawor_dir, transform_params, ...) → Path` | 3 关键点球体 |
| `render_gripper_only_video()` | 从 `render_gripper_only.py` 照搬并改新坐标 | `(hawor_dir, transform_params, ...) → Path` | 夹爪 URDF |
| `render_dual_gripper_video()` | 从 `render_gripper_only.py` 照搬并改新坐标 | `(hawor_dir, transform_params, ...) → Path` | 双夹爪并排 |
| `parse_args()` | **新建** | `→ argparse.Namespace` | CLI 参数解析 |
| `main()` | **新建** | `→ None` | 入口：自动检测+渲染调度 |

#### 2.4 `render_robot_video()` 核心重写细节

**输入参数：**
```python
def render_robot_video(
    hawor_dir: str | Path,
    ras_dir: str | Path,
    glb_path: str | Path,
    transform_params: str | Path,
    hand_idx: int = 0,
    output: str | Path | None = None,
    fps: int = 30,
    cam_width: int = 1920,
    cam_height: int = 1080,
    view: str = "fpv",
    crf: int = 18,
    start_frame: int = 0,
    num_frames: int = -1,
    fixed_base: bool = False,
    logger=None,
) -> Path:
```

**内部变换替换对照表：**

| 位置 | 旧 02 代码行 | 旧变换 | 新变换 |
|------|-------------|--------|--------|
| L180 | `scene_actor = load_glb_transformed(glb_path, scene)` | ZUP_TO_YUP → ... → RXWORLD_TO_SAPIEN | `load_glb_direct(glb_path, scene)` |
| L660 | `joints_sapien = _render_to_sapien(j)` | `(RXWORLD_TO_SAPIEN @ j.T).T` | `hand_to_glb(j, s, R_h2g, t_h2g)` |
| L670 | `wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R @ RXWORLD_TO_SAPIEN.T` | 三重矩阵变换 | `R_hand_to_glb @ wrist_R @ R_hand_to_glb.T` |
| L680 | `cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])` | R_AXIS + OpenCV→SAPIEN | `hawor_cam_to_glb_pose(R, t, s, R_h2g, t_h2g)` |
| L883 | 渲染循环内相机更新同上 | 同上 | 同上 |

**其他保持不变的逻辑：** 重定位（retargeting）、IK 求解、关节滤波、ffmpeg 重编码、qpos 保存。

#### 2.5 CLI 参数

```python
def parse_args():
    parser = argparse.ArgumentParser(description="002_render_scene.py — 新坐标系统渲染")
    parser.add_argument("--mode", default="robot_tracking",
                        choices=["hand_only", "robot_only", "robot_tracking", "topdown", "gripper_only"])
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="-1=自动检测, 0=左手, 1=右手")
    parser.add_argument("--hawor-dir", required=True)
    parser.add_argument("--ras-dir", required=True)
    parser.add_argument("--glb-path", required=True)
    parser.add_argument("--transform-params", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--view", default="fpv", choices=["fpv", "behind", "front", "topdown"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--fixed-base", action="store_true")
    parser.add_argument("--gripper-mode", default="gripper_arm",
                        choices=["gripper", "gripper_arm", "both"])
    return parser.parse_args()
```

#### 2.6 `main()` 主流程

```python
def main():
    args = parse_args()
    params = np.load(args.transform_params)
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]

    hand_indices = detect_hands(args.hawor_dir) if args.hand_idx < 0 else [args.hand_idx]
    output = resolve_output(args)

    if args.mode == "gripper_only":
        fn = render_gripper_only_video
    else:
        fn = render_robot_video

    if len(hand_indices) == 2:
        out_path = Path(output)
        out_L = out_path.with_stem(out_path.stem + "_L")
        out_R = out_path.with_stem(out_path.stem + "_R")
        fn(hand_idx=0, output=str(out_L), ...)
        fn(hand_idx=1, output=str(out_R), ...)
        _combine_videos_side_by_side(str(out_L), str(out_R), str(out_path))
        out_L.unlink(missing_ok=True)
        out_R.unlink(missing_ok=True)
        logger.info(f"双手视频已合并: {out_path}")
    else:
        fn(hand_idx=hand_indices[0], output=output, ...)

if __name__ == "__main__":
    main()
```

---

### Task 3: 修改 `00_run_pipeline.py`

**位置：** `combination/00_run_pipeline.py`

#### 3.1 添加 `--mode align-render`

在已有的 mode 选项中新增 `"align-render"`。

```python
parser.add_argument("--mode", default="full",
                    choices=["full", "render", "track_only", "align-render"])
```

#### 3.2 实现 `run_align_render()`

```python
def run_align_render(args):
    """001_align_scene.py → 002_render_scene.py"""

    hawor_dir = args.hawor_dir
    ras_dir = args.ras_dir
    glb_path = args.glb_path
    transform_params = Path(hawor_dir) / "transform_params.npz"

    # Step 1: 001_align_scene.py
    cmd_001 = [
        sys.executable, "001_align_scene.py",
        "--hawor-dir", hawor_dir,
        "--ras-dir", ras_dir,
        "--glb-path", glb_path,
        "--hand-idx", "0",
    ]
    print(f"\n{'='*60}\n[Step 1/2] 001_align_scene.py\n{'='*60}")
    subprocess.run(cmd_001, check=True)

    if not transform_params.exists():
        print(f"错误: {transform_params} 未生成")
        sys.exit(1)

    # Step 2: 002_render_scene.py
    cmd_002 = [
        sys.executable, "002_render_scene.py",
        "--hawor-dir", hawor_dir,
        "--ras-dir", ras_dir,
        "--glb-path", glb_path,
        "--transform-params", str(transform_params),
        "--hand-idx", str(args.hand_idx) if args.hand_idx >= 0 else "-1",
        "--mode", args.mode if args.mode != "align-render" else "robot_tracking",
        "--view", args.view or "fpv",
        "--fps", str(args.fps or 30),
        "--crf", str(args.crf or 18),
        "--start-frame", str(args.start_frame or 0),
        "--num-frames", str(args.num_frames or -1),
    ]
    if args.fixed_base:
        cmd_002.append("--fixed-base")

    print(f"\n{'='*60}\n[Step 2/2] 002_render_scene.py\n{'='*60}")
    subprocess.run(cmd_002, check=True)

    print("\n✓ align-render 完成")
```

#### 3.3 在 main dispatch 中调用

```python
def main():
    args = parse_args()
    if args.mode == "align-render":
        run_align_render(args)
        return
    # ... 原 full/render/track_only 逻辑保持不变
```

---

## 坐标变换对照速查表

| 数据 | 旧 02 变换 | 新 002 变换 |
|------|-----------|------------|
| GLB 顶点 | `ZUP_TO_YUP → s_inv(R_inv@v + t_inv) → RXWORLD_TO_SAPIEN` | 原样 `v` |
| 手部关节点 | `RXWORLD_TO_SAPIEN @ p` | `s * R_h2g @ p + t_h2g` |
| 手腕旋转矩阵 | `RXWORLD_TO_SAPIEN @ R @ RXWORLD_TO_SAPIEN.T` | `R_h2g @ R @ R_h2g.T` |
| 相机位置 | `R_AXIS @ t_c2w` + OpenCV→SAPIEN | `s * R_h2g @ t_c2w + t_h2g` + OpenCV→SAPIEN |
| 相机旋转 | `R_AXIS @ R_c2w` + OpenCV→SAPIEN | `R_h2g @ R_c2w` + OpenCV→SAPIEN |

其中 `s = scale_ratio`, `R_h2g = R_hand_to_glb`, `t_h2g = t_hand_to_glb`。

---

## 验证方法

### 验证 1: GLB 加载正确
```bash
python -c "
import trimesh, numpy as np
m = trimesh.load('path/to/final_scene.glb')
print('原始顶点范围:', m.vertices.min(axis=0), m.vertices.max(axis=0))
print('顶点数量:', len(m.vertices))
"
```

然后对比 SAPIEN 中加载后 `actor.pose.p` 相同的值。

### 验证 2: 单手握视频渲染
```bash
python 002_render_scene.py \
  --hawor-dir data/hawor \
  --ras-dir data/ras \
  --glb-path data/ras/final_scene.glb \
  --transform-params data/hawor/transform_params.npz \
  --hand-idx 0 \
  --mode robot_tracking \
  --view fpv \
  --output /tmp/test_L.mp4 \
  --num-frames 30
```

### 验证 3: 双手自动渲染
```bash
python 002_render_scene.py \
  --hawor-dir data/hawor \
  --ras-dir data/ras \
  --glb-path data/ras/final_scene.glb \
  --transform-params data/hawor/transform_params.npz \
  --hand-idx -1 \
  --mode robot_tracking \
  --view fps \
  --output /tmp/test_both.mp4 \
  --num-frames 30
```

### 验证 4: 完整管线
```bash
python 00_run_pipeline.py --mode align-render \
  --hawor-dir data/hawor \
  --ras-dir data/ras \
  --glb-path data/ras/final_scene.glb
```

---

## 执行顺序

```
Task 1: 追加 common.py 3 个新函数      (~5分钟)
Task 2: 创建 002_render_scene.py        (~30分钟)
  2.1 文件骨架 + 导入 + 常量删除
  2.2 辅助函数（照搬不变量）
  2.3 render_robot_video() 重写
  2.4 render_gripper_video() 照搬改坐标
  2.5 render_gripper_only / dual 照搬改坐标
  2.6 CLI + main() 自动检测调度
  2.7 验证渲染
Task 3: 修改 00_run_pipeline.py          (~5分钟)
  3.1 加 --mode align-render
  3.2 实现 run_align_render()
  3.3 验证完整管线
```
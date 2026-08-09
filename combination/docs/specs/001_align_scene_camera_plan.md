# 001 相机坐标系对齐方案 Implementation Plan

**目标**: 把 GLB 和 HaWoR 数据都映射到第一帧相机坐标系下对齐，对齐后变换回 GLB 初始坐标系作为 SAPIEN 仿真坐标系。GLB、变换参数、对齐后的场景都输出到同一个文件夹。

---

## 1. 方案流程（按你的思路）

```
Step 1: 提取
   ├── GLB 坐标系（up-axis）
   └── 第一帧相机在 GLB 坐标系下的位姿 (R_c2w_glb, t_c2w_glb)
       └── 来自 RAS extrinsics

Step 2: 反变换 — GLB → 第一帧相机坐标系
   p_cam_glb = R_w2c_glb @ p_glb + t_w2c_glb

Step 3: 变换 — HaWoR → 第一帧相机坐标系
   （HaWoR 相机第一帧位姿和它的坐标系不一样，需要变换）
   p_cam_hawor = R_h2c @ p_hawor + t_h2c

Step 4: 对齐（在相机坐标系里，只剩尺度差）
   p_cam_aligned = scale_ratio * p_cam_hawor

Step 5: 变换回 — 相机坐标系 → GLB 初始坐标系（= 仿真坐标系）
   p_sim = R_c2w_glb @ p_cam_aligned + t_c2w_glb

Step 6: 输出到同一文件夹
   ├── transform_params.npz
   ├── final_scene.glb（原始 GLB，不动）
   └── alignment_report.txt
   → SAPIEN 直接读取运行
```

---

## 2. 每一步的数学

### Step 1: 提取相机位姿

**1.1 加载 RAS 第一帧外参（Z-UP）**
```python
# extrinsics/0.txt 是 w2c [R_w2c | t_w2c]，Z-UP 坐标系
ext = np.loadtxt('extrinsics/0.txt')
if ext.shape == (3, 4):
    ext = np.vstack([ext, [0, 0, 0, 1]])
R_w2c_zup = ext[:3, :3]
t_w2c_zup = ext[:3, 3]

# c2w（相机在 Z-UP 世界里的位姿）
R_c2w_zup = R_w2c_zup.T
t_c2w_zup = -R_c2w_zup.T @ t_w2c_zup
```

**1.2 检测 GLB up-axis**
```python
# 用 GLB 顶点分布判断：地板在 z=0 还是 y=0
glb_up_axis = _detect_glb_up_axis(all_vertices)  # "z-up" 或 "y-up"
```

**1.3 把相机位姿转换到 GLB 坐标系下**
```python
if glb_up_axis == "y-up":
    # GLB 导出时做了 z→y，相机也跟着转
    R_c2w_glb = ZUP_TO_YUP @ R_c2w_zup @ ZUP_TO_YUP.T
    t_c2w_glb = ZUP_TO_YUP @ t_c2w_zup
else:  # z-up
    R_c2w_glb = R_c2w_zup
    t_c2w_glb = t_c2w_zup

# w2c（反变换用）
R_w2c_glb = R_c2w_glb.T
t_w2c_glb = -R_c2w_glb.T @ t_c2w_glb
```

### Step 2: GLB → 第一帧相机坐标系（反变换）
```python
p_cam_glb = R_w2c_glb @ p_glb + t_w2c_glb
```

### Step 3: HaWoR → 第一帧相机坐标系

HaWoR 的 `R_c2w, t_c2w` 在 Render World（OpenGL 约定），和 GLB 坐标系不一样，要变到第一帧相机坐标系：
```python
# HaWoR 第一帧 c2w（OpenGL, Render World）
R_c2w_hawor_0, t_c2w_hawor_0  # 来自 npz

# w2c（OpenGL）
R_w2c_hawor = R_c2w_hawor_0.T
t_w2c_hawor = -R_c2w_hawor_0.T @ t_c2w_hawor_0

# HaWoR 相机是 OpenGL 约定，RAS 相机是 OpenCV 约定，统一到 OpenCV
R_h2c = OPENCV_TO_OPENGL @ R_w2c_hawor
t_h2c = OPENCV_TO_OPENGL @ t_w2c_hawor

# HaWoR 数据 → 第一帧相机坐标系
# 注意：pred_trans 在 SLAM World，先 R_x 到 Render World
p_hawor_render = R_X @ p_hawor_slam  # pred_trans → Render World
p_cam_hawor = R_h2c @ p_hawor_render + t_h2c
```

### Step 4: 对齐（尺度校正）

在相机坐标系里，GLB 和 HaWoR 都以第一帧相机为原点，只剩尺度差。用相机轨迹的标准差比：
```python
# RAS 相机轨迹在相机坐标系（第一帧为原点，[0] = 0）
traj_ras = [R_w2c_glb @ t_c2w_ras[i] + t_w2c_glb for i in frames]
# HaWoR 相机轨迹在相机坐标系（第一帧为原点，[0] = 0）
traj_hawor = [R_h2c @ t_c2w_hawor[i] + t_h2c for i in frames]

sigma_ras = std(traj_ras)
sigma_hawor = std(traj_hawor)
scale_ratio = sigma_ras / sigma_hawor   # <1 表示 RAS/GLB 更小，要把 HaWoR 缩小

# 对齐：把 HaWoR 缩放到 GLB 尺度
p_cam_aligned = scale_ratio * p_cam_hawor
```

静态相机回退：`sigma < 0.01` 时用「手-GLB 距离 ≈ 0.15m」反推 scale_ratio（同 01 的启发式）。

### Step 5: 变换回 GLB 初始坐标系（= 仿真坐标系）
```python
# HaWoR 数据 → 仿真坐标系
p_sim = R_c2w_glb @ p_cam_aligned + t_c2w_glb
      = R_c2w_glb @ (scale_ratio * (R_h2c @ p_hawor_render + t_h2c)) + t_c2w_glb
```

合并成三个参数（供下游使用）：
```python
s_h2s = scale_ratio                                    # 标量
R_h2s = R_c2w_glb @ R_h2c                              # 旋转
t_h2s = scale_ratio * (R_c2w_glb @ t_h2c) + t_c2w_glb   # 平移

# 最终公式: p_sim = s_h2s * R_h2s @ p_hawor_render + t_h2s
```

GLB 顶点变换回（验证用，结果应该 = 原始 GLB）：
```python
p_sim_glb = R_c2w_glb @ p_cam_glb + t_c2w_glb  # = p_glb（恒等）
```

### Step 6: 输出到同一文件夹

所有产物输出到 `--output_dir`（默认 `./output/alignment_001`）：
```
output_dir/
├── transform_params.npz    # 变换参数
├── final_scene.glb         # 原始 GLB（复制过来，仿真直接读）
└── alignment_report.txt    # 报告
```

---

## 3. 变换参数（transform_params.npz）

```python
np.savez(params_path,
    # HaWoR → 仿真坐标系的变换
    s_h2s=s_h2s, R_h2s=R_h2s, t_h2s=t_h2s,
    # GLB 在仿真坐标系里就是原始顶点
    glb_up_axis=glb_up_axis,
    # 尺度诊断
    scale_ratio=scale_ratio,
    sigma_ras=sigma_ras, sigma_hawor=sigma_hawor,
    # 第一帧相机在仿真坐标系里的位姿（下游相机放置用）
    R_c2w_ras=R_c2w_glb, t_c2w_ras=t_c2w_glb,
    # HaWoR 第一帧相机（调试用）
    R_c2w_hawor=R_c2w_hawor_0, t_c2w_hawor=t_c2w_hawor_0,
    # 中间量（调试用）
    R_h2c=R_h2c, t_h2c=t_h2c,
)
```

**下游使用公式**:
- GLB 顶点 → 仿真坐标系: `p_sim = p_glb`（直接用原始 GLB）
- HaWoR 相机位置 `t_c2w_hawor` → 仿真坐标系: `p_sim = s_h2s * R_h2s @ t_c2w_hawor + t_h2s`
- HaWoR 手位置 `pred_trans` → 仿真坐标系: `p_sim = s_h2s * R_h2s @ (R_X @ pred_trans) + t_h2s`
- HaWoR 手朝向 `pred_rot` → 仿真坐标系: `R_sim = R_h2s @ R_X @ axis_angle_to_matrix(pred_rot)`

---

## 4. 文件结构

新建文件: `001_align_scene_camera.py`

```
001_align_scene_camera.py
├── 常量 (ZUP_TO_YUP, OPENCV_TO_OPENGL, R_X)
├── _detect_glb_up_axis(all_vertices)        # 复用 01
├── _load_ras_extrinsics(ras_output)          # 加载 RAS 外参
├── _extrinsics_to_glb_frame(...)             # 外参 → GLB up-axis
├── _compute_h2c(R_c2w_hawor_0, t_c2w_hawor_0)
├── _compute_scale_ratio(traj_ras, traj_hawor, ...)
├── _verify_alignment(...)
├── compute_and_save_transform_params(ras_output, hawor_reconstruction, output_dir, force_scale=None)
└── main()
```

不修改 `01_align_scene.py` 和下游脚本（02/04）。

---

## 5. 实施任务

### Task 1: 创建 001 文件骨架
- 文件: `001_align_scene_camera.py`
- 内容: docstring、import、常量
- 验证: `python -c "import ast; ast.parse(open('001_align_scene_camera.py').read())"`

### Task 2: 实现 `_detect_glb_up_axis`
- 从 01 复制，不改逻辑
- 验证: 合成 Z-UP 和 Y-UP 点云测试

### Task 3: 实现 `_load_ras_extrinsics`
- 加载 `extrinsics/*.txt`，返回 `t_c2w_zup[N,3], R_c2w_zup[N,3,3]`
- 验证: 第一帧和 01 输出对比一致

### Task 4: 实现 `_extrinsics_to_glb_frame`
- 输入: `R_c2w_zup, t_c2w_zup, glb_up_axis`
- 输出: `R_c2w_glb, t_c2w_glb`
- 实现: 见 Step 1.3
- 验证: `R_c2w_glb` 正交（`det≈1`, `R@R.T≈I`）

### Task 5: 实现 `_compute_h2c`
- 输入: `R_c2w_hawor_0, t_c2w_hawor_0`
- 输出: `R_h2c, t_h2c`
- 实现: 见 Step 3
- 验证: `R_h2c @ t_c2w_hawor_0 + t_h2c ≈ 0`（第一帧相机在原点）

### Task 6: 实现 `_compute_scale_ratio`
- 输入: 两条相机轨迹（在相机坐标系里）
- 输出: `scale_ratio, sigma_ras, sigma_hawor, is_static`
- 静态回退: 手-GLB 距离 ≈ 0.15m 反推
- 验证: `scale_ratio ≈ 1/s_inv`（和 01 对比）

### Task 7: 实现主流程 `compute_and_save_transform_params`
- 按 Step 1–6 顺序组装
- 计算 `s_h2s, R_h2s, t_h2s`
- 复制 `final_scene.glb` 到 output_dir
- 保存 npz
- 验证: 文件生成，键齐全

### Task 8: 实现验证 `_verify_alignment`
- 相机轨迹残差: `hawor_cam_sim[i] - ras_cam_sim[i]`，median < 0.1m
- 手-GLB 距离: cKDTree 查询，min < 0.10m
- 方向点积: 相机→GLB中心 · forward > 0
- 验证: 全部通过

### Task 9: 实现 `main()` CLI
- 参数: `--ras_output`, `--hawor_reconstruction`, `--output_dir`, `--force_scale`
- 验证: CLI 跑通

### Task 10: 端到端验证
- 真实数据跑一遍
- 和 01 对比相机轨迹残差、手-GLB 距离
- 验证: 001 不劣于 01

---

## 6. 验证标准（Definition of Done）

| 检查项 | 通过标准 |
|---|---|
| 相机轨迹残差 median | < 0.1m |
| 相机轨迹残差 max | < 0.3m |
| 手-GLB 最近顶点 min | < 0.10m |
| 相机→GLB 方向点积 | > 0 |
| `R_h2s` 正交性 | `‖R@R.T - I‖ < 1e-6` |
| `scale_ratio` 范围 | [0.01, 10] |
| 输出文件夹包含 | `transform_params.npz`, `final_scene.glb`, `alignment_report.txt` |

---

## 7. 下游影响（本计划不实施）

下游脚本（02/04）当前用 `s_inv, R_inv, t_inv` 把 GLB 变到 SAPIEN。001 改成「GLB 不动，HaWoR 数据变到 GLB 初始坐标系」，所以 02/04 之后需要更新：
- GLB 直接用原始顶点
- HaWoR 手/相机用 `s_h2s, R_h2s, t_h2s` 变换
- 不再需要 `RXWORLD_TO_SAPIEN`

下游更新是单独任务，本计划只做 001。

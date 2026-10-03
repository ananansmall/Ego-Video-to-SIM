# View — 可视化与分析工具集

MANO 手部重定向的渲染、可视化、对齐分析和验证工具。所有脚本支持命令行调用。

---

## 目录结构

```
View/
├── utils.py                      # 公共工具函数
├── compute_thumb_tips_depth.py   # 拇指指尖位置计算（深度对齐版）
├── kabsch_analysis.py            # Kabsch 最优旋转对齐分析
├── self_check_alignment.py       # 对齐质量自检
├── vis_camera_trajectories.py    # 相机轨迹 GLB 空间对比
├── render_quick.py               # 对齐对比渲染（输出 GLB，通用多例子）
├── render_viewer.py              # 交互式 3D 查看器（输出 HTML，含 --mesh 手部网格）
├── viz_mano_retargeting_web.py   # 交互式单帧夹爪位姿对比（4种方法）
├── viz_mano_multiframe.py        # 多帧夹爪重定向对比（3×3 网格）
├── traj_plots/                   # 轨迹图输出
├── videos/                       # 视频输出
└── README.md                     # 本文档
```

---

## 文件说明

### `utils.py` — 公共工具函数

被 `viz_mano_retargeting_web.py` 和 `viz_mano_multiframe.py` 共用。

| 导出 | 说明 |
|------|------|
| `MANO_SKELETON` | MANO 手部骨骼连接关系 (20条边) |
| `FINGER_TIP` | 指尖关节索引 `[4, 8, 12, 16, 20]` |
| `load_hawor_data(hawor_dir, hand_idx, frame_idx)` | 加载 HaWoR 重建数据（优先深度对齐版） |
| `matrix_from_compact_axis_angle(aa)` | 紧凑轴角 → 3×3 旋转矩阵 (Rodrigues) |
| `compute_mano_joints(rot, hp, trans, betas)` | MANO 前向运动学，返回顶点和关节点 |

---

### 1. `compute_thumb_tips_depth.py` — 拇指指尖计算

使用深度对齐的 HaWoR 数据计算拇指指尖（joint 20）的 3D 位置。

```bash
python3 View/compute_thumb_tips_depth.py \
    --hawor-dir /path/to/hawor/data \
    --depth-npz /path/to/depth_aligned.npz \
    --output /path/to/thumb_tips_depth.npz
```

---

### 2. `kabsch_analysis.py` — Kabsch 旋转对齐分析

对每个例子执行 Umeyama（Kabsch）最优旋转+尺度+平移对齐，比较最优旋转与当前 `R_hand_to_glb` 的差异，分解缺失旋转轴和角度。

```bash
python3 View/kabsch_analysis.py
```

> 数据路径硬编码在脚本的 `EXAMPLES` 字典中，需根据实际数据路径修改。

---

### 3. `self_check_alignment.py` — 对齐质量自检

验证 GLB ↔ HaWoR 对齐精度，包括坐标系约定、第一帧对齐、轨迹 PCA 方向一致性、90° 旋转组合测试。

```bash
python3 View/self_check_alignment.py
```

> 数据路径硬编码在脚本的 `EXAMPLES` 字典中。

---

### 4. `vis_camera_trajectories.py` — 相机轨迹对比

在 GLB 空间（Z-UP）下对比 HaWoR 和 RAS 相机轨迹，输出各轴范围、距离误差、逐帧位置差。

```bash
python3 View/vis_camera_trajectories.py \
    --hawor_dir /path/to/hawor/data \
    --ras_dir /path/to/ras/data \
    [--params_path /path/to/transform_params.npz]
```

- `--params_path` 可选，默认使用 `{hawor_dir}/output/alignment/transform_params.npz`
- 输出 PNG 图到当前目录

---

### 5. `render_quick.py` — HOI4D 对齐对比渲染

将原始对齐与深度对齐的手部 mesh 叠加到同一个 GLB 视图中，方便对比对齐质量。

```bash
python3 View/render_quick.py \
    --hawor-dir /path/to/hawor/hoi4d \
    --ras-dir /path/to/ras/hoi4d1_vggt_omega \
    --output /tmp/hoi4d_v3 \
    [--show-depth-cloud]
```

- 输出 GLB 文件，可用任何 3D 查看器打开

---

### 6. `render_viewer.py` — 交互式 3D 查看器（含 MANO 手部网格）

基于 Plotly 的交互式浏览器查看器，展示对齐后相机轨迹 + 手部根轨迹 + GLB 物体包围盒。
开启 `--mesh` 后额外按帧渲染完整 MANO 手部网格（Raw 与 Depth 对比）。变换链与 `render_quick.py` 保持一致。

**默认（轻量，无需 torch）**：

```bash
python3 View/render_viewer.py \
    --hawor-dir /path/to/hawor \
    --ras-dir /path/to/ras \
    [--output /path/to/output.html] \
    [--axis-scale 0.5] \
    [--camera-scale 0.05]
```

**带 MANO 手部网格（需要 torch + mano_layer）**：

```bash
python3 View/render_viewer.py \
    --hawor-dir /path/to/hawor \
    --ras-dir /path/to/ras \
    --mesh --step 10 --hands right \
    [--depth-npz /path/to/depth_aligned.npz] \
    [--output /path/to/output_mesh.html]
```

| 参数 | 说明 |
|------|------|
| `--mesh` | 开启 MANO 手部网格渲染（默认关闭，避免加载 MANO 模型） |
| `--step N` | 每隔 N 帧渲染一次手部网格（默认 10） |
| `--hands {left,right,both}` | 渲染哪只手（默认 `right`） |
| `--depth-npz` | 深度对齐版 HaWoR npz；提供后与 Raw 网格并排对比（灰色=Raw，金色=Depth） |

- 输出交互式 HTML，在浏览器中旋转/缩放查看
- 此文件已合并原 `render_hand_mesh.py` 的能力（2026-08 重组）

---

### 8. `viz_mano_retargeting_web.py` — 交互式单帧夹爪位姿对比

Plotly 交互式 3D 视图，对比 3 种夹爪姿态定义方法在同一帧的差异。

```bash
python3 View/viz_mano_retargeting_web.py /path/to/hawor/data [frame_index]
```

- 输出 `viz_mano_retargeting.html`（交互式）

---

### 9. `viz_mano_multiframe.py` — 多帧夹爪重定向对比

3×3 网格对比 3 个不同帧、4 种夹爪重定向方法，直观展示不同方法的差异。

```bash
python3 View/viz_mano_multiframe.py /path/to/hawor/data
```

- 输出 `viz_mano_multiframe.html`（交互式）

---

## 工作流程建议

```
数据准备:
  compute_thumb_tips_depth.py  →  thumb_tips_depth.npz

对齐验证:
  kabsch_analysis.py           →  分析最优旋转对齐
  self_check_alignment.py      →  验证对齐精度
  vis_camera_trajectories.py   →  对比相机轨迹

渲染输出:
  render_quick.py              →  GLB 文件（离线查看，对齐诊断最全）
  render_viewer.py             →  HTML 交互式查看器（相机+手根，可选 --mesh 手部网格）

夹爪可视化:
  viz_mano_retargeting_web.py  →  单帧交互式对比
  viz_mano_multiframe.py       →  多帧对比
```

---

## 附录 A：目录重组记录（原 REORGANIZE.md）

本目录曾存在较多重复代码（5 个文件共享 `load_ras_extrinsics` / `umeyama_full` /
`compute_metrics` 等函数）。重组方案为**提取公共工具模块**：

- `viz_mano_retargeting.py`、`viz_mano_retargeting.html`、`viz_mano_multiframe.html`
  被 `viz_mano_retargeting_web.py` / `viz_mano_multiframe.py` 取代或重建；
- `compute_thumb_tips.py`（无深度修正）被 `compute_thumb_tips_depth.py` 取代；
- `verify_kabsch3.py`、`plot_trajectory_comparison.py`（一次性脚本）已删除；
- `vis_camera_trajectories.py` 的相机轨迹对比功能可并入 `self_check_alignment.py`，
  当前仍独立保留以兼容既有调用。
- `render_hand_mesh.py` 的能力已合并进 `render_viewer.py`（通过 `--mesh` 开关启用），
  原文件已删除（2026-08 重组）。

---

## 附录 B：对齐与渲染开发笔记（原 SESSION_SUMMARY.md）

### B.1 关键问题与设计决策

- **PCA 指标不适用于 hoi4d**：hoi4d 轨迹主要在 XY 平面展开、Z 几乎不动，
  PCA 角度反映的是主轴方向差异而非对齐误差；正确指标为
  Forward 夹角（mean≈10.3°）与位置距离（mean≈0.2m）。
- **RAS depth center 偏移前后颜色冲突（已修复）**：
  原 `_offset_mask_traj` 与 `mask_center_traj` 被赋予相同颜色，
  已新增 `ras_depth_ctr_off`（brown）区分原始/偏移后。
- **偏移逻辑 bug（已修复）**：先保存偏移轨迹再修改原轨迹会导致两者相同，
  已改为先保存 `_offset_mask_traj = original + offset` 再修改 `mask_center_traj`。
- **轨迹精简**：用拇指尖(j20)+食指尖(j16)中点替代单独拇指尖渲染，去除手腕原始轨迹。

### B.2 坐标系约定

- RAS：Z-UP，`t_c2w` = 相机世界坐标
- HaWoR → GLB：`p_glb = s * R_hand_to_glb @ p_hawor + t_hand_to_glb`
- Umeyama 估计的 `scale_ratio`：GLB 单位 / HaWoR 单位

### B.3 `render_quick.py` 图例与帧偏移示例

| 颜色 | 含义 |
|------|------|
| navy | HaWoR 相机轨迹 |
| limegreen | RAS 相机轨迹 |
| red / dodgerblue | L 中点 Raw / Depth（拇指+食指） |
| darkorange / gold | R 中点 Raw / Depth（拇指+食指） |
| cyan / brown | RAS depth center（原始 / 偏移后） |
| pink / turquoise / magenta / yellow | 偏移后各中点轨迹 |
| green | depth center 3D 点云 |

帧偏移示例（第 50 帧）：相机位置 (-0.296, -0.899, 0.500)，
粉色物体 toy_0 质心 (-0.302, -0.497, 0.037)，
offset = (-0.006, +0.403, -0.464) m（距离 0.614 m）。

### B.4 下一步建议

1. 验证偏移效果：在 HTML 中观察偏移后 cyan(brown) 线是否更靠近粉色物体；
2. 批量渲染：将 3chair-bottle、hoi4d 等更多例子加入对比；
3. 深度视频：`videos/` 目录已有 121、7、hoi4d 的深度视频，可继续生成。

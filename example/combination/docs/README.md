# combination/ 顶层文档索引

> 原则：**文档名与对应脚本文件名一致**（如 `04_physics_simulation.md` ↔ `04_physics_simulation.py`）。
> 顶层总览见根目录 [../README.md](../README.md)；本目录只放跨模块/管线层的文档。

## 目录结构

```
docs/
├── README.md               本索引
├── system_architecture.md  顶层架构
├── COMMANDS.md             命令手册
├── CHANGE_LOG.md           变更日志（现役）
├── pipeline/               编号脚本（00~07）管线层文档，文件名对应脚本名
│   └── specs/              管线各脚本的设计/计划 spec（按日期命名）
├── experiments/            实验、问答、状态记录（文件名前缀对应所属脚本）
└── archive/                历史归档（2026-06 时代，只读，不再更新）
```

## 顶层总览

| 文件 | 日期 | 说明 |
|---|---|---|
| [system_architecture.md](system_architecture.md) | 2026-07-11 | Dex-Retargeting 管线系统架构分析 |
| [COMMANDS.md](COMMANDS.md) | 2026-08-04 | 各脚本调用方式大全 |
| [CHANGE_LOG.md](CHANGE_LOG.md) | 2026-08-10 | 全项目变更日志（现役） |

## pipeline/ — 管线层（文件名 ↔ 脚本名）

| 文档 | 对应脚本 | 说明 |
|---|---|---|
| [00_run_pipeline.md](pipeline/00_run_pipeline.md) | `00_run_pipeline.py` | 手部→机器人映射管线总览（01→02→04 全链路） |
| [01_align_scene.md](pipeline/01_align_scene.md) | `01_align_scene.py` / `001_align_scene.py` | RAS GLB → HaWoR 坐标系对齐完整指南 |
| [01_align_scene_howto.md](pipeline/01_align_scene_howto.md) | `01_align_scene.py` | GLB → HaWoR 对齐操作指南（含实际路径示例） |
| [002_render_scene_plan.md](pipeline/002_render_scene_plan.md) | `002_render_scene.py` | 002 渲染脚本实施方案 |
| [04_physics_simulation.md](pipeline/04_physics_simulation.md) | `04_physics_simulation.py` | 04 物理仿真脚本说明（8月版，以此为准） |
| [05_gripper_test.md](pipeline/05_gripper_test.md) | `05_gripper_test.py` | PhysX 驱动管线（夹爪物理仿真测试主文档） |

### pipeline/specs/ — 设计与计划（按日期命名）

| 文件 | 说明 |
|---|---|
| [specs/001_align_scene_camera_plan.md](specs/001_align_scene_camera_plan.md) | 001 相机坐标系对齐方案（8月） |
| [specs/2026-06-25-camera-view-and-bimanual-design.md](specs/2026-06-25-camera-view-and-bimanual-design.md) | 相机视角与双手设计 |
| [specs/2026-06-25-camera-view-and-bimanual-plan.md](specs/2026-06-25-camera-view-and-bimanual-plan.md) | 相机视角与双手计划 |
| [specs/2026-06-26-04-physics-glb-and-gripper-design-v2.md](specs/2026-06-26-04-physics-glb-and-gripper-design-v2.md) | 04 物理 GLB 与夹爪设计 v2 |
| [specs/2026-06-26-glb-alignment-and-gripper-mirror-design.md](specs/2026-06-26-glb-alignment-and-gripper-mirror-design.md) | GLB 对齐与夹爪镜像设计 |
| [specs/2026-06-26-glb-alignment-and-gripper-mirror-plan.md](specs/2026-06-26-glb-alignment-and-gripper-mirror-plan.md) | GLB 对齐与夹爪镜像计划 |

## experiments/ — 实验与问答（前缀 ↔ 所属脚本）

| 文件 | 对应脚本/主题 | 说明 |
|---|---|---|
| [experiments/05_gripper_test_status.md](experiments/05_gripper_test_status.md) | `05_gripper_test.py` | 单夹爪物理仿真完整总结（2026-07-14） |
| [experiments/05_gripper_test_test9.md](experiments/05_gripper_test_test9.md) | `05_gripper_test.py::test9_mano_trajectory` | MANO 轨迹跟随开发总结 |
| [experiments/05_gripper_test_dev_context.md](experiments/05_gripper_test_dev_context.md) | `05_gripper_test.py` | 开发上下文恢复入口（test8~test11 脉络，2026-08-28） |
| [experiments/question.md](experiments/question.md) | — | Q&A 记录（6月） |
| [experiments/questions.md](experiments/questions.md) | — | Q&A 记录（8月） |
| [experiments/questions_simulation_202606.md](experiments/questions_simulation_202606.md) | — | 仿真问题记录（6月） |
| [experiments/context/四重平滑与对话问题总结.md](experiments/context/四重平滑与对话问题总结.md) | 平滑相关 | 四重平滑问题 + render_quick 现状总结 |

## archive/ — 历史归档（只读，不改名）

| 文件 | 说明 |
|---|---|
| [archive/changelog_202606.md](archive/changelog_202606.md) | 管线修改记录（6月版） |
| [archive/changelog_sim_202606.md](archive/changelog_sim_202606.md) | 仿真变更日志（6月版，现役版的旧子集） |
| [archive/2026-06-26-all-changes.md](archive/2026-06-26-all-changes.md) | 2026-06-26 全部改动汇总 |
| [archive/04_physics_simulation_202606.md](archive/04_physics_simulation_202606.md) | 04 仿真说明 6月版（与 pipeline/ 下 8月版内容相同，保留备查） |
| [archive/pipeline.md.bak](archive/pipeline.md.bak) | pipeline.md 旧备份 |
| [archive/2026-07-05_render-dual-robot-viewer.md](archive/2026-07-05_render-dual-robot-viewer.md) | 双臂渲染 viewer 会话记录（原 SESSIONS/） |

## 模块层文档（在各模块内，不在本目录）

| 模块 | 文档位置 | 状态 |
|---|---|---|
| `trajectory_optimization/`（主项目） | `README.md`、`CHANGE_LOG.md`、`docs/`（含 `docs/trimesh/` 8 篇设计文档） | 现役 |
| `hand_track/`（共享库） | `README.md`、`CHANGE_LOG.md`、`docs/questions.md` | 现役 |
| `View/`（可视化） | `README.md` | 现役 |
| `tri_model_physics/`（旧 fork） | `README.md`、`CHANGE_LOG.md`、`docs/` | 历史 |
| `physics_pipeline/`（pybullet 方案） | `README.md`、`docs/questions.md` | 已废弃 |

**注意**：`tri_model_physics/docs/` 与 `trajectory_optimization/docs/trimesh/` 有 6 个同名文件
（2026-06-30 ~ 2026-07-10 的设计/计划文档），是旧 fork 遗留的分叉副本，互不同步——
**以 `trajectory_optimization/docs/trimesh/` 为准**。

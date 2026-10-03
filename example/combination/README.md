# combination/ — 手部重定向 → 物理仿真实验区

基于 [dex-retargeting](../../) 的实验目录：MANO 手部轨迹 → 机器人手/夹爪重定向 → 场景对齐 → 物理仿真。

**主项目：[trajectory_optimization/](trajectory_optimization/)**（夹爪轨迹优化，MuJoCo/SAPIEN，现役）。
目录地图与文档索引如下；详细文档见 [docs/README.md](docs/README.md)。

## 目录地图

```
combination/
├── 00_run_pipeline.py         ★ 一键管线入口（编排 01c/001/01/002/02/04）
├── 0x/05x/06x/07_*.py         编号脚本：对齐(01/001/01c) → 渲染(002/02) → 跟踪(03)
│                              → 物理仿真(04) → 夹爪测试(05×3) → 抓取/验证(06×2/07)
├── hand_track/                共享库：手部检测 + 机械臂/夹爪渲染（被 10+ 处 import）
├── trajectory_optimization/   ★ 主项目：夹爪轨迹优化（grasp_hawor.py / eval_stage1_v5.py）
├── View/                      可视化工具集（渲染、对齐分析、轨迹绘图）
├── physics_pipeline/          pybullet 物理管线（已废弃，仅存档）
├── tri_model_physics/         旧 fork（6月代，已被 trajectory_optimization 取代）
├── tools/                     MANO 轨迹平滑工具（smooth_mano.py）
├── test/ · test_compare/      散装调试/可视化脚本
├── docs/                      顶层文档（索引见 docs/README.md）
├── output/ 等 *_output/       生成产物（视频/npz/日志）
└── .opencode/ · graphify-out/ 工具缓存（不进 git）
```

## 运行入口

```bash
# 一键管线（需 conda env dex；数据默认 /home/an/data/hawor/...）
python 00_run_pipeline.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result

# 主项目：夹爪轨迹优化
cd trajectory_optimization
python grasp_hawor.py          # 抓取仿真
python eval_stage1_v5.py       # stage1 评估

# 调用方式大全
# → docs/COMMANDS.md
```

## 文档导航（文档名 ↔ 对应文件）

| 文档 | 对应文件 |
|---|---|
| [docs/pipeline/00_run_pipeline.md](docs/pipeline/00_run_pipeline.md) | `00_run_pipeline.py` 全链路总览 |
| [docs/pipeline/01_align_scene.md](docs/pipeline/01_align_scene.md) | `01_align_scene.py` / `001_align_scene.py` 对齐原理 |
| [docs/pipeline/01_align_scene_howto.md](docs/pipeline/01_align_scene_howto.md) | `01_align_scene.py` 操作指南 |
| [docs/pipeline/002_render_scene_plan.md](docs/pipeline/002_render_scene_plan.md) | `002_render_scene.py` |
| [docs/pipeline/04_physics_simulation.md](docs/pipeline/04_physics_simulation.md) | `04_physics_simulation.py` |
| [docs/pipeline/05_gripper_test.md](docs/pipeline/05_gripper_test.md) | `05_gripper_test.py` 主文档 |
| [docs/experiments/05_gripper_test_status.md](docs/experiments/05_gripper_test_status.md) | `05_gripper_test.py` 状态总结 |
| [docs/experiments/05_gripper_test_test9.md](docs/experiments/05_gripper_test_test9.md) | `05_gripper_test.py::test9_mano_trajectory` |
| [docs/experiments/05_gripper_test_dev_context.md](docs/experiments/05_gripper_test_dev_context.md) | `05_gripper_test.py` 开发上下文（test8~test11） |
| [docs/COMMANDS.md](docs/COMMANDS.md) / [docs/system_architecture.md](docs/system_architecture.md) | 全局命令 / 架构 |
| [docs/README.md](docs/README.md) | 完整文档索引（含模块层文档、归档） |

## 注意事项

- `physics_pipeline/`（pybullet 方案）**已废弃**；`tri_model_physics/` 为旧 fork，二者勿与主项目混用。
- 多数脚本依赖：conda env `dex`、`example/position_retargeting/`（mano_layer）、外部仓库 `/home/an/robot_world_ws/src/GalaxeaManipSim`（galaxea_sim）。
- 生成产物（output/、*.png、*.mp4、__pycache__ 等）不入 git。

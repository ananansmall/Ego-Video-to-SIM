# ReplicateAnyScene 文档索引

## pipeline/ — 代码技术文档（与当前代码直接对应）

| 文档 | 对应代码 |
|------|---------|
| [mainv2_technical_doc.md](pipeline/mainv2_technical_doc.md) | mainv2.py 全流程、GLB 体系、命令行参数 |
| [stage5_refinement_analysis.md](pipeline/stage5_refinement_analysis.md) | Stage 5 精修管线：问题排查、修复方案与完整逻辑（2026-08-25） |
| [VGGT_models_and_analysis.md](pipeline/VGGT_models_and_analysis.md) | VGGT / VGGT-Omega / VGGT4D 三模型对比与点云质量影响 |

## design/ — 模块设计与集成方案

| 文档 | 说明 |
|------|------|
| [stage6_physics_sim.md](design/stage6_physics_sim.md) | **Stage 6 权威文档**: URDF 生成 → SAPIEN 物理仿真 → pyrender 渲染 GIF。含材质标注（附录 A）、URDF 生成要点（附录 B）、10 个数据集验证表 |
| [stage6_design.md](design/stage6_design.md) | ~~Stage 6 材质标注设计~~ 已合并至 `stage6_physics_sim.md` 附录 A（存根，[原文归档](design/archive/stage6_design_v1.md)） |
| [stage61_urdf_plan.md](design/stage61_urdf_plan.md) | ~~Stage 6.1 URDF 生成方案~~ 已合并至 `stage6_physics_sim.md` 附录 B（存根，[原文归档](design/archive/stage61_urdf_plan_v1.md)） |
| [MERGE_ARCHITECTURE.md](design/MERGE_ARCHITECTURE.md) | HaWoR + ReplicateAnyScene 合并架构方案 |
| [coordinate_and_alignment.md](design/coordinate_and_alignment.md) | GLB → HaWoR 坐标系对齐操作指南 |

## research/ — 研究路线图与论文笔记

| 文档 | 性质 |
|------|------|
| [EGO_VIDEO_TO_SIM_ROADMAP.md](research/EGO_VIDEO_TO_SIM_ROADMAP.md) | Ego-Video-to-Sim 总路线图 v3.1 |
| [REFERENCED_PAPERS_DETAIL.md](research/REFERENCED_PAPERS_DETAIL.md) | 路线图配套核心论文深度解读 |
| [ForeHOI_reference_analysis.md](research/ForeHOI_reference_analysis.md) | ForeHOI 论文参考价值分析 |
| [SimFoundry_PROSE_reference_analysis.md](research/SimFoundry_PROSE_reference_analysis.md) | SimFoundry / PROSE 论文对比分析 |

## 根目录

- [questions.md](questions.md) — 项目 Q&A 持续记录（qa-log 维护）
- ../CHANGE_LOG.md — 变更日志（change-log 维护）

> 已删除: stage5_physics_engine_proposal.md（其问题分析基于旧版穿模实现 cKDTree/max_iterations=1，已被 FCL 方案取代，物理引擎方向并入 stage5_refinement_analysis.md 遗留观察项）

# Stage 6.1 URDF 生成方案（已合并）

> ⚠️ **本文档已合并，请阅读 → [stage6_physics_sim.md](stage6_physics_sim.md)**
> 📦 **原文全文已原样归档（未被覆盖）→ [archive/stage61_urdf_plan_v1.md](archive/stage61_urdf_plan_v1.md)**

## 为什么不保留两份

本文档（2026-07-30 计划版）是 **URDF 生成** 的详细设计，"Stage 6.1" 这个名字是当时
为了与"材质标注（Stage 6）"区分而临时起的。现在 Stage 6 已收敛为**物理仿真**这件事，
URDF 生成就是其中的 **步骤 ①**，不需要再单独挂一个编号。

## 内容去向

| 原文章节 | 去向 |
|---|---|
| §整体流程图 / Step 1~4 | `stage6_physics_sim.md` §10.1 |
| §1 文件位置总览 | `stage6_physics_sim.md` §4 |
| §2 冲突分析（EmbodiedGen 集成） | `stage6_physics_sim.md` §10.3（提炼为 5 条踩坑） |
| §3 冲突解决策略总表 | `stage6_physics_sim.md` §10.3 |
| §4 输出目录结构 | `stage6_physics_sim.md` §3 |
| §5~§6 实现要点（Open3D 渲染 / SAPIEN 集成） | 已落到代码，文档不再重复 |
| §7 运行方式 | `stage6_physics_sim.md` §6 |
| §8 测试计划 | ✅ 已全部执行完毕，计划不再保留 |
| §9 实现状态总览 | ✅ 已完成；`mainv2.py --enable_stage61` **至今未实现**，见 §8 已知限制 |

## 需要留档的两条历史事实

1. **`--enable_stage61` 从未实现**。`run_stage61()` 只被自己的 `__main__` 调用，
   `mainv2.py` / `mainv3.py` 全文搜不到 `urdf` / `stage61`。所以 Stage 6 目前
   必须手工两步跑脚本。
2. **文档里的 Python 解释器已过时**。原文写的是
   `conda_envs/ReplicateAnyScene/bin/python`，实际 ① 该用这个环境，但 **②③
   物理仿真+渲染必须用 `conda_envs/sapien`**（见 `stage6_physics_sim.md` §6）。

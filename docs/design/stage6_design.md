# Stage 6 设计方案（已合并）

> ⚠️ **本文档已合并，请阅读 → [stage6_physics_sim.md](stage6_physics_sim.md)**
> 📦 **原文全文已原样归档（未被覆盖）→ [archive/stage6_design_v1.md](archive/stage6_design_v1.md)**

## 为什么不保留两份

本文档（2026-07 版）把 **材质感知物理标注** 当作 Stage 6 的全部职责。但实际落地后，
Stage 6 的主线变成了**物理仿真**（URDF → SAPIEN → 渲染出 GIF），材质标注只是它的
**输入提供方之一**。两份文档口径冲突，容易误导后来人，因此合并为一份。

## 内容去向

| 原文章节 | 去向 |
|---|---|
| §1 目标（材质驱动 vs 参数驱动） | `stage6_physics_sim.md` §9.1 |
| §2 背景（旧硬编码 density/mu/restitution） | `stage6_physics_sim.md` §5 |
| §3.4 `materials_db.json` 完整内容 | 保留在 `tools/materials_db.json`（文档不再复制一份，避免两处不同步） |
| §3.6 输出文件格式 | `stage6_physics_sim.md` §9.3 |
| §3.7 / §6 SAPIEN 集成方式 | `stage6_physics_sim.md` §5（现行实现：直接覆盖 PhysxMaterial） |
| §4 `mainv2.py` 接入点 | `stage6_physics_sim.md` §9（`--enable_stage6`） |
| §5 工具实现结构 | `stage6_physics_sim.md` §9.2 |
| §7 实现优先级 | 已完成，不再保留 |
| §8 扩展性（新材质 / 新平台 / 杨氏模量） | 仍有效：只改 `tools/materials_db.json` 加字段即可，无需改代码 |
| §9 风险提示 | `stage6_physics_sim.md` §9.4 |

## 两处与原设计不一致的实现现状（以实际代码为准）

1. **VLM 调用方式**：原设计是"1 次调用、传场景 ID 标注图"；实际实现是
   **逐实例裁剪图各调 1 次**（`optimal_frames/<cat>_inst<i>_frame<n>.jpg`）。
2. **输出 key 粒度**：实际是**实例级** `bowl_inst0`，不是类别级 `bowl`。

这两点直接导致了历史上"Stage 6 标了但没人消费"的断链，详见
`stage6_physics_sim.md` §5 与 `docs/questions.md` Q30。

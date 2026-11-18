# scripts/ 索引

> 这个目录里混着**三个独立项目**，互不 import，按前缀区分。
> 详细索引见 `../docs/07_文件结构与职责.md`。

| 项目 | 文件 | 说明 |
|---|---|---|
| **粉鸭 pick→place**（本会话主线）| `robotwin_r1_duck_*` / `agentic_*` / `*_config.py` / `task_config.py` / `pipeline.py` | 已全链路验证 |
| **bp14 双臂**（并行编辑者，10-18 仍在改）| `bp14_*.py` | 不属于本任务 |
| **大范围**（并行编辑者，10-19 仍在改）| `cb_large_range_sim.py` / `large_range_track.py` | 不属于本任务 |

## 怎么跑

```bash
# 跑一次仿真（唯一入口）
python scripts/pipeline.py configs/duck_to_plate.json
python scripts/pipeline.py configs/duck_to_plate.json --set place_t=0.3 --render
python scripts/pipeline.py configs/duck_to_plate.json --set ik_seeds=8

# 让 agent 调参（走统一入口）
python scripts/agentic_optimize.py --task configs/duck_to_plate.json --budget 5
python scripts/agentic_agent.py --task configs/duck_to_plate.json --budget 5 --ik-seeds 4

# 旧式命令行（向后兼容，不带 --task）
python scripts/agentic_optimize.py --coacd --base-xy 0.12 -0.40
```

## 目录职责

| 位置 | 放什么 |
|---|---|
| `scripts/*.py` | 核心管线 + 适配层 + agent |
| `scripts/analysis/` | 分析/实验脚本（创新点证据、消融、回归测试）|
| `scripts/agentic_*_frames/` | agent 产出的对比图 |
| `configs/` | **任务配置（换任务只改这里）** |
| `results/` | agent 与分析的 JSON 产物 |
| `docs/` | 文档 |

## 分析脚本（`scripts/analysis/`）

| 脚本 | 产出 |
|---|---|
| `gt_ablation.py` | 消融：verifier 能否预测示范保真度 |
| `gt_cmp.py` | 各配置偏离真值排序 |
| `plot_demo_fidelity.py` | 创新点证据图 `demo_fidelity.png` |
| `verifier_screen.py` | 各 verifier 子项的 AUC / 分辨力 |
| `test_props_collision.py` | 验证"道具不简化"零代价 |
| `run_from_task_json.py` | 纯 JSON 驱动端到端 |
| `ik_benchmark.py` | 4/8/16 种子完整物理评测，分别记录 IK / 总时间与 GT |
| `ik_convergence_probe.py` | 抓取帧早退开/关对照与每个种子的迭代次数 |
| `agent_gt_ab.py` | 从相同差起点、独立空缓存做 prompt GT 输入消融 |
| `coordinate_baseline.py` | 同起点、同 IK、总预算 5 的坐标下降对照 |
| `test_ik_agent.py` | 无物理/VLM 调用的配置、缓存、prompt 与预算回归检查 |
| `probe_pinocchio.py` | 解析运动学和 pad 雅可比对照 SAPIEN 数值差分 |
| `quarantine_cache.py` | 按已识别来源隔离十条合成缓存，保存原始备份与审计 |
| `replay_cache_identity.py` | 任务/GT/碰撞缓存隔离，并用已保存响应与物理指标逐字重放实验 |
| `verify_agent_candidate.py` | GT 组候选以 16 种子完整重放，更新真实验收报告 |

`tuning.ik_seeds` 控制 fixed 后端的 IK 搜索种子数。验收默认 16，agent 搜索默认 4，
可用 `--ik-seeds 8/16` 覆盖。种子数进入缓存 key，旧代码或另一种子数的结果不作为当前历史。
`agentic_agent --without-gt` 隐藏 prompt 中的显式 GT 字段；两组评分函数仍相同且含 GT。
预算包含起点的首次物理评测，缓存命中不计预算；受控实验每组使用独立空缓存。

```bash
python scripts/analysis/test_ik_agent.py
python scripts/analysis/agent_gt_ab.py --arm without_gt --budget 5 --output-dir results/ik_agent_20261022
python scripts/analysis/agent_gt_ab.py --arm with_gt --budget 5 --output-dir results/ik_agent_20261022
```

⚠️ 分析脚本从 `../results/agentic_optimize_cache.json` 读缓存，**从任何目录运行都可以**
（路径已绝对化）。

缓存身份包含任务/GT 上下文、实际碰撞选项、全部核心参数和求解器/adapter 指纹。

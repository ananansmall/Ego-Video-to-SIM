# Debug: opt-sim-divergence

**Status**: [OPEN]
**Started**: 2026-07-04
**Session ID**: opt-sim-divergence

## Bug Description

**Symptom**: CMA-ES 优化 rollout 与最终仿真的物理结果严重不一致。
- 优化 rollout: best_reward=164.511 (8 接触帧, ~2cm 提升, ||offset||=0.479)
- 最终仿真: glb_1 被弹飞到 z=1.666m, xy_drift=98.73cm, 0 接触帧
- 物体重置已验证 drift=0.0000m
- 帧间状态 (_prev_demo_root_pos, _prev_root_pos, _prev_root_pos_vel) 已重置

**Reproduction**:
```
python grasp_hawor.py --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode hybrid --method grasp-lift \
    --cmaes-gen 40 --cmaes-pop 32 --cmaes-sigma 0.04 --num-frames 113
```

## Hypotheses

| ID | Hypothesis | Falsifiable via |
|----|------------|-----------------|
| H1 | SAPIEN scene 累积状态 (接触缓存/broadphase/warm-start) 导致 rollout 间物理行为漂移 | 在优化开始前 vs 第 N 次 rollout 后, 用相同参数 rollout, 对比物体轨迹 |
| H2 | 接触检测差异 (warm-start 改变接触法向/深度) | 记录同一帧在优化 vs 最终仿真的 contacts 数量与法向 |
| H3 | 物体重置不彻底 (velocity 未真正归零, 或碰撞几何缓存未刷新) | 记录物体在 rollout 开始时的线/角速度 |
| H4 | kinematic root 切换时序差异 (lock_root_pose 时序不同) | 记录 root lock 状态切换帧号, 对比优化 vs 最终仿真 |
| H5 | 时序/帧调度差异 (physics_step 调用次数/顺序不同) | 记录每次 rollout 的 physics_step 总调用次数, 对比优化 vs 最终仿真 |

## Investigation Log

### Round 1: 静态分析 + 插桩验证

**静态分析发现**:
| 差异点 | `run_optimize()` | `run()` |
|--------|------------------|---------|
| WARMUP_FRAMES | 无 | 30 帧 |
| force_cpu | True | False |
| HybridGraspController | 无 robot 参数 | 有 robot 参数 |
| robot 初始 qpos | set_qpos(init_qpos) 瞬移 | WARMUP PD 收敛 |

**插桩 1: verify rollout (H1 验证)**
- 5 代优化: best_reward=-1.057, verify_reward=-1.037, diff=0.02
- **H1 被证伪**: 优化 rollout 可重现, 无状态累积

**插桩 2: first_frame 状态对比 (H3/H4 验证)**
| 指标 | verify (修复前) | run() | 差异 |
|------|----------------|-------|------|
| gripper qpos[1] | 0.045140 | -0.007778 | **5.3cm!** |
| gripper qvel[1] | 0.166024 | 0.056567 | **0.11!** |
| obj_pos | 一致 | 一致 | 0 |
| root_pose | 一致 | 一致 | ~0 |

**根因确认**:
1. `init_qpos[gi2] = GRIPPER_INIT_OPEN = 0.04` (gripper_only 模式, L853)
2. WARMUP `target2 = -GRIPPER_INIT_OPEN = -0.04` (L3499)
3. WARMUP 让 qpos[gi2] 从 0.04 收敛到 -0.04
4. `rollout_single` 没有 WARMUP, qpos[gi2] 保持 0.04
5. 两者第一帧初始 qpos[gi2] 不同 (0.04 vs -0.04)
6. `set_qpos` 不重置 `qvel`, 前一 rollout 残留 qvel 影响

### Round 2: 修复 + 验证

**修复**: 在 `rollout_single` 中:
1. `set_qvel(0)` 重置关节速度
2. 添加 WARMUP_FRAMES (30 帧) 对齐 run() 初始化
3. WARMUP 后不重置 root_pose (对齐 run())

**修复后 first_frame 对比**:
| 指标 | verify (修复后) | run() | 差异 |
|------|----------------|-------|------|
| gripper qpos | [0.022177, -0.006649] | [0.022177, -0.006649] | **0** |
| gripper qvel | [-0.026079, 0.099702] | [-0.026079, 0.099702] | **0** |
| root_pose | [-0.1566, -0.2660, 0.1112] | [-0.1566, -0.2660, 0.1112] | **0** |
| obj_pos | [0.0335, -0.1512, 0.0173] | [0.0335, -0.1512, 0.0173] | **0** |

**物体轨迹对比**:
- max diff: 6.4mm (force_cpu 数值精度差异)
- mean diff: 0.1mm
- **基本一致!**

### Round 3: 40 代优化验证 (进行中)

待验证: 40 代优化找到的"激进"参数 (有接触) 在最终仿真中是否也能工作。

## Root Cause

**H3 修正版成立**: `set_qpos` 不重置 `qvel`, 且 `rollout_single` 缺少 WARMUP_FRAMES。

具体:
1. `rollout_single` 用 `set_qpos(init_qpos)` 瞬移, 但 `set_qpos` 不重置 `qvel`, 前一 rollout 的 qvel 残留
2. `init_qpos[gi2] = 0.04` (gripper_only), 但 WARMUP `target2 = -0.04`, run() 的 WARMUP 让 qpos[gi2] 收敛到 -0.04
3. `rollout_single` 无 WARMUP, qpos[gi2] 保持 0.04, 与 run() 的 -0.04 不同
4. 对"温和"参数 (无接触), 差异被 PD 吸收; 对"激进"参数 (有接触), 差异导致接触时序不同, 物体被弹飞

## Fix

在 `rollout_single` 中:
1. `robot.set_qvel(np.zeros_like(robot.get_qvel()))` — 重置关节速度
2. 添加 WARMUP_FRAMES 循环 — 对齐 run() 的 PD 预热
3. WARMUP 后不重置 root_pose — 对齐 run() 的根位姿处理


# Trajectory Optimization

夹爪轨迹优化模块，在 MANO 参考轨迹基础上优化少量参数（3D/6D），使夹爪在物理仿真中接触并抓取物体。

## 核心文件

| 文件 | 功能 |
|------|------|
| `traj_optimize.py` | 优化器核心：3D XYZ CEM、6D XYZ+RPY CEM、奖励函数 |
| `grasp_hawor.py` | 主脚本：加载 HAWOR 数据、跑 SAPIEN 物理仿真、执行优化、渲染结果 |
| `trajectory_loader.py` | 加载 HAWOR 数据，计算 MANO FK → gripper 位姿 |
| `physics_utils.py` | 夹爪参数（摩擦、力、开合范围）、地面高度等物理常量 |
| `grasp_controller.py` | 分阶段抓取控制器（APPROACH/CLOSE/FORCE_CONTROL/LIFT/HOLD/RELEASE） |
| `vis_trajectory.py` | 可视化：MANO vs 优化后轨迹对比图（3D + 距离曲线） |
| `grasp_demo.py` | 演示/测试脚本 |

## 两阶段优化流程

**Phase 1：3D XYZ 偏移**（~26秒）
- 优化 3 个参数 [dx, dy, dz]
- 目标：让夹爪在 CLOSE 阶段接触到物体
- 奖励：接近距离、最小距离、接触帧数
- 结果：contact=12, min_dist=2mm

**Phase 2：6D XYZ+RPY 偏移**（~28秒）
- 从 Phase 1 最优 XYZ 初始化
- 增加 [droll, dpitch, dyaw] 姿态调整
- 目标：稳定抓取并提升物体
- 奖励：在 Phase 1 基础上加入 lift、grasp_success、last_contact
- 结果：lift=3.38m, contact=5, last_contact=3/5

## 运行命令

```bash
python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views none --method grasp-lift
```

## 可视化

```bash
python vis_trajectory.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --opt-params output/gripper_only_left/opt_params.npy \
    --output-dir output/gripper_only_left \
    --hand-idx 0 \
    --save /tmp/traj_compare
```

## 输出目录

`output/gripper_only_left/` 包含优化结果：
- `opt_params.npy` — Phase 2 最优 6D 参数
- `opt_params_phase2.npy` — Phase 2 6D 参数备份
- `reward_history.npy` / `reward_history_phase2.npy` — 奖励收敛曲线
- `baseline_dists.npy` — MANO 基线距离
- `verify_*.npy` — 验证 rollout 轨迹数据
- `grasp.log` — 运行日志
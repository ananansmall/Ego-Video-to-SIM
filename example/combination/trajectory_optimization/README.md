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

## 调用命令（当前标准流程：7a 数据 + 左手 + 物体 ×0.6，2026-09-06）

> **真实操作手 = 左手**（`--side left`，hand_idx=0；右手是 HaWoR 幻觉手，勿用）。
> 物体需缩放 ×0.6（原始 7.86cm > 夹爪指缝，接近段手指会穿插；详见 EXPERIMENT_SUMMARY.md）。
> **每次全新跑**：换 output 目录名或先移走旧目录（stage2/3 会复用 stage1 的 best_grasp.npz）。

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization
export PY=/home/an/miniconda3/envs/dex/bin/python
export _OBJ_SCALE=0.6 _OBJ_SCALE_TARGET=glb_4   # 物体缩放旋钮（去掉则用原始尺寸）

# ── Stage0：001 对齐 + 预处理（开合跟随 MANO）+ 纯跟随基线视频 ──
$PY grasp_hawor.py --test-stage0 --views all --side left \
    --hawor-dir /home/an/data/hawor/7a --output-dir output/hawor7a_left
#    验收: [pre_translate] offset≈0 / 三视角视频 (cam=SLAM第一人称逐帧跟随+god+right)
#    产物: output/hawor7a_left/stage0/*.mp4, preprocess_traj.npz

# ── Stage0 对齐轨迹图（wrist 优化前黑虚/优化后黑实 + 相机轨迹青色）──
$PY View/vis_preprocess_trajectories.py --side left \
    --traj output/hawor7a_left/preprocess_traj.npz \
    --out output/hawor7a_left/stage0/preprocess_traj.png

# ── Stage1：F50 抓取位姿优化（CEM 15×32 + 验证门 3 次重复 + 抓取视频）──
$PY grasp_hawor.py --test-stage1 --views all --side left \
    --hawor-dir /home/an/data/hawor/7a --output-dir output/hawor7a_left
#    验收: [Stage1][验证门] 3/3 通过, lift_final ≈ 20cm
#    产物: stage1/best_grasp.npz + god/right 抓取视频

# （可选）交互式实时窗口：任意命令加 --viewer（需本机显示器）
```

### 可选旋钮（默认值即当前验证配置，一般不用动）
| 环境变量 | 默认 | 说明 |
|---|---|---|
| `_OBJ_SCALE` / `_OBJ_SCALE_TARGET` | 1.0 / 无 | 物体缩放（×0.6 = 4.72cm，适配指缝 5.1cm） |
| `_GRIP_FRICTION` | 2.0 | 夹爪摩擦（3.0 实测更差，勿调） |
| `_S1_GRIP_DELTA` | 0 | stage1 闭合深压（米；0.004 实测未稳定提升） |
| 随机种子 | 42+1000×轮次 | stage1 CEM 已固定，可复现 |

### 换数据/换手
```bash
# 7 号数据 / 右手（历史配置，仅对照用）:
$PY grasp_hawor.py --test-stage1 --views all --side right \
    --hawor-dir /home/an/data/hawor/7 --output-dir output/gripper_right
```

## 历史命令（旧流程，仅存档）

```bash
python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views none --method grasp-lift
```

## 可视化（旧流程）

```bash
python vis_trajectory.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --opt-params output/gripper_only_left/opt_params.npy \
    --output-dir output/gripper_only_left \
    --hand-idx 0 \
    --save /tmp/traj_compare
```

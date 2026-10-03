# MANO 轨迹优化抓取计划 — 详细方案 (第十九轮 v3)

**目标**: 在 MANO 轨迹基础上, 最小化偏离代价, 使 R1 夹爪能真实抓取粉色物体 (glb_1)
**核心约束**: 全程重力, 全程物理真实, 不 hack 物理引擎
**两个方案**: SPIDER 引导 (不改轨迹) / CMA-ES 轨迹优化 (最小偏离)

---

## 一、问题本质 (为什么 MANO 轨迹抓不住)

MANO 轨迹本身是正确的人手运动, 但映射到 R1 夹爪后有三个物理失配:

### 失配 1: 几何失配
- MANO 手指长度 ≠ R1 夹爪手指长度
- `compute_analytical_gripper_pose` 从指尖中点反推 root_pos, 几何差异导致 root 位置偏移
- 实测: MANO 在 f_grasp 处离物体 ~10cm (上方)

### 失配 2: 姿态失配
- MANO R_Y 的 Z 分量 >0.9 (手指上下叠放)
- R1 夹爪需要 R_Y 水平 (左右捏合) 才能夹地面物体
- 当前用 `_make_horizontal_closing_R` hack, 破坏了姿态跟随

### 失配 3: 控制失配
- R1 夹爪 PD 闭合需要时间 (8 子步内 qpos 只到 0.013, 间距 4.9cm)
- MANO 手指闭合是瞬时的 (人手肌腱快速收缩)
- 物体仅 1-2cm, PD 闭合慢导致夹不住

**结论**: 不是 MANO 轨迹错了, 而是映射到 R1 后物理上不可行. 需要在 MANO 轨迹基础上做最小修正, 使其物理可行.

---

## 二、方案 1: SPIDER Virtual Contact Guidance (`--method spider`)

### 2.1 核心思想

**不改 MANO 轨迹 (零偏离), 用虚拟接触引导帮助物体到达手指间.**

SPIDER 论文的核心洞察: 参考轨迹可能物理不可行, 但可以通过"虚拟接触引导"让物体被动到达抓取位置, 引导强度逐渐衰减到零, 最终轨迹在纯物理下稳定.

### 2.2 详细流程

```
输入: MANO 轨迹 (pos, R, j1, j2) + offset + 粉色物体 glb_1

全程 (113 帧):
  每帧:
    1. 夹爪位姿 = MANO 轨迹 + offset (严格跟随, 不偏离)
       - gripper_pos = mano_pos[f] + offset
       - gripper_R = mano_R[f] (直接用, 不水平化)
       - gripper_val = mano_j1[f] (跟随 MANO 手指开合)

    2. 物理控制 (全程重力, 全程真实):
       - 根: set_root_pose(gripper_pos) + set_root_linear_velocity(delta * freq)
       - 手指: set_drive_target(gripper_val) (纯 PD, 不 set_qpos)
       - 物体: 不干预 (纯被动动力学)

    3. 虚拟接触引导 (SPIDER 核心):
       计算引导强度 gain(f):
         - 前 30% 帧: gain = 0.0 (APPROACH, 无引导)
         - 30%-60% 帧: gain 从 0.8 衰减到 0.2 (抓取阶段, 强引导)
         - 60%-80% 帧: gain 从 0.2 衰减到 0.0 (运输阶段, 弱引导)
         - 80%-100% 帧: gain = 0.0 (释放, 纯物理)

       如果 gain > 0:
         - 计算手指中点: finger_mid = (finger1_pos + finger2_pos) / 2
         - 计算目标位置: obj_target = finger_mid + gripper_R[:,0] * 0.037
         - 引导物体: guided_pos = obj_pos * (1-gain) + obj_target * gain
         - 物体执行器: obj.set_pose(guided_pos) (不固定, 不清零速度)
         - 物体仍受重力, 仍受接触力, 只是位置被引导

    4. 物理步进 (8 子步, 全程重力):
       for _ in range(8):
         scene.step()  # 重力 = [0, 0, -9.81] 全程不变
```

### 2.3 关键设计点

**为什么引导物体而不是手指?**
- 引导手指 = 改变 MANO 轨迹 = 偏离
- 引导物体 = 不改 MANO 轨迹, 物体被动到达手指间
- 这是 SPIDER 的核心创新: "让物体来找手, 不是手去找物体"

**为什么增益衰减?**
- 早期高增益: 强制物体到达手指间 (克服几何失配)
- 中期低增益: 物体在手指间, 摩擦力开始起作用
- 后期零增益: 纯物理, 验证摩擦力能否维持抓取
- 如果后期零增益物体掉落, 说明摩擦力不够, 需要方案 2

**为什么不固定物体?**
- 固定物体 = weld = hack (用户否决)
- 引导物体 = 物体仍受物理影响, 只是位置被修正
- 物体仍受重力, 仍受接触力, 引导只是"帮助"

### 2.4 命令行

```bash
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode hybrid \
    --method spider \
    --spider-gain 0.8 \
    --spider-decay 0.5 \
    --num-frames 113
```

### 2.5 预期结果

- **最好情况**: 物体被引导到手指间, PD 闭合抓住, TRANSPORT 阶段摩擦力维持, 物体被运输到碗附近
- **可能问题**: TRANSPORT 阶段 gain=0 后物体掉落 (摩擦力不够)
- **如果失败**: 进入方案 2 (CMA-ES 优化轨迹)

### 2.6 验证标准

1. ✅ 全程重力 (日志确认 gravity=[0,0,-9.81])
2. ✅ 无 weld, 无 set_qpos 过程调用
3. ✅ MANO 轨迹零偏离 (gripper_pos = mano_pos + offset 全程)
4. ✅ 物体被引导到手指间 (CLOSE 末物体距手指中点 < 5mm)
5. ✅ 物体被提升 > 3cm
6. ✅ 物体在 TRANSPORT 阶段跟随 (xy_drift > 5cm)

---

## 三、方案 2: CMA-ES + Spline Keyframes (`--method grasp-lift`)

### 3.1 核心思想

**把整条 MANO 轨迹参数化为稀疏关键帧, 用 CMA-ES 优化关键帧偏移, 找到最小偏离的可行轨迹.**

Grasp-and-Lift 论文的核心: 用稀疏 spline keyframes 参数化整条轨迹, 优化器在"跟随参考"和"物理可行"之间找到最优平衡.

### 3.2 轨迹参数化

```
整条轨迹 (113 帧) 用 7 个关键帧参数化:

关键帧位置 (固定):
  kf0 = 0      (轨迹起点)
  kf1 = 0.2*N  (APPROACH 中)
  kf2 = 0.4*N  (接近物体)
  kf3 = 0.5*N  (抓取时刻, f_grasp)
  kf4 = 0.6*N  (提升)
  kf5 = 0.8*N  (运输)
  kf6 = 1.0*N  (释放)

每个关键帧 6 维偏移 (优化变量):
  [dx, dy, dz, droll, dpitch, dyaw]
  - dx,dy,dz: 位置偏移 (±3cm)
  - droll,dpitch,dyaw: 姿态偏移 (±15°)

总参数: 7 × 6 = 42 维

中间帧用 cubic spline 插值:
  offset[f] = CubicSpline(keyframe_indices, keyframe_params)[f]

最终轨迹:
  gripper_pos[f] = mano_pos[f] + offset[f][:3]
  gripper_R[f] = mano_R[f] @ euler_to_R(offset[f][3:6])
  gripper_val[f] = mano_j1[f]  (手指开合跟随 MANO, 不优化)
```

### 3.3 CMA-ES 优化

```
初始化:
  x0 = zeros(42)  (零偏移 = 完全跟随 MANO)
  sigma0 = 0.02   (初始步长 2cm/2°)

循环 (50 代):
  1. 采样 64 个候选偏移向量
     x_i = x_mean + sigma * N(0, C)

  2. 对每个候选, 物理 rollout (全程重力, 全程真实):
     a. 用 x_i 计算 spline 偏移轨迹
     b. 应用偏移: gripper_pos = mano_pos + offset_pos
                  gripper_R = mano_R @ euler_to_R(offset_rot)
     c. 物理仿真 113 帧 (纯 PD, 无引导, 无 hack)
     d. 记录: 接触帧数, 提升量, 距碗距离, 偏离量, 穿透, 掉落

  3. 计算奖励:
     reward = 
       + 1.0 * contact_frames          # 接触帧数 (越多越好)
       + 100.0 * (final_z - init_z)    # 提升量 (越高越好)
       - 50.0 * dist_to_bowl          # 距碗距离 (越近越好)
       - 10.0 * ||offset||²           # 偏离代价 (越小越好)
       - 100.0 * dropped              # 掉落惩罚
       - 1000.0 * max_penetration     # 穿透惩罚

  4. 选择 top 10% 精英, 更新 x_mean, sigma, C (CMA-ES 更新)

输出: 最优 x_best, 保存到 opt_params.npy
```

### 3.4 关键设计点

**为什么用 spline keyframes 而不是逐帧优化?**
- 逐帧优化: 113 帧 × 6 维 = 678 维 (太高维, CMA-ES 收敛慢)
- 7 个 keyframes: 42 维 (低维, CMA-ES 可收敛)
- spline 插值保证轨迹平滑 (相邻帧偏移连续)

**为什么手指开合不优化?**
- MANO 手指开合已经是正确的人手运动
- 物理失配在位置和姿态, 不在手指开合
- 减少优化维度 (从 678 维降到 42 维)

**为什么偏离代价 ||offset||²?**
- 用户要求"最小化偏离代价"
- CMA-ES 会在"抓取成功"和"偏离小"之间找平衡
- 如果零偏移就能抓取, CMA-ES 会保持零偏移

**为什么全程重力, 无引导?**
- 这是最终验证: 优化后的轨迹必须在纯物理下可行
- 如果优化后纯物理抓不住, 说明需要更多 keyframes 或更大偏移范围

### 3.5 命令行

```bash
# 优化阶段 (无头, 约 30-60 分钟)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --grasp-mode hybrid \
    --method grasp-lift \
    --cmaes-pop 64 \
    --cmaes-gen 50 \
    --cmaes-sigma 0.02 \
    --cmaes-keyframes 7 \
    --num-frames 113

# 渲染阶段 (用优化结果)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode hybrid \
    --opt-params output/gripper_only_left/opt_params.npy \
    --num-frames 113
```

### 3.6 预期结果

- **最好情况**: CMA-ES 找到一组小偏移 (||offset|| < 3cm), 使夹爪在纯物理下抓住物体并运输到碗
- **可能问题**: 42 维仍然太高, CMA-ES 收敛慢 (需要更多代数)
- **如果失败**: 减少 keyframes 到 5 个 (30 维), 或增大 sigma0

### 3.7 验证标准

1. ✅ CMA-ES 收敛 (后 10 代 reward 方差 < 1e-3)
2. ✅ 优化后偏离 ||offset|| < 5cm (最小偏离代价)
3. ✅ 优化后接触帧数 > 30
4. ✅ 优化后提升量 > 3cm
5. ✅ 优化后物体距碗 < 10cm
6. ✅ 全程重力, 无引导, 无 hack (纯物理验证)

---

## 四、两个方案对比

| 维度 | 方案 1: SPIDER | 方案 2: CMA-ES |
|------|---------------|----------------|
| **MANO 轨迹偏离** | 零 (完全跟随) | 最小 (优化 keyframes 偏移) |
| **物理可行性** | 引导帮助 (非纯物理) | 纯物理 (优化后) |
| **计算时间** | 快 (1 次运行) | 慢 (3200 次 rollout) |
| **改动量** | 中 (加物体引导) | 大 (加 CMA-ES 优化器) |
| **适合场景** | MANO 轨迹基本正确, 只需帮助 | MANO 轨迹需修正才能抓取 |
| **最终验证** | 引导衰减后纯物理 | 直接纯物理 |

**推荐**: 先试方案 1 (SPIDER), 如果引导衰减后物体掉落, 再用方案 2 (CMA-ES) 优化轨迹.

---

## 五、文件改动清单

### 方案 1 (SPIDER)
| 文件 | 改动 |
|------|------|
| `grasp_hawor.py` | 加 `--method spider` 参数 |
| `grasp_hawor.py` | 主循环加物体引导逻辑 (每帧, gain 退火) |
| `grasp_hawor.py` | `_compute_mano_neutral_target` 简化 (删除阶段判定, 全程跟 MANO) |

### 方案 2 (CMA-ES)
| 文件 | 改动 |
|------|------|
| `grasp_hawor.py` | 加 `--method grasp-lift` 参数 |
| `grasp_hawor.py` | `_compute_mano_neutral_target` 接受 spline keyframes 偏移 |
| `traj_optimize.py` | 新增 CMA-ES 优化器 |
| `traj_optimize.py` | 新增 spline keyframe 插值 |
| `traj_optimize.py` | 新增奖励函数 |

---

## 六、执行计划

### Phase 1: 实现方案 1 (SPIDER)
1. 加 `--method spider` 参数解析
2. 简化 `_compute_mano_neutral_target` (删除阶段判定, 全程跟 MANO)
3. 主循环加物体引导 (gain 退火)
4. 测试: `--method spider --num-frames 113`
5. 验证: 物体是否被引导到手指间, 是否被提起

### Phase 2: 实现方案 2 (CMA-ES) — 如果 Phase 1 失败
1. 加 `--method grasp-lift` 参数解析
2. 实现 spline keyframe 插值
3. 实现 CMA-ES 优化器 (调用 `cma` 库)
4. 实现奖励函数
5. 测试: `--method grasp-lift --num-frames 113`
6. 验证: CMA-ES 是否收敛, 优化后是否抓住

### Phase 3: 文档同步
1. 更新 CHANGE_LOG.md
2. 更新 docs/grasp_hawor_analysis.md

---

## 七、核心原则

> **MANO 轨迹是正确的, 物理失配需要最小修正.**
> - 方案 1: 零偏离, 引导物体 (不改轨迹)
> - 方案 2: 最小偏离, 优化 keyframes (改轨迹但最小化)
> - 全程重力, 全程物理真实
> - 不 hack 物理引擎 (无 set_qpos, 无 lock_root, 无 weld)

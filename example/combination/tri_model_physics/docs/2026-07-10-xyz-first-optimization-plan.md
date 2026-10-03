# 3D XYZ 优先优化方案

**目标**: 先把优化维度降到 3 维 XYZ 偏移，让夹爪在原始 MANO 轨迹基础上接触到物体；接触成功后再扩展到 6/7 维实现稳定抓取和提升。

**架构**: 在 `traj_optimize.py` 新增 3 维 CEM 优化器和奖励函数；在 `grasp_hawor.py` 的 `_compute_mano_neutral_target` 中优先处理 3 维参数，`run_optimize` 中调用 3D CEM。

**技术栈**: Python, NumPy, PyTorch (CEM 采样), SAPIEN 物理仿真

---

## Phase 1: 3 维 XYZ 偏移优化（只追求接触）

### Step 1: 新增 3D 参数辅助函数

**文件**: `traj_optimize.py`

在文件末尾新增（或插入到帧级窗口函数附近）:

```python
# ============================================================
# 3 维 XYZ 偏移参数化 (第二十九轮: 先只优化位置接触)
# ============================================================
def apply_xyz_offset(mano_pos, mano_R, xyz_offset):
    """对 MANO 位姿施加常量 XYZ 偏移, 姿态保持不变.

    Args:
        mano_pos: (3,) MANO 位置
        mano_R: (3, 3) MANO 姿态
        xyz_offset: (3,) [dx, dy, dz]

    Returns:
        (gripper_pos, gripper_R)
    """
    return mano_pos + np.asarray(xyz_offset, dtype=np.float64), mano_R.copy()
```

**验证**: `python -c "from traj_optimize import apply_xyz_offset; ..."` 导入不报错。

---

### Step 2: 新增 3D CEM 优化器

**文件**: `traj_optimize.py`

```python
# ============================================================
# 3 维 XYZ CEM 优化器
# ============================================================
def cem_xyz_optimize(
    rollout_fn,
    n_iterations=15,
    population_size=32,
    elite_frac=0.2,
    initial_std=0.10,
    pos_range=0.50,
    seed=42,
    verbose=True,
):
    """3 维 XYZ 偏移 CEM 优化.

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> 必须返回 {"reward": float}
        n_iterations: CEM 迭代轮数
        population_size: 每轮采样数
        elite_frac: 精英比例
        initial_std: 初始标准差 (m)
        pos_range: XYZ 偏移范围 ±m
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        best_params: np.ndarray (3,)
        best_reward: float
        reward_history: list
    """
    import torch

    dim = 3
    n_elite = max(2, int(population_size * elite_frac))

    bounds_low = torch.full((dim,), -pos_range, dtype=torch.float64)
    bounds_high = torch.full((dim,), pos_range, dtype=torch.float64)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    mu = torch.zeros(dim, dtype=torch.float64)
    std = torch.full((dim,), initial_std, dtype=torch.float64)

    best_params_np = np.zeros(dim, dtype=np.float64)
    best_reward = -np.inf
    reward_history = []

    for it in range(n_iterations):
        noise = torch.normal(0, 1, (population_size, dim), generator=rng, dtype=torch.float64)
        samples = torch.clamp(mu + std * noise, bounds_low, bounds_high)
        samples_np = samples.numpy()

        rewards = np.array([rollout_fn(s)["reward"] for s in samples_np])

        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_samples = samples_np[elite_idx]

        mu = torch.tensor(elite_samples.mean(axis=0), dtype=torch.float64)
        std = torch.tensor(elite_samples.std(axis=0) + 1e-4, dtype=torch.float64)

        if rewards.max() > best_reward:
            best_reward = float(rewards.max())
            best_params_np = samples_np[np.argmax(rewards)].copy()

        reward_history.append(best_reward)

        if verbose and (it % 3 == 0 or it == n_iterations - 1):
            print(f"[CEM-XYZ] iter {it}/{n_iterations}: best={best_reward:.3f}, "
                  f"mu={mu.numpy()}, std={std.numpy()}")

    return best_params_np, best_reward, reward_history
```

**验证**: 导入不报错；用 dummy rollout 跑 2 iter 测试能返回 (3,) 参数。

---

### Step 3: 新增 Phase-1 奖励函数

**文件**: `traj_optimize.py`

```python
# ============================================================
# Phase 1 奖励: 只奖励接近和接触, 不奖励 lift
# ============================================================
REWARD_WEIGHTS_XYZ = dict(
    w_close_avg=100.0,   # 全程平均距离 (越小越好)
    w_min_dist=400.0,    # 最小距离奖励 (越近越好)
    w_contact=200.0,     # 接触帧数
    w_pen=1000.0,        # 穿透惩罚
    w_launch=500.0,      # 弹飞惩罚
)


def compute_reward_xyz(rollout_result, weights=REWARD_WEIGHTS_XYZ):
    """Phase 1: 3D XYZ 优化奖励.

    只关注:
      - 夹爪整体接近物体
      - CLOSE 阶段最小距离
      - 接触帧数
      - 不穿透、不弹飞
    暂时不奖励 lift (那是 Phase 2 的事).
    """
    avg_dist = rollout_result.get("avg_dist_throughout", 1.0)
    min_dist = rollout_result.get("min_dist_in_close", 1.0)
    contact_frames = rollout_result.get("contact_frames_in_close", 0)
    penetration = rollout_result.get("max_penetration", 0.0)
    obj_init_xy = rollout_result.get("obj_init_xy", [0.0, 0.0])
    obj_final_xy = rollout_result.get("obj_final_xy", [0.0, 0.0])

    pen_pen = max(0.0, penetration - 0.01)
    xy_drift = float(np.linalg.norm(np.array(obj_final_xy) - np.array(obj_init_xy)))
    launch_pen = max(0.0, xy_drift - 0.2)

    reward = (
        - weights["w_close_avg"] * avg_dist
        + weights["w_min_dist"] / (min_dist + 0.01)
        + weights["w_contact"] * contact_frames
        - weights["w_pen"] * pen_pen
        - weights["w_launch"] * launch_pen
    )
    return reward
```

**验证**: 用模拟 rollout_result dict 测试返回值是 float。

---

### Step 4: 在 `_compute_mano_neutral_target` 中应用 3D 偏移

**文件**: `grasp_hawor.py` (L2368 之前插入)

在现有 `if opt_params is not None and len(opt_params) == 42:` 分支**之前**插入:

```python
        # 第二十九轮: 3 维 XYZ 偏移优化 (Phase 1)
        if opt_params is not None and len(opt_params) == 3:
            from traj_optimize import apply_xyz_offset
            gripper_pos, gripper_R = apply_xyz_offset(
                mano_target_pos, mano_R, opt_params
            )
            # 手指开合保持默认阶段逻辑
            if local_idx < close_start:
                gripper_val = mano_j1
            elif local_idx < close_end:
                gripper_val = 0.0
            elif has_bowl and local_idx < release_end:
                gripper_val = 0.0
            else:
                gripper_val = GRIPPER_MAX_OPEN
            return gripper_pos, gripper_R, gripper_val, phase, 0.5
```

**验证**: 运行 grasp_hawor.py 时 `opt_params` shape=(3,) 能进入该分支，不触发 42/156/9 维处理。

---

### Step 5: 在 `run_optimize` 中调用 3D CEM

**文件**: `grasp_hawor.py` (L4810 附近)

替换当前 `cem_window_optimize` 调用为:

```python
        # 第二十九轮: Phase 1 只优化 3 维 XYZ
        from traj_optimize import cem_xyz_optimize, compute_reward_xyz

        self._window_params = None
        if hasattr(self, '_kf_cache'):
            delattr(self, '_kf_cache')

        logger.info(f"Phase 1: 3D XYZ CEM 优化 (3 维)")

        best_params, best_reward, reward_history = cem_xyz_optimize(
            rollout_fn=rollout_single,
            n_iterations=15,
            population_size=32,
            elite_frac=0.2,
            initial_std=0.10,
            pos_range=0.50,
        )
```

同时需要确保 `rollout_single` 返回的 `reward` 是 Phase-1 奖励。在 `rollout_single` 末尾 (L4746 附近):

```python
            result = dict(...)
            result["reward"] = compute_reward_xyz(result)
            result["reward_coarse"] = compute_reward_xyz(result)
            return result
```

**验证**: 运行命令后日志显示 `Phase 1: 3D XYZ CEM 优化 (3 维)`，CEM 正常迭代。

---

### Step 6: 运行与验证

**命令**:

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views none --method grasp-lift
```

**成功标准 (Phase 1)**:
| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| `contact_frames_in_close` | 0-7 | >= 1 |
| `min_dist_in_close` (m) | ~0.02 | < 0.02 |
| `avg_dist_throughout` (m) | ~0.10 | < 0.05 |

**预计耗时**: 15 iter × 32 pop = 480 rollouts，单进程约 5-10 分钟。

---

## Phase 2: 扩展到 6/7 维（稳定抓取 + 提升）

### Step 7: 新增 6D/7D CEM

**条件**: Phase 1 已稳定达到 `contact_frames_in_close >= 3` 且 `min_dist < 0.02m`。

**做法**:
1. 以 Phase 1 最优 XYZ 作为初始均值的前 3 维。
2. 增加 `[droll, dpitch, dyaw]` 为第 4-6 维，初始均值 0，std 0.1 rad。
3. 可选增加 `gripper_close_bias` 为第 7 维。

**文件**: `traj_optimize.py` 新增 `cem_6d_optimize` / `cem_7d_optimize`（或复用 `cem_xyz_optimize` 并扩展 dim）。

**奖励函数**: 在 `compute_reward_xyz` 基础上加入:
- `w_lift`: 有接触时的提升量
- `w_grasp_success`: 接触 + lift > 2cm 一次性奖励
- `w_last_contact`: CLOSE 末段接触帧数

### Step 8: 在 `_compute_mano_neutral_target` 中处理 6/7 维

**文件**: `grasp_hawor.py`

类似 3D 分支，但额外:
- 应用 `[droll, dpitch, dyaw]` 旋转偏移
- 应用 `gripper_close_bias` 调整手指闭合目标

```python
        # 6/7 维扩展 (Phase 2)
        if opt_params is not None and len(opt_params) in (6, 7):
            from traj_optimize import Rotation
            xyz = opt_params[0:3]
            rpy = opt_params[3:6]
            gripper_pos = mano_target_pos + xyz
            if np.linalg.norm(rpy) > 1e-8:
                R_corr = Rotation.from_euler("xyz", rpy).as_matrix()
                gripper_R = R_corr @ mano_R
            else:
                gripper_R = mano_R.copy()
            # ... 手指逻辑
            return gripper_pos, gripper_R, gripper_val, phase, 0.5
```

---

## 文件改动汇总

| 文件 | 改动 | 行数估算 |
|------|------|---------|
| `traj_optimize.py` | 新增 `apply_xyz_offset`, `cem_xyz_optimize`, `compute_reward_xyz`, `REWARD_WEIGHTS_XYZ` | ~100 |
| `grasp_hawor.py` | 3D 分支 + CEM 调用替换 + reward 使用 Phase-1 | ~40 |
| `docs/2026-07-10-xyz-first-optimization-plan.md` | 本计划文档 | — |

---

## 风险与回退

- **风险**: 3D 偏移可能不足以让夹爪在保持 MANO 姿态的同时接触到物体（需要旋转调整）。
- **回退**: 如果 Phase 1 连续 3 次无法 `contact >= 1`，直接扩展到 6D XYZ+RPY，跳过 3D 成功门槛。

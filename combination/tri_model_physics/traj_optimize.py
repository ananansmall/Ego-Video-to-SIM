"""离线轨迹优化 (第十九轮: CMA-ES + Spline Keyframes)

在 MANO 参考轨迹基础上, 用 CMA-ES 优化 spline keyframes 偏移,
最小化偏离代价同时保证抓取成功. 物理仿真保持 SAPIEN + PD 控制.

借鉴:
  - Grasp-and-Lift: CMA-ES + spline keyframes 轨迹优化
  - do-as-i-do Stage 5: 多目标奖励 (接触+提升-偏离-穿透)
  - SPIDER: 域随机化

用法:
    python grasp_hawor.py --mode gripper_only --side left \
        --hawor-dir /home/an/data/hawor/7 \
        --ras-dir /home/an/data/ras/my_7mp4_result \
        --views god --grasp-mode hybrid --method grasp-lift
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# ============================================================
# Spline Keyframe 参数化 (42 维 = 7 keyframes × 6 dim)
# ============================================================
N_KEYFRAMES = 7
DIM_PER_KEYFRAME = 6  # [dx, dy, dz, droll, dpitch, dyaw]
N_PARAMS = N_KEYFRAMES * DIM_PER_KEYFRAME  # 42

# 关键帧位置 (相对于总帧数的比例)
KEYFRAME_RATIOS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0])

# 参数范围 (第二十轮 v2: 进一步增大位置偏移, 确保能到达物体)
POS_RANGE = 0.50  # 位置偏移 ±50cm (MANO 离物体 ~45cm Z 差, 需要足够范围)
ROT_RANGE = 0.35  # 姿态偏移 ±20° (rad) (保持: 用户要求"姿态要跟上 MANO")

_bounds = []
for _ in range(N_KEYFRAMES):
    _bounds.extend([
        [-POS_RANGE, POS_RANGE],  # dx
        [-POS_RANGE, POS_RANGE],  # dy
        [-POS_RANGE, POS_RANGE],  # dz
        [-ROT_RANGE, ROT_RANGE],  # droll
        [-ROT_RANGE, ROT_RANGE],  # dpitch
        [-ROT_RANGE, ROT_RANGE],  # dyaw
    ])
PARAM_RANGE = np.array(_bounds)

DEFAULT_PARAMS = np.zeros(N_PARAMS)
# Do as I Do: 初始化均值带 Z 偏移, 让第一代样本就站到物体高度
# 物体 Z~0.47m, MANO 手腕 Z~0.02m, 需抬高 ~0.40m
# KF1-KF5 (15%-80% 轨迹) 覆盖全部阶段, 平滑过渡
for _kf in range(1, 6):  # KF1, KF2, KF3, KF4, KF5
    DEFAULT_PARAMS[_kf * 6 + 2] = 0.40  # Z 偏移
# KF0 (approach) 和 KF6 (retreat) 保持 0, 跟随 MANO


# ============================================================
# 奖励权重 (第二十轮: 拆分位置/姿态偏离 + 持续接近 + 抓取成功大奖励)
# ============================================================
REWARD_WEIGHTS = dict(
    # 偏离代价拆分: 位置允许大幅偏离 (MANO 离物体 16cm), 姿态严格跟随
    w_track_pos=1.0,     # 位置偏离代价 (低权重: 允许大幅位置调整以到达物体)
    w_track_rot=30.0,    # 姿态偏离代价 (高权重: 用户要求"姿态要跟上 MANO")
    w_reach=300.0,       # 接近奖励 (CLOSE 阶段最后 5 帧平均距离, 鼓励持续接触)
    w_min_dist=400.0,    # 最小距离奖励: 越近越好 (鼓励夹爪真正贴近物体)
    w_contact=200.0,     # 接触帧数 (基础接触奖励)
    w_last_contact=600.0,  # CLOSE 末段接触奖励 (最后 5 帧的接触帧数, 鼓励稳定夹持)
    w_lift=1200.0,       # 提升量 (只在有接触时生效, 见 compute_reward)
    w_lift_no_contact=1200.0,  # 无接触提升惩罚 (禁止通过弹飞/推动物体骗取 lift 奖励)
    w_grasp_success=2000.0,  # 抓取成功大奖励 (接触 + lift > 2cm 时一次性给)
    w_stable_grasp=1500.0,  # 稳定抓取奖励 (contact>=5 且 lift>1cm)
    w_bowl=5.0,          # 距碗距离 (最小化)
    w_drop=200.0,        # 掉落惩罚 (增大)
    w_z_gap=300.0,       # 夹爪最低点与物体 z 差距 (引导夹爪下降到物体高度)
    w_launch=1500.0,     # 弹飞惩罚 (xy_drift > 0.2m) — 第二十六轮 v2: 加大以抑制漂移
    w_pen=1000.0,        # 穿透惩罚
    w_smooth=2.0,        # 帧间平滑度 (帧级窗口用, 惩罚相邻帧位姿突变)
)

# 粗搜阶段奖励权重 (第二十一轮: 无偏离代价, 只看接近 + 接触 + 穿透)
# 目的: 给 CMA-ES 连续的梯度信号, 让优化器找到"下降到物体"的方向
# 不惩罚偏离, 不惩罚姿态, 不奖励提升 (粗搜阶段不可能有提升)
COARSE_REWARD_WEIGHTS = dict(
    w_close_avg=100.0,       # 全程平均距离越小越好 (连续梯度! 即使没接触也有信号)
    w_min_z=200.0,           # gripper 最低点与物体 z 的差距 (引导下降)
    w_contact=80.0,          # 接触帧数
    w_pen=2000.0,            # 穿透惩罚 (不能乱穿模)
    w_launch=500.0,          # 弹飞惩罚
)


# ============================================================
# Spline Keyframe 插值
# ============================================================
def interp_keyframes(keyframe_params, total_frames):
    """用 cubic spline 插值得到整条轨迹的偏移

    Args:
        keyframe_params: (42,) 或 (7, 6) 关键帧参数
        total_frames: 总帧数

    Returns:
        (total_frames, 6) 整条轨迹的偏移, 每帧 [dx,dy,dz,droll,dpitch,dyaw]
    """
    if keyframe_params.ndim == 1:
        keyframe_params = keyframe_params.reshape(N_KEYFRAMES, DIM_PER_KEYFRAME)

    frame_indices = (KEYFRAME_RATIOS * max(total_frames - 1, 1)).astype(int)
    # 确保关键帧位置严格递增 (CubicSpline 要求)
    for i in range(1, len(frame_indices)):
        if frame_indices[i] <= frame_indices[i - 1]:
            frame_indices[i] = frame_indices[i - 1] + 1

    cs = CubicSpline(frame_indices, keyframe_params, axis=0)
    return cs(np.arange(total_frames))


def apply_keyframe_offset(mano_pos, mano_R, offset_6d):
    """应用 keyframe 偏移到 MANO 轨迹

    Args:
        mano_pos: (3,) MANO 位置
        mano_R: (3, 3) MANO 姿态
        offset_6d: (6,) [dx, dy, dz, droll, dpitch, dyaw]

    Returns:
        (gripper_pos, gripper_R)
    """
    pos_delta = offset_6d[:3]
    rot_euler = offset_6d[3:6]

    gripper_pos = mano_pos + pos_delta
    if np.linalg.norm(rot_euler) > 1e-8:
        R_correction = Rotation.from_euler("xyz", rot_euler).as_matrix()
        gripper_R = R_correction @ mano_R
    else:
        gripper_R = mano_R

    return gripper_pos, gripper_R


# ============================================================
# CMA-ES 优化器 (借鉴 Grasp-and-Lift)
# ============================================================
def cmaes_optimize(
    rollout_fn,
    n_generations=50,
    population_size=32,
    sigma0=0.25,
    sigma_end=0.005,
    sigma_decay=None,
    seed=42,
    verbose=True,
):
    """CMA-ES 优化 (sigma 退火调度)

    从较大的 sigma0 开始, sigma 指数衰减到 sigma_end, 实现
    "开始大范围探索, 逐渐缩小搜索范围" 的效果.

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> 返回包含 "reward" 的字典
        n_generations: 代数
        population_size: 种群大小
        sigma0: 初始步长 (默认 25cm/25°)
        sigma_end: 终止步长 (默认 0.5°), 与 sigma0/n_generations 共同确定退火速度
        sigma_decay: 每代衰减系数, 若设置则覆盖 sigma_end 的自动计算
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        best_params, best_reward
    """
    try:
        import cma
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])
        import cma

    # 计算退火系数
    if sigma_decay is None:
        if sigma_end is not None and sigma_end < sigma0:
            sigma_decay = (sigma_end / sigma0) ** (1.0 / max(n_generations, 1))
        else:
            sigma_decay = 1.0  # 不退火

    options = {
        'maxiter': n_generations,
        'popsize': population_size,
        'seed': seed,
        'verbose': -1 if not verbose else 1,
        'tolfun': 1e-6,
        'tolx': 1e-4,
        'bounds': [PARAM_RANGE[:, 0].tolist(), PARAM_RANGE[:, 1].tolist()],
    }

    eval_count = [0]

    def objective(x):
        x = np.asarray(x, dtype=np.float64)
        result = rollout_fn(x)
        reward = result["reward"]
        eval_count[0] += 1
        if verbose and eval_count[0] % population_size == 0:
            print(f"[CMA-ES] eval {eval_count[0]}: reward={reward:.3f}")
        return -reward  # CMA-ES 最小化

    es = cma.CMAEvolutionStrategy(DEFAULT_PARAMS.copy(), sigma0, options)
    gen = [0]

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

        # sigma 退火: 每代主动衰减
        if sigma_decay < 1.0:
            old_sigma = es.sigma
            es.sigma *= sigma_decay
            if verbose and gen[0] % 5 == 0:
                current_sigma_deg = es.sigma * 180 / np.pi  # rad→° 便于理解
                print(f"  sigma: {old_sigma:.6f} → {es.sigma:.6f}  ({current_sigma_deg:.2f}°)")
        gen[0] += 1

        if verbose:
            es.disp()

    best_params = es.result.xbest
    best_reward = -es.result.fbest

    if verbose:
        print(f"[CMA-ES] 完成: best_reward={best_reward:.3f}, "
              f"||offset||={np.linalg.norm(best_params):.4f}")

    return np.asarray(best_params, dtype=np.float64), best_reward


# ============================================================
# 两阶段 CMA-ES (第二十一轮: 粗搜 → 精调)
# ============================================================
def cmaes_two_stage_optimize(
    rollout_fn,
    n_gen_stage1=10,
    pop_stage1=64,
    sigma_stage1=0.20,
    n_gen_stage2=40,
    pop_stage2=32,
    sigma_stage2=0.04,
    seed=42,
    verbose=True,
):
    """两阶段 CMA-ES 优化

    阶段 1 (粗搜): sigma=0.20, pop=64, 10 代
      - 奖励: 接近 + 接触 + 穿透 (无偏离代价)
      - 目标: 在 42 维超球面上快速找到"下降到物体"的方向

    阶段 2 (精调): sigma=0.04, pop=32, 40 代
      - 奖励: 完整奖励函数 (偏离 + 接触 + 提升 + 穿透)
      - 起点: 阶段 1 的最优参数
      - 目标: 精细调整轨迹, 真正抓取

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> 返回包含 "reward"/"reward_coarse" 的字典
        n_gen_stage1: 阶段 1 代数
        pop_stage1: 阶段 1 种群大小
        sigma_stage1: 阶段 1 初始步长 (20cm)
        n_gen_stage2: 阶段 2 代数
        pop_stage2: 阶段 2 种群大小
        sigma_stage2: 阶段 2 初始步长 (4cm)
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        best_params, best_reward
    """
    try:
        import cma
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])
        import cma

    bounds = [PARAM_RANGE[:, 0].tolist(), PARAM_RANGE[:, 1].tolist()]

    # === 阶段 1: 粗搜 ===
    if verbose:
        print(f"\n{'='*60}")
        print(f"[两阶段 CMA-ES] 阶段 1 (粗搜): sigma={sigma_stage1}, pop={pop_stage1}, {n_gen_stage1} 代")
        print(f"{'='*60}")

    stage1_opts = {
        'maxiter': n_gen_stage1,
        'popsize': pop_stage1,
        'seed': seed,
        'verbose': -1 if not verbose else 1,
        'tolfun': 1e-6,
        'tolx': 1e-4,
        'bounds': bounds,
    }

    best_params_s1 = DEFAULT_PARAMS.copy()
    best_reward_s1 = -np.inf

    es1 = cma.CMAEvolutionStrategy(DEFAULT_PARAMS.copy(), sigma_stage1, stage1_opts)
    while not es1.stop():
        solutions = es1.ask()
        rewards_s1 = []
        for s in solutions:
            s = np.asarray(s, dtype=np.float64)
            result = rollout_fn(s)
            reward = result.get("reward_coarse", -100.0)  # 粗搜用 coarse reward
            rewards_s1.append(reward)
        es1.tell(solutions, [-r for r in rewards_s1])  # CMA-ES 最小化

        # 跟踪最优
        for r, s in zip(rewards_s1, solutions):
            if r > best_reward_s1:
                best_reward_s1 = r
                best_params_s1 = np.asarray(s, dtype=np.float64)

        if verbose:
            es1.disp()
            print(f"  [阶段 1] 当前最优: reward_coarse={best_reward_s1:.3f}")

    if verbose:
        print(f"\n[阶段 1] 完成: best_reward_coarse={best_reward_s1:.3f}, "
              f"||offset||={np.linalg.norm(best_params_s1):.4f}")

    # === 阶段 2: 精调 (从阶段 1 最优初始化) ===
    if verbose:
        print(f"\n{'='*60}")
        print(f"[两阶段 CMA-ES] 阶段 2 (精调): sigma={sigma_stage2}, pop={pop_stage2}, {n_gen_stage2} 代")
        print(f"  初始均值: 阶段 1 最优 ||params||={np.linalg.norm(best_params_s1):.4f}")
        print(f"{'='*60}")

    stage2_opts = {
        'maxiter': n_gen_stage2,
        'popsize': pop_stage2,
        'seed': seed + 1,  # 不同种子, 避免重复初始采样
        'verbose': -1 if not verbose else 1,
        'tolfun': 1e-6,
        'tolx': 1e-4,
        'bounds': bounds,
    }

    eval_count_s2 = [0]
    best_params_s2 = best_params_s1.copy()
    best_reward_s2 = -np.inf

    es2 = cma.CMAEvolutionStrategy(best_params_s1.copy(), sigma_stage2, stage2_opts)
    while not es2.stop():
        solutions = es2.ask()
        rewards_s2 = []
        for s in solutions:
            s = np.asarray(s, dtype=np.float64)
            result = rollout_fn(s)
            reward = result["reward"]  # 阶段 2 用完整 reward
            rewards_s2.append(reward)
            eval_count_s2[0] += 1
        es2.tell(solutions, [-r for r in rewards_s2])

        # 跟踪最优
        for r, s in zip(rewards_s2, solutions):
            if r > best_reward_s2:
                best_reward_s2 = r
                best_params_s2 = np.asarray(s, dtype=np.float64)

        if verbose and eval_count_s2[0] % pop_stage2 == 0:
            print(f"[阶段 2] eval {eval_count_s2[0]}: best_reward={best_reward_s2:.3f}")

    if verbose:
        print(f"\n[两阶段 CMA-ES] 完成: best_reward={best_reward_s2:.3f}, "
              f"||offset||={np.linalg.norm(best_params_s2):.4f}")

    return best_params_s2, best_reward_s2


# ============================================================
# CEM 优化器 (保留向后兼容, 第十八轮)
# ============================================================
def cem_optimize(
    rollout_fn,
    n_iterations=5,
    n_samples=16,
    elite_frac=0.25,
    initial_std=0.3,
    seed=42,
):
    """Cross-Entropy Method (第十八轮, 9 维参数, 保留向后兼容)

    Args:
        rollout_fn: Callable[[np.ndarray], dict]
        n_iterations: 迭代轮数
        n_samples: 每轮采样数
        elite_frac: 精英比例
        initial_std: 初始标准差
        seed: 随机种子

    Returns:
        best_params, best_reward
    """
    # 第十八轮 9 维参数 (向后兼容)
    old_default = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5])
    old_range = np.array([
        [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02],
        [-0.26, 0.26], [-0.26, 0.26], [-0.26, 0.26],
        [-0.01, 0.02], [0.2, 0.5], [0.1, 0.8],
    ])

    rng = np.random.default_rng(seed)
    mu = old_default.copy()
    std = (old_range[:, 1] - old_range[:, 0]) * initial_std
    n_elite = max(1, int(n_samples * elite_frac))

    best_params, best_reward = mu, -np.inf

    for it in range(n_iterations):
        samples = mu + std * rng.standard_normal((n_samples, len(mu)))
        samples = np.clip(samples, old_range[:, 0], old_range[:, 1])

        results = [rollout_fn(s) for s in samples]
        rewards = np.array([r["reward"] for r in results])

        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_samples = samples[elite_idx]

        mu = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0) + 1e-4

        if rewards[elite_idx[-1]] > best_reward:
            best_reward = rewards[elite_idx[-1]]
            best_params = samples[elite_idx[-1]].copy()

        print(f"[CEM] iter {it}: best_reward={best_reward:.3f}")

    return best_params, best_reward


# ============================================================
# 奖励计算 (第二十轮: 拆分位置/姿态偏离 + 持续接近 + 抓取成功大奖励)
# ============================================================
def compute_reward(rollout_result, weights=REWARD_WEIGHTS):
    """多目标奖励 (第二十轮 v4)

    reward = -w_track_pos * ||pos_offset||² / N_kf          # 位置偏离 (低权重, 允许大幅调整)
             -w_track_rot * ||rot_offset||² / N_kf          # 姿态偏离 (高权重, 强制跟随 MANO)
             -w_reach * avg_dist_in_close_last5             # 持续接近 (CLOSE 末段平均距离)
             +w_contact * contact_frames                    # 接触帧数
             +w_lift * lift_amount                          # 提升量
             +w_grasp_success * (lift > 0.02)               # 抓取成功大奖励
             -w_bowl * dist_to_bowl                         # 距碗距离
             -w_drop * drop_penalty                         # 掉落
             -w_launch * max(0, xy_drift - 0.2)             # 弹飞惩罚
             -w_pen * penetration_penalty                   # 穿透
    """
    params = rollout_result["params"]
    contact_frames = rollout_result.get("contact_frames_in_close", 0)
    obj_init_z = rollout_result.get("obj_init_z", 0.0)
    obj_final_z = rollout_result.get("obj_final_z", 0.0)
    obj_final_xy = rollout_result.get("obj_final_xy", [0.0, 0.0])
    obj_init_xy = rollout_result.get("obj_init_xy", [0.0, 0.0])
    bowl_xy = rollout_result.get("bowl_xy", [0.0, 0.0])
    obj_dropped = rollout_result.get("obj_dropped", True)
    penetration = rollout_result.get("max_penetration", 0.0)
    # 第二十轮: 改为 CLOSE 阶段最后 5 帧平均距离 (鼓励持续接触, 而非单帧接触)
    avg_dist_in_close_last5 = rollout_result.get("avg_dist_in_close_last5", 1.0)
    min_dist_in_close = rollout_result.get("min_dist_in_close", 1.0)  # 保留作诊断

    # 拆分位置/姿态偏离代价 (支持帧级窗口优化: 非 42 维时不计算 keyframe 偏离)
    if len(params) == N_PARAMS:
        params_2d = params.reshape(N_KEYFRAMES, DIM_PER_KEYFRAME)
        pos_offsets = params_2d[:, :3]  # (N_kf, 3)
        rot_offsets = params_2d[:, 3:]  # (N_kf, 3)
        track_pos_cost = float(np.sum(pos_offsets ** 2)) / N_KEYFRAMES
        track_rot_cost = float(np.sum(rot_offsets ** 2)) / N_KEYFRAMES
    else:
        track_pos_cost = 0.0
        track_rot_cost = 0.0

    # 持续接近奖励 (CLOSE 末段平均距离, 越小越好)
    reach_cost = float(avg_dist_in_close_last5)
    # 最小距离奖励: 越近越好 (鼓励夹爪真正贴近物体)
    min_dist = rollout_result.get("min_dist_in_close", 1.0)
    min_dist_rew = 1.0 / (min_dist + 0.01)
    # 接触奖励
    contact_rew = float(contact_frames)
    # CLOSE 末段接触奖励 (最后 5 帧的接触帧数, 鼓励稳定夹持到窗口结束)
    last_contact_count = rollout_result.get("last_contact_count", 0)
    last_contact_rew = float(last_contact_count)
    has_contact = contact_frames >= 1
    # 提升奖励: 只在有接触时生效; 无接触的提升视为弹飞/推动, 予以惩罚
    lift = max(0.0, obj_final_z - obj_init_z)
    effective_lift = lift if has_contact else 0.0
    lift_without_contact = lift if not has_contact else 0.0
    # 抓取成功大奖励: 必须同时满足接触 + 提升 > 2cm
    grasp_success_bonus = 1.0 if (has_contact and lift > 0.02) else 0.0
    # 稳定抓取奖励: 接触持续 + 有一定提升 (>1cm)
    stable_grasp_bonus = 1.0 if (contact_frames >= 5 and lift > 0.01) else 0.0
    # 距碗距离
    dist_to_bowl = float(np.linalg.norm(np.array(obj_final_xy) - np.array(bowl_xy)))
    # 掉落惩罚
    drop_pen = 1.0 if obj_dropped else 0.0
    # 弹飞惩罚 (xy 移动超过 20cm)
    xy_drift = float(np.linalg.norm(np.array(obj_final_xy) - np.array(obj_init_xy)))
    launch_pen = max(0.0, xy_drift - 0.2)
    # 穿透惩罚
    pen_pen = max(0.0, penetration - 0.01)
    # 帧间平滑度 (帧级窗口优化: 惩罚相邻帧位姿突变)
    smoothness_cost = rollout_result.get("smoothness_cost", 0.0)

    reward = (
        - weights["w_track_pos"] * track_pos_cost
        - weights["w_track_rot"] * track_rot_cost
        - weights["w_reach"] * reach_cost
        + weights["w_min_dist"] * min_dist_rew
        + weights["w_contact"] * contact_rew
        + weights["w_last_contact"] * last_contact_rew
        + weights["w_lift"] * effective_lift
        - weights["w_lift_no_contact"] * lift_without_contact
        + weights["w_grasp_success"] * grasp_success_bonus
        + weights["w_stable_grasp"] * stable_grasp_bonus
        - weights["w_bowl"] * dist_to_bowl
        - weights["w_drop"] * drop_pen
        - weights["w_launch"] * launch_pen
        - weights["w_pen"] * pen_pen
        - weights["w_smooth"] * smoothness_cost
    )
    return reward


# ============================================================
# 粗搜奖励 (第二十一轮: 连续梯度引导)
# ============================================================
def compute_reward_coarse(rollout_result, weights=COARSE_REWARD_WEIGHTS):
    """粗搜阶段奖励 (第二十一轮: 无偏离代价, 只有连续接近信号)

    reward = -w_close_avg * avg_dist_throughout     # 全程平均距离 (连续梯度!)
             -w_min_z * gripper_obj_z_gap           # gripper 最低点与物体 z 的差距
             +w_contact * contact_frames             # 接触帧数
             -w_pen * penetration_penalty            # 穿透
             -w_launch * launch_pen                  # 弹飞

    不惩罚:
      - 位置/姿态偏离 MANO (粗搜阶段允许大幅调整)
      - 距碗距离 (粗搜阶段不可能靠近碗)
      - 掉落 (粗搜阶段不会抓到, 无所谓掉落)
      - 提升 (粗搜阶段不可能有提升)

    Args:
        rollout_result: rollout_single 返回的字典
        weights: 粗搜权重

    Returns:
        float: reward
    """
    contact_frames = rollout_result.get("contact_frames_in_close", 0)
    penetration = rollout_result.get("max_penetration", 0.0)
    obj_init_xy = rollout_result.get("obj_init_xy", [0.0, 0.0])
    obj_final_xy = rollout_result.get("obj_final_xy", [0.0, 0.0])

    # 全程平均距离 (连续梯度!)
    avg_dist_throughout = rollout_result.get("avg_dist_throughout", 1.0)
    # gripper 最低点与物体 z 的差距 (引导下降)
    gripper_obj_z_gap = rollout_result.get("gripper_obj_z_gap", 1.0)

    # 穿透
    pen_pen = max(0.0, penetration - 0.01)
    # 弹飞
    xy_drift = float(np.linalg.norm(np.array(obj_final_xy) - np.array(obj_init_xy)))
    launch_pen = max(0.0, xy_drift - 0.2)

    reward = (
        - weights["w_close_avg"] * avg_dist_throughout
        - weights["w_min_z"] * gripper_obj_z_gap
        + weights["w_contact"] * contact_frames
        - weights["w_pen"] * pen_pen
        - weights["w_launch"] * launch_pen
    )
    return reward


# ============================================================
# 旧版 9 维参数辅助函数 (第十八轮向后兼容)
# ============================================================
def apply_opt_params_to_neutral_target(
    gripper_pos, gripper_R, gripper_val, phase,
    opt_params, FINGER_FORWARD_NEUTRAL=0.037
):
    """第十八轮 9 维参数辅助函数 (保留向后兼容)"""
    if opt_params is None:
        return gripper_pos, gripper_R, gripper_val, 0.5

    if len(opt_params) != 9:
        return gripper_pos, gripper_R, gripper_val, 0.5

    grasp_pos_delta = opt_params[0:3]
    grasp_R_euler = opt_params[3:6]
    finger_close_target = float(opt_params[6])
    transport_vel_limit = float(opt_params[8])

    if phase == "CLOSE":
        gripper_pos = gripper_pos + grasp_pos_delta
        if np.linalg.norm(grasp_R_euler) > 1e-6:
            R_correction = Rotation.from_euler("xyz", grasp_R_euler).as_matrix()
            gripper_R = R_correction @ gripper_R
        gripper_val = finger_close_target

    return gripper_pos, gripper_R, gripper_val, transport_vel_limit


# ============================================================
# 帧级窗口优化参数化 (第二十六轮: 只优化靠近物体的帧)
# ============================================================
def build_window_params(grasp_window, n_total):
    """构建窗口参数映射

    Args:
        grasp_window: np.ndarray, 窗口帧索引列表 (e.g. [45,46,...,84])
        n_total: 总帧数

    Returns:
        dict: window_params with keys:
            "indices": grasp_window (M,)
            "n_window": len(grasp_window) = M
            "n_total": n_total
            "dim": M * 6  (总参数维度)
    """
    M = len(grasp_window)
    dim = M * 6  # [dx,dy,dz,droll,dpitch,dyaw] per frame
    return {
        "indices": grasp_window.copy(),
        "n_window": M,
        "n_total": n_total,
        "dim": dim,
    }


def apply_window_offset(mano_pos, mano_R, local_idx, window_params, opt_params,
                        blend_frames=5):
    """应用窗口帧偏移到当前帧, 带边界平滑blend

    如果 local_idx 在窗口中: 应用该帧的 6-DOF 偏移, 边界 blend_frames 帧做线性blend
    否则: 返回 MANO 原始位姿 (无偏移, 自然跟随)

    Args:
        mano_pos: (3,) MANO 位置
        mano_R: (3,3) MANO 姿态
        local_idx: 当前帧索引
        window_params: dict from build_window_params
        opt_params: (dim,) 优化参数
        blend_frames: 边界平滑帧数 (默认 5)

    Returns:
        gripper_pos: (3,)
        gripper_R: (3,3)
    """
    M = window_params["n_window"]
    indices = window_params["indices"]

    # 找当前帧在窗口中的位置
    for wi in range(M):
        if indices[wi] == local_idx:
            base = wi * 6
            dx, dy, dz = opt_params[base:base+3]
            droll, dpitch, dyaw = opt_params[base+3:base+6]

            # 计算偏移后的位姿
            offset_pos = mano_pos + np.array([dx, dy, dz])
            if abs(droll) + abs(dpitch) + abs(dyaw) > 1e-8:
                R_corr = Rotation.from_euler("xyz", [droll, dpitch, dyaw]).as_matrix()
                offset_R = R_corr @ mano_R
            else:
                offset_R = mano_R.copy()

            # 边界平滑: 前 blend_frames 帧和后 blend_frames 帧做线性blend
            if M <= 1:
                blend = 1.0
            elif wi < blend_frames:
                blend = wi / blend_frames  # 0→1
            elif wi >= M - blend_frames:
                blend = (M - 1 - wi) / blend_frames  # 1→0
            else:
                blend = 1.0

            if blend < 1.0:
                gripper_pos = mano_pos * (1.0 - blend) + offset_pos * blend
                # 四元数 slerp 混合旋转
                q_mano = Rotation.from_matrix(mano_R).as_quat()
                q_off = Rotation.from_matrix(offset_R).as_quat()
                q_blend = q_mano * (1.0 - blend) + q_off * blend
                q_blend = q_blend / np.linalg.norm(q_blend)  # 归一化
                gripper_R = Rotation.from_quat(q_blend).as_matrix()
            else:
                gripper_pos = offset_pos
                gripper_R = offset_R

            return gripper_pos, gripper_R

    # 不在窗口中: 跟随 MANO (无偏移)
    return mano_pos.copy(), mano_R.copy()


# ============================================================
# PyTorch CEM 优化器 (帧级窗口, 第二十六轮)
# ============================================================
def cem_window_optimize(
    rollout_fn,
    window_params,
    n_iterations=50,
    population_size=200,
    elite_frac=0.1,
    initial_std=0.25,
    pos_range=0.50,
    rot_range=0.35,
    n_workers=8,
    seed=42,
    verbose=True,
):
    """PyTorch CEM 优化 (帧级窗口)

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> {"reward": float}
        window_params: dict from build_window_params
        n_iterations: CEM 迭代轮数
        population_size: 每轮采样数
        elite_frac: 精英比例 (前 10%)
        initial_std: 初始标准差
        pos_range: 位置偏移范围 ±0.50m
        rot_range: 姿态偏移范围 ±0.35rad
        n_workers: 并行 worker 数
        seed: 随机种子

    Returns:
        best_params: np.ndarray 最优参数
        best_reward: float
        reward_history: list
    """
    import torch
    import multiprocessing as mp

    dim = window_params["dim"]
    n_elite = max(2, int(population_size * elite_frac))

    # 参数范围
    bounds_low = np.full(dim, -pos_range)
    bounds_high = np.full(dim, pos_range)
    for wi in range(window_params["n_window"]):
        base = wi * 6
        bounds_low[base+3:base+6] = -rot_range
        bounds_high[base+3:base+6] = rot_range

    # PyTorch CPU tensors
    rng = torch.Generator(device='cpu').manual_seed(seed)
    mu = torch.zeros(dim, device='cpu')
    std = torch.full((dim,), initial_std, device='cpu')

    bounds_low_t = torch.tensor(bounds_low, device='cpu')
    bounds_high_t = torch.tensor(bounds_high, device='cpu')

    best_params_np = np.zeros(dim)
    best_reward = -np.inf
    reward_history = []

    pool = mp.Pool(n_workers) if n_workers > 1 else None

    for it in range(n_iterations):
        # 采样
        noise = torch.normal(0, 1, (population_size, dim), generator=rng)
        samples = mu + std * noise
        samples = torch.clamp(samples, bounds_low_t, bounds_high_t)

        # 并行评估
        samples_np = samples.numpy()
        if pool is not None:
            rewards = list(pool.map(rollout_fn, [s for s in samples_np]))
        else:
            rewards = [rollout_fn(s) for s in samples_np]

        rewards_np = np.array([r["reward"] for r in rewards])

        # 选精英
        elite_idx = np.argsort(rewards_np)[-n_elite:]
        elite_samples_np = samples_np[elite_idx]

        # 更新均值/标准差
        mu = torch.tensor(elite_samples_np.mean(axis=0), device='cpu')
        std = torch.tensor(elite_samples_np.std(axis=0) + 1e-4, device='cpu')

        # 跟踪最优
        if rewards_np.max() > best_reward:
            best_idx = np.argmax(rewards_np)
            best_reward = float(rewards_np[best_idx])
            best_params_np = samples_np[best_idx].copy()

        reward_history.append(best_reward)

        if verbose and (it % 5 == 0 or it == n_iterations - 1):
            elite_mean_reward = float(rewards_np[elite_idx].mean())
            print(f"[CEM] iter {it}/{n_iterations}: best={best_reward:.3f}, "
                  f"elite_mean={elite_mean_reward:.3f}, "
                  f"||mu||={float(torch.norm(mu)):.4f}")

    if pool is not None:
        pool.close()
        pool.join()

    return best_params_np, best_reward, reward_history


# ============================================================
# Phase 2: 6 维 XYZ+RPY 奖励 (接触 + 提升)
# ============================================================
REWARD_WEIGHTS_PHASE2 = dict(
    w_close_avg=100.0,     # 全程平均距离
    w_min_dist=400.0,      # 最小距离奖励
    w_contact=200.0,       # 接触帧数
    w_last_contact=600.0,  # 末段 5 帧接触
    w_lift=1200.0,         # 提升量 (仅在接触时生效)
    w_grasp_success=2000.0,  # 抓取成功 (接触 + lift > 2cm)
    w_stable_grasp=1500.0,   # 稳定抓取 (contact>=5 + lift>1cm)
    w_pen=1000.0,          # 穿透惩罚
    w_launch=500.0,        # 弹飞惩罚
)


def compute_reward_phase2(rollout_result, weights=REWARD_WEIGHTS_PHASE2):
    """Phase 2 奖励: 在 Phase 1 基础上增加提升和抓取成功奖励."""
    avg_dist = rollout_result.get("avg_dist_throughout", 1.0)
    min_dist = rollout_result.get("min_dist_in_close", 1.0)
    contact_frames = rollout_result.get("contact_frames_in_close", 0)
    last_contact_count = rollout_result.get("last_contact_count", 0)
    penetration = rollout_result.get("max_penetration", 0.0)
    obj_init_z = rollout_result.get("obj_init_z", 0.0)
    obj_final_z = rollout_result.get("obj_final_z", 0.0)
    obj_init_xy = rollout_result.get("obj_init_xy", [0.0, 0.0])
    obj_final_xy = rollout_result.get("obj_final_xy", [0.0, 0.0])

    has_contact = contact_frames >= 1
    lift = max(0.0, obj_final_z - obj_init_z)
    effective_lift = lift if has_contact else 0.0

    pen_pen = max(0.0, penetration - 0.01)
    xy_drift = float(np.linalg.norm(np.array(obj_final_xy) - np.array(obj_init_xy)))
    launch_pen = max(0.0, xy_drift - 0.2)

    grasp_success_bonus = 1.0 if (has_contact and lift > 0.02) else 0.0
    stable_grasp_bonus = 1.0 if (contact_frames >= 5 and lift > 0.01) else 0.0

    reward = (
        - weights["w_close_avg"] * avg_dist
        + weights["w_min_dist"] / (min_dist + 0.01)
        + weights["w_contact"] * contact_frames
        + weights["w_last_contact"] * last_contact_count
        + weights["w_lift"] * effective_lift * 100.0
        + weights["w_grasp_success"] * grasp_success_bonus
        + weights["w_stable_grasp"] * stable_grasp_bonus
        - weights["w_pen"] * pen_pen
        - weights["w_launch"] * launch_pen
    )
    return reward


# ============================================================
# 6 维 XYZ+RPY CEM 优化器 (Phase 2)
# ============================================================
def cem_6d_optimize(
    rollout_fn,
    initial_mu_xyz=None,
    n_iterations=15,
    population_size=32,
    elite_frac=0.2,
    pos_std=0.03,
    rot_std=0.10,
    pos_range=0.50,
    rot_range=0.35,
    seed=43,
    verbose=True,
):
    """6 维 XYZ+RPY CEM 优化.

    前 3 维: 位置偏移 [dx, dy, dz], 初始均值来自 Phase 1 最优
    后 3 维: 姿态偏移 [droll, dpitch, dyaw], 初始均值 0

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> 必须返回 {"reward": float}
        initial_mu_xyz: (3,) Phase 1 最优 XYZ, 作为位置维初始均值
        n_iterations: CEM 迭代轮数
        population_size: 每轮采样数
        elite_frac: 精英比例
        pos_std: 位置维初始标准差
        rot_std: 姿态维初始标准差
        pos_range: 位置范围 ±m
        rot_range: 姿态范围 ±rad
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        best_params (6,), best_reward, reward_history
    """
    import torch

    dim = 6
    n_elite = max(2, int(population_size * elite_frac))

    bounds_low = torch.tensor(
        [-pos_range, -pos_range, -pos_range, -rot_range, -rot_range, -rot_range],
        dtype=torch.float64,
    )
    bounds_high = torch.tensor(
        [pos_range, pos_range, pos_range, rot_range, rot_range, rot_range],
        dtype=torch.float64,
    )

    rng = torch.Generator(device='cpu').manual_seed(seed)

    # 初始均值: 位置从 Phase 1 继承, 姿态为 0
    mu = torch.zeros(dim, dtype=torch.float64)
    if initial_mu_xyz is not None:
        mu[:3] = torch.tensor(initial_mu_xyz, dtype=torch.float64)
    std = torch.tensor([pos_std, pos_std, pos_std, rot_std, rot_std, rot_std],
                       dtype=torch.float64)

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
            mu_np = mu.numpy()
            print(f"[CEM-6D] iter {it}/{n_iterations}: best={best_reward:.3f}, "
                  f"xyz=[{mu_np[0]:.4f},{mu_np[1]:.4f},{mu_np[2]:.4f}], "
                  f"rpy=[{mu_np[3]:.4f},{mu_np[4]:.4f},{mu_np[5]:.4f}]")

    return best_params_np, best_reward, reward_history


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
                  f"mu={mu.numpy().round(4)}, std={std.numpy().round(4)}")

    return best_params_np, best_reward, reward_history

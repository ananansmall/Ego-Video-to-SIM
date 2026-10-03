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
# ============================================================
# 逐帧参数化 (4 固定帧, 其余优化)
# ============================================================
# 锚点比例: 起点/抓取点/释放点/终点
# 对应 0%, 44.6%, 84.8%, 100% (当前数据集 113 帧: 0, 50, 95, 112)
ANCHOR_RATIOS = [0.0, 0.446429, 0.848214, 1.0]

def compute_frame_params(n_frames):
    """根据实际帧数动态计算固定帧索引和参数维度

    Args:
        n_frames: 实际帧数 (从 _mano_gripper_traj 读取)

    Returns:
        dict with: fixed_frames, F0_IDX, F50_IDX, F95_IDX, F112_IDX,
                   n_optimized, params_dim, param_range
    """
    fixed_frames = [max(0, min(n_frames - 1, round(r * (n_frames - 1))))
                    for r in ANCHOR_RATIOS]
    F0_IDX, F50_IDX, F95_IDX, F112_IDX = fixed_frames
    n_optimized = n_frames - len(fixed_frames)
    params_dim = n_optimized * DIM_PER_KEYFRAME
    param_range = np.tile(
        np.array([[-POS_RANGE, POS_RANGE],
                  [-POS_RANGE, POS_RANGE],
                  [-POS_RANGE, POS_RANGE],
                  [-ROT_RANGE, ROT_RANGE],
                  [-ROT_RANGE, ROT_RANGE],
                  [-ROT_RANGE, ROT_RANGE],
                 ]),
        (n_optimized, 1)
    )
    return dict(
        fixed_frames=fixed_frames,
        F0_IDX=F0_IDX, F50_IDX=F50_IDX, F95_IDX=F95_IDX, F112_IDX=F112_IDX,
        n_optimized=n_optimized, params_dim=params_dim, param_range=param_range,
    )

# 向后兼容: 默认 113 帧
N_FRAMES_654 = 113
FIXED_FRAMES_654 = [0, 50, 95, 112]
N_OPTIMIZED_654 = N_FRAMES_654 - len(FIXED_FRAMES_654)
PARAMS_DIM_654 = N_OPTIMIZED_654 * DIM_PER_KEYFRAME
PARAM_RANGE_654 = np.tile(
    np.array([[-POS_RANGE, POS_RANGE],
              [-POS_RANGE, POS_RANGE],
              [-POS_RANGE, POS_RANGE],
              [-ROT_RANGE, ROT_RANGE],
              [-ROT_RANGE, ROT_RANGE],
              [-ROT_RANGE, ROT_RANGE],
             ]),
    (N_OPTIMIZED_654, 1)
)


def generate_trajectory_from_params(
    mano_pos_traj,   # (N, 3) MANO 位置轨迹
    mano_R_traj,     # (N, 3, 3) MANO 姿态轨迹
    opt_params,      # 优化后的每帧偏移
    fixed_offsets=None,  # dict {frame_idx: (6,)} 固定帧偏移, None 时全零
    fixed_frames=None,   # list 固定帧索引, None 时使用 FIXED_FRAMES_654
):
    """逐帧参数化: 4 固定帧 + 其余优化帧, 直接映射, 无需插值

    Args:
        mano_pos_traj: (N, 3) MANO 位置轨迹
        mano_R_traj: (N, 3, 3) MANO 姿态轨迹
        opt_params: 优化后的每帧偏移
        fixed_offsets: dict {frame_idx: np.ndarray(6,)} 固定帧偏移
        fixed_frames: list 固定帧索引 (动态计算)

    Returns:
        opt_pos: (N, 3) 优化后位置
        opt_R: (N, 3, 3) 优化后姿态
    """
    if fixed_frames is None:
        fixed_frames = FIXED_FRAMES_654
    fixed_set = set(fixed_frames)
    N = len(mano_pos_traj)
    all_offsets = np.zeros((N, 6), dtype=np.float64)

    # 1. 填入固定帧偏移
    if fixed_offsets is not None:
        for fi, off in fixed_offsets.items():
            if fi < N:
                all_offsets[fi] = np.asarray(off, dtype=np.float64)

    # 2. 填入优化帧偏移 (容错: opt_params 维度不够时用零填充)
    opt_idx = 0
    n_opt_available = len(opt_params) // 6
    for fi in range(N):
        if fi not in fixed_set:
            if opt_idx < n_opt_available:
                all_offsets[fi] = opt_params[opt_idx * 6:(opt_idx + 1) * 6]
            # else: 保持零偏移 (维度不匹配时回退到跟随 MANO)
            opt_idx += 1

    # 3. 逐帧计算最终位姿
    opt_pos = np.array([mano_pos_traj[fi] + all_offsets[fi, :3] for fi in range(N)])
    opt_R = []
    for fi in range(N):
        euler = all_offsets[fi, 3:]
        if np.linalg.norm(euler) > 1e-8:
            R_corr = Rotation.from_euler("xyz", euler).as_matrix()
            opt_R.append(R_corr @ mano_R_traj[fi])
        else:
            opt_R.append(mano_R_traj[fi].copy())
    opt_R = np.array(opt_R)

    return opt_pos, opt_R


# ============================================================
# MPPI 优化器 (Model Predictive Path Integral)
# ============================================================
class MPPIOptimizer:
    """MPPI 轨迹优化器

    用指数加权平均更新轨迹均值, 支持时间协方差平滑采样.
    适合高维轨迹优化 (654 维).

    Usage:
        opt = MPPIOptimizer(dim=654, n_frames=109)
        for i in range(n_iterations):
            samples = opt.ask()
            costs = [rollout_fn(s)["cost"] for s in samples]
            opt.tell(samples, np.array(costs))
        best = opt.best_mean
    """

    def __init__(self, dim, n_frames, sample_size=256, sigma=0.15,
                 lambda_=0.5, temp_length=3.0, bounds=None,
                 init_mean=None, n_iterations=None):
        self.dim = dim
        self.n_frames = n_frames
        self.K = sample_size
        self.sigma = sigma
        self.lambda_ = lambda_
        self.temp_length = temp_length
        self.bounds = bounds
        self.mean = np.zeros(dim, dtype=np.float64) if init_mean is None else np.asarray(init_mean, dtype=np.float64).copy()
        self.best_mean = self.mean.copy()
        self.best_cost = np.inf
        self._iteration = 0
        self._total_iterations = n_iterations or 30
        self._rng = np.random.default_rng(42)
        # sigma 指数衰减
        self._sigma_init = sigma
        self._sigma_min = sigma * 0.1  # 最小 sigma = 10% 初始

    def _sample_smooth(self):
        """采样带时间平滑的噪声

        用高斯滤波实现时间协方差, 避免 Cholesky 分解 654x654 矩阵.
        等效于用 RBF 核 GP 采样, 但 O(n) 而非 O(n³).
        """
        from scipy.ndimage import gaussian_filter1d
        # 从 N(0, I) 采样
        z = self._rng.standard_normal((self.K, self.n_frames, 6))
        # 高斯滤波平滑 (时间维度 axis=1)
        z_smooth = gaussian_filter1d(z, sigma=self.temp_length, axis=1)
        # 缩放
        noise = z_smooth * self.sigma
        return noise.reshape(self.K, self.dim)

    def ask(self):
        """生成下一批样本"""
        noise = self._sample_smooth()
        samples = self.mean + noise
        if self.bounds is not None:
            lo = self.bounds[:, 0]
            hi = self.bounds[:, 1]
            samples = np.clip(samples, lo, hi)
        return samples

    def tell(self, samples, costs):
        """用加权平均更新均值

        Args:
            samples: (K, dim) 样本
            costs: (K,) 每个样本的 cost (越小越好)
        """
        # 归一化 cost
        c_min, c_max = costs.min(), costs.max()
        if c_max > c_min + 1e-10:
            costs_norm = (costs - c_min) / (c_max - c_min)
        else:
            costs_norm = np.zeros_like(costs, dtype=np.float64)

        # 指数权重 (最小化 cost)
        weights = np.exp(-costs_norm / max(self.lambda_, 1e-10))
        weights /= weights.sum()

        # 加权更新均值
        self.mean = np.sum(weights[:, None] * samples, axis=0)

        # 跟踪最优
        best_idx = int(np.argmin(costs))
        if costs[best_idx] < self.best_cost:
            self.best_cost = costs[best_idx]
            self.best_mean = samples[best_idx].copy()

        # sigma 指数衰减
        frac = min(self._iteration / max(self._total_iterations - 1, 1), 1.0)
        self.sigma = self._sigma_init * ((self._sigma_min / self._sigma_init) ** frac)

        self._iteration += 1
        return self.mean

    def get_info(self):
        return {
            "iteration": self._iteration,
            "best_cost": self.best_cost,
            "mean_norm": float(np.linalg.norm(self.mean)),
            "sigma": float(self.sigma),
        }


# ============================================================
# CEM 优化器 (Cross-Entropy Method)
# ============================================================
class CEMOptimizer:
    """CEM 轨迹优化器

    直接选 top-k 精英样本, 用精英均值更新轨迹.
    比 MPPI 加权平均更激进, 适合高维快速收敛.

    Usage:
        opt = CEMOptimizer(dim=654, n_frames=109)
        for i in range(n_iterations):
            samples = opt.ask()
            costs = [rollout_fn(s)["cost"] for s in samples]
            opt.tell(samples, np.array(costs))
        best = opt.best_mean
    """

    def __init__(self, dim, n_frames, sample_size=256, sigma=0.15,
                 elite_ratio=0.1, bounds=None, init_mean=None,
                 n_iterations=None):
        self.dim = dim
        self.n_frames = n_frames
        self.K = sample_size
        self.sigma = sigma
        self.elite_ratio = elite_ratio
        self.bounds = bounds
        self.mean = np.zeros(dim, dtype=np.float64) if init_mean is None else np.asarray(init_mean, dtype=np.float64).copy()
        self.best_mean = self.mean.copy()
        self.best_cost = np.inf
        self._iteration = 0
        self._total_iterations = n_iterations or 50
        self._rng = np.random.default_rng(42)
        # sigma 指数衰减
        self._sigma_init = sigma
        self._sigma_min = sigma * 0.05  # 最小 sigma = 5% 初始

    def ask(self):
        """生成下一批样本"""
        noise = self._rng.standard_normal((self.K, self.dim))
        samples = self.mean + noise * self.sigma
        if self.bounds is not None:
            lo = self.bounds[:, 0]
            hi = self.bounds[:, 1]
            samples = np.clip(samples, lo, hi)
        return samples

    def tell(self, samples, costs):
        """用精英均值更新

        Args:
            samples: (K, dim) 样本
            costs: (K,) 每个样本的 cost (越小越好)
        """
        K = len(samples)
        n_elite = max(1, int(K * self.elite_ratio))

        # 按 cost 排序, 选精英
        elite_idx = np.argsort(costs)[:n_elite]
        elites = samples[elite_idx]

        # 更新均值 = 精英均值
        self.mean = np.mean(elites, axis=0)

        # 更新 sigma = 指数衰减 + 精英标准差混合
        elite_std = np.std(elites, axis=0).mean()
        # 指数衰减目标: sigma_init → sigma_min
        frac = min(self._iteration / max(self._total_iterations - 1, 1), 1.0)
        decay_target = self._sigma_init * ((self._sigma_min / self._sigma_init) ** frac)
        # 混合: 70% 衰减调度 + 30% 精英标准差
        self.sigma = max(0.7 * decay_target + 0.3 * elite_std, self._sigma_min)

        # 跟踪最优
        best_idx = int(np.argmin(costs))
        if costs[best_idx] < self.best_cost:
            self.best_cost = costs[best_idx]
            self.best_mean = samples[best_idx].copy()

        self._iteration += 1
        return self.mean

    def get_info(self):
        return {
            "iteration": self._iteration,
            "best_cost": self.best_cost,
            "mean_norm": float(np.linalg.norm(self.mean)),
            "sigma": float(self.sigma),
        }


# ============================================================
# Phase 1: 6 维 CMA-ES 找抓取/提起位姿 (F50, F95)
# ============================================================
def cmaes_phase1_grasp_pose(rollout_fn, n_generations=30, population_size=32,
                             sigma0=0.25, x0=None, verbose=True, seed=42):
    """6 维 CMA-ES, 只优化 F50 的 6DOF 偏移

    目标: 确定 F50(抓取位姿) 和 F95(提起位姿) 的固定偏移.
    Reward 只看抓取指标 (接触 + 提升 + 最小距离), 不看轨迹跟随.

    Args:
        x0: 初始均值 (6,), 为 None 时使用全零 (盲搜)

    Returns:
        best_offset: (6,) 最优抓取偏移
    """
    try:
        import cma
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])
        import cma

    dim = 6
    bounds = [
        [-POS_RANGE, POS_RANGE],  # dx
        [-POS_RANGE, POS_RANGE],  # dy
        [-POS_RANGE, POS_RANGE],  # dz
        [-ROT_RANGE, ROT_RANGE],  # droll
        [-ROT_RANGE, ROT_RANGE],  # dpitch
        [-ROT_RANGE, ROT_RANGE],  # dyaw
    ]
    bounds_lo = [b[0] for b in bounds]
    bounds_hi = [b[1] for b in bounds]

    options = {
        'maxiter': n_generations,
        'popsize': population_size,
        'seed': seed,
        'verbose': -1 if not verbose else 1,
        'tolfun': 1e-6,
        'tolx': 1e-4,
        'bounds': [bounds_lo, bounds_hi],
    }

    def objective(x):
        x = np.asarray(x, dtype=np.float64)
        result = rollout_fn(x)
        # Phase 1 只看抓取指标
        reward = result.get("reward_grasp_only", -100.0)
        return -reward

    initial_mean = np.zeros(dim) if x0 is None else np.asarray(x0, dtype=np.float64)
    es = cma.CMAEvolutionStrategy(initial_mean, sigma0, options)
    best_offset = np.zeros(dim)
    best_reward = -np.inf

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

        for s, f in zip(solutions, fitnesses):
            r = -f
            if r > best_reward:
                best_reward = r
                best_offset = np.asarray(s, dtype=np.float64)

        if verbose:
            es.disp()

    if verbose:
        print(f"[Phase 1] 完成: best_reward={best_reward:.3f}, "
              f"offset={best_offset.round(4)}")

    return best_offset


# ============================================================
# Phase 2: MPPI 全局优化 (654 维)
# ============================================================
def mppi_phase2_global(rollout_fn, n_frames=N_OPTIMIZED_654,
                        sample_size=256, n_iterations=30,
                        sigma=0.15, lambda_=0.5, temp_length=3.0,
                        init_mean=None, verbose=True, callback=None,
                        bounds=None):
    """MPPI 全局优化: 优化 109 帧 × 6DOF = 654 维

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> {"cost": float, "reward": float, ...}
        n_frames: 优化帧数 (109)
        sample_size: 每代采样数
        n_iterations: 代数
        sigma: 采样标准差
        lambda_: 温度参数
        temp_length: 时间协方差长度 (帧)
        init_mean: 初始化均值 (dim,), None=全零
        verbose: 是否打印进度
        callback: Callable[[MPPIOptimizer, int, np.ndarray, np.ndarray], None]
                  每迭代后的回调 (opt, iteration, costs, rewards)
        bounds: (dim, 2) 参数范围, None=不限制

    Returns:
        best_params: (654,) 最优参数
        best_cost: float
    """
    dim = n_frames * 6
    opt = MPPIOptimizer(
        dim=dim, n_frames=n_frames,
        sample_size=sample_size, sigma=sigma,
        lambda_=lambda_, temp_length=temp_length,
        bounds=bounds,
        init_mean=init_mean,
        n_iterations=n_iterations,
    )

    for it in range(n_iterations):
        samples = opt.ask()
        costs = np.zeros(len(samples))
        rewards = np.zeros(len(samples))
        for i, s in enumerate(samples):
            result = rollout_fn(s)
            costs[i] = result.get("cost", 0.0)
            rewards[i] = result.get("reward", 0.0)
        opt.tell(samples, costs)

        info = opt.get_info()
        best_reward = float(rewards[np.argmin(costs)])
        if verbose:
            print(f"[MPPI] iter {it+1}/{n_iterations}: "
                  f"best_cost={info['best_cost']:.3f}, "
                  f"best_reward={best_reward:.3f}, "
                  f"||mean||={info['mean_norm']:.3f}, "
                  f"sigma={info['sigma']:.4f}")

        if callback is not None:
            callback(opt, it + 1, costs, rewards)

    return opt.best_mean, opt.best_cost


# ============================================================
# CEM 全局优化 (654 维)
# ============================================================
def cem_phase2_global(rollout_fn, n_frames=N_OPTIMIZED_654,
                       sample_size=256, n_iterations=50,
                       sigma=0.15, elite_ratio=0.1,
                       init_mean=None, verbose=True, callback=None,
                       bounds=None):
    """CEM 全局优化: 优化 108 帧 × 6DOF = 648 维

    交叉熵方法: 每次选 top-k 精英, 用精英均值更新轨迹.
    比 MPPI 加权平均更激进, 适合高维快速收敛.

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> {"cost": float, ...}
        n_frames: 优化帧数
        sample_size: 每代采样数
        n_iterations: 代数
        sigma: 初始采样标准差
        elite_ratio: 精英比例 (默认 0.1 = 10%)
        init_mean: 初始化均值 (dim,), None=全零
        verbose: 是否打印进度
        callback: 每迭代后的回调 (opt, iteration, costs, rewards)
        bounds: (dim, 2) 参数范围, None=不限制

    Returns:
        best_params: 最优参数
        best_cost: float
    """
    dim = n_frames * 6
    opt = CEMOptimizer(
        dim=dim, n_frames=n_frames,
        sample_size=sample_size, sigma=sigma,
        elite_ratio=elite_ratio,
        bounds=bounds,
        init_mean=init_mean,
        n_iterations=n_iterations,
    )

    for it in range(n_iterations):
        samples = opt.ask()
        costs = np.zeros(len(samples))
        rewards = np.zeros(len(samples))
        for i, s in enumerate(samples):
            result = rollout_fn(s)
            costs[i] = result.get("cost", 0.0)
            rewards[i] = result.get("reward", 0.0)
        opt.tell(samples, costs)

        info = opt.get_info()
        best_reward = float(rewards[np.argmin(costs)])
        if verbose:
            print(f"[CEM] iter {it+1}/{n_iterations}: "
                  f"best_cost={info['best_cost']:.3f}, "
                  f"best_reward={best_reward:.3f}, "
                  f"||mean||={info['mean_norm']:.3f}, "
                  f"sigma={info['sigma']:.4f}")

        if callback is not None:
            callback(opt, it + 1, costs, rewards)

    return opt.best_mean, opt.best_cost


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
# Phase 1 奖励: 核心是物体跟随(follow_score) + 贴合MANO + 平滑过渡
# 修正: 去掉 contact (夹爪外侧推飞物体误导优化方向)
# ============================================================
REWARD_WEIGHTS_XYZ = dict(
    w_follow=8000.0,        # 物体跟随 (核心, 物体必须跟着夹爪走)
    w_follow_last5=8000.0,  # 末段跟随 (核心, 确保持续夹住)
    w_smooth=500.0,         # 帧间平滑 (二阶差分, 保证过渡平滑)
    w_track=100.0,          # 贴合 MANO (允许偏离, 但不能太远)
    w_close_avg=100.0,      # 全程距离 (辅助)
    w_min_dist=200.0,       # 最小距离 (辅助, 鼓励夹爪靠近物体)
    w_contact=0.0,          # 去掉! 夹爪外侧接触推飞物体
    w_pen=1000.0,           # 穿透惩罚
    w_launch=500.0,         # 弹飞惩罚
)


def compute_reward_xyz(rollout_result, weights=REWARD_WEIGHTS_XYZ):
    """Phase 1: 抓取位姿优化奖励.

    核心指标: 物体跟随夹爪移动 (follow_score)
      - 改进: 加 "物体移动" + "距离合理" 两个条件, 排除假阳性
      - 假阳性1: 物体不动 → 偏移也稳定 → 加"物体必须移动"条件
      - 假阳性2: 物体被推飞 → 偏移也稳定 → 加"距离合理"条件

    辅助指标:
      - 贴合 MANO 轨迹 (w_track, F50除外)
      - 帧间平滑 (w_smooth, 二阶差分)
      - 不穿透、不弹飞
    """
    avg_dist = rollout_result.get("avg_dist_throughout", 1.0)
    min_dist = rollout_result.get("min_dist_in_close", 1.0)
    contact_frames = rollout_result.get("contact_frames_in_close", 0)
    penetration = rollout_result.get("max_penetration", 0.0)
    follow_score = rollout_result.get("follow_score", 0.0)
    follow_score_last5 = rollout_result.get("follow_score_last5", 0.0)
    smoothness_cost = rollout_result.get("smoothness_cost", 0.0)
    obj_init_z = rollout_result.get("obj_init_z", 0.0)
    obj_final_z = rollout_result.get("obj_final_z", 0.0)
    obj_init_xy = rollout_result.get("obj_init_xy", [0.0, 0.0])
    obj_final_xy = rollout_result.get("obj_final_xy", [0.0, 0.0])
    avg_dist_last5 = rollout_result.get("avg_dist_in_close_last5", 1.0)
    # w_track: 过渡帧偏移 MANO 轨迹的范数 (越小 = 越贴合 MANO, 由 rollout_single 计算)
    track_pen_val = rollout_result.get("track_pen", 0.0)

    pen_pen = max(0.0, penetration - 0.01)
    xy_drift = float(np.linalg.norm(np.array(obj_final_xy) - np.array(obj_init_xy)))
    launch_pen = max(0.0, xy_drift - 0.2)

    # w_track: 过渡帧偏离 MANO 的惩罚 (越小=越贴合 MANO, 由 rollout_single 计算)
    # track_pen_val 是过渡帧 (非 F50) 的 CEM 偏移范数均值
    track_pen = track_pen_val  # 直接使用 rollout_single 计算的过渡帧偏移量

    # min_dist 奖励加上限, 防止 min_dist 很小时主导奖励
    # min_dist=0.0002m 时, 200/0.0102=19608, 加上限 2000 防止主导
    min_dist_reward = min(weights["w_min_dist"] / (min_dist + 0.01), 2000.0)

    reward = (
        - weights["w_close_avg"] * avg_dist
        + min_dist_reward
        + weights["w_contact"] * contact_frames
        + weights["w_follow"] * follow_score
        + weights["w_follow_last5"] * follow_score_last5
        - weights["w_smooth"] * smoothness_cost
        - weights["w_track"] * track_pen
        - weights["w_pen"] * pen_pen
        - weights["w_launch"] * launch_pen
    )
    return reward


# ============================================================
# 抓取窗口 CEM 优化: 动态窗口 (N_TRANS*2+1 帧 × 6DOF)
# Stage 1: 让夹爪到达物体位置并夹住
# 修正: 动态窗口大小 + 自适应搜索范围 (F50小范围, 过渡帧中范围)
# ============================================================
def cem_grasp_window_optimize(
    rollout_fn,
    n_iterations=25,
    population_size=64,
    elite_frac=0.2,
    pos_std=0.08,
    rot_std=0.15,
    pos_range=0.30,
    rot_range=0.35,
    seed=42,
    verbose=True,
    init_mu=None,
    n_frames=5,
    f50_local_idx=2,
    pos_std_array=None,
    rot_std_array=None,
    frame_indices=None,
):
    """动态维度抓取窗口 CEM 优化 (n_frames 帧 × 6DOF).

    每帧6维: [dx, dy, dz, droll, dpitch, dyaw]
    前3维: 位置偏移, 后3维: 姿态偏移

    Args:
        rollout_fn: Callable[[np.ndarray], dict] -> 必须返回 {"reward": float}
        n_iterations: CEM 迭代轮数
        population_size: 每轮采样数
        elite_frac: 精英比例
        pos_std: 位置维初始标准差 (m) (标量, 当 pos_std_array 为 None 时使用)
        rot_std: 姿态维初始标准差 (rad) (标量, 当 rot_std_array 为 None 时使用)
        pos_range: 位置偏移范围 ±m
        rot_range: 姿态偏移范围 ±rad
        seed: 随机种子
        verbose: 是否打印进度
        init_mu: 初始均值 (dim,) 或 None
        n_frames: 窗口帧数 (默认5, 动态自适应)
        f50_local_idx: F50 在窗口中的局部索引 (默认2)
        pos_std_array: 每帧位置标准差数组 (n_frames,), 若提供则覆盖 pos_std
        rot_std_array: 每帧姿态标准差数组 (n_frames,), 若提供则覆盖 rot_std
        frame_indices: 窗口帧的全局索引列表 (用于打印)

    Returns:
        best_params: np.ndarray (n_frames*6,)
        best_reward: float
        reward_history: list
    """
    import torch

    dim = n_frames * 6
    n_elite = max(2, int(population_size * elite_frac))

    # bounds: n_frames 帧, 每帧 [dx, dy, dz, droll, dpitch, dyaw]
    bounds_low = torch.tensor(
        [-pos_range, -pos_range, -pos_range, -rot_range, -rot_range, -rot_range] * n_frames,
        dtype=torch.float64,
    )
    bounds_high = torch.tensor(
        [pos_range, pos_range, pos_range, rot_range, rot_range, rot_range] * n_frames,
        dtype=torch.float64,
    )
    rng = torch.Generator(device='cpu').manual_seed(seed)

    # 初始均值: 从物体位置初始化 (如果提供) 或零偏移
    if init_mu is not None:
        mu = torch.tensor(init_mu, dtype=torch.float64)
    else:
        mu = torch.zeros(dim, dtype=torch.float64)

    # 初始标准差: 支持每帧不同 (自适应搜索范围)
    if pos_std_array is not None and rot_std_array is not None:
        per_frame_std_list = []
        for fi in range(n_frames):
            per_frame_std_list.extend([
                pos_std_array[fi], pos_std_array[fi], pos_std_array[fi],
                rot_std_array[fi], rot_std_array[fi], rot_std_array[fi]
            ])
        std = torch.tensor(per_frame_std_list, dtype=torch.float64)
    else:
        per_frame_std = [pos_std, pos_std, pos_std, rot_std, rot_std, rot_std]
        std = torch.tensor(per_frame_std * n_frames, dtype=torch.float64)

    best_params_np = mu.numpy().copy() if init_mu is not None else np.zeros(dim, dtype=np.float64)
    best_reward = -np.inf
    reward_history = []

    # F50 硬约束: F50 偏移固定为 init_mu, 不参与 CEM 漂移
    _f50_s = f50_local_idx * 6 if f50_local_idx is not None else None
    _f50_e = _f50_s + 6 if _f50_s is not None else None
    _f50_init = (torch.tensor(init_mu[_f50_s:_f50_e], dtype=torch.float64)
                 if init_mu is not None and _f50_s is not None else None)

    for it in range(n_iterations):
        noise = torch.normal(0, 1, (population_size, dim), generator=rng, dtype=torch.float64)
        samples = torch.clamp(mu + std * noise, bounds_low, bounds_high)
        # F50 硬约束: 所有样本的 F50 偏移固定为 init_mu
        if _f50_init is not None:
            samples[:, _f50_s:_f50_e] = _f50_init
        samples_np = samples.numpy()

        rewards = np.array([rollout_fn(s)["reward"] for s in samples_np])

        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_samples = samples_np[elite_idx]

        mu = torch.tensor(elite_samples.mean(axis=0), dtype=torch.float64)
        std = torch.tensor(elite_samples.std(axis=0) + 1e-4, dtype=torch.float64)
        # F50 mu/std 恢复为 init_mu/极小 (不漂移)
        if _f50_init is not None:
            mu[_f50_s:_f50_e] = _f50_init
            std[_f50_s:_f50_e] = 1e-4

        if rewards.max() > best_reward:
            best_reward = float(rewards.max())
            best_params_np = samples_np[np.argmax(rewards)].copy()

        reward_history.append(best_reward)

        if verbose and (it % 3 == 0 or it == n_iterations - 1):
            # 打印每帧的偏移
            mu_np = mu.numpy()
            frame_strs = []
            for fi in range(n_frames):
                xyz = mu_np[fi*6:fi*6+3]
                rpy = mu_np[fi*6+3:fi*6+6]
                gidx = frame_indices[fi] if frame_indices is not None else (f50_local_idx - n_frames//2 + fi + 48)
                marker = "★" if fi == f50_local_idx else " "
                frame_strs.append(f"{marker}F{gidx}:xyz=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]rpy=[{rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f}]")
            print(f"[CEM-GraspWindow] iter {it}/{n_iterations}: best={best_reward:.1f}, "
                  f"mean_reward={rewards.mean():.1f}, dim={dim}")
            for fs in frame_strs:
                print(f"  {fs}")

    return best_params_np, best_reward, reward_history


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

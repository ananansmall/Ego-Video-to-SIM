#!/usr/bin/env python3
"""large_range_track.py — 大范围移动任务的**通用内核** (与具体数据集解耦)。

背景: chairbottle / 后续大范围数据集 都需要"底盘跟随手部 + 限位感知 IK + 多机位渲染"。
本模块只保留其中**与任务无关**的部分; 任务脚本只负责喂数据 (世界系目标位姿序列) 并调用它。

泛化点 (均从机器人运动学/数据自动推导, 无任何按任务的硬编码或手工标定):
  - comfort_frame(): 采样机器人工作空间, 自动求「底盘偏置」与「手高对齐中心」
  - fit_height():    把任意高度的手序列平移到机器人舒适工作高度 (带工作空间夹取)
  - track_base():    底盘跟随手部水平投影 (世界系) —— 大范围移动的核心
  - solve_arm_track(): 限位感知 IK, 逐帧多起点 + 连续性优选 (通用, 接任意目标序列)
  - add_path_markers()/auto_cameras()/ego_cam_pose(): 可视化与相机自动布置

坐标/约定: 本模块不含机器人常量, 一律由调用方传入 (se2_pose / fk_pad / resolve 等原语)。
"""
import numpy as np
import sapien.core as sapien


# ============================================================
# 1. 工作空间 / 舒适帧 (泛化: 免手工标定)
# ============================================================
def comfort_frame(robot, side, sel, lim, fk_pad, n=8000,
                  reach_lo=0.20, reach_hi=0.60, seed=0, forward=(0.0, 1.0)):
    """采样机器人工作空间, 自动求底盘偏置与手高对齐中心。

    底盘偏置 = 肩部水平位置 + 机器人前方(forward)方向 * 舒适前伸距离。
      - 「手→底盘」向量由此恒定, 落在臂舒适区 (避免整体出可达域);
      - 取**前方**而非工作空间质心, 因为质心会把工作点偏到机器人侧面
        (肩本身偏侧), 导致第一视角/自然作业姿态不合理。
      - forward 为机器人基座系的前方单位向量 (机器人约定常量, 与任务无关)。

    返回 dict: base_off(2,), z_center(float), head_z, ws_z, shoulder_rel
    """
    rng = np.random.default_rng(seed)
    nj = len(lim)
    Q = rng.uniform(lim[:, 0], lim[:, 1], size=(n, nj))
    base_p = np.asarray(robot.get_root_pose().p, np.float64)
    shoulder = robot.find_link_by_name(f"{side}_arm_link1").get_entity_pose().p
    P = np.array([fk_pad(robot, side, sel, q)[0] for q in Q])
    rel = P - base_p
    d = np.linalg.norm(P - shoulder, axis=1)
    m = (d > reach_lo) & (d < reach_hi)
    if m.sum() < 50:                       # 兜底: 放松半径条件
        m = d < reach_hi
    sh_rel = shoulder - base_p
    reach_front = 0.5 * (reach_lo + reach_hi)            # 底座前伸 = 臂舒适带中心 (位置+姿态都最有余量)
    fwd = np.asarray(forward, np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    base_off = sh_rel[:2] + fwd * reach_front
    z_center = float(rel[m].mean(0)[2])
    head_z = float(robot.find_link_by_name("torso_link4").get_entity_pose().p[2]) + 0.15
    return {"base_off": base_off, "z_center": z_center, "head_z": head_z, "forward": fwd,
            "ws_z": (float(rel[:, 2].min()), float(rel[:, 2].max())),
            "shoulder_rel": sh_rel}


def fit_height(pos, z_center, ws_z, margin=0.10):
    """把手序列整体平移, 使其高度中位对齐 z_center; 并夹取到工作空间 z 范围内。"""
    pos = pos.copy()
    zc = 0.5 * (pos[:, 2].min() + pos[:, 2].max())
    shift = z_center - zc
    lo, hi = ws_z[0] + margin, ws_z[1] - margin
    if pos[:, 2].min() + shift < lo:       # 顶到下限则上移
        shift = lo - pos[:, 2].min()
    if pos[:, 2].max() + shift > hi:       # 顶到上限则下移
        shift = hi - pos[:, 2].max()
    pos[:, 2] += shift
    return pos, float(shift)


# ============================================================
# 2. 底盘跟随 (泛化: 任意世界系手部轨迹)
# ============================================================
def track_base(pos_xy, base_off, base_ref_xy, smooth_win=0, smooth_fn=None):
    """底盘跟随手部水平投影: base = hand_xy - base_off。

    smooth_win>0 时对底盘水平轨迹做零相位对称滑动平均 (需传 smooth_fn, 如 rg.smooth_traj),
    抑制逐帧跟随的抖动; 平滑后底盘不再严格等于手投影, 残差由手臂吸收 (IK 仍逐帧解)。
    返回 (base_xy, dx, dy); dx/dy 相对初始底盘。
    """
    base_xy = pos_xy - base_off
    if smooth_win and smooth_win >= 3 and smooth_fn is not None:
        base_xy = smooth_fn(base_xy, smooth_win, 1)
    return base_xy, base_xy[:, 0] - base_ref_xy[0], base_xy[:, 1] - base_ref_xy[1]


def track_base_se2(pos_xy, yaw_ref, base_off, base_ref_xy, smooth_win=0, smooth_fn=None):
    """SE(2) 底盘跟随: 手保持在**机体坐标系**的舒适偏置处 (偏置随朝向参考旋转)。

    track_base 的偏置是固定世界向量, 隐含"底盘朝向基本不变"。朝向大幅旋转时
    (人绕作业点走一圈), 固定偏置会让手转到机器人侧后方 -> 手臂不可达 (实测
    chairbottle: ref_yaw 幅度 253° 而 base 固定 -> mean 误差 274mm)。这里把
    偏置用 Rz(yaw_ref) 旋转, 手始终在机体前方 base_off 处 —— 与人行走时手
    在身前的几何一致。yaw_ref 用**平滑/限速后**的朝向参考, 保证底盘路径低曲率。
    返回 (base_xy, dx, dy); dx/dy 相对初始底盘。
    """
    c = np.cos(np.asarray(yaw_ref, float))
    s = np.sin(np.asarray(yaw_ref, float))
    ox, oy = float(base_off[0]), float(base_off[1])
    off = np.stack([c * ox - s * oy, s * ox + c * oy], axis=1)   # Rz(yaw) @ base_off
    base_xy = pos_xy - off
    if smooth_win and smooth_win >= 3 and smooth_fn is not None:
        base_xy = smooth_fn(base_xy, smooth_win, 1)
    return base_xy, base_xy[:, 0] - base_ref_xy[0], base_xy[:, 1] - base_ref_xy[1]


# ============================================================
# 3. 限位感知 IK 跟踪 (泛化: 逐帧多起点 + 连续性优选)
# ============================================================
def solve_arm_track(robot, side, sel, lim, dx, dy, byaw, tgt, pad_p,
                    se2_pose, fk_pad, resolve, n_local=10, n_rand=10,
                    w_cont=2.0, w_ori=8.0, seed=0, seed_arm=None):
    """对整条目标序列做限位感知 IK (位置 + 完整姿态)。

    每帧: 以上一帧解 + 小扰动为主 (保连续), 辅以随机全局起点 (逃局部极小)。
    候选综合打分 (越小越好):
      score = 位置误差(mm) + w_ori * 姿态测地误差(deg) + w_cont * max|q-q_prev|(deg)
    - 位置与姿态**同时**计入, 避免"位置对但朝向 180° 反了"这类退化解被选中;
    - 加连续性项, 避免冗余自由度下的分支跳变 (100°+ 帧间跳)。
    seed_arm: 可选 (N,6) 外部初值 (如 WBC 解)。给定后逐帧热启动用 seed_arm[k],
      配合 n_rand=0 即为「纯局部、强连续」重解 —— 用于底盘平滑后保持臂分支不变
      (全局随机起点会跳到 180° 反向的等价分支, 是臂大跳的来源)。
    返回 (arm(N,6), pos_err(N,) mm, ori_err(N,) deg)。
    """
    N = len(pad_p)
    rng = np.random.default_rng(seed)
    nj = len(lim) if lim is not None else 6
    arm = np.zeros((N, nj)); pos_err = np.zeros(N); ori_err = np.zeros(N)

    def score_cand(qc, k):
        p, R = fk_pad(robot, side, sel, qc)
        pe = float(np.linalg.norm(p - pad_p[k])) * 1000.0
        R0 = tgt[k][1]
        c = (np.trace(R0.T @ R) - 1.0) / 2.0
        oe = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        return pe, oe

    def solve_best(k, prev):
        starts = []
        if seed_arm is not None:
            # 外部种子 (如速率限幅后的关节轨迹): 每帧都作为一个起点,
            # 让链条能沿着"摊平后的过渡"走, 而不是只认上一帧解。
            sk = np.asarray(seed_arm, float)[k]
            starts.append(sk)
            starts += [sk + rng.normal(0, 0.06, nj) for _ in range(max(2, n_local // 2))]
        if prev is not None:
            starts.append(prev)
            starts += [prev + rng.normal(0, 0.12, nj) for _ in range(n_local)]
        starts += [rng.uniform(lim[:, 0], lim[:, 1]) for _ in range(n_rand)]
        robot.set_root_pose(se2_pose(dx[k], dy[k], byaw[k]))
        best, best_s, best_pair = None, 1e18, (1e18, 1e18)
        for s0 in starts:
            qc = resolve(robot, side, sel, s0, lim[:, 0], lim[:, 1], target=tgt[k], seed=s0)
            pe, oe = score_cand(qc, k)
            s = pe + w_ori * oe
            if prev is not None:
                s += w_cont * float(np.degrees(np.abs(qc - prev)).max())
            if s < best_s:
                best_s, best, best_pair = s, qc, (pe, oe)
        return best, best_pair

    prev0 = None if seed_arm is None else np.asarray(seed_arm, float)[0]
    arm[0], (pos_err[0], ori_err[0]) = solve_best(0, prev0)
    for k in range(1, N):
        # 链式: 从**刚解出的上一帧**局部搜索, 保证每帧解是上一帧的局部延续
        # (从机制上消除冗余分支跳变)。若提供 seed_arm, 额外把 seed_arm[k] 作为一个
        # 起点 —— 用于"分支过渡摊平"后把种子拉回目标 (见 rate_limit_joints)。
        arm[k], (pos_err[k], ori_err[k]) = solve_best(k, arm[k - 1])
    return arm, pos_err, ori_err


def rate_limit_joints(arm, step_max):
    """关节轨迹速率限幅 (前后双向), 把单帧大跳变摊成多帧线性过渡。

    背景: 冗余臂在个别帧必须切换分支 (实测 j6 +165°→-157°, 物理上等价于 -38°,
    但限位 ±165° 不允许穿过 180°, 只能绕 0 走 322°)。硬步长上界会把链条锁死在
    坏分支 (误差 267mm); 放开又会单帧 126° 抽搐。正解是**允许换分支但把过渡摊开**:
    先对关节轨迹做速率限幅得到平滑"种子", 再用 IK 逐帧投影回目标。

    arm: (N,6) 弧度; step_max: 每帧最大变化 (弧度)。
    返回 (N,6) 限幅后的关节轨迹 (仅作 IK 种子, 不保证末端精度)。
    """
    q = np.array(arm, float, copy=True)
    N = len(q)
    if N < 2:
        return q
    for k in range(1, N):                      # 前向
        q[k] = np.clip(q[k], q[k - 1] - step_max, q[k - 1] + step_max)
    for k in range(N - 2, -1, -1):             # 反向 (对称, 避免整体偏移)
        q[k] = np.clip(q[k], q[k + 1] - step_max, q[k + 1] + step_max)
    return q


def solve_arm_beam(robot, side, sel, lim, dx, dy, byaw, tgt, pad_p,
                   se2_pose, fk_pad, resolve, beam=6, n_seed=10,
                   w_ori=8.0, w_move=1.5, w_limit=2.0, w_posture=0.5,
                   step_max_deg=35.0, seed=0):
    """臂的**多分支 beam search** (全局一致分支 + 硬步长 + 姿态正则)。

    为什么需要 (单链条的固有病): 逐帧链式局部搜索只会沿着一条分支走, 一旦某个关节
    贴到限位 (实测 F213 j4=-165°、j5=88°), 下一帧只能跳到**完全不同的分支**
    (实测 j1 42°→-132°, 单帧 173°) 才能跟上目标 —— 机器上不可执行。反之用硬步长
    锁死又会导致够不到 (误差 260mm+)。

    beam search 的思路: 每帧保留 B 条**不同分支**的假设, 各自从上一帧对应假设局部
    求解, 用「跟踪误差 + 姿态误差 + 步长代价」累加打分, 保留最优 B 条; 最后回溯。
    这样在还没贴近限位之前, 若有另一条分支能让后续走得更远, 它会胜出 ——
    即**提前**换到好分支, 而不是被逼到限位后突然翻。

    step_max_deg: 相邻帧单关节硬步长上限 (超过则该转移不可行, 直接剪枝)。
    返回 (arm(N,6), pos_err(N,)mm, ori_err(N,)deg)。
    """
    N = len(pad_p)
    rng = np.random.default_rng(seed)
    q_pref = 0.5 * (lim[:, 0] + lim[:, 1])
    q_margin = np.radians(12.0)

    def _barrier(q):
        below = np.maximum(0.0, (lim[:, 0] + q_margin) - q)
        above = np.maximum(0.0, q - (lim[:, 1] - q_margin))
        return np.concatenate([below, above])

    def _set_base(k):
        robot.set_root_pose(se2_pose(dx[k], dy[k], byaw[k]))

    def _eval(k, q):
        _set_base(k)
        p, R = fk_pad(robot, side, sel, q)
        pe = float(np.linalg.norm(p - pad_p[k])) * 1000.0
        R0 = tgt[k][1]
        c = (np.trace(R0.T @ R) - 1.0) / 2.0
        oe = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        return pe, oe

    def _resolve(k, q0):
        _set_base(k)
        return resolve(robot, side, sel, q0, lim[:, 0], lim[:, 1],
                       target=tgt[k], seed=q0)

    def _frame_cost(k, q):
        pe, oe = _eval(k, q)
        pen = (w_limit * float(np.linalg.norm(_barrier(q)))
               + w_posture * float(np.linalg.norm(q - q_pref)))
        return pe + w_ori * oe + pen, pe, oe

    def _dedup(cands, keep):
        """按 20° 量化去重, 保留代价最小的 keep 条。cands: [(cost, q, pe, oe, parent)]"""
        cands = sorted(cands, key=lambda t: t[0])
        out, seen = [], set()
        for c in cands:
            key = tuple(np.round(np.degrees(c[1]) / 20.0).astype(int).tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= keep:
                break
        return out

    # 第 0 帧: 全局多种子
    c0 = []
    for _ in range(n_seed):
        q = _resolve(0, rng.uniform(lim[:, 0], lim[:, 1]))
        cost, pe, oe = _frame_cost(0, q)
        c0.append((cost, q, pe, oe, -1))
    level0 = _dedup(c0, beam)

    levels = [level0]
    parents = []
    for k in range(1, N):
        new = []
        for bi, (cost_prev, q_prev, _pe, _oe, _p) in enumerate(levels[-1]):
            q = _resolve(k, q_prev)
            cost, pe, oe = _frame_cost(k, q)
            dq = float(np.degrees(np.abs(q - q_prev)).max())
            if dq > step_max_deg:
                continue                        # 硬步长: 该转移不可行
            new.append((cost_prev + cost + w_move * dq, q, pe, oe, bi))
        if not new:                             # 全灭 -> 允许全局重启兜底
            for _ in range(n_seed):
                q = _resolve(k, rng.uniform(lim[:, 0], lim[:, 1]))
                cost, pe, oe = _frame_cost(k, q)
                new.append((cost, q, pe, oe, -1))
        levels.append(_dedup(new, beam))

    # 回溯最优路径
    nj = len(lim) if lim is not None else 6
    arm = np.zeros((N, nj)); pos_err = np.zeros(N); ori_err = np.zeros(N)
    bi = int(np.argmin([c[0] for c in levels[-1]]))
    for k in range(N - 1, -1, -1):
        cost, q, pe, oe, par = levels[k][bi]
        arm[k] = q; pos_err[k] = pe; ori_err[k] = oe
        bi = par if par >= 0 else 0
    return arm, pos_err, ori_err


# ============================================================
# 3a. WBC 参数自动选择 (泛化: 免按轨迹手工调参)
# ============================================================
# 标定依据 (chairbottle, 4 档可达性难度 × 4 组 w_base, 见
# results/chairbottle_wbc_adaptive_calib.json): 折中最优在每个难度档**稳定落在
# w_base≈12, w_bcont≈5**。
#
# 为什么不需要"按轨迹分别调参": 泛化职责已由本模块的自动归一化承担 ——
#   comfort_frame 自动定底盘偏置/手高中心, fit_height 把手高夹进舒适区,
#   track_base 让底盘严格跟随手部水平投影。三者合起来把"剩余压力"压得很小,
#   实测压力代理 (目标到工作空间的间隙) 不随难度单调 (47→67→60→52mm)。
#   即: 位置误差的主导量不是可达性, 而是高度/姿态的残余冲突, 与轨迹难度不相关。
# 故采用**稳健基准**: 所有轨迹共用同一套参数, 满足"不要每条轨迹不同参数"。
# 预留扩展点: 后续接入更多大范围数据集, 若出现显著更高压力档, 只需在
# auto_wbc_params 内按 pressure 分档细化即可。
#
# 注意 base_range 必须与 w_base 配套: 标定在 base_range=1.0 下进行。低 w_base 时
# 底盘需更大活动范围才能压精度 (w_base=12 时 base_range 0.15→1.0 使位置 62→52mm);
# 靠 w_bcont=5 的帧间连续性惩罚把底盘步长仍约束在 54mm/帧内, 故放宽不会引入跳变。
#
# 2026-10-20 重标定 (chairbottle, 无效帧插值 + SLAM 段对齐修复后):
#   w_base  位置mean  臂max|Δq|  底盘步长max
#    12     66.9mm    21.8°      54mm
#     6     49.2mm    30.6°     101mm
#     3     34.7mm    23.9°     113mm   <- 最佳平衡 (精度↑近半, 臂跳变与默认相当)
#     1     28.8mm    41.3°     194mm
# 尾段 z≈-0.64 的 ~8 帧为 R1 臂向下物理不可达 (数据边界), 各档均存在, 非参数可解。
# 故稳健基准 w_base 由 12 下调至 3 (精度优先, 臂平滑度几乎不损)。
WBC_PARAMS_ROBUST = {"w_base": 3.0, "w_bcont": 5.0, "base_range": 1.0}


def auto_wbc_params(pressure_p90_mm=None, **overrides):
    """自动选择 WBC 权重 (泛化, 免手工调参): 返回稳健基准, 所有轨迹共用。

    pressure_p90_mm: 可选诊断量 (目标相对跟随底盘到工作空间点云的间隙, mm)。
      当前标定表明其区分度弱, 故不影响返回值; 保留供后续数据分档扩展与日志记录。
    overrides: 显式覆盖 (CLI/配置指定时, 值为 None 的键忽略), 覆盖基准值。
    """
    p = dict(WBC_PARAMS_ROBUST)
    p.update({k: v for k, v in overrides.items() if v is not None})
    p["_pressure_p90_mm"] = pressure_p90_mm
    return p


# ============================================================
# 3b. 全身 IK (WBC): 底盘 + 臂 联合求解 (工业界移动操作主流做法)
# ============================================================
def lead_base_reference(body_xy, reach_xy, states, lead_frames):
    """Reach each grasp/release stance lead_frames before the gripper event."""
    frames = np.arange(len(body_xy))
    active = np.asarray(states)[:, None] >= [2, 5]
    events = np.flatnonzero(np.any(active & ~np.vstack([np.zeros(2, bool), active[:-1]]), axis=1))
    correction = np.zeros_like(body_xy)
    weight = np.zeros(len(body_xy))
    for k in events:
        distance = np.maximum(k - lead_frames - frames, frames - k)
        blend = np.clip(1.0 - distance / lead_frames, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        correction += blend[:, None] * (reach_xy[k] - body_xy)
        weight += blend
    return body_xy + correction / np.maximum(1.0, weight[:, None])


def torso_link_pitch(q):
    """Absolute R1 torso link pitches. The third joint rotates about -Y."""
    return np.cumsum(np.asarray(q) * [1, 1, -1], axis=-1)


def torso_pitch_coordinates(values, lo, hi, pitch, encode=False):
    """Map a unit cube onto joint bounds intersected with all three R1 pitch limits."""
    p1, p2, p3 = np.broadcast_to(pitch, (3,))
    sum_lo, sum_hi = max(-p2, lo[2] - p3), min(p2, hi[2] + p3)
    if sum_lo > sum_hi + 1e-12:
        raise ValueError("joint bounds and torso pitch constraint have no common solution")
    low = max(lo[0], -p1, sum_lo - hi[1])
    high = min(hi[0], p1, sum_hi - lo[1])
    angles, unit = [], []
    for i, value in enumerate(values):
        if i == 1:
            low, high = max(lo[1], sum_lo - angles[0]), min(hi[1], sum_hi - angles[0])
        elif i == 2:
            low, high = max(lo[2], sum(angles) - p3), min(hi[2], sum(angles) + p3)
        if high < low - 1e-12:
            raise ValueError("joint bounds and torso pitch constraint have no common solution")
        span = max(0.0, high - low)
        angle = np.clip(value, low, low + span) if encode else low + span * value
        angles.append(angle)
        unit.append((angle - low) / span if span else 0.5)
    return np.array(unit if encode else angles)


def stable_grasp_width(gaps, finger_impulses, collider_width):
    """Recent measured gap with bilateral load, expressed in meters."""
    gaps = np.asarray(gaps)
    impulses = np.asarray(finger_impulses)
    if (gaps.shape != (3,) or impulses.shape != (3, 2)
            or not np.isfinite(impulses).all() or not np.all(impulses > 1e-7)
            or not np.isfinite(gaps).all() or np.ptp(gaps) > 0.003):
        return None
    width = float(np.median(gaps))
    return width if abs(width - collider_width) <= 0.003 else None


def supported_rest(positions, dynamic, impulses, release, frames=15):
    """Require an actual released, unfrozen object resting on its intended support."""
    contact = np.asarray(impulses) > 1e-7
    independent = dynamic & contact[:, 2] & ~contact[:, 3] & ~contact[:, :2].any(axis=1)
    if release is None or len(positions) - release < frames:
        return False
    if not np.all(independent[-frames:]) or contact[release:, 3].any():
        return False
    return bool(np.linalg.norm(np.ptp(positions[-frames:], axis=0)) <= 0.005)


def ring_approach(pad, hold_start, place_start, release, center, top, radius, half_box, carry_half_z):
    """Clear the ring during the return, then move above center and descend."""
    if not 0 <= hold_start < place_start < release < len(pad) or release - place_start < 6:
        raise ValueError("ring approach needs HOLD, PLACE and RELEASED frames")
    plan = np.asarray(pad, dtype=float).copy()
    half_box = np.asarray(half_box)
    radial_clearance = radius + np.linalg.norm(half_box)
    z_above = top + carry_half_z + 0.035
    distance = np.linalg.norm(plan[hold_start:place_start, :2] - center, axis=1)
    w = np.clip(1 - (distance - radial_clearance) / 0.10, 0, 1)
    w = w*w*(3 - 2*w)
    plan[hold_start:place_start, 2] += w * np.maximum(z_above - plan[hold_start:place_start, 2], 0)
    start = plan[place_start - 1].copy()
    mid = place_start + (release - place_start)//2
    hover = np.r_[center, max(z_above, start[2])]
    lift_end = place_start + (mid - place_start)//2
    above_start = np.r_[start[:2], hover[2]]
    for a, b, p0, p1 in ((place_start, lift_end, start, above_start),
                         (lift_end, mid, above_start, hover)):
        u = np.linspace(0, 1, b-a+1)
        w = (u*u*(3 - 2*u))[:, None]
        plan[a:b+1] = p0 + w*(p1-p0)
    u = np.linspace(0, 1, release - mid + 1)
    w = u*u*(3 - 2*u)
    plan[mid:release+1, :2] = center
    # [ee17] 落座终点: 降到环面下方让锥肩(杯口下方凸缘)排架在环缘 —— 不再是
    # 悬停在环面上方 (v13 切 dynamic 后自由落 ~350mm 才楔到锥肩, 撤爪后沿锥面
    # 下滑 477mm 的根因)。终点 = 环顶 - 0.20 (越过杯底、锥肩压在环上)。
    plan[mid:release+1, 2] = hover[2] + w * (top - 0.20 - hover[2])
    # Retract vertically before moving away. Keep continuity with the source path.
    end = min(len(plan), release + 18)
    if end > release + 1:
        u = np.linspace(0, 1, end - release)
        w = (u*u*(3 - 2*u))[:, None]
        retract = plan[release] + w * np.array([0, 0, 0.065])
        tail = np.linspace(0, 1, end - release)
        blend = (np.maximum(0, 2*tail-1)**2 * (3-2*np.maximum(0, 2*tail-1)))[:, None]
        plan[release+1:end] = (retract*(1-blend) + plan[release:end]*blend)[1:]
    return plan


def progressive_motion_residual(value, knee, radial=False):
    """Quadratic at small excursions, increasingly costly near the motion envelope."""
    magnitude = np.linalg.norm(value) if radial else value
    return value * np.hypot(1., magnitude/knee)


def solve_wbc_track(robot, side, sel, lim, ref_dx, ref_dy, tgt, pad_p,
                    se2_pose, fk_pad, w_pos=10.0, w_ori=10.0,
                    w_base=4.0, w_yaw=3.0, base_range=0.35, yaw_max=0.5,
                    use_yaw=True, n_local=4, n_rand=3, w_cont=1.0, w_bcont=0.4,
                    cont_arm_cap=60.0, base_step_max=0.05, joint_step_max=180.0,
                    w_limit=0.0, limit_margin_deg=12.0, w_posture=0.0,
                    obstacles=None, w_obs=0.0, obs_margin=0.06,
                    escape_thr=100.0, escape_n=24, max_nfev=150,
                    diff_step=1e-4, seed=0, ref_yaw=None, q_neutral=None, torso_pitch=None,
                    initial_state=None, initial_seed=None, optimize_continuity=False,
                    secondary_tgt=None, yaw_step_max=None, acceleration_weight=0.,
                    hand_task_scales=None, progressive_body_motion=False):
    """全身 IK: 每帧把 [底盘 x, y, yaw + 臂 6 关节] 放进**同一个**有界最小二乘。

    动机 (解耦方案的根因): 之前底盘按启发式跟随、臂关节逐帧解耦, 当手转动范围大时
    "末端位置"与"末端姿态"互相冲突 (底盘与手相对位置固定 -> 臂姿态工作空间覆盖不了)。
    把底盘平移(x,y)与 yaw 也当**决策变量**, 底盘就能主动微调到"让目标位姿可达"的位置,
    位置与姿态便可同时压住 —— 这正是移动操作的标准做法 (whole-body control / task-priority IK)。

    正则 (避免底盘乱飞): w_base * (ox, oy) 把底盘拉回「启发式跟随参考」, w_yaw * yaw 抑制转向;
    底盘偏移被限制在 ±base_range 内 (软正则 + 硬边界)。逐帧热启动 + 连续性惩罚保平滑。

    参数 ref_dx/ref_dy: 启发式跟随的底盘偏移 (相对初始底座) —— 作为 WBC 的参考中心。
    参数 ref_yaw(N,): **逐帧朝向参考**。None = 全 0 (旧行为, 机器人基座系前方)。
        yaw 决策量 x[2] 实际是"相对 ref_yaw[k] 的修正量", 因此正则 w_yaw*x[2] 自动
        变成"把底盘拉向 ref_yaw[k]" —— 绕圈任务据此让机器人始终正面朝向作业点,
        避免手腕被拧 90° 顶死限位 (实测 chairbottle 152/296 帧裕度 <5° 的根因)。
    返回 (base(N,3)=[dx,dy,yaw绝对角], arm(N,6), pos_err(N,)mm, ori_err(N,)deg)。

    initial_seed 只为首帧提供优化初值，首帧使用完整可行域和指定的 n_rand。
    initial_state 表示真实的上一帧状态，首帧也受相邻帧步长约束。
    optimize_continuity 将底盘和关节的实际帧间位移纳入目标，不只用连续性选初值。
    secondary_tgt accepts aligned opposite-hand goals or None per missing row.
    sel/lim include both arms, with the torso (if constrained) after the first arm.
    Returned pose errors are the worse of the active hands per row.
    progressive_body_motion penalizes large deviations from route velocity,
    base acceleration and cumulative torso motion more steeply.
    """
    from scipy.optimize import least_squares
    N = len(pad_p)
    nj = len(lim)                          # 关节 DOF (臂 6; 躯干扩展后 10)
    nX = 3 + nj                            # 决策变量 [dx,dy,yaw] + nj 关节
    if secondary_tgt is not None and len(secondary_tgt) != N:
        raise ValueError("secondary_tgt must align with the primary pose rows")
    if secondary_tgt is not None:
        optimize_continuity = True
    if yaw_step_max is not None and (not np.isfinite(yaw_step_max) or yaw_step_max <= 0.):
        raise ValueError("yaw_step_max must be finite and positive, in degrees per row")
    if not np.isfinite(acceleration_weight) or acceleration_weight < 0.:
        raise ValueError("acceleration_weight must be finite and nonnegative")
    if hand_task_scales is None:
        hand_task_scales = np.ones((N, 2))
    hand_task_scales = np.asarray(hand_task_scales, float)
    if (hand_task_scales.shape != (N, 2) or not np.isfinite(hand_task_scales).all()
            or np.any(hand_task_scales <= 0.)):
        raise ValueError("hand_task_scales must have two positive finite values per pose row")
    other_side = "left" if side == "right" else "right"
    last_motion = None
    torso = slice(9, 12)  # first three torso joints after planar variables and primary arm
    if progressive_body_motion:
        if not optimize_continuity or torso_pitch is None or yaw_step_max is None:
            raise ValueError("progressive body motion needs continuity, yaw rate and torso limits")
        translation_knee = base_step_max/2
        yaw_knee = np.radians(yaw_step_max)/2
        torso_step_knee = np.radians(joint_step_max)/5
        torso_angle_knee = np.asarray(torso_pitch)/2

        def base_residual(delta):
            return np.r_[progressive_motion_residual(delta[:2], translation_knee, radial=True),
                         progressive_motion_residual(delta[2], yaw_knee)]

    def targets_at(k):
        yield side, tgt[k]
        if secondary_tgt is not None and secondary_tgt[k] is not None:
            yield other_side, secondary_tgt[k]

    def base_reference_delta(k):
        if not k:
            return np.zeros(3)
        return np.array([ref_dx[k]-ref_dx[k-1], ref_dy[k]-ref_dy[k-1], _ry[k]-_ry[k-1]])

    if initial_seed is not None:
        if initial_state is not None:
            raise ValueError("Use initial_seed for a new pose or initial_state for a trajectory continuation")
        initial_seed = np.asarray(initial_seed, dtype=float)
        if initial_seed.shape != (nX,) or not np.isfinite(initial_seed).all():
            raise ValueError(f"initial_seed must contain {nX} finite values")
    rng = np.random.default_rng(seed)
    base = np.zeros((N, 3)); arm = np.zeros((N, nj))
    pos_err = np.zeros(N); ori_err = np.zeros(N)
    _ry = np.zeros(N) if ref_yaw is None else np.asarray(ref_yaw, float).reshape(-1)[:N]
    if len(_ry) < N:                       # 长度不足则补零, 不静默错位
        _ry = np.concatenate([_ry, np.zeros(N - len(_ry))])

    yl = -yaw_max if use_yaw else 0.0
    yh = yaw_max if use_yaw else 0.0
    lo = np.array([-base_range, -base_range, yl, *lim[:, 0]])
    hi = np.array([base_range, base_range, yh, *lim[:, 1]])
    # 零空间姿态偏好: 默认限位中点; 调用方可传 q_neutral 覆盖 (躯干关节的
    # "直立"是 0 而非限位中点 —— 中点会让躯干弯着, 姿态误差爆炸, 实测 42°)。
    q_pref = (np.asarray(q_neutral, float).reshape(-1) if q_neutral is not None
              else 0.5 * (lim[:, 0] + lim[:, 1]))

    def eval_x(x, k):
        # yaw 决策量是「相对 ref_yaw[k] 的修正量」—— 正则 w_yaw*x[2] 于是把底盘
        # 拉向 ref_yaw[k] (绕圈任务让机器人始终正面朝向作业点)。
        yaw = (x[2] + _ry[k]) if use_yaw else 0.0
        robot.set_root_pose(se2_pose(ref_dx[k] + x[0], ref_dy[k] + x[1], yaw))
        pe, oe, op = 0., 0., 0.
        for hand, (p0, R0) in targets_at(k):
            p, R = fk_pad(robot, hand, sel, x[3:3+nj])
            pe = max(pe, float(np.linalg.norm(p - p0)) * 1000.)
            cosine = (np.trace(R0.T @ R) - 1.) / 2.
            oe = max(oe, float(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))))
            if obstacles is not None and len(obstacles):
                distances = np.linalg.norm(p[:, None] - obstacles[:, :3].T, axis=0) * 1000.
                op += float((np.maximum(0., obstacles[:, 3]*1000 + obs_margin - distances)**2).sum())
        return pe, oe, op

    def frame_bounds(k, prev):
        """逐帧可行域: 底盘偏移与臂关节都叠加**相邻帧步长硬上界**, 保证不跳。

        为什么必须用硬边界 (软惩罚不够):
          - 底盘: 软惩罚上限 20 时仍出现 307mm 单帧跳 —— 因为数百 mm 的位置误差收益
            远大于惩罚上限, 优化器当然会跳。
          - 臂: 目标本身帧间转向上限仅 14.3°, 但解里出现 j6 +165→-157 (322°)。
            这是**关节角 ±180 表示翻转**: 物理上只差 38°, 但限位 ±165° 不允许穿过
            180°, 于是被迫绕 0 走完 322° —— 对真机是灾难性动作。
          底层手部运动是连续的, 真实所需关节步长远小于 30°/帧, 故硬上界是物理正确的。
          另: 冗余零空间 (底盘2 + 臂6 = 8 自由度 vs 6 维任务) 未被约束时, 解会在
          零空间里游走 (实测 pos_err=0 的帧臂仍动 270°), 硬上界同时抑制这种漂移。
        """
        if prev is None:
            return lo, hi
        blo, bhi = lo.copy(), hi.copy()
        # 各轴用 base_step_max/√2 收紧, 保证**欧氏**帧间位移 ≤ base_step_max
        # (各轴独立给 base_step_max 时对角线可达 1.41×, 实测出现 72mm/帧 > 设定 50mm)
        _bs = base_step_max / np.sqrt(2.0)
        for a in (0, 1):
            # 关键: x[0],x[1] 是「相对启发式跟随参考 ref 的偏移」, 不是底盘绝对位置。
            # 只约束偏移步长时, ref 自身每帧跳 54mm 会直接传到总位置 (实测底盘 80mm/帧
            # 超出 50 设定)。故把 ref 的帧间增量补偿进来, 约束的是**总位置**步长。
            ref = ref_dx if a == 0 else ref_dy
            _dref = ref[k] - ref[k - 1] if k else 0.0
            blo[a] = max(lo[a], prev[a] - _bs - _dref)
            bhi[a] = min(hi[a], prev[a] + _bs - _dref)
        if use_yaw and yaw_step_max is not None:
            center = prev[2] - base_reference_delta(k)[2]
            yaw_step = np.radians(yaw_step_max)
            blo[2] = max(lo[2], center - yaw_step)
            bhi[2] = min(hi[2], center + yaw_step)
        if optimize_continuity:
            # A reference envelope can outrun the platform. Preserve the
            # actual step window while catching up to the reference.
            widths = [_bs, _bs]
            if use_yaw and yaw_step_max is not None:
                widths.append(np.radians(yaw_step_max))
            centers = prev[:3] - base_reference_delta(k)
            for j, half in enumerate(widths):
                if blo[j] >= bhi[j]:
                    stop = np.clip(0., centers[j]-half+1e-6, centers[j]+half-1e-6)
                    blo[j], bhi[j] = stop-1e-6, stop+1e-6
        qlo = np.radians(joint_step_max)
        blo[3:3+nj] = np.maximum(lim[:, 0], prev[3:3+nj] - qlo)
        bhi[3:3+nj] = np.minimum(lim[:, 1], prev[3:3+nj] + qlo)
        # Robustness: a fast reference combined with a lagged previous state can
        # cross the bounds (station-envelope routes move the window each frame).
        # Keep a minimal interval around prev instead of crashing the fit; the
        # frame then solves with degraded tracking rather than aborting the run.
        crossed = blo > bhi
        if crossed.any():
            val = np.clip(prev[crossed], lo[crossed], hi[crossed])
            blo[crossed] = val - 1e-5
            bhi[crossed] = val + 1e-5
        return blo, bhi

    # 关节限位裕度: 贴近限位时给软惩罚, 避免"贴死限位后只能靠翻分支脱困"。
    q_margin = np.radians(limit_margin_deg)

    def limit_barrier(q):
        below = np.maximum(0.0, (lim[:, 0] + q_margin) - q)
        above = np.maximum(0.0, q - (lim[:, 1] - q_margin))
        return np.concatenate([below, above])

    def fit(resid, seed_x, blo, bhi, nfev):
        if optimize_continuity and not use_yaw:
            blo, bhi = blo.copy(), bhi.copy()
            blo[2], bhi[2] = -1e-8, 1e-8
        if torso_pitch is None:
            inset = 0. if optimize_continuity else 1e-4
            return least_squares(resid, np.clip(seed_x, blo + inset, bhi - inset),
                                 bounds=(blo, bhi), method="trf", xtol=1e-8, ftol=1e-8,
                                 max_nfev=nfev, diff_step=diff_step).x

        # WBC vector is [base x/y/yaw, arm 1..6, torso 1..4].
        qlo, qhi = blo[torso], bhi[torso]
        seed_x = seed_x.copy()
        seed_x[torso] = torso_pitch_coordinates(seed_x[torso], qlo, qhi, torso_pitch, encode=True)
        lower, upper = blo.copy(), bhi.copy()
        lower[torso], upper[torso] = 0.0, 1.0

        def decode(y):
            x = y.copy()
            x[torso] = torso_pitch_coordinates(y[torso], qlo, qhi, torso_pitch)
            return x

        y = least_squares(lambda y: resid(decode(y)), np.clip(seed_x, lower, upper),
                          bounds=(lower, upper), method="trf", xtol=1e-8, ftol=1e-8,
                          max_nfev=nfev, diff_step=diff_step).x
        return decode(y)

    def solve_best(k, prev):
        """热启动 -> 邻域扰动 -> 全局随机, 三层起点按序试, 取打分最优。

        起点数刻意压小 (n_local=2, n_rand=1): 实测把起点从 11 个降到 4 个,
        精度不变而解算快 2 倍 (51ms/帧 vs 109ms/帧, chairbottle 296 帧);
        真正难逃的坏分支由 solve_escape 兜底, 不靠堆随机起点。
        """
        p0, R0 = tgt[k]
        blo, bhi = frame_bounds(k, prev)
        best, best_s, best_pair = None, 1e18, (1e9, 1e9)

        def _try(s0, nfev):
            nonlocal best, best_s, best_pair

            def resid(x):
                # 与 eval_x 完全一致: yaw 决策量是「相对 ref_yaw[k] 的修正量」。
                # (漏加 _ry[k] 时优化器在错误的底盘朝向下求解, 而打分又按正确朝向
                # 评估 —— 每帧都解向错位姿, 是绕圈任务误差爆炸的直接根因。)
                yaw = (x[2] + _ry[k]) if use_yaw else 0.0
                robot.set_root_pose(se2_pose(ref_dx[k] + x[0], ref_dy[k] + x[1], yaw))
                p, R = fk_pad(robot, side, sel, x[3:3+nj])
                primary_scale, secondary_scale = hand_task_scales[k]
                parts = [primary_scale*w_pos*(p-p0), primary_scale*w_ori*(R-R0).reshape(-1),
                         w_base * np.array([x[0], x[1]]),
                         w_limit * limit_barrier(x[3:3+nj]),
                         w_posture * (x[3:3+nj] - q_pref)]
                parts.append(np.array([w_yaw * x[2]]) if use_yaw else np.zeros(1))
                if optimize_continuity and prev is not None:
                    motion = x - prev
                    motion[:3] += base_reference_delta(k)
                    parts.extend([w_bcont*motion[:3], w_cont*motion[3:]])
                    if progressive_body_motion:
                        # Keep planned travel available. Penalize corrections to
                        # the route, and measure torso excursions at the links.
                        correction = x[:3]-prev[:3]
                        parts.extend([w_bcont*base_residual(correction),
                                      w_cont*progressive_motion_residual(
                                          torso_link_pitch(motion[torso]), torso_step_knee),
                                      w_posture*progressive_motion_residual(
                                          torso_link_pitch(x[torso]), torso_angle_knee)])
                    if acceleration_weight and last_motion is not None:
                        change = motion - last_motion
                        if progressive_body_motion:
                            parts.extend([acceleration_weight*w_bcont*base_residual(change[:3]),
                                          acceleration_weight*w_cont*progressive_motion_residual(
                                              torso_link_pitch(change[torso]), torso_step_knee/2)])
                        else:
                            parts.append(acceleration_weight*w_bcont*change[:3])
                        parts.append(acceleration_weight*w_cont*change[3:])
                if secondary_tgt is not None and secondary_tgt[k] is not None:
                    p2, R2 = fk_pad(robot, other_side, sel, x[3:3+nj])
                    target_p, target_R = secondary_tgt[k]
                    parts.extend([secondary_scale*w_pos*(p2-target_p), secondary_scale*w_ori*(R2-target_R).reshape(-1)])
                if obstacles is not None and len(obstacles):
                    # 碰撞感知: pad + 肘 (link3) 双点主动避让 (进入优化目标,
                    # 否则规划穿障, 物理臂撞上被顶 —— pad 远离障碍但肘扫过
                    # 的实测 127mm 残差来源)
                    _pts = [p]
                    if secondary_tgt is not None and secondary_tgt[k] is not None:
                        _pts.append(p2)
                    for hand, _ in targets_at(k):
                        elbow = robot.find_link_by_name(f"{hand}_arm_link3")
                        if elbow is not None:
                            _pts.append(np.asarray(elbow.get_pose().p, float))
                    for _pt in _pts:
                        _dd = np.linalg.norm(_pt[:, None] - obstacles[:, :3].T,
                                             axis=0) * 1000.0
                        _pen = np.maximum(0.0, (obstacles[:, 3] * 1000 + obs_margin) - _dd)
                        parts.append(w_obs * _pen)
                return np.concatenate(parts)

            x = fit(resid, s0, blo, bhi, nfev)
            pe, oe, op = eval_x(x, k)
            s = (pe + w_ori * oe + w_obs * op if torso_pitch is None and not optimize_continuity
                 else 1000.0 * np.linalg.norm(resid(x)) / w_pos)
            if prev is not None and not optimize_continuity:
                # 臂连续性: 封顶以允许必要时换冗余分支 (不封顶时一次 180° 翻转
                # 罚 540, 会压过 260mm+ 跟踪误差 -> 贴限位, 实测 307mm 误差)。
                cont_arm = w_cont * float(np.degrees(np.abs(x[3:3+nj] - prev[3:3+nj])).max())
                s += min(cont_arm, cont_arm_cap)
            if s < best_s:
                best_s, best, best_pair = s, x, (pe, oe)
            return pe, oe

        if prev is not None:
            pe, oe = _try(prev, max_nfev)                      # 热启动
            # 早停: 热启动已亚毫米/亚度级命中 -> 扰动与随机起点不可能更好,
            # 直接收工 (本序列 300 帧实测把 158ms/帧 压到 ~50ms/帧)。
            if pe < 1.0 and oe < 1.0:
                return best, best_pair
            for _ in range(n_local):                           # 邻域扰动
                _try(prev + rng.normal(0, 0.08, nX), max_nfev)
            if best_pair[0] < 3.0 and best_pair[1] < 1.5:
                return best, best_pair                         # 已足够好, 跳过全局随机
        # 首帧无热启动: 必须给足随机种子与迭代 (只给 1 个起点时实测落到贴限位
        # 坏分支, 误差 1043mm 并连锁触发第二帧 144° 翻转)。仅此一帧, 代价可忽略。
        if prev is None and k == 0 and initial_seed is not None:
            pe, oe = _try(initial_seed, max_nfev)
            if pe < 1.0 and oe < 1.0:
                return best, best_pair
            n_r, nfev = n_rand, max_nfev
        elif prev is None:
            if torso_pitch is not None:
                _try(np.r_[0.0, 0.0, 0.0, q_pref], max_nfev)
            n_r, nfev = max(8, n_rand), max(400, max_nfev)
        else:
            n_r, nfev = n_rand, max_nfev
        for _ in range(n_r):
            _try(np.concatenate([[0.0, 0.0, 0.0], rng.uniform(lim[:, 0], lim[:, 1])]), nfev)
        return best, best_pair

    # 卡死逃逸: 局部解若跟踪仍差 (贴限位/陷入坏分支), 用全局重启重解该帧,
    # 降低连续性权重 (允许换分支), 取跟踪误差最小的解。仅对少数卡死帧触发, 不影响平滑。
    def solve_escape(k, prev):
        p0, R0 = tgt[k]
        blo, bhi = frame_bounds(k, prev)
        best, best_pe = None, 1e18
        for _ in range(escape_n):
            s0 = np.concatenate([rng.uniform(blo[:2], bhi[:2]),
                                 [rng.uniform(blo[2], bhi[2]) if use_yaw else 0.0],
                                 rng.uniform(lim[:, 0], lim[:, 1])])
            def resid_e(x):
                yaw = (x[2] + _ry[k]) if use_yaw else 0.0   # 与 eval_x 一致 (见 solve_best.resid)
                robot.set_root_pose(se2_pose(ref_dx[k] + x[0], ref_dy[k] + x[1], yaw))
                p, R = fk_pad(robot, side, sel, x[3:3+nj])
                parts = [w_pos * (p - p0), w_ori * (R - R0).reshape(-1)]
                parts.append(np.array([w_yaw * x[2]]) if use_yaw else np.zeros(1))
                return np.concatenate(parts)
            x = least_squares(resid_e, np.clip(s0, blo + 1e-4, bhi - 1e-4), bounds=(blo, bhi),
                              method="trf", xtol=1e-8, ftol=1e-8, max_nfev=max_nfev,
                              diff_step=diff_step).x
            pe, oe, _op = eval_x(x, k)
            if pe < best_pe:
                best_pe, best = pe, x
        return best, best_pe

    x0, (pos_err[0], ori_err[0]) = solve_best(0, initial_state)
    base[0] = [ref_dx[0] + x0[0], ref_dy[0] + x0[1],
               (x0[2] + _ry[0]) if use_yaw else 0.0]
    arm[0] = x0[3:3+nj]
    prev_xw = x0[2] if use_yaw else 0.0      # 上一帧的 yaw **决策量**(不是绝对角)
    n_escape = 0
    for k in range(1, N):
        prev = np.concatenate([base[k - 1, :2] - [ref_dx[k - 1], ref_dy[k - 1]],
                               [prev_xw], arm[k - 1]])
        xk, (pe_k, oe_k) = solve_best(k, prev)
        # The legacy escape objective drops posture/obstacle terms. The constrained
        # solve already tries local and random starts when tracking is poor.
        if torso_pitch is None and not optimize_continuity and pe_k > escape_thr:
            xe, pe_e = solve_escape(k, prev)
            if pe_e < pe_k * 0.5:      # 明显改善才采纳, 避免误伤
                xk, pe_k = xe, pe_e
                _, oe_k, _op2 = eval_x(xe, k)
                n_escape += 1
        pos_err[k], ori_err[k] = pe_k, oe_k
        base[k] = [ref_dx[k] + xk[0], ref_dy[k] + xk[1],
                   (xk[2] + _ry[k]) if use_yaw else 0.0]
        arm[k] = xk[3:3+nj]
        last_motion = np.r_[base[k]-base[k-1], arm[k]-arm[k-1]]
        prev_xw = xk[2] if use_yaw else 0.0
    if n_escape:
        print(f"    [WBC] 卡死逃逸触发 {n_escape} 帧 (位置误差 > {escape_thr:.0f}mm)")
    return base, arm, pos_err, ori_err


def smooth_wbc_track(robot, side, sel, lim, ref_dx, ref_dy, tgt, pad_p,
                     se2_pose, fk_pad, x_init, w_pos=10.0, w_ori=10.0,
                     w_base=0.2, w_yaw=3.0, w_smooth=6.0, base_range=2.0, yaw_max=0.6,
                     use_yaw=True, sweeps=4, diff_step=1e-4, ref_yaw=None):
    """轨迹级时序优化 (离线 MPC): 在 WBC 初值上做「跟踪 + 平滑」联合优化, 消除逐帧抖动。

    逐帧独立解在冗自由度下会在等价解之间跳变 -> 底盘/关节帧间抖动 (实测底盘 max 730mm/帧)。
    本函数用 Gauss-Seidel 迭代 (每次只更新一帧, 相邻帧固定) 解一个"数据保真 + Laplacian 平滑"问题:

        min_x  Σ_k [ w_pos*||p_k - p0_k||² + w_ori*ori(R_k,R0_k)²        (跟踪)
                     + w_base*||(ox,oy)_k||² + w_yaw*yaw_k²              (正则)
                     + w_smooth*||x_k - mean(x_{k±1})||² ]              (时序平滑)

    每帧仍用有界最小二乘求解 (限位/底盘范围硬约束), 热启动=上一轮该帧解。
    这是 MPC 的离线化: 跟踪项 + 平滑项, 只是时域取整段而非滚动窗口。
    参数 ref_yaw: 与 solve_wbc_track 相同的逐帧朝向参考。x_init 的 yaw 列是
    「相对 ref_yaw 的修正量」(调用侧从绝对角减去 ref 得到), 残差里重建绝对角时
    必须加回 —— 否则 fk 在错误朝向下评估 (与 solve_best 漏加 _ry 同一类 bug)。
    返回 (base(N,3), arm(N,6), pos_err(N,)mm, ori_err(N,)deg)。
    """
    from scipy.optimize import least_squares
    N = len(x_init)
    _ry = np.zeros(N) if ref_yaw is None else np.asarray(ref_yaw, float).reshape(-1)[:N]
    if len(_ry) < N:                       # 长度不足则补零, 不静默错位
        _ry = np.concatenate([_ry, np.zeros(N - len(_ry))])
    nj = len(lim)
    nX = 3 + nj
    X = x_init.copy()                       # (N,3+nj) = [ox,oy,yaw,q1..qnj]
    yl = -yaw_max if use_yaw else 0.0
    yh = yaw_max if use_yaw else 0.0
    lo = np.array([-base_range, -base_range, yl, *lim[:, 0]])
    hi = np.array([base_range, base_range, yh, *lim[:, 1]])

    def neigh_mean(k):
        js = [j for j in (k - 1, k + 1) if 0 <= j < N]
        return X[js].mean(axis=0)

    for _ in range(sweeps):
        for k in range(N):
            xm = neigh_mean(k)
            p0, R0 = tgt[k]

            def resid(x):
                yaw = (x[2] + _ry[k]) if use_yaw else 0.0   # 重建绝对角 (见 docstring)
                robot.set_root_pose(se2_pose(ref_dx[k] + x[0], ref_dy[k] + x[1], yaw))
                p, R = fk_pad(robot, side, sel, x[3:3+nj])
                parts = [w_pos * (p - p0), w_ori * (R - R0).reshape(-1),
                         w_base * np.array([x[0], x[1]]),
                         np.array([w_yaw * x[2]]) if use_yaw else np.zeros(1),
                         w_smooth * (x - xm)]
                return np.concatenate(parts)

            X[k] = least_squares(resid, np.clip(X[k], lo + 1e-4, hi - 1e-4), bounds=(lo, hi),
                                 method="trf", xtol=1e-9, ftol=1e-9, max_nfev=120,
                                 diff_step=diff_step).x

    base = np.zeros((N, 3)); nj = len(lim)
    nX = 3 + nj
    arm = np.zeros((N, nj))
    pos_err = np.zeros(N); ori_err = np.zeros(N)
    for k in range(N):
        yaw = (X[k, 2] + _ry[k]) if use_yaw else 0.0   # 绝对角 = 修正量 + ref
        robot.set_root_pose(se2_pose(ref_dx[k] + X[k, 0], ref_dy[k] + X[k, 1], yaw))
        p, R = fk_pad(robot, side, sel, X[k, 3:3+nj])
        p0, R0 = tgt[k]
        c = (np.trace(R0.T @ R) - 1.0) / 2.0
        pos_err[k] = np.linalg.norm(p - p0) * 1000.0
        ori_err[k] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        base[k] = [ref_dx[k] + X[k, 0], ref_dy[k] + X[k, 1], yaw]
        arm[k] = X[k, 3:3+nj]
    return base, arm, pos_err, ori_err


def _slerp(Ra, Rb, t=0.5):
    """两旋转矩阵间的球面插值 (四元数 SLERP)。"""
    from scipy.spatial.transform import Rotation as Rot
    qa = Rot.from_matrix(np.asarray(Ra, float)).as_quat()
    qb = Rot.from_matrix(np.asarray(Rb, float)).as_quat()
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = np.clip(np.dot(qa, qb), -1.0, 1.0)
    if d > 0.9995:
        q = qa
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * qa + np.sin(t * th) * qb) / np.sin(th)
    return Rot.from_quat(q).as_matrix()


def realign_slam_segments(t_c2w, thr_m=0.12, k_med=12.0):
    """修复 SLAM 世界系的"段不连续"(相机位姿粘滞跳变)。

    现象 (chairbottle 实测): HaWoR 的 SLAM 相机平移在个别帧突跳 (F450 跳 386mm、
    F454 跳 144mm) 且**跳变后保持** -> 世界系整体平移。因手在相机系内平滑, 该跳变
    原样传到手部世界轨迹 -> 目标 teleport -> IK 分支翻转 (视觉"手腕乱跳")。

    做法: 检测相机平移帧间跳变 (> max(thr_m, k_med*中位)), 对跳变帧及之后**累加抵消**
    (段对齐), 使相机轨迹连续。阈值依据: chairbottle 步长中位 8.7mm / p99 52.8mm,
    真跳变 386.9/144.2mm (第三高仅 79mm), 故 >120mm 可干净隔离。

    仅修平移 (实测旋转跳变最大仅 3.3°, 无需处理)。
    返回 (t_c2w_fixed, jumps) —— jumps 为跳变帧索引。
    """
    t = np.array(t_c2w, float, copy=True)
    N = len(t)
    if N < 2:
        return t, []
    dt = np.linalg.norm(np.diff(t, axis=0), axis=1)
    med = float(np.median(dt))
    thr = max(float(thr_m), k_med * med)
    jumps = [i + 1 for i in range(N - 1) if dt[i] > thr]
    orig = t.copy()
    acc = np.zeros(3)
    for j in jumps:
        acc = acc + (orig[j - 1] - orig[j])      # 抵消该跳变
        t[j:] += acc                              # 该帧及之后整体回移 (累加)
    return t, jumps


def clean_pose_sequence(pos, rot, ang_thr=60.0, pos_k=6.0, smooth_win=5):
    """修复目标位姿序列的伪影: ①成块的 ~180° 姿态翻转 ②离群位置跳变。

    实测 (chairbottle): 目标姿态 median 帧间仅 1.6°, 但存在 **成块的 ~175° 翻转**:
        orig101->102: 174.6°   (整块 102..105 停在反向)
        orig105->106: 172.5°   (再翻)    orig109->110: 172.7°  (翻回)
    (原始数据 250fps, 真实运动 <5°/帧, 故 170°+ 的帧间跳变必是上游 SVD 重定向的
    反射/符号歧义伪影。) IK 会忠实跟踪 -> 机械臂分支翻转 -> 视觉上"不平滑"。
    **根因在目标序列, 且是成块出现, 单帧替换会失效 (替换用的邻居本身也是反的)**。

    故采用「增量式时序一致性修复」: 逐帧与**上一帧已修正姿态**比较,
      - 测地角 <= ang_thr: 接受原帧;  - > ang_thr: 判为伪影, **保持上一帧**(块内全被吸收),
    恢复正常后自动接回 (真实渐变运动不受影响, 因其帧间角远小于阈值)。
    位置同理 (阈值 = 中位 + pos_k * MAD)。最后做一次零相位平滑去除台阶。
    返回 (pos_c, rot_c, bad_idx)。
    """
    from scipy.spatial.transform import Rotation as Rot
    pos = np.array(pos, float); rot = np.array(rot, float)
    N = len(rot)

    def geo(a, b):
        c = (np.trace(a.T @ b) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    held = set()
    rot_c = rot.copy()
    for i in range(1, N):
        if geo(rot_c[i - 1], rot[i]) > ang_thr:
            rot_c[i] = rot_c[i - 1]
            held.add(i)
    # 位置: 检出离群帧后**线性插值**补齐 (不用"保持上一帧"—— 那会级联卡死:
    #   一旦 keep, 之后真实运动使 |pos_c[i-1]-pos[i]| 持续超阈 -> 一路 hold 到结尾,
    #   实测把尾段 141 帧 (含 0.7~1.3m 真实位移) 全冻结)。检测对比**原始**前帧避免级联。
    pos_c = pos.copy()
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    med = float(np.median(dp)); mad = float(np.median(np.abs(dp - med)))
    thr_p = max(0.08, med + pos_k * mad)
    out = set()
    for i in range(1, N):
        if dp[i - 1] > thr_p:
            out.add(i); out.add(i - 1)
    if out:
        good = np.array([i for i in range(N) if i not in out])
        if len(good) >= 2:
            for j in range(3):
                pos_c[:, j] = np.interp(np.arange(N), good, pos[good, j])
            held |= out

    if smooth_win >= 3 and N > smooth_win:
        padw = smooth_win // 2
        k = np.ones(smooth_win) / smooth_win
        q = Rot.from_matrix(rot_c).as_quat()
        for i in range(1, N):                # 四元数符号对齐, 才能逐分量平滑
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        qs = np.stack([np.convolve(np.pad(q[:, j], (padw, padw), mode="reflect"), k, mode="valid")
                       for j in range(4)], axis=1)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        rot_c = Rot.from_quat(qs).as_matrix()
        pos_c = np.stack([np.convolve(np.pad(pos_c[:, j], (padw, padw), mode="reflect"), k, mode="valid")
                          for j in range(3)], axis=1)
    return pos_c, rot_c, sorted(held)


def clip_step(x, max_step):
    """逐帧步长限幅 (在线/离线限速): |x[k]-x[k-1]| 超过 max_step 时沿方向缩到 max_step。

    用于抑制少数离群帧的突然大跳 (如 WBC 底盘 max 442mm/帧), 是工业界 IK 输出后
    常见的 "关节限速 / 底盘限速" 后处理。会引入轻微滞后。
    """
    y = np.array(x, float)
    for k in range(1, len(y)):
        d = y[k] - y[k - 1]
        n = float(np.linalg.norm(d))
        if n > max_step > 0:
            y[k] = y[k - 1] + d * (max_step / n)
    return y


def post_smooth_traj(base_xy, base_yaw, arm, win=11, passes=2,
                     base_step_max=None, arm_step_max=None, smooth_fn=None):
    """输出后处理平滑: 对 (底盘位姿, 臂关节) 轨迹做零相位对称滑动平均, 可选逐帧限速。

    为什么需要这一层: WBC/IK 逐帧独立解在冗余自由度下存在等价解, 会在分支间跳变
    (底盘出现数百 mm/帧的离群帧), 单靠优化里的正则项压不平。工业界标准做法是在
    IK 输出后**再接一层时序滤波 + 限速**。代价: 末端跟踪精度略降 (平滑改变了轨迹)。
    smooth_fn: 传入 rg.smooth_traj (零相位滑动平均) 以复用主管线实现。
    返回 (base_x(N,), base_y(N,), base_yaw(N,), arm(N,6))。
    """
    if smooth_fn is None:
        raise ValueError("需要传入 smooth_fn (零相位滑动平均)")
    bx = smooth_fn(base_xy[:, 0:1], win, passes).ravel()
    by = smooth_fn(base_xy[:, 1:2], win, passes).ravel()
    byaw = smooth_fn(np.asarray(base_yaw, float).reshape(-1, 1), win, passes).ravel()
    arm_s = smooth_fn(arm, win, passes)
    if base_step_max is not None:
        bx = clip_step(bx, base_step_max)
        by = clip_step(by, base_step_max)
        byaw = clip_step(byaw, base_step_max)
    if arm_step_max is not None:
        arm_s = clip_step(arm_s, arm_step_max)
    return bx, by, byaw, arm_s


# ============================================================
# 3c. MANO 三点 -> 夹爪位姿 (对齐上游 compute_analytical_gripper_pose)
# ============================================================
FINGER_BASE_DIST = 0.026906   # 夹爪两指基距 (gripper_config.py)
FINGER_ORIGIN_X = 0.03689     # 指关节在 gripper_link 系的 x (URDF 原值)
GRIP_MAX = 0.05
# R1 夹爪关节量 q -> 两指**碰撞盒内表面**间距。实测标定 (URDF 几何推导 + 仿真实测):
#   两指 link 原点间距 = 0.0269 + 2q   (finger_joint origin y=∓0.013453, prismatic 各动 q)
#   每个碰撞盒 (0.06×0.01×0.04) 中心相对自身 link 原点朝内偏 0.01, 半厚 0.005
#   => 内距 = (0.0269 + 2q) - 2*0.01 - 2*0.005 = 2q - 0.0031
# 旧的 0.017 + 2.22q 是另一版夹爪的标定值, 用在本 URDF 上会把 25mm 物体判成需要
# q=3.5mm (真实内距仅 3.9mm) -> 两指几乎全闭去撞物体, 抓取必然失败 (agile456 实测)。
GRIP_INNER_A, GRIP_INNER_K = -0.0031, 2.0
GRIP_LINK_GAP0 = 0.0269        # q=0 时两指 link **原点**间距 (finger_joint origin y=∓0.013453)
GRIP_INNER_OFF = GRIP_LINK_GAP0 - GRIP_INNER_A   # |l1-l2| -> 内距 的减项 (= 0.0300)
# 说明: 内距 = |两指 link 原点间距| - GRIP_INNER_OFF = (0.0269+2q) - 0.030 = 2q-0.0031。
# 实测两指间距必须用这个减项 (0.030), 而不是凭估计的 0.031 —— 1mm 的偏差就足以
# 让"抓握目标 24.7mm"变成"实际内距 25.7mm"(= 物体宽度 25.2 之上, 两指悬空没接触),
# 交接重力后物体自由落体 (agile456 实测)。
GRIP_SQUEEZE = 0.0005         # 夹紧预紧 (m): 目标内距比物体薄轴小 0.5mm, 产生接触法向力
GRIP_INNER_MIN = 0.0          # 内距下限 (q=0 时两指贴合)

# 指关节原点/运动轴 (gripper_config.FINGER_GEOM_ARRAYS, 右手)
_FINGER_GEOM = {
    "right": (np.array([FINGER_ORIGIN_X, -0.013453, -0.00012053]),
              np.array([0.0, -1.0, 0.0]),
              np.array([FINGER_ORIGIN_X, 0.013453, 0.00012067]),
              np.array([0.0, 1.0, 0.0])),
    "left": (np.array([FINGER_ORIGIN_X, 0.013453, 0.00012067]),
             np.array([0.0, 1.0, 0.0]),
             np.array([FINGER_ORIGIN_X, -0.013453, -0.00012053]),
             np.array([0.0, -1.0, 0.0])),
}


def grasp_width_for_object(thin_m, squeeze=GRIP_SQUEEZE):
    """由物体薄轴自动推导夹爪闭合量 q (m) —— 物理抓取用, 无需按任务调参。

    夹爪碰撞盒内距 = GRIP_INNER_A + GRIP_INNER_K * q (见常量处 URDF 推导)。
    平行夹爪抓薄轴 thin_m 的物体, 目标内距应**略小于**物体宽度, 让 PD 伺服持续
    输出法向力 (接触预紧):
        q = (thin_m - squeeze - GRIP_INNER_A) / GRIP_INNER_K
    预紧量必须小: 内距压入超过 ~2mm 时 PhysX 会持续把物体当穿透排开。
    受关节上限 GRIP_MAX 与内距下限 GRIP_INNER_MIN 双重约束。
    """
    q = (float(thin_m) - squeeze - GRIP_INNER_A) / GRIP_INNER_K
    return float(np.clip(q, 0.0, GRIP_MAX))


def mano_gripper_pose(wrist, f1, f2, side="right", off_scale=1.0):
    """MANO 三点 (腕 j0 / 拇指尖 j4 / 食指尖 j8) -> (夹爪位置, 旋转, 指关节行程)。

    Gram-Schmidt 构造 (与上游一致): X = 腕->指尖中点 (指向), Y = 拇指->食指在 X
    垂面内的分量 (开合), Z = X×Y; 再按夹爪几何的 Y 符号纠正。位置取「指尖中点」
    而非手腕 —— 两指指尖与 MANO 指尖重合, gripper_link 落在指尖后方 37mm。

    off_scale: 指尖相对手腕的**局部矢量**额外放大倍数 (默认 1 = 不做)。
    世界链路若做过尺度归一化 (÷scale), 手的局部网格会被一起缩小 (不再是米制),
    此时必须传 off_scale=scale 把手指几何还原成米制, 否则指间距被压到几毫米、
    指关节行程恒被 clamp 到 0 (夹爪永远闭合), gripper_link 也会偏到手腕后方。
    """
    o1, a1, o2, a2 = _FINGER_GEOM[side]
    wrist = np.asarray(wrist, float)
    if off_scale != 1.0:
        f1 = wrist + (np.asarray(f1, float) - wrist) * off_scale
        f2 = wrist + (np.asarray(f2, float) - wrist) * off_scale
    midpoint = (np.asarray(f1, float) + np.asarray(f2, float)) / 2

    v_pointing = midpoint - wrist
    X = v_pointing / max(np.linalg.norm(v_pointing), 1e-9)

    v_open = np.asarray(f2, float) - np.asarray(f1, float)
    v_open_proj = v_open - np.dot(v_open, X) * X
    Y = v_open_proj / max(np.linalg.norm(v_open_proj), 1e-9)
    Z = np.cross(X, Y)

    if np.sign((o2 - o1)[1]) * np.dot(Y, v_open) < 0:      # 开合轴与夹爪几何一致
        Y, Z = -Y, -Z
    R = np.column_stack([X, Y, Z])
    if np.linalg.det(R) < 0:
        R = np.column_stack([X, -Y, -Z])

    finger_dist = float(np.linalg.norm(v_open))
    jv = max(0.0, min(GRIP_MAX, (finger_dist - FINGER_BASE_DIST) / 2))
    mid_in = ((o1 + a1 * jv) + (o2 + a2 * jv)) / 2
    return midpoint - R @ mid_in, R, jv


# ============================================================
# 4. 可视化: 地面路径标记 (起点红 + 沿轨迹灰) + 手部目标路径
# ============================================================
def no_collision(actor, scene=None, name=None):
    """把 actor 的所有碰撞形状碰撞组清零 (= 纯视觉, 不参与任何接触), **带回读断言**。

    路径标记/目标球只是为了"看得见", 但 create_box 建的静态盒**默认带碰撞**。
    物理抓取时物体掉落后就会撞在这些地面小方块上 (实测 agile456: 物体与
    tgt_path_* 产生 13 对接触, 落点被垫高到 z=0.019)。凡纯视觉 actor 一律清零。

    为什么必须回读断言: create_box 返回的 envs.utils.Actor 是**包装类**, 既没有
    `.entity` 也没有 `get_components()` —— 旧版 `getattr(actor,'entity',actor)`
    拿到的还是包装类, 循环体一次都不执行, **静默什么都不做** (实测 agile456:
    落地接触列表里仍有 ef3r_object<->tgt_path_147/154/161/168)。兜底路径走
    scene.get_all_actors() 按名字找回 Entity。失败抛异常, 不吞。
    """
    comps = []
    ent = getattr(actor, "entity", None)
    holder = ent if (ent is not None and hasattr(ent, "get_components")) else actor
    if hasattr(holder, "get_components"):
        comps = list(holder.get_components())
    if not comps and scene is not None:
        nm = name or (actor.get_name() if hasattr(actor, "get_name") else None)
        if nm:
            for _a in scene.get_all_actors():
                try:
                    if _a.get_name() == nm:
                        comps = list(_a.get_components())
                        break
                except Exception:
                    continue
    shapes = [sh for comp in comps for sh in getattr(comp, "get_collision_shapes", list)()]
    if not shapes:
        raise RuntimeError(
            f"no_collision: 找不到任何碰撞形状 (actor={type(actor).__name__}, "
            f"name={name}) —— 包装类需传 scene 兜底, 请检查调用点")
    for sh in shapes:
        sh.set_collision_groups([0, 0, 0, 0])
    # 回读断言: 写了必须真的生效
    for i, sh in enumerate(shapes):
        got = list(sh.get_collision_groups())
        if got != [0, 0, 0, 0]:
            raise RuntimeError(f"no_collision: 碰撞组回读失败 shapes[{i}]={got}")
    return len(shapes)


def add_path_markers(scene, create_box, base_xy, start_xy, n_marks=40, collide=False):
    a = create_box(scene, sapien.Pose(p=[start_xy[0], start_xy[1], 0.006]),
                   half_size=[0.06, 0.06, 0.006], color=(1, 0, 0), name="base_start", is_static=True)
    if not collide:
        no_collision(a, scene=scene, name="base_start")
    step = max(1, len(base_xy) // n_marks)
    for kk in range(0, len(base_xy), step):
        a = create_box(scene, sapien.Pose(p=[base_xy[kk, 0], base_xy[kk, 1], 0.004]),
                       half_size=[0.02, 0.02, 0.004], color=(0.1, 0.1, 0.1),
                       name=f"base_path_{kk}", is_static=True)
        if not collide:
            no_collision(a, scene=scene, name=f"base_path_{kk}")


class TargetMarker:
    """手部目标可视化: 每帧把一个醒目小球移动到当前目标位置, 并在地面留下目标路径。

    使"底盘/机械臂是否跟上了手"一眼可见 (否则画面里没有参照物)。
    """

    def __init__(self, scene, pad_p, make_box, color=(1.0, 0.2, 0.2), r=0.03, collide=False):
        mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=r, material=mat)
        self.actor = builder.build_kinematic(name="hand_target")
        # 目标水平投影的地面路径 (淡蓝小点)
        step = max(1, len(pad_p) // 40)
        for kk in range(0, len(pad_p), step):
            a = make_box(scene, sapien.Pose(p=[pad_p[kk, 0], pad_p[kk, 1], 0.003]),
                         half_size=[0.015, 0.015, 0.003], color=(0.2, 0.4, 1.0),
                         name=f"tgt_path_{kk}", is_static=True)
            if not collide:
                no_collision(a, scene=scene, name=f"tgt_path_{kk}")

    def update(self, p):
        self.actor.set_pose(sapien.Pose(p=[p[0], p[1], p[2]]))


# ============================================================
# 5. 相机: 固定机位自动布置 (按路径包围盒)
# ============================================================
def auto_cameras(scene, look_at_pose, base_xy, pad_p, W=480, H=360, fovy=50):
    """按路径包围盒自动布置固定机位 (正面/侧面/俯视), 覆盖整条轨迹。

    返回 (cams, ctr): cams 顺序 [front, side, top]; ctr 为路径中心 (供跟随视角用)。
    """
    xy = np.concatenate([base_xy, pad_p[:, :2]], axis=0)
    lo, hi = xy.min(0), xy.max(0)
    ctr = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]),
                    0.5 * (pad_p[:, 2].min() + pad_p[:, 2].max())])
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    d = 1.15 * span + 0.8 * (pad_p[:, 2].max() - pad_p[:, 2].min()) + 1.5
    fv = np.deg2rad(fovy)
    mk = lambda nm: scene.add_camera(name=nm, width=W, height=H, fovy=fv, near=0.05, far=500)
    front = mk("front"); front.entity.set_pose(look_at_pose(ctr + [0, d, 0.5 * d], ctr))
    side = mk("side"); side.entity.set_pose(look_at_pose(ctr + [-d, 0, 0.5 * d], ctr))
    top = mk("top"); top.entity.set_pose(look_at_pose(ctr + [0, 0, 1.4 * d], ctr))
    return [front, side, top], ctr


def fpv_cam_pose(base_xy, forward, head_z, look_at, fwd_off=0.12, yaw=0.0):
    """机器人第一视角 (机载 FPV): 相机在躯干顶端前方, 视线朝向当前手部目标。

    让手/夹爪始终居于画面中心 —— 否则固定朝前看时, 手偏移就看不到 (画面全白)。
    forward 为 **yaw=0 时的世界系前向** (comfort_frame 推导, 不随底盘旋转);
    yaw 传入当前底盘朝向, 前向偏移随之旋转 —— 否则底盘转弯后相机不再位于身体
    前方, 视线被自己的肩/手臂横穿遮挡 (实测 chairbottle 中段整屏手臂)。
    fwd_off 还必须大于肩部前伸 (~0.14m): 相机越过肩线, 手臂才不会横在视线上。
    """
    b = np.asarray(base_xy, np.float64)
    f = np.asarray(forward, np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    off = np.array([c * f[0] - s * f[1], s * f[0] + c * f[1], 0.0]) * float(fwd_off)
    eye = [b[0] + off[0], b[1] + off[1], float(head_z)]
    return eye, [float(look_at[0]), float(look_at[1]), float(look_at[2])]

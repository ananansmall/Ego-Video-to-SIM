"""N-5: Stage2 接近段轨迹局部优化 (用户方案, 2026-09-12).

用户原话
--------
"stage2 快到达目标点前, 可以进行一个简单的夹爪轨迹优化. 处理的原始参数是预处理后的
 夹爪轨迹参数. 首先就是通过物理引擎来计算夹爪的碰撞和物体的碰撞, 不碰到, 这个是一个
 代价; 然后是跟随预处理后的夹爪轨迹; 还有物理学可行轨迹; 最后到达是 stage1 的最终固定位姿."

问题背景 (实测)
----------------
轴向直线逼近在物体附近余量很薄: 物体在 anchor 系 y ∈ [-32.6, +47.9]mm, 虎口内侧
±55.0mm (q_open=0.05) → **+y 侧仅 7.1mm**; scale 0.6 时余量转负 →
真撞 (max_pen 3.08mm, 手指被顶开 13.57mm → PRE_F50_READY 失败).
且接近段压到 1:1 时基座速度翻倍, PD 跟不上 (跟踪 2.49 → 6.33mm) → 也失败.

本模块的做法
------------
在接近段叠加一条**低维横向偏移曲线**, 让夹爪绕开物体, 末端自然归零回到 Stage1 anchor:

    δ(u) = [ a_y · sin(π u) , a_z · sin(π u) · cos(π u) ]   (u = 接近进度 0→1)

* ``sin(πu)`` 在 u=0 与 u=1 处都为 0 → **两端约束天然满足** (起点不动, 终点仍是 anchor)
* 只有 **2 个待优化参数** (a_y 沿夹爪开合轴, a_z 沿夹爪 X 上下) → 网格搜索即可, 无需 CEM
* 偏移在**夹爪局部系**表达, 再随该控制点的姿态旋转回世界系

三项代价 (严格对应用户描述)
---------------------------
1. ``C_collide``  解析穿透: 两指碰撞盒中心 vs 物体 AABB, 有穿透则罚 (权重最大)
2. ``C_follow``   偏移量本身 (与预处理轨迹的偏差) —— 偏移越小越好
3. ``C_feasible`` 相邻控制点偏移的差分 (~速度) 与二阶差分 (~加速度), 保证物理可行

终端硬约束
----------
末控制点 u=1 → δ=0 → 严格等于 Stage1 的最终固定位姿 (由 sin(π)=0 保证, 非软约束).
"""

from __future__ import annotations

import logging

import os

import numpy as np

logger = logging.getLogger("grasp_hawor")

# 指端几何 (与 physics_utils 一致)
_FINGER_OFFSET_GRIPPER = np.array([0.03689, 0.0, 0.0])   # base → 手指原点 (夹爪 X)
# 2026-09-13: URDF 里 finger joint origin 是 ±0.020 (axis 为 ∓y, 故 q 增大时两指向内收).
# 注意 physics_utils 里 FINGER_BASE_DIST 的注释值 0.026906 与代码实算 0.040 不一致(注释过时),
# 这里以 **URDF 实际值 ±0.020** 为准.
_FINGER_HALF_GAP = 0.020
_FINGER_BOX_HALF = np.array([0.0387, 0.010, 0.006])       # 手指碰撞盒半尺寸 (长/宽/厚, 估算)


def gripper_local_axes(R: np.ndarray):
    """返回夹爪坐标系的三个轴 (世界系): X=手指伸出, Y=开合, Z=第三轴."""
    R = np.asarray(R, dtype=np.float64)
    return R[:, 0], R[:, 1], R[:, 2]


def finger_centers(base_pos, base_R, q):
    """两指碰撞盒中心 (世界系). 与 physics_env 的手指 box 定义一致 (origin ±0.01)."""
    base_pos = np.asarray(base_pos, dtype=np.float64)
    _, y_axis, _ = gripper_local_axes(base_R)
    tip = base_pos + np.asarray(base_R, dtype=np.float64) @ _FINGER_OFFSET_GRIPPER
    half_gap = _FINGER_HALF_GAP + float(q)
    return tip + y_axis * half_gap, tip - y_axis * half_gap


_FINGER_RADIUS = 0.012      # (已弃用) 旧的小球近似半径
_FINGER_N_SPHERE = 3        # (已弃用)

# ★★★ 2026-09-13 必须与 physics_env URDF 的 <collision> 逐值一致!
#   URDF 当前 collision: origin xyz="-0.0045 ±0.01 0", box size="0.09 0.01 0.04"
#   ⇒ finger1 薄板在 link 系 y ∈ [+0.005, +0.015] (中心 +0.010, 半厚 0.005)
#   ⇒ 世界系 q=0 时内侧面 = -0.020+0.015 = -0.005 → 指缝 10mm ✓
# 踩坑记录: 方案A 把物理回退成薄板后, 这里一度仍是"92×70×37 真实指形"模型 →
#   优化器以为手指比实际胖 60mm → 报出物理上不存在的 20.23mm "穿透",
#   于是 ±67mm 范围内怎么搜都"修不动"(始终选 ~0 偏移).
#   **改 physics_env 的手指碰撞体时, 必须同步改这里。**
# ★★★ 2026-09-13 (第三次同步) 薄板已删除, collision = **未缩放真实指形 STL**:
#   实测 STL bbox: x[-0.0505,0.0415] y[-0.0652,+0.0048] z[-0.0142,0.0229] (finger1)
#   ⇒ link 系中心 y = (-0.0652+0.0048)/2 = -0.0302, 半尺寸 y = 0.035
#   世界系 q=0 内侧面 = -0.020 + (-0.0302+0.035) = -0.0152 → **指缝 30.4mm** ✓
#   改 physics_env 手指碰撞体时**必须同步改这里** (已踩坑 3 次).
# ★ 任务3: collision = 未缩放真实指形 STL (实测 bbox x[-0.0505,0.0415] y[-0.0652,0.0048] z[-0.0142,0.0229])
#   link 系中心 y=(-0.0652+0.0048)/2=-0.0302, 半尺寸 (0.046, 0.035, 0.0185)
#   世界系 q=0 内侧面 = -0.020 + (-0.0302+0.035) = -0.0152 → 指缝 30.4mm ✓
_FINGER_LOCAL_CENTER = np.array([-0.0045, -0.0302, 0.0043])
_FINGER_HALF_EXTENT = np.array([0.046, 0.035, 0.0185])
# ★ 网格必须"无缝隙"覆盖手指盒, 否则漏检穿透 (2026-09-13 实测: 3×3×3 网格配 9.25mm 半径
#   时球间有 46mm 空隙 → 优化器报"无穿透"→ 给出无效偏移 → 物理穿透反升到 2.45mm).
# 现在球半径取**网格单元半对角线**, 4×4×4 网格:
#   spacing = 2·half/(N-1) = (0.0307, 0.0233, 0.0123)
#   r = 0.5·√(Σ spacing²) ≈ 0.0202
_FINGER_GRID = 4
_FINGER_SPHERE_R = float(
    0.5 * np.linalg.norm(2.0 * _FINGER_HALF_EXTENT / max(_FINGER_GRID - 1, 1))
)


# 2026-09-13: 接近段安全余量. 只要求"不穿透"不够 —— 实测穿透降到 0.11mm 后,
# 手指仍会与物体发生 33 次轻擦接触, 把物体推走 10.41mm (阈值 10mm).
# 因此 C_collide 改为"低于余量就罚", 让优化器主动保持间隙.
APPROACH_CLEARANCE_MARGIN_M = 0.003

# ★★★ 2026-09-14 分段改造参数 (用户: "不只桌面, 还有被抓取物体") ★★★
#   APPROACH_CONTACT_ALLOW_U: 接近段/末段的分界 (归一化进度 u).
#     u <  该值 → "抓取前": 不许接触目标物体 (罚穿透 + 间隙不足) ✓ 用户原话
#     u >= 该值 → "到位":   允许贴到物体表面 (只罚真穿透) ✓
APPROACH_CONTACT_ALLOW_U = float(os.environ.get("_APPROACH_CONTACT_ALLOW_U", "0.85"))

#   末段(允许接触区)对真穿透的权重 —— 比接近段轻, 因为终点本来就要贴上去
APPROACH_END_PEN_WEIGHT = float(os.environ.get("_APPROACH_END_PEN_WEIGHT", "0.20"))

#   障碍物"贴住但没穿"的极小余量 —— 只为防刮蹭, **不是**"远离" ✗
APPROACH_GRAZE_MARGIN_M = float(os.environ.get("_APPROACH_GRAZE_MARGIN_M", "0.0015"))
APPROACH_GRAZE_WEIGHT = float(os.environ.get("_APPROACH_GRAZE_WEIGHT", "0.10"))

# 2026-09-13: 接近段允许的最大横向偏移 (m). 超过就重罚 —— 对应用户要求
# "A 方案可以, 但要考虑原来的轨迹": 避让不能让基座偏离原轨迹太远, 否则
# 1) 跟踪门 (mean<5mm/max<15mm) 会挂; 2) 抓取位姿会跟着偏, 影响咬合.
APPROACH_MAX_OFFSET_M = 0.030


def sphere_aabb_clearance(point, obj_center, obj_half, radius):
    """球与 AABB 的**带符号**距离: >0 分离, <0 穿透 (m)."""
    point = np.asarray(point, dtype=np.float64)
    obj_center = np.asarray(obj_center, dtype=np.float64)
    obj_half = np.asarray(obj_half, dtype=np.float64)
    d = np.maximum(np.abs(point - obj_center) - obj_half, 0.0)
    return float(np.linalg.norm(d)) - float(radius)


def obb_obb_signed(cA, RA, hA, cB, RB, hB):
    """**两个有向盒(OBB)** 的带符号分离量 (m), 分离轴定理(SAT), 15 根候选轴.

    返回值: >0 分离 (间隙下界); <0 穿透 (绝对值 ≈ 穿透深度).

    ★ 2026-09-13: 物体 bbox 是**物体局部系**的, 物体在场景里带姿态旋转 ——
      早期把物体当成"世界系轴对齐盒"(直接用 bbox/2), 忽略其旋转,
      scale=0.8 时该误差放大到 ~20mm, 导致 N-5 报出"穿透 20.13mm → 19.22mm"
      (能找到偏移却永远修不干净). 现物体也用自身姿态的 OBB。
    """
    t = np.asarray(cA, float) - np.asarray(cB, float)
    RA = np.asarray(RA, float)
    RB = np.asarray(RB, float)
    hA = np.asarray(hA, float)
    hB = np.asarray(hB, float)
    axes = [RA[:, i].copy() for i in range(3)] + [RB[:, i].copy() for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(RA[:, i], RB[:, j])
            n = float(np.linalg.norm(c))
            if n > 1e-8:
                axes.append(c / n)
    seps = []
    for L in axes:
        rA = float(np.sum(np.abs(RA.T @ L) * hA))
        rB = float(np.sum(np.abs(RB.T @ L) * hB))
        seps.append(abs(float(np.dot(t, L))) - rA - rB)
    return float(max(seps))


def obb_aabb_signed(obb_c, obb_R, obb_h, aabb_c, aabb_h):
    """**有向盒(OBB)** 与 **轴对齐盒(AABB)** 的带符号分离量 (m), 分离轴定理(SAT).

    返回值: **>0 = 分离** (值为该方向上的间隙下界); **<0 = 穿透** (绝对值 ≈ 穿透深度).

    ★ 2026-09-13 为什么要换掉"球串"模型:
      手指碰撞体是 **90×10×40mm 的薄板** (physics_env URDF <collision>).
      用球串近似薄板时, 球半径受**最长轴(x, 90mm)**支配 (≥ 15mm), 于是必然在
      **厚度方向(y, 仅 10mm)多外扩 ~11.5mm** → 优化器报出**物理上不存在的穿透**,
      ±67mm 范围内怎么搜都"修不动"(始终选 ~0 偏移) → scale=0.8 时
      N-5 报 `穿透 16.50mm → 16.50mm` 而 `PRE_F50_READY` 跟踪被顶开 25mm。
      薄板这种形状用球串在原理上就表达不了, 必须用 OBB。

    15 根候选轴: AABB 的 3 个面法向 + OBB 的 3 个面法向 + 9 个叉积.
    """
    t = np.asarray(obb_c, dtype=np.float64) - np.asarray(aabb_c, dtype=np.float64)
    R = np.asarray(obb_R, dtype=np.float64)
    hb = np.asarray(obb_h, dtype=np.float64)
    ha = np.asarray(aabb_h, dtype=np.float64)

    axes = [np.eye(3)[i] for i in range(3)] + [R[:, i].copy() for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(np.eye(3)[i], R[:, j])
            n = float(np.linalg.norm(c))
            if n > 1e-8:
                axes.append(c / n)

    seps = []
    for L in axes:
        ra = float(np.sum(np.abs(L) * ha))          # AABB 投影半径
        rb = float(np.sum(np.abs(R.T @ L) * hb))    # OBB 投影半径
        seps.append(abs(float(np.dot(t, L))) - ra - rb)
    # 任一轴分离 ⇒ 整体分离 (取最大间隙); 否则取"最小重叠"作为穿透深度
    return float(max(seps))


def sphere_aabb_penetration(point, obj_center, obj_half, radius):
    """球 (point, radius) 与 AABB 的穿透深度; 0 表示不碰."""
    point = np.asarray(point, dtype=np.float64)
    obj_center = np.asarray(obj_center, dtype=np.float64)
    obj_half = np.asarray(obj_half, dtype=np.float64)
    d = np.maximum(np.abs(point - obj_center) - obj_half, 0.0)
    dist = float(np.linalg.norm(d))
    return max(0.0, float(radius) - dist)


def finger_box_penetration(base_pos, base_R, q, obj_center, obj_half, finger_sign,
                           obj_R=None):
    """一根手指 (沿夹爪 X 的 7.74cm 盒) 与物体 AABB 的最大穿透.

    修复史 (2026-09-13):
      ① 最早把手指盒当**世界系轴对齐盒** → 忽略手指随夹爪旋转 → 报"无穿透"而物理穿透 2.98mm;
      ② 改成"沿手指轴取球串" → 姿态对了, 但**球半径受最长轴支配**, 无法表达 10mm 薄板
         (厚度方向多外扩 ~11.5mm) → scale=0.8 时报幻影穿透"16.50mm 修不动";
      ③ **现在**: OBB(有向盒) vs AABB 分离轴测试 —— 姿态与形状都精确, 无外扩.
    """
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_R = np.asarray(base_R, dtype=np.float64)
    half_gap = _FINGER_HALF_GAP + float(q)
    # finger link 原点 (世界系) —— 与 physics_env URDF 的 finger joint origin 一致:
    # origin y = ±0.020, axis = ∓y → q 增大时向中心收拢, 故 y = ∓(0.020 + q)
    link_origin = base_pos + base_R @ np.array(
        [0.03689, -half_gap * float(finger_sign), 0.0])
    # 注: link_origin 的 y = ∓(0.020+q); 薄板中心再按 finger_sign 取 ±0.010 (见 _FINGER_LOCAL_CENTER)
    # 手指盒的局部中心 (y 偏移随左右指取符号)
    local_c = _FINGER_LOCAL_CENTER.copy()
    local_c[1] *= float(finger_sign)
    box_center = link_origin + base_R @ local_c
    # ★ 2026-09-13 改为 **OBB(有向盒) vs AABB 分离轴测试** —— 精确表达薄板, 无球串外扩.
    #   OBB 姿态 = 夹爪姿态 (finger link 经 prismatic 关节挂载, 无相对旋转).
    _oR = np.eye(3) if obj_R is None else np.asarray(obj_R, dtype=np.float64)
    signed = obb_obb_signed(box_center, base_R, _FINGER_HALF_EXTENT, obj_center, _oR, obj_half)
    # 余量代价: 穿透时按穿透深度罚; 分离但间隙 < 余量时按不足量罚
    if signed < 0.0:
        return -signed
    return max(0.0, APPROACH_CLEARANCE_MARGIN_M - signed)


def finger_box_signed(base_pos, base_R, q, obj_center, obj_half, finger_sign, obj_R=None):
    """一根手指 OBB 与物体 AABB 的**原始 signed 距离** (m).

    与 ``finger_box_penetration`` 的区别 (★ 2026-09-14 分段改造新增):
      · ``finger_box_penetration`` 把结果**钳在 [0, ∞)** —— 穿透深度和"间隙不足"
        被混成同一个数, 调用方无法区分"穿进去了"和"只是离得近" ✗
      · 本函数返回 **signed**: <0 = 真穿透 (深度 = -signed), >0 = 真实间隙 ✓
    这样调用方才能按进度分段: 接近段罚"间隙不足", 末段只罚"真穿透".

    Args: 同 finger_box_penetration.
    Returns:
        float, <0 表示穿透深度, >0 表示两形状之间的真实最小间隙
    """
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_R = np.asarray(base_R, dtype=np.float64)
    half_gap = _FINGER_HALF_GAP + float(q)
    link_origin = base_pos + base_R @ np.array(
        [0.03689, -half_gap * float(finger_sign), 0.0])
    local_c = _FINGER_LOCAL_CENTER.copy()
    local_c[1] *= float(finger_sign)
    box_center = link_origin + base_R @ local_c
    _oR = np.eye(3) if obj_R is None else np.asarray(obj_R, dtype=np.float64)
    return float(obb_obb_signed(box_center, base_R, _FINGER_HALF_EXTENT,
                                obj_center, _oR, obj_half))


def finger_pair_signed(base_pos, base_R, q, obj_center, obj_half, obj_R=None):
    """两指中**最靠近物体**的那一根的 signed 距离 (m).

    取 min: 只要有一根手指贴上去/穿进去, 就应该被检测到
    (取 max 会在"一指贴住、一指远离"时漏报).
    """
    return min(
        finger_box_signed(base_pos, base_R, q, obj_center, obj_half, +1.0, obj_R),
        finger_box_signed(base_pos, base_R, q, obj_center, obj_half, -1.0, obj_R),
    )


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 2026-09-21 【用户原则 (最高优先级)】规划几何 **必须 = 物理几何** ★★★★
#   "**规划全部用 coacd, 不要再用 bbox**, 不然这样怎么行呢?
#    你**物理仿真用什么, 你的规划就用什么** —— 这个是原则!" ✓
#   病根 ✓ (改前): 本文件的碰撞代价 (`finger_pair_signed` → `obb_obb_signed`) 把物体
#     当成 **bbox 半尺寸的有向盒** ✗; 而物理用的是 **CoACD 64 凸包** ✗
#     ⇒ 规划算的"会不会撞"与物理真实碰撞**不是同一个几何** ✗
#       (实测: 规划说"无穿透 ⇒ already clear" ✗, 物理却撞上/夹空 ✗)
#   修法 ✓ (本块): 用**引擎读出的凸包顶点** (与物理同一组 ✓) 做 OBB-vs-凸包 SAT ✓
#     ⇒ `object_hulls` 给了就用凸包 ✓; 没给才回退 bbox (那种情况物理本身就是 box ⇒ 仍然一致 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
class TargetHullModel:
    """目标物体的**真实碰撞几何** (与引擎 collision shapes 同源 ✓).

    用户原则 ✓: "物理仿真用什么, 规划就用什么" ⇒ 本类直接消费
    `_coacd_world_hulls_live()` (读引擎各凸包局部顶点 + 当前 `T_W_O` ✓)
    ⇒ **规划与物理逐顶点一致** ✓ (不再有 bbox 近似 ✗)
    """

    def __init__(self, hulls_world):
        self.pieces = []          # [(Verts(N,3), centroid(3), radius, normals(M,3)|None), ...]
        for _V in (hulls_world or []):
            V = np.asarray(_V, dtype=np.float64).reshape(-1, 3)
            if len(V) < 4:
                continue
            _n = None
            try:
                from scipy.spatial import ConvexHull
                _n = np.asarray(ConvexHull(V).equations[:, :3], dtype=np.float64)
            except Exception:  # noqa: BLE001
                _n = None
            # ★ 2026-09-22 【P0-1】SAT 需要**凸包边方向** ✓ (补齐"边×边"分离轴 ⇒ 否则会
            #   漏判分离轴 ⇒ 把本来分离的形状误判为重叠 ✗ —— 审计文档 §P0-1 ✓)
            E = None
            try:
                from scipy.spatial import ConvexHull
                _ch = ConvexHull(V)
                _simp = np.asarray(_ch.simplices, dtype=np.int64)
                _es = set()
                for _tri in _simp:
                    for _a in range(3):
                        _i, _j = int(_tri[_a]), int(_tri[(_a + 1) % 3])
                        _es.add((min(_i, _j), max(_i, _j)))
                if _es:
                    E = np.asarray([V[_j] - V[_i] for _i, _j in sorted(_es)],
                                   dtype=np.float64)
            except Exception:  # noqa: BLE001
                E = None
            _C = V.mean(axis=0)
            _r = float(np.max(np.linalg.norm(V - _C, axis=1))) if len(V) else 0.0
            # ★ 每片的 **AABB** ✓ (逐片粗筛用 ✓ —— 片很小(~1cm) ⇒ 淘汰率极高 ✓✓)
            _pc = 0.5 * (V.max(axis=0) + V.min(axis=0))
            _ph = 0.5 * (V.max(axis=0) - V.min(axis=0))
            self.pieces.append((V, _C, _r, _n, _pc, _ph, E))
        self._finalize()

    def __len__(self):
        return len(self.pieces)

    def _finalize(self):
        """预计算整体包围球 + **逐轴 AABB** ✓ (broad-phase 粗筛 ⇒ 避免 64 片全算 ✗)"""
        if not self.pieces:
            self.center_all = np.zeros(3)
            self.radius_all = 0.0
            self.aabb_c = np.zeros(3)
            self.aabb_h = np.zeros(3)
            return
        _Cs = np.stack([p[1] for p in self.pieces], axis=0)
        self.center_all = _Cs.mean(axis=0)
        self.radius_all = float(max(
            np.linalg.norm(p[1] - self.center_all) + p[2] for p in self.pieces))
        # ★ 逐轴 AABB (所有凸包顶点的包围盒 ✓) ⇒ 手指盒与它**逐轴分离**时
        #   必然与整个物体分离 ⇒ 直接返回保守距离 ✓ (无需 64 片 SAT ✓)
        _Vs = np.concatenate([p[0] for p in self.pieces], axis=0)
        self.aabb_c = 0.5 * (_Vs.max(axis=0) + _Vs.min(axis=0))
        self.aabb_h = 0.5 * (_Vs.max(axis=0) - _Vs.min(axis=0))

    def signed_obb(self, obb_c, obb_R, obb_h, prefilter_m: float = 0.12) -> float:
        """OBB 与"凸包并集"的 signed 距离: >0 = 分离(间隙), <0 = 穿透(深度) ✓

        轴集 ✓: OBB 3 面法向 + 每个凸包面的法向 (SAT ✓)
        预筛 ✓: 只算**质心在 `prefilter_m` 内**的凸包片 (64 片全算太慢 ✓)
        注 ✓: 省略"边×边"轴 ⇒ 对薄板-薄板极端姿态会略高估分离;
             对"指腹 vs 物体"足够 ✓ (且与物理同一组顶点 ✓)
        """
        obb_c = np.asarray(obb_c, dtype=np.float64)
        obb_R = np.asarray(obb_R, dtype=np.float64)
        obb_h = np.asarray(obb_h, dtype=np.float64)
        # ★★★★ 2026-09-22 【P0-1 重写】正确的**区间投影 SAT** (审计文档 §P0-1 ✓) ★★★★
        #   旧实现 ✗: `r_h = max((V - C) @ L)` + `sep = |d| - (r_o + r_h)`
        #     ⇒ 假设凸包关于**顶点均值 C 对称** ✗ (CoACD 凸包**不对称** ✓)
        #     ⇒ 实测反例 (文档 §7-1): 小 OBB 中心 (0.08,0.005,0.005) 半尺寸 1mm
        #       **完全落在**四面体 (0,0,0)(0.1,0,0)(0,0.1,0)(0,0,0.1) **内部**,
        #       旧函数却报 **"+29mm 分离"** ✗✗ (包含被误判为分离 ✗)
        #   本实现 ✓: 每个轴**分别算两个区间** `[min(V@L), max(V@L)]` 与
        #     `[min(corner@L), max(corner@L)]` ⇒ 再做区间分离/重叠判定 ✓
        #     ① 任一轴区间**分离** ⇒ 分离 ✓ (间隔 = 分离轴上的**下界** ✓ 不是欧氏距离 ✗)
        #     ② 全部轴区间重叠 ⇒ **相交** ✓ (含"包含"✓) ⇒ 深度 = 各轴重叠的 **min**
        #        (最小平移距离 ✓)
        #   轴集 ✓ (三类齐全, 否则会**漏判分离轴** ⇒ 把分离误判为重叠 ✗):
        #     · OBB 3 个面法向 ✓  · 凸包面法向 `Nr` ✓  · **OBB 边 × 凸包边**叉乘 ✓ (补齐 ✓)
        #   注 ✓: 早期"省略边叉乘轴"的注释**方向写反了** ✗ (文档 §P0-1 指出 ✓) —— 遗漏
        #     分离轴是把**分离**误判为**重叠** ✗ (不是"高估分离"✗)
        _obb_ext = np.abs(obb_R) @ obb_h
        _d_ax = np.abs(obb_c - self.aabb_c) - (_obb_ext + self.aabb_h)
        if float(np.max(_d_ax)) > 1e-3:
            return float(np.max(_d_ax))              # 整体 AABB 分离 ⇒ 轴向间隔下界 ✓
        _signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                           for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        _corners = obb_c + (_signs * np.asarray(obb_h)) @ np.asarray(obb_R).T
        _obb_edges = np.asarray(obb_R, dtype=np.float64).T * np.asarray(obb_h)[:, None]
        _axes0 = [np.asarray(obb_R, dtype=np.float64)[:, 0],
                  np.asarray(obb_R, dtype=np.float64)[:, 1],
                  np.asarray(obb_R, dtype=np.float64)[:, 2]]
        best = float("inf")
        for V, C, _r_p, Nr, _pc, _ph, E in self.pieces:
            if float(np.max(np.abs(obb_c - _pc) - (_obb_ext + _ph))) > 1e-3:
                continue
            axes = list(_axes0)
            if Nr is not None:
                axes.extend([np.asarray(_n, dtype=np.float64) for _n in Nr])
            # ★ 补齐"边×边"轴 ✓ (缺它 ⇒ 漏判分离轴 ⇒ 分离被误判为重叠 ✗)
            if E is not None and len(E):
                _seen = set()
                for _oe in _obb_edges:
                    for _ne in E:
                        _cr = np.cross(_oe, _ne)
                        if float(np.linalg.norm(_cr)) < 1e-12:
                            continue
                        _k = tuple(np.round(_cr / np.linalg.norm(_cr), 6))
                        if _k in _seen:
                            continue
                        _seen.add(_k)
                        axes.append(_cr)
            _sep_any = float("-inf")
            _ov_min = float("inf")
            _intersect = True
            for L in axes:
                _nL = float(np.linalg.norm(L))
                if _nL < 1e-12:
                    continue
                L = L / _nL
                _po = _corners @ L
                _ph2 = V @ L
                _gap = max(float(_ph2.min() - _po.max()), float(_po.min() - _ph2.max()))
                if _gap > 0.0:
                    _sep_any = max(_sep_any, _gap)   # 找到分离轴 ⇒ 该片一定分离 ✓
                    _intersect = False
                    break
                _ov_min = min(_ov_min, -_gap)
            if not _intersect:
                best = min(best, _sep_any)           # 分离 ⇒ 间隔(下界) ✓
            elif _ov_min < float("inf"):
                best = min(best, -_ov_min)           # 相交 ⇒ −(最小平移距离) ✓
        return float(best)

    def query_obb(self, obb_c, obb_R, obb_h) -> dict:
        """★ 2026-09-22 【P0-1 配套】结构化查询 ✓ —— **区分**三种语义 ✓ (审计要求 ✓)

        审计文档 §P0-1 明确要求 ✓: "区分 SAT 判定、轴向间隔下界、欧氏距离、凸包并集穿透量,
        不要统一标成'真实 signed 距离'" ✗
        返回 ✓:
          `intersect`    : 是否相交 (**含完全包含** ✓)
          `depth_m`      : 相交时 = **最小平移距离** ✓ (m, ≥0)
          `sep_lower_m`  : 分离时 = **轴向间隔下界** ✓ (m, ≥0) —— ⚠ **不是欧氏距离** ✗
          `n_pieces`     : 参与判定的凸包片数 ✓
        """
        _sig = self.signed_obb(obb_c, obb_R, obb_h)
        if _sig < 0.0:
            return {"intersect": True, "depth_m": float(-_sig),
                    "sep_lower_m": 0.0, "n_pieces": len(self.pieces)}
        return {"intersect": False, "depth_m": 0.0,
                "sep_lower_m": float(_sig), "n_pieces": len(self.pieces)}


def finger_pair_signed_hulls(base_pos, base_R, q, model: "TargetHullModel"):
    """两指 vs **CoACD 凸包真实几何** 的最小 signed 距离 (m) ✓

    与 `finger_pair_signed` 同语义 (min over 两指 ✓), 但几何换成**物理同款凸包** ✓
    返回: 距离(>0) 或 −穿透深度(<0); 模型为空 ⇒ +inf (调用方应回退 ✓)
    """
    if model is None or len(model) == 0:
        return float("inf")
    base_pos = np.asarray(base_pos, dtype=np.float64)
    base_R = np.asarray(base_R, dtype=np.float64)
    half_gap = _FINGER_HALF_GAP + float(q)
    _out = []
    for _sign in (+1.0, -1.0):
        link_origin = base_pos + base_R @ np.array([0.03689, -half_gap * _sign, 0.0])
        local_c = _FINGER_LOCAL_CENTER.copy()
        local_c[1] *= _sign
        box_center = link_origin + base_R @ local_c
        _out.append(model.signed_obb(box_center, base_R, _FINGER_HALF_EXTENT))
    return float(min(_out))


def penetration_at(base_pos, base_R, q, obj_center, obj_half, obj_R=None):
    """该控制点两指的最大穿透深度 (m). 使用手指自身轴的球串近似 (见 finger_box_penetration)."""
    return max(
        finger_box_penetration(base_pos, base_R, q, obj_center, obj_half, +1.0, obj_R),
        finger_box_penetration(base_pos, base_R, q, obj_center, obj_half, -1.0, obj_R),
    )


def offset_curve(u, a_y, a_z, a_x=0.0):
    """偏移曲线 (返回 (δ_x, δ_y, δ_z), 单位 m). u ∈ [0,1].

    ★ 2026-09-13 (用户: "要平移也是 x,y,z 三个平移吧, 这个也太少了"):
      新增 **δ_x (进给方向)** —— 原先只有 y/z 横向平移, 无法提前后退/延长接近距离,
      遇到"必须绕开而非平移"的几何就修不动 (实测 0.7/0.8 时 19.54→18.71mm 卡住)。
      a_x<0 = 该处整体后退, a_x>0 = 更靠近物体。

    ★ 2026-09-13 改为 sin²(πu):
      旧版用 sin(πu), 在 u=0/1 处导数 δ'(0)=a·π ≠ 0 —— 偏移一启动就带速度突跳,
      正是用户反馈"快到时停顿太直接/轨迹不够平滑"的来源。sin²(πu) 在两端导数都为 0
      (δ' = a·π·sin(2πu)), 起止速度连续, 峰值仍在 u=0.5, 幅值语义不变。
    """
    s = np.sin(np.pi * u) ** 2
    return a_x * s, a_y * s, a_z * s * np.cos(np.pi * u)


def evaluate_candidate(a_y, a_z, approach, obj_center, obj_half, obj_R=None,
                       w_collide=1000.0, w_follow=1.0, w_feasible=10.0,
                       a_x=0.0, obstacles=None, w_obstacle=800.0,
                       target_hulls=None):
    """对一个 (a_y, a_z) 候选打分. 返回 (总代价, 明细 dict)."""
    n = len(approach)
    us = np.asarray([c["u"] for c in approach], dtype=np.float64)
    deltas = np.array([offset_curve(u, a_y, a_z, a_x) for u in us])  # (n, 3)

    # ★★★★ 2026-09-14 分段改造 (用户: "不只桌面, 还有被抓取物体") ★★★★
    #   旧实现的语义错误 (用户原话: "collide 代价 → 0 肯定不对, 但单纯远离障碍物也不行,
    #                          单纯是在**抓取前**最好不要碰到被抓取物体, collision 目前太差"):
    #     ① 目标物体: 用 `penetration_at` 对**所有**控制点一视同仁地罚"穿透或间隙不足" ✗
    #        → u→1 (到达 F50, 本来就该贴到物体表面) 也被罚 → 优化器**不敢靠近** → 轨迹推歪 ✗
    #     ② 障碍物: `if _d < 余量: 罚 (余量-_d)²` 等价于"**离桌面越远越好**" ✗
    #        → 手指被推离桌面, 但用户要的是"**不撞**就行", 不是"远离" ✗
    #   新语义 (三段, 直接对应"抓取前最好不要碰到"):
    #     · 目标物体, 接近段 (u < APPROACH_CONTACT_ALLOW_U):
    #           罚"穿透 + 间隙不足" —— 抓取前不许碰 ✓ (重)
    #     · 目标物体, 末段 (u ≥ 该阈值):
    #           只罚"**真穿透**" —— 允许贴上去 (这是终点, 本来就该贴) ✓ (轻)
    #     · 障碍物 (桌面/其他物体), **全程**:
    #           只罚"**真穿透**" —— 不撞就行, 不再等价于"远离" ✓
    pen_max = 0.0
    pen_sq = 0.0
    pen_real_max = 0.0          # 真穿透的最大值 (signed<0), 用于日志与阈值判定
    gap_min = float("inf")      # 全段最小真实间隙 (用于日志)
    n_contact_pre = 0           # 接近段"碰到"的控制点数 (诊断: 抓取前不该碰)
    u_split = float(APPROACH_CONTACT_ALLOW_U)
    for i, c in enumerate(approach):
        R = np.asarray(c["R"], dtype=np.float64)
        pos = np.asarray(c["p"], dtype=np.float64)
        # 偏移在夹爪局部系表达, 再转世界系
        local = np.array([deltas[i, 0], deltas[i, 1], deltas[i, 2]], dtype=np.float64)
        pos_shifted = pos + R @ local
        # ★ 2026-09-21 【用户原则】有**物理同款凸包**就用凸包 ✓ (规划=物理 ✓);
        #   没有 ⇒ 回退 bbox (那种情况下物理本身就是 box ⇒ 仍然一致 ✓)
        if target_hulls is not None and len(target_hulls) > 0:
            _sd = finger_pair_signed_hulls(pos_shifted, R, c["q"], target_hulls)
        else:
            _sd = finger_pair_signed(pos_shifted, R, c["q"], obj_center, obj_half, obj_R)
        gap_min = min(gap_min, _sd)
        _u = float(c.get("u", 1.0))
        if _u < u_split:
            # 接近段: 真穿透按深度², 且"间隙不足"按不足量² —— 抓取前不许碰 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【复审 · 代价方向修正】零点跳变 ✗ ★★★★
            #   复审原话 ✓: "接近阶段的间隙不足罚分, **只在距离大于零时计算**; 一旦接触或穿透,
            #     这项罚分**突然消失** … 当前函数可能认为'**轻微穿进去**'比'**还差一点碰到**'
            #     好得多" ✗✗
            #   实测 ✓ (复审给的隔离测试): 间隙 1mm ⇒ 代价 32000; 恰好接触 ⇒ **0**;
            #     穿透 0.001mm ⇒ **0.008** ✗ ⇒ **梯度方向是反的** ✗ (加大碰撞权重只会放大它 ✗)
            #   本改 ✓ (复审建议的**连续违反量** ✓):
            #     `C_clear = [ max(0, m − d) / s_d ]²`  (本处 s_d = 1 ✓, m = MARGIN ✓)
            #     ⇒ `_sd ≤ 0` (接触/穿透) **不再置零** ✓ ⇒ d 越小罚越大 ✓ 单调连续 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _pen = max(0.0, -_sd)
            _deficit = max(0.0, APPROACH_CLEARANCE_MARGIN_M - _sd)      # ← 含 _sd ≤ 0 ✓ 连续 ✓
            _cost = _pen * _pen + _deficit * _deficit
            pen_sq += _cost
            if _pen > 0.0 or _deficit > 0.0:
                n_contact_pre += 1
            pen_max = max(pen_max, _pen + _deficit)
        else:
            # 末段: 只罚真穿透 (允许贴到物体表面) ✓ —— 权重降到 APPROACH_END_PEN_WEIGHT
            _pen = max(0.0, -_sd)
            pen_sq += APPROACH_END_PEN_WEIGHT * (_pen * _pen)
            pen_max = max(pen_max, _pen)
        pen_real_max = max(pen_real_max, max(0.0, -_sd))
    # ★★★★ 2026-09-21 【问题5 修复】碰撞代价必须按**毫米容差归一化** ✗ ★★★★
    #   静态审查结论 ✓: "`evaluate_candidate()` 对穿透用**米制平方**, 对跟随偏移却做归一化.
    #     比如单个采样点穿透 1mm, 1000 × 0.001² 只有 **0.001** 量级; 归一化跟随代价可能
    #     明显更大. **权重写成 1000, 并不意味着碰撞一定优先避免** ✗"
    #   修法 ✓: 统一按"允许穿透容差"归一 ⇒ 容差处 = 1.0, 2× 容差 = 4.0 ✓
    #     并**返回 `feasible`** ✓ ⇒ 调用方可"先硬筛可行, 再比跟随" ✓ (审查建议 ✓)
    _pen_tol_m = max(1e-6, float(os.environ.get("_N5_PEN_TOL_MM", "0.5") or 0.5) / 1000.0)
    c_collide = w_collide * ((pen_sq / (_pen_tol_m ** 2)) + (pen_max / _pen_tol_m) ** 2)

    # 1b) ★ 障碍物代价 (桌面/地面 + 所有 non-target 物体, AABB)
    #   ★ 2026-09-14 语义修正: **只罚真穿透** (signed < 0) —— 不再是"距离<余量就罚" ✗.
    #     理由 (用户): 要的是"不撞", 不是"远离"; 后者会把手指推离桌面 → 夹爪姿态被带偏 ✗.
    #     另: 接近段对"贴住桌面"(0 < d < 极小余量) 仍给一个**很小**的惩罚,
    #         避免手指贴着桌面拖行刮蹭; 末段完全放开.
    c_obstacle = 0.0
    obst_pen_max = 0.0
    if obstacles:
        for i, c in enumerate(approach):
            R = np.asarray(c["R"], dtype=np.float64)
            pos = np.asarray(c["p"], dtype=np.float64)
            local = np.array([deltas[i, 0], deltas[i, 1], deltas[i, 2]], dtype=np.float64)
            pos_shifted = pos + R @ local
            half_gap = _FINGER_HALF_GAP + float(c["q"])
            _u = float(c.get("u", 1.0))
            for _sign in (+1.0, -1.0):
                _link = pos_shifted + R @ np.array([0.03689, -half_gap * _sign, 0.0])
                _lc = _FINGER_LOCAL_CENTER.copy(); _lc[1] *= _sign
                _bc = _link + R @ _lc
                for (_oc, _oh) in obstacles:
                    _d = obb_aabb_signed(_bc, R, _FINGER_HALF_EXTENT, _oc, _oh)
                    if _d < 0.0:
                        # 真穿透: 全程重罚 ✓ (无论如何都不能穿进桌面/其他物体)
                        # ★ 2026-09-21 【问题5 同上】障碍物穿透也按**毫米容差归一化** ✓
                        c_obstacle += (_d / _pen_tol_m) ** 2
                        obst_pen_max = max(obst_pen_max, -_d)
                    elif _u < u_split and _d < APPROACH_GRAZE_MARGIN_M:
                        # 保留项: 接近段"贴住但没穿" → 很小的惩罚 (防刮蹭), 末段不罚
                        c_obstacle += APPROACH_GRAZE_WEIGHT * (APPROACH_GRAZE_MARGIN_M - _d) ** 2
        c_obstacle *= float(w_obstacle)

    # 2) 跟随代价: 偏移幅值 (归一化到上限, 便于和 C_collide 配权重).
    #    ★ 平滑性由 offset_curve 的 sin²(πu) 基函数**结构性保证** (两端导数=0),
    #      优化器只选择幅值 (a_y,a_z) —— 所以这里不需要再堆差分项:
    #      固定形状下 一/二/三阶差分都正比于 |a|², 与幅值项冗余, 只会引入权重失衡.
    dev = float(np.max(np.linalg.norm(deltas, axis=1)))
    c_follow = w_follow * (dev / APPROACH_MAX_OFFSET_M) ** 2

    # 3) 越界代价: 超过上限则重罚 (对应"不能为了避让把原轨迹带偏太多")
    over = max(0.0, dev - APPROACH_MAX_OFFSET_M) / APPROACH_MAX_OFFSET_M
    c_feasible = w_feasible * (over ** 2)

    total = c_collide + c_obstacle + c_follow + c_feasible
    # ★★★★ 2026-09-21 【问题5】可用性标志 ✓ (审查建议: "先排除穿桌/超过允许接触深度的候选,
    #   再在可行候选中最小化跟随偏差" ✓) —— 判据: 目标真穿透 & 障碍物真穿透 均 ≤ 容差 ✓
    _feasible = bool(pen_real_max <= _pen_tol_m and obst_pen_max <= _pen_tol_m)
    return total, {
        "C_collide": c_collide, "C_obstacle": c_obstacle,
        "C_follow": c_follow, "C_feasible": c_feasible,
        "pen_max_mm": pen_max * 1000.0, "a_y_mm": a_y * 1000.0, "a_z_mm": a_z * 1000.0,
        "dev_mm": dev * 1000.0,
        # ★ 2026-09-14 分段诊断 (区分"抓取前碰"与"末段贴", 便于 verify 与调参)
        "pen_real_mm": pen_real_max * 1000.0,       # 真穿透最大值 (>0 = 穿进目标物体)
        "gap_min_mm": (gap_min * 1000.0) if gap_min != float("inf") else -1.0,
        "n_contact_pre": n_contact_pre,             # 接近段触碰点数 (应 = 0 ✓)
        "obst_pen_mm": obst_pen_max * 1000.0,       # 障碍物真穿透最大值 (应 = 0 ✓)
        "u_split": u_split,
        # ★ 2026-09-21 新增 (问题5 ✓): 是否**物理可行** (穿透都在容差内 ✓)
        "feasible": _feasible,
        "pen_tol_mm": _pen_tol_m * 1000.0,
    }


def optimize_approach(approach, obj_center, obj_half, obj_R=None,
                      a_y_range=(-0.030, 0.030), a_z_range=(-0.020, 0.020),
                      n_y=13, n_z=9, refine_rounds=2,
                      a_x_values=None, obstacles=None, target_hulls=None):
    """网格搜索 (粗 → 细) 求最优 (a_y, a_z). 返回 (best_a, report).

    Args:
        approach: [{'p': (3,), 'R': (3,3), 'q': float, 'u': float}, ...] 接近段控制点
        obj_center, obj_half: 目标物体的世界系中心与半尺寸

    Returns:
        ((a_y, a_z), report_dict); 若无需偏移 (原本就无碰撞) 则返回 ((0,0), ...)
    """
    if not approach:
        return (0.0, 0.0, 0.0), {"reason": "empty approach", "pen_before_mm": 0.0}

    # 先看原始轨迹有没有穿透
    # ★ 2026-09-21 【用户原则】有**物理同款凸包** ⇒ 用凸包真值算 ✓ (规划=物理 ✓)
    if target_hulls is not None and len(target_hulls) > 0:
        pen_before = max(0.0, -min(
            finger_pair_signed_hulls(
                np.asarray(c["p"], dtype=np.float64),
                np.asarray(c["R"], dtype=np.float64), c["q"], target_hulls)
            for c in approach))
    else:
        pen_before = max(
            penetration_at(c["p"], np.asarray(c["R"]), c["q"], obj_center, obj_half, obj_R)
            for c in approach
        )
    # ★★★★ 2026-09-21 【P0 修复】早退必须做**完整可行性检查** ✗ ★★★★
    #   静态审查结论 ✓: "只要**没有穿透目标物**, 就返回 `already clear`, **不检查桌面和
    #     其他障碍物** ✗" ⇒ 复现样例: "完整候选检查: feasible=False, **桌面穿透 14.2mm**;
    #     优化器返回: 零偏移, reason='already clear'" ✗✗
    #   修法 ✓: 用 `evaluate_candidate(0,0,...)` **完整评估零偏移候选** (含障碍物/桌面 ✓)
    #     · 目标无穿透 **且** 零偏移可行 ⇒ 才早退 ✓
    #     · 目标无穿透 **但** 桌面/障碍物不可行 ⇒ **不早退** ✓ (继续搜偏移 ✓)
    _c_zero, _d_zero = evaluate_candidate(
        0.0, 0.0, approach, obj_center, obj_half, obj_R, a_x=0.0, obstacles=obstacles,
        target_hulls=target_hulls)
    _zero_feasible = bool(_d_zero.get("feasible", True))
    if pen_before <= 0.0 and _zero_feasible:
        logger.info(
            "[N-5] 接近段原本无穿透 (max_pen=0mm) 且**零偏移可行** (obst_pen=%.2fmm) "
            "⇒ 不需要偏移 → a=(0,0)",
            float(_d_zero.get("obst_pen_mm", 0.0)),
        )
        # ★ 2026-09-14 修复: 早退也必须返回 **3 元组** (a_y, a_z, a_x)
        #   调用方 (stage2_reconstruct_mixin.py:460) 解包为
        #     `(a_y, a_z, a_x), approach_offset_report = optimize_approach(...)`
        #   此前这里只返回 (0.0, 0.0) ✗ → "not enough values to unpack (expected 3, got 2)"
        #   → 整个 N-5 被 except 吞掉并回退原轨迹 (实测 hz1: 反而因此 Stage2 PASS ✓)
        return (0.0, 0.0, 0.0), {"reason": "already clear", "pen_before_mm": 0.0}

    best = None
    best_key = None
    y_lo, y_hi = a_y_range
    z_lo, z_hi = a_z_range
    for rnd in range(int(refine_rounds) + 1):
        ys = np.linspace(y_lo, y_hi, int(n_y))
        zs = np.linspace(z_lo, z_hi, int(n_z))
        for a_y in ys:
            for a_z in zs:
                _axs = [float(v) for v in a_x_values] if a_x_values else [0.0]
                for a_x in _axs:
                    cost, detail = evaluate_candidate(
                        float(a_y), float(a_z), approach, obj_center, obj_half, obj_R,
                        a_x=float(a_x), obstacles=obstacles, target_hulls=target_hulls)
                    # ★★★★ 2026-09-21 【问题5 修复】**可行优先**的两级选择 ✓ ★★★★
                    #   审查建议 ✓: "先**排除**穿桌和超过允许接触深度的候选, 再在**可行候选**中
                    #     最小化跟随偏差" ✓ (避免"碰撞代价被跟随代价交易掉" ✗)
                    _key = (0 if detail.get("feasible", True) else 1, cost)
                    if best_key is None or _key < best_key:
                        best_key = _key
                        best = (cost, (float(a_y), float(a_z)), detail, float(a_x))
        # 细化: 在下一次迭代把搜索区间收缩到最优附近
        span_y = (y_hi - y_lo) / max(n_y - 1, 1) * 2.0
        span_z = (z_hi - z_lo) / max(n_z - 1, 1) * 2.0
        y_lo, y_hi = best[1][0] - span_y, best[1][0] + span_y
        z_lo, z_hi = best[1][1] - span_z, best[1][1] + span_z

    cost, (a_y, a_z), detail, a_x = best
    # ★★★★ 2026-09-21 【P0 修复】"可行优先"≠"不可行也能交付" ✗ ★★★★
    #   静态审查结论 ✓: "**全部候选不可行时仍选一个最小代价结果**, 调用方仍可能应用偏移" ✗
    #   修法 ✓: 可行集为空 ⇒ **明确失败** ✓: 返回零偏移 + `feasible=False` + `failed` 标记 ✓
    #     ⇒ 调用方**必须检查** `feasible` ✓ (禁止继续执行不合格轨迹 ✓)
    if not bool(detail.get("feasible", True)):
        logger.error(
            "[N-5] ★ **全部候选不可行** (最优候选 pen_real=%.2fmm / obst_pen=%.2fmm > 容差 "
            "%.2fmm ✗) ⇒ **返回失败** (零偏移) ✗ —— 调用方必须检查 `feasible` 并**拒绝交付** ✗",
            float(detail.get("pen_real_mm", -1.0)), float(detail.get("obst_pen_mm", -1.0)),
            float(detail.get("pen_tol_mm", -1.0)))
        return (0.0, 0.0, 0.0), {
            "reason": "no_feasible_candidate",
            "feasible": False,
            "pen_before_mm": pen_before * 1000.0,
            "pen_after_mm": detail.get("pen_max_mm"),
            "obst_pen_mm": detail.get("obst_pen_mm"),
            "pen_real_mm": detail.get("pen_real_mm"),
            "pen_tol_mm": detail.get("pen_tol_mm"),
            "n_controls": len(approach),
        }
    report = {
        "pen_before_mm": pen_before * 1000.0,
        "pen_after_mm": detail["pen_max_mm"],
        "a_y_mm": detail["a_y_mm"], "a_z_mm": detail["a_z_mm"],
        "C_collide": detail["C_collide"], "C_follow": detail["C_follow"],
        "C_feasible": detail["C_feasible"], "total": cost,
        "n_controls": len(approach),
        # ★ 2026-09-14 分段语义诊断 (区分两种完全不同的"碰")
        "pen_real_mm": detail.get("pen_real_mm"),      # 真穿透 (应 → 0)
        "gap_min_mm": detail.get("gap_min_mm"),        # 全段最小间隙
        "n_contact_pre": detail.get("n_contact_pre"),  # 抓取前触碰点数 (应 = 0 ✓)
        "obst_pen_mm": detail.get("obst_pen_mm"),      # 桌面/其他物体真穿透 (应 = 0 ✓)
        "u_split": detail.get("u_split"),
        # ★ 2026-09-21 新增 (问题5 ✓): 选中候选是否**物理可行** + 所用容差 ✓
        "feasible": detail.get("feasible"),
        "pen_tol_mm": detail.get("pen_tol_mm"),
        "C_obstacle": detail.get("C_obstacle"),
    }
    # ★ 偏移前后的轨迹平滑度对照 (用户要求: "前后的轨迹平滑代价也得参考一下")
    sm_before = trajectory_smoothness(approach)
    sm_after = trajectory_smoothness(apply_offset(approach, a_y, a_z))
    report["smooth_before"] = sm_before
    report["smooth_after"] = sm_after
    logger.info(
        "[N-5] 接近段优化: 穿透 %.2fmm → %.2fmm (偏移 a_y=%.1fmm, a_z=%.1fmm; "
        "代价 collide=%.3g follow=%.3g feasible=%.3g)",
        report["pen_before_mm"], report["pen_after_mm"],
        report["a_y_mm"], report["a_z_mm"],
        report["C_collide"], report["C_follow"], report["C_feasible"],
    )
    # ★ 2026-09-14 分段语义报告 (用户: "抓取前最好不要碰到被抓取物体" + "不只桌面, 还有被抓取物体")
    if report.get("pen_real_mm") is not None:
        _pre = report.get("n_contact_pre") or 0
        _tag_pre = "✓ 抓取前全程无接触" if _pre == 0 else f"✗ {_pre} 个控制点在抓取前碰到物体"
        logger.info(
            "[N-5 分段] 目标物体: 真穿透 %.2fmm (应→0) | 全段最小间隙 %.2fmm | "
            "u<%.2f 触碰点数 %d → %s",
            report["pen_real_mm"], report.get("gap_min_mm") or -1.0,
            report.get("u_split") or 0.0, _pre, _tag_pre)
        logger.info(
            "[N-5 分段] 障碍物(桌面+其他): 真穿透 %.2fmm (应=0) → %s",
            report.get("obst_pen_mm") or 0.0,
            "✓ 未穿透" if (report.get("obst_pen_mm") or 0.0) < 1e-9 else "✗ 有穿透")
    if sm_before and sm_after:
        logger.info(
            "[N-5] 平滑度对照 (归一化差分, 越小越顺): "
            "速度 %.3f→%.3f  加速度 %.3f→%.3f  jerk %.3g→%.3g",
            sm_before["v_max"], sm_after["v_max"],
            sm_before["a_max"], sm_after["a_max"],
            sm_before["j_max"], sm_after["j_max"],
        )
    return (a_y, a_z, a_x), report


def trajectory_smoothness(approach):
    """轨迹平滑度指标: 按实际 Δu 归一化的一/二/三阶差分的峰值.

    物理含义 ≈ 路径的 速度 / 加速度 / jerk 峰值 (单位分别为 m/u, m/u², m/u³).
    用于对照"偏移前后"的平滑性 (用户要求参考轨迹平滑代价).
    """
    if len(approach) < 4:
        return {}
    P = np.asarray([c["p"] for c in approach], dtype=np.float64)
    us = np.asarray([c["u"] for c in approach], dtype=np.float64)
    du = max(float(us[-1] - us[0]), 1e-6) / max(len(approach) - 1, 1)
    return {
        "v_max": float(np.linalg.norm(np.diff(P, axis=0), axis=1).max() / du),
        "a_max": float(np.linalg.norm(np.diff(P, n=2, axis=0), axis=1).max() / (du ** 2)),
        "j_max": float(np.linalg.norm(np.diff(P, n=3, axis=0), axis=1).max() / (du ** 3)),
    }


def apply_offset(approach, a_y, a_z, a_x=0.0):
    """把偏移曲线应用到接近段控制点, 就地修改 p (返回新列表)."""
    out = []
    for c in approach:
        R = np.asarray(c["R"], dtype=np.float64)
        p = np.asarray(c["p"], dtype=np.float64)
        dx, dy, dz = offset_curve(float(c["u"]), a_y, a_z, a_x)
        local = np.array([dx, dy, dz], dtype=np.float64)
        out.append({**c, "p": p + R @ local, "offset_local": local})
    return out

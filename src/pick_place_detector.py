#!/usr/bin/env python3
"""Pick/Place 帧检测 (Stage 3.7)

从第一视角视频到抓取-放置交互事件检测:
  输入 _pre_dedup_masks.pkl (all_masks + world_points)
  → 数据提取与预处理 (逐实例逐帧, 有限点过滤, 每帧≥10点, 总有效帧<10丢弃)
  → 运动分析 (两种策略, 按平均mask面积分治):
      小物体 (<1000px): 质心位移法 (帧间位移/累计路径/净位移/持续帧数)
      大物体 (≥1000px): 点云重叠法 (当前帧 vs 前5帧拼接点云的重叠率序列)
  → 交互事件检测:
      小物体: 平滑位移序列找局部峰值, 回落至基线确定 Pick/Place
      大物体: 连续低重叠片段, 片段起点=Pick, 片段终点=Place
  → 后处理与合并: 先合并相邻交互 (gap≤阈值), 再过滤短交互 (持续<阈值)
  → 帧号 × frame_ratio 转换到原始视频空间, 输出 pick_place_frames.json

阈值单位原则:
  物理量(米)   → 不随 frame_ratio 变换 (VGGT 世界坐标是米制)
  时长量(帧数) → 定义在原始视频帧空间, 使用时 ×/÷ frame_ratio 换算
  比率量       → 无量纲, 不变换
"""

import os
import json
import pickle

import numpy as np

from src.sg_deduplication import get_overlap_ratio

# Stage 3.7 是否先用 VLM 判断"哪个物体被操作" (默认开)。
# 由 mainv3 的 --no_vlm_interaction 控制; 单独跑本模块时默认开启,
# 任何 VLM 失败都会静默退化成纯几何判据。
try:
    _VLM_INTERACTION
except NameError:
    _VLM_INTERACTION = True

# ── 数据提取 ──
MIN_POINTS_PER_FRAME = 10    # 每帧至少有效 3D 点数
MIN_FRAMES = 10              # 实例总有效帧数下限, 低于则跳过

# ── 大小物体分治 ──
SMALL_AREA_THRESH = 1000     # 平均 mask 面积 (px): <此为小物体, ≥此为大物体

# ── 小物体: 质心位移法 (物理量: 米) ──
CENTROID_PATH_THRESH = 0.15  # 累计路程
CENTROID_NET_THRESH = 0.08   # 首末净位移
CENTROID_PNET_THRESH = 5.0   # path/net 上限 (排除来回抖动)
# 大物体 (重叠率法) 的质心净位移下限。
# 重叠率法会把"相机移动导致静止大物体可见表面变化"误判成动态, 因此要求
# 质心真的移动过。比小物体的 0.08 松, 因为大物体质心受 mask 边界变化影响大。
LARGE_CENTROID_NET_THRESH = 0.05
# 可操作物体的 3D 尺寸上限 (米)。超过就不可能是被手拿起的东西 ——
# mask 把背景一起包进去会产生巨大伪实例 (实测 weixin_v10 device_inst0 1.31m),
# 它会因为"跨帧罩住不同区域"而质心大幅位移, 被误判成抓得最准的动态物体。
MANIPULABLE_MAX_EXTENT = 0.60

# ── 大物体: 点云重叠法 ──
OVERLAP_WINDOW = 5           # 滚动参考窗口帧数
MATCH_LOW_THRESH = 0.50      # 重叠率低阈值

# ── 时长量 (原始视频帧空间) ──
MIN_MOTION_DURATION = 15     # 动态判定: 运动/低重叠总时长下限
MIN_INTERACT_DURATION = 15   # 交互持续时长下限 (过滤短交互)

# ── 时长阈值自适应 ──
# MIN_MOTION_DURATION / MIN_INTERACT_DURATION 是针对长视频 (>150 帧) 的经验值, 定义在
# 原始视频帧空间。但对 60~62 帧的短视频 (egodex_gripper20 那批), 一次抓取动作只持续
# 13~14 帧, 于是系统性踩在 15 之下被判静态。实测漏检:
#   vertical_pick_place_115 bottle_inst0: path=1.015m net=0.250m pnet=4.1 (硬条件全过)
#                                          仅 max_run=13 < 15 → 被判静态
#   basic_pick_place_46    cube_inst0   : min_match=0.111 (几乎完全离开原位)
#                                          仅 low_count=13 < 15 → 被判静态
#   vertical_pick_place_47 box_inst1    : min_match=0.000 (与原位零重叠), low_count=14
# 放宽时长是安全的: 真正的判别力在 net/pnet/min_match —— 上面这些抖动类实例的
# net 只有 3mm、pnet 却高达 130+, 放宽时长它们依然过不了。
# 取 min(阈值, n_frames//5) 并对 60 帧给出 12, 使 max_run=13 / low_count=13~14 能通过;
# 长视频 (n_frames >= 75) 完全不受影响。
DURATION_ADAPT_DIVISOR = 5
DURATION_ADAPT_FLOOR = 5


def _effective_duration(n_frames_rgb, base):
    """时长类阈值按时长自适应: 短视频按 n_frames_rgb // 5 缩小 (下限 DURATION_ADAPT_FLOOR)。

    长视频 (n_frames_rgb >= base * DURATION_ADAPT_DIVISOR) 保持 base 不变。
    n_frames_rgb 为 0/None 时也返回 base (调用方未提供时长信息 → 保持原行为)。
    """
    if not n_frames_rgb or int(n_frames_rgb) <= 0:
        return base
    return max(DURATION_ADAPT_FLOOR,
               min(base, int(n_frames_rgb) // DURATION_ADAPT_DIVISOR))
MERGE_GAP = 20               # 相邻交互合并间隔上限
MIN_LOW_FRAMES_RGB = 5       # 低重叠证据数下限 (RGB采样帧, 防个别离群帧触发)

# ── 交互检测细节 ──
PEAK_BASELINE_FACTOR = 2.0   # 小物体峰值须 > baseline × 2
PICK_FALLBACK_FACTOR = 1.5   # 向左回落到 baseline × 1.5 → Pick
PLACE_FALLBACK_FACTOR = 2.0  # 向右回落到 baseline × 2 → Place
MIN_CONFIDENCE = 0.5         # 低于此置信度的交互丢弃
CONF_DISP_SCALE = 0.2        # 小物体置信度 = min(1, 交互位移/0.2)
CONF_SEG_LEN_SCALE = 20      # 大物体置信度贡献: 段长/20

# ── 过程窗口 (手-物体交互阶段: "移动还小但已开始位移" → 明显移动) ──
MATCH_START_THRESH = 0.80    # 大物体: wm < 0.80 视为开始位移 (仍大部分在原位)
PEAK_WINDOW_FACTOR = 0.25    # 小物体: 位移 < 峰值×0.25 视为移动还小 (相对交互幅度自适应)


# ─────────────────────────────────────────────────────────────
# Step 1: 数据提取与预处理
# ─────────────────────────────────────────────────────────────
def _extract_instance_points(all_masks, world_points):
    """逐实例逐帧提取 mask 内 3D 点。

    返回: {key: {pts_list, fids, centroids, avg_area}}
      - pts_list: 每有效帧的 (N,3) 点云
      - fids:     有效帧号 (RGB 空间)
      - centroids: 每有效帧 3D 质心 (N,3)
      - avg_area: 平均 mask 面积 (px, 按所有可见帧)
    """
    instances = {}
    for category, inst_list in all_masks.items():
        if not isinstance(inst_list, list):
            continue
        for idx, instance_masks in enumerate(inst_list):
            key = f"{category}_inst{idx}"
            sm = sorted(instance_masks, key=lambda x: x['frame_id'])
            pts_list, fids, centroids, areas = [], [], [], []
            for im in sm:
                mask = im['mask']
                fid = im['frame_id']
                n_px = int(np.count_nonzero(mask))
                if n_px == 0:
                    continue
                areas.append(n_px)
                pts = world_points[fid][mask > 0]
                pts = pts[np.all(np.isfinite(pts), axis=-1)]
                if len(pts) < MIN_POINTS_PER_FRAME:
                    continue
                pts_list.append(pts.astype(np.float64))
                fids.append(int(fid))
                centroids.append(pts.mean(axis=0))
            if len(fids) < MIN_FRAMES:
                continue  # 过滤短序列
            instances[key] = {
                'pts_list': pts_list,
                'fids': fids,
                'centroids': np.array(centroids),
                'avg_area': float(np.mean(areas)) if areas else 0.0,
            }
    return instances


# ─────────────────────────────────────────────────────────────
# Step 2: 运动分析 (两种策略)
# ─────────────────────────────────────────────────────────────
def _centroid_features(centroids):
    """小物体: 质心位移序列 → path / net / pnet / max_run。"""
    cent = np.asarray(centroids)
    disps = np.linalg.norm(cent[1:] - cent[:-1], axis=1)
    path = float(disps.sum())
    net = float(np.linalg.norm(cent[-1] - cent[0]))
    pnet = path / (net + 1e-6)
    p50 = float(np.percentile(disps, 50)) if len(disps) else 0.0
    max_run = cur = 0
    for d in disps:
        if d > p50:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return {'disps': disps, 'path': path, 'net': net,
            'pnet': pnet, 'max_run': max_run}


def _overlap_features(pts_list):
    """大物体: 当前帧 vs 前5帧拼接点云的重叠率序列 → win_match / min_match / low_count。

    win_match[j] 与 fids[j] 对齐 (前 OVERLAP_WINDOW 帧无参考, 置 1.0)。
    low_count = 低重叠总帧数 (替代旧 r50 最长连续段, 慢速移动不漏检)。
    """
    n = len(pts_list)
    wm = [1.0] * min(OVERLAP_WINDOW, n)
    for j in range(OVERLAP_WINDOW, n):
        ref = np.concatenate(pts_list[j - OVERLAP_WINDOW:j], axis=0)
        cp = pts_list[j]
        if len(cp) >= MIN_POINTS_PER_FRAME and len(ref) >= MIN_POINTS_PER_FRAME:
            wm.append(float(get_overlap_ratio(cp, ref)))
        else:
            wm.append(1.0)
    low_count = sum(1 for v in wm if v < MATCH_LOW_THRESH)
    return {'win_match': wm, 'min_match': float(min(wm)),
            'low_count': int(low_count)}


def _instance_3d_extent(pts_list):
    """实例的**物体尺寸** (米): 逐帧点云最大边长的中位数。

    为什么不能用全部帧点云的并集范围: 那测的是"轨迹跨度"而不是"物体大小"。
    一个 5cm 的方块被抓着走过 80cm, 并集范围就是 0.85m —— 会把真实的可操作
    物体误杀 (实测 0.6m 阈值下 basic_pick_place_66 的 5 个真实动态实例全被挡)。
    逐帧边长才是物体自身的大小; 取中位数是为了抗单帧 mask 抖动。
    """
    if not pts_list:
        return 0.0
    ext = []
    for p in pts_list:
        p = np.asarray(p, dtype=np.float64)
        p = p[np.all(np.isfinite(p), axis=-1)]
        if len(p) >= MIN_POINTS_PER_FRAME:
            ext.append(float((p.max(axis=0) - p.min(axis=0)).max()))
    if not ext:
        return 0.0
    return float(np.median(ext))


def compute_dynamic_info(all_masks, world_points, frame_ratio=1.0):
    """对全部实例做动态判定 (大小物体分治)。

    返回: {key: {is_dynamic, method, avg_area, fids, pts_list, centroids, features}}
    """
    info = {}
    # 时长阈值按时长自适应 (短视频缩小、长视频不变, 见 _effective_duration 的说明)
    _n_rgb = int(world_points.shape[0]) if world_points is not None else 0
    min_dur = _effective_duration(_n_rgb, MIN_MOTION_DURATION)
    for key, d in _extract_instance_points(all_masks, world_points).items():
        # 无论走哪套判据, 都先算质心特征 —— 大物体路径要用它排除"相机移动
        # 导致可见表面变化"造成的假动态。
        cent = _centroid_features(d['centroids'])
        # 物理判据: 能被单手拿起的物体有尺寸上限。
        #
        # 为什么需要: mask 质量差时会把桌面/背景一起包进去, 产出一个巨大的伪
        # 实例。实测 weixin_v10 的 device_inst0: 3D 尺寸 1.31m、20 万点, 它
        # 的 mask 跨 69 帧且每帧罩住不同区域, 于是质心"位移"1.08m、
        # min_match=0.0, 被几何判据当成抓得最准的动态物体 (置信度 1.00) ——
        # 而它根本不是可操作物体。
        # 人单手拿不起 0.6m 以上的东西, 这是与场景无关的物理约束。
        _ext = _instance_3d_extent(d['pts_list'])
        if _ext > MANIPULABLE_MAX_EXTENT:
            info[key] = {
                'is_dynamic': False,
                'method': 'too_large',
                'avg_area': d['avg_area'],
                'fids': d['fids'],
                'pts_list': d['pts_list'],
                'centroids': d['centroids'],
                'features': {'suppressed_by':
                             f"3D 尺寸 {_ext:.2f}m > {MANIPULABLE_MAX_EXTENT}m "
                             f"(单手拿不起)"},
                'centroid_features': cent,
            }
            continue
        if d['avg_area'] < SMALL_AREA_THRESH:
            feats = cent
            is_dyn = (feats['path'] > CENTROID_PATH_THRESH
                      and feats['net'] > CENTROID_NET_THRESH
                      and feats['pnet'] < CENTROID_PNET_THRESH
                      and feats['max_run'] * frame_ratio >= min_dur)
        else:
            feats = _overlap_features(d['pts_list'])
            is_dyn = (feats['min_match'] < MATCH_LOW_THRESH
                      and feats['low_count'] * frame_ratio >= min_dur
                      and feats['low_count'] >= MIN_LOW_FRAMES_RGB)
            # 大物体补一道"质心必须真的移动"的门槛。
            #
            # 为什么需要: 重叠率法测的是"当前帧点云 vs 前 5 帧点云"的接近程度。
            # 相机移动时, 同一个静止大物体的可见表面会大幅变化 (新露出来的面
            # 不在前 5 帧的点云里), 重叠率随之暴跌 → 被误判成"在动"。
            # 实测 weixin_v8: table_inst1 (0.66x0.65x0.28m) 的 min_match=0.12、
            # low_count=5, 完全满足原有三条判据, 于是桌子被判成被抓取的物体,
            # 而真正被抓的耳机盒 (min_match=0.00, 反而更"动") 因 low_count=2 被否。
            #
            # 质心位移是直接的物理量: 桌子被相机移动带来的"可见区域质心"漂移
            # 通常只有几厘米, 真被抓起的物体会有几十厘米量级的净位移。
            # 阈值取 0.05m (5cm), 比 CENTROID_NET_THRESH(0.08) 松 —— 大物体的
            # 质心受 mask 边界变化影响更大, 用同一阈值会误杀。
            if is_dyn and cent['net'] < LARGE_CENTROID_NET_THRESH:
                is_dyn = False
                feats = dict(feats)
                feats['suppressed_by'] = (
                    f"质心净位移 {cent['net']:.3f}m < {LARGE_CENTROID_NET_THRESH}m "
                    f"(相机移动导致的假动态)")
        info[key] = {
            'is_dynamic': bool(is_dyn),
            'method': 'centroid' if d['avg_area'] < SMALL_AREA_THRESH else 'overlap',
            'avg_area': d['avg_area'],
            'fids': d['fids'],
            'pts_list': d['pts_list'],
            'centroids': d['centroids'],
            'features': feats,
            'centroid_features': cent,
        }
    return info


# ─────────────────────────────────────────────────────────────
# Step 3: 交互事件检测 (Pick / Place), RGB 帧空间
# ─────────────────────────────────────────────────────────────
def _detect_interactions_small(key, fids, centroids, feats):
    """小物体: 平滑位移序列找局部峰值, 向左/右回落至基线确定 Pick/Place。"""
    disps = feats['disps']
    n = len(fids)
    if len(disps) < 3:
        return []
    sm = np.convolve(disps, np.ones(3) / 3, mode='same')
    baseline = float(np.median(sm))
    interactions = []
    i = 1
    while i < len(sm) - 1:
        if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] \
                and sm[i] > baseline * PEAK_BASELINE_FACTOR:
            # 向左找上升起点 (回落到 baseline×1.5 以下)
            L = i
            while L > 0 and sm[L] > baseline * PICK_FALLBACK_FACTOR:
                L -= 1
            # 向右找回落点 (回落到 baseline×2 以下)
            R = i
            while R < len(sm) - 1 and sm[R] > baseline * PLACE_FALLBACK_FACTOR:
                R += 1
            place_idx = min(R + 1, n - 1)  # disps[k] 覆盖 fid[k]→fid[k+1]
            disp = float(np.linalg.norm(centroids[place_idx] - centroids[L]))
            conf = min(1.0, disp / CONF_DISP_SCALE)
            if conf >= MIN_CONFIDENCE:
                # 过程窗口: 位移回落到 峰值×PEAK_WINDOW_FACTOR 以下 = "移动还小"
                win_thr = sm[i] * PEAK_WINDOW_FACTOR
                j = L
                while j > 0 and sm[j] > win_thr:
                    j -= 1
                pick_win_start = min(j + 1, L)  # 运动开始于 transition j→j+1
                j = place_idx
                while j < len(sm) - 1 and sm[j] > win_thr:
                    j += 1
                place_win_end = min(j + 1, n - 1)
                interactions.append({
                    'instance_name': key,
                    'pick_window': [float(fids[pick_win_start]), float(fids[L])],
                    'pick_frame': float(fids[L]),
                    'place_frame': float(fids[place_idx]),
                    'place_window': [float(fids[place_idx]), float(fids[place_win_end])],
                    'confidence': conf,
                })
            i = R + 1
        else:
            i += 1
    return interactions


def _detect_interactions_large(key, fids, feats):
    """大物体: 连续低重叠片段 → 片段起点=Pick, 片段终点=Place。"""
    wm = feats['win_match']
    interactions = []
    s = None
    for j, v in enumerate(wm + [1.0]):  # 哨兵收尾
        if v < MATCH_LOW_THRESH:
            if s is None:
                s = j
        elif s is not None:
            seg_min = min(wm[s:j])
            conf = min(1.0, (j - s) / CONF_SEG_LEN_SCALE + (1.0 - seg_min))
            # 过程窗口: wm < MATCH_START_THRESH (开始位移, 大部分仍在原位)
            k = s
            while k > 0 and wm[k - 1] < MATCH_START_THRESH:
                k -= 1
            k2 = j - 1
            while k2 + 1 < len(wm) and wm[k2 + 1] < MATCH_START_THRESH:
                k2 += 1
            interactions.append({
                'instance_name': key,
                'pick_window': [float(fids[k]), float(fids[s])],
                'pick_frame': float(fids[s]),
                'place_frame': float(fids[j - 1]),
                'place_window': [float(fids[j - 1]), float(fids[k2])],
                'confidence': conf,
            })
            s = None
    return interactions


# ─────────────────────────────────────────────────────────────
# Step 4: 后处理与合并 (顺序关键: 先合并后过滤)
# ─────────────────────────────────────────────────────────────
def postprocess_interactions(interactions, frame_ratio, n_frames_rgb=0):
    """合并相邻交互 (gap≤MERGE_GAP 原始帧) → 过滤短交互 (<MIN_INTERACT_DURATION 原始帧)。

    输入/合并/过滤均在 RGB 帧空间进行 (时长阈值 ÷ frame_ratio 换算),
    输出帧号 × frame_ratio 转换到原始视频帧空间。

    n_frames_rgb: 用于时长阈值自适应 (短视频会缩短下限, 见 _effective_duration)。
        为 0 时保持原行为 (不缩放), 以兼容既有调用与测试。
    """
    merge_gap_rgb = MERGE_GAP / frame_ratio
    min_interact_dur = _effective_duration(n_frames_rgb, MIN_INTERACT_DURATION)
    out = []
    for inst in sorted(set(i['instance_name'] for i in interactions)):
        items = sorted((i for i in interactions if i['instance_name'] == inst),
                       key=lambda x: x['pick_frame'])
        cur = None
        for it in items:
            if cur is None:
                cur = dict(it)
                continue
            if it['pick_frame'] - cur['place_frame'] <= merge_gap_rgb:
                cur['place_frame'] = max(cur['place_frame'], it['place_frame'])
                cur['confidence'] = max(cur['confidence'], it['confidence'])
                cur['pick_window'][0] = min(cur['pick_window'][0], it['pick_window'][0])
                cur['place_window'][1] = max(cur['place_window'][1], it['place_window'][1])
                cur['place_window'][0] = cur['place_frame']  # 同步到合并后的 place
            else:
                out.append(cur)
                cur = dict(it)
        if cur is not None:
            out.append(cur)
    # 过滤短交互 (时长下限同样自适应)
    kept = [i for i in out
            if (i['place_frame'] - i['pick_frame']) * frame_ratio >= min_interact_dur]
    # 转换到原始视频帧空间
    for i in kept:
        i['pick_frame'] = int(round(i['pick_frame'] * frame_ratio))
        i['place_frame'] = int(round(i['place_frame'] * frame_ratio))
        i['pick_window'] = [int(round(p * frame_ratio)) for p in i['pick_window']]
        i['place_window'] = [int(round(p * frame_ratio)) for p in i['place_window']]
    return kept


# ─────────────────────────────────────────────────────────────
# Step 5: 主入口 — Stage 3.7
# ─────────────────────────────────────────────────────────────
def run_stage3_7_pick_place(output_path, all_masks=None, world_points=None,
                            frame_ratio=None):
    """Pick/Place 帧检测主入口。

    参数:
        output_path:  场景输出目录 (含 _pre_dedup_masks.pkl / coordinate_alignment.json)
        all_masks:    可选, 直接传入去重 masks (默认从 pkl 读)
        world_points: 可选, 直接传入 VGGT world_points (默认从 pkl 读)
        frame_ratio:  原始视频帧数 / RGB帧数; 默认从 coordinate_alignment.json
                      的 total_frames/n_frames 读取, 都没有则 1.0
    返回:
        {'frame_ratio', 'n_frames_rgb', 'total_frames', 'interactions'}
        并写入 output_path/pick_place_frames.json
    """
    if all_masks is None or world_points is None:
        with open(os.path.join(output_path, '_pre_dedup_masks.pkl'), 'rb') as f:
            data = pickle.load(f)
        all_masks = data['all_masks']
        world_points = data['world_points']
    n_frames_rgb = int(world_points.shape[0])
    total_frames = None
    ratio_source = 'argument'
    if frame_ratio is None:
        ca_path = os.path.join(output_path, 'coordinate_alignment.json')
        if os.path.exists(ca_path):
            with open(ca_path) as f:
                cad = json.load(f)
            total_frames = cad.get('total_frames')
            if total_frames:
                frame_ratio = float(total_frames) / n_frames_rgb
                ratio_source = 'coordinate_alignment.total_frames'
        if frame_ratio is None:
            frame_ratio = 1.0
            ratio_source = 'fallback_1.0'
            # 时长类阈值 (MIN_MOTION_DURATION 等) 定义在原始视频帧空间, 小物体判
            # 动态用的是 `max_run × frame_ratio >= MIN_MOTION_DURATION`。
            # 这里拿不到 total_frames, 该条件会退化成"连续 15 个 RGB 帧"。而输入
            # 视频都是抽帧的, 抽帧后连续位移段本来就被截短, 所以这个条件会偏严:
            # 实测 002_C1_Bowl 的 brush_inst0 (path/net/pnet 三条都过, 只因
            # max_run=7 < 15 被卡) 就是这样漏掉的。
            #
            # 时长阈值现已按视频长度自适应 (见 _effective_duration): 短视频会用
            # n_frames//5 缩小下限。所以这里的风险已被部分对冲, 但仍需提示 ——
            # ratio 退化意味着"原始帧空间"与"RGB 帧空间"混同, 帧号语义仍不准确。
            print(f"   ⚠️ [Stage 3.7] 未取到 total_frames, frame_ratio 退化为 1.0 "
                  f"(来源: {ratio_source})", flush=True)
            print(f"      → 时长阈值已按 {n_frames_rgb} 帧自适应, 但帧号仍在 RGB 空间",
                  flush=True)
    else:
        total_frames = int(round(n_frames_rgb * frame_ratio))

    # ── VLM 先行: 判断哪些物体被手拿起/放下 ──
    # 几何判据 (位移/重叠率) 对"大静态物被相机移动欺骗"和"小物体轨迹太短"
    # 两类情况会失效。VLM 直接看画面回答"手拿起了什么", 是语义判断, 不依赖
    # 阈值。二者互补: VLM 给类别 + 大致帧, 几何给实例 + 精确帧。
    # 任何失败都返回 [], 自动退化成纯几何判据 (用 --no_vlm_interaction 关闭)。
    vlm_items = []
    if _VLM_INTERACTION:
        try:
            from src.vlm_interaction import vlm_detect_interactions
            _color = os.path.join(output_path, 'color')
            _frames = sorted(os.path.join(_color, f) for f in os.listdir(_color)
                             if f.lower().endswith(('.jpg', '.png'))) \
                if os.path.isdir(_color) else []
            if _frames:
                vlm_items = vlm_detect_interactions(
                    _frames, frame_ratio=frame_ratio, verbose=True)
        except Exception as _e:
            print(f"   ⚠️ [Stage 3.7] VLM 交互判断失败 ({type(_e).__name__}: "
                  f"{_e}), 退化为纯几何判据", flush=True)

    info = compute_dynamic_info(all_masks, world_points, frame_ratio)
    interactions = []
    for key in sorted(info):
        d = info[key]
        if not d['is_dynamic']:
            continue
        if d['method'] == 'centroid':
            interactions += _detect_interactions_small(
                key, d['fids'], d['centroids'], d['features'])
        else:
            interactions += _detect_interactions_large(
                key, d['fids'], d['features'])
    interactions = postprocess_interactions(interactions, frame_ratio, n_frames_rgb)

    # ── VLM × 几何 融合 ──
    # 规则 (VLM 管语义, 几何管精度):
    #   1. VLM 报的类别在几何结果里有对应实例 → 用几何的精确 pick/place 帧
    #   2. VLM 报了但几何没检出 → 用 VLM 的帧 + 该类别的第一个实例补一条
    #      (几何漏检时这是唯一的救命通道)
    #   3. 几何检出了但 VLM 没提 → 保留, 标 source='geometry' (VLM 可能漏看)
    if vlm_items:
        try:
            from src.vlm_interaction import match_vlm_to_instances
            _cats = sorted({k.rsplit('_inst', 1)[0] for k in info})
            _map = match_vlm_to_instances(vlm_items, _cats)
            _geo_by_cat = {}
            for _it in interactions:
                _geo_by_cat.setdefault(
                    _it['instance_name'].rsplit('_inst', 1)[0], []).append(_it)
            _added = 0
            for _vi in vlm_items:
                _c = _map.get(_vi['object'])
                if _c is None:
                    continue
                if _geo_by_cat.get(_c):
                    for _it in _geo_by_cat[_c]:
                        _it['source'] = 'vlm+geometry'
                    continue
                _cand = [(k, info[k]) for k in info
                         if k.rsplit('_inst', 1)[0] == _c]
                if not _cand:
                    continue
                _cand.sort(key=lambda kv: -len(kv[1].get('fids', [])))
                _key = _cand[0][0]
                _pick_rgb = int(round(_vi['pick_frame'] / max(frame_ratio, 1e-6)))
                _place_rgb = int(round(_vi['place_frame'] / max(frame_ratio, 1e-6)))
                interactions.append({
                    'instance_name': _key,
                    'pick_frame': _vi['pick_frame'],
                    'place_frame': _vi['place_frame'],
                    'pick_window': [_pick_rgb, _pick_rgb],
                    'place_window': [_place_rgb, _place_rgb],
                    'confidence': 0.9,
                    'source': 'vlm',
                })
                _added += 1
            if _added:
                print(f"   [Stage 3.7] VLM 补回 {_added} 条几何漏检的交互",
                      flush=True)
            interactions = postprocess_interactions(
                interactions, frame_ratio, n_frames_rgb)
        except Exception as _e:
            print(f"   ⚠️ [Stage 3.7] VLM 融合失败 ({type(_e).__name__}: {_e}), "
                  f"沿用纯几何结果", flush=True)

    result = {
        'frame_ratio': frame_ratio,
        # 便于事后区分"真实算出的 ratio"与"退化成 1.0",
        # 后者意味着小物体判定偏严 (见上面的告警)
        'ratio_source': ratio_source,
        'n_frames_rgb': n_frames_rgb,
        'total_frames': total_frames if total_frames else int(round(n_frames_rgb * frame_ratio)),
        'interactions': interactions,
    }
    out_path = os.path.join(output_path, 'pick_place_frames.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result

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

# ── 数据提取 ──
MIN_POINTS_PER_FRAME = 10    # 每帧至少有效 3D 点数
MIN_FRAMES = 10              # 实例总有效帧数下限, 低于则跳过

# ── 大小物体分治 ──
SMALL_AREA_THRESH = 1000     # 平均 mask 面积 (px): <此为小物体, ≥此为大物体

# ── 小物体: 质心位移法 (物理量: 米) ──
CENTROID_PATH_THRESH = 0.15  # 累计路程
CENTROID_NET_THRESH = 0.08   # 首末净位移
CENTROID_PNET_THRESH = 5.0   # path/net 上限 (排除来回抖动)

# ── 大物体: 点云重叠法 ──
OVERLAP_WINDOW = 5           # 滚动参考窗口帧数
MATCH_LOW_THRESH = 0.50      # 重叠率低阈值

# ── 时长量 (原始视频帧空间) ──
MIN_MOTION_DURATION = 15     # 动态判定: 运动/低重叠总时长下限
MIN_INTERACT_DURATION = 15   # 交互持续时长下限 (过滤短交互)
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


def compute_dynamic_info(all_masks, world_points, frame_ratio=1.0):
    """对全部实例做动态判定 (大小物体分治)。

    返回: {key: {is_dynamic, method, avg_area, fids, pts_list, centroids, features}}
    """
    info = {}
    for key, d in _extract_instance_points(all_masks, world_points).items():
        if d['avg_area'] < SMALL_AREA_THRESH:
            feats = _centroid_features(d['centroids'])
            is_dyn = (feats['path'] > CENTROID_PATH_THRESH
                      and feats['net'] > CENTROID_NET_THRESH
                      and feats['pnet'] < CENTROID_PNET_THRESH
                      and feats['max_run'] * frame_ratio >= MIN_MOTION_DURATION)
        else:
            feats = _overlap_features(d['pts_list'])
            is_dyn = (feats['min_match'] < MATCH_LOW_THRESH
                      and feats['low_count'] * frame_ratio >= MIN_MOTION_DURATION
                      and feats['low_count'] >= MIN_LOW_FRAMES_RGB)
        info[key] = {
            'is_dynamic': bool(is_dyn),
            'method': 'centroid' if d['avg_area'] < SMALL_AREA_THRESH else 'overlap',
            'avg_area': d['avg_area'],
            'fids': d['fids'],
            'pts_list': d['pts_list'],
            'centroids': d['centroids'],
            'features': feats,
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
def postprocess_interactions(interactions, frame_ratio):
    """合并相邻交互 (gap≤MERGE_GAP 原始帧) → 过滤短交互 (<MIN_INTERACT_DURATION 原始帧)。

    输入/合并/过滤均在 RGB 帧空间进行 (时长阈值 ÷ frame_ratio 换算),
    输出帧号 × frame_ratio 转换到原始视频帧空间。
    """
    merge_gap_rgb = MERGE_GAP / frame_ratio
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
    # 过滤短交互
    kept = [i for i in out
            if (i['place_frame'] - i['pick_frame']) * frame_ratio >= MIN_INTERACT_DURATION]
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
    if frame_ratio is None:
        ca_path = os.path.join(output_path, 'coordinate_alignment.json')
        if os.path.exists(ca_path):
            with open(ca_path) as f:
                cad = json.load(f)
            total_frames = cad.get('total_frames')
            if total_frames:
                frame_ratio = float(total_frames) / n_frames_rgb
        if frame_ratio is None:
            frame_ratio = 1.0
    else:
        total_frames = int(round(n_frames_rgb * frame_ratio))

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
    interactions = postprocess_interactions(interactions, frame_ratio)

    result = {
        'frame_ratio': frame_ratio,
        'n_frames_rgb': n_frames_rgb,
        'total_frames': total_frames if total_frames else int(round(n_frames_rgb * frame_ratio)),
        'interactions': interactions,
    }
    out_path = os.path.join(output_path, 'pick_place_frames.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result

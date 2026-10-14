#!/usr/bin/env python3
"""可视化 pick/place 检测结果: 在 color 帧上标注 pick/place 时刻 + 整段 MP4 渲染。

1) 单帧标注图: 对每个交互, 在 rgb = round(原始帧号/frame_ratio) 对应的 color 图上:
  - PICK: 绿色边框 / PLACE: 红色边框 / PICK_WIN: 青色 / PLACE_WIN: 黄色
2) 整段视频渲染: pick_place_vis/pick_place_annotated.mp4
  每帧根据所处阶段画边框+文字:
  - PICK_WIN(青) → PICK(绿) → CARRY(蓝细框, 搬运中) → PLACE(红) → PLACE_WIN(黄)
  左上角显示阶段/实例/帧号(rgb与原始), 底部进度条
"""
import os, sys, json, shutil
import cv2

BASE = str(os.path.join(os.path.dirname(__file__), '..', 'output_v2'))
FPS = 10  # 输出视频帧率 (检查用)

# 阶段 → (边框颜色BGR, 边框粗细, 标签)
STYLE = {
    'PICK_WIN':  ((255, 255, 0),   10, 'PICK_WIN'),
    'PICK':      ((0, 255, 0),     14, 'PICK'),
    'CARRY':     ((255, 128, 0),    4, 'CARRY'),
    'PLACE':     ((0, 0, 255),     14, 'PLACE'),
    'PLACE_WIN': ((0, 255, 255),   10, 'PLACE_WIN'),
}


def frame_phase_map(interactions, ratio, n_rgb):
    """RGB帧号 → (阶段, 交互序号)。窗口/锚点帧号转回RGB空间后归类。"""
    phase = {}
    for idx, it in enumerate(interactions):
        pick = int(round(it['pick_frame'] / ratio))
        place = int(round(it['place_frame'] / ratio))
        pw0, pw1 = int(round(it['pick_window'][0] / ratio)), int(round(it['pick_window'][1] / ratio))
        dw0, dw1 = int(round(it['place_window'][0] / ratio)), int(round(it['place_window'][1] / ratio))
        for f in range(max(0, pw0), min(pw1, n_rgb - 1) + 1):
            phase[f] = ('PICK_WIN', idx)
        for f in range(max(0, pw1 + 1), place):          # pick 与 place 之间 = 搬运
            phase[f] = ('CARRY', idx)
        for f in range(max(0, dw0), min(dw1, n_rgb - 1) + 1):
            phase[f] = ('PLACE_WIN', idx)
        if 0 <= pick < n_rgb:
            phase[pick] = ('PICK', idx)
        if 0 <= place < n_rgb:
            phase[place] = ('PLACE', idx)
    return phase


def render_video(scene_dir, data):
    """渲染整段标注 MP4 到 pick_place_vis/pick_place_annotated.mp4"""
    interactions = data['interactions']
    ratio = data['frame_ratio']
    n_rgb = data['n_frames_rgb']
    phase = frame_phase_map(interactions, ratio, n_rgb)

    sample = cv2.imread(os.path.join(scene_dir, 'color', '0.jpg'))
    h, w = sample.shape[:2]
    vis_dir = os.path.join(scene_dir, 'pick_place_vis')
    out_path = os.path.join(vis_dir, 'pick_place_annotated.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    if not writer.isOpened():  # mp4v 失败回退 avi
        out_path = os.path.join(vis_dir, 'pick_place_annotated.avi')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), FPS, (w, h))
    assert writer.isOpened(), f"VideoWriter 打开失败: {out_path}"

    for fid in range(n_rgb):
        img_path = os.path.join(scene_dir, 'color', f'{fid}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 1) 阶段边框 + 文字
        if fid in phase:
            tag, idx = phase[fid]
            color, thick, label = STYLE[tag]
            inst = interactions[idx]['instance_name']
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thick)
            text = f"[{label}] {inst} rgb={fid} orig~{int(round(fid * ratio))}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(img, (10, 10), (20 + tw, 30 + th), (0, 0, 0), -1)
            cv2.putText(img, text, (15, 25 + th), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
        else:
            text = f"rgb={fid} orig~{int(round(fid * ratio))}"
            cv2.rectangle(img, (10, 10), (130, 46), (0, 0, 0), -1)
            cv2.putText(img, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # 2) 底部进度条
        cv2.rectangle(img, (0, h - 8), (int(w * (fid + 1) / n_rgb), h), (0, 255, 255), -1)
        writer.write(img)
    writer.release()
    return out_path


ONLY = sys.argv[1:]

SCENES = {
    '2chair-bottle_vggt_omega':            2.29,
    '7_vggt_omega':                        1.00,
    '3chair-bottle_vggt_omega':            4.93,
    '121_C5_CellPhone_161deg_vggt_omega':  3.00,
    '116_C5_CellPhone_185deg_vggt_omega':  3.00,
    # 2026-10-10: 下面 4 个场景的 pick_place_frames.json 已用当前代码重新生成
    # (原产物是 2026-07-30 的旧格式, 缺 pick_window/place_window, 本脚本会 KeyError)。
    # 它们没有 coordinate_alignment.json, frame_ratio 退化为 1.0 (帧号即 RGB 空间)。
    '002_C1_Bowl_364deg_vggt_omega':       1.00,
    'beizi_vggt_omega':                    1.00,
    'bottle_vggt_omega':                   1.00,
    'phone_vggt_omega':                    1.00,
    '86_verify':                           3.01875,
}

for scene, ratio in SCENES.items():
    if ONLY and scene not in ONLY:
        continue
    scene_dir = os.path.join(BASE, scene)
    # 优先用 test_combined.py 的 combined 产物, 回退到 Stage 3.7 的标准输出
    json_path = os.path.join(scene_dir, 'pick_place_frames_combined.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(scene_dir, 'pick_place_frames.json')
    if not os.path.exists(json_path):
        continue
    with open(json_path) as f:
        data = json.load(f)
    n_rgb = data['n_frames_rgb']
    ratio = data['frame_ratio']

    vis_dir = os.path.join(scene_dir, 'pick_place_vis')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)

    print(f"\n{scene} (ratio={ratio})")
    for it in data['interactions']:
        inst = it['instance_name']
        # (标签, 原始帧号, 边框颜色BGR): PICK绿 / PLACE红 / 窗口起点青 / 窗口终点黄
        marks = [
            ('PICK', it['pick_frame'], (0, 255, 0)),
            ('PLACE', it['place_frame'], (0, 0, 255)),
            ('PICK_WIN', it['pick_window'][0], (255, 255, 0)),
            ('PLACE_WIN', it['place_window'][1], (0, 255, 255)),
        ]
        for tag, orig, color in marks:
            rgb = int(round(orig / ratio))
            rgb = max(0, min(n_rgb - 1, rgb))
            img_path = os.path.join(scene_dir, 'color', f'{rgb}.jpg')
            if not os.path.exists(img_path):
                print(f"  [skip] {img_path} 不存在")
                continue
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 12)
            label = f"{inst} {tag} orig={orig} rgb={rgb}"
            cv2.rectangle(img, (10, 10), (10 + 22 * len(label), 70), (0, 0, 0), -1)
            cv2.putText(img, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, color, 3, cv2.LINE_AA)
            out = os.path.join(vis_dir, f'{inst}_{tag}_{orig}.jpg')
            cv2.imwrite(out, img)
            print(f"  {label}  → {os.path.basename(out)}")

    # 整段 MP4 渲染
    video = render_video(scene_dir, data)
    print(f"  视频 → {video}")

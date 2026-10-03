#!/usr/bin/env python3
"""
在 HaWoR 原始视频上画 MANO 参考点, 验证投影是否正确
视频分辨率 6640×1080, mask 是中心裁切 1920×1080
"""
import numpy as np, cv2, torch, argparse
from pathlib import Path
from mano_layer import MANOLayer

def load_hawor_data(hawor_dir):
    d = {}
    npz = sorted(Path(hawor_dir).rglob('hawor_results_*.npz'))
    npz = [f for f in npz if '_depth_aligned' not in str(f) and '_aligned' not in str(f)]
    d_raw = dict(np.load(str(npz[0]), allow_pickle=True))
    
    masks_path = str(list(Path(hawor_dir).glob('tracks_*/model_masks.npy'))[0])
    masks = np.load(masks_path, allow_pickle=True)
    if masks.ndim == 0: masks = masks.item()
    if masks.ndim > 3 and masks.shape[0] == 1: masks = masks[0]
    
    d.update({'npz': d_raw, 'masks': masks})
    return d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', default='/home/an/data/hawor/7')
    parser.add_argument('--hidx', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output-dir', default='/tmp/hawor_video_test')
    args = parser.parse_args()
    
    d = load_hawor_data(args.hawor_dir)
    hidx = args.hidx
    
    masks = d['masks']
    cx = 960.0  # mask 1920宽的中心
    cy = 540.0  # mask 1080高的中心
    video_w, video_h = 6640, 1080
    
    # mask 是中心裁切
    cx_offset = (video_w - 1920) / 2  # 2360
    
    img_focal = float(d['npz']['img_focal'])
    print(f"img_focal = {img_focal}")
    print(f"mask 分辨率: {masks.shape}")
    
    # 找有效帧
    if args.frame == 0:
        for fi in range(d['npz']['pred_trans'].shape[1]):
            t = d['npz']['pred_trans'][hidx, fi]
            if not np.any(np.isnan(t)):
                args.frame = fi
                break
    fi = args.frame
    
    d_npz = d['npz']
    t = d_npz['pred_trans'][hidx, fi]
    rot = d_npz['pred_rot'][hidx, fi]
    hp = d_npz['pred_hand_pose'][hidx, fi]
    
    print(f"frame {fi}: pred_trans = {t}")
    
    # 找第一个有效的 betas
    betas_arr = d_npz['pred_betas'][hidx]
    valid_b = next((i for i in range(betas_arr.shape[0]) if not np.any(np.isnan(betas_arr[i]))), 0)
    handedness = 'right' if hidx == 1 else 'left'
    ml = MANOLayer(handedness, betas_arr[valid_b].astype(np.float32))
    
    p_t = torch.from_numpy(np.concatenate([rot.astype(np.float32), hp.astype(np.float32)])).unsqueeze(0)
    t_t = torch.from_numpy(t.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        verts, joints = ml(p_t, t_t)
    vv = verts[0].numpy()
    jv = joints[0].numpy()
    
    mask = masks[fi] if fi < masks.shape[0] else masks[-1]
    ys, xs = np.where(mask > 0)
    mask_min_x, mask_max_x = int(xs.min()), int(xs.max())
    mask_min_y, mask_max_y = int(ys.min()), int(ys.max())
    
    # 从视频读取一帧
    videos = sorted(Path(args.hawor_dir).glob('*.mp4'))
    if not videos:
        for p in Path(args.hawor_dir).rglob('*.mp4'):
            videos.append(p)
    videos = sorted(videos)
    
    cap = cv2.VideoCapture(str(videos[0]))
    video_w_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频实际分辨率: {video_w_actual}x{video_h_actual}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    _, frame = cap.read()
    cap.release()
    
    if frame is None:
        print("无法读取视频帧")
        return
    
    frame = cv2.resize(frame, (video_w_actual, video_h_actual))
    
    # === 方案 A: img_focal 投影到 6640 宽, mask 是左侧 1920 ===
    # 直接投影到视频, cx=video_w/2=3320
    out_a = frame.copy()
    fx = img_focal
    cx_v = video_w / 2
    cy_v = video_h / 2
    
    # pred_trans 投影
    u_t = fx * t[0] / t[2] + cx_v
    v_t = fx * t[1] / t[2] + cy_v
    cv2.circle(out_a, (int(round(u_t)), int(round(v_t))), 8, (0, 0, 255), -1)
    
    # MANO verts 投影
    for v in vv:
        u = fx * v[0] / v[2] + cx_v
        uval = fx * v[1] / v[2] + cy_v
        ui, vi = int(round(u)), int(round(uval))
        if 0 <= ui < video_w and 0 <= vi < video_h:
            cv2.circle(out_a, (ui, vi), 2, (255, 0, 0), -1)
    
    # mask 轮廓 (左侧1920)
    for y in range(1080):
        for x in range(1920):
            if mask[y, x] > 0:
                if y >= 1 and y < 1079 and x >= 1 and x < 1919:
                    continue  # 只画边缘
                if y == 0 or y == 1079 or x == 0 or x == 1919:
                    out_a[y, x, :] = [255, 255, 255]
    
    # === 方案 B: img_focal 投影到 6640 宽, mask 是中心 1920 ===
    out_b = frame.copy()
    # mask 中心在 (3320, 540)
    for y in range(1080):
        for x in range(1920):
            if mask[y, x] > 0:
                if y >= 1 and y < 1079 and x >= 1 and x < 1919:
                    continue
                vx = x + cx_offset
                out_b[y, int(vx), :] = [255, 255, 255]
    
    # pred_trans 投影
    cv2.circle(out_b, (int(round(u_t)), int(round(v_t))), 8, (0, 0, 255), -1)
    
    for v in vv:
        u = fx * v[0] / v[2] + cx_v
        uval = fx * v[1] / v[2] + cy_v
        ui, vi = int(round(u)), int(round(uval))
        if 0 <= ui < video_w and 0 <= vi < video_h:
            cv2.circle(out_b, (ui, vi), 2, (255, 0, 0), -1)
    
    # === 方案 C: 假设 mask 是左侧, 但用 fx = img_focal * 1920/6640 ===
    out_c = frame.copy()
    fx_c = img_focal * 1920 / video_w
    cx_c = 960.0
    cy_c = 540.0
    
    # pred_trans 投影
    u_t_c = fx_c * t[0] / t[2] + cx_c
    v_t_c = fx_c * t[1] / t[2] + cy_c
    cv2.circle(out_c, (int(round(u_t_c)), int(round(v_t_c))), 8, (0, 0, 255), -1)
    
    for v in vv:
        u = fx_c * v[0] / v[2] + cx_c
        uval = fx_c * v[1] / v[2] + cy_c
        ui, vi = int(round(u)), int(round(uval))
        if 0 <= ui < 1920 and 0 <= vi < 1080:
            cv2.circle(out_c, (ui, vi), 2, (255, 0, 0), -1)
    
    for y in range(1080):
        for x in range(1920):
            if mask[y, x] > 0:
                if y >= 1 and y < 1079 and x >= 1 and x < 1919:
                    continue
                out_c[y, x, :] = [255, 255, 255]
    
    # === 方案 D: 假设 mask 是中心, 用 fx = img_focal * 1920/6640 ===
    out_d = frame.copy()
    fx_d = img_focal * 1920 / video_w
    cx_d = 960.0 + cx_offset
    cy_d = 540.0
    
    u_t_d = fx_d * t[0] / t[2] + cx_d
    v_t_d = fx_d * t[1] / t[2] + cy_d
    cv2.circle(out_d, (int(round(u_t_d)), int(round(v_t_d))), 8, (0, 0, 255), -1)
    
    for v in vv:
        u = fx_d * v[0] / v[2] + cx_d
        uval = fx_d * v[1] / v[2] + cy_d
        ui, vi = int(round(u)), int(round(uval))
        if 0 <= ui < video_w and 0 <= vi < video_h:
            cv2.circle(out_d, (ui, vi), 2, (255, 0, 0), -1)
    
    for y in range(1080):
        for x in range(1920):
            if mask[y, x] > 0:
                if y >= 1 and y < 1079 and x >= 1 and x < 1919:
                    continue
                vx = x + cx_offset
                out_d[y, int(vx), :] = [255, 255, 255]
    
    # 保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / 'A_left1920_original_focal.png'), out_a)
    cv2.imwrite(str(out_dir / 'B_center1920_original_focal.png'), out_b)
    cv2.imwrite(str(out_dir / 'C_left1920_scaled_focal.png'), out_c)
    cv2.imwrite(str(out_dir / 'D_center1920_scaled_focal.png'), out_d)
    cv2.imwrite(str(out_dir / 'E_original_frame.png'), frame)
    
    print(f"\\n图片已保存到 {out_dir}")
    print(f"A: 左侧1920, img_focal={img_focal}, cx=3320 (原始焦距投6640)")
    print(f"B: 中心1920, img_focal={img_focal}, cx=3320 (原始焦距投6640)")
    print(f"C: 左侧1920, fx={fx_c:.0f}, cx=960 (缩放到1920宽)")
    print(f"D: 中心1920, fx={fx_d:.0f}, cx={cx_d:.0f} (缩放到1920宽)")
    
    # 统计 mask 覆盖范围
    print(f"\\nMask 边界 (1920x1080): x[{mask_min_x},{mask_max_x}], y[{mask_min_y},{mask_max_y}]")
    print(f"MANO j0 3D: {jv[0]}")
    print(f"pred_trans: {t}")

if __name__ == '__main__':
    main()

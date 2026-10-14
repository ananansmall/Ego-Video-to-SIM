"""用 FoundationPose 位姿生成 2D 叠加 GIF（替代旧的 ICP centroids 版本）。

关键差异:
  - 位姿来自 FoundationPose 的 ob_in_cam/{t}.txt（毫米级精度, 已用深度反算验证）
  - 物体轮廓用**面片光栅化**投影, 而非顶点投影+膨胀, 避免轮廓外扩
  - 直接复用 foundationpose_data/cam_K 的逐帧内参, 相机姿态变化已由 VGGT 外参处理

输出 <scene_dir>/foundationpose_out/overlay/: toy_inst1.gif + overlay_report.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_iou_eval import rasterize_mesh  # noqa: E402


def draw_overlay(ob_in_cam, mesh, data_dir, out_dir, obj, stride=1, fps=16,
                 smooth_depth=3):
    """逐帧把 mesh 轮廓 + 历史轨迹尾迹叠回原图, 存为 GIF。"""
    import imageio.v2 as imageio

    rgb_dir = os.path.join(data_dir, 'rgb')
    files = sorted(os.listdir(rgb_dir))
    os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(os.path.join(out_dir, f'{obj}.gif'),
                                fps=fps, loop=0)
    hist_uv = []
    written = 0
    for f in files[::stride]:
        t = int(os.path.splitext(f)[0])
        img = cv2.imread(os.path.join(rgb_dir, f))
        if img is None:
            continue
        canvas = img.copy()
        H, W = canvas.shape[:2]
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        P = np.loadtxt(os.path.join(data_dir, '..', 'foundationpose_out',
                                    'ob_in_cam', f'{t:06d}.txt'))

        # --- 物体轮廓（面片级) ---
        pm = rasterize_mesh(mesh, P, K, H, W)
        cnts, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(canvas, [c], -1, (0, 255, 255), 1, cv2.LINE_AA)
            overlay = canvas.copy()
            canvas[pm] = (0.62 * canvas[pm] +
                          0.38 * np.array([(0, 200, 255)])).astype(np.uint8)

        # --- 当前帧中心 ---
        Pc = P[:3, 3]
        uvc = Pc[:2] / Pc[2] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        cx, cy = int(uvc[0]), int(uvc[1])

        # --- 尾迹（历史帧中心投影到当前帧）---
        if hist_uv:
            pts = np.array(hist_uv, np.int32)
            cv2.polylines(canvas, [pts], False, (0, 140, 255), 2, cv2.LINE_AA)
        hist_uv.append((cx, cy))
        if len(hist_uv) > 60:
            hist_uv.pop(0)

        cv2.circle(canvas, (cx, cy), 5, (255, 0, 255), -1)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, obj, (cx + 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, obj, (cx + 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        written += 1
    writer.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sd = args.scene_dir
    data_dir = os.path.join(sd, 'foundationpose_data')
    out_dir = args.out_dir or os.path.join(sd, 'foundationpose_out', 'overlay')

    mesh = trimesh.load(os.path.join(data_dir, 'mesh.obj'))
    n = draw_overlay(None, mesh, data_dir, out_dir, args.obj, stride=args.stride)

    # ---- 报告: 位姿统计 + 世界系轨迹 ----
    fp_w = os.path.join(sd, 'foundationpose_out', 'ob_in_world')
    traj = {t: np.loadtxt(os.path.join(fp_w, f'{t:06d}.txt'))
            for t in range(len(os.listdir(fp_w)))}
    ks = sorted(traj)
    c0 = traj[ks[0]][:3, 3]
    disp = [float(np.linalg.norm(traj[t][:3, 3] - c0)) for t in ks]
    rep = {'obj': args.obj, 'frames': n,
           'world_start': c0.tolist(),
           'world_end': traj[ks[-1]][:3, 3].tolist(),
           'net_displacement': float(np.linalg.norm(traj[ks[-1]][:3, 3] - c0)),
           'max_displacement': float(np.max(disp))}
    with open(os.path.join(out_dir, 'overlay_report.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print(f'GIF: {out_dir}/{args.obj}.gif ({n} frames)')
    print(f'世界系净位移: {rep["net_displacement"]:.4f}m  最大位移: {rep["max_displacement"]:.4f}m')


if __name__ == '__main__':
    main()
"""
物体轨迹可视化: 投影回原视频 + 3D 轨迹图 + 首尾位姿。

坐标系约定
----------
- **世界坐标系**: VGGT-Omega 的 world 系, Z-up, 与 `extrinsics/`、`centroids` 一致。
  物体轨迹、3D 轨迹图都用这个系, 不做任何变换。
- **相机系**: `extrinsics/{i}.txt` 是 **world-to-camera** (p_cam = R @ p_world + t)。
  投影回图像时用 p_cam, z_cam 即深度。
- **内参**: 逐帧 FoV 预测, 与落盘 `intrinsic.txt`(各帧均值) 有细微差异,
  这里按 world_points + 外参反推该帧真实内参, 与 vggt-omega 反投影口径一致。

三件产出
--------
1. `<out>/overlay/<obj>.gif` 物体位姿叠加回原视频帧
   - 轨迹线(历史尾迹, 由暗到亮)
   - 当前帧质心投影点 + 名称
   - 首帧/末帧标记
   - 右上角信息栏: 首尾世界坐标、净位移、帧号
2. `<out>/traj3d.png` matplotlib 3D 轨迹图(世界坐标系)
   - 主图: 3D 轨迹线 + 首尾标记 + 起点球
   - 副图: X/Y/Z 随时间变化曲线
3. `trajectory_report.json` 首尾位姿与逐帧数据

用法
----
    python trajectory_video_overlay.py --obj toy_inst1
    python trajectory_video_overlay.py# 全部
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

TRAJ_DIR_SUFFIX = 'object_trajectories'


def load_trajectories(traj_dir):
    out = {}
    for fname in sorted(os.listdir(traj_dir)):
        if not fname.endswith('_traj.npz'):
            continue
        key = fname[:-len('_traj.npz')]
        d = np.load(os.path.join(traj_dir, fname), allow_pickle=True)
        out[key] = {
            'trajectory': d['trajectory'],
            'centroids': d['centroids'],
            'valid_frames': d['valid_frames'],
            'trusted': bool(d['trusted']) if 'trusted' in d else True,
        }
    return out


def load_dynamic_keys(scene_dir, world_points=None, all_masks=None,
                      info_rel='dynamic_pointclouds/trajectory_info.json'):
    """读 stage3(mainv2) 的动态物体判定结果, 返回对应的轨迹实例名集合。

    注意编号不同: stage3 输出在去重**后**的 all_instances 上(去重重编号),
    而轨迹基于 `_pre_dedup_masks.pkl`(去重**前**)。两者对不上, 所以这里
    用首帧质心距离做一次映射。
    """
    info_path = os.path.join(scene_dir, info_rel)
    if not os.path.isfile(info_path):
        return set()
    with open(info_path) as f:
        info = json.load(f)
    dyn = {k: v for k, v in info.items()
           if isinstance(v, dict) and v.get('is_dynamic')}
    if not dyn:
        return set()
    ref = {k: np.array(v['centroids'])[0] for k, v in dyn.items()}
    n_ref = len(next(iter(dyn.values()))['visible_frames'])

    if world_points is None or all_masks is None:
        pkl_path = os.path.join(scene_dir, '_pre_dedup_masks.pkl')
        with open(pkl_path, 'rb') as f:
            pkl = pickle.load(f)
        world_points = np.asarray(pkl['world_points'], dtype=np.float64)
        all_masks = pkl['all_masks']

    matched = set()
    for cat, insts in all_masks.items():
        for i, inst in enumerate(insts):
            key = f'{cat}_inst{i}'
            if len(inst) != n_ref:
                continue
            m0 = np.asarray(inst[0]['mask']) > 0
            if m0.sum() < 20:
                continue
            c0 = world_points[inst[0]['frame_id']][m0].mean(0)
            best = min(ref, key=lambda k: np.linalg.norm(ref[k] - c0))
            if np.linalg.norm(ref[best] - c0) < 0.1:
                matched.add(key)
                print(f"  stage3 '{best}' -> 轨迹 '{key}' "
                      f"(首帧质心差 {np.linalg.norm(ref[best] - c0) * 1000:.1f} mm)")
    return matched


def recover_intrinsic(world_points_frame, R, t, fallback):
    """由该帧 world_points 与其像素网格反推内参 (u = fx*X/Z + cx)。"""
    H, W = world_points_frame.shape[:2]
    cam = world_points_frame @ R.T + t
    z = cam[..., 2]
    m = z > 1e-06
    if m.sum() < 100:
        return fallback
    iy, ix = np.mgrid[0:H, 0:W]
    fx, cx = np.linalg.lstsq(
        np.stack([cam[..., 0][m] / z[m], np.ones(m.sum())], 1),
        ix[m].astype(np.float64), rcond=None)[0]
    fy, cy = np.linalg.lstsq(
        np.stack([cam[..., 1][m] / z[m], np.ones(m.sum())], 1),
        iy[m].astype(np.float64), rcond=None)[0]
    if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        return fallback
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


class Projector:
    """world → 像素。按帧缓存内外参。"""

    def __init__(self, scene_dir, world_points):
        self.scene_dir = scene_dir
        self.world_points = world_points
        H, W = world_points.shape[1:3]
        self.K_fallback = np.array([[286.0, 0, W / 2], [0, 286.0, H / 2], [0, 0, 1.0]])
        self._cache = {}

    def at(self, fid):
        if fid in self._cache:
            return self._cache[fid]
        ext = np.loadtxt(os.path.join(self.scene_dir, 'extrinsics', f'{fid}.txt'))
        R, t = ext[:3, :3], ext[:3, 3]
        K = recover_intrinsic(self.world_points[fid], R, t, self.K_fallback)
        self._cache[fid] = (R, t, K)
        return R, t, K

    def project(self, pts_world, fid):
        R, t, K = self.at(fid)
        cam = pts_world @ R.T + t
        z = cam[:, 2]
        ok = z > 1e-06
        uv = np.full((len(cam), 2), np.nan)
        uv[ok, 0] = cam[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
        uv[ok, 1] = cam[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
        return uv, ok


def load_frames(color_dir, n_frames):
    files = sorted(os.listdir(color_dir),
                   key=lambda x: int(os.path.splitext(x)[0]))
    idx = np.linspace(0, len(files) - 1, min(n_frames, len(files))).astype(int)
    frames = []
    for i in idx:
        img = cv2.imread(os.path.join(color_dir, files[i]))
        if img is None:
            continue
        frames.append((int(i), img))
    return frames


def draw_overlay_gif(traj_key, traj, projector, frames, out_path, trail_len=None,
                     fps=16, mesh_pts=None):
    """把物体轨迹叠加回原视频帧, 生成 GIF。

    mesh_pts: (M,3) 物体表面点(物体局部系)。给定时会在每帧画出物体投影轮廓,
               这样"位姿"是真实可见的, 而不只是一个质心点。
    trail_len: 尾迹帧数, None = 画完整轨迹。
    """
    import imageio.v2 as imageio

    valid = traj['valid_frames']
    centroids = traj['centroids']
    rotmats = traj['trajectory'][:, :3, :3]
    vidx = np.where(valid)[0]
    if len(vidx) < 2:
        print(f'  skip {traj_key}: only {len(vidx)} valid frames')
        return None

    first_f, last_f = int(vidx[0]), int(vidx[-1])
    start_w = centroids[first_f].copy()
    end_w = centroids[last_f].copy()
    net_disp = float(np.linalg.norm(end_w - start_w))

    uv_start, _ = projector.project(start_w.reshape(1, 3), first_f)
    uv_end, _ = projector.project(end_w.reshape(1, 3), last_f)

    full_uv = {}
    for f in vidx:
        uv, ok = projector.project(centroids[f].reshape(1, 3), int(f))
        full_uv[int(f)] = uv[0] if ok[0] else None

    out_frames = []
    for fid, img in frames:
        canvas = img.copy()
        H, W = canvas.shape[:2]

        if mesh_pts is not None and valid[fid]:
            world = centroids[fid] + mesh_pts @ rotmats[fid].T
            uv, ok = projector.project(world, int(fid))
            uv = uv[ok]
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)]
            if len(uv) > 30:
                layer = np.zeros((H, W, 3), np.uint8)
                layer[uv[:, 1].astype(int), uv[:, 0].astype(int)] = (0, 220, 255)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                solid = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kern)
                solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, kern)
                cnts, _ = cv2.findContours(solid[:, :, 2], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    biggest = max(cnts, key=cv2.contourArea)
                    cv2.drawContours(canvas, [biggest], -1, (0, 255, 255), 1, cv2.LINE_AA)
                blend = solid.astype(bool).any(axis=2)
                canvas[blend] = (0.45 * canvas[blend]
                                 + 0.55 * np.array([(0, 220, 255)])).astype(np.uint8)

        hist = [f for f in vidx if f <= fid]
        if trail_len:
            hist = hist[-trail_len:]
        pts = [full_uv[int(f)] for f in hist]
        pts = [p for p in pts if p is not None and 0 <= p[0] < W and 0 <= p[1] < H]
        if len(pts) > 1:
            poly = np.array([(int(p[0]), int(p[1])) for p in pts], dtype=np.int32)
            cv2.polylines(canvas, [poly], False, (0, 140, 255), 2, cv2.LINE_AA)

        future = [full_uv[int(f)] for f in vidx if f > fid]
        future = [p for p in future if p is not None and 0 <= p[0] < W and 0 <= p[1] < H]
        if len(future) > 1:
            poly = np.array([(int(p[0]), int(p[1])) for p in future], dtype=np.int32)
            cv2.polylines(canvas, [poly], False, (170, 170, 170), 1, cv2.LINE_AA)

        if valid[fid]:
            uv, ok = projector.project(centroids[fid].reshape(1, 3), int(fid))
            if ok[0]:
                cx, cy = int(uv[(0, 0)]), int(uv[(0, 1)])
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(canvas, (cx, cy), 5, (255, 0, 255), -1)
                    cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(canvas, traj_key, (cx + 9, cy - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, traj_key, (cx + 9, cy - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        for uv_pt, color, label in [(uv_start, (0, 255, 0), 'START'),
                                    (uv_end, (0, 140, 255), 'END')]:
            if np.isnan(uv_pt[(0, 0)]):
                continue
            px, py = int(uv_pt[(0, 0)]), int(uv_pt[(0, 1)])
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(canvas, (px, py), 9, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (px + 11, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, (px + 11, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

        lines = [
            f'{traj_key}  frame {fid}',
            f'start {np.round(start_w, 3)}',
            f'end   {np.round(end_w, 3)}',
            f'displacement {net_disp:.3f} m',
        ]
        cv2.rectangle(canvas, (0, 0), (265, len(lines) * 15 + 8), (255, 255, 255), -1)
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (7, 19 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        out_frames.append(canvas[:, :, ::-1])

    imageio.mimsave(out_path, out_frames, duration=1.0 / fps, loop=0)
    return {
        'gif': out_path,
        'n_frames': len(out_frames),
        'start_world': start_w.tolist(),
        'end_world': end_w.tolist(),
        'net_displacement': net_disp,
        'first_video_frame': first_f,
        'last_video_frame': last_f,
    }


def plot_traj3d(traj_key, traj, out_png, title_suffix=''):
    """matplotlib 3D 轨迹图 + X/Y/Z 时间曲线。世界坐标系, 不做变换。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = traj['valid_frames']
    centroids = traj['centroids']
    rot = Rotation.from_matrix(traj['trajectory'][:, :3, :3])
    rot_xyz = rot.as_euler('xyz', degrees=True)

    vidx = np.where(valid)[0]
    c = centroids[valid]
    rx, ry, rz = rot_xyz[valid, 0], rot_xyz[valid, 1], rot_xyz[valid, 2]
    f = vidx
    tsec = f / 2.0

    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    n = len(c)
    cmap = plt.get_cmap('turbo')
    for i in range(n - 1):
        ax.plot(c[i:i + 2, 0], c[i:i + 2, 1], c[i:i + 2, 2],
                color=cmap(i / max(n - 1, 1)), linewidth=2.4, zorder=2)
    ax.scatter(*c[0], s=150, c='#00c853', edgecolors='k', linewidths=1.4,
               label=f'start  f{f[0]}', depthshade=False, zorder=5)
    ax.scatter(*c[-1], s=170, c='#ff6d00', edgecolors='k', linewidths=1.4,
               label=f'end  f{f[-1]}', depthshade=False, zorder=5)

    for frac in (0.25, 0.5, 0.75):
        i = int(frac * (n - 1))
        ax.scatter(*c[i], s=45, c=cmap(i / max(n - 1, 1)),
                   edgecolors='k', linewidths=0.8, depthshade=False, zorder=4)
        ax.text(c[i, 0], c[i, 1], c[i, 2] + 0.02, f'f{f[i]}',
                fontsize=7.5, color='k')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D trajectory (world frame){title_suffix}', fontsize=11)
    ax.legend(fontsize=8.5, loc='best')
    try:
        ax.set_box_aspect((max(np.ptp(c[:, 0]), 0.1),
                           max(np.ptp(c[:, 1]), 0.1),
                           max(np.ptp(c[:, 2]), 0.1)))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-58)

    disp = np.linalg.norm(c[-1] - c[0])

    for j, (lab, col) in enumerate((('X', '#d32f2f'), ('Y', '#388e3c'), ('Z', '#1976d2'))):
        ax2.plot(tsec, c[:, j], color=col, linewidth=1.9, label=f'{lab}(m)')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('world coordinate (m)')
    ax2.set_title(f'position  |displacement| = {disp:.3f} m', fontsize=11)
    ax2.grid(alpha=0.3, linewidth=0.5)
    ax2.legend(fontsize=8.5)

    ax3.plot(tsec, rx, color='#d32f2f', linewidth=1.9, label='roll X')
    ax3.plot(tsec, ry, color='#388e3c', linewidth=1.9, label='pitch Y')
    ax3.plot(tsec, rz, color='#1976d2', linewidth=1.9, label='yaw Z')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('rotation (deg)')
    ax3.set_title('rotation (euler xyz)', fontsize=11)
    ax3.grid(alpha=0.3, linewidth=0.5)
    ax3.legend(fontsize=8.5)

    fig.suptitle(traj_key, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=135)
    plt.close(fig)
    return out_png


def sample_object_surface_points(obj_key, world_points, all_masks, n_points=0,
                                 seed=0):
    """取该实例首帧 mask 内的世界点作为物体表面点(物体局部系)。

    用 mask 而非 URDF mesh 的原因: 轨迹本身就是从 mask + world_points 估计的,
    这样取点与轨迹严格同源。实测投影命中率: mask 原点 100%, URDF mesh 重定位后
    仅 73%(重建 mesh 与真实分割有形状差异), 所以优先用 mask。
    n_points=0 表示不抽样, 用完整点云。
    """
    cat, inst = obj_key.rsplit('_inst', 1)
    inst = int(inst)
    if cat not in all_masks or inst >= len(all_masks[cat]):
        return None
    frames = all_masks[cat][inst]
    if not frames:
        return None
    m0 = np.asarray(frames[0]['mask']) > 0
    if m0.sum() < 50:
        return None
    pts = world_points[frames[0]['frame_id']][m0]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 50:
        return None
    origin = pts.mean(axis=0)
    if n_points and len(pts) > n_points:
        rng = np.random.RandomState(seed)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts - origin


def main():
    parser = argparse.ArgumentParser(
        description='物体轨迹: 投影回原视频 GIF + matplotlib 3D 轨迹图')
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    parser.add_argument('--traj_dir', type=str, default=None)
    parser.add_argument('--obj', type=str, default=None,
                        help='只处理指定轨迹 (如 toy_inst1)')
    parser.add_argument('--all_objects', action='store_true',
                        help='处理全部物体 (默认只处理 stage3 判定为动态的)')
    parser.add_argument('--dynamic_info', type=str,
                        default='dynamic_pointclouds/trajectory_info.json',
                        help='stage3 动态判定文件(相对 scene_dir)')
    parser.add_argument('--n_frames', type=int, default=60,
                        help='GIF 帧数(从原视频均匀采样)')
    parser.add_argument('--trail_len', type=int, default=0,
                        help='尾迹帧数, 0=画完整轨迹(默认)')
    parser.add_argument('--fps', type=float, default=16.0)
    parser.add_argument('--n_surface_points', type=int, default=0,
                        help='物体表面采样点数, 0=用完整 mask 点云(默认)')
    parser.add_argument('--no_mesh', action='store_true',
                        help='不画物体投影轮廓, 只画质心与尾迹')
    parser.add_argument('--min_frames', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    traj_dir = args.traj_dir or os.path.join(scene_dir, TRAJ_DIR_SUFFIX)
    out_dir = args.out_dir or os.path.join(scene_dir, 'trajectory_overlay')
    os.makedirs(os.path.join(out_dir, 'overlay'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'traj3d'), exist_ok=True)

    pkl_path = os.path.join(scene_dir, '_pre_dedup_masks.pkl')
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f'{pkl_path} (需先运行 mainv2.py)')
    print(f'loading {pkl_path} ...')
    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
    world_points = np.asarray(pkl['world_points'], dtype=np.float64)
    all_masks = pkl['all_masks']
    print(f'world_points: {world_points.shape}')

    trajs = load_trajectories(traj_dir)
    if args.obj:
        trajs = {k: v for k, v in trajs.items() if k == args.obj}
        if not trajs:
            raise SystemExit(f'no trajectory for --obj {args.obj}')
    elif not args.all_objects:
        dyn = load_dynamic_keys(scene_dir, world_points, all_masks,
                                info_rel=args.dynamic_info)
        if dyn:
            before = len(trajs)
            trajs = {k: v for k, v in trajs.items() if k in dyn}
            print(f'stage3 dynamic objects: {sorted(dyn)}')
            print(f'  {before} -> {len(trajs)} trajectories')
            if not trajs:
                raise SystemExit('no dynamic trajectory matched')
        else:
            print(f'WARNING: {args.dynamic_info} 无动态物体, 输出全部')
    print(f'trajectories: {list(trajs.keys())}')

    projector = Projector(scene_dir, world_points)
    color_dir = os.path.join(scene_dir, 'color')
    frames = load_frames(color_dir, args.n_frames)
    print(f'video frames: {len(frames)} (from {color_dir})')

    report = {}
    for key, traj in sorted(trajs.items()):
        v = np.where(traj['valid_frames'])[0]
        if len(v) < args.min_frames:
            print(f'\nskip {key}: only {len(v)} valid frames')
            continue
        print(f'\n[{key}] {len(v)} valid frames')
        mesh_pts = None
        if not args.no_mesh:
            mesh_pts = sample_object_surface_points(
                key, world_points, all_masks,
                n_points=args.n_surface_points)
            if mesh_pts is None:
                print(f'  no surface points for {key},质心点 only')
            else:
                print(f'  surface points: {len(mesh_pts)}')
        gif_path = os.path.join(out_dir, 'overlay', f'{key}.gif')
        info = draw_overlay_gif(key, traj, projector, frames, gif_path,
                                trail_len=args.trail_len, fps=args.fps,
                                mesh_pts=mesh_pts)
        if info is None:
            continue
        png_path = plot_traj3d(key, traj,
                               os.path.join(out_dir, 'traj3d', f'{key}.png'))
        info['traj3d_png'] = png_path
        rot0 = Rotation.from_matrix(traj['trajectory'][v[0]][:3, :3]).as_euler(
            'xyz', degrees=True)
        rot1 = Rotation.from_matrix(traj['trajectory'][v[-1]][:3, :3]).as_euler(
            'xyz', degrees=True)
        info.update({
            'start_euler_xyz_deg': rot0.tolist(),
            'end_euler_xyz_deg': rot1.tolist(),
            'path_length': float(np.linalg.norm(
                np.diff(traj['centroids'][v], axis=0), axis=1).sum()),
            'max_step': float(np.linalg.norm(
                np.diff(traj['centroids'][v], axis=0), axis=1).max()),
            'trusted': traj['trusted'],
        })
        report[key] = info

        print(f"  start world = {np.round(info['start_world'], 4)}  "
              f'euler={np.round(rot0, 1)}')
        print(f"  end   world = {np.round(info['end_world'], 4)}  "
              f'euler={np.round(rot1, 1)}')
        print(f"  displacement = {info['net_displacement']:.3f} m, path = "
              f"{info['path_length']:.3f} m, max_step = {info['max_step']:.3f} m")
        print(f"  GIF : {info['gif']} ({os.path.getsize(info['gif'])} bytes)")
        print(f'  3D  : {png_path}')

    rp = os.path.join(out_dir, 'trajectory_report.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\nReport: {rp}')
    print(f'Done. {len(report)} object(s) in {out_dir}')


if __name__ == '__main__':
    main()

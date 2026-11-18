#!/usr/bin/env python3
"""test/ee11_mesh_wrist.py — ep11 烧瓶 mesh: 腕锚定深度带过滤。

手握烧瓶时, 瓶体在**腕深度 ±10cm** 内 (tracker 相机系可直接算) —— 比 mask
百分位带更准 (百分位会被台面/背景拖偏)。过滤后 SAM3D 重建 + 几何回退对齐。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

EP = '/path/to/project_data/dataset/EgoExplorer/dataset/lerobot/episode_meta/episode_000011'
EF = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee11_win'
VID = '/path/to/workspace/ee11_mocapwin.mp4'

import torch, imageio.v2 as imageio, json
d = np.load(EP + '/smplx/smplx_episode.npz')
tr_p, tr_r = d['tracker_position'], d['tracker_rotation']
N = len(tr_p)

# 深度锚帧 + K
ef = torch.load(EF + '/inference_output.pt', map_location='cpu', weights_only=False)
dep = ef['depth'].numpy(); Ka = ef['K'].numpy(); dfr = ef['depth_frame_indices'].numpy()
# EF3R 子片段帧 = episode 视频帧 (子片段从 0s 切) ✓ 直接对应

# 目标帧: 抓取稳定点 (mocap 速度骤降 ~7.8s → F234)
tgt = 234
ai = int(np.argmin(np.abs(dfr - tgt)))
fa = int(dfr[ai])
cam_p = tr_p[fa, 0]; cam_R = tr_r[fa, 0]          # 头部相机 mocap 位姿
wrist_m = tr_p[fa, 2]                              # 右腕 mocap 位置
# 腕 → 相机系
wrist_c = cam_R.T @ (wrist_m - cam_p)
print(f"帧 F{fa}: 腕(相机系) {np.round(wrist_c,3).tolist()} | 深度 {wrist_c[2]:.2f}m")

# RGB 帧 + K
r = imageio.get_reader(VID)
img = None
for i, fr in enumerate(r):
    if i == fa:
        img = fr; break
r.close()
H, W = img.shape[:2]
import json as _json
# K 缩放到视频分辨率: EF3R K 是处理分辨率 (512 宽) — 用主点比推
Kd = Ka[ai]
scx = W / (2 * Kd[0, 2]); scy = H / (2 * Kd[1, 2])
Ks = Kd.copy(); Ks[0] *= scx; Ks[1] *= scy
print(f"K 缩放: {scx:.2f}x{scy:.2f}")

# SAM3 文本 mask (flask, F{fa}) — 复用 run_sam3_video 流程
import subprocess
work = EF + '/sam3d_wrist'
os.makedirs(work + '/masks', exist_ok=True)
fpng = os.path.join(work, f'frame_{fa:06d}.png')
imageio.imwrite(fpng, img)
SAM3 = '/path/to/project_data/mano_ws/do-as-i-do/reconstruction/scripts'
CKPT = '/path/to/project_data/mano_ws/do-as-i-do/reconstruction/weights/sam3/sam3.pt'
cmd = ("source /path/to/user_home/miniconda3/etc/profile.d/conda.sh && conda activate sam3 && "
       "unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && cd " + SAM3 + " && "
       f"CUDA_VISIBLE_DEVICES=3 python run_sam3_video.py --video {work} "
       f"--frame_idx {fa} --text 'flask' --obj_id object "
       f"--checkpoint {CKPT} --output_dir {work}/sam3_seg")
# run_sam3_video 的 --video 是目录时按文件名序; 单帧目录 ✓
subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
mp = os.path.join(work, 'sam3_seg', 'masks', f'frame_{fa:06d}_masks', 'object.png')
m = imageio.imread(mp) > 0
print(f"SAM3 mask: {int(m.sum())} px")

# 腕锚定深度带: mask 内深度 ∈ [wrist_z - 0.08, wrist_z + 0.10]
z = dep[ai]
vs, us = np.mgrid[0:z.shape[0], 0:z.shape[1]]
mm = np.isfinite(z) & (z > 0.1)
zz = z[mm]
pc = np.stack([(us[mm] - Kd[0, 2]) / Kd[0, 0] * zz,
               (vs[mm] - Kd[1, 2]) / Kd[1, 1] * zz, zz], 1)
pc_w = pc @ cam_R.T + cam_p                        # 相机系 → mocap 系
d_w = np.linalg.norm(pc_w - wrist_m, axis=1)       # 到腕距离
sel = np.zeros(len(pc_w), bool)
sel[(d_w < 0.12)] = True
pts = pc_w[sel]                                    # 烧瓶区域点 (mocap 系)
print(f"腕 12cm 球内点: {len(pts)} | 范围 {np.round(np.ptp(pts,axis=0)*1000,0).tolist()}mm")

# mask 过滤: 投影 pts 回图像 (y-up→sim z-up 无关, 直接相机系投影)
Kf = Kd.copy()
Kf[0] *= scx; Kf[1] *= scy
pc_img = pts @ cam_R * 1.0                          # 已在相机系 (pts 相对相机: 转回)
# pts 是 mocap 系 → 回相机系
pc_c = cam_R.T @ (pts.T - cam_p[:, None]) + cam_p[:, None] * 0
pc_c = (pts - cam_p) @ cam_R
ui = (pc_c[:, 0] / pc_c[:, 2] * Kf[0, 0] + Kf[0, 2]).round().astype(int)
vi = (pc_c[:, 1] / pc_c[:, 2] * Kf[1, 1] + Kf[1, 2]).round().astype(int)
ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
m_f = np.zeros((H, W), bool)
m_f[vi[ok], ui[ok]] = True
from scipy.ndimage import binary_closing
m_f = binary_closing(m_f, np.ones((9, 9)))
print(f"腕锚 mask: {int(m_f.sum())} px")

# SAM3D
mdir = os.path.join(work, f'wrist_{fa:06d}_masks')
os.makedirs(mdir, exist_ok=True)
imageio.imwrite(os.path.join(mdir, 'object_0.png'), (m_f > 0).astype(np.uint8) * 255)
SAM3D = '/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules/sam-3d-objects'
CFG = '/path/to/project_data/mano_ws/do-as-i-do/reconstruction/weights/sam3d_shared/hf/checkpoints/pipeline.yaml'
cmd = ("source /path/to/user_home/miniconda3/etc/profile.d/conda.sh && conda activate sam3d && "
       "unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && cd " + SAM3D + " && "
       "export PYTHONPATH=$PWD && "
       f"CUDA_VISIBLE_DEVICES=3 python generate_mesh_sam3d.py "
       f"--image_path {fpng} --masks_dir {mdir} --mask_files object_0.png "
       f"--config {CFG}")
r3 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
mesh_p = os.path.join(mdir, 'object_0', 'object_0.obj')
print("SAM3D:", "OK" if os.path.exists(mesh_p) else "失败 " + r3.stderr[-100:])
if not os.path.exists(mesh_p):
    sys.exit(1)

# 几何回退对齐 (深度空洞): mask 范围 × 腕深度 / fx
import trimesh
tm = trimesh.load(mesh_p, force='mesh', process=False)
V, F = np.asarray(tm.vertices, float), np.asarray(tm.faces, int)
span = np.ptp(V, axis=0)
w_px = _ys.max() - _ys.min() if len(_ys) else 200
Z = max(float(wrist_c[2]), 0.1)
s = float((w_px * Z / Ks[0, 0]) / max(span.max(), 1e-6))
Vw = (V - V.mean(0)) * s + pts.mean(0)              # mocap 系 (y-up)
print(f"mesh: {len(V)} 顶点 | 尺度 {s:.3f} | bbox mm {np.round(np.ptp(Vw,axis=0)*1000,1).tolist()}")

od = dict(np.load(EF + '/object_instance.npz', allow_pickle=True))
o = EF + '/object_instance_sam3d.npz'
np.savez(o, verts=Vw.astype(np.float32), faces=F.astype(np.int64),
         center_pre=Vw.mean(0), center_placed=Vw.mean(0),
         obj_points=pts.astype(np.float32), n_points=len(pts),
         pca_center=Vw.mean(0), pca_axes=np.eye(3),
         pca_ext=np.ptp(Vw - Vw.mean(0), axis=0).astype(float),
         source='sam3d_wrist', sam3d_scale=s, sam3d_frame=fa)
print(f"✓ {o}")

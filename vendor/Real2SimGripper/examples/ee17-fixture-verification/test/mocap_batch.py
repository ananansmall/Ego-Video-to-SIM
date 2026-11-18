#!/usr/bin/env python3
"""test/mocap_batch.py — EgoExplorer mocap 全链批量 (抓取-移动-环境-GIF-episode)。"""
import sys, os, json, subprocess, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np

EE = '/path/to/project_data/dataset/EgoExplorer/dataset/lerobot'
R2S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R2S = os.path.normpath(R2S)
PY = sys.executable


def auto_gpu():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    rows = [ln.split(",") for ln in q.stdout.strip().splitlines() if ln]
    return min(rows, key=lambda r: int(r[1]))[0].strip()



def build_env_collision(EE_DIR, EF_DIR, targets_path=None):
    """场景点云 → 路径附近 1.4m → 排除 pad 0.5m 球域 → 15cm 体素 → 碰撞盒+排斥球。"""
    sc_p = EF_DIR + '/pointcloud/scene_cloud_mocap.npz'
    if not os.path.exists(sc_p):
        return None, None
    sc = np.load(sc_p)
    P = sc['points']
    dm = np.load(targets_path or (EE_DIR + '/ef3r_right_targets.npz'))
    pad = dm['pos']
    Pd, pad_s = P[::8], pad          # 排除区采样 = 全帧 (稀疏采样会漏掉下降段
                                     # 中间的盒, 实测排斥把抓取下降推开 → 抓空)
    near = Pd[(np.min(np.linalg.norm(Pd[:, None, :2] - pad_s[None, :, :2],
                                      axis=2), axis=1) < 1.4)
              & (Pd[:, 2] > 0.05) & (Pd[:, 2] < 1.55)]
    if len(near) < 50:
        return None, None
    # 排除区 = **3D**: pad 轨迹 35cm 球域 (xy+z) —— 仅按 xy 排除时物体正下方
    # 的桌面盒仍在, 手臂下探抓取撞盒 (实测 ep5 抓起✓→✗)
    d_pad = np.min(np.linalg.norm(near[:, None, :3] - pad_s[None, :, :3],
                                  axis=2), axis=1)
    near = near[d_pad > 0.50]     # 0.5m 保护球域: 抓取下降全程无盒
                                  # (0.35m 时排斥推开下降轨迹, 实测 ep5 抓空)
    grid = np.floor(near[:, :3] / 0.15).astype(np.int64)
    occ = np.unique(grid, axis=0)
    centers = (occ + 0.5) * 0.15
    if len(centers) > 120:
        sel = np.random.RandomState(0).choice(len(centers), 120, replace=False)
        centers = centers[sel]
    env_boxes = np.hstack([centers, np.full((len(centers), 3), 0.075)])
    obstacles = np.hstack([centers, np.full((len(centers), 1), 0.12)])
    return env_boxes, obstacles


def run_episode(ep, gpu):
    EE_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee%d_mocap' % ep
    name = 'ee%d_mocap' % ep
    # EF3R 输出目录 (早前批量按任务名生成): 找含 meta.json 的 ee{ep}_* 目录
    _cands = [d for d in glob.glob('/path/to/project_data/mano_ws/EgoFound3R/'
                                   'outputs/ee%d_*' % ep)
              if os.path.exists(d + '/meta.json') and not d.endswith('_mocap')]
    if not _cands:
        return {'ep': ep, 'status': '无 EF3R 输出 (先跑 ef3r 批量)'}
    EF_DIR = _cands[0]
    r = subprocess.run([PY, R2S + '/scripts/ee_mocap_targets.py',
                        '--episode', str(ep)], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(EE_DIR + '/ef3r_right_targets.npz'):
        return {'ep': ep, 'status': '目标生成失败: ' + r.stderr[-120:]}
    if not os.path.exists(EF_DIR + '/object_instance.npz'):
        r2 = subprocess.run([PY, R2S + '/scripts/ef3r_object.py',
                             '--ef3r-dir', EF_DIR, '--hand', 'right'],
                            capture_output=True, text=True)
        if not os.path.exists(EF_DIR + '/object_instance.npz'):
            return {'ep': ep, 'status': '物体契约失败: ' + r2.stderr[-120:]}
    sc = EF_DIR + '/pointcloud/scene_cloud_mocap.npz'
    sc_arg = repr(sc) if os.path.exists(sc) else 'None'
    env_boxes, obstacles = build_env_collision(EE_DIR, EF_DIR)
    code = "\n".join([
        "import sys, os",
        "import numpy as np",
        "from numpy import array",
        "sys.path.insert(0, %r)" % (R2S + '/scripts'),
        "import cb_large_range_sim as cb",
        "cb.main(targets=%r, name=%r," % (EE_DIR + '/ef3r_right_targets.npz', name),
        "        object_npz=%r," % (EF_DIR + '/object_instance.npz'),
        "        scene_cloud=%s," % sc_arg,
        "        object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,",
        "        finger_force_limit=30.0, base_ref_source='body',",
        "        torso_gain=6.0, wrist_gain=3.0,",
        "        depenetration_vel=0.05, base_interp=True,",
        "        legacy_physics=False, phys_only=False,",
        "        obstacles=%r, w_obs=0.05, env_boxes=%r)" % (obstacles, env_boxes),
    ])
    r3 = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                        cwd=R2S + '/scripts')
    if r3.returncode != 0:
        return {'ep': ep, 'status': '仿真失败: ' + r3.stderr[-150:]}
    d = json.load(open(R2S + '/results/large_range/%s_large_range.json' % name))
    o = d.get('object') or {}
    acc = (o.get('acceptance') or {}).get('criteria') or {}
    info = {'ep': ep, 'status': 'ok',
            'travel_m': round(d['base']['travel_m'], 1),
            'grasped': o.get('grasped'), 'lift_mm': round(o.get('lift_mm', 0)),
            'follow_mm': round(d['phys_pos_err_mm']['mean'], 1),
            'accept': '%d/%d' % (sum(1 for v in acc.values() if v), len(acc))}
    subprocess.run([PY, R2S + '/scripts/export_episode.py', name], cwd=R2S,
                   capture_output=True)
    return info




# ============ 新 episode 完整流 (无 EF3R 输出时) ============

def find_video(ep):
    """episode_index → ego 相机视频 (LeRobot meta 映射, 借 agile-optim 的 pyarrow)。"""
    import json
    rows = json.loads(subprocess.run(
        ["/path/to/user_home/miniconda3/envs/agile-optim/bin/python", "-c",
         "import pandas as pd;print(pd.read_parquet('%s/meta/episodes/"
         "chunk-000/file-000.parquet').to_json(orient='records'))" % EE],
        capture_output=True, text=True).stdout)
    # episode_0000NN 的 metadata 写 episode_index = NN-1 (0 基) —— 查视频必须
    # 用 ep-1, 否则拿到**下一个** episode 的视频 (实测 ee1 切片=ep2 萃取画面,
    # off-by-one 实锤; 物体/场景语义错位)
    row = next(r for r in rows if r["episode_index"] == ep - 1)
    fi = row["videos/observation.images.camera_0/file_index"]
    return "%s/videos/observation.images.camera_0/chunk-000/file-%03d.mp4" % (EE, fi)


def _read_parquet(path):
    out = subprocess.run(["/path/to/user_home/miniconda3/envs/agile-optim/bin/python",
                          "-c", "import pandas as pd;print(pd.read_parquet(%r)."
                          "to_json(orient='records'))" % path],
                         capture_output=True, text=True)
    return json.loads(out.stdout)


def grasp_window(ep_dir, fps, N):
    """segments → (t0, t1, 物体英文名): 抓取段前 2s → 放置段尾, ≤35s。"""
    sg = _read_parquet(ep_dir + '/annotations/segments.parquet')
    grasp = next((s for s in sg
                  if any(k in s.get("caption_zh", "") for k in
                         ("抓握", "握着", "拿起", "取出", "拾取", "抓住"))), None)
    if grasp is None:
        return None, None, None
    t0 = max(0.0, float(grasp["start"]) - 2.0)
    t1 = float(grasp["end"])
    for s2 in sg[grasp["segment_index"] + 1: grasp["segment_index"] + 3]:
        t1 = float(s2["end"])
        if any(k in s2.get("caption_zh", "") for k in ("放下", "放置", "放回", "平放")):
            break
    t1 = min(t1 + 2.0, t0 + 35.0, float(sg[-1]["end"]))
    cap = grasp["caption_zh"]
    obj_zh = obj_en = None
    best = 10 ** 9
    for zh, en in [("容量瓶", "volumetric flask"), ("烧瓶", "glass flask"),
                   ("量筒", "graduated cylinder"), ("试剂瓶", "reagent bottle"),
                   ("玻璃瓶", "glass bottle"), ("小瓶", "small bottle"),
                   ("托盘", "tray"), ("纸板", "cardboard"), ("纸箱", "cardboard box"),
                   ("毛巾", "towel"), ("衣服", "clothes"), ("面包", "bread slice"),
                   ("西红柿", "tomato slice"), ("书", "book"), ("壶", "kettle"),
                   ("锅", "pot"), ("盒", "box")]:
        p = cap.find(zh)
        if 0 <= p < best:
            obj_zh, obj_en, best = zh, en, p
    return t0, t1, obj_en


def run_full_flow(ep, gpu):
    """新 episode: 切交互窗 → EF3R 子片段 (物体/场景) → mocap 目标 → 场景对齐 → 仿真。"""
    ep_dir = EE + '/episode_meta/episode_%06d' % ep
    video = find_video(ep)
    mdf = _read_parquet(EE + '/meta/episodes/chunk-000/file-000.parquet')
    fps = 30.0
    t0, t1, obj_en = grasp_window(ep_dir, fps, 10 ** 9)
    if t0 is None:
        return {'ep': ep, 'status': '无抓取段'}
    print("  [%d] 窗 %.1f-%.1fs 物体=%s" % (ep, t0, t1, obj_en), flush=True)
    # 切子片段
    clip = '/path/to/workspace/ee%d_mocapwin.mp4' % ep
    if not os.path.exists(clip):
        import imageio.v2 as imageio
        r = imageio.get_reader(video)
        fps_r = r.get_meta_data().get('fps', 30.0)
        w = imageio.get_writer(clip, fps=fps_r, quality=8)
        for i, img in enumerate(r):
            if i >= t0 * fps_r and i <= t1 * fps_r:
                w.append_data(img)
            if i > t1 * fps_r:
                break
        w.close(); r.close()
    # EF3R 全链在子片段上 (产出物体契约/场景深度/手轨迹)
    name3 = 'ee%d_win' % ep
    env = dict(os.environ, OBJ_TEXT=obj_en or "")
    out3 = '/path/to/project_data/mano_ws/EgoFound3R/outputs/' + name3
    if not os.path.exists(out3 + '/object_instance.npz'):
        r3 = subprocess.run(['bash', R2S + '/scripts/ef3r_dataset.sh', clip, name3,
                             'right', gpu], capture_output=True, text=True,
                            cwd=R2S, env=env)
        if not os.path.exists(out3 + '/object_instance.npz'):
            return {'ep': ep, 'status': 'EF3R 子片段失败: ' + r3.stderr[-130:]}
    # mocap 目标 (episode 级 smplx 切到窗; EF3R 子片段帧 = 窗内时刻一一对应)
    r4 = subprocess.run([PY, R2S + '/scripts/ee_mocap_targets.py',
                         '--episode', str(ep), '--t0', str(t0), '--t1', str(t1)],
                        capture_output=True, text=True)
    mocap_npz = ('/path/to/project_data/mano_ws/EgoFound3R/outputs/ee%d_mocap/'
                 'ef3r_right_targets.npz' % ep)
    if not os.path.exists(mocap_npz):
        return {'ep': ep, 'status': 'mocap 目标失败: ' + r4.stderr[-130:]}
    # 场景对齐: EF3R 子片段手轨迹 ↔ episode mocap 腕轨迹 (同窗) → 场景入 mocap 系
    scene_src = out3 + '/pointcloud/scene_cloud.npz'
    if os.path.exists(scene_src) and not os.path.exists(
            out3 + '/pointcloud/scene_cloud_mocap.npz'):
        dm = np.load(mocap_npz)
        de = np.load(out3 + '/ef3r_right_targets.npz')
        dsm = np.load(EP + '/smplx/smplx_episode.npz')
        n = min(len(de['pos']), len(dm['pos']))
        fw = np.array([0, 0, -0.08])
        pe = de['pos'][:n] + np.einsum("nij,j->ni", de['rot'][:n], fw)
        pm = dm['pos'][:n]
        ce, cm = pe.mean(0), pm.mean(0)
        Hm = (pe - ce).T @ (pm - cm)
        U, S, Dt = np.linalg.svd(Hm)
        dg = np.sign(np.linalg.det(Dt.T @ U.T))
        Ru = Dt.T @ np.diag([1, 1, dg]) @ U.T
        su = float(np.trace(np.diag(S) @ np.diag([1, 1, dg]))) / max(
            float(((pe - ce) ** 2).sum()), 1e-9)
        tu = cm - su * Ru @ ce
        scz = np.load(scene_src)
        Pc = scz['points'] @ Ru.T * su + tu
        Cc = scz.get('colors', np.full((len(Pc), 3), 0.6, np.float32))
        np.savez(out3 + '/pointcloud/scene_cloud_mocap.npz',
                 points=Pc.astype(np.float32), colors=Cc.astype(np.float32),
                 already_sim=np.array([1]))
        print("  [%d] 场景对齐: 尺度 %.2f (EF3R 手噪声均值已含)" % (ep, su), flush=True)
    env_boxes, obstacles = build_env_collision(EE_DIR, EF_DIR)
    # 仿真 (mocap 目标 + 已对齐场景 + 物体)
    code = "\n".join([
        "import sys, os",
        "import numpy as np",
        "from numpy import array",
        "sys.path.insert(0, %r)" % (R2S + '/scripts'),
        "import cb_large_range_sim as cb",
        "cb.main(targets=%r, name=%r," % (mocap_npz, 'ee%d_mocap' % ep),
        "        object_npz=%r," % (out3 + '/object_instance.npz'),
        "        scene_cloud=%r," % (out3 + '/pointcloud/scene_cloud_mocap.npz'),
        "        object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,",
        "        finger_force_limit=30.0, base_ref_source='body',",
        "        torso_gain=6.0, wrist_gain=3.0,",
        "        depenetration_vel=0.05, base_interp=True,",
        "        legacy_physics=False, phys_only=False,",
        "        obstacles=%r, w_obs=0.05, env_boxes=%r)" % (obstacles, env_boxes),
    ])
    r5 = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                        cwd=R2S + '/scripts')
    if r5.returncode != 0:
        return {'ep': ep, 'status': '仿真失败: ' + r5.stderr[-150:]}
    d = json.load(open(R2S + '/results/large_range/ee%d_mocap_large_range.json' % ep))
    o = d.get('object') or {}
    acc = (o.get('acceptance') or {}).get('criteria') or {}
    subprocess.run([PY, R2S + '/scripts/export_episode.py', 'ee%d_mocap' % ep],
                   cwd=R2S, capture_output=True)
    return {'ep': ep, 'status': 'ok',
            'travel_m': round(d['base']['travel_m'], 1),
            'grasped': o.get('grasped'), 'lift_mm': round(o.get('lift_mm', 0)),
            'follow_mm': round(d['phys_pos_err_mm']['mean'], 1),
            'accept': '%d/%d' % (sum(1 for v in acc.values() if v), len(acc))}


if __name__ == '__main__':
    eps = [int(x) for x in sys.argv[1:]] or [1, 8, 10, 12, 13, 11]
    gpu = auto_gpu()
    print("自动选卡: GPU %s | episodes: %s" % (gpu, eps), flush=True)
    for ep in eps:
        try:
            _has_ef3r = any(os.path.exists(d + '/meta.json') and not d.endswith('_mocap')
                            for d in glob.glob('/path/to/project_data/mano_ws/'
                                               'EgoFound3R/outputs/ee%d_*' % ep))
            r = run_full_flow(ep, gpu) if not _has_ef3r else run_episode(ep, gpu)
        except Exception as e:
            r = {'ep': ep, 'status': '异常: ' + str(e)[:120]}
        if r.get('status') == 'ok':
            print("  ep%d: 抓起=%s 抬升 %dmm | 行程 %.1fm | 跟随 %.1fmm | 验收 %s"
                  % (ep, '✓' if r['grasped'] else '✗', r['lift_mm'],
                     r['travel_m'], r['follow_mm'], r['accept']), flush=True)
        else:
            print("  ep%d: %s" % (ep, r['status']), flush=True)


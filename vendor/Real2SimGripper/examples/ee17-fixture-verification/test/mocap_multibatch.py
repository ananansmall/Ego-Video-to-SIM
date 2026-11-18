#!/usr/bin/env python3
"""test/mocap_multibatch.py — 多交互窗口批量: 每个抓取段 = 一条独立数据。

EgoExplorer 的多交互片段 (称量/过柱/开反应) 含 2-5 个抓取段, 每段 = 一次
完整抓取-操作-放置 —— 全部提取为独立数据, 数据量成倍扩展。
物体 mesh/场景 复用首窗产物 (同 episode 同物体)。
"""
import sys, os, json, subprocess, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np

EE = '/path/to/project_data/dataset/EgoExplorer/dataset/lerobot'
R2S = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = sys.executable
GRASP_K = ("抓握", "握着", "拿起", "取出", "拾取", "抓住")
REL_K = ("放下", "放置", "放回", "平放")


def shim_read(path):
    out = subprocess.run(["/path/to/user_home/miniconda3/envs/agile-optim/bin/python", "-c",
                          "import pandas as pd;print(pd.read_parquet(%r)."
                          "to_json(orient='records'))" % path],
                         capture_output=True, text=True)
    return json.loads(out.stdout)


def auto_gpu():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    rows = [ln.split(",") for ln in q.stdout.strip().splitlines() if ln]
    return min(rows, key=lambda r: int(r[1]))[0].strip()


def grasp_segments(ep_dir):
    segs = shim_read(ep_dir + '/annotations/segments.parquet')
    out = []
    for i, s2 in enumerate(segs):
        if any(k in s2.get("caption_zh", "") for k in GRASP_K):
            t0 = max(0.0, float(s2["start"]) - 2.0)
            t1 = float(s2["end"])
            for s3 in segs[i + 1:i + 3]:
                t1 = float(s3["end"])
                if any(k in s3.get("caption_zh", "") for k in REL_K):
                    break
            t1 = min(t1 + 2.0, t0 + 35.0)
            out.append((t0, t1, s2["caption_zh"][:40]))
    return out


def run_window(ep, wi, t0, t1, gpu, obj_npz, scene_npz):
    name = f'ee{ep}_mocap_w{wi}'
    EE_DIR = f'/path/to/project_data/mano_ws/EgoFound3R/outputs/ee{ep}_mocap'
    # per-window 目标目录: 与 base 批量 (共享 EE_DIR/ef3r_right_targets.npz)
    # 并行时互不覆盖
    out_dir = EE_DIR + '/w%d' % wi
    tgt = out_dir + '/ef3r_right_targets.npz'
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([PY, R2S + '/scripts/ee_mocap_targets.py',
                        '--episode', str(ep), '--t0', str(t0), '--t1', str(t1),
                        '--out', out_dir],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tgt):
        return {'wi': wi, 'status': '目标失败: ' + r.stderr[-100:]}
    # 环境碰撞 (v2 升级): 场景云→碰撞盒+排斥球, 与 mocap_batch 同调优参数
    env_boxes = obstacles = None
    try:
        from mocap_batch import build_env_collision
        env_boxes, obstacles = build_env_collision(EE_DIR, os.path.dirname(obj_npz),
                                                   targets_path=tgt)
    except Exception as _e:
        print('  env_boxes 构建失败 (无碰撞继续): %s' % str(_e)[:80], flush=True)
    code = "\n".join([
        "import sys, os",
        "import numpy as np",
        "from numpy import array",
        "sys.path.insert(0, %r)" % (R2S + '/scripts'),
        "import cb_large_range_sim as cb",
        "cb.main(targets=%r, name=%r," % (tgt, name),
        "        object_npz=%r," % obj_npz,
        "        scene_cloud=%r," % (scene_npz if scene_npz else 'None'),
        "        object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,",
        "        finger_force_limit=30.0, base_ref_source='body',",
        "        torso_gain=6.0, wrist_gain=3.0,",
        "        depenetration_vel=0.05, base_interp=True,",
        "        legacy_physics=False, phys_only=True,",
        "        obstacles=%r, w_obs=0.05, env_boxes=%r)" % (obstacles, env_boxes),
    ])
    r3 = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                        cwd=R2S + '/scripts')
    if r3.returncode != 0:
        return {'wi': wi, 'status': '仿真失败: ' + r3.stderr[-120:]}
    d = json.load(open(R2S + '/results/large_range/%s_large_range.json' % name))
    o = d.get('object') or {}
    acc = (o.get('acceptance') or {}).get('criteria') or {}
    subprocess.run([PY, R2S + '/scripts/export_episode.py', name], cwd=R2S,
                   capture_output=True)
    return {'wi': wi, 'status': 'ok',
            'travel_m': round(d['base']['travel_m'], 1),
            'grasped': o.get('grasped'), 'lift_mm': round(o.get('lift_mm', 0)),
            'follow_mm': round(d['phys_pos_err_mm']['mean'], 1),
            'env_n': 0 if env_boxes is None else len(env_boxes),
            'accept': '%d/%d' % (sum(1 for v in acc.values() if v), len(acc))}


if __name__ == '__main__':
    eps = [int(x) for x in sys.argv[1:]] or [12, 13, 15, 2, 17, 11]
    gpu = auto_gpu()
    print("自动选卡: GPU %s" % gpu, flush=True)
    total_ok = total_all = 0
    for ep in eps:
        ep_dir = EE + '/episode_meta/episode_%06d' % ep
        wins = grasp_segments(ep_dir)
        obj_glob = glob.glob('/path/to/project_data/mano_ws/EgoFound3R/'
                             'outputs/ee%d_*/object_instance.npz' % ep)
        obj_npz = obj_glob[0] if obj_glob else None
        sc_glob = glob.glob('/path/to/project_data/mano_ws/EgoFound3R/'
                            'outputs/ee%d_*/pointcloud/scene_cloud_mocap.npz' % ep)
        scene_npz = sc_glob[0] if sc_glob else None
        print("ep%d: %d 个抓取窗 | 物体: %s" % (ep, len(wins), obj_npz or '无'),
              flush=True)
        if not wins or obj_npz is None:
            continue
        for wi, (t0, t1, cap) in enumerate(wins):
            total_all += 1
            r = run_window(ep, wi, t0, t1, gpu, obj_npz, scene_npz)
            if r.get('status') == 'ok':
                total_ok += 1
                print("  w%d [%.0f-%.0fs] %s: 抓起=%s 抬升 %dmm | 跟随 %.1fmm | %s"
                      % (wi, t0, t1, cap[:20], '✓' if r['grasped'] else '✗',
                         r['lift_mm'], r['follow_mm'], r['accept']), flush=True)
            else:
                print("  w%d: %s" % (wi, r['status']), flush=True)
    print("===== 多窗口批量汇总: %d/%d 窗仿真成功 =====" % (total_ok, total_all),
          flush=True)

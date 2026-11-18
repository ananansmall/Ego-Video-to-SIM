#!/usr/bin/env python3
"""agile_queue.py — AGILE 链批量队列 (episode × 抓取窗口)。

每窗口流水 (可断点续跑, 物件齐则跳):
  1. 抓取窗 (multibatch 同款 segments 解析) + targets (复用 multibatch 的 w{wi})
  2. CONTACT = targets grip_states 首个 LIFT 帧 → 10fps → 5 的倍数
     POINT  = 接触帧 mocap 右腕 → 相机系投影 (约定无关: 纯 tracker→cam)
  3. 切窗视频 → bash tools/run_ee_window.sh (子集模式链, 1.5-2h/窗)
  4. agile_bridge_window.py → object_instance_agile.npz + 配准 JSON

用法 (robotwin_curobo env):
    python test/agile_queue.py --eps 12 17 --gpus 3 4 6 [--dry]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

R2S = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, R2S + "/scripts")
sys.path.insert(0, R2S + "/test")
EE = "/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
OUTS = "/path/to/project_data/mano_ws/EgoFound3R/outputs"
AGILE = "/path/to/project_data/mano_ws/AGILE"
PY = sys.executable
FX, FY, CX, CY = 961.5, 1000.6, 960.0, 485.2
# 手动 POINT (腕投影出画/失败的窗口): {(ep, wi): "x,y"} —— 从窗口帧目检选定
MANUAL_POINT = {(12, 0): "1765,295"}       # ep12 w0: 黄色液体量筒 (右上)


def auto_gpu(exclude=()):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    rows = [ln.split(",") for ln in q.stdout.strip().splitlines() if ln]
    for r in sorted(rows, key=lambda r: int(r[1])):
        if r[0].strip() not in exclude and int(r[1]) < 2000:
            return r[0].strip()
    return None


def cut_clip(video, t0, t1, out):
    """切窗视频。imageio 写 mp4 会把奇数高 970 对齐到 976, 破坏与 cam_K 的
    一致性 —— 用 ffmpeg 切 (保留原始分辨率), 失败回退 imageio。"""
    import shutil
    ff = "/path/to/user_home/miniconda3/envs/hawor/bin/ffmpeg"
    if shutil.which(ff) is None:
        ff = "ffmpeg"
    r = subprocess.run([ff, "-y", "-loglevel", "error",
                        "-ss", str(t0), "-i", video, "-t", str(t1 - t0),
                        "-an", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", out], capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
        return
    # 回退: imageio (会 976 对齐, 但至少能用)
    import imageio.v2 as imageio
    rv = imageio.get_reader(video)
    fps = rv.get_meta_data().get("fps", 30.0)
    w = imageio.get_writer(out, fps=fps, quality=8)
    a, b = int(t0 * fps), int(t1 * fps)
    for i, img in enumerate(rv):
        if i < a:
            continue
        if i > b:
            break
        w.append_data(img)
    w.close()
    rv.close()


def prep_window(ep, wi):
    """返回 dict(ep, wi, t0, t1, seq, src, point, contact, out_dir) 或错误串。"""
    from mocap_multibatch import grasp_segments
    ep_dir = f"{EE}/episode_meta/episode_{int(ep):06d}"
    wins = grasp_segments(ep_dir)
    if wi >= len(wins):
        return f"ep{ep} 无 w{wi}"
    t0, t1, cap = wins[wi][0], wins[wi][1], wins[wi][2]
    out_dir = f"{OUTS}/ee{ep}_mocap/w{wi}"
    tgt = out_dir + "/ef3r_right_targets.npz"
    if not os.path.exists(tgt):
        os.makedirs(out_dir, exist_ok=True)
        r = subprocess.run([PY, R2S + "/scripts/ee_mocap_targets.py",
                            "--episode", str(ep), "--t0", str(t0), "--t1", str(t1),
                            "--out", out_dir], capture_output=True, text=True)
        if r.returncode != 0:
            return f"ep{ep} w{wi} 目标失败: {r.stderr[-90:]}"
    tg = np.load(tgt, allow_pickle=True)
    gs = np.asarray(tg["grip_states"], np.int64)
    names = [str(x) for x in np.asarray(tg["grip_state_names"])]
    lift = np.flatnonzero(gs >= names.index("LIFT") if "LIFT" in names else 2)
    if not len(lift):
        return f"ep{ep} w{wi} 无 LIFT 段"
    k_lift = int(lift[0])
    contact = max(0, round(round(k_lift / 3.0) / 5.0) * 5)
    # POINT: 接触帧右腕 → 相机像素 (纯 mocap→cam, 无世界往返)
    d = np.load(f"{EE}/episode_meta/episode_{int(ep):06d}/smplx/smplx_episode.npz",
                allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tr_p = np.asarray(d["tracker_position"], np.float64)
    tr_r = np.asarray(d["tracker_rotation"], np.float64)
    H, W = 970, 1920
    # 多帧 × 双约定扫描: 接触帧 ±0.5s 内首个「z>0 且在画内」的腕投影
    best = None
    for dk in (0, -15, 15, -30, 30):
        kf = k_lift + dk
        if kf < 0 or kf >= len(gs):
            continue
        fa = int(round(t0 * 30)) + kf
        row = int(np.clip(np.searchsorted(vfi, fa), 0, len(vfi) - 1))
        wc0 = tr_r[row, 0].T @ (tr_p[row, 2] - tr_p[row, 0])
        for cand in (wc0, wc0 * np.array([1.0, -1.0, -1.0])):
            if cand[2] <= 0.02:
                continue
            px = FX * cand[0] / cand[2] + CX
            py = FY * cand[1] / cand[2] + CY
            if 5 <= px < W - 5 and 5 <= py < H - 5:
                best = (int(round(px)), int(round(py)))
                break
        if best:
            break
    if best is None:
        if (ep, wi) in MANUAL_POINT:
            best = tuple(int(v) for v in MANUAL_POINT[(ep, wi)].split(","))
            print(f"  ep{ep} w{wi}: 用手动 POINT {best}", flush=True)
        else:
            return f"ep{ep} w{wi} 腕投影无有效候选 (接触帧附近全出画)"
    px, py = best
    return dict(ep=ep, wi=wi, t0=t0, t1=t1, cap=cap,
                seq=f"ee{ep}_w{wi}_agile",
                src=f"/tmp/ee{ep}_w{wi}_agile.mp4",
                point=f"{int(round(px))},{int(round(py))}",
                contact=int(contact), out_dir=out_dir)


def run_window_job(job, gpu):
    ep, wi, seq = job["ep"], job["wi"], job["seq"]
    out_npz = f"{job['out_dir']}/object_instance_agile.npz"
    if os.path.exists(out_npz):
        return f"ep{ep} w{wi}: 已有产物 (跳过)"
    from mocap_multibatch import grasp_segments  # noqa: F401  (确保模块可用)
    obj_glob = sorted(glob.glob(f"{OUTS}/ee{ep}_*/object_instance.npz"))
    if not obj_glob:
        return f"ep{ep} w{wi}: 无先验 object_instance.npz"
    prior_obj = obj_glob[0]
    if not os.path.exists(job["src"]):
        # 源视频: LeRobot episode 映射 (file-0XX)
        import json as _json
        parq = subprocess.run(
            ["/path/to/user_home/miniconda3/envs/agile-optim/bin/python", "-c",
             "import pandas as pd;print(pd.read_parquet('%s/meta/episodes/"
             "chunk-000/file-000.parquet').to_json(orient='records'))" % EE],
            capture_output=True, text=True)
        rows = _json.loads(parq.stdout)
        row = next(r for r in rows if r["episode_index"] == ep - 1)
        fi = row["videos/observation.images.camera_0/file_index"]
        video = f"{EE}/videos/observation.images.camera_0/chunk-000/file-{fi:03d}.mp4"
        cut_clip(video, job["t0"], job["t1"], job["src"])
    env = dict(os.environ, SEQ=seq, SRC=job["src"], POINT=job["point"],
               CONTACT=str(job["contact"]), FX=str(FX), FY=str(FY),
               CX=str(CX), CY=str(CY))
    r = subprocess.run(["bash", f"{AGILE}/tools/run_ee_window.sh", gpu],
                       cwd=AGILE, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).splitlines()
    ok_chain = "DONE" in (r.stdout + r.stderr)
    if not ok_chain:
        return f"ep{ep} w{wi}: 链失败 — " + " | ".join(tail[-2:])
    br = subprocess.run([PY, R2S + "/scripts/agile_bridge_window.py",
                         "--agile-dir", f"{AGILE}/output/{seq}",
                         "--episode", str(ep),
                         "--t0", str(job["t0"]),
                         "--out-dir", job["out_dir"],
                         "--prior-object", prior_obj,
                         "--targets", f"{job['out_dir']}/ef3r_right_targets.npz"],
                        capture_output=True, text=True,
                        env=dict(os.environ,
                                 PYTHONPATH=R2S + "/scripts"))
    if br.returncode != 0:
        return f"ep{ep} w{wi}: 桥接失败 — " + br.stderr[-140:]
    return f"ep{ep} w{wi}: ✓ {out_npz}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[0],
                    help="每集取前 K 个窗 (默认 0=只有 w0)")
    ap.add_argument("--gpus", nargs="+", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for ep in a.eps:
        for wi in range(len(a.windows) and max(a.windows) + 1 or 1):
            j = prep_window(ep, wi)
            if isinstance(j, str):
                print("跳过:", j, flush=True)
            else:
                jobs.append(j)
                print(f"准备: ep{ep} w{wi} [{j['t0']:.0f}-{j['t1']:.0f}s] "
                      f"POINT={j['point']} CONTACT={j['contact']} | {j['cap'][:24]}",
                      flush=True)
    if a.dry:
        print(f"(dry) {len(jobs)} 个窗口", flush=True)
        sys.exit(0)
    gpus = a.gpus or [auto_gpu()]
    print(f"队列: {len(jobs)} 窗 | GPUs {gpus}", flush=True)
    import threading
    queue = list(jobs)
    lock = threading.Lock()
    results = []

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
            try:
                msg = run_window_job(job, gpu)
            except Exception as e:
                msg = f"ep{job['ep']} w{job['wi']}: 异常 {str(e)[:120]}"
            with lock:
                results.append(msg)
            print(f"[GPU {gpu}] {msg}", flush=True)

    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print("===== AGILE 队列完成 =====", flush=True)
    for m in results:
        print(" ", m, flush=True)

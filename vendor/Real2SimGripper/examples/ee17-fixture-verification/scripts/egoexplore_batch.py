#!/usr/bin/env python3
"""egoexplore_batch.py — EgoExplorer 数据集 → Real2SimGripper 物理仿真 episode 批量转换器。

用户定义的转换契约: EgoExplorer 注解给出**被控物体 + 交互时段**
(segments.parquet 的自然语言交互描述, 含起止秒), 其余生成走原管线
(EF3R 快路径 → R1 物理仿真 → episode 打包)。

单 episode 流程:
  1. 读 metadata.json + segments.parquet, 找**抓取段** (caption 含 抓握/握着/拿起/取出…)
  2. 从 caption_zh 提取被控物体 (名词词表 + 前置形容词, 如 "透明短颈玻璃烧瓶"),
     映射英文给 SAM3D 文本提示 (OBJ_TEXT)
  3. 切子片段: [抓取段 start-4s, 后续段直到 出现 放/置/松开 或 +12s 封顶, 全程 ≤40s]
     —— 物体抓取前静置, 抓取-搬运-放置完整覆盖
  4. ef3r_dataset.sh 一键链 (SAM3D 精模 + 物理仿真 + GIF)
  5. export_episode 打包, 汇总物理仿真验收 (抓起/抬升/判据数)

用法 (robotwin_curobo env):
    python scripts/egoexplore_batch.py --episodes 5 13 8 --gpu 0
    python scripts/egoexplore_batch.py --episodes 5 --dry-run      # 只看切窗与物体
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

EE = "/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
R2S = "/path/to/Real2SimGripper"
CLIPS = "/path/to/workspace"
PYV = "/path/to/user_home/miniconda3/envs/robotwin_curobo/bin/python"
PARQUET_PY = "/path/to/user_home/miniconda3/envs/agile-optim/bin/python"  # 有 pyarrow

GRASP_RE = re.compile(r"抓握|握着|拿起|取出|拾取|抓住|拎起|取[出回]")
RELEASE_RE = re.compile(r"放下|放置|放回|松开|放到|搁|平放|放于")
# 物体名词词表 (caption_zh → 英文, 供 SAM3D 文本提示; 未命中则跳过 SAM3D 用凸包)
OBJ_LEX = [
    ("容量瓶", "volumetric flask"), ("烧瓶", "glass flask"), ("量筒", "graduated cylinder"),
    ("试剂瓶", "reagent bottle"), ("玻璃瓶", "glass bottle"), ("小瓶", "small bottle"),
    ("瓶盖", "bottle cap"), ("烧杯", "beaker"), ("试管", "test tube"),
    ("托盘", "tray"), ("纸板", "cardboard"), ("纸箱", "cardboard box"),
    ("毛巾", "towel"), ("衣服", "clothes"), ("面包", "bread slice"),
    ("西红柿", "tomato slice"), ("肉片", "meat slice"), ("书", "book"),
    ("柱状物", "white column"), ("接头", "glass joint"), ("水壶", "kettle"),
    ("壶", "kettle"), ("锅", "pot"), ("盒", "box"),
]


def read_parquet(path):
    """pyarrow 只在 agile-optim env 里有 —— 借它的 python 读。"""
    out = subprocess.run([PARQUET_PY, "-c",
                          f"import pandas as pd; df=pd.read_parquet('{path}'); "
                          f"print(df.to_json(orient='records'))"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"parquet 读取失败: {out.stderr[-200:]}")
    return json.loads(out.stdout)


def episode_video(ep_idx):
    """episode_index → ego 相机视频路径 (LeRobot meta 映射)。"""
    rows = read_parquet(f"{EE}/meta/episodes/chunk-000/file-000.parquet")
    row = next(r for r in rows if r["episode_index"] == ep_idx)
    fi = row["videos/observation.images.camera_0/file_index"]
    v = f"{EE}/videos/observation.images.camera_0/chunk-000/file-{fi:03d}.mp4"
    if not os.path.exists(v):
        raise FileNotFoundError(v)
    return v, row


def grasp_segment(ep_dir):
    """抓取段 + 被控物体 (中文名) + 搬运窗 (秒)。"""
    sg = read_parquet(f"{ep_dir}/annotations/segments.parquet")
    idx = next((i for i, s in enumerate(sg) if GRASP_RE.search(s.get("caption_zh", ""))), None)
    if idx is None:
        return None
    seg = sg[idx]
    cap = seg["caption_zh"]
    # 物体: 词表里最早的命中, 连同前置修饰 (最多回溯 6 字)
    obj_zh, obj_en = None, None
    best_pos = 10**9
    for zh, en in OBJ_LEX:
        p = cap.find(zh)
        if 0 <= p < best_pos:
            head = re.search(rf"([\u4e00-\u9fa5]{{0,6}}{zh})", cap)
            obj_zh, obj_en, best_pos = head.group(1) if head else zh, en, p
    # 交互窗 (用户: 不用那么长, 主要看交互): 抓取前 2s → 第一个"放下"段
    # (或抓取段 +8s 封顶), 全程 ≤20s —— 物体资产只在此窗内建一次 (SAM3D 单帧),
    # 仿真中是刚体, 逐帧只有 6D 位姿, 从不逐帧重建
    t0 = max(0.0, float(seg["start"]) - 2.0)
    t1 = float(seg["end"])
    for s2 in sg[idx + 1: idx + 3]:
        t1 = float(s2["end"])
        if RELEASE_RE.search(s2.get("caption_zh", "")):
            break
    t1 = min(t1 + 2.0, t0 + 35.0, float(sg[-1]["end"]))   # 30s 级: 含完整携带行走
    return {"seg_idx": idx, "t0": t0, "t1": t1, "obj_zh": obj_zh,
            "obj_en": obj_en, "caption": cap[:80]}


def cut_clip(video, t0, t1, out):
    import imageio.v2 as imageio
    r = imageio.get_reader(video)
    fps = r.get_meta_data().get("fps", 30.0)
    w = imageio.get_writer(out, fps=fps, quality=8)
    a, b = int(t0 * fps), int(t1 * fps)
    for i, img in enumerate(r):
        if i < a:
            continue
        if i > b:
            break
        w.append_data(img)
    w.close()
    r.close()
    return b - a + 1


def run_episode(ep_idx, gpu, dry=False, out_root=None):
    ep_dir = f"{EE}/episode_meta/episode_{ep_idx:06d}"
    md = json.load(open(f"{ep_dir}/metadata.json"))
    video, _row = episode_video(ep_idx)
    g = grasp_segment(ep_dir)
    if g is None:
        return {"episode": ep_idx, "task": md.get("task_zh"), "status": "无抓取段"}
    name = f"ee{ep_idx}_{md.get('task_id', 'task')}"
    n_frames = int((g["t1"] - g["t0"]) * _row.get("fps", 30) if "fps" in _row else 0) or None
    info = {"episode": ep_idx, "task": md.get("task_zh"), "obj": g["obj_zh"],
            "obj_en": g["obj_en"], "window": f"{g['t0']:.1f}-{g['t1']:.1f}s",
            "caption": g["caption"]}
    print(f"[ep{ep_idx}] {md.get('task_zh')} | 物体: {g['obj_zh']} ({g['obj_en']}) | "
          f"窗 {g['t0']:.1f}-{g['t1']:.1f}s | {g['caption'][:50]}…")
    if dry:
        info["status"] = "dry-run"
        return info
    clip = f"{CLIPS}/{name}.mp4"
    n = cut_clip(video, g["t0"], g["t1"], clip)
    info["clip_frames"] = n
    env = dict(os.environ, OBJ_TEXT=g["obj_en"] or "")
    cmd = ["bash", "scripts/ef3r_dataset.sh", clip, name, "right", str(gpu)]
    r = subprocess.run(cmd, cwd=R2S, env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode != 0:
        info["status"] = f"管线失败 (exit {r.returncode})"
        info["log_tail"] = log.splitlines()[-3:]
        return info
    acc_p = f"{R2S}/results/large_range/{name}_large_range.json"
    d = json.load(open(acc_p))
    obj = d.get("object") or {}
    acc = obj.get("acceptance") or {}
    crit = acc.get("criteria") or {}
    info.update(status="ok", grasped=obj.get("grasped"),
                lift_mm=obj.get("lift_mm"), travel_m=round(d["base"]["travel_m"], 1),
                accept=f"{sum(1 for v in crit.values() if v)}/{len(crit)}",
                phys_mm=round(d["phys_pos_err_mm"]["mean"], 1))
    # episode 打包
    subprocess.run([PYV, f"{R2S}/scripts/export_episode.py", name,
                    "--out", out_root or f"{R2S}/dataset"], cwd=R2S,
                   capture_output=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--gpu", default="auto",
                    help="'auto' = 自动选显存最空的 GPU (共享集群抢卡常态)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.gpu == "auto":
        import subprocess as _sp
        _q = _sp.run(["nvidia-smi", "--query-gpu=index,memory.used",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True)
        _rows = [ln.split(",") for ln in _q.stdout.strip().splitlines() if ln]
        a.gpu = min(_rows, key=lambda r: int(r[1]))[0].strip()
        print(f"自动选卡: GPU {a.gpu}")
    results = []
    for ep in a.episodes:
        try:
            results.append(run_episode(ep, a.gpu, a.dry_run, a.out))
        except Exception as e:
            results.append({"episode": ep, "status": f"异常: {str(e)[:120]}"})
    print("\n===== 转换汇总 =====")
    for r in results:
        if r.get("status") == "ok":
            print(f"  ep{r['episode']} [{r['task']}] 物体={r['obj']} | "
                  f"抓起={'✓' if r['grasped'] else '✗'} 抬升 {r['lift_mm']:.0f}mm | "
                  f"行程 {r['travel_m']}m | 物理跟随 {r['phys_mm']}mm | 验收 {r['accept']}")
        else:
            print(f"  ep{r['episode']} [{r.get('task','?')}] {r['status']}")


if __name__ == "__main__":
    main()

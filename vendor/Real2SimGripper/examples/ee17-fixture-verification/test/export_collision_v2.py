#!/usr/bin/env python3
"""export_collision_v2.py — 批量重导出碰撞版 dataset (含 collision 键)。

对每个 results/large_range/ee*_mocap*.json 对应窗口:
  - 窗口名 → 找到含 object_instance.npz 的 EF3R 目录 (ee{ep}_* 非 _mocap)
  - export_episode <name> --ef3r-dir <EF_DIR>
  - 校验 dataset/<name>/acceptance.json 含 collision 键
跳过已有 collision 键的 (幂等)。
"""
import glob
import json
import os
import subprocess
import sys

ROOT = "/path/to/Real2SimGripper"
LR = f"{ROOT}/results/large_range"
DS = f"{ROOT}/dataset"
EF3R_OUT = "/path/to/project_data/mano_ws/EgoFound3R/outputs"
PY = "/path/to/user_home/miniconda3/envs/robotwin_curobo/bin/python"


def find_ef_dir(ep):
    cands = [d for d in glob.glob(f"{EF3R_OUT}/ee{ep}_*")
             if os.path.exists(d + "/object_instance.npz")
             and not d.endswith("_mocap")]
    return cands[0] if cands else (cands[0] if cands else None)


def main():
    names = []
    for p in sorted(glob.glob(f"{LR}/ee*_mocap*_large_range.json")):
        name = os.path.basename(p).replace("_large_range.json", "")
        if "_fix_" in name or "_env" in name or "_coll" in name or \
           name.startswith("ee17_posture"):
            continue
        # 只导出碰撞批实际产出的 (源 JSON 带 collision 键)
        try:
            if json.load(open(p)).get("collision") is not None:
                names.append(name)
        except Exception:
            pass
    ok = skip = fail = 0
    for name in names:
        acc = f"{DS}/{name}/acceptance.json"
        # 幂等: 已有 collision 键(非None)则跳过
        if os.path.exists(acc):
            try:
                if json.load(open(acc)).get("collision"):
                    skip += 1
                    continue
            except Exception:
                pass
        ep = int(name.split("_")[0].replace("ee", ""))
        ef = find_ef_dir(ep)
        if ef is None:
            print(f"✗ {name}: 无 EF3R object 目录", flush=True)
            fail += 1
            continue
        r = subprocess.run([PY, f"{ROOT}/scripts/export_episode.py", name,
                            "--ef3r-dir", ef], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"✗ {name}: {r.stderr[-120:]}", flush=True)
            fail += 1
            continue
        # 校验
        d = json.load(open(acc))
        coll = d.get("collision")
        if coll is not None:
            ok += 1
        else:
            fail += 1
            print(f"⚠ {name}: 导出后仍无 collision 键", flush=True)
            continue
        print(f"✓ {name}: env_boxes {coll.get('env_boxes_n')} "
              f"obstacles {coll.get('obstacles_n')}")
    print(f"\n导出: ok {ok} | skip {skip} | fail {fail} | 总 {len(names)}")


if __name__ == "__main__":
    main()

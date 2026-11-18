import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import sys, time, json
sys.path.insert(0, _os.path.dirname(_HERE))
import numpy as np
import robotwin_r1_duck_grasp as G
from task_config import load_task, apply_task
from collision_config import enable_coacd

CFG = sys.argv[1]
cfg = load_task(CFG)
apply_task(G, cfg, verbose=True)                       # ← 全部输入来自 JSON
# 碰撞模式也按 JSON 来
mode = cfg.get("sim",{}).get("collision",{}).get("mode","none")
if mode == "coacd":
    enable_coacd(G, float(cfg["sim"]["collision"].get("threshold",0.05)),
                 max_depenetration_velocity=float(cfg["sim"]["collision"].get("max_depenetration_velocity",1.0)))

t0=time.time()
ep = G.build_episode(on_step=None, collect_frames=False, params={})   # params 空 → 全用 JSON
v,m = ep["verdict"], ep["verdict"]["metrics"]
GT=np.array(cfg["demonstration_ground_truth"]["normalised_offset"])
PLATE=np.array(ep["plate_rb"]); EXT=np.array(ep["plate_ext"])
duck=np.asarray(ep["duck_trace"][-1]); off=(duck[:2]-PLATE[:2])/EXT[:2]
reas=[r for r in v["reasons"] if r!="released_on_table"]
pe=bool(v["pass"]) or (len(reas)==0 and abs(off[0])<=0.5 and abs(off[1])<=0.5)
print("\n=== RESULT (纯 JSON 驱动, params={}) ===")
print("verdict(raw) =", "PASS" if v["pass"] else "FAIL", v["reasons"])
print("pass_eff     =", "PASS" if pe else "FAIL")
print("follow_max   = %.1f mm"%(m.get("follow_max_m",float("nan"))*1000))
print("lift=%.2fcm slip=%.2fcm knock=%.2fcm"%(m["lift_m"]*100,m["hold_slip_m"]*100,m["knock_m"]*100))
print("duck_final   =", np.round(duck,4).tolist())
print("归一化偏移   = [%+.3f, %+.3f]  偏离GT=%.3f"%(off[0],off[1],np.linalg.norm(off-GT)))
print("secs=%d"%int(time.time()-t0))

import sys, time, json
sys.path.insert(0,'/path/to/Real2SimGripper/scripts')
import numpy as np, robotwin_r1_duck_grasp as G
from collision_config import enable_coacd
from task_config import load_task, apply_task
cfg=load_task('../configs/duck_to_plate.json'); apply_task(G,cfg,verbose=False)
enable_coacd(G,None,verbose=False)
GT=np.array([-0.06,0.06]); PLATE=np.array([0.18396,0.22788]); EXT=np.array([0.2504,0.2513])
BASE={'plate_geom':'geometry_8','time_scale':3.0,'grip_open_q':0.05,
      'place_t':0.0,'place_hover':0.07,'ik_seeds':16,'ik_max_iters':300}
# 已知最优点 (0.30,-0.38) gt=0.077; 地形尖锐 -> 密扫 x[0.29,0.35] x y[-0.38,-0.36]
out={}
for bx in [0.29,0.30,0.31,0.32,0.33,0.34,0.35]:
    for by in [-0.38,-0.36]:
        P=dict(BASE); P['base_xy']=[bx,by]; t0=time.time()
        try:
            ep=G.build_episode(on_step=None, collect_frames=False, params=P)
            v,m=ep['verdict'],ep['verdict']['metrics']
            duck=np.asarray(ep['duck_trace'][-1]); off=(duck[:2]-PLATE)/EXT
            out[f"x{bx}_y{by}"]=dict(base_xy=[bx,by],
                verdict="PASS" if v['pass'] else "FAIL", raw=v['reasons'],
                follow_mm=round(float(m.get('follow_max_m',float('nan'))*1000),1),
                off=[round(float(x),3) for x in off],
                gt_dist=round(float(np.linalg.norm(off-GT)),4),
                gt_x_cm=round(float(abs(off[0]-GT[0])*EXT[0]*100),1),
                gt_y_cm=round(float(abs(off[1]-GT[1])*EXT[1]*100),1),
                duck_z=round(float(duck[2]),4),
                lift_cm=round(float(m['lift_m']*100),2), slip_cm=round(float(m['hold_slip_m']*100),2),
                secs=int(time.time()-t0))
        except Exception as e:
            out[f"x{bx}_y{by}"]=dict(base_xy=[bx,by], error=str(e)[:100])
        r=out[f"x{bx}_y{by}"]
        if 'error' in r: print(f"[{bx},{by}] ERROR {r['error'][:60]}", flush=True)
        else: print(f"[{bx},{by}] {r['verdict']} follow={r['follow_mm']} gt={r['gt_dist']} (x{r['gt_x_cm']}/y{r['gt_y_cm']}cm) z={r['duck_z']} {r['secs']}s", flush=True)
json.dump(out, open('/tmp/sweep_dense.json','w'), indent=1)
best=min((v for v in out.values() if 'gt_dist' in v), key=lambda r:r['gt_dist'])
print("\n最优:", best['base_xy'], "gt=%.4f"%best['gt_dist'], "follow=%.1f"%best['follow_mm'], flush=True)
print("DENSE_DONE", flush=True)

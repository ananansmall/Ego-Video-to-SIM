import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import json, numpy as np
c = json.load(open(_CACHE))
PLATE = np.array([0.18396, 0.22788]); EXT = np.array([0.2504, 0.2513])
GT = np.array([-0.06, 0.06])
rows = []
for v in c.values():
    m = v['metrics']; p = v['params']
    df = m.get('duck_final'); pr = m.get('plate_rb')
    if df is None or pr is None: continue
    pe = m.get('pass_eff'); pe = m.get('pass', False) if pe is None else pe
    off = (np.asarray(df, float)[:2] - PLATE) / EXT
    rows.append(dict(tag=v.get('tag'), place_t=p.get('place_t'), ts=p.get('time_scale'),
                     hover=p.get('place_hover'), bx=tuple(p.get('base_xy', [])),
                     pe=bool(pe), follow=m.get('follow_max_mm'),
                     dist=float(np.linalg.norm(off - GT)),
                     y_cm=abs(off[1] - GT[1]) * EXT[1] * 100, off=off))
n = len(rows)
pe_arr = np.array([r['pe'] for r in rows])
d_arr = np.array([r['dist'] for r in rows])
print("样本数 =", n, " (verifier PASS %d / FAIL %d)" % (pe_arr.sum(), (~pe_arr).sum()))

# 1) 物理 verifier 能否预测示范保真度?
def rankdata(a):
    order = np.argsort(a, kind='mergesort'); r = np.empty(len(a)); r[order] = np.arange(len(a)); return r
from scipy.stats import spearmanr
rho, pv = spearmanr(pe_arr.astype(float), d_arr)
print("\n[1] 物理 PASS 与 示范偏离 的 Spearman 相关: rho=%+.3f (p=%.3f)" % (rho, pv))
print("    PASS 组偏离: 中位 %.3f  范围 [%.3f, %.3f]" % (np.median(d_arr[pe_arr]), d_arr[pe_arr].min(), d_arr[pe_arr].max()))
print("    FAIL 组偏离: 中位 %.3f  范围 [%.3f, %.3f]" % (np.median(d_arr[~pe_arr]), d_arr[~pe_arr].min(), d_arr[~pe_arr].max()))
ov = d_arr[~pe_arr].min() < d_arr[pe_arr].max()
print("    => 两组分布重叠: %s" % ("是 —— verifier 无法区分保真度" if ov else "否"))

# 2) 关键 ablation: 按物理 score 选 vs 按 GT 偏离选
import importlib, sys
sys.path.insert(0, _os.path.dirname(_HERE))
import agentic_optimize as AO
cache = json.load(open(_CACHE))
scored = [(AO.score(v['metrics']), v) for v in cache.values() if v.get('score') is not None and v['score'] > -1e8]
scored.sort(key=lambda x: -x[0])
def dist_of(v):
    df = v['metrics'].get('duck_final')
    if df is None: return None
    return float(np.linalg.norm((np.asarray(df)[:2]-PLATE)/EXT - GT))
by_score = max((d for d in [dist_of(v) for _, v in scored[:5]] if d is not None), default=None)
by_gt = min(d for d in [dist_of(v) for v in cache.values()] if d is not None)
top5 = [(round(s,1), v['params'].get('place_t'), v['params'].get('base_xy'), v['params'].get('time_scale')) for s, v in scored[:5]]
print("\n[2] 选型策略对比 (都在 verifier PASS 集合内挑前5)")
print("    按物理 score 前5:", top5)
print("    按物理 score 选  -> 示范偏离 %.3f" % by_score)
print("    按 GT 偏离  选  -> 示范偏离 %.3f" % by_gt)
print("    改善 %.0f%%" % ((1 - by_gt/by_score)*100))

# 3) 最优 GT 配置是什么
best = min(rows, key=lambda r: r['dist'])
print("\n[3] 示范偏离最小的配置:")
print("    place_t=%s ts=%s hover=%s base_xy=%s" % (best['place_t'], best['ts'], best['hover'], best['bx']))
print("    verifier PASS=%s  follow=%.1fmm  偏离GT=%.3f  y向误差=%.1fcm" % (best['pe'], best['follow'], best['dist'], best['y_cm']))
worst = max(rows, key=lambda r: r['dist'])
print("    偏离最大的配置: place_t=%s ts=%s base_xy=%s -> 偏离 %.3f, PASS=%s" % (worst['place_t'], worst['ts'], worst['bx'], worst['dist'], worst['pe']))

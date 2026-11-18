import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import json, numpy as np
c = json.load(open(_CACHE))
PLATE = np.array([0.18396, 0.22788]); EXT = np.array([0.2504, 0.2513])
GT = np.array([-0.045, 0.042])          # 真实视频末段归一化偏移(鸭-盘)/盘尺寸
print('真实视频真值 GT =', GT.tolist(), '  (≈盘心: 人类把鸭子放在了盘子中心)')
print()
rows = []
for k, v in c.items():
    m = v['metrics']; p = v['params']
    df = m.get('duck_final'); pr = m.get('plate_rb')
    if df is None or pr is None:
        continue
    d = np.asarray(df, dtype=float)[:2] - PLATE
    off = d / EXT
    dist = float(np.linalg.norm(off - GT))
    pe = m.get('pass_eff')
    if pe is None:
        pe = m.get('pass', False)
    rows.append((dist, p.get('place_t'), p.get('time_scale'), p.get('place_hover'),
                 bool(pe), off, m.get('follow_max_mm'), v.get('tag')))
rows.sort(key=lambda r: r[0])
print('%-8s %-5s %-6s %-7s | %-17s %-8s %-8s %s' %
      ('place_t', 'ts', 'hover', 'verdict', '仿真归一化偏移', '偏离GT', 'follow', '来源'))
for r in rows:
    print('%-8s %-5s %-6s %-7s | [%+.3f, %+.3f]   %-8.3f %-8s %s' %
          (r[1], r[2], r[3], 'PASS' if r[4] else 'FAIL',
           r[5][0], r[5][1], r[0], ('%.1f' % r[6]) if r[6] else 'n/a', r[7]))
print()
print('=== 关键结论 ===')
ok = [r for r in rows if r[4]]
if ok:
    b = min(ok, key=lambda r: r[0])
    print('  最接近真值且 PASS: place_t=%s ts=%s hover=%s  偏离GT=%.3f follow=%.1fmm'
          % (b[1], b[2], b[3], b[0], b[6]))
else:
    print('  没有任何配置同时满足 PASS + 接近真值')
b = rows[0]
print('  最接近真值的配置: place_t=%s ts=%s  偏离GT=%.3f  follow=%.1fmm  verdict=%s'
      % (b[1], b[2], b[0], b[6], 'PASS' if b[4] else 'FAIL'))

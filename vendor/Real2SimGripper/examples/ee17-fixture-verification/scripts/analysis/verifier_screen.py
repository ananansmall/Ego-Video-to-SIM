"""创新点 2 的最小验证: 哪些物理 verifier 子项对「示范保真度」有分辨力?

DexAgent 自认局限: "Undetected errors propagate through the library" —— 一个写歪的
verifier 进库就永久污染后续任务。我们的改法: 入库前测它与示范保真度的相关性。

这里用已有的 27+ 个评测配置做原型: 对每个 verifier 子项(通过/失败)算它与
「示范偏离 gt_dist」的点二列相关 + AUC(用该子项能否把"偏离大/小"分开)。
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import json, numpy as np
from scipy.stats import pointbiserialr, mannwhitneyu

c = json.load(open(_CACHE))
GT = np.array([-0.06, 0.06])
rows = []
for v in c.values():
    m = v['metrics']
    df, pr = m.get('duck_final'), m.get('plate_rb')
    if df is None or pr is None:
        continue
    off = (np.asarray(df, float)[:2] - np.asarray(pr, float)[:2]) / np.array([0.2504, 0.2513])
    gd = float(np.linalg.norm(off - GT))
    pe = m.get('pass_eff'); pe = m.get('pass', False) if pe is None else pe
    reasons = set(m.get('reasons') or [])
    rows.append(dict(gd=gd, pe=bool(pe), reasons=reasons,
                     follow=m.get('follow_max_mm'), margin=m.get('plate_margin_mm')))
n = len(rows)
print("样本 =", n, " (PASS %d / FAIL %d)" % (sum(r['pe'] for r in rows), sum(not r['pe'] for r in rows)))

# 1) 各 verifier 失败项与「偏离大」的关联
print("\n[1] 每个 verifier 失败项 是否预示「示范偏离大」")
print("    %-22s %-8s %-10s %-10s %s" % ("失败项", "出现次数", "偏离中位", "未出现偏离", "区分度(AUC)"))
gd_all = np.array([r['gd'] for r in rows])
for item in ['follow_integrity', 'released_on_table', 'hold_integrity', 'lift_integrity']:
    hit = np.array([item in r['reasons'] for r in rows])
    if hit.sum() == 0:
        continue
    # AUC: 用 item 出现/不出现 去区分 偏离大/小 (偏离>中位数算"大")
    big = gd_all > np.median(gd_all)
    try:
        u = mannwhitneyu(gd_all[hit], gd_all[~hit], alternative='greater')
        auc = u.statistic / (hit.sum() * (~hit).sum())
        p = u.pvalue
    except Exception:
        auc, p = float('nan'), float('nan')
    print("    %-22s %-8d %-10.3f %-10.3f %.3f (p=%.3f)" % (
        item, int(hit.sum()), np.median(gd_all[hit]), np.median(gd_all[~hit]), auc, p))

# 2) 整体 PASS/FAIL 的区分度
big = gd_all > np.median(gd_all)
pe = np.array([r['pe'] for r in rows])
try:
    u = mannwhitneyu(gd_all[~pe], gd_all[pe], alternative='greater')
    auc = u.statistic / ((~pe).sum() * pe.sum())
    print("\n[2] 整体 verifier PASS/FAIL 区分「偏离大/小」的 AUC = %.3f (p=%.3f)" % (auc, u.pvalue))
except Exception as e:
    print("\n[2] 算不出:", e)
print("    (AUC=0.5 表示完全没分辨力, =1.0 表示完美区分)")

# 3) 连续量的相关性: follow_max / margin 与 gt_dist
print("\n[3] 连续物理量与示范偏离的相关性 (Spearman)")
from scipy.stats import spearmanr
for k in ['follow', 'margin']:
    a = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
    g = np.array([r['gd'] for r in rows if r[k] is not None])
    if len(a) > 3:
        rho, p = spearmanr(a, g)
        print("    %-10s n=%2d  rho=%+.3f (p=%.3f)" % (k, len(a), rho, p))

# 4) 结论: 应该把哪些 verifier 降权
print("\n[4] 创新点 2 的判定")
print("    - AUC 接近 0.5 的失败项 => 与示范保真度无关, 入库时应降权(它只保证物理安全)")
print("    - AUC 明显 > 0.5 的失败项 => 与保真度正相关, 保留")
print("    - 连续量 |rho| 小的 => 不能替代示范保真度")

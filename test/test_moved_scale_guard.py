#!/usr/bin/env python3
"""moved 分支缩放 guard 回归测试。

背景 (真 bug):
  tools/rescale_meshes_by_pointcloud.py 的 sanity guard 对 moved 实例只查 k_val:
      if moved: guard_ok = _K_MIN <= k_val <= _K_MAX
  而均匀缩放下"最长轴 after ≡ 1"是由 k_val 构造的恒等式, 会掩盖另外两轴的
  垃圾缩放 —— 只要 k 落在 [0.5,2.0], mesh 与点云在另两轴差 5 倍也照样通过。

  实测 vertical_pick_place_115 bottle: k 合法, 另两轴 after 到 4.9,
  各向异性残差 3.898, 模型被拉成畸形。vertical_pick_place_135 同类 (残差 1.067)。

修复: moved 且非 hybrid 时, 补查 after_ratio 的 min/max 落在
  _MOVED_AFTER_MIN/_MOVED_AFTER_MAX = [0.5, 2.0]
  (比静止物体的 [0.70,1.55] 宽 —— 移动物体只用静止段测量, 太严会误杀)。

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_moved_scale_guard.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "tools", "rescale_meshes_by_pointcloud.py")

_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


txt = open(SRC, encoding="utf-8").read()

print("\n[1] 常量已定义")
m = re.search(r"_MOVED_AFTER_MIN,\s*_MOVED_AFTER_MAX\s*=\s*([\d.]+),\s*([\d.]+)", txt)
check("_MOVED_AFTER_MIN/_MAX 存在", m is not None)
if m:
    lo, hi = float(m.group(1)), float(m.group(2))
    check("范围比静止物体宽 (静止 0.70~1.55)", lo <= 0.70 and hi >= 1.55,
          f"[{lo},{hi}] vs [0.70,1.55]")

print("\n[2] moved 分支补查了 after_ratio")
# 按缩进定位 moved 分支主体: 取 "if moved:" 所在缩进, 一直到同缩进的 "else:"
# (不能用 find 第一个 else: —— 分支内部可能有别的 else, 会提前截断)
lines = txt.split("\n")
# 文件里有 3 处 "if moved:" (line106/172/264), 只有 line264 是 sanity guard 分支。
# 用"分支内出现 guard_ok"来精确定位, 不能只取第一个。
cands = [n for n, L in enumerate(lines) if L.strip() == "if moved:"]
i = next((n for n in cands
          if any("guard_ok" in L for L in lines[n + 1:n + 30])), -1)
check("定位到 guard 用的 moved 分支", i >= 0,
      f"候选行={[n + 1 for n in cands]} → 选中 {i + 1}")
indent = len(lines[i]) - len(lines[i].lstrip()) if i >= 0 else 0
j = next((n for n in range(i + 1, len(lines))
          if lines[n].strip().startswith("else:") and
          (len(lines[n]) - len(lines[n].lstrip())) == indent), len(lines))
body = "\n".join(lines[i:j]) if i >= 0 else ""
# 必须确认 j 真的匹配到了同缩进的 else:, 否则 j 会退化成文件末尾,
# body 吞掉整个后半文件 —— 那样下面的断言会因为"搜到了"而假通过。
check("moved 分支被正确闭合 (行数合理)", 0 < len(body.splitlines()) <= 40,
      f"{len(body.splitlines())} 行 (j={'匹配到 else:' if j < len(lines) else '!!退化为文件末尾'})")
check("分支内不含后续无关代码", "is_hybrid:" not in body.split("if is_hybrid")[0]
      if "if is_hybrid" in body else True)
check("补查了 after_ratio", "after_ratio" in body,
      "原来只有 k_val 检查, 另两轴垃圾缩放被漏掉")
check("用了 _MOVED_AFTER_MIN", "_MOVED_AFTER_MIN" in body)
check("排除了 hybrid (混合分支另有 k_z 检查)", "not is_hybrid" in body)

print("\n[3] 静止分支不受影响 (仍用严界)")
k = txt.find("_AFTER_MIN, _AFTER_MAX")
check("静止分支仍用原界", k > 0)
nonmoved = txt[txt.find("else:", txt.find("if moved:")):]
check("静止分支仍检查 after_ratio", "_AFTER_MIN" in nonmoved)

print("\n[4] 收敛检查能报出实例名")
check("all_after_names 已定义", "all_after_names = []" in txt)
check("append 了实例名", 'all_after_names.append(f"{cat}_{ii}")' in txt)
check("循环里用了 zip 配对", "zip(all_after_names, arr)" in txt)
# fmt 定义在循环内 (lambda), 循环外不可见 —— 确认没误用
seg = txt[txt.find("if all_after_ratios:"):]
check("循环外没误用循环内的 fmt", "fmt(_row)" not in seg)

print("\n[5] 语法可编译")
import py_compile
try:
    py_compile.compile(SRC, doraise=True)
    check("compile 通过", True)
except Exception as e:
    check("compile 通过", False, str(e)[:120])

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)

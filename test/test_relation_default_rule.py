#!/usr/bin/env python3
"""关系判断提示词的兜底默认值回归测试。

背景 (真 bug, 两层):
  第一层: 无条件 'If unsure, default to: "supported by floor"', 与提示词自己
          写的 "FLOOR is NOT visible" 矛盾。
  第二层: 改成按 has_floor 选默认值后仍无效 —— 实测 basic_pick_place_46 报
          "floor=✅ (43 个 mask)", SAM3 把整个桌面当成了地面 → has_floor=True
          → 依旧全标 floor。说明 has_floor 本身是不可靠信号。
  最终: 默认值不依赖 has_floor, 统一 "supported by other objects", 交给下游
        _infer_supporter 按 xy 重叠 + z 关系做几何推断。
  全标 floor 的后果: has_inter_object=False → 5.2 整段跳过 →
    ① resolve_penetrations 不跑 → 穿模残留 (46 的 box_1/box_2 重叠 92%)
    ② placement_strategies.json 不写 → on_top 无法区分 inside
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.generate_scene_json_stage1 import _build_relationship_prompt as B  # noqa: E402

_ok = True


def ck(n, c, d=""):
    global _ok
    print(f"  {'PASS' if c else 'FAIL'}  {n}  {d}")
    if not c:
        _ok = False


def default_line(prompt):
    for L in prompt.split("\n"):
        if "unsure" in L and "default to" in L:
            return L.strip()
    return ""


print("\n[1] 默认值统一为 other objects (两种 has_floor 都一样)")
for hf in (True, False):
    line = default_line(B(["cup", "table"], has_floor=hf, has_wall=False))
    ck(f"has_floor={hf}", line.startswith('- If unsure, default to: "supported by other objects"'),
       line[:70])

print("\n[2] 不再依赖 has_floor (两层 bug 的回归护栏)")
lo = default_line(B(["a"], has_floor=False, has_wall=False))
lt = default_line(B(["a"], has_floor=True, has_wall=False))
ck("两种 has_floor 产出相同默认值", lo == lt)

print("\n[3] 明确禁止 floor 当兜底")
line = default_line(B(["a"], has_floor=True, has_wall=False))
ck("提示语禁止 catch-all", "do NOT fall back" in line)
p = B(["a"], has_floor=True, has_wall=False)
ck("rule 5 禁止 floor catch-all", "do NOT use" in p and "catch-all" in p)

print("\n[4] 四条关系仍完整 + 占位符替换")
for hf in (True, False):
    pp = B(["a", "b", "c"], has_floor=hf, has_wall=False)
    ck(f"has_floor={hf} 四条关系齐全",
       all(k in pp for k in ("supported by floor", "supported by other objects",
                             "attached to wall", "embedded in wall")))
    ck(f"has_floor={hf} 无残留占位符", "{_" not in pp)

print("\n[5] 物体列表/数量注入")
p5 = B(["x1", "y2", "z3"], has_floor=False, has_wall=False)
ck("列表注入", "x1" in p5 and "z3" in p5)
ck("数量注入", "Total: 3 objects" in p5)

print("\n" + ("OK 全部通过" if _ok else "有失败"))
sys.exit(0 if _ok else 1)

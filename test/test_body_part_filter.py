#!/usr/bin/env python3
"""Stage1 人体部位过滤回归测试。

背景: VLM_DETECT_PROMPT 已写 "Ignore: hands, body parts, walls, floors,
ceilings", 但实测 86.mp4 仍输出 "arm" (画面里人的手臂)。放任会导致:
  - pick_place_detector 把操作者判成抓取事件 (86 的 arm_inst1 4 次交互)
  - 支撑关系被污染 (plate_0 → supported by arm_1, 水平重叠 0%)
  - 场景里多出一个不该有的模型

本测试锁住:
  1. 各类人体部位(含实例后缀)被识别
  2. 合法类别不被误伤 (armchair / chair / handle / footstool / alarm ...)
  3. generate_scene_json 落盘时确实过滤, 且打印原因
  4. 全被过滤时保留原清单 (不输出空场景)
  5. 用 86.mp4 的真实 Stage1 产物验证 (应去掉 arm, 保留其余 7 个)

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_body_part_filter.py
"""
import os
import sys
import json
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from generate_scene_json_stage1 import _is_body_part, generate_scene_json  # noqa: E402

_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


print("\n[1] 人体部位被识别 (含实例后缀)")
for n in ["arm", "hand", "finger", "person", "people", "human", "head", "face",
          "leg", "foot", "eye", "ear", "mouth", "baby", "man", "woman",
          "arm_0", "hand_1", "person_2", "finger_10", "Arm", "ARM"]:
    check(f"'{n}' 是人体部位", _is_body_part(n))

print("\n[2] 合法类别不被误伤 (精确匹配, 非子串)")
for n in ["armchair", "chair", "table", "handle", "footstool", "alarm", "football",
          "armband", "handbag", "body_cream", "backpack", "necklace", "toaster",
          "eyeglasses", "earphone", "mouthwash", "persona", "humanoid_robot",
          "banana", "bowl", "plate", "cup", "bottle", "toy", "block", "utensil"]:
    check(f"'{n}' 不是人体部位", not _is_body_part(n), f"→ {_is_body_part(n)}")

print("\n[3] generate_scene_json 过滤并落盘")
with tempfile.TemporaryDirectory() as td:
    reg = {
        "arm": "supported by other objects",
        "banana": "supported by floor",
        "bowl": "supported by other objects",
        "plate": "supported by other objects",
        "hand_0": "supported by other objects",
        "cup": "supported by floor",
    }
    out = os.path.join(td, "scene_test_stage1.json")
    generate_scene_json(reg, out)
    saved = json.load(open(out, encoding="utf-8"))
    check("arm 被过滤", "arm" not in saved, f"剩余: {sorted(saved)}")
    check("hand_0 被过滤", "hand_0" not in saved)
    check("合法物体保留", set(saved) == {"banana", "bowl", "plate", "cup"},
          f"剩余: {sorted(saved)}")
    check("关系值未丢", saved.get("banana") == "supported by floor")
    check("类别数从 6 → 4", len(saved) == 4)

print("\n[4] 兜底: 全部是人体部位时保留原清单")
with tempfile.TemporaryDirectory() as td:
    reg = {"arm": "supported by floor", "hand": "supported by floor"}
    out = os.path.join(td, "scene_all_body_stage1.json")
    generate_scene_json(reg, out)
    saved = json.load(open(out, encoding="utf-8"))
    check("未输出空场景", len(saved) == 2, f"剩余: {sorted(saved)}")

print("\n[5] 真实数据: 86.mp4 的 Stage1 产物")
p = os.path.join(REPO, "output_v2", "86_verify", "scene_86_stage1.json")
if os.path.isfile(p):
    reg = json.load(open(p, encoding="utf-8"))
    print(f"    原始 {len(reg)} 类: {sorted(reg)}")
    dropped = sorted(k for k in reg if _is_body_part(k))
    kept = sorted(k for k in reg if not _is_body_part(k))
    check("arm 被识别为人体部位", dropped == ["arm"], f"过滤: {dropped}")
    check("其余 7 个全部保留", len(kept) == 7, f"保留: {kept}")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "scene_86_filtered_stage1.json")
        generate_scene_json(reg, out)
        saved = json.load(open(out, encoding="utf-8"))
        check("落盘后 arm 消失且其余保留",
              "arm" not in saved and len(saved) == 7,
              f"落盘 {len(saved)} 类: {sorted(saved)}")
else:
    print("    (86_verify/scene_86_stage1.json 不存在, 跳过)")

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)

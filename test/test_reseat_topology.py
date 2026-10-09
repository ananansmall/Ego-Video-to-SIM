#!/usr/bin/env python3
"""reseat_on_top_objects / dump_placement_audit 回归测试。

覆盖四类真实故障:
  1. 多层支撑链 (A→B→C) 必须按拓扑序处理, 否则被支撑物会贴到支撑物的旧顶面
  2. 支撑环 (sheet_0 ↔ sheet_1) 必须断开, 否则互相往上贴会无限发散
  3. 'supported by other objects' 需要几何兜底挑支撑物, 不能躺平不动
  4. 无 placement_strategies 时, inside 的物体 (碗里的面包) 不能被拽到碗沿

trimesh 用哑模块顶掉 —— 被测函数只用到 mesh.copy()/apply_transform()/bounds。
"""
import os
import sys
import types
import json
import tempfile
import numpy as np

_tm = types.ModuleType("trimesh")
_tm.geometry = types.ModuleType("trimesh.geometry")
_tm.transformations = types.ModuleType("trimesh.transformations")
sys.modules.setdefault("trimesh", _tm)
sys.modules.setdefault("trimesh.geometry", _tm.geometry)
sys.modules.setdefault("trimesh.transformations", _tm.transformations)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.sp_refinement import reseat_on_top_objects, dump_placement_audit  # noqa: E402


class FakeMesh:
    """8 顶点盒, 局部坐标居中。"""

    def __init__(self, size):
        s = np.asarray(size, dtype=float) / 2.0
        self.v = np.array([[x, y, z]
                           for x in (-s[0], s[0])
                           for y in (-s[1], s[1])
                           for z in (-s[2], s[2])], dtype=float)

    @classmethod
    def _from_v(cls, v):
        o = cls.__new__(cls)
        o.v = v
        return o

    def copy(self):
        return FakeMesh._from_v(self.v.copy())

    def apply_transform(self, T):
        T = np.asarray(T, dtype=float)
        self.v = self.v @ T[:3, :3].T + T[:3, 3]
        return self

    @property
    def bounds(self):
        return np.array([self.v.min(axis=0), self.v.max(axis=0)])


def inst(size, center):
    T = np.eye(4)
    T[:3, 3] = np.asarray(center, dtype=float)
    return {"original_mesh": FakeMesh(size), "T": T}


def bottom(all_instances, cat, i=0):
    m = all_instances[cat][i]["original_mesh"].copy()
    m.apply_transform(all_instances[cat][i]["T"])
    return float(m.bounds[0, 2])


_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


def test_chain_topological_order():
    print("\n[1] 多层支撑链 (拓扑序)")
    # dict 顺序故意把最上层 cup 放最前: 旧版单次遍历会先处理 cup, 读到 box 的旧顶面
    a = {"cup": [inst((0.1, 0.1, 0.08), (0, 0, 0.74))],
         "box": [inst((0.3, 0.3, 0.10), (0, 0, 0.65))],
         "table": [inst((1.0, 1.0, 0.75), (0, 0, 0.375))]}
    r = {"table_0": "supported by floor",
         "box_0": "supported by table_0",
         "cup_0": "supported by box_0"}
    strat = {"box_0": "on_top", "cup_0": "on_top"}
    a, n = reseat_on_top_objects(a, r, placement_strategies=strat, verbose=False)
    check("box_0 贴到桌面 0.75", abs(bottom(a, "box") - 0.75) < 1e-6, f"→ {bottom(a,'box'):.4f}")
    check("cup_0 贴到 box 新顶面 0.85", abs(bottom(a, "cup") - 0.85) < 1e-6,
          f"→ {bottom(a,'cup'):.4f}")
    check("重贴计数", n == 2, f"n={n}")


def test_cycle_broken():
    print("\n[2] 支撑环 (断开面积最大者)")
    a = {"sheet": [inst((2.0, 2.0, 0.02), (0, 0, 0.03)),
                   inst((0.5, 0.5, 0.02), (0, 0, 0.11))]}
    r = {"sheet_0": "supported by sheet_1", "sheet_1": "supported by sheet_0"}
    a, n = reseat_on_top_objects(a, r, verbose=False)
    check("sheet_0 (根) 不动", abs(bottom(a, "sheet", 0) - 0.02) < 1e-6,
          f"→ {bottom(a,'sheet',0):.4f}")
    check("sheet_1 贴到 sheet_0 顶面 0.04", abs(bottom(a, "sheet", 1) - 0.04) < 1e-6,
          f"→ {bottom(a,'sheet',1):.4f}")
    check("未发散", bottom(a, "sheet", 1) < 0.5)


def test_infer_supporter_for_other_objects():
    print("\n[3] other objects 推断支撑物")
    a = {"box": [inst((0.4, 0.4, 0.15), (0, 0, 0.075))],
         "stapler": [inst((0.15, 0.10, 0.06), (0, 0, 0.23))],
         "far": [inst((0.2, 0.2, 0.10), (5, 5, 0.05))]}
    r = {"box_0": "supported by floor",
         "stapler_0": "supported by other objects",
         "far_0": "supported by floor"}
    a, n = reseat_on_top_objects(a, r, verbose=False)
    check("stapler_0 贴到 box 顶面 0.15", abs(bottom(a, "stapler") - 0.15) < 1e-6,
          f"→ {bottom(a,'stapler'):.4f}")
    check("远处 far_0 未被误选", abs(bottom(a, "far") - 0.0) < 1e-6)

    # 找不到支撑物时保持原位, 不乱猜
    b = {"ghost": [inst((0.1, 0.1, 0.06), (9, 9, 0.40))],
         "box": [inst((0.4, 0.4, 0.15), (0, 0, 0.075))]}
    rb = {"ghost_0": "supported by other objects", "box_0": "supported by floor"}
    b, nb = reseat_on_top_objects(b, rb, verbose=False)
    check("无支撑物时保持原位", abs(bottom(b, "ghost") - 0.37) < 1e-6 and nb == 0)


def test_inside_guard():
    print("\n[4] inside 保护")
    r = {"bowl_0": "supported by floor", "bread_0": "supported by bowl_0"}

    a = {"bowl": [inst((0.3, 0.3, 0.10), (0, 0, 0.15))],
         "bread": [inst((0.08, 0.08, 0.05), (0, 0, 0.145))]}
    a, n = reseat_on_top_objects(a, r, placement_strategies=None, verbose=False)
    check("无策略文件: 面包留在碗内", abs(bottom(a, "bread") - 0.12) < 1e-6 and n == 0,
          f"→ {bottom(a,'bread'):.4f}")

    a2 = {"bowl": [inst((0.3, 0.3, 0.10), (0, 0, 0.15))],
          "bread": [inst((0.08, 0.08, 0.05), (0, 0, 0.145))]}
    a2, n2 = reseat_on_top_objects(a2, r, placement_strategies={"bread_0": "inside"},
                                   verbose=False)
    check("策略 inside → 不动", abs(bottom(a2, "bread") - 0.12) < 1e-6 and n2 == 0)

    a3 = {"bowl": [inst((0.3, 0.3, 0.10), (0, 0, 0.15))],
          "bread": [inst((0.08, 0.08, 0.05), (0, 0, 0.145))]}
    a3, n3 = reseat_on_top_objects(a3, r, placement_strategies={"bread_0": "on_top"},
                                   verbose=False)
    check("策略 on_top → 贴到碗沿 0.20", abs(bottom(a3, "bread") - 0.20) < 1e-6,
          f"→ {bottom(a3,'bread'):.4f}")


def test_low_overlap_not_dragged():
    print("\n[5] 支撑物水平重叠过低")
    a = {"table": [inst((0.5, 0.5, 0.75), (0, 0, 0.375))],
         "mug": [inst((0.1, 0.1, 0.10), (3.0, 0, 0.90))]}
    r = {"table_0": "supported by floor", "mug_0": "supported by table_0"}
    a, n = reseat_on_top_objects(a, r, verbose=False)
    check("mug_0 未被拽到桌面", abs(bottom(a, "mug") - 0.85) < 1e-6, f"→ {bottom(a,'mug'):.4f}")


def test_audit():
    print("\n[6] dump_placement_audit")
    with tempfile.TemporaryDirectory() as td:
        a = {"box": [inst((0.4, 0.4, 0.15), (0, 0, 0.075))],
             "toy": [inst((0.1, 0.1, 0.06), (0, 0, 0.40))],
             "ghost": [inst((0.1, 0.1, 0.06), (9, 9, 0.40))]}
        r = {"box_0": "supported by floor",
             "toy_0": "supported by box_0",
             "ghost_0": "supported by other objects"}
        aud = dump_placement_audit(a, r, output_dir=td, verbose=False)
        check("审计文件已写", os.path.isfile(os.path.join(td, "placement_audit.json")))
        with open(os.path.join(td, "placement_audit.json"), encoding="utf-8") as f:
            check("JSON 可解析", isinstance(json.load(f), dict))
        st = {o["name"]: o["status"] for o in aud["objects"]}
        check("box_0 = ok", st.get("box_0") == "ok", str(st.get("box_0")))
        check("toy_0 = floating", st.get("toy_0") == "floating", str(st.get("toy_0")))
        check("ghost_0 = no_supporter_found", st.get("ghost_0") == "no_supporter_found",
              str(st.get("ghost_0")))
        check("n_problem 统计", aud["n_problem"] == 2, f"n_problem={aud['n_problem']}")
        g = [o for o in aud["objects"] if o["name"] == "toy_0"][0]
        check("gap 记录正确", abs(g["gap"] - 0.22) < 1e-6, f"gap={g['gap']:.4f}")


if __name__ == "__main__":
    test_chain_topological_order()
    test_cycle_broken()
    test_infer_supporter_for_other_objects()
    test_inside_guard()
    test_low_overlap_not_dragged()
    test_audit()
    print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
    sys.exit(0 if _ok else 1)

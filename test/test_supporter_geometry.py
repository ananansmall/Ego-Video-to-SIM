#!/usr/bin/env python3
"""支撑物几何判据回归测试 (5.15 纠偏 + reseat 的平板判据)。

覆盖两个真实 bug:

1) reseat 的 inside_ratio 判据对"平板支撑物"误判
   实测 basic_pick_place_141: 棋盘 0.061m 高, 10 个棋子/遥控底面全在 z=0
   (地面), 棋盘顶面 z=0.061 → sink=0.061 > 0.5×棋子高 0.045, 全被判成
   "装在棋盘里" 而跳过, 于是 10 个物体一直留在地面 (陷地 10/11)。
   棋盘是平板, 不可能容纳东西, 因此判据必须排除平板。

2) 无几何成立的支撑物时未回退 floor
   实测 vertical_pick_place_11: table_0 重建在 z=[-1.17,-0.90] (地下 1 米),
   box_0/toy_0 在 z≈0, 水平重叠 0~1%。这种关系几何上不可能成立, 保留它
   会让后续按错误基准算 gap, 物体永远修不好。
"""
import os
import sys

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.sp_refinement import (  # noqa: E402
    reseat_on_top_objects, fix_supporter_instance, _world_bounds)

_ok = True


def ck(n, c, d=""):
    global _ok
    print(f"  {'PASS' if c else 'FAIL'}  {n}  {d}")
    if not c:
        _ok = False


def box(ext, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=ext)
    m.apply_translation(center)
    return m


def inst(mesh, T=None):
    return {"original_mesh": mesh,
            "T": np.eye(4) if T is None else T}


def zmin(insts, cat, i):
    return float(_world_bounds(insts[cat][i])[0][2])


def zmax(insts, cat, i):
    return float(_world_bounds(insts[cat][i])[1][2])


print("\n[1] 平板支撑物 (棋盘): 陷在里面的棋子必须被抬上去")
# 棋盘: 0.40 x 0.40 x 0.06, 顶面 z=0.06
# 棋子: 0.06 x 0.06 x 0.09, 底面 z=0 (陷 0.06, 超过半高 0.045)
insts = {
    "chessboard": [inst(box((0.40, 0.40, 0.06), (0, 0, 0.03)))],
    "piece": [inst(box((0.06, 0.06, 0.09), (0, 0, 0.045)))],
}
rel = {"piece_0": "supported by chessboard_0",
       "chessboard_0": "supported by floor"}
before = zmin(insts, "piece", 0)
out, n = reseat_on_top_objects(insts, rel, placement_strategies=None,
                               verbose=False)
after = zmin(out, "piece", 0)
ck("被抬到棋盘顶面", abs(after - 0.06) < 0.01, f"{before:+.3f} → {after:+.3f}")
ck("确实执行了 reseat", n >= 1, f"n={n}")

print("\n[2] 容器 (碗): 藏在里面的物体不能被拽到碗沿")
# 碗: 0.20 x 0.20 x 0.12, 顶面 z=0.12
# 面包: 0.08 x 0.08 x 0.08, 底面 z=0.02 (陷 0.10, 超过半高 0.04)
insts = {
    "bowl": [inst(box((0.20, 0.20, 0.12), (0, 0, 0.06)))],
    "bread": [inst(box((0.08, 0.08, 0.08), (0, 0, 0.06)))],
}
rel = {"bread_0": "supported by bowl_0", "bowl_0": "supported by floor"}
before = zmin(insts, "bread", 0)
out, n = reseat_on_top_objects(insts, rel, placement_strategies=None,
                               verbose=False)
after = zmin(out, "bread", 0)
ck("inside 物体保持不动", abs(after - before) < 1e-6, f"{before:+.3f} → {after:+.3f}")

print("\n[3] 支撑物在地下 1 米 (几何不可能) → 回退 floor")
# table 在 z=[-1.17, -0.90], 物体在 z≈0, 水平完全错开
insts = {
    "table": [inst(box((0.80, 0.80, 0.30), (2.0, 2.0, -1.0)))],   # 远处且地下
    "box": [inst(box((0.30, 0.30, 0.20), (0, 0, 0.10)))],
}
rel = {"box_0": "supported by table_0", "table_0": "supported by floor"}
out_rel, n = fix_supporter_instance(insts, dict(rel), verbose=False)
ck("回退为 supported by floor", out_rel["box_0"] == "supported by floor",
   out_rel["box_0"])
ck("计数正确", n >= 1, f"n={n}")

print("\n[4] 支撑物合法时不改动 VLM 的选择 (尊重语义)")
# 物体正好坐在桌面上, 重叠 100%, z 贴合
insts = {
    "desk": [inst(box((0.60, 0.60, 0.05), (0, 0, 0.025)))],
    "cup": [inst(box((0.08, 0.08, 0.10), (0, 0, 0.10)))],
}
rel = {"cup_0": "supported by desk_0", "desk_0": "supported by floor"}
out_rel, n = fix_supporter_instance(insts, dict(rel), verbose=False)
ck("合法关系保持不动", out_rel["cup_0"] == "supported by desk_0",
   out_rel["cup_0"])
ck("无改动", n == 0, f"n={n}")

print("\n[5] 同类多实例: 选几何正确的那个")
# 两块板: board_0 在远处, board_1 在物体正下方
insts = {
    "board": [inst(box((0.40, 0.40, 0.05), (3.0, 0, 0.025))),
              inst(box((0.40, 0.40, 0.05), (0, 0, 0.025)))],
    "piece": [inst(box((0.06, 0.06, 0.09), (0, 0, 0.095)))],
}
rel = {"piece_0": "supported by board_0",   # VLM 选错了
       "board_0": "supported by floor", "board_1": "supported by floor"}
out_rel, n = fix_supporter_instance(insts, dict(rel), verbose=False)
ck("改判到几何正确的 board_1", out_rel["piece_0"] == "supported by board_1",
   out_rel["piece_0"])

print("\n" + ("OK 全部通过" if _ok else "有失败"))
sys.exit(0 if _ok else 1)

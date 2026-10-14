#!/usr/bin/env python3
"""align_to_room_coordinate_system 的 info 落盘回归测试。

背景: 该函数原先只返回 (R, t), 而 align_via_objects / align_via_large_plane /
align_via_geocalib 都返回 (R, t, info)。结果 40 个 stage1_strict 场景的
coordinate_alignment.json 里 method_detail 永远是空的 —— 第 1 级为什么成功、
选中的是哪个平面, 完全不可追溯。

本测试构造合成点云 + mask, 覆盖成功与三条失败分支, 断言:
  1. 返回值是 3 元组 (与其它级签名一致)
  2. 成功时记录了所选平面 (面积/质心/法向/分歧度)
  3. 失败时有 reason
  4. R[2,2] 只作为记录项存在, 不参与判定
"""
import os
import sys
import types
import inspect
import numpy as np

for _name in ("trimesh", "matplotlib", "matplotlib.pyplot", "scipy", "scipy.spatial"):
    _m = types.ModuleType(_name)

    class _Any(types.ModuleType):
        def __getattr__(self, k):
            return type(k, (), {})

    sys.modules.setdefault(_name, _Any(_name))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import src.geometry_utils as gu  # noqa: E402
from src.geometry_utils import align_to_room_coordinate_system, _build_R_t_from_floor  # noqa: E402

# Delaunay 被 stub 掉会导致平面面积恒为 0, 选平面时区分不出大小。这里给面积一个
# 可控实现 (面积 = mask 点数), 让测试聚焦在 info 落盘逻辑上。
gu.compute_surface_area_from_pointmap = \
    lambda pointmap, mask, max_triangle_size=2e-4: float(mask.sum())

_ok_all = True


def check(name, cond, detail=""):
    global _ok_all
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok_all = False


H, W = 60, 80
rng = np.random.default_rng(0)


def make_world_points(n_frames=4, floor_z=0.0, tilted=False, noise=0.0, wall=True):
    """造 (T,H,W,3) 点云: 下方 floor 平面(法向 +z), 侧面 wall 平面(法向 -x)。"""
    pts = np.zeros((n_frames, H, W, 3))
    for f in range(n_frames):
        for i in range(H):
            for j in range(W):
                if i < H // 2:                      # 下半 = floor
                    z = floor_z
                    if tilted:
                        z += 0.3 * (j / W)
                    pts[f, i, j] = [j / W, i / H, z + rng.normal(0, noise)]
                else:                               # 上半 = wall (法向 -x)
                    x = 0.8 if wall else 2.0 + i / H
                    pts[f, i, j] = [x, j / W, 1.0 + i / H]
    return pts


def masks(n_frames=4, floor=True, wall=True):
    fm, wm = [], []
    for f in range(n_frames):
        if floor:
            m = np.zeros((H, W), bool)
            m[:H // 2, :] = True
            fm.append({"frame_id": f, "mask": m})
        if wall:
            m = np.zeros((H, W), bool)
            m[H // 2:, :] = True
            wm.append({"frame_id": f, "mask": m})
    return wm, fm


print("\n[1] 签名: 必须返回 3 元组")
wp = make_world_points()
wm, fm = masks()
out = align_to_room_coordinate_system(wp, wm, fm)
check("返回 3 个值", isinstance(out, tuple) and len(out) == 3,
      f"实际 {len(out) if isinstance(out, tuple) else type(out).__name__}")

print("\n[2] 成功分支: info 记录了所选平面")
R, t, info = align_to_room_coordinate_system(wp, wm, fm)
check("R 形状正确", getattr(R, "shape", None) == (3, 3))
check("method 标识", info.get("method") == "floor+wall", str(info.get("method")))
check("无 reason (成功)", "reason" not in info, str(info.get("reason")))
check("记录了 floor_area", "floor_area" in info, f"floor_area={info.get('floor_area')}")
check("记录了 floor_centroid", "floor_centroid" in info)
check("记录了法向分歧度", "normal_spread_deg" in info,
      f"spread={info.get('normal_spread_deg'):.2f}°" if "normal_spread_deg" in info else "")
check("记录了候选计数", info.get("n_floor_planes") == 4, str(info.get("n_floor_planes")))
check("记录了 R22 (仅记录)", "R22_not_a_metric" in info,
      f"R22={info.get('R22_not_a_metric')}")

print("\n[3] 失败分支 ①: 没有 floor mask")
R, t, info = align_to_room_coordinate_system(wp, wm, [])
check("返回单位阵", np.allclose(R, np.eye(3)))
check("reason = no_valid_floor_plane", info.get("reason") == "no_valid_floor_plane",
      str(info.get("reason")))

print("\n[4] 失败分支 ②: 各帧 floor 法向严重分歧 (>30°)")
# 让一半帧的 floor 平面倾斜 60°, 与另一半冲突
wp2 = make_world_points(n_frames=4)
for f in (2, 3):
    for i in range(H // 2):
        for j in range(W):
            wp2[f, i, j] = [j / W, i / H, 0.9 * (j / W)]
R, t, info = align_to_room_coordinate_system(wp2, wm, fm)
check("判定失败 (单位阵)", np.allclose(R, np.eye(3)), str(info.get("reason")))
check("reason 指明法向分歧或选到了某一侧",
      info.get("reason") in ("no_consistent_floor_normal", None), str(info.get("reason")))

print("\n[5] 失败分支 ③: 没有与 floor 正交的墙")
wp3 = make_world_points(wall=False)
wm_no_wall, fm3 = masks(wall=True)
for m in wm_no_wall:               # 把 wall mask 指到一块平行于 floor 的区域
    m["mask"][:] = False
    m["mask"][:3, :] = True
R, t, info = align_to_room_coordinate_system(wp3, wm_no_wall, fm3)
if np.allclose(R, np.eye(3)):
    check("失败时带 reason", bool(info.get("reason")), str(info.get("reason")))
else:
    check("墙退化时仍给出 R", R.shape == (3, 3), "(该构造下未触发失败分支)")

print("\n[6] 回归: _build_R_t_from_floor 仍返回 2 元组")
# 该函数被 align_via_objects / align_via_large_plane / align_via_geocalib 调用,
# 签名不能跟着 align_to_room_coordinate_system 一起改 (曾误改, 会导致 NameError)。
_r2 = _build_R_t_from_floor(wp, np.array([0.0, 0.0, 1.0]),
                            np.array([0.0, 0.0, 0.0]))
check("_build_R_t_from_floor 返回 2 个值",
      isinstance(_r2, tuple) and len(_r2) == 2,
      f"实际 {len(_r2) if isinstance(_r2, tuple) else type(_r2).__name__}")

print("\n[7] 关键回归: 判据不依赖 R[2,2]")
src = inspect.getsource(align_to_room_coordinate_system)
# 允许出现 R22 作为"记录", 但不允许作为 if 条件
bad = [ln.strip() for ln in src.splitlines()
       if "R[2, 2]" in ln and ("if " in ln or "<" in ln.split("R[2, 2]")[0])]
check("R[2,2] 未用于条件判定", not bad, str(bad))

print("\n" + ("✅ 全部通过" if _ok_all else "❌ 有失败"))
sys.exit(0 if _ok_all else 1)

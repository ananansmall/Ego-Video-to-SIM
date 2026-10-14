#!/usr/bin/env python3
"""_gravity_consistency_quality 回归测试。

背景: 旧的 GeoCalib 质量判据是 abs(R[2,2]) < quality_threshold, 依据
"R[2,2] = floor_normal[2], 理想值 ≈ ±1.0"。该依据要求 VGGT 的 world 坐标系是
z-up, 已被实测证伪 (46 个场景 |R[2,2]| 中位数 0.42, 最小 0.047, 但对齐是好的)。
新判据改用"重力跨帧自洽性", 不依赖任何坐标惯例。

关键回归点: 判据的结果必须与 R / floor_normal 无关 —— 用签名反射断言。
"""
import os
import sys
import types
import inspect
import numpy as np

# geometry_utils 顶部 import 了 trimesh/matplotlib/scipy, 环境里没有 → 用哑模块顶掉。
# 注意 geometry_utils 用到 `-> trimesh.Scene` 这类运行时求值的注解, 所以哑模块必须
# 能对任意属性名返回一个可用的对象。
class _AnyModule(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})


for _name in ("trimesh", "matplotlib", "matplotlib.pyplot", "scipy", "scipy.spatial"):
    sys.modules.setdefault(_name, _AnyModule(_name))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.geometry_utils import _gravity_consistency_quality as Q  # noqa: E402

_ok_all = True


def check(name, cond, detail=""):
    global _ok_all
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok_all = False


def make_gravity(n=10, spread_deg=1.0, seed=0):
    """造 n 个重力向量, 围绕 -z 随机散开 spread_deg 度。"""
    rng = np.random.default_rng(seed)
    base = np.array([0.0, 0.0, -1.0])
    axis = np.array([1.0, 0.0, 0.0])
    out = []
    for _ in range(n):
        r = np.radians(rng.uniform(0, spread_deg))
        phi = rng.uniform(0, 2 * np.pi)
        # 绕任意轴小角度旋转 base
        v = base * np.cos(r) + np.cross(axis, base) * np.sin(r) * np.cos(phi) \
            + axis * np.dot(axis, base) * (1 - np.cos(r))
        out.append(v / np.linalg.norm(v))
    return np.array(out)


print("\n[1] 重力一致的正常场景")
g = make_gravity(n=10, spread_deg=1.0, seed=1)
ok, q, reason = Q(g)
check("判定通过", ok, reason)
check("median_angle 很小", q["median_angle_deg"] < 3.0, f'{q["median_angle_deg"]:.2f}°')
check("inlier_ratio = 1", abs(q["inlier_ratio"] - 1.0) < 1e-9)
check("n_frames_valid 正确", q["n_frames_valid"] == 10)

print("\n[2] 半数帧严重离散 (模拟 GeoCalib 估计不稳)")
good = make_gravity(n=5, spread_deg=1.0, seed=2)
# 造 5 个偏离很大的帧
bad = np.array([[0.0, np.sin(np.radians(60)), -np.cos(np.radians(60))]] * 5)
g = np.vstack([good, bad])
mask = np.ones(10, bool)
ok, q, reason = Q(g, inlier_mask=mask)
check("判定失败", not ok, reason)
check("median_angle 超阈值", q["median_angle_deg"] > 15.0, f'{q["median_angle_deg"]:.2f}°')
check("reason 指明离散度", "inconsistent_gravity" in reason, reason)

print("\n[3] 内点率不足")
g = make_gravity(n=10, spread_deg=1.0, seed=3)
mask = np.zeros(10, bool)
mask[:4] = True          # 4/10 = 0.4 < 0.5
ok, q, reason = Q(g, inlier_mask=mask)
check("判定失败", not ok, reason)
check("reason 指明内点率", "low_inlier_ratio" in reason, reason)
check("inlier_ratio 记录正确", abs(q["inlier_ratio"] - 0.4) < 1e-9, f'{q["inlier_ratio"]:.2f}')

print("\n[4] 边界: 内点率恰好 0.5 应通过 (用 < 判断)")
mask = np.zeros(10, bool)
mask[:5] = True
ok, _q, reason = Q(g, inlier_mask=mask)
check("0.5 通过", ok, reason)

print("\n[5] 异常输入")
ok, q, reason = Q(np.zeros((0, 3)))
check("空输入失败", not ok and reason == "geocalib_no_frames", reason)

ok, q, reason = Q(np.zeros((5, 3)))
check("零向量失败", not ok and reason == "geocalib_zero_gravity_vector", reason)

ok, q, reason = Q(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
check("反平行重力失败", not ok, reason)

ok, q, reason = Q(g, inlier_mask=np.ones(3, bool))
check("mask 长度不符失败", not ok and reason == "geocalib_mask_shape_mismatch", reason)

print("\n[6] confidences 可选, 缺省不报错")
ok, q, reason = Q(g)
check("confidences=None 正常", ok, reason)
check("mean_confidence = 0", q["mean_confidence"] == 0.0)
ok, q, reason = Q(g, confidences=np.full(10, 0.8))
check("mean_confidence 正确", abs(q["mean_confidence"] - 0.8) < 1e-9,
      f'{q["mean_confidence"]:.3f}')

print("\n[7] 关键回归: 判据与 R / floor_normal 无关")
params = set(inspect.signature(Q).parameters.keys())
check("签名不含 R", not (params & {"R", "R22", "quality_threshold"}),
      f"实际参数: {sorted(params)}")
check("签名不含 floor_normal", "floor_normal" not in params)
# 用 0.047 (= egodex_28 的 |R[2,2]|, 旧判据会判失败) 的等价"好"重力数据 → 必须通过
g28 = make_gravity(n=10, spread_deg=0.5, seed=28)
ok, q, reason = Q(g28)
check("旧判据会误杀的场景形态可通过", ok, reason)

print("\n" + ("✅ 全部通过" if _ok_all else "❌ 有失败"))
sys.exit(0 if _ok_all else 1)

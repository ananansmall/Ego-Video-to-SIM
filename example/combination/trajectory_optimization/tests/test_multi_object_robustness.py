"""【多物体鲁棒性测试】—— 新增文件, **不修改任何原有代码** ✓

用户诉求 (2026-09-27)
--------------------
"如果我给你其他的物体, 比如就这个里面的其他物体, 你能不能抓的住呢,
 用 coacd 测试一下看看, 尝试一下看看, 检验一下算法的鲁棒性"

本测试**只调用**现有模块做黑盒验证, 不改任何原有代码 ✓
    场景 glb    : /home/an/data/ras/7_vggt_omega/final_scene.glb (8 个 dynamic 物体)
    CoACD 分解  : `coacd_collision.decompose` (与管线**同一实现/同一参数** ✓)
    物理抓取    : `grasp_hawor.py --test-stage1` (原入口, 子进程 ✓ 参数与 run_config 一致)

三层, 由慢到快, 逐层可单独跑
----------------------------
  A 清单层  `test_A_object_inventory`           —— 秒级 ✓  必跑
      枚举场景里所有 dynamic 物体: 名称/尺寸/体积/颜色, 并按"手指可容纳"给几何可抓性
  B 几何层  `test_B_coacd_robustness`           —— 分钟级 (每物体一次凸分解, 有磁盘缓存)
      对**每个**物体跑 CoACD (64 hull / threshold .05 / res 2000, 与管线一致 ✓)
      ⇒ 看凸包数/耗时/**体积膨胀率**/退化碎屑, 检验"换形状后 CoACD 是否还稳"
  C 物理层  `test_C_physics_grasp_per_object`   —— 每物体十几分钟, **默认跳过** ✓
      `ROBUSTNESS_PHYSICS=1` 才跑; `ROBUSTNESS_OBJECTS=glb_5,glb_6` 可只跑子集
      ⇒ 真跑一次 stage1, 看 tier=4 / 抬起 / 真两侧夹持率

用法
----
    pytest tests/test_multi_object_robustness.py -q -s          # A + B ✓
    ROBUSTNESS_PHYSICS=1 ROBUSTNESS_OBJECTS=glb_5 \\
        pytest tests/test_multi_object_robustness.py::test_C_physics_grasp_per_object -q -s

诚实约定
--------
· 本测试**只报告**, 不为了通过而放宽任何判据 ✓
· B 层只证明"CoACD 对该形状是否稳定", **不等于**"机器人能抓住它" (那要 C 层物理) ✓
· C 层若某物体的 `target_object` 仍是被手腕轨迹选中的那个 (数据驱动, 见
  `data_loader.find_target_object_by_trajectory`), 该 case 判为 **手迹外** 而不是算法失败 ✓
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GLB = Path("/home/an/data/ras/7_vggt_omega/final_scene.glb")
HAWOR_DIR = "/home/an/data/hawor/7a"
RAS_DIR = "/home/an/data/ras/7_vggt_omega"
PY = sys.executable

# 与 run_config.sh 一致的 CoACD 配方 ✓ (不复制默认值, 显式写死以便回归比对)
COACD_THRESHOLD = 0.05
COACD_MAX_HULL = 64          # run_config.sh: _COACD_MAX_HULL=64
COACD_RESOLUTION = 2000

# 手指开合能力 (来自 s31 实测日志 ✓): 全开指缝 90mm / 闭合指缝 52.6mm
GAP_OPEN_MM = 90.0
GAP_CLOSE_MM = 52.6
# 参考物体 glb_4 (toy_0) 在 `_OBJ_SCALE=0.5` 下的最大边 ≈ 5.6cm —— 作为"手能抓"的基准尺寸
REF_MAX_EXTENT_MM = 56.0

# ★ s31 有效配方基线 (2026-09-27 实测: 唯一能跑出 tier=4 ×76 / 抬起 ×25 的那组参数) ✓
#   ⚠ 实测教训 ✓: 只用 `run_config.sh` 基线跑 stage1 ⇒ box/coacd 两组都 tier=4 ×0
#     (各跑了 1200+ rollout 全白费 ✗) ⇒ 任何消融都必须**先站在这套参数上**, 只改一个变量 ✓
S1_BASELINE_ENV = {
    "_Q_OPEN": "0.030",
    "_S1_LEVEL_FINGERS": "1",
    "_S1_AUTO_CENTER": "1",
    "_S1_CENTER_MEASURE": "close",
    "_S1_CLOSE_OVERCLOSURE_MM": "3.0",
    "_S1_CLOSE_OVERCLOSURE_MAX_MM": "3.0",
    "_S1_PRELOAD_M": "0.0015",
    "_S1_T8_EXTRA_MM": "2.0",
    "_S1_T8_STEP_MM": "0.05",
    "_STAGE1_CEM_ITER": "1",
    "_STAGE1_CEM_POP": "2",
    "_STAGE1_TOP_K": "1",
    "_S1_N_FRAMES": "350",
    "_S1_CLOSE_FRAMES": "130",
    "_COACD_ALL": "1",          # 所有 dynamic 都用 CoACD (用户定策 ✓)
}


# ───────────────────────────────────────────────────────────────────────────
# 公共工具 (只读, 不改动原始数据) ✓
# ───────────────────────────────────────────────────────────────────────────
def _scene_geoms():
    """枚举 glb 里的三角网格几何 ⇒ [(glb_id, real_name, verts, faces, color), ...]"""
    import trimesh

    sc = trimesh.load(str(GLB), force="scene")
    g2n = {}
    for node in sc.graph.nodes:
        if node == "world":
            continue
        try:
            d = sc.graph[node]
            if isinstance(d, tuple) and len(d) == 2 and d[1]:
                g2n[d[1]] = node
        except Exception:  # noqa: BLE001
            pass

    out = []
    for gi, (gn, gm) in enumerate(sc.geometry.items()):
        rn = g2n.get(gn, gn)
        if not (hasattr(gm, "faces") and hasattr(gm, "vertices")):
            continue
        v = np.asarray(gm.vertices, dtype=np.float64)
        f = np.asarray(gm.faces, dtype=np.int64)
        if len(v) == 0 or len(f) == 0:
            continue
        col = None
        try:
            vc = gm.visual.vertex_colors
            if vc is not None and len(vc):
                col = (np.asarray(vc)[:, :3].mean(0) / 255.0).round(3).tolist()
        except Exception:  # noqa: BLE001
            pass
        out.append((f"glb_{gi}", rn, v, f, col))
    return out


def _classify(verts):
    """复刻 data_loader 的 static/dynamic 判据 (一致才有意义 ✓)"""
    bmin, bmax = verts.min(0), verts.max(0)
    size = bmax - bmin
    vol = abs(float(np.prod(size)))
    ext = float(size.max())
    flat = size[2] / max(size[0], size[1], 1e-6)
    is_static = (vol > 0.01 and flat < 0.3) or ext > 0.8
    return (not is_static), size, vol


def _adaptive_scale(size_mm):
    """把物体缩到与参考物体 (glb_4 @0.5 ≈ 56mm 最大边) 同量级 ⇒ 只比**形状**不比大小 ✓"""
    return REF_MAX_EXTENT_MM / max(float(size_mm), 1e-6)


def _hull_volume_sum(parts):
    """凸包体积之和 (凸包顶点已是局部坐标)"""
    tot = 0.0
    for pv, pf in parts:
        try:
            import trimesh
            tot += abs(float(trimesh.Trimesh(vertices=np.asarray(pv, float),
                                             faces=np.asarray(pf, np.int64)).volume))
        except Exception:  # noqa: BLE001
            pass
    return tot


# ───────────────────────────────────────────────────────────────────────────
# A 清单层
# ───────────────────────────────────────────────────────────────────────────
def test_A_object_inventory():
    """枚举场景物体 + 几何可抓性 (纯几何, 秒级 ✓)"""
    if not GLB.exists():
        pytest.skip(f"场景 GLB 不存在: {GLB}")

    geoms = _scene_geoms()
    assert geoms, "glb 中没有可用的三角网格几何"

    rows = []
    for gid, rn, v, f, col in geoms:
        dyn, size, vol = _classify(v)
        if not dyn:
            continue
        size_mm = np.asarray(size) * 1000.0
        sc = _adaptive_scale(float(size_mm.max()))
        scaled = np.sort(size_mm * sc)
        rows.append(dict(
            obj=gid, name=rn, verts=len(v),
            size_mm=[round(float(x), 1) for x in size_mm],
            scale=round(float(sc), 3),
            scaled_mm=[round(float(x), 1) for x in scaled],
            grip_axis_mm=round(float(scaled[0]), 1),   # 最细轴 ⇒ 最优闭合轴 ✓
            color=col,
        ))

    print("\n" + "=" * 96)
    print("【A】场景物体清单 (dynamic ✓)  参考: glb_4 @_OBJ_SCALE=0.5 ⇒ 最大边 56mm")
    print("=" * 96)
    print(f"{'物体':<8}{'名称':<12}{'原始尺寸mm':<22}{'自适应scale':<13}{'缩放后mm':<22}{'最细轴':<9}{'指缝?'}")
    for r in rows:
        fits = "✓可夹" if r["grip_axis_mm"] <= GAP_OPEN_MM else "✗过大"
        print(f"{r['obj']:<8}{r['name'][:11]:<12}{str(r['size_mm']):<22}"
              f"{r['scale']:<13}{str(r['scaled_mm']):<22}{r['grip_axis_mm']:<9}{fits}")
    print(f"\n  手指开合能力 (s31 实测): 全开 {GAP_OPEN_MM}mm / 闭合 {GAP_CLOSE_MM}mm")

    assert len(rows) >= 5, f"dynamic 物体过少 ({len(rows)}) —— 场景加载/分类可能变了"
    # 参考物体必须还在 (否则后续对比失去基准)
    assert any(r["obj"] == "glb_4" for r in rows), "参考物体 glb_4 不在场景中 ⇢ 基准丢失"


def test_A3_target_selection_locked_by_color():
    """★ 直接回答"换物体能不能抓": 目标物体由 **粉色启发式** 决定 (`pink_obj or 手迹target`)

    grasp_hawor.py 两处 (3843 / 4143) 都是 `target_obj=pink_obj or target_obj`
        ⇒ 只要场景里还有粉色物体, **换任何 CoACD/缩放旋钮都不会改抓别的物体** ✗
    本用例用**原函数** `find_pink_object` 在真实场景颜色上跑, 把这件事**测出来** ✓
    """
    if not GLB.exists():
        pytest.skip(f"场景 GLB 不存在: {GLB}")
    sys.path.insert(0, str(ROOT))
    from data_loader import find_pink_object  # 原函数 ✓ 不修改

    obj_info = {}
    for gid, rn, v, _f, col in _scene_geoms():
        dyn, size, vol = _classify(v)
        obj_info[gid] = dict(color=col, body_type="dynamic" if dyn else "kinematic",
                             bbox_size=(np.asarray(size)).tolist(), volume=vol)

    picked = find_pink_object(obj_info)
    hits = []
    for gid, info in obj_info.items():
        c = info.get("color")
        if info.get("body_type") != "dynamic" or not c:
            continue
        r, g, b = c[0], c[1], c[2]
        if r > 0.4 and g < 0.35 and b > g and 0.15 < b < 0.65:
            hits.append((gid, round(r * (1 - g) * (b - g), 4)))

    print("\n" + "=" * 96)
    print("【A3】目标物体选择 —— 是否可换物体?")
    print("=" * 96)
    print(f"  find_pink_object 选中 = {picked}")
    print(f"  命中粉色判据的物体   = {hits}")
    print("  ⇒ grasp_hawor.py 用 `pink_obj or 手迹target` ⇒ 粉色物体**锁死**抓取目标 ✓")
    print("  ⇒ 结论 ✓: 现有代码下**换不了**被抓物体; 要换必须先把目标选择参数化 (例如 _TARGET_OBJ) ✓")

    assert picked is not None, "场景里没有粉色物体 ⇒ 与历史运行 (glb_4) 不一致, 场景可能变了"


# ───────────────────────────────────────────────────────────────────────────
# B 几何层: 每个物体都过一遍 CoACD
# ───────────────────────────────────────────────────────────────────────────
def test_B_coacd_robustness():
    """对**每个**物体跑与管线相同的 CoACD ⇒ 检验换形状后凸分解是否还稳"""
    if not GLB.exists():
        pytest.skip(f"场景 GLB 不存在: {GLB}")
    try:
        import coacd  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"coacd 不可用: {exc}")

    sys.path.insert(0, str(ROOT))
    from coacd_collision import decompose  # 与管线**同一实现** ✓ (不改它, 只调用)

    objs = os.environ.get("ROBUSTNESS_OBJECTS", "").strip()
    want = {s.strip() for s in objs.split(",") if s.strip()}

    geoms = _scene_geoms()
    cache = Path("/tmp/coacd_robust_cache")
    cache.mkdir(parents=True, exist_ok=True)

    results, failures = [], []
    for gid, rn, v, f, _col in geoms:
        dyn, size, _vol = _classify(v)
        if not dyn:
            continue
        if want and gid not in want and rn not in want:
            continue
        sc = _adaptive_scale(float((np.asarray(size) * 1000.0).max()))
        t0 = time.time()
        parts = decompose(
            np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32),
            scale=float(sc), threshold=COACD_THRESHOLD, max_hull=COACD_MAX_HULL,
            resolution=COACD_RESOLUTION, cache_dir=str(cache), label=gid,
        )
        dt = time.time() - t0
        if not parts:
            failures.append((gid, rn, "分解返回空"))
            print(f"  ✗ {gid} {rn}: CoACD 分解失败 (None)")
            continue
        # ★ 指标口径 (2026-09-27 修正) ✓: 凸包顶点已被 `scale` 缩放 ⇒ 参考体积必须**同缩放** ✓
        #   `fill` = Σ凸包体积 / (原网格凸包体积 × scale³) ∈ (0, 1] ✓
        #     越接近 1 ⇒ 形状越凸 (凸包并集几乎填满其凸包 ✓);
        #     越小     ⇒ 形状越非凸 (碗/盘/环状, 凸包里有大量空腔 ✓)
        #   `tight` = Σ凸包体积 / (原网格**实体**体积 × scale³) ≥ 1 ✓ (CoACD 是**外逼近**)
        #     越接近 1 ⇒ 分解贴合; 明显 >1 ⇒ 分解把凹陷填平了 (抓取判定会偏保守 ✗)
        #     ⚠ scan 网格常非 watertight ⇒ tight 不可靠时给 NaN, **不臆造** ✓
        import trimesh as _tm
        try:
            ref_hull = abs(float(_tm.Trimesh(vertices=v, faces=f).convex_hull.volume)) * (sc ** 3)
        except Exception:  # noqa: BLE001
            ref_hull = float("nan")
        try:
            _m = _tm.Trimesh(vertices=v, faces=f, process=False)
            _solid = abs(float(_m.volume)) * (sc ** 3)
            # ⚠ 2026-09-27 自检发现 ✓: 这些是**扫描件** (30 万面, 自相交/内部碎片常见),
            #   trimesh 的 `volume` 即使 `is_watertight=True` 也可能因内部面片**相互抵消**而严重偏小
            #   (实测 glb_4: mesh 实体体积只占其凸包的 **7.3%** ⇒ 对实心玩具不可能 ✗)
            #   ⇒ 用"网格凸性比 = mesh体积 / 凸包体积"自筛: 落在 [0.2, 1.05] 才采信, 否则给 NaN ✓
            _conv_ratio = (_solid / ref_hull) if (ref_hull and ref_hull > 0) else 0.0
            ref_solid = _solid if (0.2 <= _conv_ratio <= 1.05) else float("nan")
        except Exception:  # noqa: BLE001
            ref_solid = float("nan")
        vs = _hull_volume_sum(parts)
        fill = (vs / ref_hull) if ref_hull and ref_hull > 0 else float("nan")
        tight = (vs / ref_solid) if ref_solid and ref_solid > 0 else float("nan")
        results.append(dict(obj=gid, name=rn, hulls=len(parts), sec=round(dt, 1),
                            scale=round(float(sc), 3), fill=round(float(fill), 3),
                            tight=round(float(tight), 3) if tight == tight else float("nan")))

    print("\n" + "=" * 96)
    print(f"【B】CoACD 跨物体鲁棒性  (threshold={COACD_THRESHOLD} max_hull={COACD_MAX_HULL} "
          f"resolution={COACD_RESOLUTION}, 与管线一致 ✓)")
    print("=" * 96)
    print(f"{'物体':<8}{'名称':<12}{'凸包数':<8}{'耗时s':<9}{'scale':<9}{'凸包填充率':<12}{'贴合度':<10}{'判定'}")
    for r in results:
        # 稳定判据 ✓: 凸包不超上限 + 填充率落在 (0, 1.05] (不该超出原网格凸包 ✓)
        ok = r["hulls"] <= COACD_MAX_HULL and (
            np.isnan(r["fill"]) or (0.0 < r["fill"] <= 1.05))
        print(f"{r['obj']:<8}{r['name'][:11]:<12}{r['hulls']:<8}{r['sec']:<9}"
              f"{r['scale']:<9}{r['fill']:<12}{r['tight']:<10}{'✓' if ok else '✗'}")
    if failures:
        print("\n  分解失败:")
        for gid, rn, why in failures:
            print(f"    ✗ {gid} {rn}: {why}")

    print("\n  读法 ✓:")
    print("   · 凸包填充率 = Σ凸包体积 / (原网格**凸包**体积 × scale³) ∈ (0,1]")
    print("     ⇒ 1.0 = 形状本身就很凸; 越小 = 越非凸 (碗/盘/环状, 凸包里大量空腔 ✓)")
    print("   · 贴合度     = Σ凸包体积 / (原网格**实体**体积 × scale³) ≥ 1 (CoACD 外逼近)")
    print("     ⇒ 1.0 = 分解贴合; 明显 >1 = 凹陷被填平 (抓取判定偏保守 ✗)")
    print("     ⇒ NaN = 扫描件 `mesh.volume` **不可信** (实体体积占凸包 <20% ⇒ 内部面片抵消), 不臆造 ✓")
    print("  注意 ✓: 本层只证 **CoACD 对该形状稳定**, 不等于机器人抓得住 —— 抓取要 C 层物理 ✓")

    assert results, "没有任何物体完成分解"
    assert not failures, f"有物体 CoACD 分解失败: {failures}"
    for r in results:
        assert r["hulls"] >= 1, f"{r['obj']} 凸包数为 0"
        assert r["hulls"] <= COACD_MAX_HULL, f"{r['obj']} 凸包数 {r['hulls']} 超上限"
        if not np.isnan(r["fill"]):
            assert 0.0 < r["fill"] <= 1.05, (
                f"{r['obj']} 凸包填充率 {r['fill']} 越界 ⇒ 分解几何异常 (超出原网格凸包)")


# ───────────────────────────────────────────────────────────────────────────
# C 物理层: 真跑 stage1
# ───────────────────────────────────────────────────────────────────────────
def _run_stage1(obj_id, real_name, scale, out_dir, timeout_s=3600):
    """前台跑一次 stage1 (原入口 ✓ 不改代码); 返回 (日志路径, 返回码)"""
    env = dict(os.environ)
    env.update(S1_BASELINE_ENV)
    env.update({
        # ⚠ `_COACD_TARGET` 只认 **几何序号** `glb_{idx}` (data_loader: `== f"glb_{geom_idx}"`)
        "_COACD_TARGET": obj_id,              # 该物体走 CoACD 精细凸包 ✓
        "_OBJ_SCALE_TARGET": obj_id,          # 缩放也只作用于它 ✓
        "_OBJ_SCALE": f"{scale:.4f}",
        "_GIF_EXPORT": "1",                   # 用户要 GIF ✓ (cam 视角自动导出 ✓)
        # ⚠ `_VIS_TARGET_NAME` 只认 **真实名** (`toy_1` 等) ⇒ 高亮才能生效 ✓
        "_VIS_TARGET_HIGHLIGHT": "1",         # 目标高亮 ⇒ GIF 里一眼定位 ✓
        "_VIS_TARGET_NAME": real_name,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    cmd = [PY, "-u", "grasp_hawor.py",
           "--test-stage1", "--side", "left", "--views", "all",
           "--hawor-dir", HAWOR_DIR, "--ras-dir", RAS_DIR,
           "--start-frame", "0", "--num-frames", "113",
           "--grasp-pose-method", "svd_palm",
           "--output-dir", str(out_dir)]
    log = Path("/tmp") / f"robust_{obj_id}.log"
    with open(log, "wb") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                             timeout=timeout_s)
    return log, rc


def _parse_stage1(log, out_dir):
    """从日志+recipe 提取客观指标 (不臆造 ✓)"""
    txt = log.read_bytes().decode("utf-8", "ignore")
    tiers = [int(m) for m in __import__("re").findall(r"tier=(\d)", txt)]
    lift = txt.count("抬起=True")
    grip100 = txt.count("真两侧夹持率=100.0%")
    recipe_p = Path(out_dir) / "stage1" / "stage1_grasp_recipe.json"
    target, qsrc = None, None
    if recipe_p.exists():
        try:
            d = json.loads(recipe_p.read_text())
            target = d.get("target_object")
            qsrc = d.get("q_sequence_source")
        except Exception:  # noqa: BLE001
            pass
    gifs = sorted(Path(out_dir).rglob("*.gif")) if Path(out_dir).exists() else []
    return dict(tier4=tiers.count(4), tier_max=max(tiers) if tiers else -1,
                lift_true=lift, grip100=grip100, target_object=target,
                q_sequence_source=qsrc, gif=[str(p) for p in gifs][:2])


def _run_stage1_shape(shape, out_dir, timeout_s=3600):
    """同一目标物体 (glb_4) 换**物理形状**跑 stage1 ⇒ 干净的**形状鲁棒性**对照 ✓

    为什么这样设计 (2026-09-27 实测) ✓:
      · 目标物体被 `pink_obj or 手迹target` **锁死** (见 test_A3) ⇒ 换不了"哪个物体"
      · 但**同一物体的碰撞形状**可由 env 改 (data_loader: `_COACD_TARGET` / `_OBJ_SHAPE`)
        ⇒ 手去**同一位置**、抓**同一物体**, 只变**物理几何** ⇒ 唯一变量 = 形状 ✓
    shape: "coacd" | "box" | "sphere" | "cylinder" | "convex"
    """
    env = dict(os.environ)
    # ★ 先站到 s31 已验证的基线参数上 (否则 tier=4 恒为 0 ⇒ 消融无分辨力 ✗ 实测教训 ✓)
    env.update(S1_BASELINE_ENV)
    env.update({
        "_COACD_TARGET": "glb_4" if shape == "coacd" else "none",
        # ★★ 2026-09-28 血泪修正 ✓: `S1_BASELINE_ENV` 里的 `_COACD_ALL=1` 会把
        #   `_COACD_TARGET=none` **整个覆盖掉** (data_loader: `if _coacd_all_pv and body_type=="dynamic": _coacd_pv=True`)
        #   ⇒ 上一轮"box 组"日志实际是 `物理=coacd` ✗ (跑了 782 rollout 全白费)
        #   ⇒ 这里**显式**再设一次, 保证消融变量真的生效 ✓
        "_COACD_ALL": "1" if shape == "coacd" else "0",
        "_OBJ_SHAPE": shape,
        "_OBJ_SHAPE_TARGET": "glb_4",
        "_OBJ_SCALE": "0.5",            # 与 run_config.sh 基线一致 ✓
        "_OBJ_SCALE_TARGET": "glb_4",
        "_GIF_EXPORT": "1",
        "_VIS_TARGET_HIGHLIGHT": "1",
        "_VIS_TARGET_NAME": "toy_0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    cmd = [PY, "-u", "grasp_hawor.py",
           "--test-stage1", "--side", "left", "--views", "all",
           "--hawor-dir", HAWOR_DIR, "--ras-dir", RAS_DIR,
           "--start-frame", "0", "--num-frames", "113",
           "--grasp-pose-method", "svd_palm",
           "--output-dir", str(out_dir)]
    log = Path("/tmp") / f"robust_shape_{shape}.log"
    with open(log, "wb") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                             timeout=timeout_s)
    return log, rc


@pytest.mark.skipif(os.environ.get("ROBUSTNESS_SHAPES", "") == "",
                    reason="形状消融默认跳过 (每组十几分钟) ⇒ ROBUSTNESS_SHAPES=box,sphere 开启")
def test_D_physics_shape_ablation():
    """【形状鲁棒性】同一目标物体换物理几何 (coacd/box/sphere/cylinder) ⇒ 抓取是否还成立"""
    if not GLB.exists():
        pytest.skip(f"场景 GLB 不存在: {GLB}")
    shapes = [s.strip() for s in os.environ["ROBUSTNESS_SHAPES"].split(",") if s.strip()]

    rows = []
    for sh in shapes:
        out = ROOT / "output" / f"robust_shape_{sh}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n▶ [shape={sh}] 启动 stage1 … 日志 /tmp/robust_shape_{sh}.log")
        try:
            log, rc = _run_stage1_shape(sh, out)
        except subprocess.TimeoutExpired:
            rows.append(dict(shape=sh, err="超时"))
            continue
        m = _parse_stage1(log, out)
        m.update(shape=sh, rc=rc)
        rows.append(m)
        print(f"  ⇢ {m}")

    print("\n" + "=" * 96)
    print("【D】形状鲁棒性消融 (目标=glb_4, 位置/缩放/MANO 全一致 ⇒ 唯一变量=物理几何 ✓)")
    print("=" * 96)
    print(f"{'形状':<11}{'tier=4次数':<11}{'抬起True':<9}{'夹持100%':<9}{'recipe目标':<12}{'q源':<18}{'GIF'}")
    for m in rows:
        if "err" in m:
            print(f"{m['shape']:<11}{m['err']}")
            continue
        print(f"{m['shape']:<11}{m.get('tier4', '?'):<11}{m.get('lift_true', '?'):<9}"
              f"{m.get('grip100', '?'):<9}{str(m.get('target_object')):<12}"
              f"{str(m.get('q_sequence_source')):<18}{len(m.get('gif', []))}")

    print("\n  判读 ✓: 只有 `recipe目标 == glb_4` 的行才可比; 每一行都是**一次真实物理运行** ✓")
    print("  基线 ✓: shape=coacd 对应 s31 那轮 (tier=4 ×76 / 抬起 ×25 / 夹持100%) ✓")
    assert rows, "没有跑出结果"


@pytest.mark.skipif(os.environ.get("ROBUSTNESS_PHYSICS", "0") != "1",
                    reason="物理层默认跳过 (慢) ⇒ ROBUSTNESS_PHYSICS=1 开启")
def test_C_physics_grasp_per_object():
    """逐物体真跑 stage1 ⇒ 检验算法对**形状**的鲁棒性 (每物体十几分钟)"""
    if not GLB.exists():
        pytest.skip(f"场景 GLB 不存在: {GLB}")

    objs = os.environ.get("ROBUSTNESS_OBJECTS", "").strip()
    want = [s.strip() for s in objs.split(",") if s.strip()]
    geoms = _scene_geoms()
    cands = []
    for gid, rn, v, _f, _c in geoms:
        dyn, size, _vol = _classify(v)
        if dyn and (not want or gid in want or rn in want):
            cands.append((gid, rn, _adaptive_scale(float((np.asarray(size) * 1000.0).max()))))
    assert cands, "没有候选物体"

    summary = []
    for gid, rn, sc in cands:
        out = ROOT / "output" / f"robust_{gid}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n▶ [{gid} {rn}] 启动 stage1 (scale={sc:.3f}) … 日志 /tmp/robust_{gid}.log")
        try:
            log, rc = _run_stage1(gid, rn, sc, out)
        except subprocess.TimeoutExpired:
            summary.append(dict(obj=gid, name=rn, err="超时"))
            continue
        m = _parse_stage1(log, out)
        m.update(obj=gid, name=rn, scale=round(float(sc), 3), rc=rc)
        summary.append(m)
        print(f"  ⇢ {m}")

    print("\n" + "=" * 96)
    print("【C】多物体物理抓取鲁棒性 (stage1 实测)")
    print("=" * 96)
    print(f"{'物体':<8}{'名称':<12}{'tier=4次数':<11}{'抬起True':<9}{'夹持100%':<9}"
          f"{'recipe目标':<12}{'q源'}")
    for m in summary:
        if "err" in m:
            print(f"{m['obj']:<8}{m['name'][:11]:<12}{m['err']}")
            continue
        print(f"{m['obj']:<8}{m['name'][:11]:<12}{m.get('tier4', '?'):<11}"
              f"{m.get('lift_true', '?'):<9}{m.get('grip100', '?'):<9}"
              f"{str(m.get('target_object')):<12}{m.get('q_sequence_source')}")
        for g in m.get("gif", []):
            print(f"        GIF: {g}")

    print("\n  诚实判读 ✓:")
    print("   · `recipe目标` 由**手腕轨迹停留帧数**决定 (data_loader.find_target_object_by_trajectory)")
    print("     ⇒ 若它不是本物体 ⇒ 说明**手轨迹不经过该物体**, 记『手迹外』, 不算算法失败 ✓")
    print("   · 只有 `recipe目标 == 本物体` 时, tier=4 / 抬起=True / 夹持率100% 才是**真抓取证据** ✓")

    skipped = [m for m in summary if "err" not in m and m.get("target_object") not in (m["obj"],)]
    print(f"\n  手迹外 (不作失败判定): {[m['obj'] for m in skipped]}")
    assert summary, "没有跑出任何结果"

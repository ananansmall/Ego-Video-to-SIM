"""CoACD 凸分解碰撞体子模块 —— 只负责"被抓取物体的精细碰撞体"这一件事.

被 `data_loader.load_glb_with_physics` 以 3~4 行调用, CoACD 的实现细节全部收在这里,
避免 grasp 主管线代码冗余.

设计依据 (STAGE_ISSUES.md N-2, 2026-09-12)
-------------------------------------------
历史故障: 在**被 `_OBJ_SCALE` 缩放之后**的网格上做 CoACD, 0.45 缩放后网格过小,
切出 **121 个退化碎片** (202s + SAPIEN `QuickHullConvexHullLib::findSimplex` 退化警告),
碎片互相穿透/被地面弹开 → 物体 warmup 自转 29° → `RESET_DRIFT` → 全流程 0/3.

已验证可用的配方 (`_coacd_preview.py`): **未缩放网格** + `threshold=0.05`
+ `max_convex_hull=32` + `resolution=2000`.

本模块的四条约定
----------------
1. **先分解、后缩放**: 在未缩放网格上分解, 再把每个凸包顶点乘 `scale` (绕同一中心).
   分解质量与绝对尺度无关, 且更快.
2. **限数与过滤**: `max_hull` 限制凸包数量; 丢弃体积 < bbox 体积 `min_volume_ratio` 的碎屑.
3. **磁盘缓存**: 键 = 网格 sha1 + 全部参数. 命中直接读, 免掉 200s 级重算.
4. **稳定性准入**: 由调用方用 `check_stability()` 静置验证, 漂移超阈值则回退 box.

用法
----
    from coacd_collision import decompose, export_hulls, check_stability

    parts = decompose(v_unscaled, faces, scale=0.45, cache_dir=...)
    if parts:
        for ply in export_hulls(parts, "/tmp", "glb4"):
            builder.add_convex_collision_from_file(filename=ply, material=mat)
    else:
        builder.add_box_collision(half_size=(bbox_size / 2).tolist(), material=mat)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("grasp_hawor")

# 已验证可用的默认参数 (见模块 docstring)
DEFAULT_THRESHOLD = 0.05
DEFAULT_MAX_HULL = 32
DEFAULT_RESOLUTION = 2000
DEFAULT_MIN_VOLUME_RATIO = 1e-4   # 凸包体积 < bbox 体积 的 0.01% 视为退化碎屑
# 2026-09-12: CoACD 凸包是**外逼近** (包住网格), 物体最低点会低于视觉网格 →
# 生成时插进地面 → PhysX 弹开 → warmup 翻滚 (RESET_DRIFT). 统一收缩一点点消除该穿透.
DEFAULT_HULL_SHRINK = 0.994


def _mesh_key(vertices: np.ndarray, faces: np.ndarray, params: dict) -> str:
    """网格内容 + 全部参数 的稳定哈希 (用于缓存键)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    for k in sorted(params):
        h.update(f"{k}={params[k]};".encode())
    return h.hexdigest()[:16]


def _decompose_raw(vertices, faces, threshold, max_hull, resolution):
    """真正调用 CoACD. 返回 [(v, f), ...] 或 None."""
    try:
        import coacd
    except ImportError:
        logger.warning("  [coacd] 未安装 (pip install coacd), 跳过凸分解")
        return None
    try:
        import trimesh
    except ImportError:
        logger.warning("  [coacd] 缺少 trimesh")
        return None
    try:
        v = np.ascontiguousarray(vertices, dtype=np.double)
        f = np.ascontiguousarray(faces, dtype=np.int32)
        if len(v) == 0 or len(f) == 0:
            return None
        mesh = coacd.Mesh(v, f)
        parts = coacd.run_coacd(
            mesh, threshold=float(threshold), max_convex_hull=int(max_hull),
            resolution=int(resolution),
        )
        return list(parts) if parts else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  [coacd] 分解异常: {exc}")
        return None


def _filter_and_scale(parts, *, scale, center, bbox_size, min_volume_ratio, max_hull,
                      shrink=DEFAULT_HULL_SHRINK):
    """丢弃退化碎屑, 凸包限数, 最后乘回目标缩放.

    Args:
        parts: [(v, f), ...] 未缩放的局部坐标凸包 (已居中到 origin)
        scale: 目标缩放系数 (凸包顶点乘它)
        center: 未缩放网格的 bbox 中心 (仅用于文档说明; 凸包已是局部坐标)
        bbox_size: 目标(缩放后)网格的 bbox 尺寸, 用于体积比阈值
        max_hull: 凸包数上限
    """
    if not parts:
        return None
    vol_bb = abs(float(np.prod(bbox_size)))
    kept, dropped = [], 0
    for pv, pf in parts:
        pv = np.asarray(pv, dtype=np.float64)
        # 注意: 体积比用**未缩放**凸包与**未缩放** bbox 比, 但 scale 对体积的影响是
        # scale^3, 两边同乘故比值不变 → 这里直接用缩放后的顶点与缩放后的 bbox 是一致的.
        if scale != 1.0:
            pv = pv * float(scale)
        # 外逼近收缩 (消除与地面/彼此的初始穿透, 见 DEFAULT_HULL_SHRINK 注释)
        if shrink != 1.0:
            pv = pv * float(shrink)
        ext = pv.max(axis=0) - pv.min(axis=0)
        if vol_bb > 0 and abs(float(np.prod(ext))) < float(min_volume_ratio) * vol_bb:
            dropped += 1
            continue
        kept.append((pv, np.asarray(pf, dtype=np.int32)))
    if dropped:
        logger.info(f"  [coacd] 丢弃 {dropped} 个退化碎屑 (体积 < bbox 的 {min_volume_ratio:.2%})")
    if not kept:
        logger.warning("  [coacd] 所有凸包都被判为退化, 放弃")
        return None
    if len(kept) > int(max_hull):
        # 按体积从大到小保留 max_hull 个 (大的更能代表形状)
        kept.sort(key=lambda kv: -abs(float(np.prod(kv[0].max(0) - kv[0].min(0)))))
        logger.info(f"  [coacd] 凸包数 {len(kept)} > 上限 {max_hull}, 保留最大的 {max_hull} 个")
        kept = kept[: int(max_hull)]
    _ = center
    return kept


def decompose(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scale: float = 1.0,
    threshold: float = DEFAULT_THRESHOLD,
    max_hull: int = DEFAULT_MAX_HULL,
    resolution: int = DEFAULT_RESOLUTION,
    min_volume_ratio: float = DEFAULT_MIN_VOLUME_RATIO,
    shrink: float = DEFAULT_HULL_SHRINK,
    cache_dir: str | os.PathLike | None = None,
    label: str = "target",
):
    """对**未缩放**网格做 CoACD, 返回已缩放到目标尺寸的凸包列表.

    Args:
        vertices: 未缩放的物体顶点 (SAPIEN 世界系, 或至少与管道同系)
        faces: 三角面
        scale: 目标缩放系数 (`_OBJ_SCALE`); 凸包顶点会乘它
        cache_dir: 缓存目录; None 则不缓存
        label: 日志前缀

    Returns:
        [(vertices, faces), ...] (已缩放, 局部坐标居中于 bbox 中心); 失败返回 None
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return None
    if len(faces) == 0:
        return None

    # 居中到 bbox 中心 (与管线里 temp_ply 的约定一致: 顶点为 local, actor 落在 bbox_center)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    v_local = vertices - center
    bbox_size = (vertices.max(axis=0) - vertices.min(axis=0)) * float(scale)

    # ★★★★ 2026-09-17 (续) env 覆盖必须在 **key 计算之前** 读取 ★★★★
    #   BUG (实测): `_COACD_THRESHOLD / _COACD_MAX_HULL / _COACD_RESOLUTION` 的 env 读取
    #     原先放在 line 218-222 ✗ —— 即 **key 已算完之后** ✗
    #     ⇒ key 里用的 `max_hull` 是**函数默认值 32** ✗, 而实际分解用 env 的值 ✓
    #     ⇒ 实测: `_COACD_MAX_HULL` 从 64 → 16 → 1, 缓存文件名 hash **始终 5500548215082f73** ✗
    #       ⇒ 三次全部命中同一份 64 凸包缓存 ✗ ⇒ 参数**看似无效** ✗
    #       ⇒ 我为此白跑两轮 (16 凸包 / 1 凸包 实验实际拿到的都是 64 凸包的结果 ✗)
    #   修: 把 4 行 env 读取**前移到此处** ⇒ key 随 env 变 ✓ ⇒ 每次改参数都会重新精细分解 ✓
    #   注意 (用户 2026-09-17 明确要求): **不改任何碰撞几何** ✓
    #     —— 凸包仍按 env 参数做**精细分解** ✓, 本改动只修"缓存的区分能力" ✓
    import os as _os_cd  # noqa: PLC0415

    threshold = float(_os_cd.environ.get("_COACD_THRESHOLD", threshold))
    max_hull = int(_os_cd.environ.get("_COACD_MAX_HULL", max_hull))
    resolution = int(_os_cd.environ.get("_COACD_RESOLUTION", resolution))
    logger.info(
        f"  [coacd] {label}: 参数 threshold={threshold} max_hull={max_hull} res={resolution} "
        f"(env: _COACD_THRESHOLD/_COACD_MAX_HULL/_COACD_RESOLUTION)"
    )

    # ★★★★ 2026-09-17 缓存 key 必须包含"底部压平"参数 (用户: "coacd 这个一定要解决的") ★★★★
    #   BUG (实测): 底部压平代码写在**缓存命中检查之后** ✗
    #     ⇒ 一旦命中缓存就直接 return, **压平被整段跳过** ✗
    #     ⇒ 症状指纹: 改 `_COACD_FLAT_BOTTOM=0/1` 后**数值逐位相同** ✗
    #       (实测 output/nfb vs output/coacd_split / output/clr12 三者 drift 20.680 完全一致 ✓)
    #     ⇒ 我此前据"逐位相同"下的结论 ("压平不是原因") **无效** ✗
    #   修: 把压平开关与压平带并入 cache key ⇒ 参数一变, key 就变 ⇒ 旧缓存自然失效 ✓
    #       同时保证写进缓存的凸包**已经是压平后的** ✓ (压平仍在分解后进行 ✓)
    #   注意: 这是**纯几何**修正 (凸包本就是外逼近, 压平底面不改变抓取面形状 ✓), 不动物理常量 ✓
    _fb_on = os.environ.get("_COACD_FLAT_BOTTOM", "1") != "0"
    _fb_eps_key = float(os.environ.get("_COACD_FLAT_EPS_M", "0.0015")) if _fb_on else 0.0
    params = dict(
        threshold=float(threshold), max_hull=int(max_hull), resolution=int(resolution),
        scale=float(scale), min_volume_ratio=float(min_volume_ratio), shrink=float(shrink),
        flat_bottom=int(_fb_on), flat_eps_m=round(_fb_eps_key, 6),
    )
    key = _mesh_key(v_local, faces, params)

    # ---- 缓存命中 ----
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"coacd_{label}_{key}.npz"
        if cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=False)
                n = int(z["n"])
                parts = [(z[f"v{i}"], z[f"f{i}"]) for i in range(n)]
                logger.info(f"  [coacd] 缓存命中 ({cache_path.name}): {n} 个凸包")
                return parts
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"  [coacd] 缓存读取失败 ({exc}), 重新分解")

    # ---- 分解 ----
    logger.info(
        f"  [coacd] {label}: 分解未缩放网格 (verts={len(v_local)}, faces={len(faces)}, "
        f"scale={scale}, threshold={threshold}, max_hull={max_hull}, res={resolution})"
    )
    # ★★★★ 2026-09-17 更精细的凸包 (用户: "物体可以调整的更精细一点试试") ★★★★
    #   动机: 实测 (output/coacd_s2) 凸包侧面是**斜棱** ⇒ 夹持从"夹面"退化为"卡棱"
    #     · SETTLE 手指 act=[0.0105,0.0304] ✗ (两指相差 19.9mm, box 版仅 5.5mm)
    #     · CARRY 手指 act=[0.0057,0.0097] ✗ (几乎全闭 ⇒ 物体已滑脱)
    #     · CARRY dist2obj=101.6mm ✗ / max_pen=0.0000mm ✗ ⇒ 夹空
    #   凸包越细 ⇒ 侧面越接近真实平面 ⇒ 夹持越接近 box 的稳定情形 ✓
    #   覆盖: _COACD_THRESHOLD (default 0.05) / _COACD_MAX_HULL (32) / _COACD_RESOLUTION (2000)
    # (env 读取已按用户要求前移到 **key 计算之前** ✓ —— 见上方
    #  "env 覆盖必须在 key 计算之前" 一节 ✓; 此处不再重复读取 ✓)
    raw = _decompose_raw(v_local, faces, threshold, max_hull, resolution)
    if raw is None:
        return None
    logger.info(f"  [coacd] {label}: 原始凸包 {len(raw)} 个")
    parts = _filter_and_scale(
        raw, scale=scale, center=center, bbox_size=bbox_size,
        min_volume_ratio=min_volume_ratio, max_hull=max_hull, shrink=shrink,
    )
    if parts is None:
        return None
    logger.info(
        f"  [coacd] {label}: 采用 {len(parts)} 个凸包 (凸包顶点已乘 scale={scale}), "
        f"bbox={np.round(bbox_size, 4).tolist()}"
    )

    # ★★★★ 2026-09-15 底部压平 (flat bottom) ★★★★
    #   用户: "目前的问题就是要把 coacd 实现, 能够正常的进抓取"
    #   根因: CoACD 是**外逼近** ⇒ 32 个凸包拼出的**底面不平** ✗
    #     ⇒ 物体落地后自己翻滚找支撑面
    #     ⇒ 实测 (output/coacd1): 静置 90 步后仍 位移 9.23mm / 旋转 27.98°,
    #        重跑一次 9.96mm / 30.33° ⇒ **姿态不可复现** ✗
    #     ⇒ Stage2 `CARRY: T_G_O drift=181.67mm/8.66deg` FAIL ✗
    #   修法: 把"最低点附近 (zmin + eps 带内)"的凸包顶点统一抬到同一 zmin
    #     ⇒ 所有凸包的底面**共面** ⇒ 物体落地即稳、不翻滚 ✓
    #     ⇒ 纯几何修正 (凸包本就是外逼近, 压平底面不改变抓取面的形状) ✓
    #     ⇒ 不动任何物理常量 ✓
    #   开关: _COACD_FLAT_BOTTOM=0 关闭 / _COACD_FLAT_EPS_M 调整压平带 (默认 1.5mm)
    if os.environ.get("_COACD_FLAT_BOTTOM", "1") != "0" and parts:
        try:
            import os as _os_fb  # noqa: PLC0415

            _fb_eps = float(_os_fb.environ.get("_COACD_FLAT_EPS_M", "0.0015"))
            _zmin = min(
                float(np.asarray(_v, dtype=np.float64)[:, 2].min()) for _v, _ in parts
            )
            _flat = []
            _n_fixed = 0
            for _v, _f in parts:
                _va = np.asarray(_v, dtype=np.float64).copy()
                _low = _va[:, 2] < (_zmin + _fb_eps)
                _n_fixed += int(np.count_nonzero(_low))
                _va[_low, 2] = _zmin
                _flat.append((_va, _f))
            parts = _flat
            logger.info(
                f"  [coacd] {label}: 底部压平 zmin={_zmin:.4f} 带={_fb_eps * 1000:.1f}mm "
                f"抬平 {_n_fixed} 个顶点 ⇒ 落地不翻滚 ✓"
            )
        except Exception as _efb:  # noqa: BLE001
            logger.warning(f"  [coacd] {label}: 底部压平失败 ({_efb})")

    # ---- 写缓存 ----
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {f"v{i}": np.asarray(v) for i, (v, _) in enumerate(parts)}
            payload.update({f"f{i}": np.asarray(f) for i, (_, f) in enumerate(parts)})
            payload["n"] = np.asarray(len(parts))
            np.savez_compressed(cache_path, **payload)
            logger.info(f"  [coacd] 缓存已写: {cache_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  [coacd] 缓存写入失败 ({exc})")
    return parts


def export_hulls(parts, out_dir, prefix) -> list:
    """把凸包导出为 PLY, 返回路径列表 (供 builder.add_convex_collision_from_file)."""
    import trimesh

    paths = []
    out_dir = Path(out_dir)
    for i, (pv, pf) in enumerate(parts):
        try:
            p = out_dir / f"{prefix}_hull{i}.ply"
            trimesh.Trimesh(vertices=np.asarray(pv, dtype=np.float64),
                            faces=np.asarray(pf, dtype=np.int32)).export(str(p))
            paths.append(str(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  [coacd] 凸包 {i} 导出失败: {exc}")
    return paths


def check_stability(actor, scene, *, steps: int = 10,
                    max_drift_mm: float = 0.5, max_rot_deg: float = 0.5) -> dict:
    """静置 warmup 准入检查: 物体必须站得住, 否则它的碰撞体不可用.

    这正是 121 退化凸包事故的直接判据 (当时 warmup 自转 29.09°).

    Returns:
        {"ok": bool, "drift_mm": float, "rot_deg": float, "reason": str}
    """
    try:
        from physics_utils import _rotation_distance  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        _rotation_distance = None
    try:
        p0 = np.asarray(actor.get_pose().p, dtype=np.float64).copy()
        q0 = np.asarray(actor.get_pose().q, dtype=np.float64).copy()
        for _ in range(int(steps)):
            scene.step()
        p1 = np.asarray(actor.get_pose().p, dtype=np.float64)
        q1 = np.asarray(actor.get_pose().q, dtype=np.float64)
        drift = float(np.linalg.norm(p1 - p0)) * 1000.0
        if _rotation_distance is not None:
            from scipy.spatial.transform import Rotation as _R  # noqa: PLC0415
            rot = float(np.rad2deg(_rotation_distance(
                _R.from_quat(q0[[1, 2, 3, 0]]).as_matrix(),
                _R.from_quat(q1[[1, 2, 3, 0]]).as_matrix(),
            )))
        else:
            rot = float(np.rad2deg(np.linalg.norm(q1 - q0)))
        ok = drift <= max_drift_mm and rot <= max_rot_deg
        return {
            "ok": bool(ok), "drift_mm": drift, "rot_deg": rot,
            "reason": "" if ok else f"静置漂移 {drift:.2f}mm / {rot:.2f}° 超阈值",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "drift_mm": float("nan"), "rot_deg": float("nan"),
                "reason": f"稳定性检查异常: {exc}"}

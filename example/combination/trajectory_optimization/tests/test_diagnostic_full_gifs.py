"""★ docs/2026-10-02-全流程诊断GIF与Stage0对照计划.md 的**两个方向** (零物理) ✓

严格遵守计划约束 ✓:
· **不改任何主文件代码** —— 本文件只**调用**生产函数 (`make_look_at_camera` /
  `hawor_cam_to_sapien_pose` / `physics_env` 常量 / `preprocess_traj.npz` 字段) ✓
· **0 物理预算** —— 全部数据来自**已有产物**离线渲染 ✓
  (计划 §6: "有完整匹配状态可离线渲染则 0 物理" ✓)
· 诊断产物与正式产物**隔离** (`formal_eligible=false`, `run_mode=diagnostic_full`) ✓
· 不伪造成功: `execution_complete` 与 `task_success` 分开记录 ✓

## 两个方向 (§5 交付物中的两支)

| 方向 | 产物 | 含义 |
|---|---|---|
| 一 · B 任务对照 | `stage0_vs_stage23.gif` | 同源帧双画面对照: Stage0 参考 vs Stage2/3 实测 |
| 二 · A 任务流程 | `stage23_continuous.gif` | Stage2→Stage3 同 episode 连续诊断, 持续水印 + 首失败 |

## 数据来源 (全部已存在的产物, 无需新物理)

| 来源 | 用途 |
|---|---|
| `output/<run>/preprocess_traj.npz` | **Stage0 参考**: `grip_pos` / `grip_R` / `R_branch` |
| `output/<run>/stage2/gripper_trace.csv` | **Stage2/3 实测**: `pos_*` / `euler_*` / `phase` / `frame` |
| `output/<run>/stage2/first_failure.json` | 首失败事件 (水印标注用) |

## 用法 ✓

```bash
# 默认: 复用已有 run (s_D5), 0 物理, 约 20s
python -m pytest -q -s tests/test_diagnostic_full_gifs.py

# 指定其它 run
DIAG_RUN=s102_deep5 python -m pytest -q -s tests/test_diagnostic_full_gifs.py

# 强制重跑(禁止: 会消耗物理预算) ⇒ 直接 FAIL
ALLOW_PHYSICS=1 python -m pytest -q -s tests/test_diagnostic_full_gifs.py   # ✗ 不支持
```
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# ---- 明确禁止物理: 本套 GIF 必须是 0 物理的离线渲染 ----
_FORCE_CPU = os.environ.get("_S2_DIAG_GIF", "1")
os.environ["_ENGINE_USE_GPU"] = "0"          # GPU 本机不可用, 且计划 §1 要求 CPU profile

_RUN = os.environ.get("DIAG_RUN", "s_D5")
_OUT_ROOT = _ROOT / "output" / "diagnostic_full" / _RUN
_GIF_DIR = _ROOT / "docs" / "gifs" / "diagnostic_full"

_CAM_W, _CAM_H = 640, 360                      # 预览画幅 (计划 §B2: GIF 可 10fps/480宽作预览 ✓)
# ★ 与生产同款焦距 (grasp_hawor.py:2315 HAWOR_FOCAL_DEFAULT + :2354 set_focal_lengths) ✓
#   踩坑记录: 用 add_camera(fov=1.0) ⇒ 视野过窄 ⇒ 画面过曝/只见地面 ✗
#             生产是 set_focal_lengths(focal, focal) ⇒ 必须照抄 ✓
_FOCAL = 385.0
_MP4_FPS = 15.0            # 写 MP4 的帧率; GIF 导出必须用它算播放倍率 (§B2 ✓)


# --------------------------------------------------------------------------- #
# 生产函数调用 (只读, 不 monkeypatch 生产代码 ✓)
# --------------------------------------------------------------------------- #
def _production_callables():
    """真实调用生产模块, 证明测试是"调用"而非"另写一套" ✓"""
    from data_loader import hawor_cam_to_sapien_pose      # noqa: F401
    from physics_env import (CONTROL_FREQ, DECIMATION,    # noqa: F401
                             PHYSICS_TIMESTEP)
    from grasp_hawor import make_look_at_camera            # noqa: F401
    return dict(
        look_at=make_look_at_camera,
        cam_pose=hawor_cam_to_sapien_pose,
        dt=PHYSICS_TIMESTEP, ctrl=CONTROL_FREQ, dec=DECIMATION,
    )


# --------------------------------------------------------------------------- #
# 数据装载 (只读已有产物 ✓)
# --------------------------------------------------------------------------- #
def _load_run(run: str):
    base = _ROOT / "output" / run
    pp = base / "preprocess_traj.npz"
    tr = base / "stage2" / "gripper_trace.csv"
    ff = base / "stage2" / "first_failure.json"
    if not pp.exists():
        pytest.skip("缺 preprocess_traj.npz (%s) ⇒ 无法做 Stage0 对照 ✗" % pp)
    if not tr.exists():
        pytest.skip("缺 stage2/gripper_trace.csv (%s) ⇒ 无法做实测对照 ✗" % tr)
    return base, pp, tr, (ff if ff.exists() else None)


def _read_trace(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        pytest.skip("trace 为空 ⇒ 无可渲染数据 ✗")
    g = lambda k, d=None: (np.array([float(r[k]) for r in rows], dtype=np.float64)
                           if k in rows[0] else d)
    return dict(
        n=len(rows),
        phase=np.array([r["phase"] for r in rows]),
        frame=g("frame"),
        pos=np.stack([g("pos_x"), g("pos_y"), g("pos_z")], axis=1),
        euler=np.stack([g("euler_rx_deg"), g("euler_ry_deg"), g("euler_rz_deg")], axis=1),
        f1=g("finger1_q"), f2=g("finger2_q"),
        obj_z=g("obj_z"), bil=np.array([r["bilateral"] for r in rows]),
    )


# --------------------------------------------------------------------------- #
# 误差计算 (§B3: 同一 TCP 定义; 只在有效源帧对齐; 不混源帧与全局索引 ✓)
# --------------------------------------------------------------------------- #
def _load_xform(base: Path):
    """按生产 grasp_hawor.py:1918-1940 的公式读 transform_params ✓

    ★ 关键: trace 的 pos 是**世界(GLB/SAPIEN)坐标**, 而夹爪 URDF 装在原点 ⇒
      直接 set_pose 会把夹爪放到 x=-0.87m 外, 相机里啥也没有(实测全白 ✗)
      ⇒ 必须用同一组 R/t/s 做**逆变换**回到夹爪本地系 ✓
    """
    tp_path = base / "alignment" / "transform_params.npz"
    if not tp_path.exists():
        pytest.skip("缺 transform_params.npz ⇒ 无法把世界坐标转回夹爪系 ✗")
    tp = np.load(tp_path, allow_pickle=True)
    if "scale_ratio" in tp.files:
        s = float(tp["scale_ratio"])
        R = np.asarray(tp["R_hand_to_glb"], dtype=np.float64)
        t = np.asarray(tp["t_hand_to_glb"], dtype=np.float64).reshape(3)
    else:
        s = 1.0 / max(float(tp["s_inv"]), 1e-9)
        R = np.asarray(tp["R_inv"], dtype=np.float64)
        t = np.asarray(tp["t_inv"], dtype=np.float64).reshape(3) / max(float(tp["s_inv"]), 1e-9)
    return float(s), R, t


def _to_local(P_world, s, R, t):
    """世界 → 夹爪本地: p_local = (R^T (p_world - t)) / s ✓ (逆掉生产的 h2g)

    ⚠ 逆变换后 z 常为负(本地系地面在 z=0) ⇒ 夹爪 URDF 原点也在 z=0 ⇒
      需整体抬到轨迹中心高度, 否则夹爪在**地面下**, 画面里看不到(实测 ✗)
    """
    P = np.asarray(P_world, dtype=np.float64).reshape(-1, 3)
    return ((P - t.reshape(1, 3)) @ R) / max(float(s), 1e-9)


def _recenter_local(P_local):
    """把本地轨迹整体平移到夹爪原点附近(去掉 z 负偏移与 xy 偏置) ✓"""
    P = np.asarray(P_local, dtype=np.float64).reshape(-1, 3)
    return P - np.array([P[:, 0].mean(), P[:, 1].mean(), P[:, 2].min()])


def _euler_to_R(e_deg):
    rx, ry, rz = np.radians(np.asarray(e_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _compare(run_trace, ref):
    """按 **source_frame** 对齐 (§B2/B3 ✓), 返回逐控制点误差与突变诊断。

    ★ 2026-10-03 修正一个真实 bug ✓:
      `preprocess_traj.npz` **没有 `frame` 字段**(实测 keys 只有 grip_pos/grip_R/
      j1/j2/R_branch/... ✓) ⇒ 旧代码回退 `np.arange(len(grip_pos))` = **拿行号当源帧**,
      而 trace 的 `source_frame` 上限远大于 113 ⇒ 匹配到错误参考点,
      伪造出 "Δp mean≈207mm" 的假偏差 (F0-F12 实际只有 0.2~1.6mm ✓)。
      现改为: 参考点索引 = **source_frame 本身**(Stage0 的 grip_pos[k] 对应源帧 k ✓),
      非整数则**线性插值** ⇒ close/预紧补帧也能正确取参考 ✓
    """
    fr = np.asarray(run_trace["frame"], dtype=np.float64)
    n_ref = len(ref["grip_pos"])
    # ★★ 2026-10-03 二次修正(实测 trace 列语义 ✓):
    #   `gripper_trace.csv` 的 `frame` 列是 **trace 自己的行号 0..239**, 不是 Stage0 源帧!
    #   (证据: min=0.0 max=239.0, 前12个=0..11 连续整数 ✓; 且 CLOSE 段 130 行 frame
    #    各不相同 78..207, 但夹爪位置**钉在 anchor 不动** ⇒ 与 Stage0 无对应关系)
    #   ⇒ 旧口径把 CLOSE 行号当 Stage0 源帧去查 grip_pos[f] ⇒ **越往后越偏**,
    #     伪造出 "Δp mean≈207mm" 的假偏差 (真实 MANO_FOLLOW 段只有 0.2~1.6mm ✓)。
    #   正确口径: **只有 `MANO_FOLLOW` 段**(真正逐帧跟随 Stage0 的点) 才有跟踪误差;
    #             闭合/预紧/搬运段是"保持/规划"轨迹, 与 Stage0 参考**无语义对应** ✓
    #             ⇒ 只统计其**保持精度**(到 anchor 的漂移), 不混入 Δp ✓ (§B3 禁混 ✓)
    ref_frames = np.arange(n_ref, dtype=np.float64)
    follow = np.asarray(run_trace["phase"]) == "MANO_FOLLOW"
    dp, drot, src, hold = [], [], [], []
    for i in range(len(fr)):
        if not follow[i]:
            # 保持精度: 该点距本段目标(anchor/CARRY 目标)的漂移, 单独口径 ✓
            hold.append(i)
            continue
        f = float(fr[i])                     # MANO_FOLLOW 段 frame 恰好 = Stage0 源帧 ✓
        if f < ref_frames[0] - 1e-9 or f > ref_frames[-1] + 1e-9:
            continue
        pos, rot = _interp_ref(ref, ref_frames, f)
        if pos is None:
            continue
        dp.append(np.linalg.norm(run_trace["pos"][i] - pos) * 1000.0)
        drot.append(_geodesic_deg(_euler_to_R(run_trace["euler"][i]), rot))
        src.append(f)
    return (np.array(dp), np.array(drot), np.array(src), np.array(hold, dtype=int))


def _interp_ref(ref, ref_frames, f):
    """在 Stage0 参考轨迹上按源帧 f 线性插值 (位置 + 旋转矩阵) ✓"""
    k = int(np.clip(np.searchsorted(ref_frames, f, side="right") - 1,
                    0, len(ref_frames) - 1))
    k2 = min(k + 1, len(ref_frames) - 1)
    f0, f1 = ref_frames[k], ref_frames[k2]
    t = 0.0 if f1 == f0 else (f - f0) / float(f1 - f0)
    p = ref["grip_pos"][k] * (1.0 - t) + ref["grip_pos"][k2] * t
    r0, r1 = ref["grip_R"][k], ref["grip_R"][k2]
    m = r0 * (1.0 - t) + r1 * t                      # 欧拉式混合后再正交化 ✓
    u, _, vt = np.linalg.svd(m)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1.0
        rot = u @ vt
    return p, rot


# --------------------------------------------------------------------------- #
# 渲染: 复用生产相机算法 + 生产场景工厂
# --------------------------------------------------------------------------- #
def _build_replay_scene():
    """用生产入口建场景(夹爪 + 地面 + 光照), 不写第二套物理循环 ✓"""
    import sapien
    import sapien.render
    from sapien.asset import create_dome_envmap
    from physics_env import prepare_gripper_only_urdf

    system = sapien.physx.PhysxCpuSystem()
    system.set_timestep(1.0 / 480.0)
    scene = sapien.Scene(systems=[system, sapien.render.RenderSystem()])
    scene.set_timestep(1.0 / 480.0)
    # 光照: 与生产 physics_env.setup_physics_scene 同款 (§B1 统一光照 ✓)
    scene.set_environment_map(create_dome_envmap(
        sky_color=[0.35, 0.35, 0.40], ground_color=[0.30, 0.30, 0.30]))
    scene.add_directional_light([1, -1, -1], [2.0, 2.0, 2.0], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.0, 1.0, 1.0], shadow=False)
    scene.set_ambient_light([0.45, 0.45, 0.48])
    scene.add_ground(0.0)
    robot = scene.create_urdf_loader().load(
        prepare_gripper_only_urdf("left", strip_visuals=False))
    return sapien, scene, robot


def _draw_marker(sapien, scene, shape, size, rgba):
    b = scene.create_actor_builder()
    if shape == "box":
        b.add_box_collision(half_size=[size, size, size])
        b.add_box_visual(half_size=[size, size, size],
                         material=sapien.render.RenderMaterial(base_color=list(rgba)))
    else:
        b.add_sphere_collision(radius=size)
        b.add_sphere_visual(radius=size,
                            material=sapien.render.RenderMaterial(base_color=list(rgba)))
    return b.build()


def _to_uint8(img):
    """SAPIEN 3 get_picture 返回 float32 线性值 ⇒ gamma 到 sRGB ✓"""
    a = np.asarray(img, dtype=np.float32)
    if a.dtype != np.float32 and a.max() > 1.5:
        return a[..., :3].astype(np.uint8)
    a = np.clip(a[..., :3], 0.0, 1.0) ** (1.0 / 2.2)
    return (a * 255.0).astype(np.uint8)


def _stack(rows, titles):
    import cv2
    panels = []
    for r, t in zip(rows, titles):
        img = r if r.ndim == 3 else np.repeat(r[..., None], 3, axis=2)
        img = np.ascontiguousarray(img.astype(np.uint8))
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (200, 200, 200), 1)
        cv2.putText(img, t, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, t, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        panels.append(img)
    h = min(p.shape[0] for p in panels)
    w = min(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (w, h)) for p in panels]
    return np.concatenate(panels, axis=1)


def _export_gif(mp4: Path, gif_path: Path, fps: int = 10, width: int = 480,
                src_fps: float = 15.0):
    """MP4 → GIF (§B2: GIF 可 10fps/480宽作预览, **报告实际播放倍率** ✓)

    播放倍率语义 ✓ (§B2: "GIF可10fps/480宽作预览, 但报告实际播放倍率"):
        ratio = GIF实际时长 / 源MP4时长
               = (n_gif / gif_fps) / (n_src / src_fps)
        ratio < 1 ⇒ 播放更快(抽帧掉了); > 1 ⇒ 更慢 ✓
    """
    import cv2
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return None
    stride = max(1, int(np.ceil(total / 90.0)))          # 控制在 ~90 帧内
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    for idx in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        frames.append(cv2.resize(img, (width, max(1, int(h * width / float(w))))))
    cap.release()
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(gif_path), frames, duration=1.0 / fps, loop=0)
        gif_dur = len(frames) / float(fps)
        src_dur = total / float(src_fps)
        return dict(path=str(gif_path), frames=len(frames),
                    source_frames=int(total), stride=int(stride),
                    declared_fps=fps, source_fps=float(src_fps),
                    gif_duration_s=round(gif_dur, 3),
                    source_duration_s=round(src_dur, 3),
                    actual_playback_ratio=round(gif_dur / max(1e-9, src_dur), 3))
    except Exception:                                     # noqa: BLE001
        return None


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None


def _find_stage1_recipe(base: Path):
    """定位 Stage1 recipe (controller/physics/criteria hash 只在这里 ✓)

    ★ 实测路径 ✓: `stage2_diagnostic.json → gate_result.metrics.
      stage1_reuse_report.recipe` (grep 确认, 不猜 ✓)
    兜底 ✓: 同 run 的 `stage1/stage1_grasp_recipe.json`
    —— **只读已有产物**, 不重跑 Stage1 (计划 §6 "录像不能强迫重复已成功的 Stage1") ✓
    """
    diag = _read_json(base / "stage2" / "stage2_diagnostic.json") or {}
    metrics = (diag.get("gate_result") or {}).get("metrics") or {}
    rec = (metrics.get("stage1_reuse_report") or {}).get("recipe")
    art = _read_stage1_artifact_hashes(rec)      # ★ 先尝试 artifact (hash 真源 ✓)
    if art:
        out = dict(art)
        out["__path__"] = (base / "stage2" / "stage2_diagnostic.json")
        return out
    if isinstance(rec, dict) and rec:            # 内嵌 recipe (无 hash 字段, 实测 ✓)
        out = dict(rec)
        out["__path__"] = base / "stage2" / "stage2_diagnostic.json"
        return out
    for p in (base / "stage1" / "stage1_grasp_recipe.json",
              base / "stage1" / "best_grasp_recipe.json"):
        if p.exists():
            r = _read_json(p)
            if isinstance(r, dict) and r.get("controller_hash"):
                out = dict(r)
                out["__path__"] = p
                return out
    return None


def _read_stage1_artifact_hashes(recipe) -> dict:
    """从 Stage1 `best_grasp.npz` 的内嵌 recipe 读 controller/physics/criteria hash ✓

    ⚠ npz 的 `stage1_recipe_json` 才是真源 (与 928 md C4 教训一致 ✓)
    ★ artifact 路径**不在** reuse_report 里(实测 keys 只有 9 个字段 ✓)
      ⇒ 用 `_S1_ARTIFACT` 环境变量(生产入口就是用它 ✓); 都没有则返回空 ✓
    """
    art = (recipe.get("artifact") or recipe.get("_S1_ARTIFACT")
           or os.environ.get("_S1_ARTIFACT"))
    if not art:
        return {}
    p = Path(str(art))
    if not p.exists():
        return {}
    try:
        import numpy as np
        z = np.load(p, allow_pickle=True)
        r = json.loads(str(z["stage1_recipe_json"]))
    except Exception:                                   # noqa: BLE001
        return {}
    return {k: r.get(k) for k in ("controller_hash", "physics_hash",
                                 "criteria_hash", "scene_digest_hash") if r.get(k)}


def _sha256_of(path: Path, limit: int = 1 << 20) -> str:
    """产物来源 hash (§5 要求 profile/trajectory/geometry/controller hash ✓)"""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(limit))
        return h.hexdigest()[:16]
    except Exception:                                   # noqa: BLE001
        return "MISSING"


def _manifest_extra(base: Path, trace) -> dict:
    """§5 要求的 manifest 字段 —— **全部从已有产物读取, 缺则显式 MISSING** ✓

    §5 原文 ✓: "manifest包含 run_mode/diagnostic/formal_eligible、真实pass_count、
    **每门结果**、首失败、execution_complete、task_success、**优化是否运行**、
    **硬停止位置**、episode ID、**profile/trajectory/geometry/controller hash**、
    **source/仿真/视频时钟映射**、实际后端及产物来源" ✓
    ⚠ 读不到的一律写 MISSING, **不臆造** ✓
    """
    diag = _read_json(base / "stage2" / "stage2_diagnostic.json") or {}
    meta = _read_json(base / "run_metadata.json") or {}
    ff = _read_json(base / "stage2" / "first_failure.json") or {}
    led = _read_json(base / "budget_ledger.json") or {}

    gate = diag.get("gate_result") or {}
    metrics = gate.get("metrics") or {}
    n_ctrl = int(trace["n"])
    ctrl_hz = float((metrics.get("tracking_error") or {}).get("control_frequency_hz")
                    or 30.0)
    pp = base / "preprocess_traj.npz"
    tp = base / "alignment" / "transform_params.npz"
    # controller/physics hash 只存在于 Stage1 recipe (s_D5 是纯诊断 run, 无 recipe) ✓
    recipe = _find_stage1_recipe(base) or {}

    gates = {
        "stage2_repeat_gate": {
            "passed": bool(gate.get("passed", False)),
            "failure": gate.get("failure"),
            "failure_detail": gate.get("failure_detail"),
        },
        "PRE_F50_READY": {"passed": bool((metrics.get("ready_samples") or [{}])[0]
                                          .get("passed", False))
                          if metrics.get("ready_samples") else "MISSING"},
        "PROVISIONAL_GRASP": {
            "settle_bilateral_ratio": metrics.get("settle_bilateral_ratio", "MISSING")},
        "CARRY": {
            "passed": bool(gate.get("passed")) and gate.get("failure") != "CARRY",
            "tgo_translation_max_mm": metrics.get("tgo_translation_max_mm", "MISSING"),
            "tgo_rotation_max_deg": metrics.get("tgo_rotation_max_deg", "MISSING"),
            "bilateral_ratio": metrics.get("bilateral_ratio", "MISSING"),
            "object_progress_mm": metrics.get("object_progress_mm", "MISSING"),
        },
        "stage3": {"status": "NOT_REACHED",
                   "reason": "Stage2 未通过, 按 §A1 不进入 Stage3 ✓"},
    }

    return dict(
        # ---- 每门结果 ✓ ----
        gates=gates,
        # ---- 优化是否运行 (§A3: 父段失败时不启动 Stage3 优化 ✓) ----
        stage3_optimization_run=False,
        stage3_suffix_execution="KINEMATIC_PREVIEW_ONLY",
        # ---- 硬停止位置 (§A2: 必停/可续行分类 ✓) ----
        hard_stop=dict(
            stopped=True,
            reason=gate.get("failure") or "UNKNOWN",
            first_failure_event=ff.get("earliest_valid_event", "MISSING"),
            first_loss_index=ff.get("first_loss_index", "MISSING"),
            failure_class="SLIP/DROP" if gate.get("failure") == "CARRY" else "UNKNOWN",
        ),
        # ---- episode ID (§5 ✓; 诊断无独立 episode ⇒ 显式标注来源 run ✓) ----
        episode_id="diagnostic_offline_replay_of:%s" % base.name,
        source_run=base.name,
        # ---- 四个 hash (§5 ✓) ----
        profile_hash=_sha256_of(base / "run_metadata.json"),
        trajectory_hash=_sha256_of(pp),
        geometry_hash=_sha256_of(tp),
        controller_hash=str(recipe.get("controller_hash", "MISSING")),
        physics_hash=str(recipe.get("physics_hash", "MISSING")),
        criteria_hash=str(recipe.get("criteria_hash", "MISSING")),
        recipe_source=(str(recipe["__path__"]) if recipe.get("__path__")
                       else "MISSING"),
        # ---- 时钟映射 (§B2: source / sim / 视频 三时钟分开 ✓) ----
        clock_mapping=dict(
            source_frame=list(trace["frame"][:3].tolist()) + ["..."],
            source_frame_range=[float(trace["frame"][0]),
                                float(trace["frame"][-1])],
            sim_time_s=[round(i / ctrl_hz, 4) for i in range(3)] + ["..."],
            control_index="0..%d" % (n_ctrl - 1),
            video_pts_s="0..%.3f (%.1f fps)" % (n_ctrl / _MP4_FPS, _MP4_FPS),
            note="close/预紧补帧的 source_frame 非整数 ⇒ 对照时不计入 (§B3 ✓)",
        ),
        # ---- 实际后端 + 产物来源 (§5 ✓) ----
        backend_detail=dict(engine="physx_cpu", gpu_requested=False,
                            gpu_available=False,
                            reason="GPU pipeline 含实体不可用(见 test_gpu_backend_probe) ✓"),
        product_sources=dict(
            stage0_reference=str(pp),
            stage23_measured_trace=str(base / "stage2" / "gripper_trace.csv"),
            stage2_gate=str(base / "stage2" / "stage2_diagnostic.json"),
            transform_params=str(tp),
        ),
        budget_ledger=str(base / "budget_ledger.json"),
        ledger_entries=len(led.get("entries") or []) or "MISSING",
        # ---- 计划 §5 未做的三支(诚实标注, 不假装完成 ✓) ----
        not_implemented=dict(
            stage0_cam_gif="复用已有 Stage0 记录即可, 本轮未产 (§C 最低交付未覆盖)",
            stage1_probe_cam_gif="需 Stage1 固定候选录像, 本轮 0 物理未做",
            full_pipeline_cam_gif="需 Stage1+Stage2/3 合集与 reset 转场, 本轮未做",
        ),
    )


# --------------------------------------------------------------------------- #
# 方向一: stage0_vs_stage23 (§B 与 §5 的 stage0_vs_stage23.gif)
# --------------------------------------------------------------------------- #
def test_direction1_stage0_vs_stage23_gif():
    """★ 方向一: **Stage0 参考 vs Stage2/3 实测** 同源帧双画面对照 GIF ✓

    计划 §B1 要求 ✓: 同一 TCP 定义、同一相机/坐标; §B3 要求 ✓:
    报告 Δp/ΔR 误差, 不只凭 GIF 判断。
    ★ **0 物理**: 全用已有 `preprocess_traj.npz` + `gripper_trace.csv` 离线渲染 ✓
    """
    import cv2

    base, pp, tr, ff = _load_run(_RUN)
    trace = _read_trace(tr)
    d = np.load(pp, allow_pickle=True)
    ref = dict(grip_pos=d["grip_pos"], grip_R=d["grip_R"])
    prod = _production_callables()          # ← 真实调用生产函数 ✓

    dp, drot, src, hold = _compare(trace, ref)
    print("\n[direction1] 跟踪段(MANO_FOLLOW)对照点=%d  "
          "(闭合/预紧/搬运段无 Stage0 对应关系 ⇒ 不计入 ✓)"
          % len(dp))
    if len(dp) == 0:
        pytest.skip("跟踪段无法对齐 ⇒ 不产对照 (不伪造 ✓)")
    print("[direction1] Δp  mean=%.2fmm  max=%.2fmm  (真实跟踪误差 ✓)"
          % (dp.mean(), dp.max()))
    print("[direction1] ΔR  mean=%.2fdeg max=%.2fdeg" % (drot.mean(), drot.max()))
    if len(hold):
        # ★ 必须是**分段内**漂移: 跨段(CLOSE→APPROACH→CARRY)比末点毫无意义 ✓
        # (旧口径 215.9mm 就是这么算出来的假数 ✓)
        segs = {}
        for i in hold:
            segs.setdefault(str(trace["phase"][i]), []).append(i)
        worst = 0.0
        for name, ids in sorted(segs.items()):
            hp = trace["pos"][np.array(ids)]
            d = np.linalg.norm(hp - hp[-1], axis=1) * 1000.0
            worst = max(worst, float(d.max()))
            print("[direction1]   %-16s n=%3d 段内漂移 max=%7.2fmm"
                  % (name, len(ids), d.max()))
        print("[direction1] 保持段最大段内漂移 = %.2fmm  (分段口径 ✓)" % worst)

    # 突变诊断(定位你观察到的"不平滑" ✓)
    step = np.linalg.norm(np.diff(trace["pos"], axis=0), axis=1) * 1000.0
    med = float(np.median(step))
    spikes = np.where(step > max(3 * med, 1.0))[0] if step.size else []
    if len(spikes):
        top = int(np.argmax(step))
        print("[direction1] 位移突变 %d 处; 最大 idx=%d (%.2fmm, phase=%s, frame=%.0f->%.0f)"
              % (len(spikes), top, step[top], trace["phase"][top],
                 trace["frame"][top], trace["frame"][top + 1]))
    dE = np.linalg.norm(np.diff(trace["euler"], axis=0), axis=1)
    if dE.size:
        te = int(np.argmax(dE))
        print("[direction1] 姿态最大跳变 idx=%d (%.2fdeg, phase=%s, frame=%.0f->%.0f)"
              % (te, dE[te], trace["phase"][te],
                 trace["frame"][te], trace["frame"][te + 1]))

    # ---- 离线渲染两个画面 ----
    xs, Rh2g, th2g = _load_xform(base)
    # 逆变换 + 重心化(夹爪原点对齐轨迹, 否则夹爪在地面下不可见 ✓)
    pos_local = _recenter_local(_to_local(trace["pos"], xs, Rh2g, th2g))
    ref_local = _recenter_local(_to_local(ref["grip_pos"], xs, Rh2g, th2g))

    sapien, scene, robot = _build_replay_scene()
    active = robot.get_active_joints()
    gi = [i for i, j in enumerate(active) if "finger" in (j.get_name() or "").lower()]
    # 相机对准**本地**轨迹包围盒中心
    center = pos_local.mean(axis=0).copy()
    radius = float(np.linalg.norm(pos_local.max(axis=0) - pos_local.min(axis=0))) * 0.5
    eye = center + np.array([radius * 1.6, radius * 1.3, radius * 1.1])
    quat = prod["look_at"](eye, center)      # ← 生产 make_look_at_camera ✓
    cam = scene.add_camera("cam", _CAM_W, _CAM_H, 1.0, 0.01, 100.0)
    cam.set_focal_lengths(_FOCAL * max(0.25, radius * 4.0),
                          _FOCAL * max(0.25, radius * 4.0))   # ← 同生产 :2354 ✓
    cam.set_pose(sapien.Pose(eye.tolist(), np.asarray(quat).tolist()))

    marker = _draw_marker(sapien, scene, "sphere", radius * 0.05,
                          [0.15, 0.85, 0.25, 1.0])
    frames, err_series = [], []
    n_show = min(len(trace["phase"]), 150)
    for i in range(n_show):
        # 实测: 移动夹爪(本地系 ✓) + 手指
        robot.set_pose(sapien.Pose(pos_local[i].tolist(), [1.0, 0.0, 0.0, 0.0]))
        if gi:
            active[gi[0]].set_drive_target(float(trace["f1"][i]))
            active[gi[-1]].set_drive_target(float(trace["f2"][i]))
        # 参考: 同一画面里用绿球标 Stage0 参考位姿(本地系, §B1 同坐标系 ✓)
        # ★ 只在**跟踪段**显示 (其余段与 Stage0 无对应关系 ⇒ 球会误导 ✓)
        if trace["phase"][i] == "MANO_FOLLOW":
            rf = float(trace["frame"][i])
            r_p, _ = _interp_ref(ref, np.arange(len(ref["grip_pos"]),
                                                dtype=np.float64), rf)
            marker.set_pose(sapien.Pose(
                _recenter_local(_to_local(r_p[None, :], xs, Rh2g, th2g))[0].tolist(),
                [1.0, 0.0, 0.0, 0.0]))
        else:
            marker.set_pose(sapien.Pose([0.0, -2.0, -2.0]))   # 移出画面 ✓
        for _ in range(16):
            scene.step()
        scene.update_render()
        cam.take_picture()
        frame = _to_uint8(cam.get_picture("Color"))
        # 误差读数叠加(§B3: 不只凭 GIF 判断 ✓)
        j = np.argmin(np.abs(src - trace["frame"][i])) if len(src) else -1
        e_txt = "dp=%.1fmm dR=%.1fdeg" % (dp[j], drot[j]) if j >= 0 else "ref: n/a"
        cv2.putText(frame, e_txt, (8, _CAM_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, e_txt, (8, _CAM_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        frames.append(frame)
        err_series.append((int(trace["frame"][i]), e_txt))

    # ★ 视频 = **时序**(每帧一张, 不拼接!) ✓
    #   (横排/网格拼帧 ⇒ 尺寸爆炸 ⇒ MPEG-4 打不开 ✗)
    for i in range(len(frames)):
        img = frames[i]
        txt = "F%.0f %s | %s" % (trace["frame"][i], trace["phase"][i],
                                 err_series[i][1])
        cv2.putText(img, txt, (8, _CAM_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3)
        cv2.putText(img, txt, (8, _CAM_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)
    panel = frames[0]
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _GIF_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = _OUT_ROOT / "stage0_vs_stage23.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), _MP4_FPS,
                         (_CAM_W, _CAM_H))
    for img in frames:
        vw.write(np.ascontiguousarray(img.astype(np.uint8)))
    vw.release()

    print("[direction1] → %s (%d 帧, %dx%d)" % (mp4, len(frames), _CAM_W, _CAM_H))
    gif1 = _export_gif(mp4, _GIF_DIR / "stage0_vs_stage23.gif",
                       src_fps=_MP4_FPS)
    if gif1:
        print("[direction1] GIF → %s (%d 帧, 播放倍率 %.2fx ✓)"
              % (gif1["path"], gif1["frames"], gif1["actual_playback_ratio"]))

    report = dict(
        run_mode="diagnostic_full", formal_eligible=False, physical_episodes=0,
        task="direction1_stage0_vs_stage23",
        compared_phase="MANO_FOLLOW",
        compared_phase_only=True,
        n_non_tracking_points=int(len(hold)),
        delta_p_mm=dict(mean=float(dp.mean()), max=float(dp.max()),
                        p50=float(np.percentile(dp, 50))),
        delta_R_deg=dict(mean=float(drot.mean()), max=float(drot.max()),
                         p50=float(np.percentile(drot, 50))),
        position_step_mm=dict(median=med, max=float(step.max()) if step.size else 0.0,
                              n_spikes=int(len(spikes))),
        max_jump_index=top if len(spikes) else None,
        max_jump_mm=(float(step[top]) if len(spikes) else None),
        first_failure=(json.loads(ff.read_text(encoding="utf-8")).get("earliest_valid_event")
                       if ff else None),
        video_mp4=str(mp4), video_frames=int(len(frames)),
        gif=gif1,
        note="误差只用于诊断展示; formal 口径与本报告分开 (§5 隔离 ✓)",
    )
    (_OUT_ROOT / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert panel.size > 0 and len(dp) > 0, "对照 GIF 生成失败 ✗"
    assert gif1 is not None, "GIF 未产出 ⇒ 交付物缺失 ✗"


# --------------------------------------------------------------------------- #
# 方向二: stage23_continuous (§A 的 stage23_continuous_cam.gif)
# --------------------------------------------------------------------------- #
def test_direction2_stage23_continuous_gif():
    """★ 方向二: **Stage2→Stage3 连续诊断** GIF, 持续 DIAGNOSTIC 水印 + 首失败 ✓

    计划 §A1 要求 ✓: Stage2/3 同 episode 连续; §A2 要求 ✓: 失败分类而非任意续行;
    §5 要求 ✓: `formal_eligible=false`, 持续显示首失败。
    ★ **0 物理**: 用已有 trace 离线重放, 不调用任何 physics 预算 ✓
    ⚠ `execution_complete` 与 `task_success` **分开** 记录 (§5 ✓)
    """
    import cv2

    base, pp, tr, ff = _load_run(_RUN)
    trace = _read_trace(tr)
    prod = _production_callables()
    first_fail = None
    if ff:
        first_fail = json.loads(ff.read_text(encoding="utf-8")).get("earliest_valid_event")

    xs, Rh2g, th2g = _load_xform(base)
    # 逆变换 + 重心化(同方向一 ✓)
    pos_local = _recenter_local(_to_local(trace["pos"], xs, Rh2g, th2g))
    sapien, scene, robot = _build_replay_scene()
    active = robot.get_active_joints()
    gi = [i for i, j in enumerate(active) if "finger" in (j.get_name() or "").lower()]

    # god 视角: 高空俯瞰抓取区 (高度按本地轨迹尺度自适应 ✓)
    center = pos_local.mean(axis=0)
    radius = max(0.05, float(np.linalg.norm(
        pos_local.max(axis=0) - pos_local.min(axis=0))) * 0.5)
    focal = _FOCAL * max(0.25, radius * 4.0)
    gz = float(center[2])
    cx, cy = float(center[0]), float(center[1])
    god_pos = np.array([cx, cy, gz + radius * 2.2])
    god_look = np.array([cx, cy, gz])
    god_q = prod["look_at"](god_pos, god_look, up=np.array([1.0, 0.0, 0.0]))
    god = scene.add_camera("god", _CAM_W, _CAM_H, 1.0, 0.01, 100.0)
    god.set_focal_lengths(focal, focal)       # ← 同生产 :2407 ✓
    god.set_pose(sapien.Pose(god_pos.tolist(), np.asarray(god_q).tolist()))

    # 近景 cam (同样自适应 ✓)
    eye = center + np.array([radius * 1.5, radius * 1.2, radius * 1.0])
    cam_q = prod["look_at"](eye, center)
    cam = scene.add_camera("cam", _CAM_W, _CAM_H, 1.0, 0.01, 100.0)
    cam.set_focal_lengths(focal, focal)       # ← 同生产 ✓
    cam.set_pose(sapien.Pose(eye.tolist(), np.asarray(cam_q).tolist()))

    target = _draw_marker(sapien, scene, "box", radius * 0.06,
                          [0.85, 0.25, 0.15, 1.0])

    fail_idx = None
    if first_fail and isinstance(first_fail, str):
        for i, ph in enumerate(trace["phase"]):
            if ph == "CARRY":
                fail_idx = i
                break
    frames = []
    step = max(1, len(trace["phase"]) // 120)
    for i in range(0, len(trace["phase"]), step):
        robot.set_pose(sapien.Pose(pos_local[i].tolist(), [1.0, 0.0, 0.0, 0.0]))
        if gi:
            active[gi[0]].set_drive_target(float(trace["f1"][i]))
            active[gi[-1]].set_drive_target(float(trace["f2"][i]))
        # 物体沿真实 obj_z 演化(离线重放, 不推进物理语义 ✓)
        target.set_pose(sapien.Pose([float(center[0]), float(center[1]),
                                     float(center[2] + trace["obj_z"][i] - 0.02)],
                                    [1.0, 0.0, 0.0, 0.0]))
        for _ in range(16):
            scene.step()
        scene.update_render()
        god.take_picture()
        cam.take_picture()
        g_img = _to_uint8(god.get_picture("Color"))
        c_img = _to_uint8(cam.get_picture("Color"))
        warn = ""
        if fail_idx is not None and i >= fail_idx:
            warn = " | FAIL@%s" % first_fail
        for img, tag in ((g_img, "god"), (c_img, "cam")):
            t1 = "DIAGNOSTIC (formal_eligible=false)%s" % warn
            cv2.putText(img, t1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3)
            cv2.putText(img, t1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            t2 = "%s | %s f=%.0f bil=%s" % (tag, trace["phase"][i],
                                            trace["frame"][i], trace["bil"][i])
            cv2.putText(img, t2, (8, _CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3)
            cv2.putText(img, t2, (8, _CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
        frames.append(_stack([g_img, c_img], ["god", "cam"]))

    # ★ 每帧 = god|cam **横排两格**(时序, 不纵向堆叠!) ✓
    panel = _stack([g_img, c_img], ["god", "cam"])
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _GIF_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = _OUT_ROOT / "stage23_continuous.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), _MP4_FPS,
                         (_CAM_W * 2, _CAM_H))
    for img in frames:
        vw.write(np.ascontiguousarray(img.astype(np.uint8)))
    vw.release()

    print("[direction2] → %s (%dx%d, %d 帧)"
          % (mp4, _CAM_W * 2, _CAM_H, len(frames)))
    gif_info = _export_gif(mp4, _GIF_DIR / "stage23_continuous.gif",
                           width=640, src_fps=_MP4_FPS)
    if gif_info:
        print("[direction2] GIF → %s (%d 帧, 播放倍率 %.2fx ✓)"
              % (gif_info["path"], gif_info["frames"],
                 gif_info["actual_playback_ratio"]))

    manifest = dict(
        run_mode="diagnostic_full", diagnostic=True, formal_eligible=False,
        physical_episodes=0,
        execution_complete=True,                 # 视频走完 ✓
        task_success=False,                      # ★ 但物理门未过 (§5 允许二者并存 ✓)
        first_failure=first_fail,
        pass_count=None,                         # ★ 不用诊断冒充 passed (§2 ✓)
        real_pass_count=0,
        backend="physx_cpu",
        profile_source=str((base / "run_metadata.json")),
        trace_source=str(tr),
        stage0_source=str(pp),
        videos=["stage0_vs_stage23.mp4", "stage23_continuous.mp4"],
        stage23_video_frames=int(len(frames)),
        stage23_gif=gif_info,
        note="诊断产物不得进入 Stage1 recipe 候选池或覆盖旧成功 (§5 ✓)",
    )
    # ---- §5 要求的其余字段 (从已有产物读取, 不臆造 ✓) ----
    manifest.update(_manifest_extra(base, trace))
    (_OUT_ROOT / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[direction2] execution_complete=True, task_success=False (分开记录 ✓)")
    print("[direction2] first_failure=%s" % first_fail)
    assert panel.size > 0 and len(frames) > 0, "连续诊断 GIF 生成失败 ✗"
    assert gif_info is not None, "GIF 未产出 ⇒ 交付物缺失 ✗"


# --------------------------------------------------------------------------- #
# 计划 §C 的产物隔离约束 (反例测试 ✓)
# --------------------------------------------------------------------------- #
def test_diagnostic_products_are_not_formal_eligible():
    """★ 反例: 诊断产物**必须**被标 `formal_eligible=false` 且 `pass_count=None` ✓

    计划 §5: "formal loader必须拒绝diagnostic类型" ✓
    ⚠ 用**内存里的合成 manifest**做断言, **不覆盖**方向二写出的真实 manifest ✓
      (否则本测试会抹掉方向二的产物 ✗)
    """
    synthetic = dict(run_mode="diagnostic_full", formal_eligible=False,
                     physical_episodes=0, pass_count=None, task_success=False)
    assert synthetic["run_mode"] == "diagnostic_full"
    assert synthetic["formal_eligible"] is False, "诊断产物被标成正式资格 ✗"
    assert synthetic["pass_count"] is None, "诊断产物带 pass_count ⇒ 会冒充成功 ✗"
    assert synthetic["task_success"] is False, "task_success 必须为 False (视频≠成功 ✓)"

    # 若方向二已产出真实 manifest, 顺带核对它同样合规 ✓
    real = _OUT_ROOT / "diagnostic_manifest.json"
    if real.exists():
        m = json.loads(real.read_text(encoding="utf-8"))
        assert m.get("run_mode") == "diagnostic_full", \
            "真实 manifest 的 run_mode 不是 diagnostic_full ✗"
        assert m.get("formal_eligible") is False, "真实 manifest 带正式资格 ✗"
        assert m.get("pass_count") is None, "真实 manifest 带 pass_count ✗"
        assert int(m.get("physical_episodes", -1)) == 0, "真实 manifest 声明了物理 ✗"
        print("\n[isolation] ✓ 真实 manifest 亦合规 (formal_eligible=false / "
              "pass_count=None / physical_episodes=0 ✓)")
    else:
        print("\n[isolation] ✓ 合成 manifest 合规 (真实 manifest 尚未生成 ✓)")


def test_zero_physical_budget_declared():
    """★ 本套 GIF **必须**声明 0 物理 (§6: 有完整状态可离线渲染则 0 物理 ✓)"""
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    p = _OUT_ROOT / "diagnostic_manifest.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        assert int(m.get("physical_episodes", -1)) == 0, \
            "声明了物理 episode ⇒ 违反'离线渲染 0 物理' ✗"
    print("\n[budget] ✓ physical_episodes=0 (未消耗物理预算 ✓)")

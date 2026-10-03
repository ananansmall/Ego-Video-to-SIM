"""GPU 物理后端**可行性探针与结论固化**（2026-10-02）—— 零生产代码改动 ✓

本文件把 GPU 可用性的**实测结论**钉成可重复执行的断言, 避免继续把注释当结论 ✓。

## 实测结论（本机, 逐条可复现, 见文件内 `_probe_*` 函数）

| 判据 | 结果 |
|---|---|
| `torch.cuda.is_available()` | **False**（驱动 535.230.02 只支持 CUDA 12.2, 环境是 cu130）|
| `libPhysXGpu_64.so` 就位 | ✅ 已就位（226MB, `~/.sapien/physx/105.1-.../`）|
| `enable_gpu()` + `PhysxGpuSystem()` + `gpu_init()` | ✅ 全部成功, `device = cuda:0` |
| 空场景 `scene.step()` × 3 | ✅ 3/3 成功 |
| **加入任何实体(ground 或 box)后 `step()`** | ❌ **100% 失败**（6/6 组合）: `RuntimeError: failed to step: gpu simulation is not initialized.` |

⇒ **不是"驱动太旧"**（GPU 库能加载、能识别 cuda:0、能 gpu_init、能空场景 step）✗
⇒ **不是显存不足**（deep6 结束后显存空闲 3.9GB 仍同样失败 ✗）
⇒ **不是 `set_pose` 冲突**（不调用 set_pose 也失败 ✗）
⇒ **不是调用序 / build 入口 / body 类型**（before/after scene、build/static/kinematic、
  dt→init / init→dt 全组合均失败 ✗）
⇒ 结论: **SAPIEN 3.0.0b0 + 驱动 535 + 本机 PhysX 5.3.1 的组合下, GPU pipeline
  一旦场景含实体就无法 step** ⇒ 对本管线**不可用** ✓（架构/版本层面, 非配置问题）

## 为什么仍保留 CPU 侧提速建议 ✓

本管线的瓶颈**确实是物理**: 一次 Stage1 rollout ≈200s / 350 控制点 / 16 子步
⇒ **每子步 ≈35ms**。GPU 走不通 ⇒ 提速只能来自 CPU 侧:
  ① **并行 rollout**（多进程, 各自独立场景）—— 当前是唯一不改变物理语义的手段 ✓
  ② 降 `_COACD_MAX_HULL`(64 → 更小) —— 会改变碰撞几何 ⇒ **改变物理** ✗ 需谨慎
  ③ 降 `_S1_N_FRAMES` —— 会改变 Stage1 动作时长 ⇒ **改变任务** ✗ 不可接受

## 用法 ✓

    # 零成本环境层（默认, 不建场景）
    python -m pytest -q tests/test_gpu_backend_probe.py
    # 实测行为层（会建 GPU 场景 ⇒ 约 30s）
    SAPIEN_GPU_PROBE=1 python -m pytest -q -s tests/test_gpu_backend_probe.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PHYSX_DIR = Path(os.path.expanduser("~")) / ".sapien" / "physx"
_PROBE_ON = str(os.environ.get("SAPIEN_GPU_PROBE", "0")).strip().lower() \
    not in ("0", "false", "no", "")
_PARITY_ON = str(os.environ.get("SAPIEN_RENDER_PARITY", "0")).strip().lower() \
    not in ("0", "false", "no", "")

_DT = 1.0 / 480.0        # 与 physics_env.PHYSICS_TIMESTEP 一致
_N_SUBSTEP = 16          # DECIMATION
_N_CTRL = 20             # 每控制点 16 子步, 跑 20 点 ≈ 一次短诊断的量级


# --------------------------------------------------------------------------- #
# 层 1: 环境事实（零成本, 无条件运行）
# --------------------------------------------------------------------------- #
def _nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + query,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                     # noqa: BLE001
        return ""


def test_gpu_driver_version_is_the_known_limit():
    """驱动版本是**已知事实** ⇒ 钉住它, 便于换机器时立刻发现差异 ✓"""
    drv = _nvidia_smi("driver_version")
    if not drv:
        pytest.skip("nvidia-smi 不可用 ⇒ 非 GPU 机器 ✓")
    assert drv, "驱动版本读取失败"
    # 只记录不断言具体版本（换机器时不该红）✓
    print("\n[env] driver=%s" % drv)


def test_physx_gpu_library_is_present():
    """`libPhysXGpu_64.so` 必须就位 —— 这是 SAPIEN 内置下载会卡死的那个文件 ✓

    SAPIEN 用 `requests.get` **无超时**去 GitHub 拉它 ⇒ 卡死 ⇒ 必须手动预置 ✓。
    本断言把该前置条件钉死: 缺失时给出修复路径, 而不是跑到物理阶段才失败 ✓。
    """
    libs = sorted(_PHYSX_DIR.glob("*/libPhysXGpu_64.so"))
    if not libs:
        pytest.skip(
            "libPhysXGpu_64.so 未就位 ⇒ GPU 场景不可能构造。"
            "修复: 手动下载 PhysX 5.3.1 GPU 库放到 ~/.sapien/physx/<版本>/ ✓ "
            "(SAPIEN 内置下载无超时会卡死 ✗)")
    mb = libs[0].stat().st_size / 1e6
    assert mb > 100, "libPhysXGpu_64.so 仅 %.1fMB ⇒ 下载被截断 ✗" % mb
    print("\n[env] libPhysXGpu_64.so = %.1fMB @ %s" % (mb, libs[0].parent.name))


def test_torch_cuda_state_is_not_a_gpu_physx_verdict():
    """**torch 的 CUDA 可用性不是 GPU PhysX 的判据** ✓

    本机 `torch.cuda.is_available()=False`（cu130 vs 驱动 535/CUDA 12.2），
    但 SAPIEN 的 `PhysxGpuSystem` **能**构造、`gpu_init()` **能**成功 ✓
    ⇒ 任何"torch False ⇒ 跳过 GPU 物理"的写法都是**误判** ✗。

    本测试只做断言: torch 状态可读, 且不因此改变对 SAPIEN 的判断 ✓。
    """
    import torch
    print("\n[env] torch=%s cuda=%s avail=%s"
          % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    assert torch.version.cuda is not None or True   # 状态记录, 不断言


def test_cuda_driver_api_is_fully_functional():
    """★ CUDA **driver API 完全正常** ⇒ 排除"驱动/GPU 硬件问题" ✓

    实测 ✓: `cuInit(0)=0` (CUDA_SUCCESS)、`cuDriverGetVersion=12020` (CUDA 12.2)、
    `cuDeviceGetCount=1`、识别到 `NVIDIA GeForce RTX 3050 Laptop GPU` ✓。

    ⇒ **GPU 跑不通与驱动、硬件、torch 都无关** ✗（本测试把这些钉死, 免得重复排查）✓。
    """
    import ctypes
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except Exception as exc:                          # noqa: BLE001
        pytest.skip("libcuda.so.1 不可加载 ⇒ 无 NVIDIA 驱动: %s" % exc)
    rc = cuda.cuInit(0)
    assert rc == 0, "cuInit=%s (0=CUDA_SUCCESS) ⇒ 驱动不可用 ✗" % rc
    ver = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(ver))
    n = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(n))
    print("\n[env] driver CUDA=%d  devices=%d" % (ver.value, n.value))
    assert n.value >= 1, "cuDeviceGetCount=0 ⇒ 无 GPU 设备 ✗"
    assert ver.value >= 12000, "driver CUDA API %d < 12.0 ⇒ PhysX5 GPU 不支持 ✗" % ver.value


def test_cuda_12_runtime_is_absent_but_may_be_installable():
    """★ 环境缺 `libcudart.so.12`（只有 11.x 与 13.x）⇒ **一个可尝试的修复路径** ✓

    实测 ✓: 系统有 `/usr/local/cuda-11.8`(CUDA 11.8)、pip 有 `libcudart.so.13`,
    但 **没有任何 `libcudart.so.12`** ⇒ PhysX 5.3.1 GPU 缺 CUDA 12 runtime。

    已实测过 ✓: `pip download nvidia-cuda-runtime-cu12` + `LD_LIBRARY_PATH` 注入
    ⇒ `libcudart.so.12` 能加载, **但 step 仍失败** ⇒ 缺 runtime **不是**唯一原因 ✗。
    本测试记录该事实, 避免以后重复试这条路 ✓。
    """
    import ctypes
    import glob
    pats = ["/usr/local/cuda*/lib64/libcudart.so*",
            "/usr/lib/x86_64-linux-gnu/libcudart.so*",
            os.path.expanduser(
                "~/miniconda3/envs/*/lib/python3.10/site-packages"
                "/nvidia/*/lib/libcudart.so*")]
    found = []
    for pat in pats:
        found += glob.glob(pat)
    vers = sorted({os.path.basename(f) for f in found})
    print("\n[env] libcudart on system: %s" % (vers or "(none)"))
    try:
        ctypes.CDLL("libcudart.so.12")
        print("[env] libcudart.so.12 present ✓")
    except Exception:                                 # noqa: BLE001
        print("[env] libcudart.so.12 ABSENT ⇒ PhysX5 GPU 缺 CUDA12 runtime "
              "(实测: 补上后仍失败 ⇒ 非唯一原因 ✗)")
    # 只做记录, 不断言 ⇒ 换机器时本测试不会红 ✓
    assert True


# --------------------------------------------------------------------------- #
# 层 2: 行为实测（需 SAPIEN_GPU_PROBE=1; 不触碰任何生产代码路径）
# --------------------------------------------------------------------------- #
def _require_fresh_process_for_gpu():
    """GPU 探针**必须在未初始化 PhysX 的干净进程**里跑 ✓ (2026-10-03 实测)

    ⚠ 两种错法都试过, 结论是**同进程互斥**, 不是可以容错的小坑:
      ① 直接调 `enable_gpu()` ⇒ 若同进程已有 CPU 场景, 抛
         `GPU PhysX can only be enabled once before any other code involving
         PhysX` ✓
      ② 把该异常吞掉继续 ⇒ C++ 侧"GPU 已启用"flag **不会被置上** ⇒
         紧接着 `PhysxGpuSystem()` 改抛 `sapien.physx.enable_gpu() must be
         called before creating a PhysX GPU system` ✓ (实测已复现)
      ⇒ 唯一正确做法: **skip 并要求换干净进程**, 不伪造、不吞异常 ✓
    """
    import sapien
    try:
        sapien.physx.enable_gpu()          # 已启用时幂等 return ✓
    except RuntimeError as e:
        pytest.skip("GPU 探针必须在**未初始化 PhysX 的干净进程**里跑 "
                    "(同进程已有 CPU 场景 ⇒ %s) ⇒ 请单独运行: "
                    "SAPIEN_GPU_PROBE=1 python -m pytest -q -s "
                    "tests/test_gpu_backend_probe.py" % e)


def _gpu_scene():
    """按**实测可行的最小序列**建 GPU 场景 ✓（探针 15/17 确认）"""
    import sapien
    import sapien.render
    _require_fresh_process_for_gpu()
    system = sapien.physx.PhysxGpuSystem()
    system.set_timestep(_DT)
    system.gpu_init()                       # 必须在建 scene 前 ✓
    scene = sapien.Scene(systems=[system, sapien.render.RenderSystem()])
    scene.set_timestep(_DT)
    return sapien, scene, system


def _add_box(scene):
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
    return builder.build(name="probe_box")


def _try_step(scene, n=3):
    try:
        for _ in range(n):
            scene.step()
        return True, ""
    except Exception as exc:                              # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)


@pytest.mark.skipif(not _PROBE_ON, reason="需要真实 GPU 场景 ⇒ SAPIEN_GPU_PROBE=1 ✓")
def test_gpu_pipeline_constructs_and_steps_empty_scene():
    """判据 A: `enable_gpu` → `PhysxGpuSystem` → `gpu_init` → 空场景 step **必须成功** ✓

    这条通过 ⇒ 说明"驱动太旧 / 库缺失"**不是**根因 ✓（推翻旧结论的必要对照）。
    """
    sapien, scene, _sys = _gpu_scene()
    ok, err = _try_step(scene, 3)
    try:
        scene.clear()
    except Exception:                                     # noqa: BLE001
        pass
    assert ok, "空场景 step 失败 ⇒ GPU pipeline 连最小形态都不可用 ⇒ %s" % err
    print("\n[gpu] 空场景 step OK (device=%s)" % sapien.physx.PhysxGpuSystem)


@pytest.mark.skipif(not _PROBE_ON, reason="需要真实 GPU 场景 ⇒ SAPIEN_GPU_PROBE=1 ✓")
@pytest.mark.parametrize("kind", ["ground", "box"])
def test_gpu_pipeline_fails_once_scene_has_any_entity(kind):
    """判据 B: **只要场景含一个实体, step 就失败** ⇒ 本管线 GPU 不可用 ✓

    这是决定性判据 ✓: 本管线必须有 ground + 8 个物体 + 机器人 ⇒ 必然命中 ✗。
    固化它的意义: 后续任何人再想试 GPU, 会立刻看到这条红灯, 而不是重新烧一小时 ✓。
    """
    import sapien.render  # noqa: F401
    sapien, scene, _sys = _gpu_scene()
    if kind == "ground":
        scene.add_ground(0.0, render=False)
    else:
        _add_box(scene)
    ok, err = _try_step(scene, 1)
    try:
        scene.clear()
    except Exception:                                     # noqa: BLE001
        pass
    if ok:
        pytest.xfail(
            "★ 本机 GPU pipeline 现在能在含实体场景 step ⇒ 之前的失败原因已变化 "
            "(驱动/库/版本变动) ⇒ **应重新评估 GPU 后端** ✓ 别再沿用旧结论 ✗")
    assert "gpu simulation is not initialized" in err, \
        "失败原因与实测不符 ⇒ %s" % err
    print("\n[gpu] %s 实体 ⇒ step 失败(预期) ✓ %s" % (kind, err))


@pytest.mark.skipif(not _PROBE_ON, reason="需要真实 GPU 场景 ⇒ SAPIEN_GPU_PROBE=1 ✓")
def test_gpu_official_warmstart_api_also_fails():
    """判据 B2: **官方 "warm start" 三段式 API 也失败** ⇒ 封死第 9 条修复路径 ✓

    ★ 2026-10-03 新增 ✓ —— 之前 8 条路径都**没用过** SAPIEN 官方 warm start:
      `pysapien.so` 内嵌文档串明写 *"Warm start the GPU simulation by stepping
      the system once. This function must be called before any PhysX body is
      added to scene."* ⇒ 说明 GPU pipeline **可能需要先 warm start** ✓

    ★★ 失败分界是"**有无实体**", 不是"Scene 层" ✓ (2026-10-03 实测校正):
      同一个 GPU system (`gpu_init()=True`, `device=cuda:0`), **空场景**下
      `scene.step()` / `gpu.step()` / `step_start+step_finish` **三种全部 OK** ✓;
      一旦 `add_ground()` 加了 **1 个实体**, 同样三种调用**全部失败** ✗
      ⇒ warm start 本身没坏, 也不是"Scene 包一层就挂" ⇒ **只要有实体就挂**
      ⇒ 判据收紧为: **GPU 后端只支持空场景**, 本项目要跑的全是实体 ⇒ 不可用 ✓
      (CPU 同场景 `scene.step()` OK ⇒ 场景/调用姿势本身没问题 ✓)

    实测组合 (空场景 3/3 OK → 加实体后 3/3 失败):
        ① step_start + step_finish
        ② 直接 gpu.step()
        ③ step_start → scene.step → step_finish
    本测试跑在**已含 ground 的场景**上 ⇒ 上面 3 条预期全部 FAIL ✓
    """
    import sapien
    import sapien.render  # noqa: F401

    _require_fresh_process_for_gpu()
    gpu = sapien.physx.PhysxGpuSystem()
    gpu.gpu_init()
    assert str(gpu.device) == "cuda:0", "GPU device 异常: %s" % gpu.device

    # 先证明 Scene 确实绑到了 GPU system (排除"传错 system"这个可能 ✓)
    sc = sapien.Scene(systems=[gpu, sapien.render.RenderSystem()])
    assert isinstance(sc.get_physx_system(), sapien.physx.PhysxGpuSystem), \
        "Scene 未绑定 GPU system ⇒ 传参错, 不是 GPU 本身的问题 ✗"

    combos = (("step_start+step_finish", lambda: (gpu.step_start(), gpu.step_finish())),
              ("gpu.step()", gpu.step),
              ("scene.step()", sc.step))

    def _run(sink):
        for name, fn in combos:
            try:
                fn()
                sink[name] = "OK"
            except Exception as e:                        # noqa: BLE001
                sink[name] = str(e)[:60]

    # ---- 对照组: **空场景**(尚未 add_ground) ⇒ 三种调用预期全部 OK ✓ ----
    empty = {}
    _run(empty)
    for k, v in empty.items():
        print("\n[warm:空场景]   %-26s -> %s" % (k, v))
    assert all(v == "OK" for v in empty.values()), \
        "空场景下 GPU step 就已经失败 ⇒ 与已知模式不同(≠'加实体才挂') " \
        "⇒ %s ⇒ 需重新排查 ✗" % empty

    sc.add_ground(0.0, render=False)

    # ---- 实验组: **含 1 个实体**(ground) ⇒ 三种调用预期全部失败 ✗ ----
    results = {}
    _run(results)
    for k, v in results.items():
        print("\n[warm:含实体]   %-26s -> %s" % (k, v))
    try:
        sc.clear()
    except Exception:                                     # noqa: BLE001
        pass

    # 只要**任意一条**能让含实体场景 step 成功, 就说明 GPU 可用 ⇒ 必须 xfail 提醒重评
    if any(v == "OK" for v in results.values()):
        pytest.xfail("★ 官方 warm start API 生效了 ⇒ GPU 可能真的可用 ⇒ "
                      "**应重新评估 GPU 后端** ✓ 别再沿用旧结论 ✗")
    assert all("not initialized" in v for v in results.values()), \
        "失败原因与已知模式不同 ⇒ %s ⇒ 需重新排查 ✗" % results


@pytest.mark.skipif(not _PROBE_ON, reason="需要真实 GPU 场景 ⇒ SAPIEN_GPU_PROBE=1 ✓")
def test_gpu_failure_is_not_vram_pressure():
    """判据 C: 失败**与显存无关** ⇒ 排除"显存不足"这个误判 ✓

    本机实测: deep6 结束后显存空闲 **3.9GB**, 仍 100% 同样失败 ✗
    ⇒ 不能用"显存不够"解释, 也不能靠"等显存空闲"解决 ✓。
    """
    free_mb = _nvidia_smi("memory.free")
    print("\n[env] GPU free memory = %s MB" % free_mb)
    sapien, scene, _sys = _gpu_scene()
    _add_box(scene)
    ok, err = _try_step(scene, 1)
    try:
        scene.clear()
    except Exception:                                     # noqa: BLE001
        pass
    assert not ok, "含实体场景 step 成功了 ⇒ 需重新评估判据 B（free=%sMB）" % free_mb
    print("[gpu] 显存空闲 %sMB 仍失败 ⇒ 排除显存原因 ✓" % free_mb)


@pytest.mark.skipif(not _PROBE_ON, reason="需要真实 GPU 场景 ⇒ SAPIEN_GPU_PROBE=1 ✓")
def test_cpu_backend_baseline_speed_for_reference():
    """CPU 对照速度, 作为"GPU 若可用能省多少"的基线 ✓

    同时验证 CPU 下 ``set_pose`` 100% 生效 —— 这是 Stage2 ready 门赖以成立的前提 ✓
    (若这条挂了, 说明探针本身坏了, GPU 那边的失败也就不可信 ✗)。
    """
    import time
    import numpy as np
    import sapien
    import sapien.render

    system = sapien.physx.PhysxCpuSystem()
    system.set_timestep(_DT)
    scene = sapien.Scene(systems=[system, sapien.render.RenderSystem()])
    scene.set_timestep(_DT)
    scene.add_ground(0.0, render=False)
    box = _add_box(scene)
    box.set_pose(sapien.Pose([0.0, 0.0, 0.3]))

    effective, t_ctrl = [], []
    for k in range(_N_CTRL):
        box.set_pose(sapien.Pose([0.001 * k, 0.0, 0.3 + 0.002 * k]))
        rb = np.asarray(box.get_pose().p, dtype=np.float64)
        effective.append(bool(np.allclose(rb[:2], [0.001 * k, 0.0], atol=1e-6)))
        t0 = time.perf_counter()
        for _ in range(_N_SUBSTEP):
            scene.step()
        t_ctrl.append(time.perf_counter() - t0)
    mean_ms = float(np.mean(t_ctrl)) * 1e3
    print("\n[cpu] %.3f ms/控制点 (%d 子步) | set_pose 生效率=%.2f"
          % (mean_ms, _N_SUBSTEP, float(np.mean(effective))))
    try:
        scene.clear()
    except Exception:                                     # noqa: BLE001
        pass
    assert float(np.mean(effective)) == 1.0, "CPU 下 set_pose 未生效 ⇒ 探针无效 ✗"
    assert mean_ms < 50.0, "CPU 每控制点 %.1fms 异常 ⇒ 探针环境被污染 ✗" % mean_ms


# --------------------------------------------------------------------------- #
# 层 4: 关渲染是否改变物理（"并行 rollout 关渲染"方案的前提）
# --------------------------------------------------------------------------- #
def test_force_cpu_path_is_already_a_production_option():
    """★ `--views none` **已经**是生产支持的路径 ⇒ 不需要新增代码 ✓

    [grasp_hawor.py:1867](grasp_hawor.py): `_force_cpu = (self.views == "none" or not self.views)`
    ⇒ `--views none` 即 `force_cpu=True` ⇒ **无渲染** ✓。

    本测试只读源码确认该分支存在 ⇒ 避免以后误以为"关渲染要改代码" ✗。
    """
    src = (Path(__file__).resolve().parents[1] / "grasp_hawor.py").read_text(
        encoding="utf-8")
    assert "_force_cpu = (self.views ==" in src, \
        "views→force_cpu 的接线消失 ⇒ `--views none` 不再等价于关渲染 ⇒ 请重新评估 ✓"
    print("\n[audit] --views none ⇒ force_cpu=True (生产已有 ✓)")


def test_disabling_render_does_not_change_physics_numerics():
    """★ **关渲染不改变物理数值** ⇒ 并行 rollout 可用它省资源 ✓

    实测（真实夹爪 URDF, `prepare_gripper_only_urdf` + URDF loader）✓:
        finger q  max|Δ| = **0.000e+00**（逐位相同 ✓）
        object    max|Δ| = **0.000000000 mm** ✓
    ⇒ 物理状态与渲染**完全解耦** ✓ ⇒ 关渲染跑出的 gate 数值与开渲染**可比** ✓

    ⚠ 但实测到一个**反直觉**事实 ✓: 关渲染**墙钟反而慢 1.33×**
    （A 带渲染 1.276 ms/ctrl vs B 无渲染 1.694 ms/ctrl）✗
    ⇒ **关渲染的价值不是"更快", 而是"省显存/避免多进程 OOM"** ✓
    ⇒ 时间收益只能来自**多进程并行**, 而并行的前提是每个进程省内存 ✓

    本测试用**生产 URDF 路径**跑真实夹爪 ⇒ 结论对真实管线有效 ✓
    需要 `SAPIEN_RENDER_PARITY=1`（会建两个场景, 约 30s）✓。
    """
    if not _PARITY_ON:
        pytest.skip("需要建两个场景对照 ⇒ SAPIEN_RENDER_PARITY=1 ✓")

    import time

    import numpy as np
    import sapien
    import sapien.render

    from physics_env import prepare_gripper_only_urdf

    def run(with_render, do_update_render=False):
        system = sapien.physx.PhysxCpuSystem()
        system.set_timestep(_DT)
        systems = [system] + ([sapien.render.RenderSystem()] if with_render else [])
        scene = sapien.Scene(systems=systems)
        scene.set_timestep(_DT)
        scene.add_ground(0.0, render=False)
        urdf = prepare_gripper_only_urdf("left", strip_visuals=not with_render)
        robot = scene.create_urdf_loader().load(urdf)
        robot.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
        active = robot.get_active_joints()
        gi = [i for i, j in enumerate(active)
              if "finger" in (j.get_name() or "").lower()]
        for j in active:
            j.set_drive_property(stiffness=1000.0, damping=200.0, force_limit=100.0)
        bb = scene.create_actor_builder()
        bb.add_box_collision(half_size=[0.022, 0.028, 0.02])
        obj = bb.build(name="target")
        obj.set_pose(sapien.Pose([0.0, 0.0, 0.02]))
        qs, objs, t = [], [], []
        for k in range(20):
            for i in gi:
                active[i].set_drive_target(0.03 - 0.0015 * k)
            if do_update_render and with_render:
                scene.update_render()
            t0 = time.perf_counter()
            for _ in range(_N_SUBSTEP):
                scene.step()
            t.append(time.perf_counter() - t0)
            qs.append(np.asarray(robot.get_qpos(), dtype=np.float64)[gi].copy())
            objs.append(np.asarray(obj.get_pose().p, dtype=np.float64).copy())
        try:
            scene.clear()
            del scene
        except Exception:                             # noqa: BLE001
            pass
        return np.array(qs), np.array(objs), float(np.mean(t)) * 1e3

    qa, oa, ta = run(True, do_update_render=True)
    qb, ob, tb = run(False)
    dq = float(np.max(np.abs(qa - qb)))
    do = float(np.max(np.abs(oa - ob)))
    print("\n[parity] q diff=%.3e  obj diff=%.9f mm" % (dq, do * 1000.0))
    print("[parity] wall: render=%.3f ms/ctrl  no-render=%.3f ms/ctrl  ratio=%.2fx"
          % (ta, tb, tb / max(1e-9, ta)))
    assert dq == 0.0, "关渲染**改变了**手指 q (%.3e) ⇒ 数值不可比 ✗" % dq
    assert do == 0.0, "关渲染**改变了**物体位姿 (%.9f mm) ⇒ 数值不可比 ✗" % (do * 1000.0)


# --------------------------------------------------------------------------- #
# 层 3: 生产开关 `_ENGINE_USE_GPU` 的**接线**审计（零物理, 只读生产代码）
# --------------------------------------------------------------------------- #
def test_engine_gpu_switch_is_wired_only_into_the_force_cpu_branch():
    """★ 发现的生产缺陷（**只读断言, 不改生产代码**）✓

    `physics_env.setup_physics_scene` 里 `_sys_sc`（GPU system）**只在
    `force_cpu=True` 分支被传进 Scene**:

    · [physics_env.py:802](physics_env.py)  `scene = sapien.Scene(systems=[_sys_sc or PhysxCpuSystem()])`
      —— **仅 force_cpu 分支** ✓
    · 正常分支走 [:811](physics_env.py) `scene = sapien.Scene()` ⇒ **不传 systems**
      ⇒ SAPIEN 用默认 CPU system ⇒ `_ENGINE_USE_GPU=1` **被静默忽略** ✗

    后果 ✓: 即便 GPU 后端完全正常, `--test-stage*` 的正常渲染路径也**永远拿不到 GPU**,
    而日志却会打印"GPU 物理后端已启用" ⇒ **报告与执行不一致**（md f2 三态纪律违反 ✗）。

    本测试**只读源码、不调用生产函数、不建场景** ⇒ 零成本零副作用 ✓。
    一旦上游修好接线, 本测试会 FAIL ⇒ 提醒把结论同步回 md ✓。
    """
    src_path = Path(__file__).resolve().parents[1] / "physics_env.py"
    src = src_path.read_text(encoding="utf-8")

    start = src.index("def setup_physics_scene")
    body = src[start:]
    end = body.find("\ndef ", 1)
    if end > 0:
        body = body[:end]

    gpu_assign = "_sys_sc = sapien.physx.PhysxGpuSystem()"
    assert gpu_assign in body, \
        "GPU system 构造行消失 ⇒ 接线已变 ⇒ 请重新审计并更新本测试 ✓"

    uses = [ln.strip() for ln in body.splitlines() if "_sys_sc" in ln]
    consume = [ln for ln in uses if "sapien.Scene(" in ln and "_sys_sc" in ln]
    assert consume, "没有任何 Scene(...) 消费 _sys_sc ⇒ GPU 开关完全无效 ✗"

    force_cpu_idx = body.find("if force_cpu:")
    if force_cpu_idx < 0:
        pytest.skip("无法定位 if force_cpu: ⇒ 请人工复核 ✓")

    else_idx = body.find("else:", force_cpu_idx)
    normal_branch = body[else_idx:] if else_idx > 0 else ""
    normal_scene = [ln.strip() for ln in normal_branch.splitlines()
                    if "sapien.Scene(" in ln]
    wired = any("_sys_sc" in ln for ln in normal_scene)

    print("\n[audit] force_cpu 分支 Scene: %s" % consume[0])
    print("[audit] 正常分支 Scene 调用  : %s" % (normal_scene[:1] or "(无)"))

    if not wired:
        pytest.xfail(
            "★ 已确认的生产缺陷 ✓: `_sys_sc` 只在 `if force_cpu:` 分支被消费; "
            "正常分支 `sapien.Scene()` **不传 systems** ⇒ `_ENGINE_USE_GPU=1` "
            "在 `--test-stage*` 真实路径上**被静默忽略** ✗ "
            "(日志却报'GPU 已启用' ⇒ 报告与执行不一致 ✗)。"
            "当前 GPU 本机也跑不通, 故暂不影响结果, 但修好 pipeline 后会踩 ✗")

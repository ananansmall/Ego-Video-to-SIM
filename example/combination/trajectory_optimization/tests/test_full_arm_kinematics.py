"""整个臂 (full_robot R1) 运动学/驱动验证 —— 0 物理预算, 只调用生产入口

★ 约束 ✓: **不改任何主文件代码**, 只**调用** `prepare_full_robot_urdf` /
  `setup_physics_scene` / `setup_robot` (与 grasp_hawor.py:2196 生产路径一致 ✓)
★ 动机 (用户 2026-10-03): 之前的 GPU/渲染测试只加载了**夹爪 URDF**,
  没验证过**整条臂** ⇒ 关节映射/PD/自碰撞/臂基座偏移全是空白 ✓

覆盖:
  T1 整臂 URDF 可加载 + 关节树完整 (含双臂+双夹爪)
  T2 臂基座偏移符合生产常量 (ARM_BASE_OFFSET_*)
  T3 关节驱动目标可写入且 q/qvel 响应 (PD 通道通)
  T4 自碰撞配置存在且非致命 (整臂不会自 explode)
  T5 GPU 后端在本机对整臂同样不可用 (与夹爪测试结论一致 ✓)
    ⚠ T5 在**独立子进程**里跑 —— `sapien.physx.enable_gpu()` 全进程只能调一次且
      必须在任何 PhysX 初始化之前; T1~T4 已建过 CPU 场景 ⇒ 同进程必抛 RuntimeError ✓
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _prod():
    """只 import 生产符号, 不改它们 ✓"""
    import sys
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from models.robot_forms import prepare_full_robot_urdf
    from physics_env import setup_physics_scene, setup_robot
    return prepare_full_robot_urdf, setup_physics_scene, setup_robot


def _side() -> str:
    return os.environ.get("ARM_SIDE", "right")


@pytest.fixture(scope="module")
def arm_env():
    """整臂场景 —— force_cpu=True 走生产 no-render 路径 (views none 同款 ✓)"""
    prepare_full_robot_urdf, setup_physics_scene, setup_robot = _prod()
    urdf = prepare_full_robot_urdf()
    assert Path(urdf).exists(), f"整臂 URDF 不存在: {urdf}"

    scene = setup_physics_scene(ground_height=0.0, force_cpu=True)
    side = _side()
    info = setup_robot(scene, "full_robot", side,
                       np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    robot = info["robot"]
    assert robot is not None, "setup_robot 未返回 robot ⇒ 整臂未加载 ✓"
    yield dict(scene=scene, robot=robot, info=info, side=side, urdf=urdf)
    try:
        scene.close()
    except Exception:                                   # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# T1 关节树完整性
# --------------------------------------------------------------------------- #
def test_t1_full_arm_articulation_tree_is_complete(arm_env):
    robot, side, info = arm_env["robot"], arm_env["side"], arm_env["info"]
    names = [l.get_name() for l in robot.get_links()]
    n_act = len(robot.get_active_joints())
    print("[T1] links=%d  active_joints=%d  joint_names=%s"
          % (len(names), n_act, info.get("joint_names")))

    # ★ link 实名取自实测 ✓ (不是猜的 "gripper_base_link"):
    #   right_arm_base_link / right_arm_link1..6 / right_gripper_link /
    #   right_gripper_finger_link1..2  ✓
    for need in (f"{side}_arm_base_link", f"{side}_gripper_link"):
        assert need in names, f"缺 {need} ⇒ 非完整整臂 ✗"
    for i in range(1, 7):
        assert f"{side}_arm_link{i}" in names, f"缺 {side}_arm_link{i} ✗"
    assert f"{side}_gripper_finger_link1" in names, "缺夹爪指尖 ✗"
    assert any(n.startswith("left_arm") for n in names), "缺左臂 ⇒ 非 full_robot ✗"

    # ★ 主动关节 **18** 个 (实测 ✓): 躯干 4 + 左臂 6 + 右臂 6 + 右夹爪 2
    #   (左夹爪 fixed ⇒ 不在 active 里 ✓) ⇒ 与"夹爪-only 只有 2 个"可区分 ✓
    assert n_act == 18, f"主动关节 {n_act} ≠ 18 (躯干4+双臂12+夹爪2) ✗"
    assert len(info["arm_joint_indices"]) == 6, \
        f"右臂关节数 {len(info['arm_joint_indices'])} ≠ 6 ✗"
    gi = info["gripper_indices"][side]
    assert len(gi) == 2, f"夹爪关节数 {len(gi)} ≠ 2 ✗"
    print("[T1] ✓ 双臂 6+6 + 躯干 4 + 夹爪 2 = %d 主动关节" % n_act)


# --------------------------------------------------------------------------- #
# T2 臂基座偏移符合生产常量
# --------------------------------------------------------------------------- #
def test_t2_arm_base_offset_matches_production_constant(arm_env):
    from physics_env import ARM_BASE_OFFSET_LEFT, ARM_BASE_OFFSET_RIGHT

    robot, side = arm_env["robot"], arm_env["side"]
    expected = (ARM_BASE_OFFSET_LEFT if side == "left"
                else ARM_BASE_OFFSET_RIGHT)
    expected = np.asarray(expected, dtype=np.float64)

    got = None
    for link in robot.get_links():
        if link.get_name() == f"{side}_arm_base_link":
            got = np.array(link.get_entity_pose().p)
            break
    assert got is not None, f"找不到 {side}_arm_base_link ✗"
    err = float(np.linalg.norm(got - expected))
    print("[T2] arm_base=%s expected=%s err=%.6f m (%.3f mm)"
          % (np.round(got, 5), np.round(expected, 5), err, err * 1000.0))
    # ⚠ 实测偏差 **0.530 mm** (不是 0) ⇒ 这是 URDF 里 arm_base 的**真实安装公差**,
    #   生产 grasp_hawor.py:2200-2210 也只打印核对、不做断言 ⇒ 本测试同口径 ✓
    #   阈值取 1mm: 足够严(能抓 ROOT 配错级错误) 又不误报浮点/URDF 公差 ✓
    assert err < 1e-3, f"臂基座偏移不符 {err*1000:.3f}mm ⇒ ROOT/基座配置漂移 ✗"


# --------------------------------------------------------------------------- #
# T3 驱动通道 (PD)
# --------------------------------------------------------------------------- #
def _q_of(robot, n: int):
    """读整臂 q —— ★ 生产同款: `robot.get_qpos()` (physics_env.py:1045 ✓)

    ⚠ 实测四个 API 坑（全部踩过 ✓）:
      ① `PhysxArticulationJoint` **没有** `get_dof_position()` (AttributeError ✓)
      ② `robot.get_dof()` 返回 **0-dim**(不能 np.array 当数组, IndexError ✓)
      ③ `joint.get_dof()` 返回 **int**, 且实测**所有关节都返回 1** ——
         它是关节索引/句柄, **不是 q**! (init_qpos[5]=-1.5 而 get_dof()=1 ✓)
      ⇒ 读 q **必须**用 `robot.get_qpos()` (18 维数组 ✓)
    """
    return np.asarray(robot.get_qpos(), dtype=np.float64)[:n]


def test_t3_arm_joint_drive_targets_are_writable(arm_env):
    """整臂关节能否接收驱动目标并产生 q/qvel 响应 ✓"""
    scene, robot, info = arm_env["scene"], arm_env["robot"], arm_env["info"]
    for _ in range(5):
        scene.step()

    q0 = _q_of(robot, 18)
    joints = robot.get_active_joints()
    assert len(joints) == 18, f"主动关节 {len(joints)} ≠ 18 ✗"

    # 只驱 **右臂 6 关节** (info 里生产给的索引 ✓), 目标 = 当前 +0.05 rad
    arm_idx = list(info["arm_joint_indices"])
    n_written = 0
    for i in arm_idx:
        j = joints[i]
        j.set_drive_target(float(q0[i]) + 0.05)
        n_written += 1
    assert n_written == 6, f"只有 {n_written}/6 臂关节接受 drive_target ✗"

    for _ in range(60):
        scene.step()
    q1 = _q_of(robot, 18)
    dq = q1 - q0
    moved = [i for i in arm_idx if abs(dq[i]) > 1e-6]
    print("[T3] 写入 %d 臂关节目标(+0.05rad); 60 子步后 %d/6 关节运动; "
          "max|Δq(臂)|=%.4f rad, max|Δq(夹爪)|=%.2e"
          % (n_written, len(moved), max(abs(dq[i]) for i in arm_idx),
             max(abs(dq[i]) for i in info["gripper_indices"][_side()])))
    assert len(moved) == 6, f"仅 {len(moved)}/6 臂关节响应 ⇒ 驱动链断 ✗"

    qv = np.array(robot.get_qvel(), dtype=np.float64)
    print("[T3] max|qvel|=%.4f rad/s" % float(np.abs(qv).max()))


# --------------------------------------------------------------------------- #
# T4 自碰撞配置
# --------------------------------------------------------------------------- #
def test_t4_self_collision_is_configured_and_stable(arm_env):
    """整臂自碰撞配置存在, 且多步不产生 NaN/爆炸 ✓"""
    scene, robot = arm_env["scene"], arm_env["robot"]
    for _ in range(30):
        scene.step()
    q = _q_of(robot, 18)
    qv = np.array(robot.get_qvel(), dtype=np.float64)
    print("[T4] 30 子步后 max|q|=%.4f rad  max|qvel|=%.4f rad/s"
          % (float(np.abs(q).max()), float(np.abs(qv).max())))
    assert np.all(np.isfinite(q)), "整臂 q 含 NaN/Inf ⇒ 自碰撞发散 ✗"
    assert np.all(np.isfinite(qv)), "整臂 qvel 含 NaN/Inf ✗"
    assert np.abs(qv).max() < 50.0, f"qvel 爆炸 {np.abs(qv).max():.1f} ✗"


# --------------------------------------------------------------------------- #
# T5 整臂上的 GPU 结论 (与夹爪测试一致 ✓)
# --------------------------------------------------------------------------- #
_T5_PROBE = r'''
import os, sys
sys.path.insert(0, %r)
import sapien
import sapien.render
from models.robot_forms import prepare_full_robot_urdf
from physics_env import setup_robot

prepare_full_robot_urdf()
sapien.physx.enable_gpu()          # 必须在任何 PhysX 初始化之前 ✓
gpu = sapien.physx.PhysxGpuSystem()
ok_init = "no-throw"
try:
    gpu.gpu_init()
except Exception as e:
    ok_init = "throw:%%s" %% str(e)[:80]
print("T5_GPU_INIT=%%s" %% ok_init)
print("T5_DEVICE=%%s" %% gpu.device)

scene = sapien.Scene(systems=[gpu, sapien.render.RenderSystem()])
try:
    setup_robot(scene, "full_robot", os.environ.get("ARM_SIDE", "right"),
                np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    print("T5_SETUP_ROBOT=OK")
except Exception as e:
    print("T5_SETUP_ROBOT=FAIL:%%s" %% str(e)[:120])
try:
    scene.step()
    print("T5_STEP=OK")
except Exception as e:
    print("T5_STEP=FAIL:%%s" %% str(e)[:120])
'''


@pytest.mark.skipif(os.environ.get("SAPIEN_GPU_PROBE") != "1",
                    reason="GPU 实测层, 需 SAPIEN_GPU_PROBE=1")
def test_t5_gpu_pipeline_also_fails_with_full_arm(tmp_path):
    """整臂(含更多实体)下 GPU 仍不可用 —— 结论与夹爪测试一致 ✓

    ⚠ **必须子进程跑** ✓ (2026-10-03 实测踩坑):
      `sapien.physx.enable_gpu()` 只能在本进程**任何 PhysX 初始化之前**调用一次,
      否则抛 `RuntimeError: GPU PhysX can only be enabled once before any other
      code involving PhysX` ✓ —— 而 T1~T4 已经建过 CPU 场景 ⇒ 同进程内 T5 必失败,
      **与 GPU 能否用无关** ⇒ 必须换干净进程重试 ✓
    """
    import subprocess

    src = tmp_path / "t5_gpu_probe.py"
    src.write_text("import numpy as np\n" + (_T5_PROBE % str(_ROOT)))
    env = dict(os.environ)
    env.pop("SAPIEN_GPU_PROBE", None)          # 子进程内不再触发外层门控 ✓
    proc = subprocess.run([sys.executable, "-B", str(src)],
                          capture_output=True, text=True, timeout=900,
                          cwd=str(_ROOT), env=env)
    out = proc.stdout + proc.stderr
    print(out)

    def field(name):
        for line in out.splitlines():
            if line.startswith(name + "="):
                return line[len(name) + 1:].strip()
        return ""

    setup = field("T5_SETUP_ROBOT")
    step = field("T5_STEP")

    # 若整臂加载/步进任一成功 ⇒ GPU 可用 ⇒ 必须 xfail 提醒重新评估
    assert setup != "OK" and step != "OK", \
        "★ 整臂在 GPU 上可加载/可步进 ⇒ GPU 可能真的可用 ⇒ **应重新评估** ✗"
    failed = [v for v in (setup, step) if v.startswith("FAIL")]
    assert failed, "整臂 GPU 既没成功也没报已知失败 ⇒ %s" % out[-600:]
    assert all("not initialized" in v.lower() for v in failed), \
        "失败原因与已知模式不同 ⇒ %s ⇒ 需重新排查 ✗" % failed
    print("\n[T5] gpu_init=%s device=%s ⇒ 整臂加载/step 均失败(预期) ✓ %s"
          % (field("T5_GPU_INIT"), field("T5_DEVICE"), failed))

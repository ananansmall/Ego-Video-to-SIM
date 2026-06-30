#!/usr/bin/env python3
"""tri_model_physics 主入口 — 三形式(完整机器人/浮动臂/纯夹爪) × 双引擎(SAPIEN/PyBullet) 物理仿真

用法:
    # 单形式+单引擎
    python run_tri_model.py --backend sapien --form floating_arm \\
        --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result

    # 一键运行全部6组合
    python run_tri_model.py --all --hawor-dir ... --ras-dir ...

    # 仅测试模型加载
    python run_tri_model.py --test-models

    # 指定帧数
    python run_tri_model.py --backend pybullet --form gripper_only \\
        --hawor-dir ... --ras-dir ... --num-frames 50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 设置 Vulkan
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
_intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
if 'VK_ICD_FILENAMES' not in os.environ:
    if os.path.exists(_nvidia_icd):
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if r.returncode == 0:
                os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
            else:
                os.environ['VK_ICD_FILENAMES'] = _intel_icd
        except Exception:
            os.environ['VK_ICD_FILENAMES'] = _intel_icd
    else:
        os.environ['VK_ICD_FILENAMES'] = _intel_icd

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = COMBINATION_DIR.parent.parent

# 添加路径
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(Path("/home/an/robot_world_ws/src/GalaxeaManipSim")))

from models.robot_forms import ALL_FORMS, get_robot_form_info, get_init_qpos
from physics_utils import (
    FULL_ROBOT_URDF, FLOATING_ARM_RIGHT_URDF, R1_MESH_DIR,
)

ALL_BACKENDS = ["sapien", "pybullet", "mujoco"]


def setup_logging(log_path=None):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        logging.getLogger().addHandler(fh)
    return logging.getLogger(__name__)


def test_models(logger):
    """测试三种形式的URDF加载与结构信息"""
    logger.info("=" * 60)
    logger.info("  测试: 三种机器人形式 URDF 加载")
    logger.info("=" * 60)

    results = {}
    for form_name in ALL_FORMS:
        for side in ["right", "left"]:
            try:
                info = get_robot_form_info(form_name, side)
                qpos_dict = get_init_qpos(form_name, side)
                logger.info(f"\n  [{form_name}/{side}]")
                logger.info(f"    URDF: {info.urdf_path}")
                logger.info(f"    描述: {info.description}")
                logger.info(f"    臂关节: {info.arm_joint_names}")
                logger.info(f"    夹爪关节: {info.gripper_joint_names}")
                logger.info(f"    初始qpos: {qpos_dict}")
                logger.info(f"    有臂: {info.has_arm}, 有夹爪: {info.has_gripper}, 浮动: {info.is_floating}")

                # 验证文件存在
                urdf_path = Path(info.urdf_path)
                if urdf_path.exists():
                    logger.info(f"    URDF文件: 存在")
                else:
                    logger.error(f"    URDF文件: 不存在!")

                results[f"{form_name}/{side}"] = True
            except Exception as e:
                logger.error(f"  [{form_name}/{side}] 失败: {e}")
                results[f"{form_name}/{side}"] = False

    logger.info(f"\n{'=' * 60}")
    success = sum(1 for v in results.values() if v)
    logger.info(f"  模型测试结果: {success}/{len(results)} 通过")
    return results


def run_single(form_name, backend, hawor_dir, ras_dir, transform_params_path,
               side="right", num_frames=-1, start_frame=0, headless=True,
               output_dir=None, logger=None):
    """运行单个形式+引擎组合

    Args:
        form_name: "full_robot" / "floating_arm" / "gripper_only"
        backend: "sapien" / "pybullet"
        hawor_dir: HaWoR 数据目录
        ras_dir: RAS 数据目录
        transform_params_path: 变换参数路径
        side: "right" / "left"
        num_frames: 帧数
        start_frame: 起始帧
        headless: 是否无头
        output_dir: 输出目录
        logger: 日志器

    Returns:
        bool: 是否成功
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  运行: {form_name} × {backend} ({side})")
    logger.info(f"{'=' * 60}")

    try:
        if backend == "sapien":
            from sapien_backend.sapien_runner import SapienRunner
            runner = SapienRunner(form_name, side, headless)
        elif backend == "pybullet":
            from pybullet_backend.pybullet_runner import PyBulletRunner
            runner = PyBulletRunner(form_name, side, headless)
        elif backend == "mujoco":
            from mujoco_backend.mujoco_runner import MuJoCoRunner
            runner = MuJoCoRunner(form_name, side, headless)
        else:
            logger.error(f"  未知后端: {backend}")
            return False

        # 视频输出路径
        video_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(output_dir / f"{form_name}_{backend}_{side}.mp4")

        # "SAPIEN 计算, 其他回放" 策略:
        # - sapien: 计算并保存 target_qpos (8 DOF)
        # - pybullet/mujoco: 加载 sapien 的 target_qpos 并回放
        target_qpos_trajectory = None
        if output_dir and backend in ("pybullet", "mujoco"):
            target_path = output_dir / f"{form_name}_sapien_{side}_target_qpos.npy"
            if target_path.exists():
                import numpy as np
                target_qpos_trajectory = np.load(str(target_path))
                logger.info(f"  加载 SAPIEN target_qpos: {target_path.name} shape={target_qpos_trajectory.shape}")

        runner.build()
        result = runner.run_tracking(
            hawor_dir, ras_dir, transform_params_path,
            start_frame=start_frame, num_frames=num_frames,
            output_video=video_path,
            target_qpos_trajectory=target_qpos_trajectory,
        )

        # 保存结果
        if output_dir:
            import numpy as np
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            qpos_arr = np.array(result["qpos_sequence"])
            np.save(str(output_dir / f"{form_name}_{backend}_{side}_qpos.npy"), qpos_arr)
            logger.info(f"  结果已保存: {output_dir / f'{form_name}_{backend}_{side}_qpos.npy'}")
            # sapien 额外保存 target_qpos (供 pybullet/mujoco 回放)
            if backend == "sapien" and "target_qpos_trajectory" in result:
                target_arr = np.array(result["target_qpos_trajectory"])
                np.save(str(output_dir / f"{form_name}_sapien_{side}_target_qpos.npy"), target_arr)
                logger.info(f"  target_qpos 已保存: {output_dir / f'{form_name}_sapien_{side}_target_qpos.npy'}")

        if backend == "pybullet":
            runner.disconnect()

        logger.info(f"  完成: {form_name} × {backend} | {len(result['qpos_sequence'])} 帧")
        return True

    except Exception as e:
        logger.error(f"  失败: {form_name} × {backend}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="三形式 × 双引擎 物理仿真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", type=str, default="sapien",
                        choices=ALL_BACKENDS, help="仿真后端")
    parser.add_argument("--form", type=str, default="floating_arm",
                        choices=ALL_FORMS, help="机器人形式")
    parser.add_argument("--side", type=str, default="right",
                        choices=["right", "left"], help="手臂侧别")
    parser.add_argument("--hawor-dir", type=str, default=None,
                        help="HaWoR 数据目录")
    parser.add_argument("--ras-dir", type=str, default=None,
                        help="RAS 数据目录")
    parser.add_argument("--transform-params", type=str, default=None,
                        help="变换参数 npz 路径")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="帧数 (-1=全部)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="起始帧")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="无头模式")
    parser.add_argument("--gui", action="store_true",
                        help="GUI模式 (覆盖headless)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--all", action="store_true",
                        help="运行全部9组合 (3形式×3引擎)")
    parser.add_argument("--test-models", action="store_true",
                        help="仅测试模型加载")
    parser.add_argument("--log", type=str, default=None,
                        help="日志文件路径")

    args = parser.parse_args()
    logger = setup_logging(args.log)

    # 仅测试模型
    if args.test_models:
        test_models(logger)
        return

    # 需要数据参数
    if args.hawor_dir is None or args.ras_dir is None:
        # 尝试默认路径
        default_hawor = "/home/an/data/hawor/7"
        default_ras = "/home/an/data/ras/my_7mp4_result"
        if os.path.exists(default_hawor) and os.path.exists(default_ras):
            args.hawor_dir = default_hawor
            args.ras_dir = default_ras
            logger.info(f"  使用默认数据路径")
        else:
            logger.error("  需要指定 --hawor-dir 和 --ras-dir")
            sys.exit(1)

    # 变换参数
    if args.transform_params is None:
        # 自动查找
        ras_path = Path(args.ras_dir)
        tp = ras_path / "alignment" / "transform_params.npz"
        if not tp.exists():
            # 在 combination/output 下查找
            combo_output = COMBINATION_DIR / "output"
            for d in combo_output.iterdir():
                tp_candidate = d / "alignment" / "transform_params.npz"
                if tp_candidate.exists():
                    tp = tp_candidate
                    break
        args.transform_params = str(tp)

    headless = not args.gui if args.gui else args.headless

    if args.all:
        # 运行全部9组合 (3形式×3引擎)
        # 注意: SAPIEN/PyBullet/MuJoCo 在同一进程中会因 GL 上下文冲突
        # (gladLoadGL error), 因此每个后端用独立子进程运行
        import subprocess
        results = {}
        log_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output"
        log_dir.mkdir(parents=True, exist_ok=True)
        for backend in ALL_BACKENDS:
            for form_name in ALL_FORMS:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"  子进程运行: {form_name} × {backend}")
                logger.info(f"{'=' * 60}")
                # 每个子进程单独的日志文件
                sub_log = str(log_dir / f"{form_name}_{backend}_{args.side}.log")
                cmd = [
                    sys.executable, str(SCRIPT_DIR / "run_tri_model.py"),
                    "--backend", backend,
                    "--form", form_name,
                    "--side", args.side,
                    "--hawor-dir", args.hawor_dir,
                    "--ras-dir", args.ras_dir,
                    "--transform-params", args.transform_params,
                    "--num-frames", str(args.num_frames),
                    "--start-frame", str(args.start_frame),
                    "--log", sub_log,
                ]
                if args.output_dir:
                    cmd += ["--output-dir", args.output_dir]
                if headless:
                    cmd += ["--headless"]

                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600,
                        cwd=str(SCRIPT_DIR),
                    )
                    ok = proc.returncode == 0
                    # 将子进程 stdout 也写入主日志
                    if proc.stdout:
                        for line in proc.stdout.strip().split('\n')[-10:]:
                            logger.info(f"    [{form_name}/{backend}] {line}")
                    if not ok:
                        err_lines = proc.stderr.strip().split('\n')[-5:]
                        logger.error(f"  失败: {form_name} × {backend}")
                        for line in err_lines:
                            logger.error(f"    {line}")
                    else:
                        logger.info(f"  完成: {form_name} × {backend} (日志: {sub_log})")
                except subprocess.TimeoutExpired:
                    logger.error(f"  超时: {form_name} × {backend}")
                    ok = False
                except Exception as e:
                    logger.error(f"  异常: {form_name} × {backend}: {e}")
                    ok = False

                results[f"{form_name}/{backend}"] = ok

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  全部组合结果:")
        for name, ok in results.items():
            logger.info(f"    {'✓' if ok else '✗'} {name}")
        logger.info(f"  通过: {sum(results.values())}/{len(results)}")
    else:
        run_single(
            args.form, args.backend,
            args.hawor_dir, args.ras_dir, args.transform_params,
            side=args.side, num_frames=args.num_frames,
            start_frame=args.start_frame, headless=headless,
            output_dir=args.output_dir, logger=logger,
        )


if __name__ == "__main__":
    main()

"""
path_config.py — 统一管理 Ego-Video-to-SIM 的所有依赖路径

设计原则:
  - 不使用任何硬编码绝对路径 (如 /home/an/...)
  - 所有路径相对项目根目录 (Ego-Video-to-SIM/) 计算
  - 子模块缺失时给出友好提示
  - 既支持 git submodule 安装, 也支持环境变量覆盖

用法:
  from path_config import LIBS_DIR, GALAXEA_SIM_PATH
  from path_config import setup_sys_path
  setup_sys_path()
"""

import os
import sys
from pathlib import Path

# =============================================================================
# 基础路径
# =============================================================================

# path_config.py 所在目录的父目录 = Ego-Video-to-SIM 项目根
EGO_VIDEO_TO_SIM_ROOT = Path(__file__).resolve().parent.parent

# libs/ 子目录: 自包含依赖库
LIBS_DIR = EGO_VIDEO_TO_SIM_ROOT / "libs"

# 各依赖库的本地路径
DEX_RETARGETING_PATH = LIBS_DIR / "dex_retargeting"
GALAXEA_SIM_PATH = LIBS_DIR / "galaxea_sim"
POSITION_RETARGETING_PATH = LIBS_DIR / "position_retargeting"

# HaWoR / ReplicateAnyScene (前向工具)
HAWOR_DIR = EGO_VIDEO_TO_SIM_ROOT / "HaWoR"
REPLICATE_ANY_SCENE_DIR = EGO_VIDEO_TO_SIM_ROOT / "ReplicateAnyScene"


# =============================================================================
# R1 机器人资源路径 (在 galaxea_sim 内)
# =============================================================================

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
R1_URDF_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs"
R1_SETTINGS_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs"

R1_FLOATING_RIGHT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_right.urdf"
R1_FLOATING_LEFT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_left.urdf"
R1_RIGHT_SETTINGS = R1_SETTINGS_DIR / "settings_right.yaml"
R1_LEFT_SETTINGS = R1_SETTINGS_DIR / "settings_left.yaml"


# =============================================================================
# MANO 手部模型路径 (在 position_retargeting/manopth 内)
# =============================================================================

MANO_PATH = POSITION_RETARGETING_PATH / "manopth" / "assets"
MANO_LEFT_PKL = MANO_PATH / "MANO_LEFT.pkl"
MANO_RIGHT_PKL = MANO_PATH / "MANO_RIGHT.pkl"


# =============================================================================
# 路径检查 + 错误提示
# =============================================================================

class PathConfigError(RuntimeError):
    """路径配置错误 (子模块未初始化)"""
    pass


def check_submodules(strict=True):
    """检查所有子模块是否已初始化

    Args:
        strict: True=失败时抛异常, False=仅打印警告

    Raises:
        PathConfigError: strict=True 且子模块缺失
    """
    missing = []
    for name, path in [
        ("libs/dex_retargeting", DEX_RETARGETING_PATH),
        ("libs/galaxea_sim", GALAXEA_SIM_PATH),
        ("libs/position_retargeting", POSITION_RETARGETING_PATH),
    ]:
        if not path.exists() or not any(path.iterdir()):
            missing.append(name)

    if missing:
        msg = (
            f"子模块未初始化: {', '.join(missing)}\n"
            f"请在项目根目录执行:\n"
            f"  git submodule update --init --recursive\n"
            f"或运行:\n"
            f"  bash scripts/setup.sh"
        )
        if strict:
            raise PathConfigError(msg)
        else:
            print(f"WARNING: {msg}")
    return len(missing) == 0


def setup_sys_path():
    """将所有依赖库路径加入 sys.path, 使 import 生效

    在 combination/*.py 文件开头调用:
        from path_config import setup_sys_path
        setup_sys_path()
    """
    check_submodules(strict=True)

    # 按优先级加入 sys.path
    paths = [
        str(EGO_VIDEO_TO_SIM_ROOT),                            # 顶层
        str(LIBS_DIR),                                          # libs/
        str(DEX_RETARGETING_PATH / "src"),                      # dex_retargeting
        str(POSITION_RETARGETING_PATH),                         # position_retargeting
        str(GALAXEA_SIM_PATH),                                  # galaxea_sim
        str(HAWOR_DIR),                                         # HaWoR
        str(REPLICATE_ANY_SCENE_DIR),                          # ReplicateAnyScene
    ]

    for p in paths:
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)


# =============================================================================
# 调试信息
# =============================================================================

if __name__ == "__main__":
    print(f"Ego-Video-to-SIM Root: {EGO_VIDEO_TO_SIM_ROOT}")
    print(f"libs/: {LIBS_DIR}")
    print(f"  dex_retargeting: {DEX_RETARGETING_PATH} "
          f"{'✓' if DEX_RETARGETING_PATH.exists() else '✗'}")
    print(f"  galaxea_sim:     {GALAXEA_SIM_PATH} "
          f"{'✓' if GALAXEA_SIM_PATH.exists() else '✗'}")
    print(f"  position_retargeting: {POSITION_RETARGETING_PATH} "
          f"{'✓' if POSITION_RETARGETING_PATH.exists() else '✗'}")
    print()
    print("R1 resources:")
    print(f"  URDF: {R1_FLOATING_RIGHT_URDF} "
          f"{'✓' if R1_FLOATING_RIGHT_URDF.exists() else '✗'}")
    print(f"  Settings: {R1_RIGHT_SETTINGS} "
          f"{'✓' if R1_RIGHT_SETTINGS.exists() else '✗'}")

    try:
        check_submodules(strict=False)
    except PathConfigError as e:
        print(f"\nERROR: {e}")

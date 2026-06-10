# Ego-Video-to-SIM

从第一人称操作视频到机器人仿真的项目集合。将 HaWoR 手部重建与 RAS 场景重建融合，实现人手操作到 R1 机器人仿真的完整映射。

## 快速开始

```bash
# 克隆（含子模块）
git clone --recursive https://github.com/ananansmall/Ego-Video-to-SIM.git
cd Ego-Video-to-SIM

# 一键初始化
bash scripts/setup.sh

# 运行管线
cd combination
python 00_run_pipeline.py --hawor-dir <path> --ras-dir <path>
```

## 子模块

| 子模块 | 说明 |
|--------|------|
| [combination/](combination/) | 完整仿真管线：场景对齐 → SAPIEN 渲染 → 物理仿真 → 视频对齐 |
| [pv_retargeting/](pv_retargeting/) | PV 重定向：R1 机器人手部跟踪与单臂跟随 |

## 依赖

### 本地依赖 (libs/)

| 库 | 来源 | 用途 |
|------|------|------|
| `dex_retargeting` | [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 手部到机器人夹爪映射 |
| `galaxea_sim` | [OpenGalaxea/GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim) | RelaxedIK + R1 机器人模型 |
| `position_retargeting` | dex-retargeting/example | MANO 手部模型层 |

### pip 安装

```bash
pip install numpy opencv-python sapien torch joblib pytransform3d tqdm trimesh scipy natsort matplotlib imageio-ffmpeg
```

### 上游项目

| 项目 | 用途 |
|------|------|
| [HaWoR](https://github.com/ubc-vision/HaWoR) | 手部重建 |
| [ReplicateAnyScene](https://github.com/ubc-vision/ReplicateAnyScene) | 场景重建 |
| [MANO](https://mano.is.tue.mpg.com/) | 手部网格模型 |

## 路径管理

`combination/path_config.py` 统一管理所有依赖路径，不使用硬编码绝对路径：

```python
# 在 combination/*.py 中使用:
from path_config import setup_sys_path, GALAXEA_SIM_PATH
setup_sys_path()  # 自动配置 sys.path
```

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: dex_retargeting` | `git submodule update --init --recursive` |
| `ModuleNotFoundError: relaxed_ik_solver` | 检查 `libs/galaxea_sim/` 是否存在 |
| 硬编码路径错误 | 已统一改为相对路径，通过 `path_config.py` 管理 |

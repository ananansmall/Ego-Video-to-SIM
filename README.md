# Ego-Video-to-SIM

从第一人称操作视频到机器人仿真的项目集合。将 HaWoR 手部重建与 RAS 场景重建融合，实现人手操作到 R1 机器人仿真的完整映射。

## Real2SimGripper 源码下载

[直接下载 Real2SimGripper 50ee6bf 源码 ZIP](https://raw.githubusercontent.com/ananansmall/Ego-Video-to-SIM/main/downloads/Real2SimGripper-50ee6bf.zip) · [浏览主仓里的源码目录](vendor/Real2SimGripper/)

ZIP 解压后直接得到 Real2SimGripper 目录，可从其他电脑匿名下载。公开版包含 R1 跟踪、仿真、示例脚本和文本配置。数据、模型权重和运行产物需要另行准备。文件范围、路径替换和来源见[导出清单](vendor/Real2SimGripper/SOURCE_EXPORT.json)。

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

## R1 Chairbottle 验收

本次报告和媒体随 [Real2SimGripper 子模块](https://github.com/ananansmall/Real2SimGripper/tree/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827) 发布。使用有子仓访问权限的 GitHub 账号打开文件。

- [Chairbottle 统一验收报告](https://github.com/ananansmall/Real2SimGripper/blob/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827/examples/r1_multicontact/runs/pixel-umi-ear-20261118/REPORT.md)
- [整段黑红四视图](https://github.com/ananansmall/Real2SimGripper/blob/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827/examples/r1_multicontact/runs/pixel-umi-ear-20261118/moving_views_joint_seat/moving_views.mp4)
- [独立机器人第一视角](https://github.com/ananansmall/Real2SimGripper/blob/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827/examples/r1_multicontact/runs/pixel-umi-ear-20261118/moving_views_joint_seat/r1_body_fpv.mp4)
- [独立右腕](https://github.com/ananansmall/Real2SimGripper/blob/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827/examples/r1_multicontact/runs/pixel-umi-ear-20261118/moving_views_joint_seat/r1_wrist_fpv.mp4)

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

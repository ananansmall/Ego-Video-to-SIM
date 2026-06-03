# Ego-Video-to-SIM

从第一人称操作视频到机器人仿真的项目集合。将 HaWoR 手部重建与 RAS 场景重建融合，实现人手操作到 R1 机器人仿真的完整映射。

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

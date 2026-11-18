# CHANGE_LOG

## [2026-06-30] 同步 dex-retargeting combination 最新代码

**类型**: 新增 + 更新
**影响范围**: combination/hand_track/, combination/04_physics_simulation.py, combination/tri_model_physics/

### 背景
从 `dex-retargeting/example/combination` 同步最新代码到 `Ego-Video-to-SIM/combination`。

### 修改内容

#### 1. hand_track/ 新增文件 (6 个)
- `align_strategy.py` — 新对齐策略: 先对齐夹爪两点 + 中点-手腕连线确定位姿
- `configs/r1_gripper_left.yml` — 左手夹爪配置
- `configs/r1_gripper_right.yml` — 右手夹爪配置
- `gripper_config.py` — 夹爪 URDF 生成与几何配置
- `render_dexterous_only.py` — 灵巧手渲染管线 (allegro/inspire/shadow/ability/leap/svh)
- `verify_optimizer_3points.py` — 3 点优化器验证脚本

#### 2. hand_track/ 更新文件 (6 个)
- `common.py` — 相机帧一致性修复 (R_AXIS 替代 RXWORLD_TO_SAPIEN)
- `README.md` — 新增灵巧手渲染说明 + 对齐策略文档
- `render_auto.py` — 支持 --dexterous 和 --robot-name 参数
- `render_gripper_only.py` — 集成新对齐策略 + 灵巧手支持
- `CHANGE_LOG.md` — 更新为完整变更历史
- `docs/questions.md` — 新增 Q&A 文档

#### 3. 04_physics_simulation.py 更新
- 从 dex-retargeting 同步最新版本 (171KB, 含物理仿真完整实现)

#### 4. tri_model_physics/ 新增目录 (完整物理仿真抓取模块)
- `grasp_hawor.py` — 核心抓取脚本 (SAPIEN 物理引擎, 支持 full_robot/gripper_only 模式)
- `grasp_demo.py` — 抓取演示
- `grasp_controller.py` — 抓取控制器
- `physics_utils.py` — 物理工具函数
- `trajectory_loader.py` — 轨迹加载器
- `video_recorder.py` — 视频录制器
- `analyze_files.py` — 文件分析工具
- `run_tri_model.py` — 三模型运行脚本
- `models/` — 机器人模型 (robot_forms.py, urdf_templates.py)
- `sapien_backend/` — SAPIEN 后端
- `pybullet_backend/` — PyBullet 后端
- `mujoco_backend/` — MuJoCo 后端
- `tests/` — 测试套件
- `docs/` — 文档 (grasp_hawor_analysis.md, questions.md)
- `CHANGE_LOG.md` — 完整变更历史
- `README.md` — 使用说明

### 验证
- 文件完整性: ✓ 所有新增文件已复制到目标仓库
- 目录结构: ✓ tri_model_physics/ 完整 (不含 output/__pycache__)
- 代码一致性: ✓ hand_track/ 与 dex-retargeting 版本一致
## [2026-11-18] 更新 Real2SimGripper 验收入口

- 将 Real2SimGripper 子模块更新到 `50ee6bf`。
- 总仓 README 新增 [Chairbottle 验收报告](https://github.com/ananansmall/Real2SimGripper/blob/50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827/examples/r1_multicontact/runs/pixel-umi-ear-20261118/REPORT.md) 和对应视频。

## [2026-11-19] 公开 Real2SimGripper 源码下载

- 在 vendor/Real2SimGripper 提供 50ee6bf 的源码快照。在 downloads 提供相同内容的 ZIP，主仓 README 提供下载链接。
- 校验 ZIP、普通源码目录、Python 语法和 JSON 配置，公开快照排除数据、二进制运行产物和内部记录。

## [2026-11-19] 更新 Real2SimGripper 公共源码快照

- 基于 4119071 及工作区改动，更新 vendor/Real2SimGripper 与独立下载 ZIP，包含原片入口、公共回放、轨迹报告、五样本文字证据。
- Python、Shell 与文本配置通过静态检查，两项接触窗测试及 R1、Stage 0、轨迹报告入口的帮助命令通过。

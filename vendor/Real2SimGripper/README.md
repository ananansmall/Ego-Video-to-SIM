# Real2SimGripper 当前源码快照

本版源自仿真工作区，基线提交为 411907188b7befd85b582ec66ff1bd6949eae2a4。包含截至导出时工作区中的相关未提交改动和新增入口。每份文件的来源和 SHA-256 见 [SOURCE_EXPORT.json](SOURCE_EXPORT.json)。

| 内容 | 入口 |
| --- | --- |
| 原片注册、RAS、EgoFound3R、Stage 0 和接触窗衔接 | [run_video_to_r1.sh](examples/egoexplore/video_to_r1/run_video_to_r1.sh) |
| R1 连续物理回放、接触与承重 | [replay_multicontact.py](examples/r1_multicontact/replay_multicontact.py) |
| 分阶段轨迹误差 | [unified_trajectory_report.py](scripts/unified_trajectory_report.py) |
| 五样本实测与 EgoDex 桌面对照 | [五样本报告](examples/r1_multicontact/runs/five-case-review/REPORT.md) |

运行原片链需要 RAS、EgoFound3R、AGILE 对应的环境和模型，以及可用的样本配置与注册输入。R1 回放还需要 SAPIEN 环境、R1 资产和包含 paired_targets.npz 与 geometry.json 的输入包。请将 /path/to/ 下的占位路径改成实际路径。版本包用于下载和审阅代码。报告中的历史实测和当前入口的限制按原证据记载。

接触窗合并可用以下方式快速验证。需要 Python、NumPy 和 SciPy。

```bash
python examples/egoexplore/video_to_r1/contact_merge_smoke.py
python examples/egoexplore/video_to_r1/stage2_contact_smoke.py
```

[RoboTwin MIT 许可声明](LICENSE-RoboTwin) · [Ego-Video-to-SIM 许可证](LICENSE)

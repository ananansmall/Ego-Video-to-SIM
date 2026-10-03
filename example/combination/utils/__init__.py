"""
utils — 手部检测、机械臂配置、数据加载等共享工具

核心模块已迁移至 hand_track/:
  - hand_detector:    从 HaWoR 数据自动检测手部类型 (左手/右手/双手)
  - robot_arm_config: 根据手部类型生成对应的机械臂配置

使用方式:
  sys.path.insert(0, str(Path(...) / "hand_track"))
  from hand_detector import HandDetector, Handedness
  from robot_arm_config import RobotArmMapper
"""

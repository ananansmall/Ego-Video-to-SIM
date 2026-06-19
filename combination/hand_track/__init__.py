"""
hand_track — 手部检测 + 机械臂映射测试模块

模块:
  - hand_detector:    从 HaWoR 数据自动检测手部类型 (左手/右手/双手)
  - robot_arm_config: 根据手部类型生成对应的机械臂配置 (URDF/关节/IK)
  - run_all_hawor:    批量自动检测 + 映射管线 (基于 03_track_robot.py 逻辑)
  - render_auto:      自动检测手部 + GLB 场景渲染 (基于 02_render_scene.py 逻辑)
  - test_pipeline:    手部检测 + 机械臂映射验证测试

用法:
  # 批量处理所有 hawor 目录
  python hand_track/run_all_hawor.py --no-render

  # 处理指定目录
  python hand_track/run_all_hawor.py --hawor-dirs /home/an/data/hawor/7 /home/an/data/hawor/laptop

  # 运行检测测试
  python hand_track/test_pipeline.py
"""

"""tri_model_physics — 三形式(完整机器人/浮动臂/纯夹爪) × 双引擎(SAPIEN/PyBullet) 物理仿真

参考:
  - combination/02_render_scene.py: run_robot_tracking 轨迹加载与夹爪映射
  - combination/04_physics_simulation.py: SAPIEN PD驱动物理仿真
  - combination/physics_pipeline/pybullet_pipeline.py: PyBullet物理管线
  - combination/hand_track/render_gripper_only.py: 纯夹爪解析映射
  - GalaxeaManipSim/galaxea_sim/assets/r1: URDF模型源
  - malik-group/do-as-i-do: 阶段化管线 + 凸分解碰撞mesh思路
"""

__version__ = "0.1.0"

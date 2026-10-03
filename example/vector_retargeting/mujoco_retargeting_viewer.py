"""
使用 MuJoCo 可视化机器人手部重定向轨迹

功能：
- 读取 detect_from_video.py 生成的 pickle 文件
- 在 MuJoCo 中加载机器人模型（URDF）
- 逐帧播放机器人关节运动
- 支持实时查看和速度控制

使用方法：
    python mujoco_retargeting_viewer.py \
        --pickle-path data/panda_joints.pkl \
        --robot-name panda
"""

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError(
        "MuJoCo is not installed. Install it with:\n"
        "pip install mujoco"
    )

from dex_retargeting.constants import RobotName, ROBOT_NAME_MAP
from dex_retargeting.retargeting_config import RetargetingConfig


class MuJoCoRetargetingViewer:
    """MuJoCo 重定向轨迹查看器"""
    
    def __init__(self, pickle_path: str, robot_name: str, urdf_path: Optional[str] = None):
        """
        初始化查看器
        
        Args:
            pickle_path: pickle 文件路径（包含关节轨迹数据）
            robot_name: 机器人名称（如 panda, allegro 等）
            urdf_path: 可选的 URDF 路径，如果不提供则从配置中自动查找
        """
        self.pickle_path = pickle_path
        self.robot_name = robot_name
        
        # 加载轨迹数据
        print(f"加载轨迹数据: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        self.meta_data = pickle_data['meta_data']
        self.joint_trajectory = pickle_data['data']
        self.joint_names = self.meta_data['joint_names']
        
        print(f"轨迹帧数: {len(self.joint_trajectory)}")
        print(f"关节数量: {len(self.joint_names)}")
        print(f"关节名称: {self.joint_names}")
        
        # 确定 URDF 路径
        if urdf_path is None:
            # 从 meta_data 中获取配置路径
            config_path = self.meta_data.get('config_path')
            if config_path:
                # 设置 URDF 搜索目录
                robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
                RetargetingConfig.set_default_urdf_dir(str(robot_dir))
                
                config = RetargetingConfig.load_from_file(config_path)
                urdf_path = config.urdf_path
                
                # MuJoCo 需要使用绝对路径，并且要处理网格文件路径
                # 优先使用 GLB 格式的 URDF（如果存在）
                urdf_path_obj = Path(urdf_path)
                if not urdf_path_obj.is_absolute():
                    urdf_path_obj = urdf_path_obj.absolute()
                
                # 检查是否有对应的 _glb.urdf 版本
                glb_urdf_path = str(urdf_path_obj).replace('.urdf', '_glb.urdf')
                if Path(glb_urdf_path).exists():
                    urdf_path = glb_urdf_path
                    print(f"使用 GLB 格式 URDF: {urdf_path}")
                else:
                    urdf_path = str(urdf_path_obj)
            else:
                # 默认路径
                robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
                robot_folder = ROBOT_NAME_MAP.get(RobotName[robot_name], robot_name)
                urdf_path = str(robot_dir / robot_folder / f"{robot_folder}_glb.urdf")
        
        self.urdf_path = urdf_path
        print(f"URDF 路径: {urdf_path}")
        
        # 加载 MuJoCo 模型（从 URDF 转换）
        print("初始化 MuJoCo...")
        
        # 获取 URDF 所在目录作为 asset 目录
        urdf_dir = Path(self.urdf_path).parent
        
        # 使用 mujoco.mj_loadXML 加载 URDF（支持 URDF 格式）
        import tempfile
        import shutil
        
        # 创建一个临时的 MJCF 文件，使用 compiler 指定 meshdir
        with tempfile.TemporaryDirectory() as tmpdir:
            # 复制 URDF 到临时目录
            urdf_filename = Path(self.urdf_path).name
            temp_urdf_path = Path(tmpdir) / urdf_filename
            shutil.copy2(self.urdf_path, temp_urdf_path)
            
            # 如果有 meshes 目录，也复制过去
            meshes_dir = urdf_dir / "meshes"
            if meshes_dir.exists():
                temp_meshes_dir = Path(tmpdir) / "meshes"
                shutil.copytree(meshes_dir, temp_meshes_dir)
            
            # 尝试直接加载 URDF
            try:
                self.model = mujoco.MjModel.from_xml_path(str(temp_urdf_path))
                self.data = mujoco.MjData(self.model)
                print("MuJoCo 模型加载成功")
            except Exception as e:
                print(f"MuJoCo 加载 URDF 失败: {e}")
                print("\n提示: MuJoCo 对 URDF 的支持有限，建议使用 SAPIEN 渲染器")
                print("或者将 URDF 转换为 MuJoCo 的 MJCF 格式")
                raise
        
        # 设置明亮照明
        self.configure_lighting(bright=True)
        
        # 获取 MuJoCo 关节 ID 映射
        self.joint_id_map = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and joint_name in self.joint_names:
                self.joint_id_map[joint_name] = i
        
        print(f"找到 {len(self.joint_id_map)} 个可控制关节")
        
        # 仿真参数
        self.sim_speed = 1.0  # 仿真速度倍数
        self.frame_delay = 1.0 / 30.0  # 每帧延迟（假设30fps）
    
    def set_joint_positions(self, qpos):
        """
        设置机器人关节位置
        
        Args:
            qpos: 关节角度数组
        """
        for name, angle in zip(self.joint_names, qpos):
            if name in self.joint_id_map:
                joint_id = self.joint_id_map[name]
                # MuJoCo 中直接设置 qpos
                joint_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[joint_addr] = angle
    
    def play_trajectory(self):
        """播放完整的关节轨迹"""
        print("\n开始播放轨迹...")
        print("控制说明:")
        print("  - SPACE: 暂停/继续")
        print("  - R: 重新开始")
        print("  - +/-: 调整播放速度")
        print("  - L: 切换照明模式")
        print("  - ESC: 退出\n")
        
        paused = False
        restart = False
        current_frame = 0
        bright_light = True  # 明亮照明模式
        trajectory_finished = False  # 标记轨迹是否已播放完成
        
        def key_callback(keycode):
            nonlocal paused, restart, current_frame, bright_light
            
            if keycode == 32:  # SPACE
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"[{status}] 当前帧: {current_frame}/{len(self.joint_trajectory)}")
            
            elif keycode == 82:  # R
                restart = True
                current_frame = 0
                trajectory_finished = False  # 重置完成标记
                print("[重置] 从头开始播放")
            
            elif keycode == 61 or keycode == 43:  # + or =
                self.sim_speed = min(self.sim_speed * 1.2, 5.0)
                print(f"[速度] {self.sim_speed:.1f}x")
            
            elif keycode == 45:  # -
                self.sim_speed = max(self.sim_speed / 1.2, 0.1)
                print(f"[速度] {self.sim_speed:.1f}x")
            
            elif keycode == 76:  # L - 切换照明
                bright_light = not bright_light
                self.configure_lighting(bright_light)
                mode = "明亮" if bright_light else "标准"
                print(f"[照明] {mode}模式")
        
        # 使用 MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # 设置初始相机视角为左视图
            self.set_initial_camera_view(viewer)
            
            while viewer.is_running():
                if restart:
                    current_frame = 0
                    restart = False
                
                if not paused and current_frame < len(self.joint_trajectory):
                    # 获取当前帧的关节角度
                    qpos = self.joint_trajectory[current_frame]
                    
                    # 设置关节位置
                    self.set_joint_positions(qpos)
                    
                    # 更新仿真
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # 控制播放速度
                    time.sleep(self.frame_delay / self.sim_speed)
                    
                    # 打印进度（每10帧）
                    if current_frame % 10 == 0:
                        print(f"帧: {current_frame}/{len(self.joint_trajectory)}", end='\r')
                    
                    current_frame += 1
                
                elif current_frame >= len(self.joint_trajectory):
                    # 只在第一次完成时输出提示
                    if not trajectory_finished:
                        print(f"\n播放完成！共 {len(self.joint_trajectory)} 帧")
                        print("按 R 重新播放，或关闭窗口退出")
                        trajectory_finished = True
                    # 保持最后一帧
                    viewer.sync()
                    time.sleep(0.1)
                
                else:
                    # 暂停状态
                    viewer.sync()
                    time.sleep(0.05)
    
    def set_initial_camera_view(self, viewer):
        """
        设置初始相机视角
        
        Args:
            viewer: MuJoCo viewer 对象
        """
        # 获取相机对象
        cam = viewer.cam
        
        # 设置相机类型为跟踪模式
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
        # 设置相机位置（左视图）
        # azimuth: 水平旋转角度（度），0=后，90=左，180=前，270=右
        # elevation: 垂直仰角（度），0=水平，90=俯视
        # distance: 相机距离目标的距离
        cam.azimuth = 90    # 左视图
        cam.elevation = -20  # 稍微向下看
        cam.distance = 1.5   # 距离
        
        # 设置相机跟踪的目标（通常是机器人基座或第一个body）
        if self.model.nbody > 0:
            cam.trackbodyid = 1  # 跟踪第一个body
        
        print("相机视角已设置为左视图")
        print("提示: 可以用鼠标拖动调整视角，按 Ctrl+R 重置相机")
    
    def configure_lighting(self, bright=True):
        """
        配置场景照明
        
        Args:
            bright: True 为明亮模式，False 为标准模式
        """
        if bright:
            # 明亮模式：增强环境光和方向光
            self.model.light_active[:] = 1  # 激活所有光源
            
            # 设置更强的环境光
            self.model.light_ambient[:] = [0.6, 0.6, 0.6]  # 环境光强度
            
            # 设置更强的漫反射光
            self.model.light_diffuse[:] = [1.0, 1.0, 1.0]  # 漫反射强度
            
            # 设置 specular 高光
            self.model.light_specular[:] = [0.5, 0.5, 0.5]
            
            # 如果有光源，设置为白色强光
            for i in range(self.model.nlight):
                self.model.light_diffuse[i] = [1.5, 1.5, 1.5]
                self.model.light_ambient[i] = [0.5, 0.5, 0.5]
        else:
            # 标准模式：MuJoCo 默认照明
            self.model.light_ambient[:] = [0.3, 0.3, 0.3]
            self.model.light_diffuse[:] = [0.7, 0.7, 0.7]
            self.model.light_specular[:] = [0.3, 0.3, 0.3]


def main(
    pickle_path: str,
    robot_name: str = "panda",
    urdf_path: Optional[str] = None,
):
    """
    使用 MuJoCo 可视化机器人重定向轨迹
    
    Args:
        pickle_path: pickle 文件路径，由 detect_from_video.py 生成
        robot_name: 机器人名称（panda, allegro, shadow 等）
        urdf_path: 可选的 URDF 文件路径，不提供则自动查找
    """
    # 创建查看器
    viewer = MuJoCoRetargetingViewer(
        pickle_path=pickle_path,
        robot_name=robot_name,
        urdf_path=urdf_path
    )
    
    # 播放轨迹
    viewer.play_trajectory()


if __name__ == "__main__":
    tyro.cli(main)

"""
在 MuJoCo 中可视化 DexYCB 数据集的 YCB 物体

功能：
- 加载 DexYCB 数据集中的 YCB 物体网格
- 根据数据集中的位姿信息放置物体
- 在 MuJoCo 中实时渲染

使用方法：
    python mujoco_ycb_objects.py \
        --dexycb-dir /home/an/data/DexYCB \
        --data-id 0
"""

import sys
import time
import tempfile
from pathlib import Path

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

from dataset import DexYCBVideoDataset, YCB_CLASSES


class YCBObjectsViewer:
    """YCB 物体 MuJoCo 查看器"""
    
    def __init__(self, dexycb_dir: str, data_id: int = 0):
        """
        初始化查看器
        
        Args:
            dexycb_dir: DexYCB 数据集根目录
            data_id: 要可视化的数据索引
        """
        self.dexycb_dir = Path(dexycb_dir)
        self.data_id = data_id
        
        # 加载 DexYCB 数据集
        print(f"加载 DexYCB 数据集: {dexycb_dir}")
        self.dataset = DexYCBVideoDataset(self.dexycb_dir, hand_type="right")
        print(f"数据集大小: {len(self.dataset)} 条轨迹")
        
        if data_id >= len(self.dataset):
            raise ValueError(f"data_id {data_id} 超出范围 (0-{len(self.dataset)-1})")
        
        # 加载指定数据
        print(f"\n加载第 {data_id} 条轨迹...")
        self.sampled_data = self.dataset[data_id]
        
        # 打印数据信息
        print(f"捕获名称: {self.sampled_data['capture_name']}")
        print(f"物体ID: {self.sampled_data['ycb_ids']}")
        print(f"物体名称: {[YCB_CLASSES[yid] for yid in self.sampled_data['ycb_ids']]}")
        print(f"手部姿态帧数: {self.sampled_data['hand_pose'].shape[0]}")
        print(f"物体数量: {len(self.sampled_data['ycb_ids'])}")
        
        # 获取物体信息
        self.ycb_ids = self.sampled_data['ycb_ids']
        self.object_mesh_files = self.sampled_data['object_mesh_file']
        self.object_poses = self.sampled_data['object_pose']  # (frames, num_objects, 7)
        self.extrinsics = self.sampled_data['extrinsics']  # (4, 4)
        
        # 创建 MuJoCo XML 模型
        self._create_mujoco_model()
        
        # 配置明亮照明
        self.configure_lighting(bright=True)
    
    def _create_mujoco_model(self):
        """创建包含 YCB 物体的 MuJoCo XML 模型"""
        import tempfile
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="mujoco_ycb_")
        temp_path = Path(self.temp_dir)
        
        # 复制所有物体网格到临时目录
        mesh_paths = []
        for i, (ycb_id, mesh_file) in enumerate(zip(self.ycb_ids, self.object_mesh_files)):
            mesh_path = Path(mesh_file)
            if mesh_path.exists():
                # 复制到临时目录
                dest_mesh = temp_path / f"obj_{i}_{mesh_path.name}"
                import shutil
                shutil.copy2(mesh_path, dest_mesh)
                mesh_paths.append(dest_mesh)
                print(f"复制物体 {i}: {YCB_CLASSES[ycb_id]} -> {dest_mesh.name}")
            else:
                print(f"警告: 物体网格文件不存在: {mesh_file}")
                mesh_paths.append(None)
        
        # 生成 MJCF XML
        xml_content = self._generate_mjcf_xml(mesh_paths)
        xml_file = temp_path / "model.xml"
        with open(xml_file, 'w') as f:
            f.write(xml_content)
        
        print(f"\n生成 MuJoCo 模型: {xml_file}")
        
        # 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(str(xml_file))
            self.data = mujoco.MjData(self.model)
            print("MuJoCo 模型加载成功")
        except Exception as e:
            print(f"MuJoCo 加载失败: {e}")
            raise
    
    def _generate_mjcf_xml(self, mesh_paths):
        """生成 MJCF XML 内容"""
        object_names = [YCB_CLASSES[yid] for yid in self.ycb_ids]
        
        # 生成 asset 部分（网格定义）
        assets_xml = ""
        for i, (mesh_path, obj_name) in enumerate(zip(mesh_paths, object_names)):
            if mesh_path:
                assets_xml += f'        <mesh name="obj_{i}" file="{mesh_path.name}"/>\n'
        
        # 生成 worldbody 部分（物体实例）
        bodies_xml = ""
        for i, obj_name in enumerate(object_names):
            bodies_xml += f'''
        <body name="object_{i}" pos="0 0 0">
            <geom type="mesh" mesh="obj_{i}" rgba="0.8 0.8 0.8 1"/>
            <joint name="obj_{i}_joint" type="free"/>
        </body>'''
        
        # 完整的 MJCF
        mjcf = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="ycb_objects">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    
    <asset>
{assets_xml}
    </asset>
    
    <worldbody>
        <!-- 地面 -->
        <geom name="floor" type="plane" pos="0 0 -0.5" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>
        
        <!-- 光源 -->
        <light name="light1" pos="1 1 2" dir="-1 -1 -2" directional="false"/>
        <light name="light2" pos="-1 -1 2" dir="1 1 -2" directional="false"/>
        
        <!-- YCB 物体 -->
{bodies_xml}
    </worldbody>
</mujoco>'''
        
        return mjcf
    
    def set_object_poses(self, frame_idx):
        """
        设置物体位姿
        
        Args:
            frame_idx: 帧索引
        """
        if frame_idx >= self.object_poses.shape[0]:
            return
        
        # 获取当前帧的物体位姿
        object_pose_frame = self.object_poses[frame_idx]  # (num_objects, 7)
        
        # 相机外参（用于转换坐标系）
        extrinsic_mat = self.extrinsics
        
        for i in range(len(self.ycb_ids)):
            pos_quat = object_pose_frame[i]  # [rx, ry, rz, x, y, z]
            
            # DexYCB 的格式: 前3维是旋转轴角，后3维是平移
            # 需要转换为 MuJoCo 的四元数格式
            from pytransform3d import rotations
            
            # 轴角转四元数
            axis_angle = pos_quat[:3]
            angle = np.linalg.norm(axis_angle)
            if angle > 1e-6:
                axis = axis_angle / angle
                quat_xyzw = rotations.quaternion_from_axis_angle(axis * angle)
            else:
                quat_xyzw = np.array([1, 0, 0, 0])
            
            # 平移
            trans = pos_quat[3:6]
            
            # 应用相机外参转换
            # 将物体从相机坐标系转换到世界坐标系
            trans_homogeneous = np.append(trans, 1)
            trans_world = extrinsic_mat @ trans_homogeneous
            trans_world = trans_world[:3]
            
            # 设置 MuJoCo 中的物体位姿
            joint_name = f"obj_{i}_joint"
            joint_id = self.model.joint(joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            
            # MuJoCo 的 free joint qpos 格式: [x, y, z, qw, qx, qy, qz]
            # 注意：四元数需要从 xyzw 转换为 wxyz
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            self.data.qpos[qpos_addr:qpos_addr+3] = trans_world
            self.data.qpos[qpos_addr+3:qpos_addr+7] = quat_wxyz
    
    def play_trajectory(self, fps=10):
        """播放物体运动轨迹"""
        print(f"\n开始播放 YCB 物体轨迹...")
        print(f"播放速度: {fps} FPS")
        print("控制说明:")
        print("  - SPACE: 暂停/继续")
        print("  - R: 重新开始")
        print("  - +/-: 调整播放速度")
        print("  - L: 切换照明模式")
        print("  - ESC: 退出\n")
        
        paused = False
        restart = False
        current_frame = 0
        bright_light = True
        trajectory_finished = False
        sim_speed = 1.0
        frame_delay = 1.0 / fps
        
        def key_callback(keycode):
            nonlocal paused, restart, current_frame, bright_light, sim_speed
            
            if keycode == 32:  # SPACE
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"[{status}] 当前帧: {current_frame}/{self.object_poses.shape[0]}")
            
            elif keycode == 82:  # R
                restart = True
                current_frame = 0
                trajectory_finished = False
                print("[重置] 从头开始播放")
            
            elif keycode == 61 or keycode == 43:  # + or =
                sim_speed = min(sim_speed * 1.2, 5.0)
                print(f"[速度] {sim_speed:.1f}x")
            
            elif keycode == 45:  # -
                sim_speed = max(sim_speed / 1.2, 0.1)
                print(f"[速度] {sim_speed:.1f}x")
            
            elif keycode == 76:  # L
                bright_light = not bright_light
                self.configure_lighting(bright_light)
                mode = "明亮" if bright_light else "标准"
                print(f"[照明] {mode}模式")
        
        # 使用 MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # 设置初始相机视角
            self.set_initial_camera_view(viewer)
            
            num_frames = self.object_poses.shape[0]
            
            while viewer.is_running():
                if restart:
                    current_frame = 0
                    restart = False
                
                if not paused and current_frame < num_frames:
                    # 设置物体位姿
                    self.set_object_poses(current_frame)
                    
                    # 更新仿真
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # 控制播放速度
                    time.sleep(frame_delay / sim_speed)
                    
                    # 打印进度
                    if current_frame % 10 == 0:
                        print(f"帧: {current_frame}/{num_frames}", end='\r')
                    
                    current_frame += 1
                
                elif current_frame >= num_frames:
                    if not trajectory_finished:
                        print(f"\n播放完成！共 {num_frames} 帧")
                        print("按 R 重新播放，或关闭窗口退出")
                        trajectory_finished = True
                    viewer.sync()
                    time.sleep(0.1)
                
                else:
                    viewer.sync()
                    time.sleep(0.05)
    
    def set_initial_camera_view(self, viewer):
        """设置初始相机视角"""
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.azimuth = 90    # 左视图
        cam.elevation = -20
        cam.distance = 1.5
        
        if self.model.nbody > 0:
            cam.trackbodyid = 1
        
        print("相机视角已设置为左视图")
        print("提示: 可以用鼠标拖动调整视角，按 Ctrl+R 重置相机")
    
    def configure_lighting(self, bright=True):
        """配置场景照明"""
        if bright:
            self.model.light_active[:] = 1
            self.model.light_ambient[:] = [0.6, 0.6, 0.6]
            self.model.light_diffuse[:] = [1.0, 1.0, 1.0]
            self.model.light_specular[:] = [0.5, 0.5, 0.5]
            
            for i in range(self.model.nlight):
                self.model.light_diffuse[i] = [1.5, 1.5, 1.5]
                self.model.light_ambient[i] = [0.5, 0.5, 0.5]
        else:
            self.model.light_ambient[:] = [0.3, 0.3, 0.3]
            self.model.light_diffuse[:] = [0.7, 0.7, 0.7]
            self.model.light_specular[:] = [0.3, 0.3, 0.3]


def main(
    dexycb_dir: str,
    data_id: int = 0,
    fps: int = 10
):
    """
    使用 MuJoCo 可视化 DexYCB 数据集的 YCB 物体
    
    Args:
        dexycb_dir: DexYCB 数据集根目录
        data_id: 要可视化的数据索引
        fps: 播放帧率
    """
    # 创建查看器
    viewer = YCBObjectsViewer(
        dexycb_dir=dexycb_dir,
        data_id=data_id
    )
    
    # 播放轨迹
    viewer.play_trajectory(fps=fps)


if __name__ == "__main__":
    tyro.cli(main)

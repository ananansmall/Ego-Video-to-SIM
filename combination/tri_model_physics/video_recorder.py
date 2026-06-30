"""视频录制器 — SAPIEN/PyBullet/MuJoCo 通用

支持:
  - SAPIEN: 通过 CameraEntity + scene.update_render() + get_picture()
  - PyBullet: 通过 getCameraImage()
  - MuJoCo: 通过 mujoco.MjData + renderer
  - 使用 imageio 保存 mp4
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """通用视频录制器"""

    def __init__(self, output_path, fps=30, resolution=(640, 480)):
        """
        Args:
            output_path: 输出mp4路径
            fps: 帧率
            resolution: (width, height)
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.width, self.height = resolution
        self.frames = []
        self._writer = None

    def add_frame(self, rgb_array):
        """添加一帧 (RGB, HWC)"""
        if rgb_array is None:
            return
        # 确保是 uint8
        if rgb_array.dtype != np.uint8:
            rgb_array = (rgb_array * 255).astype(np.uint8) if rgb_array.max() <= 1.0 else rgb_array.astype(np.uint8)
        self.frames.append(rgb_array)

    def save(self):
        """保存视频"""
        if not self.frames:
            logger.warning("  无帧可保存")
            return False

        try:
            import imageio
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self.output_path, self.frames, fps=self.fps, codec='libx264')
            logger.info(f"  视频已保存: {self.output_path} ({len(self.frames)} 帧, {self.fps}fps)")
            self.frames = []
            return True
        except Exception as e:
            logger.error(f"  视频保存失败: {e}")
            return False


class SapienCamera:
    """SAPIEN 相机 (对齐 02_render_scene.py 的相机配置)"""

    def __init__(self, scene, pos=(2, 0, 1), target=(0, 0, 0),
                 resolution=(1920, 1080), fovy=None, focal=600.0):
        """初始化 SAPIEN 相机

        Args:
            scene: SAPIEN scene
            pos: 相机位置
            target: 相机看向的目标点 (仅用于初始化, 每帧会由 set_pose 覆盖)
            resolution: (width, height)
            fovy: 垂直视场角 (弧度). 若为 None, 则根据 focal 和 height 计算
            focal: HaWoR 焦距 (像素), 用于计算 fovy
        """
        import sapien
        self.scene = scene
        # FOV 计算: 对齐 02_render_scene.py 的 cam_fov = 2*arctan(height/2/focal)
        if fovy is None:
            fovy = 2 * np.arctan(resolution[1] / 2.0 / focal)
        self.camera = scene.add_camera(
            name="video_camera",
            width=resolution[0],
            height=resolution[1],
            fovy=fovy,
            near=0.01,
            far=100,
        )
        # 使用 set_local_pose (对齐 02_render_scene.py)
        pos = np.array(pos, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        self.camera.set_local_pose(sapien.Pose(pos.tolist(), self._look_at_quat(pos, target).tolist()))

    @staticmethod
    def _look_at_quat(pos, target):
        """计算从pos看向target的四元数 (wxyz)

        参考 02_render_scene.py 的 make_look_at_camera:
        SAPIEN 相机坐标系: x=forward(看向目标), y=-right, z=up
        """
        forward = target - pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        # 旋转矩阵: x=forward, y=-right, z=up
        R = np.eye(3)
        R[:, 0] = forward
        R[:, 1] = -right
        R[:, 2] = cam_up

        # 转为四元数 (wxyz)
        from pytransform3d import rotations as pr
        return pr.quaternion_from_matrix(R)

    def set_pose(self, pos, quat):
        """设置相机位姿 (位置 + 四元数 wxyz)

        供 runner 每帧动态更新相机位姿使用 (如 HaWoR 相机轨迹).
        使用 set_local_pose 对齐 02_render_scene.py.
        """
        import sapien
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        self.camera.set_local_pose(sapien.Pose(pos.tolist(), quat.tolist()))

    def capture(self):
        """捕获一帧"""
        self.scene.update_render()
        self.camera.take_picture()
        rgb = self.camera.get_picture("Color")[..., :3]  # (H, W, 3)
        return (rgb * 255).astype(np.uint8)


class PyBulletCamera:
    """PyBullet 相机"""

    def __init__(self, physics_client, pos=(2, 0, 1), target=(0, 0, 0), resolution=(640, 480)):
        import pybullet as p
        self.p = p
        self.client = physics_client
        self.pos = list(pos)
        self.target = list(target)
        self.resolution = resolution

    def capture(self):
        """捕获一帧"""
        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=self.pos,
            cameraTargetPosition=self.target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.client,
        )
        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=45, aspect=self.resolution[0]/self.resolution[1],
            nearVal=0.1, farVal=100,
            physicsClientId=self.client,
        )
        # 在无头 DIRECT 模式下优先使用 CPU 软件渲染 (TINY_RENDERER),
        # 硬件 OpenGL 渲染器在无头环境下容易返回空白/白屏。
        try:
            _, _, px, _, _ = self.p.getCameraImage(
                self.resolution[0], self.resolution[1],
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=self.p.ER_TINY_RENDERER,
                physicsClientId=self.client,
            )
        except Exception:
            _, _, px, _, _ = self.p.getCameraImage(
                self.resolution[0], self.resolution[1],
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client,
            )
        # px 是 RGBA, shape=(H, W, 4)
        rgb = np.array(px, dtype=np.uint8).reshape(self.resolution[1], self.resolution[0], 4)[:, :, :3]
        return rgb


class MuJoCoCamera:
    """MuJoCo 相机

    支持自由视角 (lookat + distance/azimuth/elevation) 和跟踪视角。
    """

    def __init__(self, model, data, resolution=(640, 480),
                 pos=(0.5, 0.0, 0.3), target=(0.0, 0.0, 0.0)):
        import mujoco
        self.model = model
        self.data = data
        self.resolution = resolution
        self.renderer = mujoco.Renderer(
            model, height=resolution[1], width=resolution[0]
        )
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.trackbodyid = -1
        self.set_pose(pos, target)

    def set_pose(self, pos, target):
        """设置自由相机位置和目标点"""
        import mujoco
        pos = np.asarray(pos, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.camera.lookat[:] = target
        # distance = |pos - target|
        distance = float(np.linalg.norm(pos - target))
        self.camera.distance = max(0.01, distance)
        # azimuth: 在 xy 平面上的角度
        dx, dy = pos[0] - target[0], pos[1] - target[1]
        self.camera.azimuth = float(np.degrees(np.arctan2(-dy, dx)))  # MuJoCo 约定
        # elevation: 相对水平面的仰角
        horizontal = float(np.linalg.norm([dx, dy]))
        self.camera.elevation = float(np.degrees(np.arctan2(pos[2] - target[2], horizontal)))

    def capture(self):
        """捕获一帧"""
        import mujoco
        self.renderer.update_scene(self.data, camera=self.camera)
        pixels = self.renderer.render()
        return pixels

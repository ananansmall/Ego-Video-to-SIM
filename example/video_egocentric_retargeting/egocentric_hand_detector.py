"""
第一人称视角手部检测器

基于 MediaPipe Hands 的手部关键点检测器，专门针对第一人称（egocentric）
视角视频进行优化。支持自拍模式（selfie）下的手别标签翻转。

核心功能：
  - detect(): 检测图像中所有匹配手别的手部，返回3D/2D关键点、手腕位姿
  - detect_single(): 检测单只手（返回第一个匹配结果）
  - draw_skeleton_on_image(): 在图像上绘制手部骨架

坐标系说明：
  - MediaPipe 坐标系：X右, Y下, Z朝相机（远离屏幕）
  - MANO 坐标系：通过 OPERATOR2MANO 矩阵变换得到
  - OPERATOR2MANO_RIGHT: 右手从操作者坐标系到 MANO 坐标系的旋转
  - OPERATOR2MANO_LEFT: 左手从操作者坐标系到 MANO 坐标系的旋转

Selfie 模式说明：
  - 第一人称视角视频通常是镜像的（如自拍），MediaPipe 的手别标签会反转
  - selfie=True 时，MediaPipe 的 "Right" 标签对应实际右手
  - selfie=False 时，MediaPipe 的 "Right" 标签对应实际左手
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "vector_retargeting"))

import mediapipe as mp
import numpy as np

# 操作者坐标系到 MANO 坐标系的旋转矩阵
# 右手：X→-Z, Y→-X, Z→Y
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
# 左手：X→-Z, Y→X, Z→-Y
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

# MediaPipe 指尖关键点索引（拇指4, 食指8, 中指12, 无名指16, 小指20）
MEDIAPIPE_TO_MANO_TIPS = [4, 8, 12, 16, 20]

# MediaPipe 21 个手部关键点名称
MEDIAPIPE_JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class EgocentricHandDetector:
    """
    第一人称视角手部检测器

    使用 MediaPipe Hands 检测手部关键点，并进行坐标系变换
    以适配后续的重定向和 IK 求解流程。

    Args:
        hand_type: 目标手别 ("Right" / "Left")
        min_detection_confidence: 检测置信度阈值（默认 0.7）
        min_tracking_confidence: 跟踪置信度阈值（默认 0.7）
        selfie: 是否为自拍模式（第一人称视角，默认 True）
            selfie=True: MediaPipe "Right" 标签 = 实际右手
            selfie=False: MediaPipe "Right" 标签 = 实际左手
    """

    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        selfie=True,
    ):
        # 初始化 MediaPipe Hands 模型
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,       # 视频模式，利用帧间连续性
            max_num_hands=2,               # 最多检测2只手
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        self.hand_type = hand_type
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = inverse_hand_dict[hand_type] if selfie else hand_type

    @staticmethod
    def _compute_wrist_pos_camera(keypoint_2d, keypoint_3d_array, img_w, img_h):
        """
        用2D关键点+深度估计计算相机坐标系下的手腕3D位置

        MediaPipe world_landmarks 是手心局部坐标（原点在手心），
        不能反映手在相机空间中的实际位置。本方法利用2D关键点的
        图像位置和手部度量尺寸来估计深度，再反投影到相机坐标系。

        相机坐标系：+X右, +Y下, +Z前(远离相机)

        深度估计原理：
          取手部多个关键点对的度量距离(world_landmarks)和图像距离(landmarks)，
          利用针孔相机模型 Z = f * metric_dist / image_dist 估计深度。
          假设 egocentric 相机水平FOV约70°。

        Args:
            keypoint_2d: MediaPipe NormalizedLandmarkList (2D归一化坐标)
            keypoint_3d_array: 21个关键点的3D坐标 (21, 3)，world_landmarks
            img_w: 图像宽度(像素)
            img_h: 图像高度(像素)

        Returns:
            ndarray: 手腕在相机坐标系下的3D位置 (3,)，单位：米
        """
        wrist_x = keypoint_2d.landmark[0].x
        wrist_y = keypoint_2d.landmark[0].y

        ref_pairs = [(0, 9), (0, 5), (5, 9), (0, 17), (5, 17)]
        depth_estimates = []

        fov_h = np.radians(70)
        f_norm = 1.0 / (2 * np.tan(fov_h / 2))
        aspect = img_w / max(img_h, 1)

        for i1, i2 in ref_pairs:
            metric_dist = np.linalg.norm(keypoint_3d_array[i2] - keypoint_3d_array[i1])
            dx = keypoint_2d.landmark[i2].x - keypoint_2d.landmark[i1].x
            dy = (keypoint_2d.landmark[i2].y - keypoint_2d.landmark[i1].y) * aspect
            img_dist = np.sqrt(dx * dx + dy * dy)

            if img_dist > 1e-6 and metric_dist > 1e-6:
                depth_est = f_norm * metric_dist / img_dist
                depth_estimates.append(depth_est)

        if depth_estimates:
            depth = float(np.median(depth_estimates))
        else:
            depth = 0.5

        depth = np.clip(depth, 0.1, 3.0)

        X = (wrist_x - 0.5) * depth / f_norm
        Y = (wrist_y - 0.5) * depth / f_norm
        Z = depth

        return np.array([X, Y, Z], dtype=np.float64)

    def detect(self, rgb):
        """
        检测图像中所有匹配手别的手部

        处理流程：
          1. MediaPipe Hands 检测所有手部
          2. 过滤出匹配目标手别的手部
          3. 提取 3D/2D 关键点
          4. 估计手腕坐标系（SVD 法线估计）
          5. 变换到 MANO 坐标系
          6. 计算相机坐标系下的手腕3D位置（2D+深度估计）

        Args:
            rgb: RGB 图像 (H, W, 3)，uint8 格式

        Returns:
            (hands_data, results):
                hands_data: 匹配手别的手部数据列表，每个元素包含：
                    - joint_pos: 21 个关节 3D 坐标 (21, 3)，MANO 坐标系，以手腕为原点
                    - wrist_pose: 手腕齐次变换矩阵 (4, 4)
                    - keypoint_2d: MediaPipe 2D 关键点对象
                    - wrist_pos_raw: 手腕原始 3D 位置 (3,)，手心局部坐标
                    - wrist_pos_camera: 手腕相机坐标系 3D 位置 (3,)，米
                    - wrist_rot_mano: 手腕旋转矩阵 (3, 3)，MANO 坐标系
                results: MediaPipe 原始检测结果
        """
        results = self.hand_detector.process(rgb)

        if not results.multi_hand_landmarks:
            return [], results

        img_h, img_w = rgb.shape[:2]
        hands_data = []
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label != self.detected_hand_type:
                continue

            keypoint_3d = results.multi_hand_world_landmarks[i]
            keypoint_2d = results.multi_hand_landmarks[i]

            keypoint_3d_array = self._parse_keypoint_3d(keypoint_3d)
            wrist_pos_raw = keypoint_3d_array[0].copy()

            wrist_pos_camera = self._compute_wrist_pos_camera(
                keypoint_2d, keypoint_3d_array, img_w, img_h
            )

            keypoint_3d_centered = keypoint_3d_array - keypoint_3d_array[0:1, :]
            mediapipe_wrist_rot = self._estimate_frame_from_hand_points(
                keypoint_3d_centered
            )
            joint_pos = keypoint_3d_centered @ mediapipe_wrist_rot @ self.operator2mano

            wrist_rot_mano = mediapipe_wrist_rot @ self.operator2mano
            wrist_pose = np.eye(4)
            wrist_pose[:3, :3] = wrist_rot_mano
            wrist_pose[:3, 3] = wrist_pos_raw

            hands_data.append(
                {
                    "joint_pos": joint_pos,
                    "wrist_pose": wrist_pose,
                    "keypoint_2d": keypoint_2d,
                    "wrist_pos_raw": wrist_pos_raw,
                    "wrist_pos_camera": wrist_pos_camera,
                    "wrist_rot_mano": wrist_rot_mano,
                }
            )

        return hands_data, results

    def detect_single(self, rgb):
        """
        检测单只手（返回第一个匹配目标手别的结果）

        Args:
            rgb: RGB 图像 (H, W, 3)，uint8 格式

        Returns:
            dict or None: 手部数据字典，格式同 detect() 返回的 hands_data 元素
        """
        hands_data, _ = self.detect(rgb)
        if len(hands_data) == 0:
            return None
        return hands_data[0]

    @staticmethod
    def _parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        """
        解析 MediaPipe 3D 关键点为 numpy 数组

        MediaPipe 的 world_landmarks 使用米为单位的 3D 坐标，
        原点在手腕处，X 朝右，Y 朝下，Z 朝相机方向。

        Args:
            keypoint_3d: MediaPipe NormalizedLandmarkList 对象

        Returns:
            ndarray: 21 个关键点的 3D 坐标 (21, 3)
        """
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        """
        解析 MediaPipe 2D 关键点为像素坐标数组

        Args:
            keypoint_2d: MediaPipe NormalizedLandmarkList 对象（归一化坐标）
            img_size: 图像尺寸 (H, W)

        Returns:
            ndarray: 21 个关键点的像素坐标 (21, 2)
        """
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def _estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        从手部关键点几何中估计手腕坐标系（旋转矩阵）

        使用手腕(0)、食指MCP(5)、中指MCP(9)三个关键点：
          1. X 轴：从中指MCP指向手腕的方向（去除法向分量）
          2. 法向量(Y)：对三个点进行 SVD 分解，取最小奇异值对应的向量
          3. Z 轴：X × Y 的叉积

        Args:
            keypoint_3d_array: 以手腕为原点的 21 个关键点 (21, 3)

        Returns:
            ndarray: 手腕坐标系旋转矩阵 (3, 3)，列向量分别为 X, Y, Z 轴
        """
        assert keypoint_3d_array.shape == (21, 3)
        # 选取手腕(0)、食指MCP(5)、中指MCP(9)三个关键点
        points = keypoint_3d_array[[0, 5, 9], :]

        # X 轴方向：从中指MCP指向手腕
        x_vector = points[0] - points[2]

        # SVD 估计平面法向量（Y 轴方向）
        points_centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points_centered)
        normal = v[2, :]  # 最小奇异值对应的向量 = 平面法向量

        # X 轴去除法向分量，确保 X ⊥ Y
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        # Z 轴 = X × Y
        z = np.cross(x, normal)

        # 确保法向量方向正确（食指MCP应在Z轴正方向）
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1

        # 构建旋转矩阵：列向量分别为 X, Y, Z 轴
        frame = np.stack([x, normal, z], axis=1)
        return frame

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
        """
        在图像上绘制 MediaPipe 手部骨架

        使用 MediaPipe 内置的绘制工具，在图像上标注 21 个关键点
        和连接线。白色背景风格下使用红色关键点和细连接线。

        Args:
            image: BGR 图像（会被原地修改）
            keypoint_2d: MediaPipe NormalizedLandmarkList 对象
            style: 绘制风格（目前仅支持 "white"）

        Returns:
            image: 绘制了骨架的图像（原地修改）
        """
        if style == "white":
            from mediapipe.python.solutions.drawing_utils import DrawingSpec
            from mediapipe.python.solutions.hands import HandLandmark
            from mediapipe.python.solutions import hands_connections

            # 关键点样式：红色圆点，半径4
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )
            # 连接线样式：细线
            connection_style = {}
            for pair in hands_connections.HAND_CONNECTIONS:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )
        return image

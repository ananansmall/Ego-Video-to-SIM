import os
import sys
import gc
import re
import subprocess
import tempfile
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vggt-omega'))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

CHECKPOINT_PATH = "/path/to/project_data/mano_ws/vggt-omega/checkpoints/VGGT-Omega-1B-512/vggt_omega_1b_512.pt"


def load_vggt_omega_model(checkpoint_path=CHECKPOINT_PATH):
    """
    加载 VGGT-Omega 模型权重。

    与原版 VGGT 的区别: 不使用 from_pretrained(), 手动 torch.load + load_state_dict。

    Args:
        checkpoint_path: .pt 权重文件路径

    Returns:
        VGGTOmega 模型实例, eval 模式, 在 CPU 上 (需外部 .to("cuda"))
    """
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def load_vggt_omega_frames(video_path, max_frames, image_resolution=512):
    """
    从视频/图片文件夹中提取帧并进行 VGGT-Omega 预处理。

    与原版 VGGT 的区别: resolution=512 (VGGT为518), patch_size=16 (VGGT为14)。
    VGGT-Omega 的 Aggregator 内部做 ImageNet 归一化, 无需外部处理。

    Args:
        video_path:       视频文件路径 或 图片文件夹路径
        max_frames:       最大帧数, 超出则均匀采样
        image_resolution: 输入分辨率, 必须为 16 的倍数 (默认 512)

    Returns:
        torch.Tensor (S, 3, H, W), 值域 [0, 1]
    """

    if os.path.isdir(video_path):
        images = os.listdir(video_path)
        images = [img for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]
        images = sorted(
            images,
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1,
        )
        total_frames = len(images)
        if total_frames == 0:
            raise ValueError(f"No image files found in directory: {video_path}")
        if total_frames > max_frames and max_frames > 0:
            indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
            images = [os.path.join(video_path, images[i]) for i in indices]
        else:
            images = [os.path.join(video_path, img) for img in images]
        return load_and_preprocess_images(images, image_resolution=image_resolution)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vsync", "0",
                    os.path.join(temp_dir, "frame_%04d.png"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return load_vggt_omega_frames(temp_dir, max_frames, image_resolution=image_resolution)


def unproject_depth_to_world_points(depth_map, extrinsic, intrinsic):
    """
    将深度图反投影为世界坐标系下的 3D 点云 (向量化, 一次处理所有帧)。

    数学公式 (与原版 VGGT 的 unproject_depth_map_to_point_map 数学等价):
      X_cam = [(u - cx) / fx * d,  (v - cy) / fy * d,  d]    (针孔模型反投影)
      X_world = R^T @ (X_cam - T)                                (SE3逆变换)

    实现差异:
      - 本函数: 向量化 numpy + einsum, 一次处理所有帧
      - 原版 VGGT: 逐帧循环 + closed_form_inverse_se3 构造完整4x4逆矩阵

    Args:
        depth_map: numpy (S, H, W, 1), VGGT-Omega DenseHead 输出的深度 (exp激活, 始终正值)
        extrinsic: numpy (S, 3, 4), camera-from-world 外参
        intrinsic: numpy (S, 3, 3), 相机内参 (FoV→fx/fy, 主点在中心)

    Returns:
        numpy (S, H, W, 3), 世界坐标系下的 3D 点
    """
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack([
        (x - cx) / fx * depth,
        (y - cy) / fy * depth,
        depth,
    ], axis=-1)

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def _predictions_to_pcd(predictions, conf_thres=50.0):
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    images = predictions["images"]

    if images.ndim == 4 and images.shape[1] == 3:
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images
    colors = (colors * 255).clip(0, 255).astype(np.uint8)

    vertices = points.reshape(-1, 3)
    vertex_colors = colors.reshape(-1, 3)
    conf_flat = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf_flat)
    if mask.sum() == 0:
        return trimesh.PointCloud(vertices=np.zeros((0, 3)), colors=np.zeros((0, 3)))

    if conf_thres > 0:
        conf_threshold = np.percentile(conf_flat[mask], conf_thres)
        mask &= conf_flat >= conf_threshold

    mask &= conf_flat > 1e-5

    vertices = vertices[mask]
    vertex_colors = vertex_colors[mask]

    return trimesh.PointCloud(vertices=vertices, colors=vertex_colors)


def vggt_omega_predict(images, model):
    '''
    使用 VGGT-Omega 模型对帧序列进行推理, 输出格式与 vggt_predict() 完全一致。

    推理流程:
      1. model(images) → VGGTOmega 内部:
         - Aggregator: 24层交替帧内/帧间注意力 (patch_size=16), 内部做 ImageNet 归一化
         - CameraHead: 单次前向 (非迭代), 输出 9D pose_enc [T(3), quat(4), FoV(2)]
           - T: linear (无激活), quat: linear (无激活), FoV: relu + 0.01 (保证正值)
         - DenseHead (output_dim via pixel_shuffle):
           depth = exp(depth_logits),  depth_conf = 1 + exp(conf_logits)
           conf 初始化: 权重=0, 偏置=log(0.05) → 初始 conf ≈ 1.05
         - ❌ 无 PointHead, 无 TrackHead
      2. encoding_to_camera(): 9D编码 → extrinsic (S,3,4) + intrinsic (S,3,3)
         - 与原版 pose_encoding_to_extri_intri() 数学完全相同
      3. unproject_depth_to_world_points(): depth + extrinsic + intrinsic → world_points
         - 向量化反投影, 数学等价于原版 unproject_depth_map_to_point_map
      4. world_points_conf = depth_conf (近似替代, 语义不同)

    与 vggt_predict() 的核心差异:
      - 无 PointHead: world_points 只能通过深度反投影间接得到
      - world_points_conf 用 depth_conf 近似替代 (深度置信度 vs 3D点位置置信度)
      - CameraHead 无迭代精修, 单次前向输出
      - DenseHead 使用 pixel_shuffle 上采样 (DPTHead 使用两阶段卷积)
      - 推理使用 torch.inference_mode() (VGGT 使用 no_grad + autocast)

    Args:
        images: torch.Tensor (S, 3, 512, 512), 来自 load_vggt_omega_frames(), 值域 [0,1]
        model:  VGGTOmega 模型实例 (eval 模式, 在 CUDA 上)

    Returns:
        dict (格式与 vggt_predict() 完全一致):
          - point_cloud_data:   trimesh.PointCloud, 置信度百分位50过滤后的合并点云
          - colors:             numpy (S, H, W, 3) uint8, RGB帧
          - depths:             numpy (S, H, W), 深度图 (exp激活, 始终正值)
          - extrinsics:         numpy (S, 4, 4), camera-from-world 外参
          - world_points:       numpy (S, H, W, 3), 深度反投影得到的世界坐标
          - world_points_conf:  numpy (S, H, W), depth_conf 近似替代 (≥1.05)
          - intrinsic:          numpy (3, 3), 所有帧内参取平均
    '''
    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    world_points = unproject_depth_to_world_points(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    predictions_np["world_points_from_depth"] = world_points

    point_cloud_data = _predictions_to_pcd(predictions_np, conf_thres=35.0)

    colors = (predictions_np['images'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions_np['depth'].squeeze(-1)
    extrinsics = np.pad(
        predictions_np['extrinsic'],
        ((0, 0), (0, 1), (0, 0)),
        mode='constant', constant_values=0,
    )
    extrinsics[:, 3, 3] = 1
    world_points_out = predictions_np['world_points_from_depth'].copy()
    world_points_conf = predictions_np['depth_conf'].copy()
    intrinsic_out = np.mean(predictions_np['intrinsic'], axis=0)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "world_points": world_points_out,
        "world_points_conf": world_points_conf,
        "intrinsic": intrinsic_out,
    }

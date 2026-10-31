"""
MASt3R-based 2D Correspondence Matching for Stage 4.

Implements the paper's Step 1 (Rendering and Matching) using MASt3R:
  - Render object mesh at current pose -> rendered RGB image
  - Run MASt3R inference between real RGB and rendered RGB
  - Extract dense 2D-2D correspondences via reciprocal nearest neighbors
  - Lift 2D correspondences to 3D via depth unprojection

This replaces the depth-consistency-based pixel matching in projection_alignment.py
with the actual MASt3R model as described in the paper (Section 3.4).

Dependencies:
  - mast3r package at /mnt/data_8THDD/lza/workspace/robot_world_ws/src/mast3r
  - MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric checkpoint
  - GPU with CUDA (MASt3R requires GPU for inference)
"""

import os
import sys
import numpy as np
import torch

MAST3R_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'mast3r')
if MAST3R_ROOT not in sys.path:
    sys.path.insert(0, MAST3R_ROOT)

CHECKPOINT_PATH = os.path.join(MAST3R_ROOT, 'checkpoints',
                                'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
HF_MODEL_NAME = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
HF_LOCAL_CACHE = os.path.join(MAST3R_ROOT, 'checkpoints',
                               'models--naver--MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')


def _find_local_hf_snapshot():
    """Find the latest snapshot directory in the local HuggingFace cache."""
    snapshots_dir = os.path.join(HF_LOCAL_CACHE, 'snapshots')
    if not os.path.isdir(snapshots_dir):
        return None
    for d in sorted(os.listdir(snapshots_dir), reverse=True):
        snapshot_path = os.path.join(snapshots_dir, d)
        if os.path.isfile(os.path.join(snapshot_path, 'config.json')):
            return snapshot_path
    return None


_cached_model = None
_cached_checkpoint = None
_cached_device = None


class MASt3RMatcher:
    """MASt3R-based matcher for establishing 2D correspondences between real and rendered images."""

    def __init__(self, checkpoint_path=None, device='cuda', image_size=512):
        global _cached_model, _cached_checkpoint, _cached_device

        if checkpoint_path is None:
            if os.path.isfile(CHECKPOINT_PATH):
                checkpoint_path = CHECKPOINT_PATH
            else:
                local_snapshot = _find_local_hf_snapshot()
                if local_snapshot:
                    checkpoint_path = local_snapshot
                else:
                    checkpoint_path = HF_MODEL_NAME
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = image_size

        if _cached_model is not None and _cached_checkpoint == checkpoint_path and _cached_device == device:
            self.model = _cached_model
        else:
            if _cached_model is not None:
                del _cached_model
                torch.cuda.empty_cache()
            self.model = None

    def _load_model(self):
        """Lazy-load MASt3R model."""
        global _cached_model, _cached_checkpoint, _cached_device

        if self.model is not None:
            return

        print(f"    [MASt3R] Loading model from {self.checkpoint_path}...")
        from mast3r.model import AsymmetricMASt3R

        is_local_file = os.path.isfile(self.checkpoint_path)
        if not is_local_file and '/' not in self.checkpoint_path:
            raise FileNotFoundError(
                f"MASt3R checkpoint not found: {self.checkpoint_path}\n"
                f"Please download it from https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            )

        self.model = AsymmetricMASt3R.from_pretrained(self.checkpoint_path).to(self.device)
        self.model.eval()

        _cached_model = self.model
        _cached_checkpoint = self.checkpoint_path
        _cached_device = self.device

        print(f"    [MASt3R] Model loaded on {self.device}")

    def match_images(self, rgb_real, rgb_rendered):
        """
        Run MASt3R between a real image and a rendered image to get 2D correspondences.

        Args:
            rgb_real: (H, W, 3) uint8 numpy array, real RGB image
            rgb_rendered: (H', W', 3) uint8 numpy array, rendered RGB image

        Returns:
            corres: dict with keys:
                'xy1': (N, 2) pixel coordinates in real image
                'xy2': (N, 2) pixel coordinates in rendered image
                'conf': (N,) confidence scores
        """
        self._load_model()

        from dust3r.utils.image import load_images
        from mast3r.fast_nn import extract_correspondences_nonsym
        from dust3r.inference import inference
        from dust3r.utils.device import to_numpy

        # Save images to temporary files for load_images
        import tempfile
        import cv2

        tmp_dir = tempfile.mkdtemp(prefix='mast3r_stage4_')
        real_path = os.path.join(tmp_dir, 'real.png')
        ren_path = os.path.join(tmp_dir, 'rendered.png')
        cv2.imwrite(real_path, cv2.cvtColor(rgb_real, cv2.COLOR_RGB2BGR))
        cv2.imwrite(ren_path, cv2.cvtColor(rgb_rendered, cv2.COLOR_RGB2BGR))

        # Load and preprocess images
        imgs = load_images([real_path, ren_path], size=self.image_size, verbose=False)
        img1, img2 = imgs[0], imgs[1]

        # Create pairs for inference
        pairs = [(img1, img2)]

        # Run MASt3R inference
        with torch.no_grad():
            result = inference(pairs, self.model, self.device, batch_size=1, verbose=False)

        pred1, pred2 = result['pred1'], result['pred2']

        desc1 = pred1['desc'].squeeze(0)       # (H1, W1, D)
        desc2 = pred2['desc'].squeeze(0)       # (H2, W2, D)
        desc_conf1 = pred1['desc_conf'].squeeze(0)  # (H1, W1)
        desc_conf2 = pred2['desc_conf'].squeeze(0)  # (H2, W2)

        xy1, xy2, conf = extract_correspondences_nonsym(
            desc1, desc2, desc_conf1, desc_conf2,
            subsample=8, device=self.device,
        )

        H_real, W_real = rgb_real.shape[:2]
        H_ren, W_ren = rgb_rendered.shape[:2]

        H1, W1 = desc1.shape[:2]
        H2, W2 = desc2.shape[:2]

        xy1 = to_numpy(xy1)
        xy2 = to_numpy(xy2)
        conf = to_numpy(conf)

        # Scale from MASt3R resolution to original resolution
        if len(xy1) > 0:
            xy1[:, 0] = xy1[:, 0] * W_real / W1
            xy1[:, 1] = xy1[:, 1] * H_real / H1
            xy2[:, 0] = xy2[:, 0] * W_ren / W2
            xy2[:, 1] = xy2[:, 1] * H_ren / H2

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            'xy1': xy1,
            'xy2': xy2,
            'conf': conf,
        }

    def establish_3d_correspondences(self, mesh, current_T, rgb_real, rgb_rendered,
                                      depth_real, depth_rendered,
                                      extrinsic, intrinsic,
                                      world_points_frame,
                                      conf_threshold=1.0):
        """
        Full pipeline: MASt3R matching -> 2D-3D lifting -> 3D-3D correspondences.

        This implements the paper's Step 1+2:
          Step 1: Rendering and Matching (MASt3R)
          Step 2: 3D Lifting and Aggregation

        Args:
            mesh: trimesh.Trimesh object (vertices in local coordinates)
            current_T: (4, 4) current transformation matrix (local -> world)
            rgb_real: (H, W, 3) uint8 real RGB image
            rgb_rendered: (H', W', 3) uint8 rendered RGB image
            depth_real: (H, W) float32 VGGT depth map
            depth_rendered: (H', W') float32 rendered depth map
            extrinsic: (4, 4) camera extrinsic
            intrinsic: (3, 3) camera intrinsic
            world_points_frame: (H, W, 3) VGGT 3D points for this frame
            conf_threshold: minimum MASt3R confidence for filtering

        Returns:
            mesh_pts: (M, 3) 3D points from mesh (in world coordinates)
            vggt_pts: (M, 3) 3D points from VGGT (in world coordinates)
            conf: (M,) confidence scores
        """
        from stage4.projection_alignment import unproject_depth_to_world

        mask_ren = depth_rendered > 0
        if mask_ren.sum() < 30:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

        rgb_for_matching = rgb_rendered.copy()
        black_bg = (rgb_rendered.sum(axis=2) == 0)
        if black_bg.sum() > 0:
            rgb_for_matching[black_bg] = [128, 128, 128]

        corres = self.match_images(rgb_real, rgb_for_matching)

        xy1 = corres['xy1']  # real image pixels
        xy2 = corres['xy2']  # rendered image pixels
        conf = corres['conf']

        if len(xy1) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

        # Adaptive confidence filtering
        min_correspondences = 50
        thresholds_to_try = [conf_threshold, 1.0, 0.75, 0.5]

        xy1_filtered, xy2_filtered, conf_filtered = None, None, None
        for thresh in thresholds_to_try:
            valid_conf = conf >= thresh
            xy1_f = xy1[valid_conf]
            xy2_f = xy2[valid_conf]
            conf_f = conf[valid_conf]
            if len(xy1_f) >= min_correspondences:
                xy1_filtered, xy2_filtered, conf_filtered = xy1_f, xy2_f, conf_f
                break

        if xy1_filtered is None:
            if len(xy1) > 0:
                top_k = min(min_correspondences, len(xy1))
                top_idx = np.argsort(conf)[-top_k:]
                xy1_filtered = xy1[top_idx]
                xy2_filtered = xy2[top_idx]
                conf_filtered = conf[top_idx]
            else:
                return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

        xy1, xy2, conf = xy1_filtered, xy2_filtered, conf_filtered

        # Step 2: 3D Lifting
        # For real image pixels -> look up VGGT 3D points
        H_real, W_real = depth_real.shape
        H_ren, W_ren = depth_rendered.shape

        # Round to integer pixel coordinates
        u1 = np.clip(np.round(xy1[:, 0]).astype(int), 0, W_real - 1)
        v1 = np.clip(np.round(xy1[:, 1]).astype(int), 0, H_real - 1)
        u2 = np.clip(np.round(xy2[:, 0]).astype(int), 0, W_ren - 1)
        v2 = np.clip(np.round(xy2[:, 1]).astype(int), 0, H_ren - 1)

        # Get VGGT 3D points at real image pixels
        vggt_pts = world_points_frame[v1, u1]

        # Get mesh 3D points by back-projecting rendered depth (论文方法 Step 2)
        # P_ren = π⁻¹(q_j, D_ren,v(q_j); K, T_v)
        # pyrender 返回的 depth 数值上等价于 OpenCV z-forward 深度，可直接反投影
        world_points_ren = unproject_depth_to_world(depth_rendered, extrinsic, intrinsic)
        mesh_pts = world_points_ren[v2, u2]
        valid_mesh = mask_ren[v2, u2] & (depth_rendered[v2, u2] > 1e-6)

        in_mask_ren = mask_ren[v2, u2]

        # Filter out invalid points
        valid = (~np.isnan(vggt_pts).any(axis=1) &
                 (depth_real[v1, u1] > 1e-6) &
                 valid_mesh &
                 (depth_rendered[v2, u2] > 1e-6) &
                 in_mask_ren)

        return mesh_pts[valid], vggt_pts[valid], conf[valid]

    def delete(self):
        """Release reference but keep cached model for reuse."""
        self.model = None

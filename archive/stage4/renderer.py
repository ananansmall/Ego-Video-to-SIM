import os
import numpy as np
import trimesh
import pyrender

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class MeshRenderer:
    """
    Offscreen mesh renderer using pyrender.
    Renders a 3D mesh from specified camera viewpoints to produce
    RGB images, depth maps, and segmentation masks.
    """

    def __init__(self, intrinsic_matrix, width, height):
        """
        Args:
            intrinsic_matrix: (3, 3) camera intrinsic matrix K
            width: image width
            height: image height
        """
        self.width = width
        self.height = height
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        self.camera = pyrender.IntrinsicsCamera(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            znear=0.01, zfar=100.0
        )

        self.light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=3.0
        )
        self.ambient = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=1.0
        )

        self._renderer = pyrender.OffscreenRenderer(self.width, self.height)

    def _extrinsic_to_camera_pose(self, extrinsic):
        """
        Convert VGGT extrinsic (world-to-camera) to pyrender camera pose (camera-to-world).

        VGGT extrinsic format: [R|t] where X_cam = R @ X_world + t
        This is w2c (world-to-camera) in OpenCV convention (x-right, y-down, z-forward).
        World coordinate system: z-up.

        pyrender/OpenGL convention: camera-to-world, x-right, y-up, z-backward.
        World coordinate system: y-up.

        We convert the world from z-up to y-up so pyrender can render correctly.
        After rendering, unprojected points must be converted back from y-up to z-up
        to match VGGT's coordinate system.

        Conversion: cam_pose = zup_to_yup @ inv(ext) @ opencv_to_opengl
        """
        c2w_opencv = np.linalg.inv(extrinsic)

        opencv_to_opengl = np.array([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]], dtype=np.float64)
        c2w_opengl = c2w_opencv @ opencv_to_opengl

        zup_to_yup = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
        cam_pose = zup_to_yup @ c2w_opengl
        return cam_pose

    def render_mesh(self, mesh, transform_matrix, extrinsic):
        """
        Render a mesh from a given viewpoint.

        Args:
            mesh: trimesh.Trimesh object (in local coordinates)
            transform_matrix: (4, 4) transformation matrix (local -> world)
            extrinsic: (4, 4) camera extrinsic matrix (world -> camera, VGGT format)

        Returns:
            color: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth map (in camera space, meters)
            mask: (H, W) bool binary mask of the rendered object
        """
        scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[0.3, 0.3, 0.3]
        )

        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(transform_matrix)

        zup_to_yup = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
        transformed_mesh.apply_transform(zup_to_yup)

        try:
            if hasattr(transformed_mesh.visual, 'material') and \
               hasattr(transformed_mesh.visual.material, 'baseColorFactor') and \
               not transformed_mesh.visual.material.baseColorFactor:
                transformed_mesh.visual.material.baseColorFactor = [0.8, 0.8, 0.8, 1.0]
        except (AttributeError, TypeError):
            pass

        pr_mesh = pyrender.Mesh.from_trimesh(transformed_mesh, smooth=False)
        scene.add(pr_mesh)

        camera_pose = self._extrinsic_to_camera_pose(extrinsic)
        scene.add(self.camera, pose=camera_pose)

        light_pose = camera_pose.copy()
        scene.add(self.light, pose=light_pose)

        ambient_pose = np.eye(4)
        ambient_pose[2, 3] = 1.0
        scene.add(self.ambient, pose=ambient_pose)

        color, depth = self._renderer.render(scene)
        mask = depth > 0

        return color, depth, mask

    def render_mesh_multiview(self, mesh, transform_matrix, extrinsics):
        """
        Render a mesh from multiple viewpoints.

        Args:
            mesh: trimesh.Trimesh object
            transform_matrix: (4, 4) transformation matrix
            extrinsics: list of (4, 4) camera extrinsic matrices

        Returns:
            colors: list of (H, W, 3) uint8 RGB images
            depths: list of (H, W) float32 depth maps
            masks: list of (H, W) bool binary masks
        """
        colors, depths, masks = [], [], []
        for ext in extrinsics:
            c, d, m = self.render_mesh(mesh, transform_matrix, ext)
            colors.append(c)
            depths.append(d)
            masks.append(m)
        return colors, depths, masks

    def __del__(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.delete()

    def delete(self):
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None


def compute_mask_iou(mask1, mask2):
    """
    Compute Intersection over Union between two binary masks.

    Args:
        mask1: (H, W) bool array
        mask2: (H, W) bool array

    Returns:
        iou: float, IoU value
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_mean_mask_iou(rendered_masks, real_masks):
    """
    Compute mean IoU across multiple views.

    Args:
        rendered_masks: list of (H, W) bool arrays
        real_masks: list of (H, W) bool arrays

    Returns:
        mean_iou: float
    """
    ious = []
    for ren_m, real_m in zip(rendered_masks, real_masks):
        ious.append(compute_mask_iou(ren_m, real_m))
    return np.mean(ious) if ious else 0.0

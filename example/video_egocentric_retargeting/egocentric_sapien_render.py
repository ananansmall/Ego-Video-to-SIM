import os
import sys
import tempfile
from pathlib import Path

_vulkan_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(_vulkan_icd) and "VK_ICD_FILENAMES" not in os.environ:
    os.environ["VK_ICD_FILENAMES"] = _vulkan_icd

import cv2
import numpy as np
import sapien
import tqdm
import tyro
from pytransform3d import rotations as pr_rot
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if not (PROJECT_ROOT / "assets" / "robots" / "hands").exists():
    _alt = Path("/home/an/robot_world_ws/src/dex-retargeting")
    if (_alt / "assets" / "robots" / "hands").exists():
        PROJECT_ROOT = _alt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "example" / "vector_retargeting"))

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector

MANO_FINGERS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def rotation_from_z_to_dir(direction):
    direction = np.array(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = direction / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, d)
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


class EgocentricSAPIENRenderer:
    def __init__(
        self,
        robot_name: RobotName = RobotName.r1,
        hand_type: HandType = HandType.right,
        retargeting_type: RetargetingType = RetargetingType.position,
        fps: int = 30,
        img_width: int = 1280,
        img_height: int = 720,
        mano_scale: float = 2.0,
        show_axes: bool = True,
        show_trail: bool = True,
        show_mano: bool = True,
    ):
        self.robot_name = robot_name
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fps = fps
        self.img_width = img_width
        self.img_height = img_height
        self.mano_scale = mano_scale
        self.show_axes = show_axes
        self.show_trail = show_trail
        self.show_mano = show_mano

        self.is_right = hand_type == HandType.right
        self.operator2mano = OPERATOR2MANO_RIGHT if self.is_right else OPERATOR2MANO_LEFT

        self._setup_sapien_scene()
        self._setup_robot()
        self._setup_retargeting()
        self._setup_materials()

        self.mano_trail = []
        self.robot_ee_trail = []
        self.mano_nodes = []
        self.axis_nodes = []
        self.trail_nodes = []

    def _setup_sapien_scene(self):
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)

        self.scene.set_environment_map(
            create_dome_envmap(sky_color=[0.3, 0.3, 0.3], ground_color=[0.15, 0.15, 0.15])
        )
        self.scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )
        self.scene.add_directional_light([0, 0, -1], [1.5, 1.4, 1.4], shadow=False)
        self.scene.set_ambient_light(np.array([0.3, 0.3, 0.3]))

        ground_mat = sapien.render.RenderMaterial()
        ground_mat.set_base_color(np.array([0.4, 0.4, 0.4, 1]))
        ground_mat.set_roughness(0.8)
        self.scene.add_ground(-1, render_material=ground_mat)

        self.camera = self.scene.add_camera(
            "egocentric_cam", self.img_width, self.img_height, 0.9, 0.01, 100
        )

        sapien.render.set_log_level("error")
        self.internal_scene = self.scene.render_system._internal_scene
        self.context = sapien.render.SapienRenderer()._internal_context

    def _setup_robot(self):
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        config_path = get_default_config_path(
            self.robot_name, self.retargeting_type, self.hand_type
        )
        override = {"add_dummy_free_joint": True}
        config = RetargetingConfig.load_from_file(config_path, override=override)

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(
            str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
        )
        temp_dir = tempfile.mkdtemp(prefix="egocentric_sapien-")
        temp_path = f"{temp_dir}/{urdf_path.name}"
        robot_urdf.write_xml_file(temp_path)

        self.robot = loader.load(temp_path)
        self.robot.set_pose(sapien.Pose([0, 0, 0]))

        self.retargeting_config = config
        self.robot_config_path = config_path

        self.robot_active_joints = self.robot.get_active_joints()
        self.robot_links = self.robot.get_links()

        self.dummy_joint_indices = []
        self.finger_joint_indices = []
        for i, j in enumerate(self.robot_active_joints):
            if "dummy" in j.name:
                self.dummy_joint_indices.append(i)
            else:
                self.finger_joint_indices.append(i)

    def _setup_retargeting(self):
        self.retargeting: SeqRetargeting = self.retargeting_config.build()

        opt = self.retargeting.optimizer
        num_fixed = len(opt.idx_pin2fixed)
        self.fixed_qpos = np.zeros(num_fixed, dtype=np.float32)

        fixed_names = opt.fixed_joint_names
        for i, name in enumerate(fixed_names):
            if "right_arm" in name:
                joint_idx = int(name.replace("right_arm_joint", "").replace("_joint", "")) - 1
                defaults = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
                if joint_idx < len(defaults):
                    self.fixed_qpos[i] = defaults[joint_idx]
            elif "left_arm" in name:
                joint_idx = int(name.replace("left_arm_joint", "").replace("_joint", "")) - 1
                defaults = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
                if joint_idx < len(defaults):
                    self.fixed_qpos[i] = defaults[joint_idx]

        sapien_joint_names = [j.name for j in self.robot_active_joints]
        self.retarget2sapien = np.array(
            [self.retargeting.joint_names.index(n) for n in sapien_joint_names]
        ).astype(int)

        self.detector = SingleHandDetector(
            hand_type="Right" if self.is_right else "Left",
            selfie=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def _setup_materials(self):
        self.mat_mano_hand = self.context.create_material(
            np.array([0.96, 0.75, 0.69, 1.0]), np.array([0.1, 0.1, 0.1, 1.0]), 0.5, 0.0, 0
        )
        self.mat_mano_finger = [
            self.context.create_material(np.array([1.0, 0.2, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([1.0, 0.6, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.2, 1.0, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.2, 0.4, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.8, 0.2, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        ]
        self.mat_axis = [
            self.context.create_material(np.array([1, 0, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0, 1, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0, 0, 1, 1]), np.zeros(4), 0.5, 0.0, 0),
        ]
        self.mat_trail = self.context.create_material(
            np.array([1.0, 0.8, 0.0, 0.8]), np.zeros(4), 0.5, 0.0, 0
        )
        self.mat_robot_trail = self.context.create_material(
            np.array([0.0, 0.5, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0
        )

    def _clear_nodes(self, node_list):
        for _ in range(len(node_list)):
            node = node_list.pop()
            self.internal_scene.remove_node(node)

    def _add_sphere(self, pos, radius, mat, node_list):
        sphere_mesh = self.context.create_uvsphere_mesh(12, 6)
        model = self.context.create_model([sphere_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(pos.astype(np.float32))
        node.set_scale(np.array([radius, radius, radius], dtype=np.float32))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _add_capsule(self, p1, p2, radius, mat, node_list):
        mid = ((p1 + p2) / 2).astype(np.float32)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        quat = rotation_from_z_to_dir(direction).astype(np.float32)
        capsule_mesh = self.context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = self.context.create_model([capsule_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(mid)
        node.set_rotation(quat)
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _render_mano_skeleton(self, joint_pos, wrist_rot, origin_pos):
        self._clear_nodes(self.mano_nodes)

        if joint_pos is None or wrist_rot is None:
            return

        scale = self.mano_scale
        offset = wrist_rot[:, 1] * 0.2 if wrist_rot is not None else np.array([0.0, 0.2, 0.0])
        mano_center = origin_pos + offset
        joints_world = []
        for i in range(21):
            j_world = mano_center + (wrist_rot @ (joint_pos[i] * scale)).astype(np.float64)
            joints_world.append(j_world)

        for fi, finger in enumerate(MANO_FINGERS):
            for j in range(len(finger) - 1):
                self._add_capsule(
                    joints_world[finger[j]], joints_world[finger[j + 1]],
                    0.008, self.mat_mano_finger[fi], self.mano_nodes
                )

        for i in range(21):
            mat = self.mat_mano_hand if i == 0 else self.mat_mano_finger[min(i // 4, 4)]
            self._add_sphere(joints_world[i], 0.012, mat, self.mano_nodes)

    def _render_axes(self, wrist_pos, wrist_rot, robot_ee_pos=None, robot_ee_rot=None):
        self._clear_nodes(self.axis_nodes)

        if not self.show_axes:
            return

        if wrist_pos is not None and wrist_rot is not None:
            axis_len = 0.25
            for ai in range(3):
                axis_end = wrist_pos + wrist_rot[:, ai] * axis_len
                self._add_capsule(wrist_pos, axis_end, 0.012, self.mat_axis[ai], self.axis_nodes)

        if robot_ee_pos is not None and robot_ee_rot is not None:
            axis_len = 0.12
            for ai in range(3):
                axis_end = robot_ee_pos + robot_ee_rot[:, ai] * axis_len
                self._add_capsule(robot_ee_pos, axis_end, 0.006, self.mat_axis[ai], self.axis_nodes)

        for link in self.robot_links:
            link_pose = link.get_entity_pose()
            link_pos = np.array(link_pose.p).astype(np.float64)
            link_quat = np.array(link_pose.q)
            link_R = pr_rot.matrix_from_quaternion(link_quat)
            for ai in range(3):
                axis_end = link_pos + link_R[:, ai] * 0.04
                self._add_capsule(link_pos, axis_end, 0.002, self.mat_axis[ai], self.axis_nodes)

    def _render_trails(self, mano_pos, robot_ee_pos):
        self._clear_nodes(self.trail_nodes)

        if not self.show_trail:
            return

        if mano_pos is not None:
            self.mano_trail.append(mano_pos.copy())
            if len(self.mano_trail) > 200:
                self.mano_trail.pop(0)

        if robot_ee_pos is not None:
            self.robot_ee_trail.append(robot_ee_pos.copy())
            if len(self.robot_ee_trail) > 200:
                self.robot_ee_trail.pop(0)

        if len(self.mano_trail) > 2:
            step = max(1, len(self.mano_trail) // 80)
            for i in range(step, len(self.mano_trail), step):
                alpha = i / len(self.mano_trail)
                trail_mat = self.context.create_material(
                    np.array([1.0 * alpha, 0.8 * alpha, 0.0, 0.3 + 0.7 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0
                )
                self._add_capsule(
                    self.mano_trail[i - step], self.mano_trail[i],
                    0.005, trail_mat, self.trail_nodes
                )
            self._add_sphere(self.mano_trail[-1], 0.018, self.mat_trail, self.trail_nodes)

        if len(self.robot_ee_trail) > 2:
            step = max(1, len(self.robot_ee_trail) // 80)
            for i in range(step, len(self.robot_ee_trail), step):
                alpha = i / len(self.robot_ee_trail)
                robot_trail_mat = self.context.create_material(
                    np.array([0.0, 0.5 * alpha, 1.0 * alpha, 0.3 + 0.7 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0
                )
                self._add_capsule(
                    self.robot_ee_trail[i - step], self.robot_ee_trail[i],
                    0.005, robot_trail_mat, self.trail_nodes
                )
            self._add_sphere(self.robot_ee_trail[-1], 0.018, self.mat_robot_trail, self.trail_nodes)

    def _set_robot_pose_from_wrist(self, wrist_pos, wrist_rot_mano):
        base_link = self.robot.get_links()[0]
        base_quat_wxyz = pr_rot.quaternion_from_matrix(wrist_rot_mano)
        base_quat_wxyz = np.array([base_quat_wxyz[3], base_quat_wxyz[0], base_quat_wxyz[1], base_quat_wxyz[2]])
        self.robot.set_pose(sapien.Pose(wrist_pos.astype(np.float32), base_quat_wxyz.astype(np.float32)))

    def process_video(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            import os as _os
            path_info = [
                f"  Path: {video_path}",
                f"  Exists: {_os.path.exists(video_path)}",
            ]
            if _os.path.exists(video_path):
                path_info.append(f"  Size: {_os.path.getsize(video_path)} bytes")
            raise ValueError("Cannot open video:\n" + "\n".join(path_info))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"Video: {total_frames} frames, {video_fps:.1f} fps, {orig_w}x{orig_h}")
        print(f"Robot: {self.robot_name}, Hand: {self.hand_type}, Scale: {self.mano_scale}")
        print(f"Output: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.img_width, self.img_height))

        fov_h = np.radians(70)
        f_norm = 1.0 / (2 * np.tan(fov_h / 2))
        aspect = orig_w / max(orig_h, 1)

        R_cam2base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

        warmup_frames = 5
        prev_qpos = None

        for frame_idx in tqdm.tqdm(range(total_frames), desc="Rendering"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[..., ::-1]

            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = self.detector.detect(rgb)

            if frame_idx < 10:
                print(f"  Frame {frame_idx}: num_box={num_box}, joint_pos={'OK' if joint_pos is not None else 'None'}, keypoint_2d={'OK' if keypoint_2d is not None else 'None'}, wrist_rot={'OK' if mediapipe_wrist_rot is not None else 'None'}")

            mano_origin = None
            wrist_rot_world = None
            robot_ee_pos = None
            robot_ee_rot = None

            if num_box > 0 and joint_pos is not None:
                indices = self.retargeting.optimizer.target_link_human_indices
                retargeting_type = self.retargeting.optimizer.retargeting_type

                if retargeting_type == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                qpos = self.retargeting.retarget(ref_value, self.fixed_qpos)
                robot_qpos = qpos[self.retarget2sapien]
                self.robot.set_qpos(robot_qpos)
                prev_qpos = robot_qpos

                if keypoint_2d is not None:
                    wrist_x = keypoint_2d.landmark[0].x
                    wrist_y = keypoint_2d.landmark[0].y

                    ref_pairs = [(0, 9), (0, 5), (5, 9), (0, 17), (5, 17)]
                    depth_estimates = []
                    for i1, i2 in ref_pairs:
                        metric_dist = np.linalg.norm(joint_pos[i2] - joint_pos[i1])
                        dx = keypoint_2d.landmark[i2].x - keypoint_2d.landmark[i1].x
                        dy = (keypoint_2d.landmark[i2].y - keypoint_2d.landmark[i1].y) * aspect
                        img_dist = np.sqrt(dx * dx + dy * dy)
                        if img_dist > 1e-6 and metric_dist > 1e-6:
                            depth_est = f_norm * metric_dist / img_dist
                            depth_estimates.append(depth_est)

                    depth = float(np.median(depth_estimates)) if depth_estimates else 0.5
                    depth = np.clip(depth, 0.1, 3.0)

                    X = (wrist_x - 0.5) * depth / f_norm
                    Y = (wrist_y - 0.5) * depth / f_norm
                    Z = depth
                    wrist_pos_camera = np.array([X, Y, Z])

                    wrist_pos_base = R_cam2base @ wrist_pos_camera
                    wrist_rot_base = R_cam2base @ mediapipe_wrist_rot

                    wrist_rot_mano = mediapipe_wrist_rot @ self.operator2mano
                    wrist_rot_world_mat = (R_cam2base @ wrist_rot_mano @ self.operator2mano.T).astype(np.float64)

                    mano_origin = wrist_pos_base.astype(np.float64)
                    wrist_rot_world = wrist_rot_world_mat

                    self._set_robot_pose_from_wrist(mano_origin, wrist_rot_world)

                    ee_link = self.robot_links[-1]
                    ee_pose = ee_link.get_entity_pose()
                    robot_ee_pos = np.array(ee_pose.p).astype(np.float64)
                    ee_quat = np.array(ee_pose.q)
                    robot_ee_rot = pr_rot.matrix_from_quaternion(ee_quat)
            else:
                if prev_qpos is not None:
                    self.robot.set_qpos(prev_qpos)

            self.scene.step()

            if frame_idx >= warmup_frames:
                if self.show_mano and mano_origin is not None and joint_pos is not None:
                    self._render_mano_skeleton(joint_pos, wrist_rot_world, mano_origin)
                else:
                    self._clear_nodes(self.mano_nodes)

                self._render_axes(mano_origin, wrist_rot_world, robot_ee_pos, robot_ee_rot)
                self._render_trails(mano_origin, robot_ee_pos)

            self.scene.update_render()

            if frame_idx >= warmup_frames:
                if not hasattr(self, '_ego_smooth_pos'):
                    self._ego_smooth_pos = mano_origin.copy() if mano_origin is not None else np.array([0.3, 0.0, 0.0])
                alpha_smooth = 0.08
                if mano_origin is not None:
                    self._ego_smooth_pos = (1 - alpha_smooth) * self._ego_smooth_pos + alpha_smooth * mano_origin
                smooth_target = self._ego_smooth_pos.copy()

                cam_pos = smooth_target + np.array([-0.30, 0.0, 0.45])
                cam_look_at = smooth_target + np.array([0.10, 0.0, -0.05])
                cam_up_hint = np.array([0, 0, 1])

                forward = cam_look_at - cam_pos
                forward_len = np.linalg.norm(forward)
                if forward_len > 1e-6:
                    forward = forward / forward_len
                else:
                    forward = np.array([1, 0, 0])

                right = np.cross(forward, cam_up_hint)
                right_len = np.linalg.norm(right)
                if right_len > 1e-6:
                    right = right / right_len
                else:
                    right = np.array([0, 1, 0])
                up = np.cross(right, forward)

                cam_R = np.eye(4)
                cam_R[:3, 0] = right
                cam_R[:3, 1] = -up
                cam_R[:3, 2] = forward
                cam_R[:3, 3] = cam_pos

                from scipy.spatial.transform import Rotation as R_scipy
                cam_quat_xyzw = R_scipy.from_matrix(cam_R[:3, :3]).as_quat()
                cam_quat_wxyz = np.array([cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]])

                self.camera.set_local_pose(
                    sapien.Pose(cam_pos.astype(np.float32), cam_quat_wxyz.astype(np.float32))
                )

                self.camera.take_picture()
                rgb_render = self.camera.get_picture("Color")[..., :3]
                rgb_render = (np.clip(rgb_render, 0, 1) * 255).astype(np.uint8)
                bgr_render = np.ascontiguousarray(rgb_render[..., ::-1])

                if num_box > 0 and keypoint_2d is not None:
                    SingleHandDetector.draw_skeleton_on_image(bgr_render, keypoint_2d, style="white")

                status = "DETECTED" if num_box > 0 else "LOST"
                status_color = (0, 255, 0) if num_box > 0 else (0, 0, 255)
                cv2.putText(bgr_render, f"Frame {frame_idx} | {status}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                if mano_origin is not None:
                    cv2.putText(bgr_render, f"Wrist: ({mano_origin[0]:.2f}, {mano_origin[1]:.2f}, {mano_origin[2]:.2f})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                writer.write(bgr_render)
            else:
                self.camera.take_picture()

        writer.release()
        cap.release()
        self.retargeting.verbose()
        print(f"\nVideo saved: {output_path}")


def main(
    video_path: str,
    output_path: str = "",
    robot_name: RobotName = RobotName.r1,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.position,
    fps: int = 30,
    mano_scale: float = 2.0,
    show_axes: bool = True,
    show_trail: bool = True,
    show_mano: bool = True,
):
    if not output_path:
        video_stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{video_stem}_egocentric_sapien.mp4")

    renderer = EgocentricSAPIENRenderer(
        robot_name=robot_name,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
        fps=fps,
        mano_scale=mano_scale,
        show_axes=show_axes,
        show_trail=show_trail,
        show_mano=show_mano,
    )
    renderer.process_video(video_path, output_path)


if __name__ == "__main__":
    tyro.cli(main)

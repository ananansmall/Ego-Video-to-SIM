import pickle
import sys
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import tqdm
import tyro

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from egocentric_hand_detector import EgocentricHandDetector
from arm_motion_analyzer import ArmMotionAnalyzer
from robot_hand_retargeter import RobotHandRetargeter
from contact_preservation import ContactPreservationFilter, TrajectorySmoother


def process_video_to_trajectory(
    video_path: str,
    output_path: str,
    robot_name: RobotName = RobotName.allegro,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.position,
    fps: int = 30,
    start_frame: int = 0,
    num_frames: int = -1,
    arm_offset: List[float] = None,
    visualize_detection: bool = False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(video_fps / fps))

    if num_frames < 0:
        num_frames = total_frames

    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}, FPS: {video_fps:.1f}")
    print(f"  Processing: frames {start_frame}-{start_frame + num_frames}, skip={frame_skip}")

    detector = EgocentricHandDetector(
        hand_type=hand_type.value if isinstance(hand_type, HandType) else hand_type,
        selfie=True,
    )

    arm_analyzer = ArmMotionAnalyzer(
        hand_type=hand_type,
        arm_offset=np.array(arm_offset) if arm_offset else np.zeros(3),
    )

    retargeter = RobotHandRetargeter(
        robot_name=robot_name,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
    )

    contact_filter = ContactPreservationFilter()
    trajectory_smoother = TrajectorySmoother(window_size=3, alpha=0.3)

    data = []
    wrist_trajectory = []
    meta_data = retargeter.get_meta_data()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0
    detected = 0

    with tqdm.tqdm(total=num_frames, desc="Processing") as pbar:
        while cap.isOpened() and processed < num_frames:
            frame_idx = start_frame + processed * frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            rgb = frame[..., ::-1]
            hand_data = detector.detect_single(rgb)

            if hand_data is None:
                data.append(None)
                wrist_trajectory.append(None)
                processed += 1
                pbar.update(1)
                continue

            detected += 1
            joint_pos = hand_data["joint_pos"]
            wrist_pos = hand_data["wrist_pos_raw"]
            wrist_rot = hand_data["wrist_rot_mano"]

            from scipy.spatial.transform import Rotation as R
            wrist_quat = R.from_matrix(wrist_rot).as_quat()

            qpos = retargeter.retarget(joint_pos, wrist_pos, wrist_quat)

            finger_dist = ArmMotionAnalyzer.compute_finger_aperture(joint_pos)
            filtered_qpos, is_grasping = contact_filter.update(finger_dist, qpos)
            smoothed_qpos = trajectory_smoother.smooth(filtered_qpos)

            arm_info = arm_analyzer.compute_wrist_6dof(hand_data)

            data.append(smoothed_qpos)
            wrist_trajectory.append(
                {
                    "wrist_pos": arm_info["wrist_pos"],
                    "wrist_quat": arm_info["wrist_quat"],
                    "ee_pos": arm_info["ee_pos"],
                    "ee_quat": arm_info["ee_quat"],
                    "finger_distance": finger_dist,
                    "is_grasping": is_grasping,
                    "frame_idx": frame_idx,
                }
            )

            if visualize_detection:
                bgr_with_skeleton = frame.copy()
                detector.draw_skeleton_on_image(
                    bgr_with_skeleton, hand_data["keypoint_2d"]
                )
                status = "GRASP" if is_grasping else "OPEN"
                cv2.putText(
                    bgr_with_skeleton,
                    f"{status} d={finger_dist:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if is_grasping else (0, 0, 255),
                    2,
                )
                cv2.imshow("egocentric_retargeting", bgr_with_skeleton)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            pbar.update(1)

    cap.release()
    if visualize_detection:
        cv2.destroyAllWindows()

    valid = sum(1 for d in data if d is not None)
    print(f"\nDetection rate: {valid}/{processed} ({valid/max(processed,1)*100:.1f}%)")
    print(f"Retargeting type: {retargeting_type.value}")
    print(f"Robot: {robot_name.value}, DOF: {meta_data['dof']}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "data": data,
                "wrist_trajectory": wrist_trajectory,
                "meta_data": meta_data,
                "video_info": {
                    "video_path": str(video_path),
                    "start_frame": start_frame,
                    "num_frames": processed,
                    "fps": fps,
                    "frame_skip": frame_skip,
                },
            },
            f,
        )

    print(f"Saved to: {output_path}")
    retargeter.retargeting.verbose()
    return data, wrist_trajectory, meta_data


def main(
    video_path: str,
    output_path: str,
    robot_name: RobotName = RobotName.allegro,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.position,
    fps: int = 30,
    start_frame: int = 0,
    num_frames: int = -1,
    arm_offset_x: float = 0.0,
    arm_offset_y: float = 0.0,
    arm_offset_z: float = 0.0,
    visualize: bool = False,
):
    process_video_to_trajectory(
        video_path=video_path,
        output_path=output_path,
        robot_name=robot_name,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
        fps=fps,
        start_frame=start_frame,
        num_frames=num_frames,
        arm_offset=[arm_offset_x, arm_offset_y, arm_offset_z],
        visualize_detection=visualize,
    )


if __name__ == "__main__":
    tyro.cli(main)

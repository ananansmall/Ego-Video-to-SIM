import pickle
from pathlib import Path

import cv2
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector


def retarget_video(
    retargeting: SeqRetargeting, video_path: str, output_path: str, config_path: str
):
    """
    从视频中逐帧检测手部姿态并重定向到机器人关节角度
    
    Args:
        retargeting: 重定向器对象，包含优化器和配置
        video_path: 输入视频文件路径
        output_path: 输出pickle文件路径
        config_path: 配置文件路径
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 存储每帧的机器人关节角度
    data = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # 初始化手部检测器（右手，非自拍模式）
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        # 获取视频总帧数用于进度条
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # BGR转RGB（MediaPipe需要RGB格式）
                rgb = frame[..., ::-1]
                # 检测手部关键点：返回检测数量、3D关节位置、2D关键点、手腕旋转矩阵
                num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(
                    rgb
                )
                # 如果未检测到手，跳过该帧
                if num_box == 0:
                    continue

                # 根据重定向类型计算参考值
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    # 位置模式：直接使用关键点的绝对位置
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    # 向量模式（包括DexPilot）：使用关键点之间的相对向量
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                # 调用重定向优化器计算机器人关节角度
                qpos = retargeting.retarget(ref_value)
                data.append(qpos)
                pbar.update(1)

        # 构建元数据：包含配置路径、自由度数量、关节名称
        meta_data = dict(
            config_path=config_path,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
        )

        # 创建输出目录并保存为pickle文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        # 打印重定向统计信息（耗时、误差等）
        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """
    从视频检测人手姿态并重定向为机器人关节轨迹
    
    Args:
        robot_name: 机器人标识符，必须是支持的机器人之一（如allegro、panda等）
        video_path: 输入视频的.mp4文件路径
        output_path: 输出数据的.pickle文件路径
        retargeting_type: 重定向类型（vector/position/dexpilot），对应不同算法
        hand_type: 指定追踪的手（左手或右手）
            注意：重定向仅适用于同类型手，左手只能重定向到左手机器人，右手同理
    """
    # 根据机器人名称、重定向类型和手类型获取默认配置文件路径
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    # 设置URDF模型目录为项目assets目录
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    # 从配置文件加载并构建重定向器
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    # 执行视频重定向处理
    retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
try:
    np.unicode = np.unicode_
except AttributeError:
    np.unicode = np.str_


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int):
    """
    可视化人手和机器人手的抓取轨迹
    
    该函数从DexYCB数据集中加载数据，创建相应的查看器（仅人手或人手+机器人），
    并渲染手部与物体的交互轨迹。
    
    Args:
        robots: 机器人名称元组，如果为None则仅可视化人手
        data_root: DexYCB数据集的根目录路径
        fps: 渲染帧率（每秒帧数）
    """
    # 加载DexYCB视频数据集，指定使用右手数据
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    
    # 根据是否指定机器人，创建不同的查看器
    if robots is None:
        # 仅创建人手查看器
        viewer = HandDatasetSAPIENViewer(headless=False)
    else:
        # 创建包含机器人手的查看器
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=False
        )

    # 数据ID，可以修改此值来可视化不同的轨迹
    data_id = 4

    # 从数据集中采样指定ID的数据
    sampled_data = dataset[data_id]
    
    # 打印数据信息（排除姿态数据以保持输出简洁）
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    
    # 加载物体和手部数据到查看器
    viewer.load_object_hand(sampled_data)
    
    # 渲染DexYCB数据轨迹
    viewer.render_dexycb_data(sampled_data, fps)


def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10):
    """
    主函数：渲染DexYCB数据集中的人手和机器人抓取轨迹
    
    该函数负责初始化环境配置，验证数据路径，并调用可视化函数。
    人手轨迹直接来自数据集，而机器人轨迹通过位置重定向算法生成。
    
    Args:
        dexycb_dir: DexYCB数据集的根目录路径
        robots: 要渲染的机器人名称列表，如果为None则仅渲染人手轨迹
        fps: 渲染手部-物体轨迹的频率（帧/秒）
    """
    # 将输入路径转换为绝对路径
    data_root = Path(dexycb_dir).absolute()
    
    # 设置机器人URDF文件的默认目录（指向hands文件夹）
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # 验证数据路径是否存在
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    # 调用可视化函数
    viz_hand_object(robots, data_root, fps)


if __name__ == "__main__":
    tyro.cli(main)

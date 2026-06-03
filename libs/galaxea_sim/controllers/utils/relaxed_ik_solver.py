"""
RelaxedIK 高层求解器封装

功能：提供双臂独立的 IK 求解接口，自动处理四元数格式转换和容差配置。

架构层次：
  用户代码 → RelaxedIKSolver (本文件) → RelaxedIKRust (python_wrapper.py) 
         → librelaxed_ik_lib.so (Rust 核心)

关键特性：
- 左右臂独立求解器实例
- 自动 wxyz → xyzw 四元数格式转换
- 可配置的求解容差（位置+姿态）
"""

from galaxea_sim.controllers.utils.python_wrapper import RelaxedIKRust
import pathlib

class RelaxedIKSolver:
    """
    RelaxedIK 双臂求解器管理类
    
    封装左右臂两个独立的 RelaxedIK 求解器实例，提供统一的调用接口。
    负责四元数格式转换（wxyz → xyzw）和容差参数管理。
    """
    
    def __init__(self, left_setting_file_path, right_setting_file_path, tolerances=None):
        """
        初始化双臂 RelaxedIK 求解器
        
        Args:
            left_setting_file_path: 左臂配置文件路径（相对于 assets 目录）
                                   例如: "r1_lite/configs/settings_left.yaml"
            right_setting_file_path: 右臂配置文件路径（相对于 assets 目录）
                                    例如: "r1_lite/configs/settings_right.yaml"
            tolerances: IK 求解容差列表 [tx, ty, tz, rx, ry, rz]
                       - tx,ty,tz: 位置容差（米），默认 0.01m = 1cm
                       - rx,ry,rz: 旋转容差（弧度），默认 0.01rad ≈ 0.57°
                       如果为 None，使用默认值 [0.01] * 6
        """
        # 初始化左右臂 Rust 求解器实例
        # 注意：路径需要转换为绝对路径，因为 Rust 库会读取相对路径的配置文件
        self.relaxed_ik_left = RelaxedIKRust(str(pathlib.Path(__file__).parent.parent / left_setting_file_path))
        self.relaxed_ik_right = RelaxedIKRust(str(pathlib.Path(__file__).parent.parent / right_setting_file_path))
        
        # 设置默认容差（如果未提供）
        self.tolerances = tolerances if tolerances else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    def _convert_wxyz_to_xyzw(self, quat_wxyz):
        """
        四元数格式转换：wxyz → xyzw
        
        Python 生态（pytransform3d、SAPIEN）使用 [w, x, y, z] 格式，
        而 Rust 核心库期望 [x, y, z, w] 格式。
        
        Args:
            quat_wxyz: wxyz 格式四元数 [w, x, y, z]
        
        Returns:
            xyzw 格式四元数 [x, y, z, w]
        
        Example:
            >>> solver._convert_wxyz_to_xyzw([1, 0, 0, 0])  # 单位四元数
            [0, 0, 0, 1]
        """
        return [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    def solve_position_left(self, target_pos, target_quat_wxyz):
        """
        求解左臂逆运动学
        
        给定末端执行器的目标位姿，计算对应的 7 个关节角度。
        
        Args:
            target_pos: 目标位置 [x, y, z]，单位：米
                       坐标系：base_link 帧（非世界坐标系！）
            target_quat_wxyz: 目标姿态四元数 [w, x, y, z]
                             坐标系：base_link 帧
        
        Returns:
            list: 7 个关节角度（弧度），对应左臂的 7 个关节
                  顺序与 URDF 中定义的关节顺序一致
        
        Raises:
            Exception: 如果目标不可达或优化失败
        
        Note:
            - 内部自动将 wxyz 转换为 xyzw 传给 Rust 层
            - 使用当前容差配置（self.tolerances）
        """
        # 转换四元数格式：wxyz → xyzw
        target_quat_xyzw = self._convert_wxyz_to_xyzw(target_quat_wxyz)
        return self.relaxed_ik_left.solve_position(target_pos, target_quat_xyzw, self.tolerances)

    def solve_position_right(self, target_pos, target_quat_wxyz):
        """
        求解右臂逆运动学
        
        给定末端执行器的目标位姿，计算对应的 6 个关节角度（R1 Lite）或 7 个（R1 Pro）。
        
        Args:
            target_pos: 目标位置 [x, y, z]，单位：米
                       坐标系：right_arm_base_link 帧（非世界坐标系！）
            target_quat_wxyz: 目标姿态四元数 [w, x, y, z]
                             坐标系：right_arm_base_link 帧
        
        Returns:
            list: 6 或 7 个关节角度（弧度），对应右臂关节
                  R1 Lite: 6 个关节
                  R1 Pro: 7 个关节
        
        Raises:
            Exception: 如果目标超出工作空间或优化失败
        
        Example:
            >>> joints = solver.solve_position_right(
            ...     target_pos=[0.3, 0.0, -0.2],      # base_link 帧坐标
            ...     target_quat_wxyz=[0.707, 0, 0.707, 0]  # 绕 Y 轴旋转 90°
            ... )
            >>> print(f"右臂关节角: {joints}")
        
        Note:
            - 坐标系变换公式: p_base = R⁻¹ @ (p_world - t_base)
            - 朝向变换公式: R_base = R⁻¹ @ R_world
        """
        # 转换四元数格式：wxyz → xyzw
        target_quat_xyzw = self._convert_wxyz_to_xyzw(target_quat_wxyz)
        return self.relaxed_ik_right.solve_position(target_pos, target_quat_xyzw, self.tolerances)

    def solve_position_both(self, target_pos_left, target_quat_wxyz_left, target_pos_right, target_quat_wxyz_right):
        """
        同时求解双臂逆运动学
        
        分别独立求解左右臂的 IK，适用于需要同步控制双臂的场景。
        
        Args:
            target_pos_left: 左臂目标位置 [x, y, z]（base_link 帧）
            target_quat_wxyz_left: 左臂目标姿态 [w, x, y, z]（base_link 帧）
            target_pos_right: 右臂目标位置 [x, y, z]（right_arm_base_link 帧）
            target_quat_wxyz_right: 右臂目标姿态 [w, x, y, z]（right_arm_base_link 帧）
        
        Returns:
            tuple: (left_joints, right_joints)
                   - left_joints: 左臂 7 个关节角度
                   - right_joints: 右臂 6 或 7 个关节角度
        
        Note:
            - 左右臂求解完全独立，互不影响
            - 适合 Gym 环境的双臂协同任务
        """
        # 转换左右臂四元数格式：wxyz → xyzw
        target_quat_xyzw_left = self._convert_wxyz_to_xyzw(target_quat_wxyz_left)
        target_quat_xyzw_right = self._convert_wxyz_to_xyzw(target_quat_wxyz_right)
        
        # 分别求解左右臂
        left_ik_solution = self.solve_position_left(target_pos_left, target_quat_xyzw_left)
        right_ik_solution = self.solve_position_right(target_pos_right, target_quat_xyzw_right)
        return left_ik_solution, right_ik_solution

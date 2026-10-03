# CODE_GRAPH
MISSION: COMPACT PROJECT MAP FOR LLM AGENTS.
PROTOCOL: Follow llm-agent-rules.md
MEMORY: See llm-agent-project-learnings.md

> Legend: * core, (↑out ↓in deps), s: symbols, d: desc

- assets/example/render_urdf_isaacgym.py (3↑ 0↓) | d: Wheel: https://drive.google.com/uc?id=1CmLNiENlLzTP1PRhc6Oj6Z_XArdSTXiR
  - s: [generate_joint_limit_trajectory [(joint_limits: np.ndarray, loop_steps: int)], main [()], of [position for all joints], parse_args [()], visualize_urdf [(urdf_file, simulate, disable_self_collision, fix_root)]]
- assets/example/render_urdf_sapien.py (4↑ 0↓) | d: Contains 12 symbols.
  - s: [@staticmethod compute_rotation_from_normal [(normal: np.ndarray)], ContactViewer [(Viewer)], __init__ [(self, renderer: sapien.render.SapienRenderer = None, shader_dir="", resolutions], build_collision_visual_shape [(], compute_rotation_from_normal [(normal: np.ndarray)], draw_contact [(self)], draw_contact_arrow [(self, pos: np.ndarray, normal: np.ndarray, color: np.ndarray)], fetch_contact [(], generate_joint_limit_trajectory [(robot: sapien.physx.PhysxArticulation, loop_steps: int)], main [()], parse_args [()], visualize_urdf [(use_rt, urdf_file, simulate, disable_self_collision, fix_root)]]
- assets/scripts/loop_all_hand_collision.py (3↑ 0↓) | d: 
  - s: []
- assets/scripts/robot_info.py (1↑ 0↓) | d: Contains 1 symbols.
  - s: [get_robot_path [(filepath: str)]]
- assets/tools/blender_io_glb.py (1↑ 0↓) | d: Import all obj file in a directory to blender scene
  - s: []
- assets/tools/blender_io_usd.py (1↑ 0↓) | d: Import all obj file in a directory to blender scene
  - s: []
- assets/tools/generate_sapien_collision_mesh.py (1↑ 0↓) | d: Setup
  - s: [main [(urdf_path: str, output_dir: str, /)], parse_urdf [(urdf_path, output_dir)]]
- assets/tools/generate_urdf_animation_isaacgym.py (4↑ 0↓) | d: Contains 4 symbols.
  - s: [generate_joint_limit_trajectory [(joint_limits: np.ndarray, loop_steps: int)], main [(], of [position for all joints], render_urdf [(urdf_path, simulate, disable_self_collision, fix_root, headless, output_video_p]]
- assets/tools/generate_urdf_animation_sapien.py (4↑ 0↓) | d: Contains 3 symbols.
  - s: [generate_joint_limit_trajectory [(robot: sapien.physx.PhysxArticulation, loop_steps: int)], main [(], render_urdf [(urdf_path, use_rt, simulate, disable_self_collision, fix_root, headless, output]]
- assets/tools/generate_urdf_collision_figure_sapien.py (4↑ 0↓) | d: Contains 3 symbols.
  - s: [build_collision_visual_shape [(], main [(], render_urdf [(urdf_path, fix_root, disable_self_collision, headless, output_image_path)]]
- assets/tools/split_urdf_collision_mesh.py (3↑ 0↓) | d: Setup
  - s: [parse_urdf [(urdf_path, new_mesh_dir)]]
- example/combination/001_align_scene.py (1↑ 0↓) | d: Contains 7 symbols.
  - s: [_kabsch_displacement_align [(src_pts, dst_pts, n_frames)], _load_hawor [(hawor_reconstruction)], _load_ras_extrinsics [(ras_output)], _svd_align_cam0 [(src_pts, dst_pts)], compute_and_save_transform_params [(ras_output, hawor_reconstruction, output_dir, force_scale=None, svd_align=False], main [()], umeyama_scale [(src_pts, dst_pts)]]
- example/combination/002_render_scene.py (0↑ 0↓) | d: 
  - s: []
- example/combination/00_run_pipeline.py (1↑ 0↓) | d: Contains 4 symbols.
  - s: [find_reconstruction_file [(hawor_dir)], main [()], run_align_render [(args)], run_step [(cmd, step_name, log_file=None)]]
- example/combination/01_align_scene.py (2↑ 0↓) | d: Contains 4 symbols.
  - s: [_detect_glb_up_axis [(all_vertices)], compute_and_save_transform_params [(ras_output, hawor_reconstruction, output_dir, force_scale=None)], main [()], umeyama_scale [(src_pts, dst_pts)]]
- example/combination/01c_depth_align.py (2↑ 0↓) | d: Contains 11 symbols.
  - s: [apply_depth_correction_glb [(hawor_data, correction_factors, hand_idx,], compute_depth_correction_glb [── 核心算法 ──], depth_map_to_glb_point [(u, v, z_depth, K, R_c2w, t_c2w, depth_scale=1000.0)], extract_hand_depth_ras [(depth_img, mask, depth_scale=1000.0, K=None)], find_model_masks [(hawor_dir)], find_reconstruction_file [── 文件查找 ──], load_ras_depth_dir [── RAS 数据加载 ──], load_ras_extrinsics [(ras_dir)], load_ras_intrinsics [(ras_dir)], main [── 主函数 ──], z_to_euclidean_depth [── 深度图操作 ──]]
- example/combination/02_render_scene.py (0↑ 0↓) | d: 
  - s: []
- example/combination/03_track_robot.py (10↑ 0↓) | d: Contains 24 symbols.
  - s: [HaworR1Pipeline [:], TrajectorySmoother [:], __init__ [(self, fps=30, max_velocity=1.5, max_acceleration=4.0,], _analyze_hand_trajectory [(self, hawor_data, mano_layer, R_coord, start_frame, num_frames)], _bidirectional_lowpass [(self, trajectory)], _bidirectional_lpf_order1 [(self, trajectory)], _bidirectional_lpf_order2 [(self, trajectory)], _clamp_acceleration [(self, trajectory)], _clamp_jerk [(self, trajectory)], _clamp_velocity [(self, trajectory)], _compute_metrics [(self, trajectory_smooth, trajectory_raw)], _compute_workspace_mapping [(self, hand_stats, base_link_p, base_link_R)], _determine_coord_transform [(self, hawor_data, mano_layer, start_frame, num_frames)], _fill_invalid_frames [(self, trajectory, valid_mask)], _iterative_clamp [(self, trajectory)], _precompute [(], _render_video [(], _setup_logger [(output_video: str)], compute_mano_joints [(mano_layer, rot, hand_pose, trans)], load_hawor_data [(hawor_dir, hand_idx=0)], main [()], run [(self, start_frame: int = 0, num_frames: int = -1)], smooth_trajectory [(self, qpos_sequence, smooth_indices)], transform_hawor_to_sapien [(points, R_transform)]]
- example/combination/04_physics_simulation.py (0↑ 0↓) | d: 
  - s: []
- example/combination/04_render_dual_arm.py (10↑ 0↓) | d: Contains 18 symbols.
  - s: [@staticmethod _prepare_urdf [(src_urdf_path, arm_prefix)], ArmInstance [============================================================================= 4. 动态臂创建 (核心泛化逻辑) =============================================================================], HandDetectionResult [:], Handedness [1. 手部类型检测 (内联实现, 复制自 02_render_scene.py _detect_hands) =============================================================================], __init__ [手部检测结果 (替代缺失的 hand_detector.HandDetectionResult)."""], _detect_glb_up_axis [(all_vertices)], _detect_hands_local [(hawor_path)], _find_reconstruction_file_local [(hawor_path)], _prepare_urdf [(src_urdf_path, arm_prefix)], compute_mano_joints [(mano_layer, rot, hand_pose, trans)], detect_hands [(hawor_dir, logger)], load_glb_transformed [(glb_path, transform_params_path, scene, logger=None)], load_hawor_npz [2. 数据加载 (从 npz 加载两手数据, 同时获取 pred_valid 等) =============================================================================], make_look_at_camera [(eye, target, up=np.array([0, 0, 1.0])], reencode_with_ffmpeg [(input_path, output_path, crf=18, fps=30, logger=None)], render_coordinate_axes [(scene, origin=(0, 0, 0)], setup_scene [============================================================================= 3. SAPIEN 场景辅助 =============================================================================], simple_ground [(scene, size=4.0, color=(0.8, 0.8, 0.85, 1.0)]]
- example/combination/05_compute_contact.py (3↑ 0↓) | d: Compute contact mask from HaWoR MANO data + RAS scene mesh.
  - s: [_fill_nan_frames [(data)], compute_contact_mask_from_fingers [(joints_world, thresh=GRASP_DIST_THRESH)], compute_contact_mask_from_object [(joints_world, verts_world, obj_verts,], compute_mango_fk [(mano_layer_class, side, trans, rot, hand_pose, betas)], extract_object_mesh_from_ras [(ras_dir, mesh_id=None)], load_hawor_data [(hawor_dir, hand_idx=0)], load_mano_layer [(side="right")], main [()], write_text_report [(output_dir, contact_mask, contact_dists, valid_mask,]]
- example/combination/05_gripper_test.py (0↑ 0↓) | d: 05_gripper_test.py — 二指夹爪控制 + 物理抓取测试
  - s: []
- example/combination/05_video_alignment.py (6↑ 0↓) | d: Contains 23 symbols.
  - s: [PoseOptimizer [:], __init__ [(self, hawor_data, cam_data, img_shape, mano_layer, logger=None)], _detect_hand_idx [(hawor_path)], _find_reconstruction_file [(hawor_path)], _get_gt_2d_joints [(self, frame_idx)], _get_sim_2d_joints [(self, frame_idx, offset_trans=np.zeros(3)], compute_frame_error [(self, frame_idx, offset_trans=np.zeros(3)], compute_mano_joints_cam_space [(cam_data, frame_idx, mano_layer_cam=None)], compute_reprojection_error [(joints_2d_gt, joints_2d_sim, valid_mask=None)], draw_2d_skeleton [(img, joints_2d, valid, color=(0, 255, 0)], load_cam_space_data [(hawor_dir, hand_idx=0)], load_hawor_data [(hawor_dir, hand_idx=0)], load_video_frames [(hawor_dir)], loss_fn [(params)], main [()], optimize_offset [(self, frame_indices=None, method="L-BFGS-B")], overlay_sim_on_video [(video_frame, sim_frame, alpha=0.5)], project_3d_to_2d [(points_3d_cam, focal, cx, cy)], rotation_matrix_to_angle_axis [(R)], run_optimize [(hawor_dir, output_dir, hand_idx=0, logger=None)], run_overlay [(hawor_dir, sim_video_path, output_path, hand_idx=0, alpha=0.5, logger=None)], run_reproj_analysis [(hawor_dir, output_path, hand_idx=0, logger=None)], world_to_cam_hawor [(points_world, R_c2w, t_c2w)]]
- example/combination/06_simple_grasp_test.py (2↑ 0↓) | d: Contains 5 symbols.
  - s: [capture [()], generate_urdf [(prefix="right")], main [═══════════════════════════════════════════════════════════ 主流程 ═══════════════════════════════════════════════════════════], set_target [(x, y, z, rz, ry, rx, f1, f2)], step_physics [(n=DECIMATION)]]
- example/combination/06_visualize_alignment.py (3↑ 0↓) | d: Contains 21 symbols.
  - s: [AlignmentVisualizer [:], __init__ [(self, glb_meshes, camera_fwd, camera_traj, hand_traj_lines,], _build_axes [()], _build_hand_trajectories [(avg_joints)], _build_hawor_camera_arrows [(positions, forwards, length=0.12, interval=10)], _build_hawor_camera_trajectory [(positions)], _build_legend [()], _detect_glb_up_axis [(all_vertices)], _find_mano_path [()], _on_r [(self, vis)], _slam_to_sapien [(pts)], compute_avg_joints [(all_mano_kps)], compute_mano_keypoints [(hawor_data, mano_layers)], compute_wrist_fallback [(hawor_data)], else [:], init_mano_layers [(hawor_data)], load_glb_meshes [(glb_path, params)], load_hawor_cameras [(hawor_data)], load_transform_params [(path)], main [()], run [(self)]]
- example/combination/07_gripper_verify.py (1↑ 0↓) | d: 夹爪物理仿真验证 — 4 个独立测试，每个有明确 PASS/FAIL 判据
  - s: [analytical_ik [(target_pos, target_R, finger_dist)], create_scene_with_gripper [═══════════════════════════════════════════════════════════ 工具函数 ═══════════════════════════════════════════════════════════], generate_urdf [(prefix="right")], main [═══════════════════════════════════════════════════════════ 主函数 ═══════════════════════════════════════════════════════════], set_target [===== 完全复用 06_simple_grasp_test.py 的流程 =====], step_physics [(robot, scene, n=DECIMATION)], test_A_fk [═══════════════════════════════════════════════════════════ 测试 A: FK 正确性 ═══════════════════════════════════════════════════════════], test_B_pd_tracking [═══════════════════════════════════════════════════════════ 测试 B: PD 关节跟踪 ═══════════════════════════════════════════════════════════], test_C_contact [═══════════════════════════════════════════════════════════ 测试 C: 接触力检测 ═══════════════════════════════════════════════════════════], test_D_grasp [═══════════════════════════════════════════════════════════ 测试 D: 物理抓取 ═══════════════════════════════════════════════════════════], test_E_track_and_grasp [═══════════════════════════════════════════════════════════ 测试 E: 跟踪 + 抓取 (同时验证) ═══════════════════════════════════════════════════════════]]
- example/combination/View/compute_thumb_tips_depth.py (3↑ 0↓) | d: Contains 1 symbols.
  - s: [main [()]]
- example/combination/View/kabsch_analysis.py (1↑ 0↓) | d: Contains 7 symbols.
  - s: [analyze_kabsch [(name, paths)], compute_pca_angle [(pts_a, pts_b)], load_ras_extrinsics [(ras_dir)], main [()], resample_hawor [(t_hawor, n_ras)], rotation_to_axis_angle [(R)], umeyama_full [(src, dst)]]
- example/combination/View/render_quick.py (4↑ 0↓) | d: Camera trajectory      : HaWoR / RAS camera position  t_c2w  ->  xform(t_c2w, s,
  - s: [add_plotly_traj [(pts, color, name, width=2, size=3)], compute_mano_joints [(d, hawor_dir)], draw_axes_arrows [(ax, origin, scale=5.0, lw=6)], draw_bbox [(ax, bounds, color, alpha=0.2)], draw_camera_axes [(ax, origin, R_cam, scale=0.35)], load_data [(hawor_dir, ras_dir, use_depth, params_p, depth_npz=None)], load_ras_cameras [(ras_dir)], load_ras_depth_data [(ras_dir)], main [()], plot_traj [===== 所有轨迹，每个独立颜色 =====], unproject_mask_center_to_glb [(mask, depth_img, K, R_c2w, t_c2w, depth_scale=1000.0)], xform [(pts, s, R, t)]]
- example/combination/View/render_viewer.py (2↑ 0↓) | d: Camera trajectory      : HaWoR / RAS camera position  t_c2w  ->  xform(t_c2w, s,
  - s: [add_axes [(fig, axis_scale)], add_camera_frustums [(fig, t0_glb, R_cam_glb, cam_s)], add_glb_bboxes [(fig, geoms)], add_world_point [(fig, pt, name, color, size=6, symbol='diamond')], get_glb_objects [(ras_dir)], load_ras_cameras [(ras_dir)], main [()], render_hand_meshes [(fig, d_src, s, R_HAND, t, hidx, step, rgb, label)], safe_layout [(fig, all_pts, title)], xform [(pts, s, R, t)]]
- example/combination/View/self_check_alignment.py (1↑ 0↓) | d: Contains 6 symbols.
  - s: [analyze_example [(name, paths)], compute_metrics [(t_ha, R_ha, t_ra, R_ra)], detect_glb_up_axis [(ras_dir)], load_ras_extrinsics [(ras_dir)], main [()], rotation_matrix [(axis, angle_deg)]]
- example/combination/View/utils.py (2↑ 0↓) | d: ── MANO hand skeleton connectivity ──
  - s: [compute_mano_joints [(rot, hp, trans, betas)], load_hawor_data [(hawor_dir, hand_idx=1, frame_idx=50)], matrix_from_compact_axis_angle [(aa)]]
- example/combination/View/vis_camera_trajectories.py (1↑ 0↓) | d: Contains 2 symbols.
  - s: [_compute_metrics [(t_ha, R_ha, t_ra, R_ra)], main [()]]
- example/combination/View/viz_mano_multiframe.py (4↑ 0↓) | d: Contains 5 symbols.
  - s: [arrow [Direction arrows], build_scene [(fd, idx_label="")], compute_frame_data [(hawor_dir, frame_idx)], compute_mano_based_gripper_pose [(j, finger_origin_x=0.03689)], make_multiframe_html [(hawor_dir, n_frames=9)]]
- example/combination/View/viz_mano_retargeting_web.py (5↑ 0↓) | d: Contains 5 symbols.
  - s: [compute_mano_based_gripper_pose [(j, prefix="right",], direction [# Default display: 3-target version], get_gripper_pose [(robot, retarget_qpos)], main [(hawor_dir, frame=50)], make_arrow [── Direction arrows ──]]
- example/combination/damping_benchmark.py (0↑ 0↓) | d: damping_benchmark.py — test8 阻尼对比实验 (通过临时补丁运行)"""
  - s: []
- example/combination/debug_analyze_mp4.py (0↑ 0↓) | d: 分析 mp4: 统计 橙色(夹爪若橙色) / 皮肤色(MANO) / 夹爪候选(非皮肤非背景, 灰金属) 像素占比."""
  - s: []
- example/combination/debug_gripper_visible.py (0↑ 0↓) | d: 诊断: 让夹爪在离屏画面中清晰可见.
  - s: [add_orange_overlays [橙色高亮叠加: 对每个 link 建一个 kinematic actor + 基础几何体(box), 颜色固定橙色. SAPIEN 基础几何体(add_box_visual)使用 RenderShape 的 base_color, 不受 GLB 内嵌材质影响.], main [()], sync_overlays [(overlays)]]
- example/combination/diagnose_mano_scale.py (4↑ 0↓) | d: diagnose_mano_scale.py — 诊断 MANO→夹爪映射的尺度/位移/开合问题
  - s: [main [()], stats [(name, arr, unit="")]]
- example/combination/hand_track/__init__.py (0↑ 0↓) | d: 批量处理所有 hawor 目录
  - s: []
- example/combination/hand_track/align_strategy.py (2↑ 0↓) | d: Contains 6 symbols.
  - s: [compute_arm_root_pose [(gripper_pos, gripper_R, gripper_offset_pos, gripper_offset_R)], compute_gripper_pose_aligned [(mano_wrist, mano_finger1, mano_finger2, prefix="right",], p [(msg)], print_verification [(results, logger=None)], project_wrist_to_pose_line [(wrist_pos, mano_wrist, mano_midpoint)], verify_alignment [(robot, prefix, mano_wrist, mano_finger1, mano_finger2, scene)]]
- example/combination/hand_track/common.py (10↑ 0↓) | d: Contains 14 symbols.
  - s: [_fill_nan_frames [(data)], _find_reconstruction_file [─── 数据加载 ────────────────────────────────────────────────────────────────], _is_active_hand [(trans_valid)], compute_mano_joints [(mano_layer, rot, hand_pose, trans)], detect_hands [(hawor_dir)], load [) ───], load_glb_transformed [(glb_path, transform_params_path, scene, logger=None)], load_hawor_c2w [(hawor_dir)], load_hawor_data [(hawor_dir, hand_idx=0)], load_ras_cameras [(ras_dir)], prepare_arm_urdf [─── URDF 预处理 ─────────────────────────────────────────────────────────────], ras_cam_to_sapien_pose [(t_c2w_ras, R_c2w_ras)], set_render_transform_params [(R_h2g, t_h2g, Rx_hand, s)], setup_scene [─── SAPIEN 场景 ─────────────────────────────────────────────────────────────]]
- example/combination/hand_track/gripper_config.py (3↑ 0↓) | d: ── 路径常量 ──
  - s: [EmaTargetSmoother [── 平滑器 ──], PositionEmaSmoother [:], __init__ [EMA 目标平滑器 (位置 + 四元数朝向)"""], compute_analytical_gripper_pose [── 解析位姿计算 ──], compute_gripper_offset_in_root [(robot, prefix)], compute_mano_based_gripper_pose [(j, prefix="right", scale=1.0)], generate_gripper_urdf [── URDF 生成函数 ──], generate_gripper_with_arm_urdf [(prefix="right", finger_origin_x=0.03689,], init_gripper_retargeting [── Retargeting 初始化 ──], prepare_full_arm_urdf [(prefix="right", finger_origin_x=0.03689,], reset [(self)], smooth [(self, pos, ori_quat)]]
- example/combination/hand_track/render_auto.py (3↑ 0↓) | d: Contains 3 symbols.
  - s: [_combine_videos_side_by_side [(left_video, right_video, output, fps, crf, logger)], _ensure_transform_params [(ras_dir, hawor_dir, output_dir, logger)], main [()]]
- example/combination/hand_track/render_dexterous_only.py (9↑ 0↓) | d: Contains 7 symbols.
  - s: [_compute_wrist_quat_sapien [(hand_pose_frame)], _ensure_transform_params [(ras_dir, hawor_dir, output_dir, logger)], _load_dexterous_robot [(robot_name_str: str, hand_type: HandType, scene, logger)], _robot_name_to_enum [(name: str)], main [()], render_dexterous_only_video [(hawor_dir, ras_dir, transform_params_path, output,], render_dual_dexterous_video [(hawor_dir, ras_dir, transform_params_path, output,]]
- example/combination/hand_track/render_gripper_only.py (10↑ 0↓) | d: Contains 7 symbols.
  - s: [_combine_videos_side_by_side [(left_video, right_video, output, fps, crf, logger)], _compute_gripper_pose [── 对齐策略选择 ──], _compute_gripper_pose_by_strategy [(strategy, mano_wrist, mano_finger1, mano_finger2,], _ensure_transform_params [(ras_dir, hawor_dir, output_dir, logger)], main [()], render_dual_gripper_video [(hawor_dir, ras_dir, transform_params_path, output,], render_gripper_only_video [(hawor_dir, ras_dir, transform_params_path, output,]]
- example/combination/hand_track/verify_optimizer_3points.py (4↑ 0↓) | d: 验证: 对比 2 点 vs 3 点 retargeting 配置的精度"""
  - s: [_render_to_sapien [(pts)], main [()], test_config [(hawor_dir, prefix, hand_idx, target_links, human_indices,]]
- example/combination/mano_gym_bridge_test.py (8↑ 0↓) | d: Contains 16 symbols.
  - s: [@property language_instruction [(self)], @property table_height [自定义环境：只保留右臂 + 手机（红色方块）"""], @property tabletop_center_x [(self)], ManoPhonePickEnv [═══════════════════════════════════════════════════════════ Step 3: 自定义 Gym 环境（加载手机场景） ═══════════════════════════════════════════════════════════], _check_termination [(self)], _get_info [(self)], _setup_green_bottle [(self)], _setup_red_bottle [(self)], capture [()], generate_mano_trajectory [═══════════════════════════════════════════════════════════ Step 1: 生成合成 MANO 轨迹（模拟抓手机） ═══════════════════════════════════════════════════════════], language_instruction [(self)], main [()], retarget_mano_to_gripper [═══════════════════════════════════════════════════════════ Step 2: Gram-Schmidt 重定向到夹爪位姿 ═══════════════════════════════════════════════════════════], solution [(self)], table_height [自定义环境：只保留右臂 + 手机（红色方块）"""], tabletop_center_x [(self)]]
- example/combination/physics_pipeline/analyze_images.py (1↑ 0↓) | d: 分析渲染的图片 - 检查方向和内容"""
  - s: []
- example/combination/physics_pipeline/compare_02.py (1↑ 0↓) | d: 提取 02 输出视频的帧，对比方向"""
  - s: []
- example/combination/physics_pipeline/diagnose_camera.py (2↑ 0↓) | d: 诊断相机位置和渲染效果"""
  - s: []
- example/combination/physics_pipeline/diagnose_camera2.py (2↑ 0↓) | d: 诊断相机位置 - 使用正确的 HaWoR 路径"""
  - s: []
- example/combination/physics_pipeline/diagnose_table.py (2↑ 0↓) | d: 诊断 GLB 物体 Z 值分布和桌子位置问题
  - s: []
- example/combination/physics_pipeline/pybullet_pipeline.py (5↑ 0↓) | d: Contains 3 symbols.
  - s: [_compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2, prefix="right")], _generate_gripper_only_urdf [(prefix="right")], prepare_urdf [(src_urdf_path)]]
- example/combination/physics_pipeline/run_physics_pipeline.py (1↑ 0↓) | d: Contains 3 symbols.
  - s: [find_reconstruction_file [(hawor_dir)], main [()], run_step [(cmd, step_name, log_file=None)]]
- example/combination/physics_pipeline/test_render.py (4↑ 0↓) | d: 测试 PyBullet 渲染 - 检查相机视角和夹爪加载 (修复版)"""
  - s: []
- example/combination/render_mano_minimal.py (4↑ 0↓) | d: render_mano_minimal.py — 极简渲染: 只画手腕 + 两指尖(拇指尖/食指尖)
  - s: [main [()]]
- example/combination/render_mano_skeleton.py (4↑ 0↓) | d: render_mano_skeleton.py — 用真实 HaWoR 数据渲染 MANO 21 关节骨架
  - s: [_render_to_sapien [(pts)], main [()]]
- example/combination/run_align_benchmark.py (2↑ 0↓) | d: Contains 2 symbols.
  - s: [main [()], parse_datasets [(ds_list)]]
- example/combination/test/View/batch_vis_3d.py (1↑ 0↓) | d: batch_vis_3d.py — 批量生成四个例子的 3D 轨迹对比图 (RAS vs Default vs SVD)"""
  - s: [load_hawor [(hawor_dir)], load_ras [(ras_dir)], load_transform [(params_path)], pca_angle [(a, b)], resample [(p, n)], transform_hawor [(R_hand_to_glb, scale_ratio, t_hand_to_glb, R_c2w_h, t_c2w_h)]]
- example/combination/test/View/batch_vis_3d_html.py (0↑ 0↓) | d: batch_vis_3d_html.py — 生成可交互的 Plotly 3D 轨迹对比"""
  - s: [load_hawor [(hawor_dir)], load_ras [(ras_dir)], load_transform [(path)], pca_angle [(a, b)], resample [(p, n)], rot_to_aa [(R)], transform_hawor [(Rh, s, t, R_c2w_h, t_c2w_h)]]
- example/combination/test/View/batch_vis_cameras.py (2↑ 0↓) | d: Contains 6 symbols.
  - s: [add_line [(pts, color, label, lw=2, size=3)], load_hawor_cameras [(hawor_dir, params_path, s_override=None, R_override=None, t_override=None)], load_ras_cameras [(ras_dir)], plot_3d_html [(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir)], plot_3d_matplotlib [(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir)], resample [(p, n)]]
- example/combination/test/View/check_first_frame.py (0↑ 0↓) | d: Contains 1 symbols.
  - s: [load_ras_extrinsics [(ras_dir)]]
- example/combination/test/View/check_slam_traj.py (0↑ 0↓) | d: Contains 5 symbols.
  - s: [load_ras_extrinsics [(ras_dir)], pca_angle [PCA 对比], quat_to_rotmat [(q)], rot_to_axis_angle [(R)], umeyama_full [用 Kabsch 在 SLAM cam0 空间找最优旋转]]
- example/combination/test/View/check_slam_traj2.py (0↑ 0↓) | d: Contains 5 symbols.
  - s: [load_ras_extrinsics [(ras_dir)], pca_angle [(pts_a, pts_b)], quat_to_rotmat [(q)], rot_to_axis_angle [(R)], umeyama_full [(src, dst)]]
- example/combination/test/View/debug_fk.py (3↑ 0↓) | d: Full debug: print FK chain of gripper retargeting."""
  - s: []
- example/combination/test/View/debug_kabsch.py (2↑ 0↓) | d: Debug: 对比 test_multi_displacement 和 001_align_scene 的差异"""
  - s: [load_ras_extrinsics [直接复制 001_align_scene.py 中的函数逻辑], pca_angle [(a, b)]]
- example/combination/test/View/debug_poses.py (2↑ 0↓) | d: Debug script: dump all pose vectors for manual inspection."""
  - s: [compute_mano_joints [(rot, hp, trans, betas)], matrix_from_compact_axis_angle [(aa)]]
- example/combination/test/View/debug_poses2.py (4↑ 0↓) | d: Debug: print all vectors to verify correctness."""
  - s: [compute_mano_joints [(rot, hp, trans, betas)], matrix_from_compact_axis_angle [(aa)]]
- example/combination/test/View/debug_v.py (3↑ 0↓) | d: Debug: verify gripper pointing directions are correct."""
  - s: [mat_from_aa [(aa)]]
- example/combination/test/View/test_multi_displacement.py (0↑ 0↓) | d: Contains 5 symbols.
  - s: [kabsch_rotation [(src, dst)], load_ras_extrinsics [(ras_dir)], pca_angle [(a, b)], resample [(points, n_target)], rot_to_axis_angle [(R)]]
- example/combination/test/View/test_two_frame_align.py (0↑ 0↓) | d: Contains 5 symbols.
  - s: [kabsch_rotation [(src, dst)], load_ras_extrinsics [(ras_dir)], pca_angle [(a, b)], resample [(points, n_target)], rot_to_axis_angle [(R)]]
- example/combination/test/View/test_two_frame_align_v2.py (0↑ 0↓) | d: Contains 5 symbols.
  - s: [kabsch_rotation [(src, dst)], load_ras_extrinsics [(ras_dir)], pca_angle [(a, b)], resample [(points, n_target)], rot_to_axis_angle [(R)]]
- example/combination/test/View/vis_alignment_3d.py (3↑ 0↓) | d: Contains 9 symbols.
  - s: [compute_joints [(ml, rot, hp, t)], draw_bbox [(ax, bounds, color, alpha=0.25)], draw_camera_frame [(ax, t, R, scale, lw=2.5)], draw_origin_axes [(ax, scale, lw=3)], find_npz [(hawor_dir)], find_params [(hawor_dir)], load_hands [(npz, s, R, t, idxs)], main [()], xform [(pts, s, R, t)]]
- example/combination/test/View/vis_camera_trajectories.py (1↑ 0↓) | d: Contains 2 symbols.
  - s: [_compute_metrics [(t_ha, R_ha, t_ra, R_ra)], main [()]]
- example/combination/test/View/vis_depth_align.py (4↑ 0↓) | d: Contains 5 symbols.
  - s: [bounding [boxes."""], centroidsX, compute_joints [(mano_layer, rot, hand_pose, trans)], create_mano_layer [(side, betas)], run_depth_align [─── Pipeline: 01c depth alignment ──────────────────────────────────────────]]
- example/combination/test/View/vis_depth_video.py (2↑ 0↓) | d: Contains 4 symbols.
  - s: [derive_focal_length [(pred_trans, masks, hidx, n_frames)], load_ras_depth_files [(ras_dir)], main [()], project_mano_to_ras [(p_cam_mano, fx, cx=960.0, cy=540.0,]]
- example/combination/test/View/vis_interactive_3d.py (3↑ 0↓) | d: Contains 6 symbols.
  - s: [draw_axes_arrows [(ax, origin, scale=5.0, lw=6)], draw_bbox [(ax, bounds, color, alpha=0.2)], draw_camera_axes [(ax, origin, R_cam, scale=1.5)], load_all_wrists [(npz_path, s, R, t)], load_mano_skeletons [(npz_path, s, R, t, n_frames_max)], main [============================================================ Main ============================================================]]
- example/combination/test/View/vis_mano_on_hawor_video.py (2↑ 0↓) | d: Contains 2 symbols.
  - s: [load_hawor_data [(hawor_dir)], main [()]]
- example/combination/test/View/vis_mask_depth.py (2↑ 0↓) | d: Contains 5 symbols.
  - s: [load_ras_depth_dir [(ras_dir)], load_ras_extrinsics [(ras_dir)], load_ras_intrinsics [(ras_dir)], main [()], unproject_depth_to_glb [(u, v, z_val, K, R_c2w, t_c2w, depth_scale=1000.0)]]
- example/combination/test/View/vis_palm_projection.py (2↑ 0↓) | d: 可视化 MANO vertex 268 在 RAS 深度图上的投影位置"""
  - s: [compute_palm_vertex [(d, hawor_dir, hidx=1)], load_ras_depth_data [(ras_dir)], main [()]]
- example/combination/test/View/vis_trajectories.py (1↑ 0↓) | d: Contains 5 symbols.
  - s: [load_ras_extrinsics [(ras_dir)], pca [计算 PCA 对比], pca_angle [(a, b)], quat_to_rotmat [(q)], resample [(points, n_target)]]
- example/combination/test/View/vis_wrist_depth_test.py (1↑ 0↓) | d: 测试: HaWoR 手腕在深度图上的投影 → 与 mask 走同一 pipeline"""
  - s: [main [()]]
- example/combination/test/analyze_distance.py (3↑ 0↓) | d: 
  - s: []
- example/combination/test/test_glb_alignment.py (2↑ 0↓) | d: Contains 9 symbols.
  - s: [find_frame_correspondence [(n_ras, n_hawor)], generate_sapien_glb_with_origins [(glb_verts_sapien, ras_origin_sapien, hawor_origin_sapien,], load_glb_dump_yup [(glb_path)], load_glb_geom_zup [(glb_path)], load_hawor_cameras [(hawor_reconstruction)], load_ras_cameras [(ras_output)], main [()], plot_origins_matplotlib [(ras_origin_sapien, hawor_origin_sapien, glb_centers, output_path)], umeyama_align [(src_pts, dst_pts)]]
- example/combination/test/test_hand_detection_and_mapping.py (3↑ 0↓) | d: Contains 27 symbols.
  - s: [_assert [(condition: bool, msg: str)], main [────────────────────────────────────────────── 主函数 ──────────────────────────────────────────────], test_arm_starting_qpos [()], test_description [()], test_detect_from_cam_space [()], test_detect_from_cam_space_left_only [()], test_detect_from_npz [()], test_detect_from_npz_no_pred_valid [()], test_end_to_end_both [()], test_end_to_end_left [()], test_end_to_end_right [────────────────────────────────────────────── Part 3: 端到端集成测试 ──────────────────────────────────────────────], test_get_config_for_hand_idx [()], test_mapper_both [()], test_mapper_left [()], test_mapper_none [()], test_mapper_right [────────────────────────────────────────────── Part 2: RobotArmMapper 测试 ──────────────────────────────────────────────], test_operator2mano [()], test_pred_valid_1d [()], test_pred_valid_2d_both [()], test_pred_valid_2d_left_only [()], test_pred_valid_2d_none [()], test_pred_valid_2d_right_only [────────────────────────────────────────────── Part 1: HandDetector 测试 ──────────────────────────────────────────────], test_pred_valid_2d_sparse [()], test_prepare_arm_urdf [()], ... +3 more]
- example/combination/test_compare/compare_hawor_dirs.py (2↑ 0↓) | d: Contains 18 symbols.
  - s: [load_ras [(ras_dir)], load_rec [(hawor_dir)], main [================== Main ==================], plot_camera_frustum_2d [(rec_7, rec_vggt)], plot_camera_trajectory_3d [================== 原 8 张图 (7 vs 7_vggt) ==================], plot_diff_heatmap [(rec_7, rec_vggt, hand_idx=0)], plot_hand_pose_pca_comparison [(rec_7, rec_vggt, hand_idx=0)], plot_hand_trajectory_3d [(rec_7, rec_vggt, hand_idx=0)], plot_keyframe_hand_overlay [(rec_7, rec_vggt, hand_idx=0)], plot_summary_table [(rec_7, rec_vggt)], plot_three_way_camera_2d [(ras, rec_7, rec_vggt)], plot_three_way_camera_3d [(ras, rec_7, rec_vggt)], plot_three_way_fov [()], plot_three_way_frustum [(ras, rec_7, rec_vggt)], plot_three_way_intrinsic [================== 新增 6 张三方对比图 ==================], plot_three_way_motion_range [()], plot_timeseries_comparison [(rec_7, rec_vggt, hand_idx=0)], rotation_to_euler_deg [(R)]]
- example/combination/test_gripper_limits.py (9↑ 0↓) | d: 自动检测活跃手 (推荐)
  - s: [compute_joint_limit [─── Step 3: 生成 URDF ──────────────────────────────────────────────────────], generate_urdf_with_limit [(joint_limit, urdf_out_dir)], load_hawor_mano_data [─── Step 1: 加载 HaWoR 数据 ─────────────────────────────────────────────────], method_A_max_in_data [─── Step 2: 三种 D_max 计算 ────────────────────────────────────────────────], method_B_monte_carlo [(mano_layer, betas_mean, n_samples=50000, sigma_range=3.0)], method_C_grid_search [(mano_layer, betas_mean, key_dims=5, sigma_range=3.0)]]
- example/combination/trajectory_optimization/__init__.py (0↑ 0↓) | d: tri_model_physics — 三形式(完整机器人/浮动臂/纯夹爪) × 双引擎(SAPIEN/PyBullet) 物理仿真
  - s: []
- example/combination/trajectory_optimization/data_loader.py (4↑ 0↓) | d: data_loader.py — 数据加载: GLB 场景 + MANO/HaWoR + 物体查找 + 相机 + 数学工具 + 控制器
  - s: [@staticmethod _mano_curl [(mano_gripper_val)], AdaptiveGraspController [:], HybridGraspController [============================================================ 混合抓取控制器: MANO 参数驱动 + 接触力控 ============================================================], JointFilter [============================================================ 7. 平滑滤波 ============================================================], __init__ [(self, alpha=LP_ALPHA_JOINT)], _find_nearest_object [(self, gripper_pos)], _get_obj_pos [(self, obj_name)], _grasped_is_falling [(self, window=5, threshold=0.005)], _grasped_is_lifting [(self, window=5, threshold=0.005)], _log_event [(self, phase, curl, obj, dist)], _mano_curl [(mano_gripper_val)], _obj_is_falling [(self, obj_name, window=5, threshold=0.005)], _obj_is_lifting [(self, obj_name, window=5, threshold=0.005)], _update_histories [(self, wrist_pos_z, nearest_obj, nearest_obj_pos)], _wrist_is_falling [(self, window=5, threshold=0.003)], _wrist_is_rising [(self, window=5, threshold=0.003)], compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2, prefix="right")], compute_glb_ground_z [============================================================ GLB 加载 ============================================================], compute_mano_joints [(mano_layer, pred_rot, pred_hand_pose, pred_trans)], find_bowl [(obj_info, exclude_names=None)], find_pink_object [(obj_info)], find_target_object_by_trajectory [============================================================ 物体查找 (按颜色/几何识别) ============================================================], hawor_cam_to_sapien_pose [============================================================ 相机 ============================================================], load_glb_with_physics [(glb_path, transform_params_path, scene, fast_collision=True,], ... +9 more]
- example/combination/trajectory_optimization/grasp_controller.py (3↑ 0↓) | d: 夹取GLB物体控制逻辑
  - s: [@dataclass GraspState [:], GraspController [:], GraspState [:], __init__ [(self, gripper_joint_names=None, finger_link_names=None)], check_contacts_pybullet [(self, robot_id, target_body_ids, physics_client)], check_contacts_sapien [(self, robot, target_actors)], compute_gripper_target [(self, desired_open, is_closing=False)], reset [(self)], update_grasp_state [(self, contact_count, object_pos=None)]]
- example/combination/trajectory_optimization/grasp_hawor.py (0↑ 0↓) | d: grasp_hawor.py — SAPIEN 物理仿真: 用 R1 机器人真实抓取 GLB 物体
  - s: []
- example/combination/trajectory_optimization/models/__init__.py (0↑ 0↓) | d: 三种机器人形式定义: full_robot / floating_arm / gripper_only"""
  - s: []
- example/combination/trajectory_optimization/models/robot_forms.py (5↑ 0↓) | d: 三种机器人形式定义: full_robot / floating_arm / gripper_only
  - s: [@dataclass RobotFormInfo [:], RobotFormInfo [:], _parse_urdf_joints [(urdf_path)]]
- example/combination/trajectory_optimization/models/urdf_templates.py (2↑ 0↓) | d: 纯夹爪URDF模板生成
  - s: [generate_gripper_only_urdf [(prefix="right")], generate_pybullet_gripper_urdf [(prefix="right")]]
- example/combination/trajectory_optimization/mujoco_backend/__init__.py (0↑ 0↓) | d: MuJoCo 后端"""
  - s: []
- example/combination/trajectory_optimization/mujoco_backend/mujoco_env.py (3↑ 0↓) | d: MuJoCo 物理仿真环境 — 三形式通用
  - s: [_make_mjcf_from_urdf [(urdf_path, side="right")], replace_gripper_joint [替换 gripper_finger_joint fixed → prismatic]]
- example/combination/trajectory_optimization/mujoco_backend/mujoco_runner.py (11↑ 0↓) | d: MuJoCo 三形式跟踪执行器"""
  - s: [LPFilter [:], MuJoCoRunner [:], __init__ [(self, alpha=0.5)], _compute_camera_pose [(self)], _compute_camera_pose_frame [(self, local_idx, hawor_data, start_frame)], _get_unified_qpos [(self)], _step_arm [(self, joints_sapien, hawor_data, global_idx)], _step_gripper_only [(self, joints_sapien)], _step_replay [(self, target_8dof, joints_sapien)], _update_camera_per_frame [(self, camera, hawor_data, global_idx)], build [(self)], disconnect [(self)], init_ik [(self)], init_retargeting [(self)], next [(self, x)], run_tracking [(self, hawor_dir, ras_dir, transform_params_path,]]
- example/combination/trajectory_optimization/physics_env.py (5↑ 0↓) | d: physics_env.py — SAPIEN 物理环境: URDF 准备 + 场景 + 机器人 + 物理步进 + 接触检测
  - s: [_is_floating_root [============================================================ 4. 物理步进 (参考 04 _physics_step — 纯 PD 驱动) ============================================================], _patched_build_entities [(self, fix_root_link=None)], _patched_check [因此必须 patch build_entities, 在 override 之后把 "undefined" 改回 "free". 1. 放宽 _check: 允许 "free" 作为根关节类型 (原代码只允许 "fixed" / "undefined")], _patched_parse_articulation [(self, root, fix_base)], fetch_contacts [============================================================ 5. 接触检测 (参考 04 _fetch_contacts) ============================================================], get_finger_contacts [============================================================ 7b. 辅助函数: 接触检测 + 力获取 + 物体在夹爪内判断 ============================================================], get_grasp_force [(side, scene, obj_actors, robot)], is_obj_in_gripper_frame [(gripper_pos, gripper_R, obj_pos,], physics_step [(robot, arm_joint_indices, gripper_idx1, gripper_idx2,], prepare_full_robot_urdf [============================================================ 1. URDF 准备 — 修复多行 joint 正则 (根因修复) ============================================================], prepare_gripper_only_urdf [(side="right", strip_visuals=False)], rotmat_to_zyx_euler [(R)], setup_physics_scene [2. SAPIEN 场景 + GLB 加载 (参考 04_physics_simulation.py) ============================================================], setup_robot [(scene, mode, side, base_pos, base_quat)]]
- example/combination/trajectory_optimization/physics_utils.py (1↑ 0↓) | d: 共享物理参数与工具函数
  - s: [compute_object_mass [(bbox_min, bbox_max, density=OBJECT_DENSITY)], is_large_scene_object [(bbox_min, bbox_max)]]
- example/combination/trajectory_optimization/stage1_grasp_mixin.py (3↑ 0↓) | d: stage1_grasp_mixin.py — Stage 1: 抓取姿态优化 (6DOF CMA-ES)
  - s: [Stage1Mixin [:], _check_mano_hard_constraint [(pos, R, ref_R_for_constraint=None)], _eval_pose [(grasp_pose)], _get_mano_f50_pose [=== _get_mano_f50_pose (L4973) ===], cem_grasp_pose_optimize [=== cem_grasp_pose_optimize (L5621) ===], compute_gripper_qpos [Stage 1: 抓取姿态优化 Mixin""" === compute_gripper_qpos (L4961) ===], generate_grasp_candidates [=== generate_grasp_candidates (L5109) ===], rollout_grasp_only [=== rollout_grasp_only (L5188) ===], score_grasp_quality [=== score_grasp_quality (L4993) ===], stage1_reward [=== stage1_reward (L5574) ===]]
- example/combination/trajectory_optimization/stage2_reconstruct_mixin.py (1↑ 0↓) | d: stage2_reconstruct_mixin.py — Stage 2: 轨迹重建 (Minimum Jerk 平滑插值)
  - s: [Stage2Mixin [:], _finite_diff [(arr, idx)], _lerp [(a, b, t)], _min_jerk_quintic [(p0, v0, p1, v1, T, t)], reconstruct_trajectory [Stage 2: 轨迹重建 Mixin""" === reconstruct_trajectory (L5869) ===]]
- example/combination/trajectory_optimization/stage3_optimize_mixin.py (4↑ 0↓) | d: stage3_optimize_mixin.py — Stage 3: 全局轨迹优化 (CEM + 平滑过渡)
  - s: [Stage3Mixin [:], cem_stage3_optimize [=== cem_stage3_optimize (L6578) ===], rollout_v4_stage3 [=== rollout_v4_stage3 (L6143) ===], stage3_reward [Stage 3: 轨迹优化 Mixin""" === stage3_reward (L6064) ===]]
- example/combination/trajectory_optimization/test/diagnose_full.py (4↑ 0↓) | d: 完整诊断: 追踪 MANO SLAM→SAPIEN→gripper base 的每一步"""
  - s: [_mano_to_sapien [(pts_slam)], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/test/diagnose_mano.py (1↑ 0↓) | d: 诊断: 对比 Original MANO palm vs Stage 3 mano_pos"""
  - s: [xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/test/grasp_demo.py (7↑ 0↓) | d: GalaxeaManipSim 物理仿真抓取 demo — 使用自定义 gym 环境
  - s: [@property language_instruction [(self)], @property table_height [用 create_box 替代 rand_create_glb, 不需要 robotwin_models"""], @property tabletop_center_x [(self)], BoxPickEnv [============ 1. 创建自定义环境类 (继承 DualBottlesPickEasyEnv) ============], _check_termination [(self)], _get_info [(self)], _setup_green_bottle [(self)], _setup_red_bottle [(self)], capture_frame [()], get_right_arm_qpos [()], get_right_ee_pose [()], get_right_gripper_qpos [()], language_instruction [(self)], main [()], solution [(self)], table_height [用 create_box 替代 rand_create_glb, 不需要 robotwin_models"""], tabletop_center_x [(self)]]
- example/combination/trajectory_optimization/test/vis_alignment.py (4↑ 0↓) | d: Contains 9 symbols.
  - s: [compute_mano_joints [(d, hand_idx)], draw_axes_arrows [(ax, origin, scale=0.15, lw=5)], draw_bbox [(ax, p0, p1, color, alpha=0.2)], draw_camera_axes [(ax, origin, R_cam, scale=0.15)], load_glb_objects [(ras_dir)], load_ras_cameras [(ras_dir)], main [()], plot_traj [(ax, pts, color, label, lw=1.0, alpha=0.8)], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/test/vis_stage_trajectories.py (3↑ 0↓) | d: vis_stage_trajectories.py — Stage 1/2/3 轨迹可视化 (GLB space, Y-up)
  - s: [_euler [(R)], sapien_to_glb [Inverse: p_glb = T_H2G + SCALE * R_H2G @ R_AXIS.T @ sapien (from data_loader inverse) Or, since we already have a global "sapien->glb":], sapien_to_glb_fixed [旧的 sapien_to_glb (T_H2G + SCALE*R_H2G@R_AXIS.T@sapien) 把轨迹甩到 Y≈-3.8 (错位 2.6m). 验证: R_AXIS.T @ mano_pos 与原始 mano_palm_glb 仅差 0.066m.], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/test/vis_stage_trajectories_left_v2.py (2↑ 0↓) | d: Contains 4 symbols.
  - s: [mano_sapien_to_glb [(pts)], sapien_to_glb [(pts)], trajectory [──], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/tests/diagnose.py (4↑ 0↓) | d: Contains 6 symbols.
  - s: [diagnose_full [(side, data_root)], diagnose_mano [(side, data_root)], load_hawor_joints [(side, data_root)], load_params [(side, data_root)], main [()], mano_to_sapien [(pts_slam, SCALE, R_hand, T_H2G)]]
- example/combination/trajectory_optimization/tests/test_stage1_visualize.py (1↑ 0↓) | d: Stage 1 最优 grasp 可视化: 加载 best_grasp.npz 后跑真实 rollout_grasp_only(80帧), 并渲染视频."""
  - s: [main [()], make_recording_step [(sim_instance)], wrapped_step [(robot, arm_joint_indices, gripper_idx1, gripper_idx2,]]
- example/combination/trajectory_optimization/tests/vis_stage1_alignment.py (2↑ 0↓) | d: 可视化 Stage 1 对齐: MANO 手部轨迹 + GLB 场景 + Grasp Pose
  - s: [glb_to_sapien [(pts, R_htg, t_htg, scale, s_inv, R_inv, t_inv)], hawor_to_sapien [(pts, s_inv, R_inv, t_inv)], load_mano_traj [(hawor_dir)], load_transform_params [(params_path)], main [()], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/tests/vis_stage1_full.py (2↑ 0↓) | d: Contains 9 symbols.
  - s: [draw_axes_arrows [(ax, origin, scale=0.1, lw=4)], draw_bbox [(ax, bounds, color, alpha=0.2)], load_glb_geoms_sapien [(glb_path, s_inv, R_inv, t_inv)], load_hawor_data_sapien [(hawor_dir, params_p, s_inv, R_inv, t_inv, use_depth=False)], load_ras_cameras_sapien [(ras_dir, s_inv, R_inv, t_inv)], main [()], parse_stage1_log [(log_path)], transform_ras_to_sapien [(pts, s_inv, R_inv, t_inv)], xform [(pts, s, R, t)]]
- example/combination/trajectory_optimization/tests/vis_stage1_simple.py (0↑ 0↓) | d: Contains 1 symbols.
  - s: [main [()]]
- example/combination/trajectory_optimization/tests/vis_stage1_trajectory.py (0↑ 0↓) | d: Contains 4 symbols.
  - s: [draw_axes [(ax, origin=[0,0,0], scale=0.1)], draw_bbox [(ax, bounds, color, alpha=0.25)], main [()], parse_log [(log_path)]]
- example/combination/trajectory_optimization/tests/vis_stage2_trajectory.py (5↑ 0↓) | d: vis_stage2_trajectory.py — Visualize Stage 2 trajectory: F44~F56 MANO→F50→MANO t
  - s: [center [Compute offset as in grasp_hawor.py _compute_neutral_offsets: offset = obj_center - (mano_pos[F50] + R[F50] @ finger_forward)], closest [to best_grasp pos], compute_mano_gripper_traj [(hawor_data, hand_idx, start_frame, num_frames, side="right")], create_mano_layer [(side, betas)], find_closest_glb_object_sapien [(glb_path, ref_pos, s_inv, R_inv, t_inv)], find_mano_model [()], main [()], position [Panel 2: Position X/Y/Z vs frame number for both Stage 2 and MANO+offset]]
- example/combination/trajectory_optimization/tests/vis_trajectory.py (6↑ 0↓) | d: vis_trajectory_optimize.py — 可视化 MANO 轨迹 vs 优化轨迹
  - s: [apply_optimized_offset [(mano_pos, mano_R, offsets)], compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2, prefix="right")], compute_gripper_trajectory [(hawor_data, hand_idx, start_frame, num_frames)], create_mano_layer [(side, betas)], find_mano_model [()], glb_to_sapien [(pos_glb, R_h2g, t_h2g, scale)], highlight [if target_pos is not None:], load_glb_centroids [(glb_path)], load_sim_output [(output_dir)], load_transform_params [(output_dir)], main [()]]
- example/combination/trajectory_optimization/traj_optimize.py (3↑ 0↓) | d: 离线轨迹优化 (第十九轮: CMA-ES + Spline Keyframes)
  - s: [CEMOptimizer [============================================================ CEM 优化器 (Cross-Entropy Method) ============================================================], MPPIOptimizer [MPPI 优化器 (Model Predictive Path Integral) ============================================================], __init__ [(self, dim, n_frames, sample_size=256, sigma=0.15,], _sample_smooth [(self)], apply_keyframe_offset [(mano_pos, mano_R, offset_6d)], apply_opt_params_to_neutral_target [(], apply_window_offset [(mano_pos, mano_R, local_idx, window_params, opt_params,], apply_xyz_offset [============================================================ 3 维 XYZ 偏移参数化 (第二十九轮: 先只优化位置接触) ============================================================], ask [(self)], build_window_params [============================================================ 帧级窗口优化参数化 (第二十六轮: 只优化靠近物体的帧) ============================================================], cem_6d_optimize [============================================================ 6 维 XYZ+RPY CEM 优化器 (Phase 2) ============================================================], cem_grasp_window_optimize [Stage 1: 让夹爪到达物体位置并夹住 修正: 动态窗口大小 + 自适应搜索范围 (F50小范围, 过渡帧中范围) ============================================================], cem_optimize [============================================================ CEM 优化器 (保留向后兼容, 第十八轮) ============================================================], cem_phase2_global [============================================================ CEM 全局优化 (654 维) ============================================================], cem_window_optimize [============================================================ PyTorch CEM 优化器 (帧级窗口, 第二十六轮) ============================================================], cem_xyz_optimize [============================================================ 3 维 XYZ CEM 优化器 ============================================================], cmaes_optimize [============================================================ CMA-ES 优化器 (借鉴 Grasp-and-Lift) ============================================================], cmaes_phase1_grasp_pose [============================================================ Phase 1: 6 维 CMA-ES 找抓取/提起位姿 (F50, F95) ============================================================], cmaes_two_stage_optimize [============================================================ 两阶段 CMA-ES (第二十一轮: 粗搜 → 精调) ============================================================], compute_frame_params [(n_frames)], compute_reward [============================================================ 奖励计算 (第二十轮: 拆分位置/姿态偏离 + 持续接近 + 抓取成功大奖励) ============================================================], compute_reward_coarse [============================================================ 粗搜奖励 (第二十一轮: 连续梯度引导) ============================================================], compute_reward_phase2 [(rollout_result, weights=REWARD_WEIGHTS_PHASE2)], compute_reward_xyz [(rollout_result, weights=REWARD_WEIGHTS_XYZ)], ... +6 more]
- example/combination/trajectory_optimization/trajectory_loader.py (3↑ 0↓) | d: 轨迹加载器 — 复用 002_render_scene.py 的 HaWoR+GLB 轨迹加载逻辑
  - s: [_load_glb_pybullet [(glb_path, scale, R_inv, t_inv, physics_client, logger=None)], _load_glb_sapien [(glb_path, scale, R_inv, t_inv, scene, logger=None)], compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2)], compute_mano_joints [(mano_layer, pred_rot, pred_hand_pose, pred_trans)], load_glb_transformed [(glb_path, transform_params_path, scene=None, backend="sapien", logger=None)], load_hawor_c2w [(hawor_dir)], load_hawor_data [(hawor_dir, hand_idx=0)]]
- example/combination/tri_model_physics/__init__.py (0↑ 0↓) | d: tri_model_physics — 三形式(完整机器人/浮动臂/纯夹爪) × 双引擎(SAPIEN/PyBullet) 物理仿真
  - s: []
- example/combination/tri_model_physics/analyze_files.py (2↑ 0↓) | d: 逐帧分析 HaWoR npz + RAS GLB 两个文件应该如何结合
  - s: [analyze_sapien_coords [(hawor_data, objects, side_name="left")], frame_by_frame [(hawor_data, objects, side_name="left")], load_hawor [(hawor_dir)], load_objects [(ras_dir, transform_params_path)], main [()]]
- example/combination/tri_model_physics/diag_alignment.py (3↑ 0↓) | d: 诊断脚本: 检查 MANO 手部轨迹与物体在 SAPIEN 坐标系中的对齐情况
  - s: [main [()]]
- example/combination/tri_model_physics/grasp_controller.py (3↑ 0↓) | d: 夹取GLB物体控制逻辑
  - s: [@dataclass GraspState [:], GraspController [:], GraspState [:], __init__ [(self, gripper_joint_names=None, finger_link_names=None)], check_contacts_pybullet [(self, robot_id, target_body_ids, physics_client)], check_contacts_sapien [(self, robot, target_actors)], compute_gripper_target [(self, desired_open, is_closing=False)], reset [(self)], update_grasp_state [(self, contact_count, object_pos=None)]]
- example/combination/tri_model_physics/grasp_demo.py (7↑ 0↓) | d: GalaxeaManipSim 物理仿真抓取 demo — 使用自定义 gym 环境
  - s: [@property language_instruction [(self)], @property table_height [用 create_box 替代 rand_create_glb, 不需要 robotwin_models"""], @property tabletop_center_x [(self)], BoxPickEnv [============ 1. 创建自定义环境类 (继承 DualBottlesPickEasyEnv) ============], _check_termination [(self)], _get_info [(self)], _setup_green_bottle [(self)], _setup_red_bottle [(self)], capture_frame [()], get_right_arm_qpos [()], get_right_ee_pose [()], get_right_gripper_qpos [()], language_instruction [(self)], main [()], solution [(self)], table_height [用 create_box 替代 rand_create_glb, 不需要 robotwin_models"""], tabletop_center_x [(self)]]
- example/combination/tri_model_physics/grasp_hawor.py (0↑ 0↓) | d: grasp_hawor.py — SAPIEN 物理仿真: 用 R1 机器人真实抓取 GLB 物体
  - s: []
- example/combination/tri_model_physics/models/__init__.py (0↑ 0↓) | d: 三种机器人形式定义: full_robot / floating_arm / gripper_only"""
  - s: []
- example/combination/tri_model_physics/models/robot_forms.py (5↑ 0↓) | d: 三种机器人形式定义: full_robot / floating_arm / gripper_only
  - s: [@dataclass RobotFormInfo [:], RobotFormInfo [:], _parse_urdf_joints [(urdf_path)]]
- example/combination/tri_model_physics/models/urdf_templates.py (2↑ 0↓) | d: 纯夹爪URDF模板生成
  - s: [generate_gripper_only_urdf [(prefix="right")], generate_pybullet_gripper_urdf [(prefix="right")]]
- example/combination/tri_model_physics/mujoco_backend/__init__.py (0↑ 0↓) | d: MuJoCo 后端"""
  - s: []
- example/combination/tri_model_physics/mujoco_backend/mujoco_env.py (3↑ 0↓) | d: MuJoCo 物理仿真环境 — 三形式通用
  - s: [_make_mjcf_from_urdf [(urdf_path, side="right")], replace_gripper_joint [替换 gripper_finger_joint fixed → prismatic]]
- example/combination/tri_model_physics/mujoco_backend/mujoco_runner.py (11↑ 0↓) | d: MuJoCo 三形式跟踪执行器"""
  - s: [LPFilter [:], MuJoCoRunner [:], __init__ [(self, alpha=0.5)], _compute_camera_pose [(self)], _compute_camera_pose_frame [(self, local_idx, hawor_data, start_frame)], _get_unified_qpos [(self)], _step_arm [(self, joints_sapien, hawor_data, global_idx)], _step_gripper_only [(self, joints_sapien)], _step_replay [(self, target_8dof, joints_sapien)], _update_camera_per_frame [(self, camera, hawor_data, global_idx)], build [(self)], disconnect [(self)], init_ik [(self)], init_retargeting [(self)], next [(self, x)], run_tracking [(self, hawor_dir, ras_dir, transform_params_path,]]
- example/combination/tri_model_physics/physics_utils.py (1↑ 0↓) | d: 共享物理参数与工具函数
  - s: [compute_object_mass [(bbox_min, bbox_max, density=OBJECT_DENSITY)], is_large_scene_object [(bbox_min, bbox_max)]]
- example/combination/tri_model_physics/pybullet_backend/__init__.py (0↑ 0↓) | d: PyBullet物理仿真后端"""
  - s: []
- example/combination/tri_model_physics/pybullet_backend/pybullet_env.py (3↑ 0↓) | d: PyBullet 物理仿真环境 — 三形式通用
  - s: [@staticmethod _make_prismatic_gripper_urdf [(urdf_path)], PyBulletEnv [:], __init__ [PyBullet 物理仿真环境 — 支持三种机器人形式"""], _load_robot [(self)], _make_prismatic_gripper_urdf [(urdf_path)], _make_pybullet_compatible_urdf [(urdf_path)], build [(self)], get_contacts [(self, body_id)], get_link_pose [(self, link_name_or_index)], get_qpos [(self)], replace_collision_stl [(match)], replace_glb_with_obj [(match)], replace_gripper_joint [2. 夹爪关节 fixed → prismatic], replace_package [(match)], set_base_position [(self, pos)], set_root_pose [(self, pos, quat_wxyz)], sort_key [排序], step_physics [(self, target_qpos=None, kinematic_arm=False)]]
- example/combination/tri_model_physics/pybullet_backend/pybullet_runner.py (11↑ 0↓) | d: PyBullet 三形式跟踪执行器
  - s: [LPFilter [:], PyBulletRunner [:], __init__ [低通滤波器"""], _compute_camera_pose [(self)], _compute_camera_pose_frame [(self, local_idx, hawor_data, start_frame)], _get_unified_qpos [(self)], _step_arm [(self, joints_sapien, hawor_data, global_idx)], _step_gripper_only [(self, joints_sapien)], _step_replay [(self, target_8dof, joints_sapien)], _update_camera_per_frame [(self, camera, hawor_data, global_idx)], build [(self)], disconnect [(self)], init_ik [(self)], init_retargeting [(self)], next [(self, x)], run_tracking [(self, hawor_dir, ras_dir, transform_params_path,]]
- example/combination/tri_model_physics/run_tri_model.py (6↑ 0↓) | d: tri_model_physics 主入口 — 三形式(完整机器人/浮动臂/纯夹爪) × 双引擎(SAPIEN/PyBullet) 物理仿真
  - s: [main [()], run_single [(form_name, backend, hawor_dir, ras_dir, transform_params_path,], setup_logging [(log_path=None)], test_models [(logger)]]
- example/combination/tri_model_physics/sapien_backend/__init__.py (0↑ 0↓) | d: SAPIEN物理仿真后端"""
  - s: []
- example/combination/tri_model_physics/sapien_backend/sapien_env.py (4↑ 0↓) | d: SAPIEN 物理仿真环境 — 三形式通用
  - s: [@staticmethod _prepare_sapien_urdf [(urdf_path, arm_prefix="right")], SapienEnv [:], __init__ [SAPIEN 物理仿真环境 — 支持三种机器人形式"""], _prepare_sapien_urdf [(urdf_path, arm_prefix="right")], build [(self)], setup_sapien_scene [(headless=True)]]
- example/combination/tri_model_physics/sapien_backend/sapien_runner.py (11↑ 0↓) | d: SAPIEN 三形式跟踪执行器
  - s: [LPFilter [:], SapienRunner [:], __init__ [低通滤波器"""], _compute_camera_pose [(self)], _compute_camera_pose_frame [(self, local_idx, hawor_data, start_frame)], _compute_optimal_fixed_base [(self, wrist_positions_sapien)], _compute_tracking_base_pos [(self, initial_base_pos, gripper_pos_fk, arm_base_quat)], _get_gripper_pose_from_retargeting [(self, retarget_qpos)], _get_unified_qpos [(self)], _hawor_cam_to_sapien_pose [(R_c2w, t_c2w)], _make_look_at_camera [(eye, target, up=np.array([0, 0, 1.0])], _place_robot_base [(self, hawor_data, mano_layer, start_frame, num_frames)], _step_arm [(self, joints_sapien, hawor_data, global_idx)], _step_gripper_only [(self, joints_sapien)], _update_camera_per_frame [(self, camera, hawor_data, global_idx)], _warm_start_retargeting [(self, hawor_data, mano_layer, start_frame, num_frames)], _warmup [(self, hawor_data, mano_layer, start_frame, num_frames)], build [(self)], init_ik [(self)], init_retargeting [(self)], next [(self, x)], run_tracking [(self, hawor_dir, ras_dir, transform_params_path,]]
- example/combination/tri_model_physics/tests/__init__.py (0↑ 0↓) | d: 测试用例"""
  - s: []
- example/combination/tri_model_physics/tests/test_full_pipeline.py (3↑ 0↓) | d: 测试: 端到端集成测试 — 每种组合运行少量帧"""
  - s: [TestPyBulletIntegration [:], TestSapienIntegration [:], find_test_data [()], test_data [PyBullet 端到端测试"""], test_floating_arm_pybullet [(self, test_data)], test_floating_arm_sapien [(self, test_data)], test_gripper_only_pybullet [(self, test_data)], test_gripper_only_sapien [(self, test_data)]]
- example/combination/tri_model_physics/tests/test_grasp.py (3↑ 0↓) | d: 测试: 夹取GLB物体"""
  - s: [TestGraspController [:], TestGraspState [:], test_compute_gripper_target [(self)], test_contact_lost_resets_grasp [(self)], test_gripper_target_clamped [(self)], test_init [测试夹取控制器"""], test_initial_state [测试抓取状态"""], test_no_contact_no_grasp [(self)], test_reset [(self)], test_sustained_contact_triggers_grasp [(self)]]
- example/combination/tri_model_physics/tests/test_models.py (3↑ 0↓) | d: 测试: 三种机器人形式 URDF 加载与结构验证"""
  - s: [TestGripperURDFGeneration [:], TestInitQpos [:], TestRobotFormInfo [:], TestURDFParsing [:], TestURDFPaths [:], test_floating_arm_info [(self)], test_floating_arm_init_qpos [验证初始关节角"""], test_floating_arm_right_urdf_exists [(self)], test_full_robot_info [验证三种形式的 RobotFormInfo 结构"""], test_full_robot_urdf_exists [验证 URDF 文件路径存在"""], test_generate_left_gripper [(self)], test_generate_right_gripper [验证纯夹爪 URDF 生成"""], test_gripper_only_info [(self)], test_gripper_only_init_qpos [(self)], test_left_side [(self)], test_mesh_dir_exists [(self)], test_parse_floating_arm_joints [验证 URDF 解析"""], test_parse_gripper_only_urdf [(self)], test_unknown_form_raises [(self)]]
- example/combination/tri_model_physics/tests/test_pybullet_backend.py (3↑ 0↓) | d: 测试: PyBullet 后端"""
  - s: [TestPyBulletEnv [:], TestPyBulletRunner [:], test_create_scene [测试 PyBullet 环境"""], test_load_floating_arm [(self)], test_load_gripper_only [(self)], test_pd_control [(self)], test_runner_build [测试 PyBullet 执行器构建"""], test_step_physics [(self)]]
- example/combination/tri_model_physics/tests/test_sapien_backend.py (3↑ 0↓) | d: 测试: SAPIEN 后端"""
  - s: [TestSapienEnv [:], TestSapienRunner [:], test_create_scene [测试 SAPIEN 环境"""], test_load_floating_arm [(self)], test_load_gripper_only [(self)], test_runner_build [测试 SAPIEN 执行器构建"""], test_set_qpos [(self)], test_step_physics [(self)]]
- example/combination/tri_model_physics/tests/test_trajectory.py (3↑ 0↓) | d: 测试: 轨迹加载与解析映射"""
  - s: [TestAnalyticalGripperPose [:], TestCoordinateTransform [:], TestHaWoRDataLoading [:], test_basic_pose [测试解析夹爪位姿计算"""], test_closed_gripper [(self)], test_degenerate_case [(self)], test_load_existing_data [(self)], test_load_nonexistent_dir [测试 HaWoR 数据加载 (需要实际数据)"""], test_open_gripper [(self)], test_rxworld_to_sapien_orthogonal [测试坐标变换"""], test_transform_preserves_norm [(self)]]
- example/combination/tri_model_physics/traj_optimize.py (2↑ 0↓) | d: 离线轨迹优化 (第十九轮: CMA-ES + Spline Keyframes)
  - s: [apply_keyframe_offset [(mano_pos, mano_R, offset_6d)], apply_opt_params_to_neutral_target [============================================================ 旧版 9 维参数辅助函数 (第十八轮向后兼容) ============================================================], apply_window_offset [(mano_pos, mano_R, local_idx, window_params, opt_params,], apply_xyz_offset [============================================================ 3 维 XYZ 偏移参数化 (第二十九轮: 先只优化位置接触) ============================================================], build_window_params [============================================================ 帧级窗口优化参数化 (第二十六轮: 只优化靠近物体的帧) ============================================================], cem_6d_optimize [============================================================ 6 维 XYZ+RPY CEM 优化器 (Phase 2) ============================================================], cem_optimize [============================================================ CEM 优化器 (保留向后兼容, 第十八轮) ============================================================], cem_window_optimize [============================================================ PyTorch CEM 优化器 (帧级窗口, 第二十六轮) ============================================================], cem_xyz_optimize [============================================================ 3 维 XYZ CEM 优化器 ============================================================], cmaes_optimize [============================================================ CMA-ES 优化器 (借鉴 Grasp-and-Lift) ============================================================], cmaes_two_stage_optimize [============================================================ 两阶段 CMA-ES (第二十一轮: 粗搜 → 精调) ============================================================], compute_reward [============================================================ 奖励计算 (第二十轮: 拆分位置/姿态偏离 + 持续接近 + 抓取成功大奖励) ============================================================], compute_reward_coarse [============================================================ 粗搜奖励 (第二十一轮: 连续梯度引导) ============================================================], compute_reward_phase2 [(rollout_result, weights=REWARD_WEIGHTS_PHASE2)], compute_reward_xyz [(rollout_result, weights=REWARD_WEIGHTS_XYZ)], interp_keyframes [============================================================ Spline Keyframe 插值 ============================================================], objective [(x)]]
- example/combination/tri_model_physics/trajectory_loader.py (3↑ 0↓) | d: 轨迹加载器 — 复用 02_render_scene.py 的 HaWoR+GLB 轨迹加载逻辑
  - s: [_load_glb_pybullet [(glb_path, scale, R_inv, t_inv, physics_client, logger=None)], _load_glb_sapien [(glb_path, scale, R_inv, t_inv, scene, logger=None)], compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2)], compute_mano_joints [(mano_layer, pred_rot, pred_hand_pose, pred_trans)], load_glb_transformed [(glb_path, transform_params_path, scene=None, backend="sapien", logger=None)], load_hawor_c2w [(hawor_dir)], load_hawor_data [(hawor_dir, hand_idx=0)]]
- example/combination/tri_model_physics/video_recorder.py (2↑ 0↓) | d: 视频录制器 — SAPIEN/PyBullet/MuJoCo 通用
  - s: [@staticmethod _look_at_quat [(pos, target)], MuJoCoCamera [:], PyBulletCamera [:], SapienCamera [:], VideoRecorder [:], __init__ [通用视频录制器"""], _look_at_quat [(pos, target)], add_frame [(self, rgb_array)], capture [(self)], save [(self)], set_pose [(self, pos, quat)]]
- example/combination/tri_model_physics/vis_trajectory.py (6↑ 0↓) | d: vis_trajectory_optimize.py — 可视化 MANO 轨迹 vs 优化轨迹
  - s: [apply_optimized_offset [(mano_pos, mano_R, offsets)], compute_analytical_gripper_pose [(mano_wrist, mano_finger1, mano_finger2, prefix="right")], compute_gripper_trajectory [(hawor_data, hand_idx, start_frame, num_frames)], create_mano_layer [(side, betas)], find_mano_model [()], highlight [if target_pos is not None:], load_alignment_transform [(output_dir)], load_glb_centroids [(glb_path)], main [()], positions [──], sapien_to_glb [(pos, R_hand_to_glb, t_hand_to_glb, scale_ratio)]]
- example/combination/utils/__init__.py (2↑ 0↓) | d: 
  - s: []
- example/combination/verify_fix.py (0↑ 0↓) | d: R_AXIS from common.py
  - s: []
- example/combination/viz_mano_full.py (4↑ 0↓) | d: Contains 5 symbols.
  - s: [_find_reconstruction_file [============================================================================ Standalone HaWoR loader ============================================================================], compute_mano_joints [(rot, hand_pose, trans, betas)], load_hawor_data_standalone [(hawor_dir, hand_idx=1)], main [============================================================================ Main ============================================================================], matrix_from_compact_axis_angle [============================================================================ MANO FK ============================================================================]]
- example/combination/viz_mano_gripper.py (4↑ 0↓) | d: Contains 5 symbols.
  - s: [_find_reconstruction_file [Standalone HaWoR data loader (no sapien dependency) ============================================================================], compute_mano_joints [(rot, hand_pose, trans, betas)], load_hawor_data_standalone [(hawor_dir, hand_idx=1)], main [============================================================================ Main visualization ============================================================================], matrix_from_compact_axis_angle [MANO layer wrapper + fallback ============================================================================]]
- example/position_retargeting/dataset.py (2↑ 0↓) | d: DexYCB Toolkit Copyright (C) 2021 NVIDIA Corporation Licensed under the GNU Gene
  - s: [@staticmethod is_object_move [(single_object_pose: np.ndarray)], DexYCBVideoDataset [:], __getitem__ [(self, item)], __init__ [(self, data_dir, hand_type="right", filter_objects=[])], __len__ [(self)], _filter_object_motion_frame [(self, capture_filter, object_pose, frame_margin=40)], _load_camera_parameters [(self)], _load_mano [(self)], _object_mesh_file [(self, object_id)], in [YCB_CLASSES.items()], inside [if len(list(set(ycb_ids) & set(filter_ids))) < 1:], is_object_move [(single_object_pose: np.ndarray)], main [(dexycb_dir: str)]]
- example/position_retargeting/hand_robot_viewer.py (9↑ 0↓) | d: 导入临时目录处理模块
  - s: [RobotHandDatasetSAPIENViewer [(HandDatasetSAPIENViewer)], __init__ [(], load_object_hand [(self, data: Dict)], render_dexycb_data [(self, data: Dict, fps=5, y_offset=0.8)]]
- example/position_retargeting/hand_viewer.py (9↑ 0↓) | d: 导入路径处理模块
  - s: [HandDatasetSAPIENViewer [:], __init__ [手部数据集SAPIEN可视化器类"""], _compute_hand_geometry [(self, hand_pose_frame, use_camera_frame=False)], _load_ycb_object [(self, ycb_id, ycb_mesh_file)], _update_hand [(self, vertex)], clear_all [(self)], clear_node [(self)], compute_smooth_shading_normal_np [(vertices, indices)], load_object_hand [(self, data: Dict)], render_dexycb_data [(self, data: Dict, fps=10)]]
- example/position_retargeting/mano_layer.py (3↑ 0↓) [deprecated: numpy types] | d: DexYCB Toolkit Copyright (C) 2021 NVIDIA Corporation Licensed under the GNU Gene
  - s: [MANOLayer [(Module)], __init__ [Wrapper layer for manopth ManoLayer."""], forward [(self, p, t)]]
- example/position_retargeting/manopth/examples/manopth_demo.py (6↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/examples/manopth_mindemo.py (2↑ 0↓) | d: Select number of principal components for pose space
  - s: []
- example/position_retargeting/manopth/mano/__init__.py (0↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano/webuser/__init__.py (0↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano/webuser/lbs.py (2↑ 0↓) | d: Contains 2 symbols.
  - s: [global_rigid_transformation [(pose, J, kintree_table, xp)], verts_core [(pose, v, J, weights, kintree_table, want_Jtr=False, xp=chumpy)]]
- example/position_retargeting/manopth/mano/webuser/posemapper.py (1↑ 0↓) | d: Contains 5 symbols.
  - s: [Rodrigues [(ch.Ch)], compute_dr_wrt [(self, wrt)], compute_r [(self)], lrotmin [(p)], posemap [(s)]]
- example/position_retargeting/manopth/mano/webuser/serialization.py (4↑ 0↓) | d: Contains 2 symbols.
  - s: [load_model [(fname_or_dict)], ready_arguments [(fname_or_dict)]]
- example/position_retargeting/manopth/mano/webuser/smpl_handpca_wrapper_HAND_only.py (4↑ 0↓) | d: Contains 2 symbols.
  - s: [load_model [(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None)], ready_arguments [(fname_or_dict, posekey4vposed='pose')]]
- example/position_retargeting/manopth/mano/webuser/verts.py (3↑ 0↓) | d: Contains 3 symbols.
  - s: [ischumpy [(x)], verts_core [(pose,], verts_decorated [(trans,]]
- example/position_retargeting/manopth/mano_v1_2/__init__.py (1↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/__init__.py (1↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/MANO___hello_world.py (3↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/MANO___render.py (11↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/SMPL+H___hello_world.py (3↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/SMPL+H___render.py (11↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/mano_v1_2/webuser/lbs.py (2↑ 0↓) | d: Contains 3 symbols.
  - s: [global_rigid_transformation [(pose, J, kintree_table, xp)], result_meta [(object)], verts_core [(pose, v, J, weights, kintree_table, want_Jtr=False, xp=chumpy)]]
- example/position_retargeting/manopth/mano_v1_2/webuser/posemapper.py (1↑ 0↓) | d: Contains 5 symbols.
  - s: [Rodrigues [(ch.Ch)], compute_dr_wrt [(self, wrt)], compute_r [(self)], lrotmin [(p)], posemap [(s)]]
- example/position_retargeting/manopth/mano_v1_2/webuser/serialization.py (4↑ 0↓) | d: Contains 4 symbols.
  - s: [backwards_compatibility_replacements [(dd)], load_model [(fname_or_dict)], ready_arguments [(fname_or_dict)], save_model [(model, fname)]]
- example/position_retargeting/manopth/mano_v1_2/webuser/smpl_handpca_wrapper.py (5↑ 0↓) | d: Contains 2 symbols.
  - s: [load_model [(fname_or_dict='./models/SMPLH_female.pkl', ncomps=12, flat_hand_mean=False, v_t], ready_arguments [(fname_or_dict, posekey4vposed='pose')]]
- example/position_retargeting/manopth/mano_v1_2/webuser/smpl_handpca_wrapper_HAND_only.py (4↑ 0↓) | d: Contains 2 symbols.
  - s: [load_model [(fname_or_dict='./models/MANO_RIGHT.pkl', ncomps=6, flat_hand_mean=False, v_temp], ready_arguments [(fname_or_dict, posekey4vposed='pose')]]
- example/position_retargeting/manopth/mano_v1_2/webuser/verts.py (3↑ 0↓) | d: Contains 3 symbols.
  - s: [ischumpy [(x)], verts_core [(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=chumpy)], verts_decorated [(trans, pose,]]
- example/position_retargeting/manopth/manopth/__init__.py (0↑ 0↓) | d: 
  - s: []
- example/position_retargeting/manopth/manopth/argutils.py (0↑ 0↓) | d: Contains 2 symbols.
  - s: [print_args [(args)], save_args [(args, save_folder, opt_prefix='opt', verbose=True)]]
- example/position_retargeting/manopth/manopth/demo.py (4↑ 0↓) | d: Contains 3 symbols.
  - s: [cam_equal_aspect_3d [(ax, verts, flip_x=False)], display_hand [(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True)], generate_random_hand [(batch_size=1, ncomps=6, mano_root='mano/models')]]
- example/position_retargeting/manopth/manopth/manolayer.py (4↑ 0↓) | d: Contains 3 symbols.
  - s: [ManoLayer [(Module)], __init__ [(self,], forward [(self,]]
- example/position_retargeting/manopth/manopth/rodrigues_layer.py (2↑ 0↓) | d: Contains 3 symbols.
  - s: [batch_rodrigues [(axisang)], quat2mat [(quat)], th_get_axis_angle [(vector)]]
- example/position_retargeting/manopth/manopth/rot6d.py (0↑ 0↓) [TODO: ] | d: Contains 4 symbols.
  - s: [compute_rotation_matrix_from_ortho6d [(poses)], cross_product [(u, v)], normalize_vector [(v)], robust_compute_rotation_matrix_from_ortho6d [(poses)]]
- example/position_retargeting/manopth/manopth/rotproj.py (0↑ 0↓) | d: GPU implementation of svd is VERY slow ~ 2 10^-3 per hit vs 5 10^-5 on cpu
  - s: [batch_rotprojs [(batches_rotmats)]]
- example/position_retargeting/manopth/manopth/tensutils.py (1↑ 0↓) | d: Contains 5 symbols.
  - s: [make_list [(tensor)], subtract_flat_id [(rot_mats)], th_pack [(tensor)], th_posemap_axisang [(pose_vectors)], th_with_zeros [(tensor)]]
- example/position_retargeting/manopth/setup.py (1↑ 0↓) | d: Contains 1 symbols.
  - s: [check_dependencies [()]]
- example/position_retargeting/manopth/test/test_demo.py (1↑ 0↓) | d: Contains 1 symbols.
  - s: [test_generate_random_hand [()]]
- example/position_retargeting/mujoco_ycb_objects.py (3↑ 0↓) | d: Contains 10 symbols.
  - s: [YCBObjectsViewer [:], __init__ [YCB 物体 MuJoCo 查看器"""], _create_mujoco_model [(self)], _generate_mjcf_xml [(self, mesh_paths)], configure_lighting [(self, bright=True)], key_callback [(keycode)], main [(], play_trajectory [(self, fps=10)], set_initial_camera_view [(self, viewer)], set_object_poses [(self, frame_idx)]]
- example/position_retargeting/pv_retargeting/r1_hand_tracking_video.py (9↑ 0↓) | d: Contains 27 symbols.
  - s: [@staticmethod _compute_gripper_orientation_from_hand [阶段3: 手部轨迹分析 遍历所有帧, 通过MANO计算手腕位置, 统计质心/范围/标准差 ──────────────────────────────────────────────────────────────────], R1TrackingPipeline [:], __init__ [(], _analyze_hand_trajectory [(self, hand_pose, viewer, start_frame, num_frames)], _bidirectional_lowpass [(self, trajectory)], _bidirectional_lpf_order1 [(self, trajectory)], _bidirectional_lpf_order2 [(self, trajectory)], _clamp_acceleration [(self, trajectory)], _clamp_jerk [(self, trajectory)], _clamp_velocity [(self, trajectory)], _compute_gripper_orientation_from_hand [阶段3: 手部轨迹分析 遍历所有帧, 通过MANO计算手腕位置, 统计质心/范围/标准差 ──────────────────────────────────────────────────────────────────], _compute_metrics [(self, trajectory_smooth, trajectory_raw)], _compute_workspace_mapping [阶段5: 工作空间映射 将手部轨迹从DexYCB坐标系平移到R1臂可达空间内 ──────────────────────────────────────────────────────────────────], _draw_annotations [(], _draw_axes [(self, bgr, origin, R, length, camera, label, label_color, pos_text="")], _fill_invalid_frames [(self, trajectory, valid_mask)], _iterative_clamp [(self, trajectory)], _joint_comfort_score [(joint_values, joint_limits)], _output_evaluation [────────────────────────────────────────────────────────────────── 阶段10: 评估报告 ──────────────────────────────────────────────────────────────────], _precompute [Warmup: 从机器人当前关节角线性插值到第一帧IK结果 防止启动时的大幅度跳变 ──────────────────────────────────────────────────────────────────], _project_point [(self, point_3d, camera)], _render_video [逐帧: set_qpos → scene.step() → camera.take_picture() → 写入 ──────────────────────────────────────────────────────────────────], _resolve_galaxea_sim_path [()], _setup_logger [(output_video: str)], ... +3 more]
- example/position_retargeting/pv_retargeting/r1_single_arm_follow.py (12↑ 0↓) | d: Contains 4 symbols.
  - s: [_detect_hand_type_from_dataset [(data_root: Path, data_id: int)], _find_project_root [()], _prepare_arm_urdf [(src_urdf_path: Path, arm_prefix: str)], try [:]]
- example/position_retargeting/render_hand_object.py (7↑ 0↓) | d: For numpy version compatibility
  - s: [main [(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10)], np [.unicode = np.unicode_], viz_hand_object [(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int)]]
- example/position_retargeting/test/r1_ik_position_test.py (6↑ 0↓) | d: Contains 7 symbols.
  - s: [_draw_axes [(bgr, origin, R, length, camera, label, label_color, pos_text="")], _project_point [(point_3d, camera)], _set_qpos_with_drive [(r1_robot, qpos, active_joints, joint_names)], _warmup_ik_solver [(ik_solver, target_pos_base, target_quat_base_wxyz, n_iters=IK_WARMUP_ITERS, log], generate_trajectory [(traj_type, num_frames, base_link_p, base_link_R, arm_max_reach=ARM_MAX_REACH)], main [()], solve_ik [(pos_world, quat_world)]]
- example/position_retargeting/test/verify_correct_urdf.py (4↑ 0↓) | d: Contains 4 symbols.
  - s: [floating_arm_fk [(joint_angles)], main [()], rot_axis_angle [(axis, angle)], strip_visual_from_urdf [(urdf_path, output_path)]]
- example/position_retargeting/visualize_hand_object.py (7↑ 0↓) | d: For numpy version compatibility
  - s: [main [(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10)], try [:], viz_hand_object [(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int)]]
- example/profiling/generate_human_data_from_video.py (3↑ 0↓) | d: Contains 2 symbols.
  - s: [detect_video [(video_path: str, output_path: str, is_right: bool)], main [(]]
- example/profiling/profile_online_retargeting.py (5↑ 0↓) | d: Contains 2 symbols.
  - s: [main [()], profile_retargeting [(retargeting: SeqRetargeting, data: List[np.ndarray])]]
- example/simulation/r1_hand_tracking_sim.py (8↑ 0↓) | d: Contains 26 symbols.
  - s: [PyBulletRenderer [:], TrajectorySmoother [:], __init__ [(self, fps=30, max_velocity=1.5, max_acceleration=4.0,], _bidirectional_lowpass [(self, trajectory)], _clamp_acceleration [(self, trajectory)], _clamp_jerk [(self, trajectory)], _clamp_velocity [(self, trajectory)], _compute_metrics [(self, trajectory_smooth, trajectory_raw)], _fill_invalid_frames [(self, trajectory, valid_mask)], _hold_all_joints [(self)], _iterative_clamp [(self, trajectory)], _joint_comfort_score [(joint_values, joint_limits)], _make_pybullet_urdf [(urdf_path_str)], _set_initial_qpos [(self)], compute_hand_geometry [(mano_layer, hand_pose_frame, camera_extrinsic)], disconnect [(self)], get_base_pose_world [(self)], get_ee_pose_world [(self)], get_link_pose_world [(self, link_index)], project_point [(self, point_3d, camera_yaw, camera_pitch, camera_distance, camera_target)], quat_xyzw_to_wxyz [(q_xyzw)], render_frame [(self, camera_yaw, camera_pitch, camera_distance, camera_target)], replace_glb [(match)], set_qpos_by_names [(self, joint_names, qpos_values)], ... +2 more]
- example/simulation/r1_simulation.py (2↑ 0↓) | d: Contains 20 symbols.
  - s: [LPFilter [:], TrajectorySmoother [:], __init__ [(self, alpha=0.5)], _bidirectional_lowpass [(self, trajectory)], _clamp_acceleration [(self, trajectory)], _clamp_jerk [(self, trajectory)], _clamp_velocity [(self, trajectory)], _iterative_clamp [(self, trajectory)], _make_pybullet_compatible_urdf [(urdf_path)], compute_workspace_mapping [(trajectory_world, base_pos, base_quat_wxyz)], next [(self, x)], quat_conjugate [(q)], quat_multiply [(q1, q2)], quat_to_rot [(q)], quat_xyzw_to_wxyz [(q_xyzw)], replace_collision_stl [(match)], replace_glb_with_obj [(match)], smooth [(self, trajectory)], transform_pose_forward [(base_pos, base_quat_wxyz, rel_pos, rel_quat_wxyz)], transform_pose_inverse [(base_pos, base_quat_wxyz, target_pos, target_quat_wxyz)]]
- example/vector_retargeting/capture_webcam.py (2↑ 0↓) | d: Contains 1 symbols.
  - s: [main [(video_path: str, video_capture_device: Union[str, int] = 0)]]
- example/vector_retargeting/detect_from_video.py (5↑ 0↓) | d: Contains 2 symbols.
  - s: [main [(], retarget_video [(]]
- example/vector_retargeting/mujoco_retargeting_viewer.py (4↑ 0↓) | d: Contains 8 symbols.
  - s: [MuJoCoRetargetingViewer [:], __init__ [MuJoCo 重定向轨迹查看器"""], configure_lighting [(self, bright=True)], key_callback [(keycode)], main [(], play_trajectory [(self)], set_initial_camera_view [(self, viewer)], set_joint_positions [(self, qpos)]]
- example/vector_retargeting/render_robot_hand.py (5↑ 0↓) | d: Convert webp ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset de
  - s: [main [(], render_by_sapien [Convert webp ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp]]
- example/vector_retargeting/show_realtime_retargeting.py (9↑ 0↓) | d: Contains 3 symbols.
  - s: [main [(], produce_frame [(queue: multiprocessing.Queue, camera_path: Optional[str] = None)], start_retargeting [(queue: multiprocessing.Queue, robot_dir: str, config_path: str)]]
- example/vector_retargeting/single_hand_detector.py (4↑ 0↓) | d: Contains 11 symbols.
  - s: [@staticmethod draw_skeleton_on_image [(], @staticmethod estimate_frame_from_hand_points [(keypoint_3d_array: np.ndarray)], @staticmethod parse_keypoint_2d [(], @staticmethod parse_keypoint_3d [(], SingleHandDetector [:], __init__ [(], detect [(self, rgb)], draw_skeleton_on_image [(], estimate_frame_from_hand_points [(keypoint_3d_array: np.ndarray)], parse_keypoint_2d [(], parse_keypoint_3d [(]]
- example/video_egocentric_retargeting/arm_motion_analyzer.py (2↑ 0↓) | d: Contains 11 symbols.
  - s: [@staticmethod compute_finger_aperture [(joint_pos, thumb_idx=4, index_idx=8)], @staticmethod compute_gripper_value [(], @staticmethod compute_hand_openness [(joint_pos)], @staticmethod compute_palm_normal [(joint_pos)], ArmMotionAnalyzer [:], __init__ [(self, hand_type=HandType.right, arm_offset=np.zeros(3)], compute_finger_aperture [(joint_pos, thumb_idx=4, index_idx=8)], compute_gripper_value [(], compute_hand_openness [(joint_pos)], compute_palm_normal [(joint_pos)], compute_wrist_6dof [(self, hand_data, camera_extrinsics=None)]]
- example/video_egocentric_retargeting/contact_preservation.py (2↑ 0↓) | d: Contains 6 symbols.
  - s: [ContactPreservationFilter [:], TrajectorySmoother [:], __init__ [(], reset [(self)], smooth [(self, qpos)], update [(self, finger_distance, qpos)]]
- example/video_egocentric_retargeting/egocentric_hand_detector.py (4↑ 0↓) | d: Contains 14 symbols.
  - s: [@staticmethod _compute_wrist_pos_camera [(keypoint_2d, keypoint_3d_array, img_w, img_h)], @staticmethod _estimate_frame_from_hand_points [(keypoint_3d_array: np.ndarray)], @staticmethod _parse_keypoint_3d [(keypoint_3d)], @staticmethod draw_skeleton_on_image [(image, keypoint_2d, style="white")], @staticmethod parse_keypoint_2d [(keypoint_2d, img_size)], EgocentricHandDetector [:], __init__ [(], _compute_wrist_pos_camera [(keypoint_2d, keypoint_3d_array, img_w, img_h)], _estimate_frame_from_hand_points [(keypoint_3d_array: np.ndarray)], _parse_keypoint_3d [(keypoint_3d)], detect [(self, rgb)], detect_single [(self, rgb)], draw_skeleton_on_image [(image, keypoint_2d, style="white")], parse_keypoint_2d [(keypoint_2d, img_size)]]
- example/video_egocentric_retargeting/egocentric_sapien_render.py (10↑ 0↓) | d: Contains 16 symbols.
  - s: [EgocentricSAPIENRenderer [:], __init__ [(], _add_capsule [(self, p1, p2, radius, mat, node_list)], _add_sphere [(self, pos, radius, mat, node_list)], _clear_nodes [(self, node_list)], _render_axes [(self, wrist_pos, wrist_rot, robot_ee_pos=None, robot_ee_rot=None)], _render_mano_skeleton [(self, joint_pos, wrist_rot, origin_pos)], _render_trails [(self, mano_pos, robot_ee_pos)], _set_robot_pose_from_wrist [(self, wrist_pos, wrist_rot_mano)], _setup_materials [(self)], _setup_retargeting [(self)], _setup_robot [(self)], _setup_sapien_scene [(self)], main [(], process_video [(self, video_path: str, output_path: str)], rotation_from_z_to_dir [(direction)]]
- example/video_egocentric_retargeting/gripper_video_pipeline.py (14↑ 0↓) | d: Contains 22 symbols.
  - s: [GripperSAPIENRenderer [:], __init__ [(], _add_capsule [(self, p1, p2, radius, mat, node_list)], _add_sphere [(self, pos, radius, mat, node_list)], _clear_nodes [(self, node_list)], _render_axes [(self, wrist_pos, wrist_rot, robot_ee_pos=None, robot_ee_rot=None)], _render_mano_skeleton [(self, joint_pos, wrist_rot, origin_pos)], _render_trails [(self, mano_pos, robot_ee_pos)], _setup_materials [(self)], _setup_retargeting [(self)], _setup_robot [(self)], _setup_sapien_scene [(self)], _update_camera [(self, wrist_pos_base)], composite_gripper_on_frame [(original_frame, gripper_bgr, gripper_mask, hand_mask)], compute_hand_mask [(keypoint_2d, img_h, img_w, dilation_pixels=30)], main [()], render_frame [(self, keypoint_2d=None, frame_idx=0, num_box=0, mano_origin=None)], render_software_gripper_overlay [(], render_trajectory_video [(pkl_path, output_video, fps=30)], rotation_from_z_to_dir [(direction)], run_gripper_replacement_pipeline [(], update_gripper [(self, wrist_pos_camera, mediapipe_wrist_rot, qpos, joint_pos=None)]]
- example/video_egocentric_retargeting/r1_sapien_sim.py (0↑ 0↓) | d: 
  - s: []
- example/video_egocentric_retargeting/render_trajectory.py (6↑ 0↓) | d: Contains 2 symbols.
  - s: [main [(], render_trajectory [(]]
- example/video_egocentric_retargeting/robot_hand_retargeter.py (4↑ 0↓) | d: Contains 6 symbols.
  - s: [RobotHandRetargeter [:], __init__ [(], compute_ref_value [(self, joint_pos)], get_meta_data [(self)], retarget [(self, joint_pos, wrist_pos=None, wrist_quat=None, fixed_qpos=None)], warm_start [(self, wrist_pos, wrist_quat)]]
- example/video_egocentric_retargeting/run_pipeline.py (10↑ 0↓) | d: Contains 3 symbols.
  - s: [main [()], render_software_trajectory [(pkl_path, output_video, fps=30)], run_full_pipeline [(]]
- example/video_egocentric_retargeting/video_to_trajectory.py (8↑ 0↓) | d: Contains 2 symbols.
  - s: [main [(], process_video_to_trajectory [(]]
- src/dex_retargeting/__init__.py (2↑ 0↓) | d: 
  - s: []
- src/dex_retargeting/constants.py (2↑ 0↓) | d: Contains 6 symbols.
  - s: [HandType [(enum.Enum)], RetargetingType [(enum.Enum)], RobotName [(enum.Enum)], from [typing import Optional], get_default_config_path [(], interaction [data]]
- src/dex_retargeting/kinematics_adaptor.py (3↑ 0↓) | d: Index mapping
  - s: [@abstractmethod backward_jacobian [(self, jacobian: np.ndarray)], @abstractmethod forward_qpos [(self, qpos: np.ndarray)], KinematicAdaptor [:], MimicJointKinematicAdaptor [(KinematicAdaptor)], __init__ [(self, robot: RobotWrapper, target_joint_names: List[str])], backward_jacobian [(self, jacobian: np.ndarray)], forward_qpos [(self, qpos: np.ndarray)]]
- src/dex_retargeting/optimizer.py (6↑ 0↓) | d: Contains 18 symbols.
  - s: [@abstractmethod get_objective_function [(], @property fixed_joint_names [(self)], @staticmethod generate_link_indices [(num_fingers)], @staticmethod set_dexpilot_cache [(num_fingers, eta1, eta2)], DexPilotOptimizer [(Optimizer)], Optimizer [:], PositionOptimizer [(Optimizer)], VectorOptimizer [(Optimizer)], __init__ [(], fixed_joint_names [(self)], generate_link_indices [(num_fingers)], get_link_indices [(self, target_link_names)], get_objective_function [(], objective [(x: np.ndarray, grad: np.ndarray)], retarget [(self, ref_value, fixed_qpos, last_qpos)], set_dexpilot_cache [(num_fingers, eta1, eta2)], set_joint_limit [(self, joint_limits: np.ndarray, epsilon=1e-3)], set_kinematic_adaptor [(self, adaptor: KinematicAdaptor)]]
- src/dex_retargeting/optimizer_utils.py (0↑ 0↓) | d: Contains 4 symbols.
  - s: [LPFilter [:], __init__ [(self, alpha)], next [(self, x)], reset [(self)]]
- src/dex_retargeting/retargeting_config.py (10↑ 0↓) | d: Contains 14 symbols.
  - s: [@classmethod from_dict [(cls, cfg: Dict[str, Any], override: Optional[Dict] = None)], @classmethod load_from_file [(], @classmethod set_default_urdf_dir [(cls, urdf_dir: Union[str, Path])], @dataclass RetargetingConfig [:], RetargetingConfig [:], __post_init__ [(self)], build [(self)], check [self.type = self.type.lower()], from_dict [(cls, cfg: Dict[str, Any], override: Optional[Dict] = None)], get_retargeting_config [(config_path: Union[str, Path])], load_from_file [(], not [in self._TYPE:], parse_mimic_joint [(], set_default_urdf_dir [(cls, urdf_dir: Union[str, Path])]]
- src/dex_retargeting/robot_wrapper.py (1↑ 0↓) | d: Create robot model and data
  - s: [@property dof [(self)], @property dof_joint_names [(self)], @property joint_limits [(self)], @property joint_names [-------------------------------------------------------------------------- # Robot property -------------------------------------------------------------------------- #], @property link_names [(self)], RobotWrapper [:], __init__ [(self, urdf_path: str, use_collision=False, use_visual=False)], compute_forward_kinematics [-------------------------------------------------------------------------- # Kinematics function -------------------------------------------------------------------------- #], compute_single_link_local_jacobian [(self, qpos, link_id: int)], dof [(self)], dof_joint_names [(self)], get_joint_index [-------------------------------------------------------------------------- # Query function -------------------------------------------------------------------------- #], get_joint_parent_child_frames [(self, joint_name: str)], get_link_index [(self, name: str)], get_link_pose [(self, link_id: int)], get_link_pose_inv [(self, link_id: int)], joint_limits [(self)], joint_names [-------------------------------------------------------------------------- # Robot property -------------------------------------------------------------------------- #], link_names [(self)]]
- src/dex_retargeting/seq_retarget.py (6↑ 0↓) | d: Contains 11 symbols.
  - s: [@property joint_names [(self)], SeqRetargeting [:], __init__ [(], can [only be used when the first joints of robot are free joints], get_qpos [(self, fixed_qpos: Optional[np.ndarray] = None)], joint_names [(self)], reset [(self)], retarget [(self, ref_value, fixed_qpos=np.array([])], set_qpos [(self, robot_qpos: np.ndarray)], verbose [(self)], warm_start [(]]
- src/dex_retargeting/yourdfpy.py (7↑ 0↓) [deprecated: warning,TODO: thatA filename handler that joins a file name with] | d: Code from yourdfpy with small modification for deprecated warning We also remove
  - s: [@dataclass Calibration [:], @dataclass Cylinder [:], @dataclass Geometry [:], @dataclass Material [:], @dataclass Mimic [:], @dataclass SafetyController [:], @dataclass Sphere [:], @dataclass Texture [:], @dataclass(eq=False) Actuator [:], @dataclass(eq=False) Box [:], @dataclass(eq=False) Color [:], @dataclass(eq=False) Mesh [:], @dataclass(eq=False) Transmission [:], @dataclass(eq=False) TransmissionJoint [:], @dataclass(eq=False) Visual [:], @property actuated_dof_indices [(self)], @property actuated_joint_indices [(self)], @property actuated_joint_names [(self)], @property actuated_joints [(self)], @property center_cfg [(self)], @property joint_map [(self)], @property joint_names [(self)], @property link_map [(self)], @property num_actuated_joints [(self)], ... +49 more]
- tests/test_optimizer.py (6↑ 0↓) | d: Contains 13 symbols.
  - s: [@staticmethod compute_pin_qpos [(], @staticmethod generate_position_retargeting_data_gt [(], @staticmethod generate_vector_retargeting_data_gt [(], @staticmethod sample_qpos [(optimizer: Optimizer)], TestOptimizer [:], compared [to position retargeting to set initial qpos and perform optimization], compute_pin_qpos [(], generate_position_retargeting_data_gt [(], generate_vector_retargeting_data_gt [(], sample_qpos [(optimizer: Optimizer)], test_dexpilot_optimizer [(self, robot_name, hand_type)], test_position_optimizer [(self, robot_name, hand_type)], test_vector_optimizer [(self, robot_name, hand_type)]]
- tests/test_retargeting_config.py (3↑ 0↓) | d: Contains 5 symbols.
  - s: [TestRetargetingConfig [:], test_add_dummy_joint [(self, config_path)], test_dict_config_parsing [(self)], test_multi_dict_config_parsing [(self)], test_path_config_parsing [(self, config_path)]]

## EDGES
[BoxPickEnv] -> [inherits] -> [DualBottlesPickEasyEnv]
[BoxPickEnv] -> [inherits] -> [DualBottlesPickEasyEnv]
[ContactViewer] -> [inherits] -> [Viewer]
[DexPilotOptimizer] -> [inherits] -> [Optimizer]
[HandType] -> [inherits] -> [enum]
[MANOLayer] -> [inherits] -> [Module]
[ManoLayer] -> [inherits] -> [Module]
[ManoPhonePickEnv] -> [inherits] -> [DualBottlesPickEasyEnv]
[MimicJointKinematicAdaptor] -> [inherits] -> [KinematicAdaptor]
[PositionOptimizer] -> [inherits] -> [Optimizer]
[RetargetingType] -> [inherits] -> [enum]
[RobotHandDatasetSAPIENViewer] -> [inherits] -> [HandDatasetSAPIENViewer]
[RobotName] -> [inherits] -> [enum]
[Rodrigues] -> [inherits] -> [ch]
[Rodrigues] -> [inherits] -> [ch]
[URDFAttributeValueError] -> [inherits] -> [URDFError]
[URDFBrokenRefError] -> [inherits] -> [URDFError]
[URDFMalformedError] -> [inherits] -> [URDFError]
[URDFSaveValidationError] -> [inherits] -> [URDFError]
[URDFUnsupportedError] -> [inherits] -> [URDFError]
[VectorOptimizer] -> [inherits] -> [Optimizer]
[assets/example/render_urdf_isaacgym.py] -> [isaacgym, pathlib, the]
[assets/example/render_urdf_sapien.py] -> [sapien, sapien.utils, the, typing]
[assets/scripts/loop_all_hand_collision.py] -> [pathlib, robot_info, tqdm]
[assets/scripts/robot_info.py] -> [pathlib]
[assets/tools/blender_io_glb.py] -> [pathlib]
[assets/tools/blender_io_usd.py] -> [pathlib]
[assets/tools/generate_sapien_collision_mesh.py] -> [pathlib]
[assets/tools/generate_urdf_animation_isaacgym.py] -> [isaacgym, pathlib, the, typing]
[assets/tools/generate_urdf_animation_sapien.py] -> [pathlib, sapien.asset, sapien.utils, typing]
[assets/tools/generate_urdf_collision_figure_sapien.py] -> [pathlib, sapien, sapien.utils, typing]
[assets/tools/split_urdf_collision_mesh.py] -> [pathlib, xml.dom, xml.etree]
[centroidsX] -> [inherits] -> [m]
[example/combination/001_align_scene.py] -> [glob]
[example/combination/00_run_pipeline.py] -> [pathlib]
[example/combination/01_align_scene.py] -> [glob, scipy.spatial]
[example/combination/01c_depth_align.py] -> [glob, pathlib]
[example/combination/03_track_robot.py] -> [dex_retargeting, dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, galaxea_sim.controllers.utils.relaxed_ik_solver, hand_robot_viewer, mano_layer, pathlib, pytransform3d]
[example/combination/04_render_dual_arm.py] -> [dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, mano_layer, pathlib, pytransform3d, sapien.asset, sapien.core, tqdm]
[example/combination/05_compute_contact.py] -> [mano_layer, pathlib, scipy.spatial.distance]
[example/combination/05_video_alignment.py] -> [copy, mano_layer, pathlib, pytransform3d, scipy.optimize, tqdm]
[example/combination/06_simple_grasp_test.py] -> [pathlib, scipy.spatial.transform]
[example/combination/06_visualize_alignment.py] -> [glob, mano_layer, pathlib]
[example/combination/07_gripper_verify.py] -> [pathlib]
[example/combination/View/compute_thumb_tips_depth.py] -> [depth-aligned, mano_layer, pathlib]
[example/combination/View/kabsch_analysis.py] -> [pathlib]
[example/combination/View/render_quick.py] -> [mano_layer, matplotlib.ticker, mpl_toolkits.mplot3d, pathlib]
[example/combination/View/render_viewer.py] -> [mano_layer, pathlib]
[example/combination/View/self_check_alignment.py] -> [pathlib]
[example/combination/View/utils.py] -> [mano_layer, pathlib]
[example/combination/View/vis_camera_trajectories.py] -> [mpl_toolkits.mplot3d]
[example/combination/View/viz_mano_multiframe.py] -> [gripper_config, pathlib, plotly.subplots, utils]
[example/combination/View/viz_mano_retargeting_web.py] -> [datetime, gripper_config, pathlib, plotly, utils]
[example/combination/diagnose_mano_scale.py] -> [hand_track.common, hand_track.gripper_config, mano_layer, pathlib]
[example/combination/hand_track/align_strategy.py] -> [gripper_config, pytransform3d]
[example/combination/hand_track/common.py] -> [common, dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, mano_layer, pathlib, pytransform3d, sapien.asset, tqdm]
[example/combination/hand_track/gripper_config.py] -> [dex_retargeting.retargeting_config, pathlib, pytransform3d]
[example/combination/hand_track/render_auto.py] -> [common, pathlib, render_gripper_only]
[example/combination/hand_track/render_dexterous_only.py] -> [common, dex_retargeting, dex_retargeting.constants, dex_retargeting.retargeting_config, mano_layer, pathlib, pytransform3d, sapien.utils, tqdm]
[example/combination/hand_track/render_gripper_only.py] -> [align_strategy, common, dex_retargeting.constants, dex_retargeting.retargeting_config, gripper_config, mano_layer, pathlib, pytransform3d, sapien.utils, tqdm]
[example/combination/hand_track/verify_optimizer_3points.py] -> [common, dex_retargeting.constants, dex_retargeting.retargeting_config, mano_layer]
[example/combination/mano_gym_bridge_test.py] -> [copy, galaxea_sim.envs.robotwin.dual_bottles_pick_easy, galaxea_sim.planners.bimanual, galaxea_sim.robots.r1, galaxea_sim.utils.robotwin_utils, pathlib, sapien, scipy.spatial.transform]
[example/combination/physics_pipeline/analyze_images.py] -> [pathlib]
[example/combination/physics_pipeline/compare_02.py] -> [pathlib]
[example/combination/physics_pipeline/diagnose_camera.py] -> [pathlib, pybullet_pipeline]
[example/combination/physics_pipeline/diagnose_camera2.py] -> [pathlib, pybullet_pipeline]
[example/combination/physics_pipeline/diagnose_table.py] -> [pathlib, pybullet_pipeline]
[example/combination/physics_pipeline/pybullet_pipeline.py] -> [dex_retargeting.constants, hand_track, mano_layer, pathlib, pytransform3d]
[example/combination/physics_pipeline/run_physics_pipeline.py] -> [pathlib]
[example/combination/physics_pipeline/test_render.py] -> [mano_layer, pathlib, pybullet_pipeline, pytransform3d]
[example/combination/render_mano_minimal.py] -> [hand_track.common, mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/combination/render_mano_skeleton.py] -> [hand_track.common, mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/combination/run_align_benchmark.py] -> [datetime, pathlib]
[example/combination/test/View/batch_vis_3d.py] -> [mpl_toolkits.mplot3d]
[example/combination/test/View/batch_vis_cameras.py] -> [mpl_toolkits.mplot3d, pathlib]
[example/combination/test/View/debug_fk.py] -> [gripper_config, mano_layer, pathlib]
[example/combination/test/View/debug_kabsch.py] -> [importlib, importlib.machinery]
[example/combination/test/View/debug_poses.py] -> [gripper_config, mano_layer]
[example/combination/test/View/debug_poses2.py] -> [FINGER_GEOM_ARRAYS, gripper_config, mano_layer, pathlib]
[example/combination/test/View/debug_v.py] -> [gripper_config, mano_layer, pathlib]
[example/combination/test/View/vis_alignment_3d.py] -> [mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/combination/test/View/vis_camera_trajectories.py] -> [mpl_toolkits.mplot3d]
[example/combination/test/View/vis_depth_align.py] -> [mano_layer, mpl_toolkits.mplot3d.art3d, pathlib, raw]
[example/combination/test/View/vis_depth_video.py] -> [mano_layer, pathlib]
[example/combination/test/View/vis_interactive_3d.py] -> [mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/combination/test/View/vis_mano_on_hawor_video.py] -> [mano_layer, pathlib]
[example/combination/test/View/vis_mask_depth.py] -> [mpl_toolkits.mplot3d, pathlib]
[example/combination/test/View/vis_palm_projection.py] -> [mano_layer, pathlib]
[example/combination/test/View/vis_trajectories.py] -> [mpl_toolkits.mplot3d]
[example/combination/test/View/vis_wrist_depth_test.py] -> [pathlib]
[example/combination/test/analyze_distance.py] -> [mano_layer, pathlib, scipy.spatial]
[example/combination/test/test_glb_alignment.py] -> [glob, mpl_toolkits.mplot3d]
[example/combination/test/test_hand_detection_and_mapping.py] -> [hand_detector, pathlib, robot_arm_config]
[example/combination/test_compare/compare_hawor_dirs.py] -> [matplotlib.gridspec, pathlib]
[example/combination/test_gripper_limits.py] -> [hand_track.common, itertools, mano_layer, manopth.manolayer, pathlib, pytransform3d, sapien, sapien.asset, tqdm]
[example/combination/trajectory_optimization/data_loader.py] -> [pathlib, physics_env, pytransform3d, trajectory_loader]
[example/combination/trajectory_optimization/grasp_controller.py] -> [dataclasses, physics_utils, typing]
[example/combination/trajectory_optimization/models/robot_forms.py] -> [dataclasses, models.urdf_templates, pathlib, physics_utils, typing]
[example/combination/trajectory_optimization/models/urdf_templates.py] -> [pathlib, physics_utils]
[example/combination/trajectory_optimization/mujoco_backend/mujoco_env.py] -> [models.robot_forms, pathlib, physics_utils]
[example/combination/trajectory_optimization/mujoco_backend/mujoco_runner.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, grasp_controller, mano_layer, mujoco_backend.mujoco_env, pathlib, physics_utils, pytransform3d, trajectory_loader, video_recorder]
[example/combination/trajectory_optimization/physics_env.py] -> [pathlib, pytransform3d, sapien.asset, sapien.wrapper.articulation_builder, sapien.wrapper.urdf_loader]
[example/combination/trajectory_optimization/physics_utils.py] -> [pathlib]
[example/combination/trajectory_optimization/stage1_grasp_mixin.py] -> [data_loader, physics_env, scipy.spatial.transform]
[example/combination/trajectory_optimization/stage2_reconstruct_mixin.py] -> [scipy.spatial.transform]
[example/combination/trajectory_optimization/stage3_optimize_mixin.py] -> [data_loader, physics_env, scipy.spatial.transform, traj_optimize]
[example/combination/trajectory_optimization/test/diagnose_full.py] -> [data_loader, data_loader.py, mano_layer, trajectory_loader]
[example/combination/trajectory_optimization/test/diagnose_mano.py] -> [mano_layer]
[example/combination/trajectory_optimization/test/grasp_demo.py] -> [copy, galaxea_sim.envs.robotwin.dual_bottles_pick_easy, galaxea_sim.planners.bimanual, galaxea_sim.robots.r1, galaxea_sim.utils.robotwin_utils, pathlib, sapien]
[example/combination/trajectory_optimization/test/vis_alignment.py] -> [mano_layer, matplotlib.ticker, mpl_toolkits.mplot3d, pathlib]
[example/combination/trajectory_optimization/test/vis_stage_trajectories.py] -> [data_loader, mano_layer, mpl_toolkits.mplot3d]
[example/combination/trajectory_optimization/test/vis_stage_trajectories_left_v2.py] -> [mano_layer, mpl_toolkits.mplot3d]
[example/combination/trajectory_optimization/tests/diagnose.py] -> [data_loader, mano_layer, pathlib, trajectory_loader]
[example/combination/trajectory_optimization/tests/test_stage1_visualize.py] -> [grasp_hawor]
[example/combination/trajectory_optimization/tests/vis_stage1_alignment.py] -> [mpl_toolkits.mplot3d, pathlib]
[example/combination/trajectory_optimization/tests/vis_stage1_full.py] -> [mpl_toolkits.mplot3d, pathlib]
[example/combination/trajectory_optimization/tests/vis_stage2_trajectory.py] -> [mano_layer, mpl_toolkits.mplot3d, pathlib, physics_utils, trajectory_loader]
[example/combination/trajectory_optimization/tests/vis_trajectory.py] -> [mano_layer, mpl_toolkits.mplot3d, pathlib, scipy.spatial.transform, traj_optimize, trajectory_loader]
[example/combination/trajectory_optimization/traj_optimize.py] -> [scipy.interpolate, scipy.ndimage, scipy.spatial.transform]
[example/combination/trajectory_optimization/trajectory_loader.py] -> [pathlib, physics_utils, pytransform3d]
[example/combination/tri_model_physics/analyze_files.py] -> [pathlib, trajectory_loader]
[example/combination/tri_model_physics/diag_alignment.py] -> [pathlib, physics_utils, trajectory_loader]
[example/combination/tri_model_physics/grasp_controller.py] -> [dataclasses, physics_utils, typing]
[example/combination/tri_model_physics/grasp_demo.py] -> [copy, galaxea_sim.envs.robotwin.dual_bottles_pick_easy, galaxea_sim.planners.bimanual, galaxea_sim.robots.r1, galaxea_sim.utils.robotwin_utils, pathlib, sapien]
[example/combination/tri_model_physics/models/robot_forms.py] -> [dataclasses, models.urdf_templates, pathlib, physics_utils, typing]
[example/combination/tri_model_physics/models/urdf_templates.py] -> [pathlib, physics_utils]
[example/combination/tri_model_physics/mujoco_backend/mujoco_env.py] -> [models.robot_forms, pathlib, physics_utils]
[example/combination/tri_model_physics/mujoco_backend/mujoco_runner.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, grasp_controller, mano_layer, mujoco_backend.mujoco_env, pathlib, physics_utils, pytransform3d, trajectory_loader, video_recorder]
[example/combination/tri_model_physics/physics_utils.py] -> [pathlib]
[example/combination/tri_model_physics/pybullet_backend/pybullet_env.py] -> [models.robot_forms, pathlib, physics_utils]
[example/combination/tri_model_physics/pybullet_backend/pybullet_runner.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, grasp_controller, mano_layer, pathlib, physics_utils, pybullet_backend.pybullet_env, pytransform3d, trajectory_loader, video_recorder]
[example/combination/tri_model_physics/run_tri_model.py] -> [models.robot_forms, mujoco_backend.mujoco_runner, pathlib, physics_utils, pybullet_backend.pybullet_runner, sapien_backend.sapien_runner]
[example/combination/tri_model_physics/sapien_backend/sapien_env.py] -> [models.robot_forms, pathlib, physics_utils, sapien.asset]
[example/combination/tri_model_physics/sapien_backend/sapien_runner.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, grasp_controller, mano_layer, pathlib, physics_utils, pytransform3d, sapien_backend.sapien_env, trajectory_loader, video_recorder]
[example/combination/tri_model_physics/tests/test_full_pipeline.py] -> [pathlib, pybullet_backend.pybullet_runner, sapien_backend.sapien_runner]
[example/combination/tri_model_physics/tests/test_grasp.py] -> [grasp_controller, pathlib, physics_utils]
[example/combination/tri_model_physics/tests/test_models.py] -> [models.robot_forms, pathlib, physics_utils]
[example/combination/tri_model_physics/tests/test_pybullet_backend.py] -> [pathlib, pybullet_backend.pybullet_env, pybullet_backend.pybullet_runner]
[example/combination/tri_model_physics/tests/test_sapien_backend.py] -> [pathlib, sapien_backend.sapien_env, sapien_backend.sapien_runner]
[example/combination/tri_model_physics/tests/test_trajectory.py] -> [pathlib, physics_utils, trajectory_loader]
[example/combination/tri_model_physics/traj_optimize.py] -> [scipy.interpolate, scipy.spatial.transform]
[example/combination/tri_model_physics/trajectory_loader.py] -> [pathlib, physics_utils, pytransform3d]
[example/combination/tri_model_physics/video_recorder.py] -> [pathlib, pytransform3d]
[example/combination/tri_model_physics/vis_trajectory.py] -> [mano_layer, mpl_toolkits.mplot3d, pathlib, scipy.spatial.transform, traj_optimize, trajectory_loader]
[example/combination/utils/__init__.py] -> [hand_detector, robot_arm_config]
[example/combination/viz_mano_full.py] -> [gripper_config, mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/combination/viz_mano_gripper.py] -> [gripper_config, mano_layer, mpl_toolkits.mplot3d, pathlib]
[example/position_retargeting/dataset.py] -> [collections, pathlib]
[example/position_retargeting/hand_robot_viewer.py] -> [dex_retargeting, dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, hand_viewer, pathlib, pytransform3d, tqdm, typing]
[example/position_retargeting/hand_viewer.py] -> [dataset, mano_layer, pathlib, pytransform3d, sapien, sapien.asset, sapien.utils, tqdm, typing]
[example/position_retargeting/mano_layer.py] -> [manopth.manolayer, pathlib, torch.nn]
[example/position_retargeting/manopth/examples/manopth_demo.py] -> [manopth, manopth.demo, manopth.manolayer, matplotlib, mpl_toolkits.mplot3d, tqdm]
[example/position_retargeting/manopth/examples/manopth_mindemo.py] -> [manopth, manopth.manolayer]
[example/position_retargeting/manopth/mano/webuser/lbs.py] -> [mano.webuser.posemapper, this]
[example/position_retargeting/manopth/mano/webuser/posemapper.py] -> [this]
[example/position_retargeting/manopth/mano/webuser/serialization.py] -> [chumpy.ch, mano.webuser.posemapper, mano.webuser.verts, this]
[example/position_retargeting/manopth/mano/webuser/smpl_handpca_wrapper_HAND_only.py] -> [chumpy.ch, mano.webuser.posemapper, mano.webuser.verts, this]
[example/position_retargeting/manopth/mano/webuser/verts.py] -> [chumpy.ch, mano.webuser.posemapper, this]
[example/position_retargeting/manopth/mano_v1_2/__init__.py] -> [this]
[example/position_retargeting/manopth/mano_v1_2/webuser/__init__.py] -> [this]
[example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/MANO___hello_world.py] -> [the, this, webuser.smpl_handpca_wrapper_HAND_only]
[example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/MANO___render.py] -> [OpenCV, OpenDR, mpl_toolkits.mplot3d, opendr.camera, opendr.lighting, opendr.renderer, os, psbody.mesh, psbody.mesh.sphere, this, webuser.smpl_handpca_wrapper_HAND_only]
[example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/SMPL+H___hello_world.py] -> [the, this, webuser.smpl_handpca_wrapper]
[example/position_retargeting/manopth/mano_v1_2/webuser/hello_world/SMPL+H___render.py] -> [OpenCV, OpenDR, mpl_toolkits.mplot3d, opendr.camera, opendr.lighting, opendr.renderer, os, psbody.mesh, psbody.mesh.sphere, this, webuser.smpl_handpca_wrapper]
[example/position_retargeting/manopth/mano_v1_2/webuser/lbs.py] -> [posemapper, this]
[example/position_retargeting/manopth/mano_v1_2/webuser/posemapper.py] -> [this]
[example/position_retargeting/manopth/mano_v1_2/webuser/serialization.py] -> [chumpy.ch, posemapper, this, verts]
[example/position_retargeting/manopth/mano_v1_2/webuser/smpl_handpca_wrapper.py] -> [chumpy.ch, pickle, posemapper, this, verts]
[example/position_retargeting/manopth/mano_v1_2/webuser/smpl_handpca_wrapper_HAND_only.py] -> [chumpy.ch, this, webuser.posemapper, webuser.verts]
[example/position_retargeting/manopth/mano_v1_2/webuser/verts.py] -> [chumpy.ch, posemapper, this]
[example/position_retargeting/manopth/manopth/demo.py] -> [manopth.manolayer, matplotlib, mpl_toolkits.mplot3d, mpl_toolkits.mplot3d.art3d]
[example/position_retargeting/manopth/manopth/manolayer.py] -> [mano.webuser.smpl_handpca_wrapper_HAND_only, manopth, manopth.tensutils, torch.nn]
[example/position_retargeting/manopth/manopth/rodrigues_layer.py] -> [manopth, torch.autograd]
[example/position_retargeting/manopth/manopth/tensutils.py] -> [manopth]
[example/position_retargeting/manopth/setup.py] -> [setuptools]
[example/position_retargeting/manopth/test/test_demo.py] -> [manopth.demo]
[example/position_retargeting/mujoco_ycb_objects.py] -> [dataset, pathlib, pytransform3d]
[example/position_retargeting/pv_retargeting/r1_hand_tracking_video.py] -> [dataset, dex_retargeting, dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, hand_robot_viewer, pathlib, pytransform3d]
[example/position_retargeting/pv_retargeting/r1_single_arm_follow.py] -> [dataset, dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, galaxea_sim.controllers.utils.relaxed_ik_solver, mano_layer, pathlib, pytransform3d, sapien.asset, sapien.utils, tqdm]
[example/position_retargeting/render_hand_object.py] -> [dataset, dex_retargeting.constants, dex_retargeting.retargeting_config, hand_robot_viewer, hand_viewer, pathlib, typing]
[example/position_retargeting/test/r1_ik_position_test.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, hand_robot_viewer, pathlib, pytransform3d]
[example/position_retargeting/test/verify_correct_urdf.py] -> [dex_retargeting, galaxea_sim.controllers.utils.relaxed_ik_solver, pathlib, pytransform3d]
[example/position_retargeting/visualize_hand_object.py] -> [dataset, dex_retargeting.constants, dex_retargeting.retargeting_config, hand_robot_viewer, hand_viewer, pathlib, typing]
[example/profiling/generate_human_data_from_video.py] -> [dex_retargeting.constants, example.vector_retargeting.single_hand_detector, pathlib]
[example/profiling/profile_online_retargeting.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, pathlib, typing]
[example/simulation/r1_hand_tracking_sim.py] -> [dataset, dex_retargeting.constants, dex_retargeting.optimizer_utils, dex_retargeting.retargeting_config, galaxea_sim.controllers.utils.relaxed_ik_solver, mano_layer, pathlib, pytransform3d]
[example/simulation/r1_simulation.py] -> [galaxea_sim.controllers.utils.relaxed_ik_solver, pathlib]
[example/vector_retargeting/capture_webcam.py] -> [pathlib, typing]
[example/vector_retargeting/detect_from_video.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, pathlib, single_hand_detector]
[example/vector_retargeting/mujoco_retargeting_viewer.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, pathlib, typing]
[example/vector_retargeting/render_robot_hand.py] -> [dex_retargeting.retargeting_config, pathlib, sapien.asset, sapien.utils, typing]
[example/vector_retargeting/show_realtime_retargeting.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, loguru, pathlib, queue, sapien.asset, sapien.utils, single_hand_detector, typing]
[example/vector_retargeting/single_hand_detector.py] -> [mediapipe.framework.formats, mediapipe.python.solutions, mediapipe.python.solutions.drawing_utils, mediapipe.python.solutions.hands]
[example/video_egocentric_retargeting/arm_motion_analyzer.py] -> [dex_retargeting.constants, scipy.spatial.transform]
[example/video_egocentric_retargeting/contact_preservation.py] -> [dex_retargeting.optimizer_utils, pathlib]
[example/video_egocentric_retargeting/egocentric_hand_detector.py] -> [mediapipe.python.solutions, mediapipe.python.solutions.drawing_utils, mediapipe.python.solutions.hands, pathlib]
[example/video_egocentric_retargeting/egocentric_sapien_render.py] -> [dex_retargeting, dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, pathlib, pytransform3d, sapien, sapien.asset, scipy.spatial.transform, single_hand_detector]
[example/video_egocentric_retargeting/gripper_video_pipeline.py] -> [--grasp-threshold, arm_motion_analyzer, contact_preservation, dex_retargeting, dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, egocentric_hand_detector, pathlib, pytransform3d, robot_hand_retargeter, sapien.asset, scipy.spatial.transform, typing]
[example/video_egocentric_retargeting/render_trajectory.py] -> [dex_retargeting.retargeting_config, loguru, pathlib, sapien.asset, sapien.utils, typing]
[example/video_egocentric_retargeting/robot_hand_retargeter.py] -> [dex_retargeting.constants, dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, pathlib]
[example/video_egocentric_retargeting/run_pipeline.py] -> [--grasp-threshold, arm_motion_analyzer, contact_preservation, dex_retargeting.constants, egocentric_hand_detector, pathlib, render_trajectory, robot_hand_retargeter, scipy.spatial.transform, typing]
[example/video_egocentric_retargeting/video_to_trajectory.py] -> [arm_motion_analyzer, contact_preservation, dex_retargeting.constants, egocentric_hand_detector, pathlib, robot_hand_retargeter, scipy.spatial.transform, typing]
[position] -> [inherits] -> [find]
[result_meta] -> [inherits] -> [object]
[src/dex_retargeting/__init__.py] -> [importlib.util, the]
[src/dex_retargeting/constants.py] -> [pathlib, typing]
[src/dex_retargeting/kinematics_adaptor.py] -> [abc, dex_retargeting.robot_wrapper, typing]
[src/dex_retargeting/optimizer.py] -> [abc, dex_retargeting.kinematics_adaptor, dex_retargeting.robot_wrapper, non-linear, the, typing]
[src/dex_retargeting/retargeting_config.py] -> [dataclasses, dex_retargeting, dex_retargeting.kinematics_adaptor, dex_retargeting.optimizer, dex_retargeting.optimizer_utils, dex_retargeting.robot_wrapper, dex_retargeting.seq_retarget, dex_retargeting.yourdfpy, pathlib, typing]
[src/dex_retargeting/robot_wrapper.py] -> [typing]
[src/dex_retargeting/seq_retarget.py] -> [analytical, dex_retargeting.constants, dex_retargeting.optimizer, dex_retargeting.optimizer_utils, pytransform3d, typing]
[src/dex_retargeting/yourdfpy.py] -> [anytree, dataclasses, functools, lxml, pytransform3d, the, typing]
[tests/test_optimizer.py] -> [dex_retargeting.constants, dex_retargeting.optimizer, dex_retargeting.retargeting_config, dex_retargeting.robot_wrapper, pathlib, time]
[tests/test_retargeting_config.py] -> [dex_retargeting.retargeting_config, dex_retargeting.seq_retarget, pathlib]
# baseline_manifest — R00 冻结成功基线与诊断现场

生成时间: 2026-09-28 10:18:33 · **只读** (未重跑 CEM, 未改 Stage0/质量/摩擦/驱动力) ✓

## 1. 代码与输入

- git HEAD: `92c2f6e` (2026-09-18) CoACD 线: 缓存key修复(压平/分解参数) + 闭合段16帧 + 闭合下限 + 位姿CoACD代价探针
- 工作树未提交文件数: **121** (现场含未提交修改 ⇒ 版本以文件 sha1 为准)
- 场景 glb: `/home/an/data/ras/7_vggt_omega/final_scene.glb` sha1=`367cfc16143e81f0`
- HaWoR: `/home/an/data/hawor/7a` (存在=True)

## 2. Stage1 证书范围 (不扩大 ✓)

- 证书仅覆盖: **当前物体 (glb_4 / toy_0) + 当前初始位姿 + 当前判据 (旧实际判据)** 的 3/3 复现 ✓; **不扩大**为"任意轨迹/任意物体都能抓" ✗
- 通过 episode: ['s1ep-000128', 's1ep-000129', 's1ep-000130'] (3/3)

## 3. 两个现场

### s31 — 成功基线 (Stage1 3/3)

- recipe sha1=`e41daf81647fcd3c` · target=`glb_4` · q源=`actual_executed` · task_grasp_frame=`50` · export_valid=`True`
- 数组长度: {'T_W_G_cmd': 350, 'actual_base_commands': 350, 'actual_control_times': 350, 'actual_q_commands': 350, 'close_ctrl_q': 350, 'close_ctrl_t': 16, 'observations': None, 'optimized_T_W_G': 4, 'phase': 350, 'q_cmd': 350}
- 相位: boundaries={'close': [[0, 130]], 'lift': [[140, 350]], 'n': 350, 'settle': [[130, 140]]} counts={'close': 130, 'settle': 15, 'travel': 0}
- hashes: {'physics_hash': '9ad57488f9fd56c83971b2f3', 'controller_hash': '89c4aff1e7380163faff223f', 'criteria_hash': '783e2d6e4d6c9a89a7226669', 'canonical_state_hash': '49d39cb1610b5313f2275cda', 'initial_state_hash': '35e7e08b090941804b33866a'}
- 终验: n_pass=3/3 · 通过=['s1ep-000128', 's1ep-000129', 's1ep-000130'] · **selected_episode=s1ep-000124** ⚠ 与通过列表不一致 (R03 靶子)
- 判据: {'drift_mm': 20.0, 'gnd_mm': 0.0, 'lift_cm': 3.0, 'tg_min': 0.5}
- best_grasp sha1=`2d0fcf7e06cadae2` size=753008 keys=['R', 'close_controls', 'enclose_z', 'euler', 'force_limit', 'gripper_qpos', 'locked_gripper_qpos', 'obj_lift', 'obj_xy_drift', 'peak_grip_force', 'pos', 'q_open', 'ref_base_R', 'reward', 'scene_digest_hash', 'settle_controls', 'stage1_recipe_json', 'strategy', 'target_object', 'verification']
- stage2 trace 存在=False · stage3 视频=['cam_view_stage3_gripper_only_left.mp4', 'god_view_stage3_gripper_only_left.mp4', 'right_view_stage3_gripper_only_left.mp4']

### s44 — 失败现场 (Stage2 CARRY 0/3 ⇒ Stage3 未运行)

- recipe sha1=`e41daf81647fcd3c` · target=`glb_4` · q源=`actual_executed` · task_grasp_frame=`50` · export_valid=`True`
- 数组长度: {'T_W_G_cmd': 350, 'actual_base_commands': 350, 'actual_control_times': 350, 'actual_q_commands': 350, 'close_ctrl_q': 350, 'close_ctrl_t': 16, 'observations': None, 'optimized_T_W_G': 4, 'phase': 350, 'q_cmd': 350}
- 相位: boundaries={'close': [[0, 130]], 'lift': [[140, 350]], 'n': 350, 'settle': [[130, 140]]} counts={'close': 130, 'settle': 15, 'travel': 0}
- hashes: {'physics_hash': '9ad57488f9fd56c83971b2f3', 'controller_hash': '89c4aff1e7380163faff223f', 'criteria_hash': '783e2d6e4d6c9a89a7226669', 'canonical_state_hash': '49d39cb1610b5313f2275cda', 'initial_state_hash': '35e7e08b090941804b33866a'}
- 终验: n_pass=3/3 · 通过=['s1ep-000128', 's1ep-000129', 's1ep-000130'] · **selected_episode=s1ep-000124** ⚠ 与通过列表不一致 (R03 靶子)
- 判据: {'drift_mm': 20.0, 'gnd_mm': 0.0, 'lift_cm': 3.0, 'tg_min': 0.5}
- best_grasp sha1=`2d0fcf7e06cadae2` size=753008 keys=['R', 'close_controls', 'enclose_z', 'euler', 'force_limit', 'gripper_qpos', 'locked_gripper_qpos', 'obj_lift', 'obj_xy_drift', 'peak_grip_force', 'pos', 'q_open', 'ref_base_R', 'reward', 'scene_digest_hash', 'settle_controls', 'stage1_recipe_json', 'strategy', 'target_object', 'verification']
- stage2 trace 存在=True · stage3 视频=['cam_view_stage3_gripper_only_left.mp4', 'god_view_stage3_gripper_only_left.mp4', 'right_view_stage3_gripper_only_left.mp4']

## 4. s44 Stage2 失败证据 (原始行)

- `2026-09-27 22:53:42,632 [grasp_hawor] INFO: [Stage2 gate] FAIL(CARRY) bundle=2a3962ea4b9a controls=192 tracking=0.41/2.68mm`
- `2026-09-27 22:53:48,275 [grasp_hawor] INFO: [Stage2 gate] FAIL(CARRY) bundle=2a3962ea4b9a controls=192 tracking=0.41/2.68mm`
- `2026-09-27 22:53:53,964 [grasp_hawor] INFO: [Stage2 gate] FAIL(CARRY) bundle=2a3962ea4b9a controls=192 tracking=0.41/2.68mm`
- `2026-09-27 22:53:53,964 [grasp_hawor] ERROR: [test-stage3] Stage2 repeat gate FAIL: 0/3 failures=['CARRY', 'CARRY', 'CARRY']; Stage3 不运行`

## 5. F65 来源差异

- 扫描窗口: F35–F65 (31 帧, 见 s31 日志 [Stage1][多帧扫描])
- 任务抓取帧: 50
- 任务抓取帧 = recipe.task_grasp_frame (s31: 50)
- close 相位 = phase_boundaries.close (s31: [0,130])
- ⇒ F65 是**扫描窗口右端**, 不是任务终点; 两者含义不同, 不可互相替代 ✓

## 6. 未完成清单 (按 md §H.1 顺序)

- [ ] R01 时间线截断/数组完整性 (P0, 零物理) —— 禁止 q-only 尾裁剪删 CARRY
- [ ] R02 闭合→稳定→搬运命令真实复用 (P0, 先离线)
- [ ] R03 成功 record 与入口门绑定 (从真实通过 run 恢复, 不伪造)
- [ ] R04 场景/初态真实核对 (API 自检先行, 不再持续放行 UNKNOWN)
- [ ] R05 用完整矩阵证据定位 33.37° 与起步失稳 (P1, 最多一次诊断)
- [ ] R06 完整搬运 + 反馈鲁棒性 + 重放
- [ ] R07 Stage1 最小自洽性修复 (不重找抓取姿态)
- [ ] R08 Stage3 入口/后缀/最终重放自洽

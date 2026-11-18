#!/bin/bash
# =============================================================================
# ef3r_dataset.sh — 视频 → 机器人数据集 一键管线 (EF3R 快路径, 分钟级/条)
#
#   用法: ef3r_dataset.sh <video.mp4> <name> [side] [gpu]
#   例:   ef3r_dataset.sh /path/to/workspace/456483158.mp4 pickplace_456483158 right 0
#
# 阶段 (全部断点续跑: 已有产物自动跳过):
#   1. EgoFound3R 推理        ~30s   (手/物体接触/深度/相机, vggt-omega)
#   2. export_tracks          ~3s    (关节轨迹/相机 npy)
#   3. state_labels           ~10s   (PICK/LIFT/HOLD/PLACE/RELEASED 分段)
#   4. fewframe 校准          ~2s    (接触概率选抓取帧 + 人手握宽 + 物体在手偏移)
#   5. ef3r_to_targets        ~5s    (SVD 重定向 + gravity_align 重力对齐 → sim targets)
#   6. R1 物理抓取仿真        ~35s   (phys-only; 加 --render 出 GIF ~+2min)
#
# 实测总计 ~1.4 分钟/条 (phys-only), 对比 AGILE 链 ~1.5-2h/条。
# 物体模型/背景/任务分段全部来自 EgoFound3R 自身输出, 不需 AGILE 预处理。
# =============================================================================
set -e -o pipefail
VIDEO=${1:?usage: ef3r_dataset.sh <video.mp4> <name> [side] [gpu]}
NAME=${2:?usage: ef3r_dataset.sh <video.mp4> <name> [side] [gpu]}
SIDE=${3:-right}
GPU=${4:-0}
# 全阶段统一 GPU (EF3R 推理/SAM3/SAM3D/仿真) —— 此前只有仿真阶段设置,
# 推理继承默认卡, 别人占卡时 OOM (实测 GPU0 被占 36G 全链失败)
export CUDA_VISIBLE_DEVICES=$GPU

E3R=/path/to/project_data/mano_ws/EgoFound3R
R2S=/path/to/Real2SimGripper
PYV=/path/to/user_home/miniconda3/envs/vggt-omega/bin/python
PYR=/path/to/user_home/miniconda3/envs/robotwin_curobo/bin/python
E=$E3R/outputs/$NAME
export E SIDE NAME

stage() { echo -e "\n===== [$NAME] $1 ====="; }

# 1) EF3R 推理
if [ ! -f "$E/inference_output.pt" ]; then
  stage "1/6 EgoFound3R 推理"
  cd $E3R && $PYV infer_marker_video.py --video-path "$VIDEO" \
    --checkpoint checkpoints/EgoFound3R.pt --output-dir outputs/$NAME \
    --device cuda --no-overlay-frame-images --allow-output-overwrite
else
  stage "1/6 EgoFound3R 推理 (已有产物, 跳过)"
fi

# 1b) 手侧自适应: 前摄/镜像存储的视频会被 EF3R 判成另一只手 (实测 7.mp4: 右手
# 全部检出为左手 113/113)。右手无效而左手有效 → 水平翻转视频后整链重跑
# (无文字内容翻转无损语义; 翻转件落 <视频目录>/<名>_flipped.mp4)。
_vr=$($PYV - <<PYEOF
import numpy as np
try:
    v = np.load("$E/hands/valid.npy")
    print(int(v[:, 1].sum()), int(v[:, 0].sum()))
except Exception:
    print("-1 -1")
PYEOF
)
_r=${_vr%% *}; _l=${_vr##* }
if [ "$_r" = "0" ] && [ "$_l" -gt 0 ] 2>/dev/null && [[ "$VIDEO" != *_flipped.mp4 ]]; then
  stage "1b/6 检测到镜像视频 (右手 0 帧/左手 $_l 帧) → 翻转重跑"
  FVIDEO="$(dirname "$VIDEO")/${NAME}_flipped.mp4"
  $PYR - <<PYEOF
import imageio.v2 as imageio
r = imageio.get_reader("$VIDEO"); w = imageio.get_writer("$FVIDEO", fps=30, quality=8)
for img in r: w.append_data(img[:, ::-1])
w.close()
PYEOF
  rm -rf "$E"
  exec "$0" "$FVIDEO" "$NAME" "$SIDE" "$GPU" "$@"
fi

# 2-4) EF3R 工具层
stage "2/6 export_tracks"
cd $E3R && $PYV tools/export_tracks.py --run $E
stage "3/6 state_labels"
[ -f "$E/state_labels.json" ] || $PYV tools/render_action_states.py --run $E \
  --video "$VIDEO" --side $SIDE --no-mesh
stage "4/6 fewframe 校准 + 物体实例"
$PYV $R2S/scripts/ef3r_fewframe.py --ef3r-dir $E
# 物体契约 (PICK 段手质心 + 深度反投影 + 凸包): 仿真阶段必需, 此前漏在管线外
$PYV $R2S/scripts/ef3r_object.py --ef3r-dir $E --hand $SIDE

# 4b) SAM3D 物体精模 (用户标准链 "先 SAM3D 建模被抓物体"): SAM3 文本/点提示
# 分割 → SAM3D 带纹理 mesh → PCA 对齐回 EF3R 世界系。OBJ_TEXT="pink duck" 启用;
# 失败自动回退凸包版 (仿真不中断)。
stage "4b/6 SAM3D 物体精模"
if [ -n "$OBJ_TEXT" ]; then
  $PYV $R2S/scripts/ef3r_sam3d.py --ef3r-dir $E --video "$VIDEO" --gpu $GPU \
    --text "$OBJ_TEXT" || echo "  ⚠ SAM3D 失败, 保留凸包版"
else
  echo "  跳过 (OBJ_TEXT='<物体描述>' 启用, 如 OBJ_TEXT='pink rubber duck')"
fi

# 5) targets (SVD 重定向 + 重力对齐)
stage "5/6 ef3r_to_targets (SVD 重定向 + gravity_align)"
cd $R2S && $PYR scripts/ef3r_to_targets.py --ef3r-dir $E --side $SIDE

# 6) R1 物理抓取仿真
stage "6/6 R1 物理抓取仿真"
cd $R2S/scripts && CUDA_VISIBLE_DEVICES=$GPU $PYR - <<PYEOF
import sys, os; sys.path.insert(0, '.')
import cb_large_range_sim as cb
E = '$E'
_sc = E+'/pointcloud/scene_cloud.npz'
_obj = E+'/object_instance_sam3d.npz'
_obj = _obj if os.path.exists(_obj) else E+'/object_instance.npz'
cb.main(targets=E+'/ef3r_${SIDE}_targets.npz', name='$NAME',
        object_npz=_obj,
        scene_cloud=(_sc if os.path.exists(_sc) else None),
        object_physics=True, w_ori=1.0, torso_wbc=True, arm_beam=4,
        phys_only=('-norender' in sys.argv))
PYEOF

stage "DONE"
echo "数据集产物: $E/ef3r_${SIDE}_targets.npz (机器人轨迹) + $E/object_instance.npz (物体)"
echo "           + $E/pick_place_calib.npz (抓取校准) + results/large_range/${NAME}_phys_large_range.json (验收)"
[[ " $* " == *" -norender "* ]] || echo "GIF: demo_output/large_range/${NAME}_phys_large_range.gif (重跑 run_ef3r_phys.py 不带 --phys-only)"

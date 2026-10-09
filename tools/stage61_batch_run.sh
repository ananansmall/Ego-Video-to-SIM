#!/usr/bin/env bash
# 逐场景独立进程跑 stage6.1 物理仿真。
#
# 为什么必须独立进程: 在同一进程里循环创建/销毁 sapien.Scene 会耗尽 EGL/Vulkan
# 上下文, 表现为 `elf_dynamic_array_reader.h: tag not found` 刷屏后 `code=11` 崩溃。
# 这不是显存不足 (实测 nvidia-smi 显示 GPU0/1 各有 38~47GB 空闲), 而是渲染上下文
# 在反复创建销毁后失效。所以每个场景必须开新进程。
#
# 用法: bash tools/stage61_batch_run.sh [n_frames] [scene ...]
#   PYRENDER=1  方案 B: 绕过 SAPIEN 渲染器, SAPIEN 只跑物理 + pyrender 离线渲染。
#               SAPIEN 渲染确定性失效的场景 (7_vggt_omega / phone_vggt_omega /
#               egodex_0_rescaled) 必须用这个。

set -u
cd "$(dirname "$0")/.."
PY=/mnt/data/lza/conda_envs/sapien/bin/python
RENDER_ARGS=""
if [ "${PYRENDER:-0}" = "1" ]; then RENDER_ARGS="--pyrender_render"; fi
if [ -n "${PYRENDER:-}" ] && [ "$PYRENDER" != "0" ] && [ "$PYRENDER" != "1" ]; then
  RENDER_ARGS="--pyrender_render --render_mesh $PYRENDER"
fi
NFRAMES="${1:-100}"
shift || true
if [ $# -gt 0 ]; then SCENES="$@"; else
  SCENES="2chair-bottle_rescaled 3chair-bottle_rescaled phone_vggt_omega \
          082_C5_CellPhone_rescaled 121_C5_CellPhone_161deg_vggt_omega \
          151_C7_Plate_214deg_vggt_omega 7_vggt_omega egodex_0_rescaled"
fi

for S in $SCENES; do
  D="output_v2/$S"
  if [ ! -f "$D/urdf/scene.urdf" ]; then
    echo "[SKIP] $S — 无 urdf/scene.urdf"
    continue
  fi
  echo "[RUN ] $S (n_frames=$NFRAMES) ${RENDER_ARGS}"
  timeout 1800 "$PY" tools/stage61_physics_sim.py \
      --output_dir "$D" --n_frames "$NFRAMES" $RENDER_ARGS 2>&1 \
    | grep -viE "deprecat|elf_dynamic_array_reader|renderer = sapien" \
    | grep -E "个物体|质量范围|惯量为|初始穿透|装入|visual:|距离|仿真结束|渲染自检|ERROR|WARN|GIF:"
  RC=${PIPESTATUS[0]}
  if [ "$RC" -ne 0 ]; then
    echo "[FAIL] $S 退出码 $RC"
  fi
  # 进程退出后 GPU 上下文才彻底释放, 留一点间隔
  sleep 4
done
echo "[DONE]"
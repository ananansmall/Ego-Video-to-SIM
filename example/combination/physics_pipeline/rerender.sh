#!/bin/bash
# 物理仿真渲染脚本 — 让02_render_scene.py具有物理仿真属性
# 核心改动: 重力补偿(compute_passive_force+set_qf) + dynamic物体 + 地面支撑
# 需要在有GPU的终端直接运行
#
# 调用方式:
#   bash rerender.sh [render|demo|smooth] [smooth值] [数据目录名]
#
# 模式:
#   render  - 单趟渲染 (smooth=1默认) (默认模式)
#   demo    - 交互式3D查看器 (需要显示器)
#   smooth  - 两趟渲染 (smooth=2, 后处理双向滤波)
#
# smooth值: 0=不平滑, 1=在线EMA(默认), 2=后处理双向滤波
#
# 数据目录名: output/下的子目录名, 默认 7_my_7mp4_result
#   例如: bash rerender.sh render 1 7_my_7mp4_result
#         bash rerender.sh render 1 another_scene

cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

MODE=${1:-render}
SMOOTH=${2:-1}
SESSION=${3:-7_my_7mp4_result}

OUTPUT_DIR="./output/${SESSION}"
mkdir -p "${OUTPUT_DIR}/videos"

# 检查对齐参数是否存在
ALIGN_PARAMS="${OUTPUT_DIR}/alignment/transform_params.npz"
if [ ! -f "$ALIGN_PARAMS" ]; then
    echo "⚠ 未找到对齐参数: $ALIGN_PARAMS"
    echo "  请先运行: python run_physics_pipeline.py --hawor-dir ... --ras-dir ..."
    echo "  或指定正确的数据目录名"
    exit 1
fi

COMMON_ARGS="--mode physics_tracking \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --transform-params ${ALIGN_PARAMS} \
  --fps 30 \
  --fast-collision \
  --smooth $SMOOTH"

if [ "$MODE" = "demo" ]; then
    echo "=== 交互式Demo模式 (实时3D查看器, smooth=$SMOOTH) ==="
    echo "  数据目录: $OUTPUT_DIR"
    /home/an/miniconda3/envs/dex/bin/python 04_physics_simulation.py \
      $COMMON_ARGS \
      --output "${OUTPUT_DIR}/videos/physics_sim_demo.mp4" \
      --viewer \
      2>&1 | tee "${OUTPUT_DIR}/physics_pipeline.log"
elif [ "$MODE" = "smooth" ]; then
    echo "=== 后处理平滑模式 (smooth=2, 两趟渲染) ==="
    echo "  数据目录: $OUTPUT_DIR"
    /home/an/miniconda3/envs/dex/bin/python 04_physics_simulation.py \
      $COMMON_ARGS \
      --output "${OUTPUT_DIR}/videos/physics_sim_smooth.mp4" \
      --smooth 2 \
      --crf 18 \
      2>&1 | tee "${OUTPUT_DIR}/physics_pipeline.log"
else
    echo "=== 单趟渲染模式 (smooth=$SMOOTH) ==="
    echo "  数据目录: $OUTPUT_DIR"
    echo "  smooth=0: 不平滑 | smooth=1: 在线EMA(默认) | smooth=2: 后处理双向滤波"
    /home/an/miniconda3/envs/dex/bin/python 04_physics_simulation.py \
      $COMMON_ARGS \
      --output "${OUTPUT_DIR}/videos/physics_sim_render.mp4" \
      --crf 18 \
      2>&1 | tee "${OUTPUT_DIR}/physics_pipeline.log"
fi

echo ""
echo "完成!"
echo "  输出: ${OUTPUT_DIR}/videos/"
echo "  日志: ${OUTPUT_DIR}/physics_pipeline.log"
echo ""
echo "用法: bash rerender.sh [render|demo|smooth] [smooth值] [数据目录名]"
echo "  render  - 单趟渲染 (smooth=1默认) (默认模式)"
echo "  demo    - 交互式3D查看器"
echo "  smooth  - 两趟渲染 (smooth=2, 后处理双向滤波)"
echo ""
echo "  数据目录名: output/下的子目录名, 默认 7_my_7mp4_result"
echo "  例: bash rerender.sh render 1 7_my_7mp4_result"
echo "      bash rerender.sh demo 1 another_scene"

#!/usr/bin/env bash
# ep4 RAS resume: rebuilt interpreter (embodiedgen + shims) + verified ffmpeg path.
set -uo pipefail
BASE=$(cd -- "$(dirname -- "$0")" && pwd)
RAS=$BASE/ReplicateAnyScene
MP=/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules
export PYTHONPATH="$BASE/ReplicateAnyScene/pyshims:$MP/sam3:$MP/sam-3d-objects:/path/to/project_data/mano_ws/vggt-omega"
export PATH="$(dirname "/path/to/Real2SimGripper/examples/egoexplore/ep5/ras-mainv3-gravity-luna-xhigh.20261116/bin/ffmpeg"):$PATH"
export PATH="$BASE/bin:$PATH"
export IMAGEIO_FFMPEG_EXE="/path/to/Real2SimGripper/examples/egoexplore/ep5/ras-mainv3-gravity-luna-xhigh.20261116/bin/ffmpeg"
export LIDRA_SKIP_INIT=1
export CONDA_PREFIX=/path/to/user_home/miniconda3/envs/embodiedgen
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/path/to/user_home/miniconda3/envs/embodiedgen
export CPATH="/path/to/user_home/miniconda3/envs/embodiedgen/include:/path/to/user_home/miniconda3/envs/embodiedgen/targets/x86_64-linux/include:/path/to/user_home/miniconda3/envs/embodiedgen/lib/python3.10/site-packages/nvidia/cuda_runtime/include:/path/to/user_home/miniconda3/envs/embodiedgen/lib/python3.10/site-packages/nvidia/cuda_nvcc/include:/path/to/user_home/miniconda3/envs/embodiedgen/lib/python3.10/site-packages/nvidia/nvtx/include:"
export LIBRARY_PATH="/path/to/user_home/miniconda3/envs/embodiedgen/lib:/path/to/user_home/miniconda3/envs/embodiedgen/targets/x86_64-linux/lib:/path/to/user_home/miniconda3/envs/embodiedgen/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -x "/path/to/Real2SimGripper/examples/egoexplore/ep5/ras-mainv3-gravity-luna-xhigh.20261116/bin/ffmpeg"
command -v ffmpeg
cd "$RAS"
/path/to/user_home/miniconda3/envs/embodiedgen/bin/python -u mainv3.py \
  --input_video /path/to/project_data/dataset/EgoExplorer/dataset/lerobot/videos/observation.images.camera_0/chunk-000/file-003.mp4 \
  --output_path $BASE/ras \
  --vggt_model vggt_omega \
  --max_frames 41 \
  --vlm_checkpoint /path/to/workspace/models/Qwen3.8-27B

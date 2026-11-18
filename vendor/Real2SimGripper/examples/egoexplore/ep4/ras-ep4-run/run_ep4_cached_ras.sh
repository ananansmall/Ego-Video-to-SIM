#!/usr/bin/env bash
# ep4 RAS: exact replica of the proven full-scene-dev.tYpPDq/launch_cached_ras.sh
# recipe, pointed at ep4's own video/frozen map (docs/15 sec 6 entry).
set -euo pipefail
repo=/path/to/Real2SimGripper
helper="$repo/examples/456483158-mobile-grasp/scene_agile/full-scene-dev.tYpPDq"
source_repo="/path/to/user_home/ /ReplicateAnyScene "
out="$repo/examples/egoexplore/ep4/ras-ep4-run/ras-output"
map="$repo/examples/egoexplore/ep4/ras-ep4.1BaO9z/ras_map/ras_raw.npz"
video=/path/to/project_data/dataset/EgoExplorer/dataset/lerobot/videos/observation.images.camera_0/chunk-000/file-003.mp4
vlm=/path/to/workspace/models/Qwen3.8-27B
mroot=/path/to/project_data/mano_ws/do-as-i-do/reconstruction
root=$(mktemp -d /tmp/ras-ep4-cached.XXXXXX)
cp -a "$source_repo/." "$root/"
cd "$root"
ln -s "$mroot/modules" sam3
ln -s "$mroot/modules/sam-3d-objects" sam-3d-objects
mkdir -p models/SAM3 models/SAM3D
ln -s "$mroot/weights/sam3/sam3.pt" models/SAM3/sam3.pt
ln -s "$mroot/weights/sam3d_shared/hf/checkpoints" models/SAM3D/checkpoints
cp "$repo/examples/egoexplore/ep4/ras-ep4-run/helpers/cached_map.py" "$repo/examples/egoexplore/ep4/ras-ep4-run/helpers/run_cached_ras.py" .
python3 "$repo/examples/egoexplore/ep4/ras-ep4-run/helpers/patch_ras_overlay.py" "$root"
export RAS_FROZEN_MAP="$map" RAS_VLM_DIR="$vlm" PYTHONUNBUFFERED=1
export PYTHONPATH=.:/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules/sam3:/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules/sam-3d-objects:"$helper/ras_extra"
export CONDA_PREFIX=/path/to/user_home/miniconda3/envs/sam3d CUDA_HOME=/path/to/user_home/miniconda3/envs/sam3d
export PATH="/path/to/user_home/miniconda3/envs/embodiedgen/bin:$HOME/miniconda3/envs/sam3d/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export IMAGEIO_FFMPEG_EXE="$repo/examples/egoexplore/ep5/ras-mainv3-gravity-luna-xhigh.20261116/bin/ffmpeg"
export PATH="$(dirname "$IMAGEIO_FFMPEG_EXE"):$PATH"
test -s "$video"; test -s "$map"; test -x "$(command -v ffmpeg)"
"$HOME/miniconda3/envs/sam3d/bin/python" run_cached_ras.py "$video" "$out"

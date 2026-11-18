#!/usr/bin/env bash
# T3 chairbottle: R1 chain entry.  Usage:
#   bash run_chairbottle.sh /absolute/output /absolute/mainv3-output
set -euo pipefail
work="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$work/../.." && pwd)"
replay="$repo/examples/456483158-mobile-grasp/scene_agile"
out="${1:?Usage: run_chairbottle.sh /absolute/output /absolute/ras-output}"
ras="${2:?Pass the mainv3 output for this clip}"
ef3r=/path/to/project_data/mano_ws/EgoFound3R/outputs/chairbottle
agile=/path/to/project_data/mano_ws/AGILE/output/chairbottle_v2
video=/path/to/project_data/mano_ws/HaWoR/example/chairbottle.mp4
python="${EG_PYTHON:-$HOME/miniconda3/envs/embodiedgen/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$out"
"$python" "$work/prepare_chairbottle.py" "$ras" "$video" "$ef3r" "$agile/agile_right_targets_592.npz" \
  "$out" "${@:3}" | tee "$out/preparation.log"
"$python" "$replay/replay_paired_scene.py" "$out/refined" "$out/refined_replay" --plan "$out/refined_ik/ik_plan.npz" \
  | tee "$out/physics.log"
"$python" "$replay/replay_paired_scene.py" "$out/refined" "$out/refined_replay" \
  --render-trace "$out/refined_replay/exec_trace.npz"
"$python" "$repo/scripts/render_source_and_sim.py" --source "$video" --source-start-frame 0 --source-stop-frame 592 \
  --sim "$out/refined_replay/four_views.mp4" --out "$out/refined_replay/source_and_sim.mp4" \
  --panels "Front | PhysX" "Side | PhysX" "R1 wrist | PhysX" "Overhead | PhysX"
"$python" "$replay/review_full_scene.py" "$out"

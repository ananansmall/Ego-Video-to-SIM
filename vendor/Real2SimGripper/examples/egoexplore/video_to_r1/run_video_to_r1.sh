#!/usr/bin/env bash
# Run the data-independent stages on one original video. The final stage needs
# the observed contact windows and the calibrated retarget pair for that run.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo 'Usage: run_video_to_r1.sh /path/to/source.mp4 /path/to/run' >&2
    exit 2
fi
video=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
: "${RAS_ROOT:?Set RAS_ROOT to the configured ReplicateAnyScene checkout}"
: "${RAS_PY:?Set RAS_PY to the RAS interpreter}"
: "${EF3R_ROOT:?Set EF3R_ROOT to the EgoFound3R checkout}"
: "${EF3R_PY:?Set EF3R_PY to the EgoFound3R interpreter}"
: "${R1_PY:?Set R1_PY to the R1 interpreter}"
: "${RAS_CHECKPOINT:?Set RAS_CHECKPOINT to the VGGT-Omega checkpoint}"
: "${RAS_VLM:?Set RAS_VLM to the existing vision-language checkpoint}"
: "${RAS_MAP_SCRIPT:?Set RAS_MAP_SCRIPT to the video map extractor}"
ras_root=$(realpath "$RAS_ROOT")
ef3r_root=$(realpath "$EF3R_ROOT")
current_hash=$(sha256sum "$video")
if [ -f "$out/source.sha256" ] && ! grep -Fqx -- "$current_hash" "$out/source.sha256"; then
    echo 'The run directory already belongs to another source video' >&2
    exit 2
fi
printf '%s\n' "$current_hash" > "$out/source.sha256"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export RAS_FROZEN_MAP="$out/ras_map/ras_raw.npz"

if [ ! -s "$out/ef3r/inference_output.pt" ]; then
    (cd "$ef3r_root"; CUDA_VISIBLE_DEVICES="${EF3R_GPU:-0}" "$EF3R_PY" infer_marker_video.py \
        --video-path "$video" --output-dir "$out/ef3r" --device cuda \
        --checkpoint "$ef3r_root/checkpoints/EgoFound3R.pt") 2>&1 | tee "$out/ef3r.log"
fi
if [ ! -s "$RAS_FROZEN_MAP" ]; then
    (cd "$ras_root"; PYTHONPATH="$ras_root${RAS_PYTHONPATH:+:$RAS_PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="${RAS_GPU:-0}" "$RAS_PY" "$RAS_MAP_SCRIPT" \
        --video "$video" --checkpoint "$RAS_CHECKPOINT" --output "$out/ras_map" \
        --max-frames "${RAS_MAX_FRAMES:-50}") 2>&1 | tee "$out/ras_map.log"
fi
if [ ! -s "$out/ras/final_scene.glb" ]; then
    command -v ffmpeg >/dev/null
    (cd "$ras_root"; PYTHONPATH="$ras_root${RAS_PYTHONPATH:+:$RAS_PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="${RAS_GPU:-0}" "$RAS_PY" mainv3.py \
        --input_video "$video" --output_path "$out/ras" --vggt_model vggt_omega \
        --max_frames "${RAS_MAX_FRAMES:-50}" --vlm_checkpoint "$RAS_VLM") 2>&1 | tee "$out/ras.log"
    test -s "$out/ras/final_scene.glb"
fi
"$R1_PY" "$here/register_video_observations.py" --video "$video" \
    --ef3r "$out/ef3r/inference_output.pt" --ras-map "$RAS_FROZEN_MAP" \
    --ras-alignment "$out/ras/coordinate_alignment.json" --out "$out/registration"

# Stage 0: freeze the original retarget, stance reference and object observations
# from the sample config, then render the unified replay.
config="${SAMPLE_CONFIG:-$out/sample_config.json}"
if [ ! -s "$config" ]; then
    echo "Waiting for the sample input configuration at $config (frames, object, supports, stance, grasp)" >&2
    exit 3
fi
if [ ! -s "$out/stage0/stage0_trace.npz" ]; then
    "$R1_PY" "$repo/scripts/stage0_trace.py" --config "$config" --out "$out/stage0"
fi
if [ ! -s "$out/stage0/render/stage0_replay.mp4" ]; then
    "$R1_PY" "$repo/scripts/stage0_render.py" --trace "$out/stage0/stage0_trace.npz" \
        --video "$video" --config "$config" --out "$out/stage0/render"
fi

if [ ! -s "$out/agile_windows.json" ] || [ ! -s "$out/retarget/paired_targets.npz" ]; then
    echo 'Waiting for registered object tracks, a retarget pair, and native AGILE contact-window results' >&2
    exit 3
fi
if [ ! -s "$out/pair/paired_targets.npz" ]; then
    "$R1_PY" "$here/merge_agile_contact_windows.py" "$out/retarget" "$out/agile_windows.json" "$out/pair"
fi
: "${R1_RUNNER:?Set R1_RUNNER to the calibrated continuous R1 entry point}"
"$R1_PY" "$R1_RUNNER" "$out/pair" "$out/physics" ${R1_RUNNER_ARGS:-}
if [ ! -s "$out/physics/four_views.mp4" ]; then
    echo 'R1_RUNNER produced no four_views.mp4; add R1_RUNNER_ARGS=--render (or run its render mode)' >&2
    exit 4
fi
"$R1_PY" "$repo/scripts/render_source_and_sim.py" --source "$video" \
    --sim "$out/physics/four_views.mp4" --out "$out/physics/source_and_sim.mp4"

# Unified report across Stage 0, the optimized pair, the IK plan and physics.
"$R1_PY" "$repo/scripts/unified_trajectory_report.py" --stage0-trace "$out/stage0/stage0_trace.npz" \
    --stage0-csv "$out/stage0/render/stage0_frame_errors.csv" \
    --pair "$out/pair/paired_targets.npz" --ik "$out/physics/ik_plan.npz" \
    --physics "$out/physics" --out "$out/report"

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_sweep.sh —— 参数扫描 runner (2026-09-20)
#
#  以 run_config.sh 为**基线** (唯一参数来源 ✓), 只在命令行覆盖**实验变量** ✓
#  ⇒ 满足 run_config.sh 的原则① "一个测试会话只改一个变量" ✓
#
#  用法:
#    ./run_sweep.sh <输出目录> VAR=val [VAR2=val2 ...]
#  例:
#    ./run_sweep.sh output/e0_base  _NO_GRIPPER_GROUND=0
#    ./run_sweep.sh output/e1_mass  _NO_GRIPPER_GROUND=0 _OBJ_DENSITY=6000
#
#  日志: /tmp/<basename 输出目录>.log
# ═══════════════════════════════════════════════════════════════════════════
set -u
cd "$(dirname "$0")" || exit 1

OUT="${1:?用法: ./run_sweep.sh <输出目录> VAR=val ...}"
shift
OVERRIDES=("$@")

# ★ 必须先清空位置参数再 source —— run_config.sh 末尾有"有 $1 就直接跑"的分支,
#   而 source 会继承调用方的 $1 ⇒ 不清空会**重复启动一个进程** (两个进程写同一份日志) ✗
set --
source ./run_config.sh >/dev/null

for kv in "${OVERRIDES[@]:-}"; do
    [ -n "$kv" ] || continue
    export "$kv"
done

PY=/home/an/miniconda3/envs/dex/bin/python
echo "[run_sweep] out=$OUT  覆盖=${OVERRIDES[*]:-}"
echo "[run_sweep] 生效: _NO_GRIPPER_GROUND=$_NO_GRIPPER_GROUND _S1_TABLE_PRESCREEN=${_S1_TABLE_PRESCREEN:-0} " \
     "_OBJ_DENSITY=${_OBJ_DENSITY:-1000} _OBJ_MIN_MASS=${_OBJ_MIN_MASS:-0.05} " \
     "_VIRTUAL_PD_K=$_VIRTUAL_PD_K _GRIP_FRICTION=$_GRIP_FRICTION " \
     "_GRIP_OVERCLOSURE=${_GRIP_OVERCLOSURE:-0.008}"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ 任务01 (2026-09-28) 【Stage1 执行语义透传: 默认 reuse, 搜索必须显式】 ✓ ★★★★
#   md ✓: "入口显式选择 reuse / verify / search; **默认 reuse**" ✓
#     ⇒ 本脚本**不**偷偷加 `search` ✓; 要跑搜索实验必须显式:
#         ./run_sweep.sh output/xxx ... _STAGE1_ACTION=search   (≤48 次预算 ✓)
#     需要复用**已冻结**的成功产物 (跨目录 ✓): `_STAGE1_ARTIFACT=/path/best_grasp.npz`
#     或 `_STAGE1_ARTIFACT_DIR=/path/to/stage1` ✓ (产物路径独立于 output-dir ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
_S1_ACTION_ARGS=()
[ -n "${_STAGE1_ACTION:-}" ] && _S1_ACTION_ARGS=(--stage1-action "$_STAGE1_ACTION")
exec nohup $PY -u grasp_hawor.py $PIPELINE_LINE --output-dir "$OUT" \
    "${_S1_ACTION_ARGS[@]}" \
    > "/tmp/$(basename "$OUT").log" 2>&1 &

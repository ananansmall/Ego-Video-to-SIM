#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_config.sh —— 管线**唯一参数来源** (2026-09-14 用户定策)
#
#  用户要求 (原话):
#    "你这个物理参数不要老是乱改, 一次测试就一套, 特别是夹爪的物理参数"
#
#  用法:
#    source run_config.sh          # 加载一套参数 (此后所有测试都用它 ✓)
#    ./run_config.sh <输出目录>     # 直接跑一次全管线
#
#  原则:
#    ① 一个测试会话**只改一个**变量, 其余从本文件走 ✓
#    ② 夹爪物理参数 (GRIPPER_* / FINGER_*) **不得**在命令行临时覆盖 ✗
#    ③ 改动必须写回本文件并注明日期与原因 ✓
# ═══════════════════════════════════════════════════════════════════════════

# ── 物体 (与历史成功运行 s5 一致 ✓) ──────────────────────────────────────
export _OBJ_SCALE=0.5                       # 物体缩放
export _OBJ_SCALE_TARGET=glb_4              # 缩放目标
export _OBJ_SHAPE=box                       # 物体碰撞形状 (box=简化盒 ✓)
export _OBJ_SHAPE_TARGET=

# ── 夹爪物理 (★ 不得临时覆盖 ✗) ──────────────────────────────────────────
export _GRIP_FRICTION=4.0                   # 摩擦
export _Q_OPEN=0.05                         # ★ 开合 (s5 成功值 ✓)
                                            #   注意: 不设时走"按尺寸自动" = 0.0225 ✗
                                            #   实测: 历史成功运行全部用 0.05 ✓
export _GRIP_FORCE_DOWN=1                   # 强制夹爪 X 朝下 (幂等 ✓)
# ★ 2026-09-20 用户定策: 开合方向**必须跟随 MANO** ⇒ 水平化关闭
#   为什么: 水平化 (绕 X 转 θ, 实测 θ=-60°~-123°) 会把开合轴从"MANO 真实捏合轴"
#     (夹角 **35.0°**, 已是"Y ⊥ approach"约束下的几何极限) 扭到 **77°~103°**。
#   实测 (唯一变量就是这个开关):
#     关: |Y·g|=0.836  Z·up=+0.403  与捏合轴 35.0°  抖动 3 帧  glb_4 27.5cm ✅跟随MANO
#     开: |Y·g|=0.000  Z·up=+0.932  与捏合轴 77°    抖动 4 帧  glb_4  0.9cm ❌不跟随
#   代价: Stage1 的"姿态偏好"软项 12分 → 最多 2分 (stage1_grasp_mixin.py:1813-1828),
#         **无硬门、不淘汰候选**; 该 warning 文案("已清零")是早期残留, 实际只是软扣分。
export _GRIP_CLOSE_AXIS_HORIZONTAL=0
# ★ 2026-09-20 新增: 姿态分支 DP (整段 Viterbi 全局最优, physics_utils.track_rotation_branch)
#   根因: `align_gripper_approach_to_gravity` 逐帧硬阈值, 在 approach 轴近水平时把
#         1° 的过零噪声放大成 180° 台阶 (实测台阶处 |X_raw.z| 仅 0.0047; 102 份历史
#         运行**全部**有 2 处 179° 台阶).
#   实测: 台阶 204 处 → 0, 最大帧间跳变 179.4° → 3.5°.
#   ⚠ 与 _GRIP_FORCE_DOWN **二选一**: DP=1 时 Stage2 的逐帧强制朝下自动降级为只诊断
#     (否则会把 DP 有意保留的近水平帧翻回去, 台阶重现). 回退: _ROT_BRANCH_DP=0
export _ROT_BRANCH_DP=1
# ★ 2026-09-20 新增: 开合轴 (Y) 的**符号**策略 (整段统一翻, 不是逐帧)
#   auto (默认) = 物理判据"夹爪手背 (Z 轴) 朝上"; |Z·up| 太小时回退到 MANO 捏合轴
#   mano = 与 MANO 拇指尖-食指尖同向 (语义正确, 但实测 Z 朝下 ⇒ 夹爪本体压向桌面)
#   legacy = 旧链路 (patch#4 时代) 的符号   keep = 不动
#   实测 (同一份数据, 唯一变量就是这个符号):
#     Z·up=+0.932 (auto)  → 手指抖动 4 帧,   glb_4 位移  0.9cm, glb_5   2.1cm ✅
#     Z·up=-0.932 (mano)  → 手指抖动 62 帧,  glb_4 位移 128.6cm, glb_5 481.8cm ❌
#   ⇒ R1 夹爪本体在 Z 负侧, Z 朝下时全程刮蹭被撞飞的物体.
export _GRIP_Y_SIGN=auto

# ── 虚拟关节 PD (与 s5 一致 ✓) ───────────────────────────────────────────
export _VIRTUAL_PD_K=50000
export _VIRTUAL_PD_D=600

# ── 场景 ─────────────────────────────────────────────────────────────────
export _SMOOTH_MINT=1                       # MANO 轨迹平滑
export _COACD_TARGET=glb_4                  # ★ 2026-09-20 用户定策: "stage1 一定是用 coacd 排的"
                                            #   ⇒ 目标物体必须走 CoACD (不再用 bbox 近似 ✓)
                                            #   注: 历史成功版 (v1/v2) 用的是 none(盒体) ✗;
                                            #       现在 Stage1 的排序/硬门/闭合目标三处**全部**依赖
                                            #       CoACD 凸包 ⇒ 必须开启, 否则退回 bbox 路径 ✗
export _NO_GRIPPER_GROUND=0                 # 夹爪↔桌面碰撞: 1=关 / 0=开 ✓
# ★ 2026-09-22 【必须纠正】审计文档 §1 明确 ✓:
#   "`run_config.sh` 当前仍是 `_NO_GRIPPER_GROUND=1`, 即**关闭夹爪—桌面碰撞**;
#    默认运行成功**不能验收用户要求的桌面版本**" ✗
#   ⇒ 改为 **0 (开启)** ✓ —— 用户要求的就是**真实桌面**下的结果 ✓
#     (此前多次"成功"都是在**没有桌面碰撞**下取得的 ⇒ 不构成验收 ✓)

# ── CoACD 凸分解 (2026-09-20 用户定策: "那现在这个用 64 啊" ✓) ────────────
export _COACD_MAX_HULL=64                   # ★ 凸包上限 = 64 (不是函数默认的 32 ✗)
                                            #   依据 ✓: 所有**已验证**的运行 (s1show / s1horiz /
                                            #     final_ok 一线) 用的都是 64 凸包那份缓存
                                            #     (key=fbc2248c5fc8e011, 闭合轴范围 [−18.8,+18.8]mm ✓);
                                            #     默认 32 ⇒ cache key 变了 ⇒ **现场重分解** ✗
                                            #     ⇒ 几何/接触与已验证结果不一致 ✗
                                            #   注意 ✓: 它是 cache key 的一部分 ⇒ 改它会换一份凸包 ✓
export _COACD_FLAT_BOTTOM=1                 # 底部压平 (落地不翻滚 ✓)

# ── Stage1 候选 (本会话新增 ✓) ───────────────────────────────────────────
export _STAGE1_MULTIFRAME=1                 # 多帧扫描
export _STAGE1_FRAME_SPAN=15                # 半宽 ±15 帧
export _STAGE1_ROT_SPAN_DEG=5               # 候选姿态扰动 ±5° (0 = 关闭 ✓)

# ── Stage2 接近段优化 N-5 ────────────────────────────────────────────────
export _APPROACH_OPT=0                      # 0=关 / 1=开
# ★ 2026-09-21 用户原则: "规划全部用 coacd" ⇒ 几何一致性问题**已解决** ✓
#   (`TargetHullModel` 读引擎 collision shapes ✓ 见 md §53.1 ✓), 但**暂默认关** ✗ 原因:
#   规划几何换成凸包后, N-5 实测 `RuntimeError: N-5 infeasible: no candidate within
#   tolerance` ✗✗ (geom4/geom5 两轮进程直接死 ✓) —— 因为 Stage1 的 anchor
#   **指尖离桌面=0.0mm / 最近侧=12.5mm** ⇒ 指体与物体/桌面**本身就相交**,
#   任何偏移都消不掉 ⇒ 无可行候选 ✗
#   ⇒ 先修 **anchor (Stage1 离桌面余量 G5 硬门 ✓ 见 md §46.6)** ✓, 再开本开关 ✓
#     (现已默认 `_N5_REQUIRE_FEASIBLE=0` ⇒ 不再崩, 只 ERROR + 零偏移 ✓)

# ── 诊断开关 ─────────────────────────────────────────────────────────────
export _STAGE3_WATCH_FRAMES=12              # Stage3 起手自检帧数

# ── 命令行固定参数 (与 s5 一致 ✓) ────────────────────────────────────────
PIPELINE_LINE="--test-stage3 --side left --views all \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --stage2-approach-mode axial --stage2-approach-start-m 0.16 \
  --stage2-approach-time-scale 1.0 --stage2-carry-time-scale 2.0"

# 直接运行模式:  ./run_config.sh output/xxx
if [ -n "${1:-}" ]; then
    cd "$(dirname "$0")" || exit 1
    PY=/home/an/miniconda3/envs/dex/bin/python
    echo "[run_config] 输出目录=$1  参数来自 run_config.sh (唯一来源 ✓)"
    exec nohup $PY -u grasp_hawor.py $PIPELINE_LINE --output-dir "$1" \
        > "/tmp/$(basename "$1").log" 2>&1 &
fi

echo "[run_config] 参数已加载 ✓  可用 \$PIPELINE_LINE 组装命令"
echo "  _Q_OPEN=$_Q_OPEN  _NO_GRIPPER_GROUND=$_NO_GRIPPER_GROUND  _APPROACH_OPT=$_APPROACH_OPT"

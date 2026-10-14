#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 5.5 编排入口: stage5 产物 → 确定性 Z 贴合 → opencode agent 调 XY → 碰撞兜底 → final_scene_stage55.glb

不修改任何 stage5 文件; 只读 final_scene_stage5.glb, 全部修改作用于
工作副本 stage55_work.glb, 最终导出 final_scene_stage55.glb。

用法:
  python tools/run_stage55.py --scene_dir output_v2/121_C5_CellPhone_161deg_vggt_omega \
      [--model sensenova/sensenova-6.8-flash-lite] [--skip-agent] [--agent-timeout 1800]

流程:
  1. 复制 stage5 GLB → stage55_work.glb (已存在则断点续跑)
  2. 确定性预处理: snap_support (Z 贴合, root-first, ≤0.15m)
  3. opencode agent 循环: check_glb/check_support/render → set_xy/set_z → 复检 (≤8 轮)
  4. 确定性兜底: nudge (沿碰撞向量推挤, ≤4 轮)
  5. snap_support 收尾 → 导出 final_scene_stage55.glb + stage55_report.json + 预览图
退出码: 0 = 零碰撞且支撑 ok; 1 = 存在残余问题; 2 = 输入缺失
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage55_cli as qc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_PY = sys.executable
DEFAULT_MODEL = "sensenova/sensenova-6.8-flash-lite"
OPENCODE_BIN = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")


def find_input_glb(scene_dir, override=None):
    if override:
        return override
    for name in ("final_scene_stage5.glb", "final_scene_stage4_5.glb", "final_scene.glb"):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            return p
    return None


def build_agent_prompt(scene_dir, work_glb, max_rounds):
    py = DEFAULT_PY
    cli = os.path.join(HERE, "stage55_cli.py")
    ref = os.path.join(scene_dir, "optimal_frames")
    ref_note = ""
    if os.path.isdir(ref):
        frames = sorted(f for f in os.listdir(ref) if f.endswith((".jpg", ".png")))
        if frames:
            ref_note = f"\n实拍参考帧 (真实场景照片, 用于判断物体语义位置): {os.path.join(ref, frames[0])} 等"
    return f"""你是场景布局精修 agent (stage5.5)。任务: 只调整物体位置, 消除碰撞并把物体放稳。

场景目录: {scene_dir}
工作 GLB: {work_glb}  (z-up 坐标系, 单位米; 这是唯一允许修改的文件)
{ref_note}

用 bash 运行工具 (PY={py}):
  {py} {cli} inspect --glb {work_glb}
      # 列出所有实例: 名称 bbox 体积
  {py} {cli} check_glb --glb {work_glb}
      # 碰撞报告 JSON: 每对含 depth_m / mover / world_dxdy (建议修复向量)
  {py} {cli} check_support --glb {work_glb}
      # 支撑报告 JSON: 每个物体 floating/embedded/ok + fix_dz
  {py} {cli} set_xy --glb {work_glb} --obj <名称> --dx <米> --dy <米>
      # XY 平移 (z-up 世界系)
  {py} {cli} set_z --glb {work_glb} --obj <名称> --dz <米>
  {py} {cli} render --glb {work_glb}
      # 输出四视图预览 PNG 路径, 用 read 工具查看图片判断布局是否合理
  {py} {cli} compare --glb {work_glb}
      # 【核心工具】真实机位下 渲染|实拍 并排对照图 (左=RENDER 右=REAL),
      # 用 read 查看, 逐物体对比布局差异

你的目标是让渲染布局与实拍照片一致 (视觉对齐), 几何检查只是兜底。
迭代循环, 最多 {max_rounds} 轮, 每轮走完整闭环:
 1. compare 生成对照图并逐帧查看
 2. 逐物体审计并写下判断: 实拍里它应在哪 (放在什么上面/紧挨着谁/前后
    关系), 渲染里它现在哪, 差异是什么。常见差异: 放错支撑物 (check_support
    判 ok 只代表几何上搭着东西, 看不见放错) / 与邻居粘连或穿插
    (mutual_overlap 不一定是自然贴靠) / 位置正确
 3. 有差异的用 set_xy 修正, 位移幅度按对照图中的比例估算, 一次到位
 4. 碰撞对另用 check_glb 的 world_dxdy 修复向量平移 mover (取 0.5~1.0 倍)
 5. 重新 compare 验证: 该差异消失则处理下一处; 仍有明显布局差异则进入
    下一轮
退出条件: 布局与实拍一致或达轮数上限。最后跑 check_glb + check_support
兜底 (n_clashes=0 且无大幅 floating/embedded), 输出 JSON 摘要:
{{"rounds": N, "moves": M, "clashes_left": K, "summary": "..."}}
约束:
 - 只能通过上述 CLI 修改 {work_glb}, 禁止直接编辑 GLB 文件
 - 禁止修改、删除任何其他项目文件
 - 不要旋转物体, 只平移
 - 禁止调用 nudge / snap_support / set_z: 位置修正必须由你给出具体的
   set_xy 语义决策 (哪个物体、往哪挪、为什么), 这是你的核心职责
 - large_gap 悬空/嵌入物体 (check_support 中 large_gap=true) 是你的首要任务:
   对照实拍参考帧判断它该在哪个支撑物上, 用 set_xy 挪到该支撑物 footprint
   上方 (XY 覆盖支撑物即可, 剩余高度差由你结束后的确定性 Z 贴合处理)
 - 你的回合必须在执行完所有 set_xy 修改后才能结束; 禁止只输出计划/分析
   文字而不跑任何修改命令。若确实无需修改, 也必须输出
   {{"moves": 0, "summary": "..."}} 并说明逐项检查了什么
 - 不要花时间去阅读管线源码/日志来推断该做什么——你的唯一信息来源是
   CLI 检测输出 + 实拍参考帧图片, 直接决策直接执行
 - 若某碰撞对反复修改 2 轮仍存在, 跳过它 (后续确定性兜底会处理)
"""


def run_agent(scene_dir, work_glb, model, max_rounds, timeout_s, log_path):
    """运行 opencode agent, 返回 (ok, stdout摘要)"""
    if not os.path.exists(OPENCODE_BIN):
        print(f"⚠️ opencode 不存在 ({OPENCODE_BIN}), 跳过 agent, 仅确定性流程")
        return False, ""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "permission": {"edit": "allow", "bash": {"*": "allow"}, "webfetch": "deny"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    prompt = build_agent_prompt(scene_dir, work_glb, max_rounds)
    env = dict(os.environ)
    env["OPENCODE_CONFIG"] = cfg_path
    cmd = [OPENCODE_BIN, "run", "--model", model, prompt]
    print(f"🤖 opencode agent 启动: {model} (超时 {timeout_s}s)", flush=True)
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, cwd=PROJ, env=env, stdout=logf,
                                  stderr=subprocess.STDOUT, timeout=timeout_s)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠️ agent 超时, 使用当前工作 GLB 继续", flush=True)
        ok = False
    finally:
        os.unlink(cfg_path)
    tail = ""
    if os.path.isfile(log_path):
        with open(log_path) as f:
            tail = f.read()[-2000:]
    print(f"   agent {'完成' if ok else '异常退出'} (日志: {log_path})", flush=True)
    return ok, tail


def main():
    ap = argparse.ArgumentParser(description="Stage 5.5 编排入口")
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--input_glb", default=None, help="覆盖输入 GLB (默认 final_scene_stage5.glb)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="opencode 模型")
    ap.add_argument("--max-agent-rounds", type=int, default=8)
    ap.add_argument("--agent-timeout", type=int, default=1800)
    ap.add_argument("--skip-agent", action="store_true", help="只跑确定性流程 (调试用)")
    ap.add_argument("--no-nudge", action="store_true", help="跳过确定性碰撞兜底")
    args = ap.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    assert os.path.isdir(scene_dir), f"场景目录不存在: {scene_dir}"
    src = find_input_glb(scene_dir, args.input_glb)
    assert src, f"未找到 stage5 产物 GLB (final_scene_stage5.glb 等) 于 {scene_dir}"

    work_glb = os.path.join(scene_dir, "stage55_work.glb")
    final_glb = os.path.join(scene_dir, "final_scene_stage55.glb")
    report_path = os.path.join(scene_dir, "stage55_report.json")

    # 1. 工作副本 (断点续跑: 已存在则不覆盖)
    if not os.path.isfile(work_glb):
        shutil.copy(src, work_glb)
        print(f"📋 工作副本: {src} → {work_glb}")
    else:
        print(f"📋 复用已有工作副本: {work_glb} (删除它可重头跑)")

    report = {"scene_dir": scene_dir, "input_glb": os.path.basename(src)}

    # 记录初始状态
    insts0 = qc.load_instances(work_glb)
    report["initial"] = {
        "n_clashes": len(qc.check_glb(insts0)),
        "n_support_issues": sum(1 for r in qc.check_support(insts0).values()
                                if r.get("status") in ("floating", "embedded")),
    }
    print(f"初始: 碰撞对 {report['initial']['n_clashes']}, "
          f"支撑问题 {report['initial']['n_support_issues']}", flush=True)

    # 2. 确定性 Z 贴合
    applied = qc.snap_support(work_glb)
    print(f"🔧 snap_support: {len(applied)} 处 Z 贴合", flush=True)
    report["snap_pre"] = applied

    # 3. opencode agent 调 XY
    if not args.skip_agent:
        ok, tail = run_agent(scene_dir, work_glb, args.model, args.max_agent_rounds,
                             args.agent_timeout,
                             os.path.join(scene_dir, "stage55_agent_run.log"))
        report["agent"] = {"ok": ok, "model": args.model, "stdout_tail": tail[-800:]}
    else:
        print("⏭️ --skip-agent: 跳过 agent", flush=True)
        report["agent"] = {"skipped": True}

    # 4. 确定性碰撞兜底
    if not args.no_nudge:
        res = qc.nudge(work_glb, rounds=4)
        print(f"🔨 nudge: {json.dumps(res['history'])}", flush=True)
        report["nudge"] = res

    # 5. 收尾: Z 贴合 + 导出
    # agent 已完成语义 XY 决策, 此处剩余的垂直 gap 是合法落座距离,
    # 用大 cap 让 large_gap 悬空物体也能落到正确支撑物上
    qc.snap_support(work_glb, cap=0.60)
    shutil.copy(work_glb, final_glb)
    print(f"💾 最终输出: {final_glb}")

    # 最终 QC
    insts_f = qc.load_instances(final_glb)
    clashes = qc.check_glb(insts_f)
    support = qc.check_support(insts_f)
    report["final"] = {
        "n_clashes": len(clashes),
        "clashes": clashes,
        "support_issues": {k: v for k, v in support.items()
                           if v.get("status") in ("floating", "embedded")},
    }
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 报告: {report_path}")
    print(f"最终: 碰撞对 {len(clashes)}, 支撑问题 "
          f"{len(report['final']['support_issues'])}")

    # 预览图
    try:
        from render_glb_preview import render_preview_png
        render_preview_png(final_glb)
    except Exception as e:
        print(f"⚠️ 预览渲染失败: {e}")

    sys.exit(0 if not clashes and not report["final"]["support_issues"] else 1)


if __name__ == "__main__":
    main()

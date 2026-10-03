"""T07.2 反例: 执行记录的**位置与字段** (不跑物理 ✓)

md T07.1 原话 ✓: "记录点放在**所有 clip/限速/保护完成之后、唯一命令下发处**;
  每个控制周期都记录 …**不能只记某个 else 分支**"; 且需含**旋转**与 source_phase;
  一次 step 完成后挂接相同 index 的 post-step observation (禁止错一帧) ✓

本测试**静态检查生产源码结构** ⇒ 秒级、不开物理场景 ✓
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "stage1_grasp_mixin.py")


def _lines():
    with open(_SRC, encoding="utf-8") as f:
        return f.read().splitlines()


def test_record_site_is_immediately_before_the_single_dispatch():
    ls = _lines()
    rec = [i for i, l in enumerate(ls) if '_tr_c.setdefault("q_cmd", []).append' in l]
    assert len(rec) == 1, "命令记录点必须**唯一**(实测 %d 处) ⇒ 禁止只记某个 else 分支 ✗" % len(rec)
    i = rec[0]
    # 其后 40 行内必须出现唯一的下发点 physics_step ✓
    tail = ls[i:i + 40]
    assert any("physics_step(" in l for l in tail), "记录点必须在 physics_step 之前(同段) ✗"
    # 记录点之前 25 行内必须已过 W2 出口保护 (限速/跳变保护) ✓
    head = ls[max(0, i - 60):i]
    assert any("_q_last_cmd" in l or "W2 命令跳变" in l for l in head), (
        "记录点必须在 W2 出口保护**之后** (记录 clip 后的值, 不是 clip 前 ✗)")


def test_post_step_observation_follows_dispatch_same_index():
    ls = _lines()
    rec = [i for i, l in enumerate(ls) if '_tr_c.setdefault("q_cmd", []).append' in l][0]
    obs = [i for i, l in enumerate(ls) if '_tr_o.setdefault("q_act", []).append' in l]
    assert len(obs) == 1, "post-step observation 也必须**唯一** ✗"
    assert obs[0] > rec, "observation 必须在下发之后 (command→step→observation ✓)"
    seg = "\n".join(ls[rec:obs[0] + 1])
    assert "physics_step(" in seg, "observation 必须紧跟同一次 step ✗"
    assert "错帧" in seg or "control_index" in seg, "必须做同 index 自检(禁止错一帧 ✗)"


def test_record_contains_rotation_phase_and_source_phase():
    ls = _lines()
    body = "\n".join(ls)
    for needle in ('"T_W_G"', '"phase"', '"source_phase"', '"control_index"',
                   '"q_act"', '"obj_z"', '"valid"'):
        assert needle in body, "完整记录缺少字段 %s ✗" % needle
    # 旋转必须是 4×4 (T_W_G 由 target_R + current_pos 组装 ✓)
    assert re.search(r"np\.eye\(4[^)]*\)", body), "T_W_G 必须由 4×4 齐次矩阵组装 ✓"


def test_only_one_recording_site_in_file():
    """旧的"某个分支里单独记录"必须已被移除 ✗ (唯一记录点 ✓)."""
    ls = _lines()
    legacy = [i for i, l in enumerate(ls) if '_tr2["t"].append' in l]
    assert not legacy, "残留旧记录点(分支内) ⇒ 违反唯一记录点 ✗"

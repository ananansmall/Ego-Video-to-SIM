"""★ 统一 yaml 配置的零物理测试 (2026-10-02) ✓"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_exists_and_parses():
    import yaml
    p = ROOT / "config" / "sim.yaml"
    assert p.exists(), "统一配置文件必须存在 ✓"
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(d, dict) and d, "yaml 必须非空 ✓"
    # ⚠ meta 节是**说明性**的 (note/rollout_cost_s), 不是 env 键 ⇒ 排除 ✓
    keys = [k for name, sec in d.items() if isinstance(sec, dict) and name != "meta"
            for k in sec]
    assert keys, "至少要有配置节 ✓"
    assert all(str(k).startswith("_") for k in keys), (
        "除 meta 外, 所有真实配置键必须以 `_` 开头 (env 名 ✓)")


def test_key_entries_present():
    """关键项必须在 yaml 里有唯一来源 (步长/预算保留/抬升分段) ✓"""
    import yaml
    d = yaml.safe_load((ROOT / "config" / "sim.yaml").read_text(encoding="utf-8"))
    flat = {k: v for sec in d.values() if isinstance(sec, dict) for k, v in sec.items()}
    for k in ("_PHYS_DT", "_CONTROL_HZ", "_S1_VERIFY_RESERVE",
              "_S2_LIFT_POINTS", "_S2_LIFT_OFF_M", "_ENGINE_CCD"):
        assert k in flat, "缺少关键配置项 %s ✓" % k


def test_does_not_override_existing_env(monkeypatch):
    """★ 优先级: 已有 env (CLI/外部) **优先** ⇒ 绝不覆盖 ✗"""
    monkeypatch.setenv("_S2_LIFT_OFF_M", "0.099")
    sim_config.load_sim_config(verbose=False)
    assert os.environ["_S2_LIFT_OFF_M"] == "0.099", "已有 env 不得被 yaml 覆盖 ✗"


def test_applies_when_absent(monkeypatch):
    monkeypatch.delenv("_S2_LIFT_POINTS", raising=False)
    sim_config.load_sim_config(verbose=False)
    assert os.environ.get("_S2_LIFT_POINTS") == "15", "缺失时应由 yaml 提供默认 ✓"


def test_physics_step_rate_is_480():
    """物理步频必须与 yaml 一致 (480 步/秒 ⇒ timestep = 1/480) ✓"""
    import importlib
    monkeypatch_env = os.environ.copy()
    monkeypatch_env.pop("_PHYS_DT", None)
    os.environ.pop("_PHYS_DT", None)
    sim_config.load_sim_config(verbose=False)
    import physics_env
    importlib.reload(physics_env)
    assert abs(1.0 / physics_env.PHYSICS_TIMESTEP - 480.0) < 1e-6
    assert physics_env.DECIMATION == 16, "30Hz / (1/480) ⇒ 16 子步 ✓"
    os.environ.update(monkeypatch_env)

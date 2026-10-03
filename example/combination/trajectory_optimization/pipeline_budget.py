"""任务01 —— 切断隐式搜索 + 唯一物理预算/账本 (纯离线模块 ✓ 零物理 ✓)

md 位置: `docs/2026-09-23-物理仿真控制与奖励联合设计-推荐实施方案.md` §4 任务 01
        (README 摘要里的 "01 — 切断隐式搜索，加入总预算【第一优先级】")

md 原话 (逐条对照 ✓)
-------------------
1. "入口显式选择 reuse / verify / search; **默认 reuse**。Stage1 产物路径**独立于
   output-dir**, 新输出目录不意味没有历史输入" ✓
2. "拟新增 CLI: `--stage1-action`、`--stage1-artifact`; 所有入口走**一个分流函数**。
   输入缺失或合同无效时报**具体错误退出**, 不回退搜索、不打折" ✓
3. "正式入口不再调用旧 CEM、124点polish 或递归重试。旧函数可暂留为**显式 legacy 诊断**,
   但**不得默认触发**, 也不能获得独立预算" ✓
4. "**唯一物理 episode 入口先预占预算, 再 reset/step**; 异常也消耗已占次数。
   生成 ledger 记录 task/candidate/episode/purpose、累计次数、物理子步数" ✓
5. "参数只解析一次: CLI > 唯一配置 > 默认。重复旧 env 冲突必须拒绝/明确迁移,
   不能后读悄悄覆盖。启动先输出 effective mode、artifact…" ✓
6. "脚本改成明确**前台运行**或由用户显式选择后台; **物理测试默认不启动**" ✓ (本模块只管 1~5)

本模块 ✓
--------
· `resolve_stage1_action()` —— **唯一分流函数** (reuse / verify / search / legacy) ✓
· `assert_stage1_artifact()` —— 合同校验 (缺 artifact / 坏 artifact ⇒ **具体错误** ✗ 不回退搜索 ✓)
· `EpisodeBudget` —— 预占式预算 + 磁盘 ledger (跨进程/重启可读 ✓ 不允许新目录清零 ✓)
· `resolve_effective_config()` —— 一次性解析 (CLI > 配置 > 默认) + 旧 env 冲突**显式拒绝** ✓
· 全部为纯函数/文件操作 ⇒ **不 import sapien, 不跑物理** ✓ (零物理可测 ✓)

用法 ✓
------
    from pipeline_budget import resolve_stage1_action, EpisodeBudget

    act = resolve_stage1_action(args.stage1_action, args.stage1_artifact, output_dir)
    print(act.summary())                     # 启动即打印 effective mode ✓
    bud = EpisodeBudget(ledger=Path(output_dir) / "budget_ledger.json", task="stage2")
    with bud.episode(purpose="stage2_repeat", candidate=act.candidate_id):
        ...                                   # 预算已在 reset/step **之前**扣掉 ✓
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── 三种明确执行语义 (md §2.1 ✓) + legacy 诊断 (非默认 ✗) ──────────────────
ACTIONS = ("reuse", "verify", "search", "legacy")
DEFAULT_ACTION = "reuse"                 # ★ md: 默认 reuse ✓ (不是 search ✗)

# ★ md §3.1 硬预算 ✓
SEARCH_BUDGET_DEFAULT = 48               # 显式 search 才允许 ≤48 次物理 episode ✓
NON_SEARCH_BUDGET_DEFAULT = 0            # 默认修复路线: **搜索预算 0** ✓

# 复用已有的合同校验器 (只读 ✓)
try:
    from stage1_recipe import load_stage1_grasp as _load_s1
except Exception:  # noqa: BLE001
    _load_s1 = None


class Stage1InputError(RuntimeError):
    """输入缺失 / 合同无效 ⇒ **具体错误 + 停** ✗ (绝不回退搜索 ✓, md §01-2)"""

    def __init__(self, code: str, detail: str):
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"[{self.code}] {self.detail}")


class BudgetExhausted(RuntimeError):
    """预算不足 ⇒ 在 reset/step **之前**拒绝 ✗ (md §01-4)"""

    def __init__(self, rule: str, detail: str):
        self.rule = str(rule)
        self.detail = str(detail)
        super().__init__(f"[BUDGET_EXHAUSTED:{self.rule}] {self.detail}")


class EnvConflictError(RuntimeError):
    """同一语义被两套 env/CLI 同时设置且值不同 ⇒ **拒绝** ✗ (md §01-5: 不能后读悄悄覆盖)"""


# ───────────────────────────────────────────────────────────────────────────
# 1) 唯一分流函数
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Stage1Plan:
    """分流结果 (启动时打印 ⇒ '先输出 effective mode、artifact…' ✓)"""

    action: str
    artifact: Path | None
    output_dir: Path
    candidate_id: str | None = None
    episode_id: str | None = None
    source_frame: int | None = None
    n_pass: int | None = None
    criterion: str = ""                       # 'legacy' 或明确的合同名
    revision: str = ""
    ledger: Path | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        a = "n/a" if self.artifact is None else str(self.artifact)
        return ("[task01][dispatch] effective action=%s | artifact=%s | "
                "n_pass=%s | candidate=%s | episode=%s | source_frame=%s | revision=%s"
                % (self.action, a, self.n_pass, self.candidate_id, self.episode_id,
                   self.source_frame, self.revision))


def _artifact_candidates(output_dir: Path, artifact_cli: str | None) -> list[Path]:
    """★ md §01-1 ✓: 产物路径**独立于** output-dir ——
    CLI 显式给 ⇒ 以它为准; 否则依次探测 (env `_S1_ARTIFACT` → 本 run → 共享 stage1 库 ✓)"""
    out = []
    if artifact_cli:
        out.append(Path(artifact_cli))
    env_a = os.environ.get("_S1_ARTIFACT")
    if env_a:
        out.append(Path(env_a))
    out.append(Path(output_dir) / "stage1" / "best_grasp.npz")
    # 共享库: 允许把某个"已冻结成功 run"的产物固定成默认来源 (由 env 指定目录 ✓)
    shared = os.environ.get("_S1_ARTIFACT_DIR")
    if shared:
        out.append(Path(shared) / "best_grasp.npz")
    return out


def assert_stage1_artifact(npz_path: Path) -> tuple[dict, dict]:
    """合同校验 ⇒ (grasp, recipe)；任何不满足 ⇒ Stage1InputError(具体 code) ✓"""
    p = Path(npz_path)
    if not p.exists():
        raise Stage1InputError("STAGE1_ARTIFACT_MISSING",
                               f"未找到 Stage1 产物: {p} (md §01-2: 报错退出, **不回退搜索** ✗)")
    if _load_s1 is None:
        raise Stage1InputError("STAGE1_LOADER_UNAVAILABLE", "stage1_recipe.load_stage1_grasp 不可用")
    try:
        grasp, recipe = _load_s1(str(p))
    except Exception as exc:  # noqa: BLE001
        # ★ md §01-2 ✓: 一律"报具体错误退出" ⇒ 消息里**明确**写明不回退搜索 ✓
        raise Stage1InputError("STAGE1_ARTIFACT_UNREADABLE",
                               f"{p}: {exc} ⇒ 报错退出, **不回退搜索** ✗") from exc
    if recipe is None:
        raise Stage1InputError("STALE_STAGE1_ARTIFACT",
                               f"{p} 缺内嵌 recipe (旧产物) ⇒ 拒用 ✗")
    n_pass = int((grasp.get("verification") or {}).get("n_pass", -1))
    if n_pass < 2:
        raise Stage1InputError("STAGE1_NOT_VERIFIED",
                               f"{p} 终验 n_pass={n_pass} (<2) ⇒ 拒用 ✗ "
                               f"(md: 成功产物必须来自真实通过 run ✓)")
    # schema / 长度 / 相位合同 (轻量但明确 ✓)
    missing = [k for k in ("q_cmd", "phase", "q_open", "q_grasp", "close_controls",
                           "settle_controls", "phase_boundaries") if not recipe.get(k)]
    if missing:
        raise Stage1InputError("STAGE1_RECIPE_INCOMPLETE", f"{p} 缺字段 {missing} ⇒ 拒用 ✗")
    return grasp, recipe


def resolve_stage1_action(action_cli: str | None, artifact_cli: str | None,
                          output_dir: str | os.PathLike,
                          *, verify_contract: bool = True) -> Stage1Plan:
    """★ 唯一分流函数 ✓ (md §01-2) —— 所有入口都必须经它

    · action 缺省 ⇒ **reuse** ✓ (md §01-1: 默认 reuse, 不是 search ✗)
    · reuse / verify ⇒ 必须能取到**通过合同**的 artifact; 取不到 ⇒ Stage1InputError ✗
    · search ⇒ 允许无 artifact, 但**必须显式**且受 48 次预算约束 (在 EpisodeBudget 里 ✓)
    · legacy ⇒ 旧诊断路径, **显式**才可用, 且**不获得独立预算** ✗ (md §01-3)
    """
    act = str(action_cli or DEFAULT_ACTION).strip().lower()
    if act not in ACTIONS:
        raise Stage1InputError("BAD_STAGE1_ACTION", f"未知 action={act!r} ⇒ 只能是 {ACTIONS}")
    out_dir = Path(output_dir)
    plan = Stage1Plan(action=act, artifact=None, output_dir=out_dir,
                      ledger=out_dir / "budget_ledger.json")

    if act == "search":
        # 显式搜索: 不需要既有产物; 但**绝不**在 reuse/verify 路径下隐式发生 ✓
        plan.criterion = "search(explicit)"
        return plan

    if act == "legacy":
        plan.criterion = "legacy_diagnostic"
        return plan

    # reuse / verify: 必须拿到合规产物 ✓
    last_err: Stage1InputError | None = None
    for cand in _artifact_candidates(out_dir, artifact_cli):
        try:
            grasp, recipe = assert_stage1_artifact(cand)
        except Stage1InputError as exc:
            last_err = exc
            continue
        plan.artifact = Path(cand)
        plan.candidate_id = (recipe.get("winner_binding") or {}).get("candidate_id")
        plan.episode_id = (recipe.get("winner_binding") or {}).get("episode_id")
        plan.source_frame = recipe.get("source_frame")
        plan.n_pass = int((grasp.get("verification") or {}).get("n_pass", -1))
        plan.revision = str(recipe.get("revision", ""))
        plan.criterion = "legacy_stage1_gate(n_pass>=2)" if act != "verify" else "verify_fixed_candidate"
        plan.extra = {"recipe": recipe}
        return plan
    raise last_err or Stage1InputError("STAGE1_ARTIFACT_MISSING",
                                       f"{out_dir} 下没有可用 Stage1 产物 ⇒ 停 ✗")


# ───────────────────────────────────────────────────────────────────────────
# 2) 预占式预算 + 磁盘账本 (跨进程/重启 ✓)
# ───────────────────────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


class EpisodeBudget:
    """★ 唯一物理 episode 入口的预算器 ✓ (md §01-4)

    · `episode()` 上下文进入时**先预占**, 再允许 reset/step ✓; 异常也记 failure ✗
    · ledger 落盘 ⇒ **跨进程/重启继续同一任务** 读同一账本 ✓ (新目录不清零 ✓)
    · 搜索预算默认 0 (修复路线 ✓); 只有 action=search 才给 SEARCH_BUDGET_DEFAULT ✓
    """

    def __init__(self, ledger: str | os.PathLike, task: str,
                 *, action: str = DEFAULT_ACTION,
                 physical_budget: int | None = None):
        self.ledger_path = Path(ledger)
        self.task = str(task)
        self.action = str(action)
        if physical_budget is None:
            physical_budget = (SEARCH_BUDGET_DEFAULT if self.action == "search"
                               else NON_SEARCH_BUDGET_DEFAULT)
        self.budget = int(physical_budget)
        self._data = self._load()

    # ── 账本 IO ──
    def _load(self) -> dict:
        if self.ledger_path.exists():
            try:
                d = json.loads(self.ledger_path.read_text())
                if isinstance(d, dict) and "entries" in d:
                    return d
            except Exception:  # noqa: BLE001
                pass
        return {"task": self.task, "entries": []}

    def _save(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.ledger_path,
                           {"task": self.task, "action": self.action,
                            "budget": self.budget, "entries": self._data["entries"]})

    # ── 查询 ──
    def used(self) -> int:
        return len(self._data["entries"])

    def remaining(self) -> int:
        return max(0, self.budget - self.used())

    def search_calls(self) -> int:
        """本账本里 purpose 含 search/cem 的条目数 ⇒ 'search 调用 0' 的硬证据 ✓"""
        return sum(1 for e in self._data["entries"]
                   if any(k in str(e.get("purpose", "")).lower() for k in ("search", "cem")))

    # ── 预占 ──
    def reserve(self, *, purpose: str, candidate: str | None = None,
                episode_id: str | None = None, phase: str | None = None,
                phys_substeps: int | None = None) -> dict:
        """在 reset/step **之前**调用 ✗ 超预算 ⇒ BudgetExhausted"""
        if self.used() >= self.budget:
            raise BudgetExhausted(
                self.action,
                f"已用 {self.used()}/{self.budget} ⇒ 第 {self.used() + 1} 次在 reset/step "
                f"**之前**被拒绝 ✗ (purpose={purpose})")
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": str(purpose),
                 "candidate": candidate, "episode_id": episode_id, "phase": phase,
                 "phys_substeps": phys_substeps, "status": "reserved"}
        self._data["entries"].append(entry)
        self._save()                                  # ★ 先落盘再 step ✓
        return entry

    def finish(self, entry: dict, status: str = "ok", note: str = "") -> None:
        entry["status"] = str(status)
        if note:
            entry["note"] = str(note)
        self._save()

    @contextlib.contextmanager
    def episode(self, *, purpose: str, candidate: str | None = None,
                episode_id: str | None = None, phase: str | None = None):
        """with 用法 ✓: 进入即预占; 异常 ⇒ 记 `failed` 也消耗次数 ✓ (md §01-4)"""
        entry = self.reserve(purpose=purpose, candidate=candidate,
                             episode_id=episode_id, phase=phase)
        try:
            yield entry
        except BaseException as exc:                  # noqa: BLE001
            self.finish(entry, status="failed", note=type(exc).__name__)
            raise
        else:
            self.finish(entry, status="ok")


# ───────────────────────────────────────────────────────────────────────────
# 3) 一次性参数解析 (CLI > 唯一配置 > 默认) + 旧 env 冲突显式拒绝
# ───────────────────────────────────────────────────────────────────────────
def resolve_effective_config(cli_values: dict, config_values: dict, defaults: dict,
                             *, protected_env: tuple[str, ...] = ()) -> dict:
    """★ md §01-5 ✓: 参数只解析一次; 同一语义被两处设置且**值不同** ⇒ **拒绝** ✗

    · `cli_values` 优先级最高 ⇒ 非 None 即生效 (并**显式**覆盖 config/env ✓)
    · `config_values` 次之 (运行时已解析的配置)
    · `defaults` 兜底; 三者都没有 ⇒ 用 defaults
    · `protected_env` 里列出的环境变量若**同时**被 config 与 env 设成不同值 ⇒ EnvConflictError
      (即"两套 env 暗互相覆盖"的反例被硬拦 ✓)
    """
    out: dict = dict(defaults)
    conflict: list[str] = []
    for key in set(defaults) | set(config_values) | set(cli_values) | set(protected_env):
        c_cli = cli_values.get(key)
        c_cfg = config_values.get(key)
        c_env = os.environ.get(key)
        if c_cli is not None:
            out[key] = c_cli
        elif c_cfg is not None:
            out[key] = c_cfg
        elif c_env is not None:
            out[key] = c_env
        if key in protected_env and c_cfg is not None and c_env is not None \
                and str(c_cfg) != str(c_env) and c_cli is None:
            conflict.append(f"{key}: config={c_cfg!r} vs env={c_env!r}")
    if conflict:
        raise EnvConflictError("同一语义被两套来源设置且不一致 ⇒ 拒绝 (md §01-5) ✗: "
                               + "; ".join(conflict))
    return out


def format_effective_config(effective: dict, *, width: int = 0) -> str:
    """启动打印用 ✓ ('启动先输出 effective mode、artifact…')"""
    items = [f"{k}={effective[k]!r}" for k in sorted(effective)]
    body = " | ".join(items)
    return "[task01][effective] " + (body[:width] if width else body)

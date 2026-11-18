"""Prompt-only A/B: independent empty caches, identical poor start, five episodes per arm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import agentic_agent as A
import agentic_optimize as AO
from agent_tools import AgentTools
import agentic_replay as R
import pipeline as PIPE
from vlm_lifecycle import ensure_vlm, release_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('without_gt', 'with_gt'), required=True)
    ap.add_argument('--budget', type=int, default=5)
    ap.add_argument('--output-dir', required=True)
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / (a.arm + '.json')
    if report_path.exists() or (out / (a.arm + '_cache.json')).exists():
        raise FileExistsError(report_path)
    cfg = PIPE.load_task(str(ROOT / 'configs/duck_to_plate.json'))
    cfg['robot']['base_xy'] = [0.12, -0.40]
    cfg.setdefault('tuning', {})['ik_seeds'] = 4
    PIPE.apply_adapters(cfg)
    AO.TASK_CFG = cfg
    AO.ROBOT_NAME = cfg['robot']['name']
    AO.COLLISION_TAG = cfg['sim']['collision']['mode']
    seed = AO.canon({**cfg['tuning'], **AO.BASE, 'ik_seeds':4})
    cache = {}
    tools = AgentTools(cache, AO.SPACE, cfg)
    decisions = []
    evaluations = []
    original_evaluate = tools.evaluate
    def evaluate(params):
        start = time.perf_counter()
        before = tools.n_eval
        met = original_evaluate(params)
        evaluations.append(dict(params=params, metrics=met, score=AO.score(met),
                                elapsed_s=time.perf_counter()-start,
                                cache_hit=tools.n_eval == before))
        (out / (a.arm + '_evaluations.json')).write_text(json.dumps(evaluations, indent=2))
        (out / (a.arm + '_cache.json')).write_text(json.dumps(cache, indent=2))
        return met
    tools.evaluate = evaluate
    def decide(prompt):
        start = time.perf_counter()
        response = R.vlm_call(prompt, [])
        decisions.append(dict(prompt=prompt, response=response,
                              elapsed_s=time.perf_counter()-start))
        (out / (a.arm + '_decisions.json')).write_text(json.dumps(decisions, indent=2))
        return R.parse_json(response)
    A.vlm_decide = decide
    start = time.perf_counter()
    report = dict(arm=a.arm, budget=a.budget, budget_includes_seed=True, ik_seeds=4,
                  model=R.MODEL, endpoint=R.ENDPOINT, temperature=0.0,
                  max_tokens=700, seed=seed, cfg=cfg,
                  score_contains_gt_in_both_arms=True,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(A.__file__), Path(AO.__file__), Path(PIPE.G.__file__))})
    try:
        if not ensure_vlm():
            raise RuntimeError('VLM unavailable; refuse to label greedy fallback as a VLM A/B')
        report['model_startup_s'] = time.perf_counter() - start
        search_start = time.perf_counter()
        best = A.agent_loop(seed, tools, a.budget, include_gt=a.arm == 'with_gt', strict_vlm=True)
        if tools.n_eval != a.budget:
            raise RuntimeError(f'incomplete budget: {tools.n_eval}/{a.budget}')
        report.update(status='complete', best=best, best_score=AO.score(best['metrics']),
                      search_s=time.perf_counter()-search_start)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report.update(elapsed_s=time.perf_counter()-start, n_eval=tools.n_eval,
                      actions=tools.log, evaluations=evaluations, decisions=decisions)
        report_path.write_text(json.dumps(report, indent=2))
        release_vlm()
    print('AB_RESULT', json.dumps({k:report[k] for k in ('arm','status','n_eval','best_score','elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()

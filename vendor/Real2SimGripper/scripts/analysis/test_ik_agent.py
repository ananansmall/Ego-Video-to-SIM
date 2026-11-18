"""Small regression check: python scripts/analysis/test_ik_agent.py (no physics/VLM calls)."""
import ast
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get('IK_AGENT_SCRIPTS', str(ROOT / 'scripts')))
import agentic_agent as A
import agentic_optimize as AO
from agent_tools import AgentTools
import task_config as TC
G = AO.G


def main():
    for value in (4, 8, 16):
        assert G.validate_ik_seeds(value) == value
        TC.apply_task(G, {'tuning': {'ik_seeds': value}}, verbose=False)
        assert G.PARAM_DEFAULTS['ik_seeds'] == value
    for value in (0, -1, 4.5, True, '4', None):
        try:
            G.validate_ik_seeds(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    TC.apply_task(G, {}, verbose=False)
    assert G.PARAM_DEFAULTS['ik_seeds'] == 16
    tree = ast.parse(Path(G.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'solve_ik']
    assert len(calls) == 3 and all(any(k.arg == 'n_seeds' for k in n.keywords) for n in calls)

    seed = AO.canon({**AO.BASE, 'ik_seeds': 4})
    assert len({AO.key_of({**seed, 'ik_seeds': n}) for n in (4, 8, 16)}) == 3
    assert AO.key_of({}) == AO.key_of({'ik_seeds': 16})
    with patch.object(AO, 'TASK_CFG', {'tuning': {'ik_seeds': 8}}):
        assert AO.key_of({}) == AO.key_of({'ik_seeds': 8})
        assert AO.key_of({'ik_seeds': 4}) != AO.key_of({})
    AO.TASK_CFG = {'tuning': {'ik_seeds':4}}
    met = dict(pass_eff=True, on_plate=True, gt_dist=0.4, gt_x_cm=1.2, gt_y_cm=10.3,
               follow_max_mm=81.0, plate_offset_mm=103.0, plate_margin_mm=28.0,
               duck_final=[0.1, 0.2, 0.8], knock_mm=12.0)
    rec = dict(params=seed, metrics=met, score=AO.score(met), robot=AO.ROBOT_NAME,
               collision=AO.COLLISION_TAG)
    cache = {'obsolete-code': copy.deepcopy(rec), AO.key_of(seed): copy.deepcopy(rec)}
    other = {**rec, 'params':{**seed,'ik_seeds':16}, 'score':9999.0}
    cache[AO.key_of(other['params'])] = other
    tools = AgentTools(cache, cfg={})
    assert len(tools.query_history()) == 1 and len(tools.tried_configs()) == 1
    assert AO.best_cached({'obsolete-code': rec}) is None
    assert AO.best_cached(cache)['params']['ik_seeds'] == 4
    assert any(c['moved'].startswith('base_xy') for c in tools.propose_neighbors(seed))
    tools.log_action({'tool':'query_history'}, {'nested':[{'gt_dist':0.4,'gt_x_cm':1.2}]})
    before = copy.deepcopy(tools.log)
    hidden = A.build_prompt({'params':seed,'metrics':met}, tools, 4, include_gt=False)
    visible = A.build_prompt({'params':seed,'metrics':met}, tools, 4, include_gt=True)
    assert 'gt_' not in hidden
    assert all(k in visible for k in ('gt_dist','gt_x_cm','gt_y_cm'))
    assert tools.log == before

    # Constant repeated/invalid/free actions must make progress or terminate within the budget.
    def fake_run(params, collect_frames=False):
        return None, dict(met, gt_dist=abs(params.get('place_t', 0.6)))
    for act in ({'tool':'evaluate','params':{}}, {'tool':'query_history'},
                {'tool':'render'}, {'tool':'nonsense'}, {'tool':'propose_neighbors','params':{'place_t':0.6}}):
        tools = AgentTools({}, cfg={})
        with patch.object(AO, '_run', fake_run), patch.object(A, 'vlm_decide', return_value=act):
            A.agent_loop(seed, tools, 5, strict_vlm=True)
        assert tools.n_eval == 5, (act, tools.n_eval)
    tools = AgentTools({}, cfg={})
    with patch.object(AO, '_run', fake_run):
        tools.evaluate(seed)
        tools.evaluate(seed)
    assert tools.n_eval == 1
    print('PASS: IK validation/propagation, cache separation, GT redaction, base candidates, bounded budgets')


if __name__ == '__main__':
    main()

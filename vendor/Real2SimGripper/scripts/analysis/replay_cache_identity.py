"""Check cache context and replay saved decisions/metrics without physics or model calls."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get('IK_AGENT_SCRIPTS', str(ROOT / 'scripts')))
import agentic_agent as A
import agentic_optimize as AO
from agent_tools import AgentTools
import agentic_replay as R
import pipeline as PIPE


def context_check():
    cfg = {'_path': 'one.json', 'tuning': {'ik_seeds': 4},
           'scene': {'glb': 'one.glb'}, 'sim': {'collision': {'mode': 'coacd'}},
           'demonstration_ground_truth': {'normalised_offset': [-0.06, 0.06]}}
    state = {'threshold': None, 'max_depenetration_velocity': 1.0}
    with patch.object(AO, 'TASK_CFG', cfg), patch.object(AO, 'COLLISION_TAG', 'coacd'), \
            patch('collision_config.state', return_value=state):
        first = AO.key_of({})
        assert first == AO.key_of({'ik_seeds': 4})
        cfg['_path'] = 'two.json'
        assert AO.key_of({}) == first
        cfg['sim']['collision']['threshold'] = 0.05
        assert AO.key_of({}) == first
        cfg['sim']['collision']['threshold'] = 0.20
        assert AO.key_of({}) != first
        del cfg['sim']['collision']['threshold']
        for field, value in [('threshold', 0.20), ('max_depenetration_velocity', None)]:
            old = state[field]
            state[field] = value
            assert AO.key_of({}) != first
            state[field] = old
        for field, value in [('scene', {'glb': 'two.glb'}),
                             ('demonstration_ground_truth', {'normalised_offset': [0.0, 0.0]}),
                             ('object', {'mass_kg': 0.08})]:
            changed = copy.deepcopy(cfg)
            changed[field] = value
            with patch.object(AO, 'TASK_CFG', changed):
                assert AO.key_of({}) != first
        assert AO.key_of({'place_lift': 0.3}) != first
        assert AO.key_of({}) == AO.key_of({'place_lift': AO.G.PARAM_DEFAULTS['place_lift']})


def replay(folder, arms=('without_gt', 'with_gt', 'coordinate')):
    checks = {}
    for arm in arms:
        report = json.loads((folder / (arm + '.json')).read_text())
        cfg = report['cfg']
        PIPE.apply_adapters(cfg)
        AO.TASK_CFG, AO.ROBOT_NAME = cfg, cfg['robot']['name']
        AO.COLLISION_TAG = cfg['sim']['collision']['mode']
        rows = report.get('evaluations', report.get('history'))
        index = 0
        decisions = iter(report.get('decisions', []))
        prompt_count = 0

        def measured_run(params, collect_frames=False):
            nonlocal index
            assert not collect_frames
            row = rows[index]
            assert AO.canon(params, keys=AO.G.PARAM_DEFAULTS) == AO.canon(row['params'], keys=AO.G.PARAM_DEFAULTS), (arm, index)
            index += 1
            return None, copy.deepcopy(row['metrics'])

        def recorded_decision(prompt):
            nonlocal prompt_count
            record = next(decisions)
            assert prompt == record['prompt'], (arm, prompt_count, 'prompt drift')
            prompt_count += 1
            return R.parse_json(record['response'])

        cache = {}
        with patch.object(AO, '_run', measured_run), patch.object(AO, 'save_cache'), \
                patch.object(A, 'vlm_decide', recorded_decision):
            if arm == 'coordinate':
                best, history = AO.pattern_search(report['seed'], cache, AO.SPACE, budget=4, from_seed=True)
                assert len(history) == 5
            else:
                tools = AgentTools(cache, AO.SPACE, cfg)
                best = A.agent_loop(report['seed'], tools, 5, include_gt=arm == 'with_gt', strict_vlm=True)
                assert tools.n_eval == 5 and prompt_count == len(report['decisions'])
        assert index == len(rows) == 5
        assert AO.canon(best['params']) == AO.canon(report['best']['params'])
        assert best['metrics'] == report['best']['metrics']
        assert AO.score(best['metrics']) == AO.score(report['best']['metrics'])
        checks[arm] = dict(evaluations=index, exact_prompts=prompt_count,
                           best_metrics_identical=True, score=AO.score(best['metrics']))
    return checks


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', type=Path)
    args = ap.parse_args()
    context_check()
    result = {'cache_context': 'PASS'}
    if args.trace_dir:
        result['replay'] = replay(args.trace_dir)
        (args.trace_dir / 'release_trace_check.json').write_text(json.dumps(result, indent=2))
    print('PASS:', json.dumps(result))

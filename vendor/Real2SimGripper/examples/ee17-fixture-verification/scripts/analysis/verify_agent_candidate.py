"""Replay the GT arm's selected search candidate using accurate IK, without cache."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import pipeline as PIPE
import agentic_optimize as AO


def main():
    folder = ROOT / 'results/ik_agent_20261022'
    arm = json.loads((folder / 'with_gt.json').read_text())
    control = json.loads((folder / 'without_gt.json').read_text())
    assert arm['status'] == 'complete' and arm['n_eval'] == arm['budget']
    assert control['status'] == 'complete' and control['n_eval'] == control['budget']
    for key in ('seed', 'cfg', 'budget', 'model', 'temperature', 'max_tokens', 'source_sha256'):
        assert arm[key] == control[key], f'A/B mismatch: {key}'
    current_sha = {name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                   for name in arm['source_sha256']}
    assert current_sha == arm['source_sha256'], 'Experiment sources changed before replay'
    cfg = arm['cfg']
    cfg['tuning']['ik_seeds'] = 16
    params = {**arm['best']['params'], 'ik_seeds':16}
    start = time.perf_counter()
    met = PIPE.run(cfg, params=params, collect_frames=False)['metrics']
    report = dict(params=params, metrics=met, score=AO.score(met),
                  search_metrics=arm['best']['metrics'], elapsed_s=time.perf_counter()-start,
                  solver_sha256=hashlib.sha256(Path(PIPE.G.__file__).read_bytes()).hexdigest())
    (folder / 'with_gt_accurate.json').write_text(json.dumps(report, indent=2))
    AO.TASK_CFG = cfg
    AO.ROBOT_NAME = cfg['robot']['name']
    AO.COLLISION_TAG = cfg['sim']['collision']['mode']
    cache = AO.load_cache()
    for name in ('without_gt_cache.json', 'with_gt_cache.json'):
        measured = json.loads((folder / name).read_text())
        assert all(k == AO.key_of(v['params']) for k,v in measured.items())
        cache.update(measured)
    cache[AO.key_of(params)] = dict(params=params, metrics=met, score=report['score'],
                                   secs=report['elapsed_s'], tag='gt-ab-accurate',
                                   robot=AO.ROBOT_NAME, collision=AO.COLLISION_TAG)
    AO.save_cache(cache)
    verified = dict(best_params=params, best_metrics=met, metadata=met, best_score=report['score'],
                    validation_status='PASS' if met['pass_eff'] else 'FAIL',
                    source='results/ik_agent_20261022/with_gt_accurate.json', ik_seeds=16,
                    search_budget=arm['budget'], search_n_eval=arm['n_eval'])
    for name in ('agentic_optimize_best.json', 'agentic_agent_best.json'):
        path = ROOT / 'results' / name
        backup = folder / ('before_' + name)
        if path.exists() and not backup.exists():
            shutil.copyfile(path, backup)
        path.write_text(json.dumps(verified, indent=2))
    print('ACCURATE_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

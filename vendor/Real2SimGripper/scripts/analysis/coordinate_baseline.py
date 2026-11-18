"""Budget-matched coordinate-search control for the agent-loop experiment."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import agentic_agent as A
import agentic_optimize as AO
import pipeline as PIPE


def main():
    out = ROOT / 'results/ik_agent_20261022'
    control = json.loads((out/'without_gt.json').read_text())
    cfg, seed = control['cfg'], control['seed']
    assert control['status']=='complete' and control['n_eval']==5
    sha = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (Path(A.__file__),Path(AO.__file__),Path(PIPE.G.__file__))}
    assert sha == control['source_sha256']
    AO.CACHE_JSON = str(out/'coordinate_cache.json')
    if Path(AO.CACHE_JSON).exists():
        raise FileExistsError(AO.CACHE_JSON)
    PIPE.apply_adapters(cfg)
    AO.TASK_CFG = cfg
    AO.ROBOT_NAME = cfg['robot']['name']
    AO.COLLISION_TAG = cfg['sim']['collision']['mode']
    cache = {}
    start = time.perf_counter()
    best, history = AO.pattern_search(seed,cache,AO.SPACE,budget=4,from_seed=True)
    assert len(history)==5 and len(cache)==5, 'Control must use five unique full episodes'
    report = dict(status='complete',budget=5,budget_includes_seed=True,n_eval=len(history),
                  seed=seed,cfg=cfg,source_sha256=sha,search_s=time.perf_counter()-start,
                  best=best,history=history,mode='coordinate',ik_seeds=4)
    (out/'coordinate.json').write_text(json.dumps(report,indent=2))
    AO.CACHE_JSON = str(ROOT/'results/agentic_optimize_cache.json')
    merged = AO.load_cache()
    merged.update(cache)
    AO.save_cache(merged)
    print('COORDINATE_RESULT',json.dumps(dict(score=best['score'],metrics=best['metrics'],
                                           elapsed_s=report['search_s'])),flush=True)


if __name__ == '__main__':
    main()

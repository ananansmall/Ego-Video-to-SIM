"""Instrument the existing solver on the real grasp frame; compare threshold on/off."""
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pipeline as PIPE


class ProbeFinished(Exception):
    pass


def main():
    cfg = PIPE.load_task(str(ROOT / 'configs/duck_to_plate.json'))
    source = inspect.getsource(PIPE.G.solve_ik)
    source = source.replace('    best, bx = None, x0.copy()',
                            '    seed_stats = []\n    best, bx = None, x0.copy()')
    source = source.replace('    for s in seeds:', '    for s in seeds:\n        iterations = 0')
    source = source.replace('            continue\n        for _ in range(1500):',
                            '            seed_stats.append(dict(iterations=0, cost=float(c), early=True))\n'
                            '            continue\n        for _ in range(1500):')
    source = source.replace('            g = np.zeros(6)',
                            '            iterations += 1\n            g = np.zeros(6)')
    source = source.replace('\n        if best is None or c < best:\n',
                            '\n        seed_stats.append(dict(iterations=iterations, cost=float(c), early=bool(c < IK_CONV_TOL)))\n'
                            '        if best is None or c < best:\n')
    # Only the post-loop comparison has eight spaces; the pre-loop one has twelve.
    assert 'iterations=iterations' in source and 'iterations += 1' in source
    source = source.replace('    return bx, best', '    return bx, best, seed_stats')
    ns = dict(PIPE.G.__dict__)
    exec(compile(source, '<instrumented-existing-ik>', 'exec'), ns)
    solver = ns['solve_ik']
    rows = []
    def probe(*args, **kwargs):
        for seeds in (4, 8, 16):
            sols = []
            for tol in (1e-6, 0.0):
                ns['IK_CONV_TOL'] = tol
                start = time.perf_counter()
                x, c, stats = solver(*args, **{**kwargs, 'n_seeds':seeds})
                sols.append(x)
                rows.append(dict(seeds=seeds, tol=tol, cost=float(c),
                                 elapsed_s=time.perf_counter()-start, per_seed=stats))
            assert np.array_equal(*sols), f'early threshold changed grasp solution at {seeds}'
        path = ROOT / 'results/ik_agent_20261022/convergence_probe.json'
        path.write_text(json.dumps(rows, indent=2))
        print('PROBE_RESULT', json.dumps(rows), flush=True)
        raise ProbeFinished()
    PIPE.G.solve_ik = probe
    try:
        PIPE.run(cfg, params=dict(base_xy=[0.30,-0.38], place_t=0.0,
                 time_scale=3.0, place_hover=0.07, grip_open_q=0.05), collect_frames=False)
    except ProbeFinished:
        pass


if __name__ == '__main__':
    main()

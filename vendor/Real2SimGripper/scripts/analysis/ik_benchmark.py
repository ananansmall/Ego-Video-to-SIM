"""Run one uncached full episode; override seed count without editing solver."""
import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import pipeline as PIPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, required=True)
    ap.add_argument('--no-early-stop', action='store_true')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    # A queued and a manual invocation may request the same expensive experiment.
    output_lock = open(a.output + '.lock', 'a')
    fcntl.flock(output_lock, fcntl.LOCK_EX)
    cfg = PIPE.load_task(str(ROOT / 'configs/duck_to_plate.json'))
    params = dict(plate_geom='geometry_8', time_scale=3.0, grip_open_q=0.05,
                  place_t=0.0, place_hover=0.07, base_xy=[0.30, -0.38])
    params['ik_seeds'] = a.seeds
    solver_sha = hashlib.sha256(Path(PIPE.G.__file__).read_bytes()).hexdigest()
    if Path(a.output).exists():
        previous = json.loads(Path(a.output).read_text())
        assert (previous['seeds'], previous['early_stop'], previous['params'],
                previous['config'], previous['solver_sha256']) == (
                    a.seeds, not a.no_early_stop, params, cfg, solver_sha), 'Existing result differs'
        print('ALREADY_COMPLETE', a.output, flush=True)
        return
    original = PIPE.G.solve_ik
    timings = []
    costs = []
    default_tol = PIPE.G.IK_CONV_TOL
    def timed(*args, **kwargs):
        kwargs['n_seeds'] = a.seeds
        start = time.perf_counter()
        result = original(*args, **kwargs)
        timings.append(time.perf_counter() - start)
        costs.append(float(result[1]))
        if len(timings) % 25 == 0:
            print(f'BENCH seeds={a.seeds} frames={len(timings)} ik_s={sum(timings):.1f}', flush=True)
        return result
    PIPE.G.solve_ik = timed
    if a.no_early_stop:
        PIPE.G.IK_CONV_TOL = -float('inf')
    start = time.perf_counter()
    result = PIPE.run(cfg, params=params, collect_frames=False)
    report = dict(seeds=a.seeds, early_stop=not a.no_early_stop,
                  elapsed_s=time.perf_counter()-start, ik_s=sum(timings),
                  ik_frames=len(timings), ik_frame_s=timings, params=params,
                  ik_costs=costs, default_convergence_tol=default_tol,
                  threshold_never_reached=bool(costs) and all(c >= default_tol for c in costs),
                  metrics=result['metrics'], config=cfg,
                  solver_sha256=solver_sha)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print('BENCH_RESULT', json.dumps({k:v for k,v in report.items() if k not in ('config','ik_frame_s','ik_costs')}), flush=True)


if __name__ == '__main__':
    main()

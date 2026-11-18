"""Build the held-object desk-exit short test packages (F0–155).

Both exit-route variants share run A's frozen approach, grasp (F24–36), lift
arc, pad and object trajectory and controller setup; only `base_reference_se2`
changes.  Variant A keeps run A's direct diagonal exit; variant B splices run
B's east-south detour lane (F42–151) into the identical pad/object trajectory.
"""
import json
from pathlib import Path
from shutil import copy2
import sys

import numpy as np

RUN_A = Path(__file__).resolve().parent / 'chair-full-luna-20261116'
RUN_B = Path(__file__).resolve().parent / 'chair-full-luna-20261116-route-b'
N = 199  # F0..198: grasp, desk-exit transit, wood seat, load-bearing release F156-168, verified wood anchor F198


def load_package(run):
    package = run / 'package'
    with np.load(package / 'paired_targets.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, package


def truncate(data, n):
    out = {}
    for key, value in data.items():
        if hasattr(value, 'shape') and value.ndim and value.shape[0] == len(data['src_frames']):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def write_variant(out_dir, arrays, geometry_src, package_src):
    out = out_dir / 'package'
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'paired_targets.npz', **arrays)
    geometry = json.loads((geometry_src / 'geometry.json').read_text())
    # The short window has no release or regrasp commands; drop the release
    # metadata so the runner does not index frames outside the truncation.
    mobile = geometry.pop('mobile_release', {})
    if mobile.get('scene_asset'):
        geometry['scene_asset'] = mobile['scene_asset']
    geometry.pop('multi_release_events', None)
    (out / 'geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    for name in ('case_metric.obj', 'case_collision.obj', 'material.mtl', 'material_0.png'):
        copy2(package_src / name, out / name)


def main():
    a_data, a_pkg = load_package(RUN_A)
    b_data, b_pkg = load_package(RUN_B)
    frames = np.asarray(a_data['src_frames'], int)
    if frames[0] != 0 or len(frames) < N:
        raise ValueError('run A package does not start at F0 or is too short')
    base_a = np.asarray(a_data['base_reference_se2'], float)
    base_b = np.asarray(b_data['base_reference_se2'], float)
    if not np.allclose(base_a[:42], base_b[:42], atol=1e-9):
        raise ValueError('route packages disagree before F42; the same-start premise fails')
    if not np.allclose(base_a[151], base_b[151], atol=1e-9):
        raise ValueError('route lanes do not converge at the wood station F151')

    variant_a = truncate(a_data, N)
    variant_b = truncate(a_data, N)
    variant_b['base_reference_se2'] = base_b[:N].copy()

    out_a = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'runs/exit-route-a')
    out_b = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / 'runs/exit-route-b')
    write_variant(out_a, variant_a, a_pkg, a_pkg)
    write_variant(out_b, variant_b, a_pkg, a_pkg)
    moved = int(np.sum(np.linalg.norm(variant_b['base_reference_se2'][:, :2]
                                      - variant_a['base_reference_se2'][:, :2], axis=1) > 1e-9))
    print(json.dumps({'frames': N, 'route_b_changed_frames': moved,
                      'out_a': str(out_a.resolve()), 'out_b': str(out_b.resolve())}, indent=2))


if __name__ == '__main__':
    main()

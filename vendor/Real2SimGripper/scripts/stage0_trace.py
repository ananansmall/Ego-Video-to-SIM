"""Build the frozen Stage 0 trace and manifest from a sample config.

The sample directory provides only input configuration, source evidence and run
artifacts; frame numbers, object identity, supports, stance and grasp pose are
read from the sample data referenced by the config.  This module owns the
stage0_trace.npz schema consumed by scripts/stage0_render.py and
scripts/unified_trajectory_report.py.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

TRACE_KEYS = ("src_frames", "video_fps", "playback_fps", "base_se2", "T_world_pad",
              "grip", "T_world_object", "object_visible")
FLAG_KEYS = ("hand_observed", "hand_filled", "object_observed", "object_filled",
             "object_unavailable")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo):
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def base_from_camera(t_c2w, R_c2w, pad_xy, sigma, offset_back):
    """Stance reference from the reconstructed wearer camera: smoothed camera xy,
    heading facing the observed hand, offset back along the heading (same rule as
    ep16 build_stage0_trace)."""
    base_xy = gaussian_filter1d(t_c2w[:, :2], sigma=sigma, axis=0, mode='nearest')
    ray = pad_xy - base_xy
    forward = R_c2w[:, :2, 2]
    short = np.linalg.norm(ray, axis=1) < .10
    ray[short] = forward[short]
    yaw = gaussian_filter1d(np.unwrap(np.arctan2(ray[:, 1], ray[:, 0])), sigma=sigma, mode='nearest')
    base_xy -= offset_back * np.column_stack([np.cos(yaw), np.sin(yaw)])
    return np.column_stack([base_xy, yaw])


def _as_bool(values, label):
    values = np.asarray(values)
    if values.dtype == bool:
        return values
    if values.dtype.kind in 'iuf':
        return values.astype(float) > .5
    raise ValueError(f'{label}: expected a boolean flag array')


def validate_trace(t):
    """Shared stage0_trace.npz validation (strict shapes, finiteness, frames)."""
    required = TRACE_KEYS + ('hand_observed', 'object_observed')
    missing = [k for k in required if k not in t]
    if missing:
        raise ValueError(f'trace missing {missing}')
    n = len(t['src_frames'])
    if n < 1 or t['src_frames'].dtype.kind not in 'iu' or not np.isfinite(t['src_frames']).all() \
            or np.any(np.diff(t['src_frames']) <= 0):
        raise ValueError('src_frames must be nonempty, finite, and strictly increasing integers')
    for key, shape in (('base_se2', (n, 3)), ('T_world_pad', (n, 4, 4)), ('grip', (n, 2)),
                       ('T_world_object', (n, 4, 4)), ('object_visible', (n,)),
                       ('hand_observed', (n,)), ('object_observed', (n,))):
        if t[key].shape != shape:
            raise ValueError(f'{key} shape {t[key].shape}, expected {shape}')
    if not np.isfinite(float(t['video_fps'])) or not np.isfinite(float(t['playback_fps'])) \
            or min(float(t['video_fps']), float(t['playback_fps'])) <= 0:
        raise ValueError('video_fps and playback_fps must be positive')
    for key in ('base_se2', 'T_world_pad', 'grip', 'T_world_object'):
        if not np.isfinite(t[key]).all():
            raise ValueError(f'{key} contains non-finite values')
    if np.any(t['grip'] < 0) or np.any(t['grip'] > 0.05):
        raise ValueError('physical finger positions must be in [0, 0.05] metres')
    for key in ('object_visible', 'hand_observed', 'object_observed'):
        t[key] = _as_bool(t[key], key)
    t['hand_filled'] = ~t['hand_observed']
    t['object_filled'] = t['object_visible'] & ~t['object_observed']
    t['object_unavailable'] = ~t['object_visible']
    return t


PATH_KEYS = ('video', 'pair_npz', 'trace_npz', 'scene_points', 'asset_dir', 'geometry',
             'object_mesh', 'provenance', 'calibration_reference')


def load_config(path, repo):
    config = json.loads(Path(path).read_text())
    def walk(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key in PATH_KEYS and isinstance(value, str):
                    candidate = Path(value)
                    out[key] = str(candidate if candidate.is_absolute() else repo / candidate)
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(config)


def build_trace(config):
    """Assemble the validated trace dict plus per-field provenance."""
    if 'trace_npz' in config.get('retarget', {}):
        source = config['retarget']['trace_npz']
        with np.load(source, allow_pickle=False) as z:
            t = {k: z[k] for k in z.files}
        provenance = {'kind': 'frozen_trace', 'trace_npz': source,
                      'trace_sha256': sha256(source)}
        if 'object_ids' in config['retarget']:
            t['object_ids'] = np.full(len(t['src_frames']), config['retarget']['object_id'])
    else:
        mapping = config['retarget']
        source = mapping['pair_npz']
        with np.load(source, allow_pickle=False) as z:
            available = set(z.files)
            absent = [k for k in (mapping['map'].values()) if k not in available]
            if absent:
                raise ValueError(f'{source}: pair is missing keys {absent}')
            n = len(z['src_frames'])
            frames = np.asarray(z['src_frames'])
            t = {
                'src_frames': frames,
                'video_fps': np.asarray(z[mapping.get('video_fps', 'video_fps')]),
                'T_world_pad': np.asarray(z[mapping['map']['T_world_pad']], float),
                'grip': np.asarray(z[mapping['map']['grip']], float),
                'T_world_object': np.asarray(z[mapping['map']['T_world_object']], float),
                'hand_observed': _as_bool(z[mapping['map']['hand_observed']], 'hand_observed'),
            }
            visible_key = mapping['map'].get('object_visible')
            t['object_visible'] = _as_bool(z[visible_key], 'object_visible') if visible_key \
                else np.ones(n, bool)
            anchors = mapping['map'].get('object_observed_anchors')
            if anchors is not None:
                anchor_frames = np.atleast_1d(np.asarray(z[anchors]).ravel())
                t['object_observed'] = np.isin(frames, anchor_frames)
                t['object_depth_anchors'] = anchor_frames.astype(int)
            else:
                observed_key = mapping['map'].get('object_observed')
                t['object_observed'] = _as_bool(z[observed_key], 'object_observed') if observed_key \
                    else t['object_visible'].copy()
            for extra in ('grip_states', 'grip_state_names'):
                if extra in z.files:
                    t[extra] = z[extra]
            stride = int(mapping.get('stride', 1))
            if stride < 1:
                raise ValueError('retarget.stride must be >= 1')
            t['playback_fps'] = float(t['video_fps']) / stride
            camera = config.get('base', {}).get('from_camera')
            if camera:
                t['base_se2'] = base_from_camera(
                    np.asarray(z[camera['t_c2w']], float), np.asarray(z[camera['R_c2w']], float),
                    t['T_world_pad'][:, :2, 3], float(camera.get('smooth_sigma', 1.6)),
                    float(camera.get('offset_back_m', .12)))
            explicit = config.get('base', {}).get('base_se2')
            if explicit is not None and 'base_se2' not in t:
                t['base_se2'] = np.asarray(explicit, float).reshape(n, 3)
        provenance = {'kind': 'retarget_pair', 'pair_npz': source,
                      'pair_sha256': sha256(source),
                      'field_map': {k: v for k, v in mapping['map'].items()}}
    object_track = config.get('retarget', {}).get('object_track')
    if object_track:
        object_path = object_track['pair_npz']
        with np.load(object_path, allow_pickle=False) as obj:
            object_frames = obj['src_frames']
            if (object_frames.ndim != 1 or object_frames.dtype.kind not in 'iu'
                    or np.any(np.diff(object_frames) <= 0)):
                raise ValueError(f'{object_path}: invalid object source frames')
            _, target_rows, object_rows = np.intersect1d(
                t['src_frames'], object_frames, return_indices=True)
            if not len(target_rows):
                raise ValueError(f'{object_path}: object and retarget source frames do not overlap')
            t['T_world_object'] = np.repeat(np.eye(4)[None], len(t['src_frames']), axis=0)
            t['T_world_object'][target_rows] = obj['T_world_object'][object_rows]
            t['object_visible'] = np.isin(t['src_frames'], object_frames)
            t['object_observed'] = t['object_visible'] & np.isin(
                t['src_frames'], obj['source_anchor_frames'])
            t['object_depth_anchors'] = t['src_frames'][t['object_observed']]
        provenance['object_track'] = {'pair_npz': object_path, 'pair_sha256': sha256(object_path),
                                      'shared_source_frame_rows': int(len(target_rows))}
    validate_trace(t)
    if 'object_ids' not in t and config.get('retarget', {}).get('object_id'):
        t['object_ids'] = np.full(len(t['src_frames']), config['retarget']['object_id'])
    return t, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='sample input configuration JSON')
    parser.add_argument('--out', required=True, help='output directory (…/stage0)')
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    config = load_config(args.config, repo)
    t, provenance = build_trace(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'stage0_trace.npz', **t)
    frames = t['src_frames']
    video = config.get('video')
    manifest = {
        'source_video': video,
        'source_video_sha256': sha256(video) if video and Path(video).is_file() else None,
        'git_commit': git_commit(repo),
        'schema': 'stage0_trace_v1',
        'frame_mapping': {'first': int(frames[0]), 'last': int(frames[-1]), 'rows': int(len(frames)),
                          'video_fps': float(t['video_fps']), 'playback_fps': float(t['playback_fps']),
                          'source_frames_are_sampled': bool(np.any(np.diff(frames) != 1))},
        'retarget': provenance,
        'flags': {key: int(np.asarray(t[key]).sum()) for key in FLAG_KEYS},
        'segments': {'names': [str(s) for s in t['grip_state_names']],
                     'counts': np.bincount(t['grip_states'], minlength=len(t['grip_state_names'])).tolist()}
                     if 'grip_states' in t else None,
        'gravity': config.get('gravity'),
        'assets': {'scene': config.get('scene'), 'object_mesh': config.get('object_mesh'),
                   'object_mesh_sha256': sha256(config['object_mesh'])
                   if config.get('object_mesh') and Path(config['object_mesh']).is_file() else None},
        'trace': str((out / 'stage0_trace.npz').resolve()),
    }
    if config.get('provenance') and Path(config['provenance']).is_file():
        manifest['provenance'] = json.loads(Path(config['provenance']).read_text())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': int(len(frames)), 'flags': manifest['flags'],
                      'first': manifest['frame_mapping']['first'],
                      'last': manifest['frame_mapping']['last']}, indent=2))


if __name__ == '__main__':
    main()

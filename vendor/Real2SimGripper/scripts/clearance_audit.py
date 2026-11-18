"""Per-frame whole-body and held-object clearance audit for a physics trace.

Consumes an exec_trace.npz (+ optional robot_contact_audit.json next to it) and
the sample package (geometry.json support names + scene_assets meshes), and
records per source frame: minimum whole-body distance to each support, held
object distance to supports and to the robot (grippers excluded), gripper slip
of the held object, and the real PhysX contact pairs.  Generic; frame numbers,
support names and meshes come from the sample package.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from sapien.core import Pose
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cb_large_range_sim import _build_scene, bp  # noqa: E402

GRIPPER_LINKS = ("right_gripper_link", "right_gripper_finger_link1", "right_gripper_finger_link2",
                 "left_gripper_link", "left_gripper_finger_link1", "left_gripper_finger_link2")
MAX_SHAPE_POINTS = 48


def subsample(points, cap=MAX_SHAPE_POINTS):
    if len(points) <= cap:
        return points
    idx = np.linspace(0, len(points) - 1, cap).astype(int)
    return points[idx]


def shape_vertices(sh):
    """Sample points on a collision shape in its local frame (mesh, convex, box,
    sphere or capsule)."""
    kind = type(sh).__name__
    if hasattr(sh, 'get_vertices'):
        v = np.asarray(sh.get_vertices(), dtype=float)
        return v if (v.ndim == 2 and v.shape[1] == 3 and len(v)) else None
    if kind.endswith('Box'):
        h = np.asarray(sh.get_half_size(), float)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return signs * h
    if kind.endswith('Sphere'):
        r = float(sh.radius)
        axes = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])
        return np.vstack([[0, 0, 0], axes])
    if kind.endswith('Capsule'):
        r, hl = float(sh.radius), float(sh.half_length)
        axis = np.linspace(-hl, hl, 5)
        pts = [np.array([0, 0, a]) for a in axis]
        pts += [np.array([np.cos(t) * r, np.sin(t) * r, a]) for a in (-hl, hl) for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        return np.asarray(pts)
    return None


def link_shape_points(links, skip_names=()):
    """Cache collision-shape vertices in each link's local frame."""
    cache = {}
    for link in links:
        name = link.entity.get_name()
        if name in skip_names:
            continue
        shapes = []
        for sh in link.get_collision_shapes():
            v = shape_vertices(sh)
            if v is None:
                continue
            local = sh.get_local_pose().to_transformation_matrix()
            shapes.append((subsample(v), local))
        if shapes:
            cache[name] = (link, shapes)
    return cache


def shape_world(points, local, link_pose):
    return (link_pose @ local @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def _point_segment_sq(p, s0, s1):
    d = s1 - s0
    tt = np.clip(((p - s0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-18), 0, 1)
    q = s0 + tt[:, None] * d
    return ((q - p) ** 2).sum(1)


def point_triangle_distance_sq(p, tris):
    """Exact squared distance from point p to each triangle (m,3,3)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b - a, c - a
    ap = p - a
    plane = np.cross(ab, ac)
    nn = np.maximum((plane * plane).sum(1), 1e-18)
    q = a + (((ap * plane).sum(1) / nn)[:, None] * plane)
    aq = q - a
    # barycentric coordinates of the plane projection
    dot_ab_ab = (ab * ab).sum(1); dot_ab_ac = (ab * ac).sum(1); dot_ac_ac = (ac * ac).sum(1)
    rhs0 = (ab * aq).sum(1); rhs1 = (ac * aq).sum(1)
    det = np.maximum(dot_ab_ab * dot_ac_ac - dot_ab_ac ** 2, 1e-18)
    s = (rhs0 * dot_ac_ac - rhs1 * dot_ab_ac) / det
    t = (dot_ab_ab * rhs1 - dot_ab_ac * rhs0) / det
    inside = (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)
    face_sq = ((q - p) ** 2).sum(1)
    edge_sq = np.minimum(np.minimum(_point_segment_sq(p, a, b), _point_segment_sq(p, b, c)),
                         _point_segment_sq(p, c, a))
    return np.where(inside, np.minimum(face_sq, edge_sq), edge_sq)


class MeshQuery:
    """Exact-enough point-to-mesh distance without rtree: k nearest vertices,
    then exact point-triangle distance over their incident triangles."""

    def __init__(self, mesh, k=12):
        self.mesh = mesh
        self.kdt = cKDTree(mesh.vertices)
        self.k = min(k, len(mesh.vertices))
        faces = np.asarray(mesh.faces, int)
        v2t = [[] for _ in mesh.vertices]
        for ti, tri in enumerate(faces):
            for v in tri:
                v2t[v].append(ti)
        self.v2t = [np.asarray(t, int) for t in v2t]
        self.triangles = np.asarray(mesh.triangles, float)
        self.bounds = mesh.bounds

    def min_distance(self, points, margin=0.05):
        lo, hi = self.bounds
        if np.any(points.min(0) - margin > hi) or np.any(points.max(0) + margin < lo):
            return np.inf
        _, idx = self.kdt.query(points, k=self.k)
        idx = np.atleast_2d(idx)
        best = np.inf
        for p, cand in zip(points, idx):
            cand = cand[cand < len(self.v2t)]
            tris = np.unique(np.concatenate([self.v2t[c] for c in cand])) if len(cand) else []
            if not len(tris):
                continue
            best = min(best, float(np.sqrt(point_triangle_distance_sq(p, self.triangles[tris]).min())))
        return best


def object_points(path, cap=400):
    mesh = trimesh.load(path, force='mesh', process=False)
    return subsample(np.asarray(mesh.vertices, float), cap)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', required=True, help='exec_trace.npz')
    ap.add_argument('--package', required=True, help='package dir with geometry.json + scene_assets')
    ap.add_argument('--mesh-dir', help='dir with {support}.obj; default package/scene_assets')
    ap.add_argument('--supports', nargs='*', help='override support names (default: geometry.json keys)')
    ap.add_argument('--object-mesh', help='held object mesh; default package case_metric.obj')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--frames', nargs=2, type=int, metavar=('FIRST', 'LAST'),
                    help='audit a source-frame subrange only')
    ap.add_argument('--object-support', help='support the object may legitimately rest on at the end')
    ap.add_argument('--slip-limit-mm', type=float, default=10.0)
    args = ap.parse_args()
    package = Path(args.package)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else package / 'scene_assets'
    geometry = json.loads((package / 'geometry.json').read_text())
    named = args.supports or list(geometry.get('supports', {}))
    supports = [n for n in named if (mesh_dir / f'{n}.obj').is_file()]
    if not supports:
        raise ValueError(f'no {mesh_dir}/{{' + ','.join(geometry.get("supports", {})) + '}.obj meshes')
    meshes = {n: MeshQuery(trimesh.load(mesh_dir / f'{n}.obj', force='mesh', process=False))
              for n in supports}
    obj_path = Path(args.object_mesh) if args.object_mesh else package / 'case_metric.obj'
    obj_pts_local = object_points(obj_path)

    with np.load(args.trace, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    frames = np.asarray(trace['source_frames'], int)
    base = np.asarray(trace['base_se2'], float)
    q = np.asarray(trace['q_actual'], float)
    obj_world = np.asarray(trace['actual_object'], float)
    tgo = np.asarray(trace['target_gripper_object'], float)
    ago = np.asarray(trace['actual_gripper_object'], float)
    audit_path = Path(args.trace).parent / 'robot_contact_audit.json'
    contacts = {}
    if audit_path.is_file():
        for entry in json.loads(audit_path.read_text()):
            contacts[int(entry['source_frame'])] = entry.get('pairs', [])

    lo, hi = (0, len(frames)) if not args.frames else \
        (int(np.searchsorted(frames, args.frames[0])), int(np.searchsorted(frames, args.frames[1], 'right')))
    scene, robot = _build_scene()
    links = {l.entity.get_name(): l for l in robot.get_links()}
    body = link_shape_points(robot.get_links())
    obj_actor_builder = scene.create_actor_builder()
    obj_actor_builder.add_visual_from_file(str(obj_path))
    obj_actor = obj_actor_builder.build_kinematic(name='audited_object')

    actual_pad = np.asarray(trace['actual_pad'], float)
    actual_object = np.asarray(trace['actual_object'], float)
    bilateral = np.asarray(trace['bilateral'], float) if 'bilateral' in trace else np.zeros(len(frames))
    carry_start = int(np.argmax(bilateral > 0)) if (bilateral > 0).any() else -1
    pad_obj0 = np.linalg.solve(actual_pad[carry_start], actual_object[carry_start]) \
        if carry_start >= 0 else None
    rows, minima = [], {n: (np.inf, -1, '') for n in supports}
    obj_min = {'supports': {n: (np.inf, -1) for n in supports}, 'robot': (np.inf, -1, '')}
    slip_max = (0.0, -1)
    for i in range(lo, hi):
        f = int(frames[i])
        robot.set_root_pose(bp.se2_pose(*base[i]))
        robot.set_qpos(q[i])
        obj_actor.set_pose(Pose(p=obj_world[i, :3, 3],
                                q=Rotation.from_matrix(obj_world[i, :3, :3]).as_quat()[[3, 0, 1, 2]]))
        scene.update_render()
        obj_pts = (obj_world[i] @ np.c_[obj_pts_local, np.ones(len(obj_pts_local))].T).T[:, :3]
        body_pts, per_link = {}, {}
        for name, (link, shapes) in body.items():
            lp = link.get_entity_pose().to_transformation_matrix()
            pts = np.vstack([shape_world(v, local, lp) for v, local in shapes])
            body_pts[name] = pts
            per_link[name] = pts
        row = {'source_frame': f}
        for sup in supports:
            best, best_link = np.inf, ''
            for name, pts in per_link.items():
                d = meshes[sup].min_distance(pts)
                if d < best:
                    best, best_link = d, name
            row[f'body_vs_{sup}_mm'] = best * 1000 if np.isfinite(best) else ''
            if best < minima[sup][0]:
                minima[sup] = (best, f, best_link)
            d_obj = meshes[sup].min_distance(obj_pts)
            row[f'object_vs_{sup}_mm'] = d_obj * 1000 if np.isfinite(d_obj) else ''
            if d_obj < obj_min['supports'][sup][0]:
                obj_min['supports'][sup] = (d_obj, f)
        robot_tree_points = np.vstack([p for n, p in per_link.items() if n not in GRIPPER_LINKS])
        kdt = cKDTree(robot_tree_points)
        d_obj_robot, _ = kdt.query(obj_pts, k=1)
        d_obj_robot = float(d_obj_robot.min())
        hit = [n for n, pts in per_link.items() if n not in GRIPPER_LINKS
               if float(cKDTree(pts).query(obj_pts, k=1)[0].min()) < 1e-4]
        row['object_vs_robot_mm'] = d_obj_robot * 1000
        if d_obj_robot < obj_min['robot'][0]:
            obj_min['robot'] = (d_obj_robot, f, hit[0] if hit else '')
        rel = np.linalg.norm(ago[i, :3, 3] - tgo[i, :3, 3]) * 1000
        rot = np.degrees(np.arccos(np.clip((np.trace(tgo[i, :3, :3].T @ ago[i, :3, :3]) - 1) / 2, -1, 1)))
        row['grip_offset_mm'] = rel
        row['grip_offset_deg'] = rot
        if pad_obj0 is not None and bilateral[i] > 0:
            drift = np.linalg.norm(
                (np.linalg.solve(actual_pad[i], actual_object[i]) @ np.linalg.inv(pad_obj0))[:3, 3]) * 1000
            row['grip_slip_mm'] = drift
            if drift > slip_max[0]:
                slip_max = (drift, f)
        else:
            row['grip_slip_mm'] = ''
        row['contact_pairs'] = ';'.join(';'.join(p) for p in contacts.get(f, []))
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'clearance_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    collision_frames = sorted(f for f in contacts
                              if (args.frames is None or args.frames[0] <= f <= args.frames[1]))
    summary = {
        'trace': str(Path(args.trace).resolve()),
        'audited_source_frames': [int(frames[lo]), int(frames[hi - 1])],
        'frames_audited': len(rows),
        'supports': supports,
        'body_min_clearance_mm': {n: {'value': minima[n][0] * 1000, 'source_frame': minima[n][1],
                                      'link': minima[n][2]} for n in supports},
        'object_min_clearance_mm': {'supports': {n: {'value': obj_min['supports'][n][0] * 1000,
                                                     'source_frame': obj_min['supports'][n][1]}
                                                 for n in supports},
                                    'robot': {'value': obj_min['robot'][0] * 1000,
                                              'source_frame': obj_min['robot'][1],
                                              'link': obj_min['robot'][2]}},
        'grip_slip_mm': {'max': slip_max[0], 'source_frame': slip_max[1]},
        'real_collision_frames': collision_frames,
        'first_collision': {'source_frame': collision_frames[0], 'pairs': contacts[collision_frames[0]]}
        if collision_frames else None,
        'limits': {'slip_limit_mm': args.slip_limit_mm},
        'checks': {
            'no_robot_furniture_contact': bool(not collision_frames),
            'no_object_robot_contact': bool(obj_min['robot'][0] > 1e-4),
            'object_slip_under_limit': bool(slip_max[0] <= args.slip_limit_mm),
        },
    }
    summary['pass'] = all(summary['checks'].values())
    (args.out / 'clearance_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('pass', 'checks', 'first_collision',
                                              'grip_slip_mm')}, indent=2))


if __name__ == '__main__':
    main()

"""Full-frame exit-route comparison packages for the chairbottle sample.

All variants keep run A's frozen approach, grasp (F24–36), lift arc, pad and
object trajectory, release windows and controller setup; only the exit lane
inside `base_reference_se2` (F42–151) changes.  Variants:
  b  run B's east–south detour lane spliced onto run A's trajectory
  c  a fresh clearance-aware corridor lane: A* over the base collision hull
     against the furniture mesh footprints (z-band filtered), inflated by
     --buffer metres, resampled to the transit window
"""
import heapq
import json
from pathlib import Path
from shutil import copytree
import sys

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from shapely.prepared import prep

import sapien.core as sapien  # noqa: F401  (scene import side effects)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from cb_large_range_sim import _build_scene

HERE = Path(__file__).resolve().parent
RUN_A = HERE / 'chair-full-luna-20261116'
RUN_B = HERE / 'chair-full-luna-20261116-route-b'
LANE_FIRST, LANE_LAST = 24, 216   # entry+exit lane window; converge at the wood
# station by F216 where the verified regrasp route (second npz) takes over —
# the release gate (F156-168) gates on wood support load, not base arrival
STEP_AXIS_MAX = 0.025 / np.sqrt(2)  # runner per-frame per-axis budget (replay_multicycle)


def load_a_package():
    package = RUN_A / 'package'
    with np.load(package / 'paired_targets.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, package


def base_footprint_hull():
    """Base collision hull (xz) and its z band, from the fixed-base robot."""
    scene, robot = _build_scene()
    root = robot.get_root_pose().to_transformation_matrix()
    root_inv = np.linalg.inv(root)
    names = ('base_link', 'wheel_motor_link1', 'wheel_motor_link2', 'wheel_motor_link3',
             'wheel_motor_link4', 'steer_motor_link1', 'steer_motor_link2', 'steer_motor_link3',
             'steer_motor_link4')
    links = {l.entity.get_name(): l for l in robot.get_links()}
    pts = []
    for name in names:
        link = links.get(name)
        if link is None:
            continue
        local = root_inv @ link.get_entity_pose().to_transformation_matrix()
        for sh in link.get_collision_shapes():
            v = np.asarray(sh.get_vertices(), dtype=float)
            if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
                continue
            pts.append((local @ sh.get_local_pose().to_transformation_matrix()
                        @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    pts = np.concatenate(pts)
    hull = MultiPoint(pts[:, :2]).convex_hull
    return hull, float(pts[:, 2].min()), float(pts[:, 2].max())


def furniture_footprints(mesh_dir, zmin, zmax, buffer):
    """Inflated xy footprint of each furniture mesh, keeping triangles that
    intersect the base's z band."""
    obstacles = {}
    for name in ('desk', 'table', 'chair'):
        mesh = trimesh.load(mesh_dir / f'{name}.obj', force='mesh', process=False)
        tris = np.asarray(mesh.triangles)
        polys = []
        for tri in tris:
            if tri[:, 2].max() < zmin - .01 or tri[:, 2].min() > zmax + .01:
                continue
            poly = Polygon(tri[:, :2])
            if poly.area > 1e-8:
                polys.append(poly)
            else:
                line = LineString(tri[:, :2])
                if line.length > 1e-5:
                    polys.append(line.buffer(.006))
        obstacles[name] = unary_union(polys).buffer(buffer) if polys else Polygon()
    return obstacles


def corridor_lane(mesh_dir, buffer, step=0.02):
    """A* the inflated base footprint from the desk station to the wood station."""
    hull, zmin, zmax = base_footprint_hull()
    obstacles = furniture_footprints(mesh_dir, zmin, zmax, buffer)
    blocked_all = prep(unary_union(list(obstacles.values())))
    start = (1.3, .97, 4.71238898038469)          # table_base, yaw 270 deg
    goal = (.8465444450153005, .252193293382574, 3.141592653589793)  # wood_base, yaw 180 deg
    # yaw profile: hold 270 on the outbound ring, turn to 180 on final approach
    fixed_yaw = np.radians(270)
    rotated = rotate(hull, np.degrees(fixed_yaw), origin=(0, 0))
    x_lo, x_hi = -0.4, 2.3
    y_lo, y_hi = -1.2, 1.7
    nx = int(round((x_hi - x_lo) / step)) + 1
    ny = int(round((y_hi - y_lo) / step)) + 1
    blocked = np.zeros((nx, ny), bool)
    for ix in range(nx):
        for iy in range(ny):
            poly = translate(rotated, xoff=x_lo + ix * step, yoff=y_lo + iy * step)
            blocked[ix, iy] = blocked_all.intersects(poly)

    def node(x, y):
        return (int(round((x - x_lo) / step)), int(round((y - y_lo) / step)))

    def world(n):
        return (x_lo + n[0] * step, y_lo + n[1] * step)

    sn, gn = node(*start[:2]), node(*goal[:2])
    if blocked[sn] or blocked[gn]:
        raise ValueError('start or goal station inside the inflated corridor; reduce --buffer')
    heap = [(np.hypot(sn[0] - gn[0], sn[1] - gn[1]), 0., sn)]
    cost = {sn: 0.}
    prev = {}
    while heap:
        _, g, u = heapq.heappop(heap)
        if u == gn:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            path.reverse()
            break
        if g > cost.get(u, 1e99):
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy):
                    continue
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < nx and 0 <= v[1] < ny) or blocked[v]:
                    continue
                if dx and dy and (blocked[u[0] + dx, u[1]] or blocked[u[0], u[1] + dy]):
                    continue
                ng = g + (np.sqrt(2) if dx and dy else 1) * step
                if ng < cost.get(v, 1e99):
                    cost[v] = ng
                    prev[v] = u
                    heapq.heappush(heap, (ng + np.hypot(gn[0] - v[0], gn[1] - v[1]) * step, ng, v))
    else:
        raise ValueError('no corridor path at this buffer')
    waypoints = [world(v) for v in path]
    waypoints = _shortcut(waypoints, hull, fixed_yaw, blocked_all)
    return waypoints, float(cost[gn])


def _shortcut(waypoints, hull, yaw, blocked_all, sample_step=0.01):
    """String-pulling: drop intermediate waypoints while the straight segment
    stays collision free for the yaw-270 base hull (sampled every 2 cm)."""
    rotated = rotate(hull, np.degrees(yaw), origin=(0, 0))

    def segment_free(a, b):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(int(length / sample_step), 2)
        for t in np.linspace(0, 1, n):
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if blocked_all.intersects(translate(rotated, xoff=p[0], yoff=p[1])):
                return False
        return True

    out = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        while j > i + 1 and not segment_free(waypoints[i], waypoints[j]):
            j -= 1
        out.append(waypoints[j])
        i = j
    return out


def resample_lane(waypoints, n_frames, yaw_start, yaw_end):
    """Time-parameterize the corridor polyline into n_frames base_reference rows
    so every per-frame xy move stays under the runner's per-axis budget.  Any
    slack becomes a hold at the start (the base may leave the desk station while
    the gripper is still closing; entry-route timing is part of the route)."""
    coords = np.asarray(waypoints, float)
    seg = np.diff(coords, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    if (seg_len <= 0).any():
        keep = np.r_[True, seg_len > 0]
        coords = coords[keep]
        seg = np.diff(coords, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
    # frames needed per segment: the dominant axis must not exceed the budget
    # Station-envelope routes (2 cm tolerance) crashed the WBC fit at
    # 17.7 mm/axis when the base lagged the window; use route A's verified
    # reference rate instead.
    axis_rate = 0.0125
    seg_frames = np.ceil(np.abs(seg).max(axis=1) / axis_rate).astype(int)
    seg_frames = np.maximum(seg_frames, 1)
    total = int(seg_frames.sum())
    if total > n_frames - 1:
        raise ValueError(f'corridor needs {total} frames at {axis_rate * 1000:.1f} mm/axis, '
                         f'window has {n_frames - 1}')
    # accumulate positions per frame with per-axis clamped progress
    path = LineString(coords)
    length = path.length
    cum = np.r_[0, np.cumsum(seg_len)]
    frame_dist = np.zeros(n_frames)
    moving = n_frames - total  # slack frames held at the start
    idx = 0
    acc = 0.
    for f in range(moving, n_frames):
        budget = axis_rate
        # advance along the current segment while either axis has budget left
        while idx < len(seg) and budget > 1e-12:
            seg_left = cum[idx + 1] - acc
            direction = seg[idx] / max(seg_len[idx], 1e-12)
            # distance we may travel this frame on this segment
            allowed = budget / max(np.abs(direction).max(), 1e-12)
            take = min(seg_left, allowed)
            acc += take
            budget -= take * np.abs(direction).max()
            frame_dist[f] = acc
            if acc >= cum[idx + 1] - 1e-12:
                idx += 1
        if idx >= len(seg):
            for g in range(f + 1, n_frames):
                frame_dist[g] = length
            break
    dists = np.r_[np.zeros(moving), frame_dist[moving:]]
    xy = np.array([path.interpolate(d).coords[0] for d in dists])
    yaw = np.full(n_frames, yaw_start)
    tail = 40
    u = np.clip((np.arange(n_frames) - (n_frames - tail)) / tail, 0, 1)
    yaw = yaw_start + u * (yaw_end - yaw_start)
    rows = np.column_stack([xy, yaw])
    step_dev = np.abs(np.diff(rows, axis=0)).max(axis=0)
    if (step_dev[0] > STEP_AXIS_MAX * 1.001 or step_dev[1] > STEP_AXIS_MAX * 1.001
            or step_dev[2] > np.radians(5.0)):
        raise ValueError(f'lane violates the per-axis budget: {step_dev}')
    return rows


def write_variant(out_dir, arrays, package_src):
    out = out_dir / 'package'
    if out.exists():
        raise FileExistsError(out)
    copytree(package_src, out, ignore=lambda src, names: {'paired_targets.npz'})
    np.savez_compressed(out / 'paired_targets.npz', **arrays)
    return out


def main():
    a, package_src = load_a_package()
    frames = np.asarray(a['src_frames'], int)
    base_a = np.asarray(a['base_reference_se2'], float).copy()
    base_b = np.asarray(np.load(RUN_B / 'package/paired_targets.npz')['base_reference_se2'], float)
    if not np.allclose(base_a[:LANE_FIRST], base_b[:LANE_FIRST], atol=1e-9):
        raise ValueError('routes disagree before the splice window')
    lane_len_b = float(np.sum(np.linalg.norm(np.diff(base_a[LANE_FIRST:LANE_LAST + 1, :2], axis=0), axis=1)))

    variants = {}
    variant = a.copy()
    variant['base_reference_se2'] = base_b.copy()
    variants['b'] = variant

    mesh_dir = package_src / 'scene_assets'
    buffer = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    waypoints, length = corridor_lane(mesh_dir, buffer)
    lane = resample_lane(waypoints, LANE_LAST - LANE_FIRST + 1,
                         base_a[LANE_FIRST, 2], base_a[LANE_LAST, 2])
    variant_c = a.copy()
    route_c = base_a.copy()
    route_c[LANE_FIRST:LANE_LAST + 1] = lane
    variant_c['base_reference_se2'] = route_c
    variants['c'] = variant_c

    runs = HERE / 'runs'
    out_info = {}
    for name, arrays in variants.items():
        out = runs / ('exit-route-c' if name == 'c' else 'exit-route-b-full')
        if out.exists():
            out_info[name] = str(out.resolve())  # b-full is frozen; only (re)build c
            continue
        write_variant(out, arrays, package_src)
        out_info[name] = str(out.resolve())
    info = {'buffer_m': buffer, 'corridor_waypoints': len(waypoints), 'corridor_length_m': length,
            'lane_b_length_m': lane_len_b, 'lane_c_frames_changed': int((np.linalg.norm(
                route_c[:, :2] - base_a[:, :2], axis=1) > 1e-9).sum()),
            'step_budget_axis_m': STEP_AXIS_MAX, 'outputs': out_info}
    (HERE / 'runs/exit-lane-report.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

"""Unified source-indexed kinematic Stage 0 replay for R1.

Consumes a stage0_trace.npz (schema owned by scripts/stage0_trace.py) plus the
sample scene (RAS point cloud or same-video GLB scene asset) and renders the
frozen original retarget: left source video, right four world views, and an
independent wrist MP4.  Draws the original-retarget pad trajectory and the
independent object trajectory in every view, and marks original observation,
gap-filled rows and unavailable segments on both the trajectories and the
per-timeline status strips.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cb_large_range_sim import _build_scene, bp, rg
from stage0_trace import validate_trace

STATUS_COLORS = {  # display RGB; the canvas array is RGB for imageio
    "observed": (80, 220, 80), "filled": (255, 160, 60), "unavailable": (230, 80, 80),
    "pad_tail": (255, 255, 255), "object": (60, 200, 230),
}

def _bgr(color):
    """The canvas arrays are RGB for imageio; pass display-RGB through unchanged."""
    return color

def pose(T):
    return sapien.Pose(p=T[:3, 3], q=Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])

def base_pose(x, y, yaw):
    return sapien.Pose(p=[x, y, 0.0], q=[np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])

def load_trace(trace_path):
    with np.load(trace_path, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    return validate_trace(t)

def load_points(cloud_path):
    with np.load(cloud_path, allow_pickle=False) as z:
        xyz = np.asarray(z["xyz"], np.float32); rgb = np.asarray(z["rgb"])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError("scene point cloud needs finite xyz and matching rgb arrays")
    rgb = np.asarray(rgb, np.float32)
    if rgb.max(initial=0) > 1: rgb /= 255.0
    return xyz, rgb

def add_point_cloud(scene, xyz, rgb):
    cloud = sapien.render.RenderPointCloudComponent()
    cloud.set_vertices(np.ascontiguousarray(xyz, dtype=np.float32))
    colors = np.column_stack((rgb, np.ones(len(rgb), np.float32)))
    cloud.set_attribute("color", np.ascontiguousarray(colors, dtype=np.float32))
    cloud.set_attribute("scale", np.full((len(xyz), 1), 0.0225, np.float32))
    entity = sapien.Entity()
    entity.set_name("ras_scene_points")
    entity.add_component(cloud)
    scene.add_entity(entity)
    return cloud

def build_world(scene, args, xyz, rgb):
    """Point cloud and/or GLB scene asset plus the calibrated support boxes."""
    cloud = add_point_cloud(scene, xyz, rgb) if xyz is not None else None
    supports = {}
    geometry_path = Path(args.geometry) if args.geometry else None
    if geometry_path:
        geometry = json.loads(geometry_path.read_text())
        supports = {k: geometry["supports"][k] for k in args.support_names}
    for name, data in supports.items():
        center = np.array(data["center"]) - [0, 0, .0225]
        quat = Rotation.from_euler("z", data["yaw"]).as_quat()[[3, 0, 1, 2]]
        color = (.82, .83, .82) if name == "table" else (.31, .32, .33)
        rg.create_box(scene, sapien.Pose(p=center, q=quat),
                      [*data["half_extents"], .0225], color=color, name=name, is_static=True)
    if args.scene_asset_dir:
        glb = Path(args.scene_asset_dir) / "video_scene.glb"
        if not glb.is_file():
            glb = Path(args.scene_asset_dir) / "final_scene_world.glb"
        if not glb.is_file():
            raise ValueError(f"no video_scene.glb or final_scene_world.glb in {args.scene_asset_dir}")
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(glb))
        builder.build_static(name="same_video_scene")
        for entity in scene.entities:
            if entity.name in supports:
                for component in entity.components:
                    if hasattr(component, "visibility"):
                        component.visibility = 0.
    return cloud, supports

def add_object(scene, path):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(path))
    return builder.build_kinematic(name="stage0_object")

def hide_object(actor, visible):
    for component in actor.get_components():
        if hasattr(component, "visibility"):
            component.visibility = 1.0 if visible else 0.0

def frame_rgb(cap, frame_id, current):
    while current[0] <= frame_id:
        ok, bgr = cap.read()
        if not ok: raise RuntimeError(f"source video ended before frame {frame_id}")
        selected = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if current[0] == frame_id else None
        current[0] += 1
        if selected is not None: return selected
    raise ValueError(f"requested source frame {frame_id} was skipped")

def camera(scene, name, target, eye, width=480, height=360, fov=55):
    cam = scene.add_camera(name, width, height, np.deg2rad(fov), 0.02, 100.0)
    cam.entity.set_pose(rg.look_at_pose(eye, target))
    return cam

def project(cam, points):
    """Project world points with the OpenCV-style intrinsic/extrinsic matrices."""
    extr = cam.get_extrinsic_matrix(); intr = cam.get_intrinsic_matrix()
    hom = np.column_stack((points, np.ones(len(points)))).T
    cam_pts = extr @ hom
    z = cam_pts[2]
    pix = intr @ cam_pts
    uv = pix[:2] / np.maximum(pix[2], 1e-9)
    inside = (z > 1e-6) & (uv[0] >= 0) & (uv[0] < cam.get_width() - 1) \
        & (uv[1] >= 0) & (uv[1] < cam.get_height() - 1)
    return uv.T, inside

def check_projection(cam, eye, target):
    """One-time sanity probe: a point 0.2 m along the view axis must be centred."""
    forward = (np.asarray(target, float) - np.asarray(eye, float))
    forward /= np.linalg.norm(forward)
    uv, inside = project(cam, [np.asarray(eye, float) + .2 * forward])
    w, h = cam.get_width(), cam.get_height()
    if not inside[0] or abs(uv[0, 0] - w / 2) > .12 * w or abs(uv[0, 1] - h / 2) > .12 * h:
        raise ValueError(f"camera {cam.entity.name}: projection convention probe failed at {uv[0]}")

def draw_trajectory(img, cam, pts_world, status, tail=30):
    """Polyline coloured by per-row observation status, with a white tail and head."""
    if len(pts_world) < 2: return img
    uv, inside = project(cam, pts_world)
    inside &= np.asarray(status) != "unavailable"
    for i in range(len(pts_world) - 1):
        if not (inside[i] and inside[i + 1]): continue
        color = _bgr(STATUS_COLORS["pad_tail"] if status[i] == "pad" else STATUS_COLORS[status[i]])
        thick = 2 if i >= len(pts_world) - tail else 1
        cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int)),
                 color, thick, cv2.LINE_AA)
    head = uv[-1]
    if inside[-1]:
        cv2.circle(img, tuple(np.round(head).astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return img

def draw_status_strip(img, timeline_flags, current, x0=8, y0=318, height=9):
    """Observation strips over the full timeline: row 0 hand, row 1 object."""
    n = len(timeline_flags["hand"])
    width = img.shape[1] - x0 - 8
    for row, key in enumerate(("hand", "object")):
        flags = timeline_flags[key]  # per-row char: o/f/u
        y = y0 + row * (height + 4)
        cv2.putText(img, "HAND" if key == "hand" else " OBJ", (x0 - 6, y + height),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
        for i, state in enumerate(flags):
            color = _bgr(STATUS_COLORS["observed" if state == "o" else "filled" if state == "f" else "unavailable"])
            xa = x0 + int(i * width / n); xb = x0 + int((i + 1) * width / n)
            cv2.rectangle(img, (xa, y), (max(xb, xa + 1), y + height), color, -1)
        cx = x0 + int(current * width / n)
        cv2.rectangle(img, (cx - 1, y - 2), (cx + 1, y + height + 2), (255, 255, 255), -1)
    return img

def label(img, text, xy=(12, 28), scale=.7):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def tile_label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 38), (24, 24, 24), -1)
    return label(img, text, (12, 27), .55)

def build_joint_indices(robot):
    names = {j.get_name(): i for i, j in enumerate(robot.get_active_joints())}
    arm = [names[f"right_arm_joint{i}"] for i in range(1, 7)]
    torso = [names[f"torso_joint{i}"] for i in range(1, 5)]
    fingers = [names[f"right_gripper_finger_joint{i}"] for i in (1, 2)]
    left = [names[f"left_arm_joint{i}"] for i in range(1, 7)]
    return arm, torso, fingers, left

def solve_arm(robot, selected, target, seed, neutral, limits):
    def candidate(initial):
        q = bp.limit_aware_resolve(robot, "right", selected, initial, limits[:, 0], limits[:, 1],
                                   w_pos=20., w_ori=1., target=target, seed=initial)
        p, R = bp._gripper_pad_fk(robot, "right", selected, q)
        pos_m = float(np.linalg.norm(p - target[0]))
        angle_rad = float(np.arccos(np.clip((np.trace(target[1].T @ R) - 1) / 2, -1, 1)))
        score = pos_m + .022 * angle_rad + .005 * float(np.linalg.norm(q - seed))
        return q, pos_m, angle_rad, score
    result = candidate(seed)
    source = "sequential"
    if result[1] > .030 or result[2] > .5:
        fallback = candidate(neutral)
        if fallback[3] < result[3]:
            result = fallback
            source = "neutral_fallback"
    return (*result[:3], source)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--object-mesh", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--scene-points", type=Path, help="scene_points.npz (xyz, rgb)")
    group.add_argument("--scene-asset-dir", type=Path, help="directory with video_scene.glb")
    ap.add_argument("--geometry", type=Path, help="geometry.json with calibrated support boxes")
    ap.add_argument("--support-names", nargs="*", default=None)
    ap.add_argument("--object-label", default="SAM3D object")
    ap.add_argument("--config", type=Path, help="sample input config; fills scene/object arguments")
    ap.add_argument("--max-frames", type=int, default=0, help="diagnostic prefix only; 0 renders all trace rows")
    a = ap.parse_args()
    if a.config:
        from stage0_trace import load_config
        config = load_config(str(a.config), ROOT)
        scene_cfg = config.get("scene", {})
        a.scene_points = a.scene_points or scene_cfg.get("scene_points")
        a.scene_asset_dir = a.scene_asset_dir or scene_cfg.get("asset_dir")
        a.geometry = a.geometry or scene_cfg.get("geometry")
        a.support_names = a.support_names or scene_cfg.get("supports", ["table", "chair"])
        a.object_mesh = a.object_mesh or config.get("object_mesh")
        a.object_label = a.object_label if a.object_label != "SAM3D object" \
            else config.get("object_label", a.object_label)
    if a.scene_points is None and a.scene_asset_dir is None:
        ap.error('provide --scene-points or --scene-asset-dir for the sample scene')
    if not a.object_mesh:
        ap.error('provide --object-mesh (or object_mesh in the config)')
    a.trace = Path(a.trace); a.video = Path(a.video); a.object_mesh = Path(a.object_mesh)
    if a.scene_points: a.scene_points = Path(a.scene_points)
    if a.scene_asset_dir: a.scene_asset_dir = Path(a.scene_asset_dir)
    if a.geometry: a.geometry = Path(a.geometry)
    t = load_trace(a.trace)
    n = len(t["src_frames"]) if a.max_frames <= 0 else min(len(t["src_frames"]), a.max_frames)
    xyz, rgb = load_points(a.scene_points) if a.scene_points else (None, None)
    a.out.mkdir(parents=True, exist_ok=True)
    scene, robot = _build_scene()
    cloud, supports = build_world(scene, a, xyz, rgb)
    actor = add_object(scene, a.object_mesh)
    arm, torso, fingers, left = build_joint_indices(robot)
    selected = arm + torso
    joints = robot.get_active_joints()
    q = robot.get_qpos()
    q[left] = rg.R1_LEFT_START
    q[arm] = rg.R1_RIGHT_PARK
    q[torso] = 0.0
    q[fingers] = 0.045
    robot.set_qpos(q)
    limits = np.vstack((bp._limits("right"), [joints[i].get_limits()[0] for i in torso]))
    limits[6:, 0] = np.maximum(limits[6:, 0], -.25)
    limits[6:, 1] = np.minimum(limits[6:, 1], .25)
    neutral = np.r_[np.asarray(rg.R1_RIGHT_PARK, dtype=float), np.zeros(4)]
    seed = neutral.copy()
    link = robot.find_link_by_name("right_gripper_link")
    object_xy = t["T_world_object"][:n, :2, 3][t["object_visible"][:n]]
    anchor_pts = [t["base_se2"][:n, :2], object_xy]
    if xyz is not None:
        anchor_pts.insert(0, xyz[:, :2])
    for name in supports:
        anchor_pts.append(np.asarray(supports[name]["center"])[:2][None])
    center = np.median(np.vstack(anchor_pts), axis=0)
    if xyz is not None:
        zlo = float(np.quantile(xyz[:, 2], .02)); zhi = float(np.quantile(xyz[:, 2], .98))
    else:
        zs = np.array([supports[name]["center"][2] for name in supports] or [0.8])
        zlo, zhi = zs.min() - .3, zs.max() + .3
    focus = np.array([center[0], center[1], np.clip((zlo + zhi) / 2, 0.5, 1.5)])
    span = max(float(np.ptp(t["base_se2"][:n, 0])), float(np.ptp(t["base_se2"][:n, 1])),
               float(np.ptp(object_xy, axis=0).max()) if len(object_xy) > 1 else 0.0, 2.0)
    for name in supports:
        span = max(span, 2 * float(np.asarray(supports[name]["half_extents"]).max()))
    dist = max(2.3, span * 1.35)
    front = camera(scene, "world_front", focus, focus + [-dist, -dist, dist * .55], fov=55)
    side = camera(scene, "side", focus, focus + [2.4, -2.8, 1.0], fov=55)
    overhead = camera(scene, "overhead", focus, focus + [.03, -.10, dist * 1.5], fov=55)
    wrist = camera(scene, "wrist", focus, focus + [0., -dist, dist * .3], fov=88)
    for cam, eye in ((front, focus + [-dist, -dist, dist * .55]), (side, focus + [2.4, -2.8, 1.0]),
                     (overhead, focus + [.03, -.10, dist * 1.5]), (wrist, focus + [0., -dist, dist * .3])):
        check_projection(cam, eye, focus)
    camera_to_gripper = rg.look_at_pose([.005, -.095, .085], [rg.PAD_FWD, 0., -.012]).to_transformation_matrix()
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise RuntimeError(f"cannot open source video: {a.video}")
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if int(t["src_frames"][n - 1]) >= source_count: raise ValueError("trace source frame outside video")
    current = [0]
    fps = float(t["playback_fps"])
    pad_world = t["T_world_pad"][:n, :3, 3]
    obj_world = t["T_world_object"][:n, :3, 3]
    hand_status = np.where(t["hand_observed"][:n], "o", "f")
    object_status = np.where(t["object_unavailable"][:n], "u",
                             np.where(t["object_observed"][:n], "o", "f"))
    strips = {"hand": hand_status, "object": object_status}
    pad_status = ["pad" if s == "o" else "filled" for s in hand_status]
    obj_status = [{"o": "observed", "f": "filled", "u": "unavailable"}[s] for s in object_status]
    depth_anchors = set(map(int, t.get("object_depth_anchors", [])))
    csv_path = a.out / "stage0_frame_errors.csv"
    composite_path = a.out / "stage0_replay.mp4"; wrist_path = a.out / "stage0_wrist.mp4"
    views = (front, side, overhead, wrist)
    errors, fallback_frames = [], []
    with csv_path.open("w", newline="") as stream, \
         imageio.get_writer(composite_path, fps=fps, macro_block_size=8) as combined, \
         imageio.get_writer(wrist_path, fps=fps, macro_block_size=8) as wrist_writer:
        writer = csv.writer(stream)
        writer.writerow(["source_frame", "pad_position_error_mm", "pad_orientation_error_deg",
                         *[f"right_arm_joint_{i}_rad" for i in range(1, 7)],
                         *[f"torso_joint_{i}_rad" for i in range(1, 5)],
                         "ik_seed_source", "hand_observed", "hand_filled", "object_observed",
                         "object_filled", "object_unavailable", "object_depth_anchor"])
        for k in range(n):
            src = int(t["src_frames"][k])
            source = frame_rgb(cap, src, current)
            robot.set_root_pose(base_pose(*map(float, t["base_se2"][k])))
            target = (t["T_world_pad"][k, :3, 3], t["T_world_pad"][k, :3, :3])
            seed, pos_m, angle_rad, seed_source = solve_arm(robot, selected, target, seed, neutral, limits)
            q = robot.get_qpos()
            q[selected] = seed; q[fingers] = t["grip"][k]
            robot.set_qpos(q)
            actual = bp._gripper_pad_fk(robot, "right", selected, seed)
            pos_mm = float(np.linalg.norm(actual[0] - target[0]) * 1000)
            c = np.clip((np.trace(target[1].T @ actual[1]) - 1) / 2, -1, 1)
            ori_deg = float(np.degrees(np.arccos(c)))
            errors.append([src, pos_mm, ori_deg])
            if seed_source != "sequential": fallback_frames.append(src)
            writer.writerow([src, pos_mm, ori_deg, *seed.tolist(), seed_source,
                             bool(t["hand_observed"][k]), bool(t["hand_filled"][k]),
                             bool(t["object_observed"][k]), bool(t["object_filled"][k]),
                             bool(t["object_unavailable"][k]), src in depth_anchors])
            actor.set_pose(pose(t["T_world_object"][k]))
            visible = bool(t["object_visible"][k])
            side_target = (actual[0] + obj_world[k]) / 2 if visible else actual[0]
            side_eye = side_target + np.array([1.5, -1.5, .7])
            side.entity.set_pose(rg.look_at_pose(side_eye, side_target))
            hide_object(actor, visible)
            hand = link.get_entity_pose().to_transformation_matrix()
            wrist.entity.set_pose(sapien.Pose(hand @ camera_to_gripper))
            scene.update_render()
            images = []
            for cam in views:
                if cam is side and cloud is not None: cloud.disable()
                cam.take_picture()
                im = np.uint8(255 * np.clip(cam.get_picture("Color")[:, :, :3], 0, 1))
                if cam is not wrist:
                    im = draw_trajectory(im, cam, pad_world[:k + 1], pad_status[:k + 1])
                    im = draw_trajectory(im, cam, obj_world[:k + 1], obj_status[:k + 1])
                images.append(im)
                if cam is side and cloud is not None: cloud.enable()
            wrist_frame = draw_trajectory(images[3].copy(), wrist, pad_world[:k + 1], pad_status[:k + 1])
            wrist_writer.append_data(tile_label(wrist_frame, f"Wrist | source F{src:04d}"))
            src_panel = cv2.resize(source, (640, 360), interpolation=cv2.INTER_AREA)
            src_panel = draw_status_strip(src_panel, strips, k)
            canvas = np.zeros((720, 1600, 3), np.uint8)
            canvas[:, :, :] = 25
            label(canvas, "Stage 0 kinematic replay", (16, 30), .75)
            label(canvas, f"{a.object_label} | green/cyan observed, orange gap-filled, red unavailable",
                  (16, 60), .5)
            label(canvas, f"Source F{src:04d} | t={src / float(t['video_fps']):.2f}s", (16, 95), .48)
            canvas[180:540, 0:640] = src_panel
            canvas[0:360, 640:1120] = tile_label(images[0].copy(), "World / front")
            canvas[0:360, 1120:1600] = tile_label(images[1].copy(), "Side / detail")
            canvas[360:720, 640:1120] = tile_label(images[2].copy(), "Overhead")
            canvas[360:720, 1120:1600] = tile_label(images[3].copy(), "Wrist")
            combined.append_data(canvas)
    cap.release()
    e = np.asarray(errors)
    report = {"mode": "kinematic_only", "frames_rendered": n, "frames_requested": len(t["src_frames"]),
      "source_frame_first_last": [int(t["src_frames"][0]), int(t["src_frames"][n - 1])],
      "playback_fps": fps, "source_frames_are_sampled": bool(np.any(np.diff(t["src_frames"][:n]) != 1)),
      "scene": {"points": str(a.scene_points) if a.scene_points else None,
                "asset_dir": str(a.scene_asset_dir) if a.scene_asset_dir else None,
                "supports": sorted(supports)},
      "ik_solver": "sequential fixed-base 6-arm+4-torso R1 limit_aware_resolve; w_pos=20,w_ori=1; neutral-seed fallback above 30mm or 0.5rad residual",
      "ik_neutral_fallback_frames": fallback_frames,
      "pad_fk_position_error_mm": {"median": float(np.median(e[:, 1])), "p95": float(np.quantile(e[:, 1], .95)),
                                   "max": float(e[:, 1].max())},
      "pad_fk_orientation_error_deg": {"median": float(np.median(e[:, 2])), "p95": float(np.quantile(e[:, 2], .95)),
                                       "max": float(e[:, 2].max())},
      "hand_observed_rows": int(t["hand_observed"][:n].sum()), "hand_filled_rows": int(t["hand_filled"][:n].sum()),
      "object_observed_rows": int(t["object_observed"][:n].sum()),
      "object_filled_rows": int(t["object_filled"][:n].sum()),
      "object_unavailable_rows": int(t["object_unavailable"][:n].sum()),
      "object_depth_anchor_frames": sorted(depth_anchors),
      "trajectory_overlay": True, "object_label": a.object_label,
      "per_frame_csv": str(csv_path.resolve()),
      "composite": str(composite_path.resolve()), "wrist": str(wrist_path.resolve()),
      "trace": str(a.trace.resolve()), "object_mesh": str(a.object_mesh.resolve())}
    (a.out / "stage0_render_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

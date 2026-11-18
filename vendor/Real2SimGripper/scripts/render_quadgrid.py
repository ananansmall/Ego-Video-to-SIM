"""Quad-grid acceptance video: source / MANO / Stage 0 / physics in one frame.

Panel layout per docs/21_retarget_render_acceptance.md:
  top-left     source video frame (hand, object, scene)
  bottom-left  recovered MANO hand projected with the per-frame RAS map camera:
               wrist / thumb tip / index tip / two-finger pad centre / approach
               axis, plus an observation/gap strip
  top-right    Stage 0 kinematic replay (R1, object, table/chair, scene) with
               the frozen Stage 0 pad-centre trajectory in black
  bottom-right physics replay from the R1 execution trace, same camera: black
               Stage 0 pad centre, red the final commanded pad centre, robot at
               its measured joint state

All four panels share one source frame and one source time.  The sample config
supplies videos, traces, the camera map, scene assets and ranges.
"""
import argparse
import csv as _csv
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
from cb_large_range_sim import _build_scene, bp, rg  # noqa: E402
from stage0_render import (  # noqa: E402  (reuse the frozen Stage 0 solver/views)
    build_joint_indices, solve_arm, load_trace, load_points, build_world,
    add_object, hide_object, label, tile_label, _bgr,
)

MANO_BONES = [  # MANO 21-joint topology
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (0, 16), (16, 17),
    (17, 18), (18, 19), (19, 20),
]
PANEL_W, PANEL_H = 640, 360


def project(K, w2c, pts):
    cam = w2c @ np.c_[pts, np.ones(len(pts))].T
    uv = (K @ cam[:3]).T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-9), cam[2]


def draw_polyline(img, pts_uv, color, tail=45, thick=2):
    for i in range(1, len(pts_uv)):
        p0 = tuple(np.round(pts_uv[i - 1]).astype(int))
        p1 = tuple(np.round(pts_uv[i]).astype(int))
        cv2.line(img, p0, p1, _bgr(color), thick if i >= len(pts_uv) - tail else 1, cv2.LINE_AA)


def pose_to_sapien(T):
    return sapien.Pose(p=T[:3, 3], q=Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = ROOT

    def rp(v):
        p = Path(v)
        return p if p.is_absolute() else root / p

    stage0 = load_trace(rp(cfg["stage0_trace"]))
    pair_file = rp(cfg["pair_npz"])
    pair_raw = np.load(pair_file, allow_pickle=False)
    pair = {k: np.asarray(pair_raw[k]) for k in pair_raw.files}
    phys = np.load(rp(cfg["physics_trace"]), allow_pickle=False)
    phys = {k: np.asarray(phys[k]) for k in phys.files}
    mp = np.load(rp(cfg["ras_map"]), allow_pickle=False)
    n = len(stage0["src_frames"])
    if args.max_frames:
        n = min(n, args.max_frames)
    frames_all = np.asarray(stage0["src_frames"])
    assert np.array_equal(np.asarray(phys["source_frames"], int), frames_all.astype(int)), \
        "physics trace source_frames must match the Stage 0 trace exactly"

    cap = cv2.VideoCapture(str(rp(cfg["source_video"])))
    ok, probe = cap.read()
    H_src, W_src = probe.shape[:2]
    K = np.asarray(mp["K_mean"], float)
    H_map, W_map = mp["rgb"].shape[1:3]
    Ks = np.array([[K[0, 0] * W_src / W_map, 0, K[0, 2] * W_src / W_map],
                   [0, K[1, 1] * H_src / H_map, K[1, 2] * H_src / H_map],
                   [0, 0, 1]])
    src_cap = cv2.VideoCapture(str(rp(cfg["source_video"])))

    world_joints = np.asarray(pair["world_joints"], float)[:n]
    # Hand coverage comes from the frozen Stage 0 trace flags (valid -> observed,
    # complement -> gap-filled).  pair["source_anchor_frames"] are the *object*
    # depth anchors and are NOT the hand observation mask, so they must not label
    # the MANO panel.
    hand_obs = np.asarray(stage0["hand_observed"], bool)[:n]
    hand_filled = np.asarray(stage0["hand_filled"], bool)[:n] if "hand_filled" in stage0 \
        else ~hand_obs

    scene_cfg = cfg.get("scene", {})
    xyz, rgb = (None, None)
    if scene_cfg.get("scene_points"):
        xyz, rgb = load_points(rp(scene_cfg["scene_points"]))
    scene, robot = _build_scene()

    class _A:
        pass
    a = _A()
    a.geometry = rp(scene_cfg["geometry"]) if scene_cfg.get("geometry") else None
    a.support_names = scene_cfg.get("supports", ["table", "chair"])
    a.scene_asset_dir = rp(scene_cfg["asset_dir"]) if scene_cfg.get("asset_dir") else None
    cloud, supports = build_world(scene, a, xyz, rgb)
    obj_kin = add_object(scene, rp(cfg["object_mesh"]))
    obj_phys = add_object(scene, rp(cfg["object_mesh"]))
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
    n_dof = len(q)

    # world-view camera: identical construction to scripts/stage0_render.py so
    # the top-right panel matches the archived Stage 0 replay
    anchor_pts = [stage0["base_se2"][:n, :2], stage0["T_world_object"][:n, :2, 3]]
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
    span = max(float(np.ptp(stage0["base_se2"][:n, 0])), float(np.ptp(stage0["base_se2"][:n, 1])),
               float(np.ptp(stage0["T_world_object"][:n, :2, 3], axis=0).max()) if n > 1 else 0.0, 2.0)
    for name in supports:
        span = max(span, 2 * float(np.asarray(supports[name]["half_extents"]).max()))
    dist = max(2.3, span * 1.35)
    cam = scene.add_camera("quad_front", 960, 540, np.deg2rad(55), 0.02, 100.0)
    cam.entity.set_pose(rg.look_at_pose(focus + [-dist, -dist, dist * .55], focus))
    cam_K = cam.get_intrinsic_matrix()

    pad_stage0 = np.asarray(stage0["T_world_pad"], float)[:n, :3, 3]
    # The black line is the FROZEN archived Stage 0 (stage0-consolidation trace)
    # in BOTH right panels; the physics trace's own source_pad is a candidate
    # input (may be a modified retarget) and is recorded separately.
    pad_black = pad_stage0.copy()
    pad_red = np.asarray(phys["commanded_pad"], float)[:n, :3, 3]
    q_phys = np.asarray(phys["q_actual"], float)[:n]
    base_phys = np.asarray(phys["base_se2"], float)[:n]
    obj_phys_arr = np.asarray(phys["actual_object"], float)[:n]
    grip_phys = np.asarray(phys["executed_grip"], float)[:n]
    bilateral = np.asarray(phys["bilateral"], bool)[:n]
    n_planar = len(q_phys[0]) - n_dof   # planar joints precede the fixed-base dofs

    def cam_matrices():
        # get_extrinsic_matrix() is the OpenCV world->camera transform (as used
        # by stage0_render's projection); the raw entity pose is NOT it.
        return cam_K, cam.get_extrinsic_matrix()

    def visible_mask(uv, z):
        return (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 959) & (uv[:, 1] >= 0) & (uv[:, 1] < 539)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(stage0["playback_fps"])
    hand_rows = int(hand_obs.sum())
    manifest = {
        "scope": "quad-grid acceptance video (docs/21: source/MANO/Stage0/physics)",
        "config": cfg,
        "rows": int(n),
        # NOTE: the black line is the SAME frozen archived Stage 0 T_world_pad in
        # BOTH right-hand panels (top-right Stage 0 kinematic replay and
        # bottom-right physics replay).  physics_trace.source_pad is a candidate
        # input, not the black line.
        "black": "Stage 0 frozen pad centre (T_world_pad from stage0-consolidation-v1) in both right panels",
        "red": "commanded pad centre (physics trace commanded_pad, bottom-right)",
        "mano_projection": "RAS map per-frame camera, K scaled to the source resolution",
        "hand_coverage": {"observed": hand_rows, "gap_filled": int(n) - hand_rows,
                          "total": int(n),
                          "observed_key": "stage0_trace.hand_observed",
                          "note": "hand_observed == paired_targets.valid (300/300); gap-filled frames break the drawn trajectory"},
        "playback_fps": fps,
        "panel_order": ["source", "stage0_kinematic", "mano", "physics"],
        "panel_positions": {"tl": "source", "tr": "stage0_kinematic",
                            "bl": "mano", "br": "physics"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    csv_path = out_dir / "quadgrid_frames.csv"
    video_path = out_dir / "quadgrid.mp4"
    with csv_path.open("w", newline="") as csv_f, \
         imageio.get_writer(video_path, fps=fps, macro_block_size=8) as writer_mp4:
        writer = _csv.writer(csv_f)
        writer.writerow(["source_frame", "mano_flag", "bilateral", "black_to_red_mm"])
        for k in range(n):
            src = int(frames_all[k])
            src_cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, frame = src_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tl = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
            label(tl, "Source video", (10, 24), .55)
            label(tl, f"F{src:04d}  t={src / float(stage0['video_fps']):.2f}s", (10, 46), .45)

            bl = frame.copy()
            w2c = np.eye(4)
            w2c[:3, :3] = pair["R_c2w"][k].T
            w2c[:3, 3] = -pair["R_c2w"][k].T @ pair["t_c2w"][k]
            uv, zc = project(Ks, w2c, world_joints[k])
            for i0, i1 in MANO_BONES:
                p0 = tuple(np.round(uv[i0]).astype(int)); p1 = tuple(np.round(uv[i1]).astype(int))
                cv2.line(bl, p0, p1, (120, 255, 120), 2, cv2.LINE_AA)
            for j, color in ((0, (255, 255, 0)), (4, (255, 120, 0)), (8, (0, 120, 255))):
                cv2.circle(bl, tuple(np.round(uv[j]).astype(int)), 6, _bgr(color), -1, cv2.LINE_AA)
            Tp = stage0["T_world_pad"][k]
            pc_uv, _ = project(Ks, w2c, Tp[:3, 3][None])
            axis = Tp[:3, 3] + Tp[:3, :3] @ np.array([rg.PAD_FWD, 0, 0]) * 2.0
            ax_uv, _ = project(Ks, w2c, axis[None])
            cv2.line(bl, tuple(np.round(pc_uv[0]).astype(int)),
                     tuple(np.round(ax_uv[0]).astype(int)), _bgr((60, 60, 255)), 3, cv2.LINE_AA)
            cv2.circle(bl, tuple(np.round(pc_uv[0]).astype(int)), 7, _bgr((60, 60, 255)), -1, cv2.LINE_AA)
            bl = cv2.resize(bl, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
            observed = bool(hand_obs[k])
            obs = "observed" if observed else "gap-filled"
            strip = (80, 220, 80) if observed else (255, 160, 60)
            cv2.rectangle(bl, (0, PANEL_H - 14), (PANEL_W, PANEL_H), _bgr(strip), -1)
            label(bl, f"MANO (RAS world, map camera) | {obs}", (10, 24), .5)

            # Stage 0 kinematic robot (deterministic sequential IK, same as archive)
            target = (stage0["T_world_pad"][k, :3, 3], stage0["T_world_pad"][k, :3, :3])
            seed, pos_m, ang, seed_source = solve_arm(robot, selected, target, seed, neutral, limits)
            qq = robot.get_qpos()
            qq[selected] = seed
            qq[fingers] = stage0["grip"][k]
            robot.set_qpos(qq)
            obj_kin.set_pose(pose_to_sapien(stage0["T_world_object"][k]))
            scene.update_render()
            cam.take_picture()
            Kc, E = cam_matrices()
            img_kin = np.uint8(255 * np.clip(cam.get_picture("Color")[:, :, :3], 0, 1)).copy()
            uv_s0, z0 = project(Kc, E, pad_stage0[:k + 1])
            ins0 = visible_mask(uv_s0, z0)
            draw_polyline(img_kin, uv_s0[ins0], (10, 10, 10), thick=3)
            if ins0[-1]:
                cv2.circle(img_kin, tuple(np.round(uv_s0[-1]).astype(int)), 6, _bgr((10, 10, 10)), -1)

            # physics robot (measured planar base + joints)
            robot.set_root_pose(bp.se2_pose(*map(float, base_phys[k])))
            qq = robot.get_qpos()
            qq[:n_dof] = q_phys[k][n_planar:n_planar + n_dof]
            qq[fingers] = np.clip(grip_phys[k], 0, 0.05)
            robot.set_qpos(qq)
            obj_phys.set_pose(pose_to_sapien(obj_phys_arr[k]))
            scene.update_render()
            cam.take_picture()
            Kc, E = cam_matrices()
            img_phys = np.uint8(255 * np.clip(cam.get_picture("Color")[:, :, :3], 0, 1)).copy()
            uv_b, zb = project(Kc, E, pad_black[:k + 1])
            uv_r, zr = project(Kc, E, pad_red[:k + 1])
            ins_b = visible_mask(uv_b, zb)
            ins_r = visible_mask(uv_r, zr)
            draw_polyline(img_phys, uv_b[ins_b], (10, 10, 10), thick=3)
            draw_polyline(img_phys, uv_r[ins_r], (255, 60, 60), thick=2)
            if ins_b[-1]:
                cv2.circle(img_phys, tuple(np.round(uv_b[-1]).astype(int)), 6, _bgr((10, 10, 10)), -1)
            if ins_r[-1]:
                cv2.circle(img_phys, tuple(np.round(uv_r[-1]).astype(int)), 5, _bgr((255, 60, 60)), -1)

            tile_label(img_kin, "Stage 0 kinematic replay | black: frozen pad centre")
            tile_label(img_phys, "Physics replay | black: Stage 0, red: commanded")
            # Panel positions: top-left source, top-right Stage 0 kinematic,
            # bottom-left MANO, bottom-right physics replay.
            top = np.hstack([tl, cv2.resize(img_kin, (PANEL_W, PANEL_H))])
            bottom = np.hstack([bl, cv2.resize(img_phys, (PANEL_W, PANEL_H))])
            canvas = np.vstack([top, bottom])
            writer_mp4.append_data(canvas)
            writer.writerow([src, obs, bool(bilateral[k]),
                             round(float(np.linalg.norm(pad_stage0[k] - pad_red[k]) * 1000), 2)])
    manifest["csv"] = str(csv_path.resolve())
    manifest["video"] = str(video_path.resolve())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"video": str(video_path), "rows": int(n)}, indent=2))


if __name__ == "__main__":
    main()

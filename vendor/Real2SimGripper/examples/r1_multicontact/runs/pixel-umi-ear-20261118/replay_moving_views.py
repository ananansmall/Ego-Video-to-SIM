"""Replay a frozen Chairbottle execution trace with synchronized source and moving R1 views."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
REPLAYER = ROOT / "examples/chairbottle/chair-full-luna-20261116/replay_multicycle.py"
W, H = 640, 360


def load_replayer():
    spec = importlib.util.spec_from_file_location("chair_replay", REPLAYER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def label(img, title, caption):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (W - 1, 45), (20, 22, 25), -1)
    cv2.putText(img, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .59, (251, 251, 251), 1, cv2.LINE_AA)
    cv2.putText(img, caption, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, .45, (234, 234, 234), 1, cv2.LINE_AA)
    return img


def letterbox(im):
    h, w = im.shape[:2]
    s = min(W/w, H/h)
    out = np.zeros((H, W, 3), np.uint8)
    w1, h1 = round(s*w), round(s*h)
    x, y = (W-w1)//2, (H-h1)//2
    out[y:y+h1, x:x+w1] = cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB), (w1,h1), interpolation=cv2.INTER_AREA)
    return out


def camera_rgb(cam):
    cam.take_picture()
    return np.uint8(255*np.clip(cam.get_picture("Color")[:, :, :3], 0, 1))


def project_trail(points, camera_pose, intrinsics):
    local = (points - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    optical = local[:, [1, 2, 0]] * [-1., -1., 1.]
    if np.any(optical[:, 2] < .025):
        raise ValueError("the overview camera must see the full gripper trajectory")
    xy = optical @ intrinsics.T
    return np.rint(xy[:, :2] / xy[:, 2, None]).astype(np.int32)


def draw_trails(img, target, actual, i):
    img = img.copy()
    start = max(0, i-44)
    for points, color in ((target, (14, 14, 16)), (actual, (224, 33, 42))):
        if i > 0:
            segment = points[start:i+1]
            cv2.polylines(img, [segment], False, (251, 251, 247), 5, cv2.LINE_AA)
            cv2.polylines(img, [segment], False, color, 3, cv2.LINE_AA)
        cv2.circle(img, tuple(points[i]), 4, color, -1, cv2.LINE_AA)
    return img


def feature_rows(trace, fps, base_origin, finger_indices):
    ids = np.asarray(trace["source_frames"], int)
    t = np.cumsum(trace["physics_substeps"]) * float(trace["dt"])
    dt = np.diff(t, prepend=t[0])
    pad = trace["actual_pad"]
    target = trace["target_pad"]
    poses = pad[:, :3, 3]
    rot = Rotation.from_matrix(pad[:, :3, :3])
    target_rot = Rotation.from_matrix(target[:, :3, :3])
    q = rot.as_quat()
    obj = trace["actual_object"][:, :3, 3]
    base = trace["base_se2"].copy()
    base[:, :2] += base_origin[:2]
    dyaw = np.rad2deg(np.diff(np.unwrap(base[:, 2]), prepend=base[0,2]))
    base_dp = np.linalg.norm(np.diff(base[:, :2], axis=0, prepend=base[:1,:2]), axis=1)
    pad_dp = np.linalg.norm(np.diff(poses, axis=0, prepend=poses[:1]), axis=1)
    pad_dr = np.r_[0., np.degrees((rot[:-1].inv()*rot[1:]).magnitude())]
    with np.errstate(divide="ignore", invalid="ignore"):
        bv = np.divide(base_dp, dt, out=np.zeros_like(t), where=dt>0)
        yr = np.divide(dyaw, dt, out=np.zeros_like(t), where=dt>0)
        pv = np.divide(pad_dp, dt, out=np.zeros_like(t), where=dt>0)
        wr = np.divide(pad_dr, dt, out=np.zeros_like(t), where=dt>0)
    error_pos = np.linalg.norm(poses-target[:,:3,3], axis=1)*1000
    error_rot = np.degrees((target_rot.inv()*rot).magnitude())
    fingers = trace["q_actual"][:, finger_indices]
    return np.column_stack((ids, ids/fps, t, base[:,:2], np.degrees(np.unwrap(base[:,2])),
                            bv, yr, poses, q, pv, wr, *target[:,:3,3].T, error_pos, error_rot,
                            fingers, obj, trace["bilateral"].astype(int)))


FIELDS = (
    "source_frame source_time_s sim_time_s base_x_m base_y_m base_yaw_offset_unwrapped_deg "
    "base_speed_m_s base_yaw_rate_deg_s pad_x_m pad_y_m pad_z_m "
    "pad_qx pad_qy pad_qz pad_qw pad_speed_m_s pad_rotation_rate_deg_s "
    "command_pad_x_m command_pad_y_m command_pad_z_m actual_vs_command_mm actual_vs_command_deg "
    "right_gripper_finger1_joint_m right_gripper_finger2_joint_m object_x_m object_y_m object_z_m bilateral"
).split()


def run(package, trace_path, output):
    sim = load_replayer()
    rg, bp = sim.rg, sim.bp
    geom = json.loads((package/"geometry.json").read_text())
    source = Path(geom["source_video_path"])
    if sha(source) != geom["source_video_sha256"]:
        raise ValueError("source SHA differs from the frozen package")
    with np.load(package/"paired_targets.npz",allow_pickle=False) as pairs, np.load(trace_path,allow_pickle=False) as z:
        trace = {k:z[k].copy() for k in z.files}
        if not np.array_equal(pairs["src_frames"], trace["source_frames"]):
            raise ValueError("source-frame indices differ between the pair and physics trace")
        if str(trace["package_sha256"]) != sim.package_sha256(package):
            raise ValueError("trace belongs to a different package")
        ids = np.asarray(trace["source_frames"], int)
        fps = float(pairs["video_fps"])
        frozen_pad = pairs["T_world_pad"][:, :3, 3]
    if np.any(np.diff(ids) != 1):
        raise ValueError("this playback expects contiguous source frames")
    fps_out = len(ids)/sum(trace["physics_substeps"]*float(trace["dt"]))
    scene, robot, material = sim.build_scene(geom["supports"], scene_meshes=package/"scene_assets")
    stage = scene.create_actor_builder()
    stage.add_visual_from_file(str(package/"scene_assets/final_scene_world.glb"))
    stage.build_static(name="same_video_scene")
    obj = sim.make_rigid_body(scene, package/"case_metric.obj", trace["actual_object"][0],
                              package/"material_0.png", material)
    torso, wrist = (robot.find_link_by_name(name) for name in ("torso_link4","right_gripper_link"))
    chest, hand, overview = (scene.add_camera(n, W, H, np.deg2rad(f), .025, 30)
                             for n,f in (("torso_fpv",67),("right_wrist_fpv",88),("overview",56)))
    torso_mount = rg.look_at_pose([.20, 0., .09], [1.0, 0., -.24]).to_transformation_matrix()
    wrist_mount = rg.look_at_pose([.005,-.095,.085],[rg.PAD_FWD,0.,-.012]).to_transformation_matrix()
    center = np.mean([s["center"] for s in geom["supports"].values()], axis=0)
    center[2] = .7
    overview_pose = rg.look_at_pose(center+[-2.7,-2.1,1.45], center)
    overview.entity.set_pose(overview_pose)
    extr = overview_pose.to_transformation_matrix()
    intr = overview.get_intrinsic_matrix()
    target_trail = project_trail(frozen_pad, extr, intr)
    actual_trail = project_trail(trace["actual_pad"][:, :3, 3], extr, intr)
    camera_states = []
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened() or len(ids) < 2:
        raise ValueError("cannot open original first-person video or invalid frame count")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    output.mkdir(parents=True, exist_ok=True)
    rows = feature_rows(trace, fps, np.asarray(rg.R1_BASE_P), trace["finger_joint_indices"])
    assert rows.shape == (len(ids),len(FIELDS))
    config = dict(fps=fps_out, codec="libx264", quality=7, macro_block_size=2,
                  ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart","-threads","2"])
    with ExitStack() as stack:
        writers = [stack.enter_context(imageio.get_writer(output/name, **config)) for name in (
            "moving_views.mp4", "source_ego.mp4", "r1_body_fpv.mp4","r1_wrist_fpv.mp4")]
        for i, frame in enumerate(ids):
            ok, bgr = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
                raise ValueError(f"not the requested original frame {frame}")
            robot.set_root_pose(bp.se2_pose(*trace["base_se2"][i]))
            robot.set_qpos(trace["q_actual"][i])
            obj.set_pose(sapien.Pose(trace["actual_object"][i]))
            t_torso = torso.get_entity_pose().to_transformation_matrix() @ torso_mount
            t_wrist = wrist.get_entity_pose().to_transformation_matrix() @ wrist_mount
            chest.entity.set_pose(sapien.Pose(t_torso))
            hand.entity.set_pose(sapien.Pose(t_wrist))
            scene.update_render()
            source_panel = letterbox(bgr)
            scene_imgs = [camera_rgb(c) for c in (chest,hand,overview)]
            scene_imgs[-1] = draw_trails(scene_imgs[-1], target_trail, actual_trail, i)
            base_line = f"Base {rows[i,6]:.2f} m/s | yaw rate {rows[i,7]:+.1f} deg/s"
            captions = [f"Source F{frame:03d} | {frame/fps:.2f}s",
                        base_line, f"Pad {rows[i,15]:.2f} m/s | {rows[i,16]:.1f} deg/s",
                        f"F{frame:03d} | black: frozen pair target | red: actual pads"]
            panels = [label(im,title,caption) for im,title,caption in zip(
                [source_panel,*scene_imgs], ["Original first person","R1 moving torso view (virtual)",
                "R1 moving right wrist view","R1 external view"],captions,strict=True)]
            writers[0].append_data(np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:]))))
            for writer, panel in zip(writers[1:], panels[:3], strict=True):
                writer.append_data(panel)
            camera_states.append([t_torso,t_wrist])
    capture.release()
    with (output/"motion_features.csv").open("w",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows([int(r[0]), *map(lambda x:f"{x:.8g}",r[1:-1]),int(r[-1])] for r in rows)
    np.savez_compressed(output/"camera_poses.npz",source_frames=ids,sim_time_s=rows[:,2],
                        T_world_sapien_camera=np.array(camera_states),
                        K=np.stack([c.get_intrinsic_matrix() for c in (chest,hand)]))
    summary = dict(frames=len(ids),source_range=ids[[0,-1]].tolist(),playback_fps=fps_out,
                   base_path_m=float(np.linalg.norm(np.diff(rows[:,3:5],axis=0),axis=1).sum()),
                   pad_path_m=float(np.linalg.norm(np.diff(rows[:,8:11],axis=0),axis=1).sum()),
                   torso_mount=torso_mount.tolist(), wrist_mount=wrist_mount.tolist(),
                   camera_axes="+X forward, +Y left, +Z up; columns torso virtual, right wrist virtual",
                   scope="Original source pixels and virtual cameras attached to replayed planar base plus measured joints. Source-camera-to-R1 metric registration is not asserted.",
                   trail="45-frame history at pad center; black frozen pair target, red actual executed pad",
                   source=str(source),source_sha256=sha(source),trace=str(trace_path),trace_sha256=sha(trace_path),
                   package_sha256=str(trace["package_sha256"]))
    (output/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package","trace","output"):
        parser.add_argument(name,type=Path)
    a = parser.parse_args()
    run(a.package.resolve(),a.trace.resolve(),a.output.resolve())

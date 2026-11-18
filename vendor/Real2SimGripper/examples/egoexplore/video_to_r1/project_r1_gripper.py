"""Reproject camera-space R1 retarget candidates into their original video frames."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh


def vector(text):
    return np.fromstring(text, sep=" ")


def origin(element):
    node = element.find("origin")
    xyz = vector(node.get("xyz", "0 0 0")) if node is not None else np.zeros(3)
    rpy = vector(node.get("rpy", "0 0 0")) if node is not None else np.zeros(3)
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def load_gripper(urdf, arm):
    tree = ET.parse(urdf)
    links = [f"{arm}_gripper_link", *(f"{arm}_gripper_finger_link{i}" for i in (1, 2))]
    meshes = []
    for name in links:
        visual = tree.find(f"./link[@name='{name}']/visual")
        mesh = visual.find("geometry/mesh")
        transform, position = origin(visual)
        vertices = np.asarray(trimesh.load(urdf.parent / mesh.get("filename"), force="mesh").convex_hull.vertices)
        vertices = vertices * vector(mesh.get("scale", "1 1 1"))
        meshes.append(vertices @ transform.T + position)
    joints = [tree.find(f"./joint[@name='{arm}_gripper_finger_joint{i}']") for i in (1, 2)]
    assert all(joint.get("type") == "prismatic" and joint.find("parent").get("link") == links[0] for joint in joints)
    axes = np.stack([vector(joint.find("axis").get("xyz")) for joint in joints])
    offsets = [origin(joint) for joint in joints]
    assert np.allclose(axes[0], -axes[1])
    return meshes, axes, offsets


def project(points, intrinsic):
    homogeneous = np.asarray(points) @ intrinsic.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def draw(view, meshes, finger_axes, offsets, hand, intrinsic, estimate, color, centered):
    wrist, thumb, index = hand
    legacy_pad, rotation, opening = estimate
    hand_center = (thumb + index) / 2
    joints = np.stack([p + r @ a * opening for (r, p), a in zip(offsets, finger_axes)])
    neutral_pad_offset = np.mean([p for _, p in offsets], axis=0)
    anchor = hand_center if centered else legacy_pad
    base = anchor - rotation @ neutral_pad_offset
    mesh_vertices = [meshes[0], *(v @ r.T + p for v, (r, _), p in zip(meshes[1:], offsets, joints))]
    for part, vertices in enumerate(mesh_vertices):
        camera = vertices @ rotation.T + base
        if np.min(camera[:, 2]) <= 1e-4:
            continue
        pixels = project(camera, intrinsic)
        polygon = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
        cv2.polylines(view, [polygon], True, color, 1 if part == 0 else 2, cv2.LINE_AA)
    points = joints @ rotation.T + base
    approach = anchor + .070 * rotation[:, 0]
    if np.min(np.r_[points[:, 2], anchor[2], approach[2]]) > 1e-4:
        joint_px = np.rint(project(points, intrinsic)).astype(int)
        p, direction = np.rint(project([anchor, approach], intrinsic)).astype(int)
        cv2.arrowedLine(view, tuple(p), tuple(direction), color, 2, cv2.LINE_AA, tipLength=.23)
        cv2.line(view, tuple(joint_px[0]), tuple(joint_px[1]), color, 1, cv2.LINE_AA)
        for f in joint_px:
            cv2.circle(view, tuple(f), 4, color, 1, cv2.LINE_AA)
    for pixel in np.rint(project([thumb, index], intrinsic)).astype(int):
        cv2.circle(view, tuple(pixel), 3, (250, 250, 250), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--interval", type=int, nargs=2, action="append", metavar=("FIRST", "LAST"))
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--mask-track", type=Path, help="SAM3 exact source-frame detections")
    parser.add_argument("--track-id", type=int)
    args = parser.parse_args()
    from_legacy = Path(__file__).resolve().parents[1] / "ep5/replay-oven-agile.fill-high"
    sys.path.insert(0, str(from_legacy))
    from cb_svd_retarget import svd_gripper_pose

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if args.video.resolve() != Path(data["video_path"]).resolve():
        raise ValueError("Source video differs from EgoFound3R checkpoint")
    frames = np.asarray(data["source_frame_indices"], dtype=int)
    wrists_and_tips = np.asarray(data["hand_joint_xyz_metric_full"], dtype=float)[:, :, [0, 4, 8]]
    intrinsic = np.asarray(data["hand_projection_K_full"], dtype=float).copy()
    valid = np.asarray(data["hand_metric_valid_full"]) & ~np.asarray(data["hand_missing_fill_full"])
    valid &= np.asarray(data["hand_projection_K_valid_full"])[:, None]
    valid &= np.isfinite(wrists_and_tips).all(axis=(2, 3)) & (wrists_and_tips[:, :, :, 2] > 1e-4).all(axis=2)
    meshes, axes, offsets = load_gripper(args.urdf, args.arm)
    video = cv2.VideoCapture(str(args.video))
    width_in, height_in = (int(video.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.array_equal(frames, np.arange(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        raise ValueError("Checkpoint frame IDs do not match the source video")
    height = int(round(height_in / width_in * args.width / 2) * 2)
    fps = float(video.get(cv2.CAP_PROP_FPS))
    model_height, model_width = data["image_size_hw"]
    intrinsic[:, 0] *= args.width / model_width
    intrinsic[:, 1] *= height / model_height

    args.out.mkdir(parents=True, exist_ok=True)
    mask_paths = {}
    if args.mask_track:
        if args.track_id is None:
            raise ValueError("--mask-track requires --track-id")
        mask_source = json.loads((args.mask_track / "frames.json").read_text())["source"]
        if args.video.resolve() != Path(mask_source).resolve():
            raise ValueError("SAM3 track belongs to a different source video")
        with (args.mask_track / "tracks.csv").open() as stream:
            mask_paths = {int(row["source_frame"]): args.mask_track / row["mask"]
                          for row in csv.DictReader(stream) if int(row["track_id"]) == args.track_id}
        if not mask_paths:
            raise ValueError("Requested SAM3 track has no valid masks")
    rows, estimates = [], {}
    sides = ("left", "right")
    for frame, side in np.argwhere(valid):
        hand = wrists_and_tips[frame, side]
        if np.linalg.norm(hand[1] - hand[2]) < 1e-5 or np.linalg.norm(hand[1:].mean(0) - hand[0]) < 1e-5:
            continue
        estimate = svd_gripper_pose(*hand, prefix=args.arm)
        estimates[frame, side] = estimate
        center = hand[1:].mean(0)
        shift = np.linalg.norm(estimate[0] - center)
        pixels = project([estimate[0], center], intrinsic[frame])
        rows.append(dict(source_frame=int(frame), hand=sides[side], opening_mm=1000 * estimate[2],
                         center_shift_mm=1000 * shift, image_shift_px=float(np.linalg.norm(pixels[0] - pixels[1]))))

    intervals = args.interval or [[int(frames[0]), int(frames[-1])]]
    report = dict(source=str(args.video.resolve()), checkpoint=str(args.checkpoint.resolve()),
                  urdf=str(args.urdf.resolve()), urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                  width_each=args.width, height_each=height, fps=fps, source_intervals=intervals,
                  status="camera_space_retarget_candidate",
                  definition="Existing SVD pad, versus same orientation and opening centered on observed fingertips. URDF mesh outline with prismatic joint origins. White dots are EF3R estimates.",
                  mask_track=str(args.mask_track.resolve()) if args.mask_track else None, mask_track_id=args.track_id,
                  estimated_rows=len(rows), world_registration_used=False, physical_contact_measured=False,
                  coverage=[{side: sum(lo <= row["source_frame"] <= hi and row["hand"] == side for row in rows) for side in sides} for lo, hi in intervals])
    with (args.out / "gripper_reprojection.csv").open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=["source_frame", "hand", "opening_mm", "center_shift_mm", "image_shift_px"])
        output.writeheader()
        output.writerows(rows)

    ffmpeg = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
                               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{2*args.width}x{height}",
                               "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "4",
                               "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                               str(args.out / "gripper_reprojection.mp4")], stdin=subprocess.PIPE)
    drawn_source_frames = []
    colors = ((245, 195, 55), (40, 180, 255))
    for lo, hi in intervals:
        if not 0 <= lo <= hi < len(frames):
            raise ValueError(f"Invalid preview interval [{lo}, {hi}]")
        video.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for frame in range(lo, hi + 1):
            success, raw = video.read()
            if not success or int(video.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                raise ValueError(f"Cannot decode exact source frame {frame}")
            source = cv2.resize(raw, (args.width, height), interpolation=cv2.INTER_AREA)
            if frame in mask_paths:
                mask = cv2.imread(str(mask_paths[frame]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Missing mask for source frame {frame}")
                mask = cv2.resize(mask, (args.width, height), interpolation=cv2.INTER_NEAREST)>0
                source[mask] = source[mask] * .65 + np.array([65, 220, 45]) * .35
            panels = [source.copy(), source.copy()]
            for centered, panel in enumerate(panels):
                for side in (0, 1):
                    if (frame, side) in estimates:
                        draw(panel, meshes, axes, offsets, wrists_and_tips[frame, side], intrinsic[frame],
                             estimates[frame, side], colors[side], bool(centered))
                cv2.rectangle(panel, (0, 0), (args.width, 51), (18, 18, 18), -1)
                label = "B  Centered URDF jaws" if centered else "A  Existing retarget"
                cv2.putText(panel, f"{label}    source F{frame}   {frame/fps:.2f}s", (12, 21), 0, .54, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(panel, "Blue: left hand  Amber: right hand  White: estimated fingertips", (12, 42),
                            0, .46, (240, 240, 240), 1, cv2.LINE_AA)
                if frame in mask_paths:
                    cv2.putText(panel, "Green: SAM3 object", (args.width-175, 42),
                                0, .46, (65, 220, 45), 1, cv2.LINE_AA)
            ffmpeg.stdin.write(np.concatenate(panels, axis=1).tobytes())
            drawn_source_frames.append(frame)
    video.release()
    ffmpeg.stdin.close()
    if ffmpeg.wait() or not drawn_source_frames:
        raise RuntimeError("Failed to render the source overlay")
    report["rendered_source_frames"] = drawn_source_frames
    (args.out / "gripper_reprojection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

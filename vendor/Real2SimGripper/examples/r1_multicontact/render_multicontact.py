#!/usr/bin/env python3
"""Render a generic multi-contact trace as four views and robot wrist FPV."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from cb_large_range_sim import rg
from replay_multicontact import add_object, add_supports, asset, resolved_supports


def render(package, output, source=None):
    targets = np.load(package/"paired_targets.npz")
    meta = json.loads((package/"geometry.json").read_text())
    trace_path = output/"multicontact_trace.npz"
    with np.load(trace_path) as trace:
        frames = trace["source_frames"].copy()
        qpos = trace["q_actual"].copy()
        objects = trace["actual_object"].copy()
        playback_fps = float(trace["playback_fps"] if "playback_fps" in trace else targets["video_fps"])
    supports = resolved_supports(package, meta["supports"])
    ids = np.asarray(targets["object_ids"]).astype(str)
    object_id = ids[0]
    scene, robot = __import__("cb_large_range_sim")._build_scene(planar_base=True)
    material = scene.create_physical_material(1., .9, 0.)
    if "scene_visual" in meta:
        supports_for_render = {name: {k:v for k,v in item.items() if k != "visual_mesh"}
                               for name,item in supports.items()}
    else:
        supports_for_render = supports
    add_supports(scene, supports_for_render, material, float(meta.get("support_height_m", .045)))
    if "scene_visual" in meta:
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(asset(package, meta["scene_visual"])))
        builder.build_static(name="source_scene_visual")
    body = add_object(scene, package, object_id, meta["objects"][object_id],
                      objects[0], material)
    front, side, fpv, overhead = [
        scene.add_camera(name, 480, 360, np.deg2rad(fov), .025, 30.)
        for name, fov in (("front", 51), ("side", 51),
                          ("robot_fpv", 88), ("overhead", 54))]
    centers = np.array([d["center"] for d in supports.values()], dtype=float)
    middle = centers.mean(axis=0)
    middle[2] += float(meta.get("render_target_height_m", .5))
    framing = max(1., float(np.linalg.norm(centers[:,:2]-middle[:2], axis=1).max()/.7))
    front.entity.set_pose(rg.look_at_pose(middle+framing*np.array([-1.55, -1.35, .95]), middle))
    side.entity.set_pose(rg.look_at_pose(middle+framing*np.array([1.68, -.39, .85]), middle))
    overhead.entity.set_pose(rg.look_at_pose(middle+framing*np.array([.03, -.10, 2.10]), middle))
    camera_to_gripper = rg.look_at_pose(
        [.005, -.095, .085], [rg.PAD_FWD, 0., -.012]).to_transformation_matrix()
    output.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output/"four_views.mp4", fps=playback_fps, macro_block_size=8) as tiled, \
         imageio.get_writer(output/"robot_fpv.mp4", fps=playback_fps, macro_block_size=8) as wrist:
        for q, obj in zip(qpos, objects):
            robot.set_root_pose(sapien.Pose())
            robot.set_qpos(q)
            body.set_pose(sapien.Pose(p=obj[:3, 3],
                q=Rotation.from_matrix(obj[:3, :3]).as_quat()[[3, 0, 1, 2]]))
            hand = robot.find_link_by_name("right_gripper_link").get_entity_pose().to_transformation_matrix()
            fpv.entity.set_pose(sapien.Pose(hand @ camera_to_gripper))
            scene.update_render()
            pictures = []
            for view in (front, side, fpv, overhead):
                view.take_picture()
                pictures.append(np.uint8(255*np.clip(view.get_picture("Color")[:, :, :3], 0, 1)))
            tiled.append_data(np.vstack((np.hstack(pictures[:2]), np.hstack(pictures[2:]))))
            wrist.append_data(pictures[2])
    if source is not None:
        sys.path.insert(0, str(ROOT/"scripts"))
        from render_source_and_sim import compose
        fps = float(targets["video_fps"])
        compose(source, output/"four_views.mp4", output/"source_and_sim.mp4",
                ("Front", "Side", "Robot FPV", "Overhead"),
                fps, int(frames[0]), int(frames[-1])+1,
                trace=trace_path, playback_fps=playback_fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    render(args.package.resolve(), args.output.resolve(), args.source)


if __name__ == "__main__":
    main()


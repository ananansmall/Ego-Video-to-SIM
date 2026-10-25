"""
物体交互追踪管线入口 (run_pipeline.py)
========================================

管线步骤:
  Step 0: GLB-视频对齐验证 (01_glb_video_align.py)
  Step 1: VGGT4D+TrackHead 联合点追踪 (02_point_tracker.py)
  Step 2: 运动耦合检测 + 抓取时序 (03_grasp_controller.py)
  Step 3: 精确6DoF轨迹估计 (04_trajectory_refiner.py)
  Step 4: 闭环验证 (05_closed_loop_verifier.py)

用法:
  python run_pipeline.py --video /path/to/beizi.mp4
  python run_pipeline.py --video /path/to/beizi.mp4 --skip_simulation
  python run_pipeline.py --video /path/to/beizi.mp4 --skip_closed_loop
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAWOR_ROOT = os.path.join(PROJECT_ROOT, "HaWoR")
RAS_ROOT = os.path.join(PROJECT_ROOT, "ReplicateAnyScene")

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    path = os.path.join(_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod_ga = _load_module("glb_video_align", "01_glb_video_align.py")
_mod_pt = _load_module("point_tracker", "02_point_tracker.py")
_mod_gc = _load_module("grasp_controller", "03_grasp_controller.py")
_mod_tr = _load_module("trajectory_refiner", "04_trajectory_refiner.py")
_mod_cl = _load_module("closed_loop_verifier", "05_closed_loop_verifier.py")

align_glb_to_video = _mod_ga.align_glb_to_video
run_point_tracking = _mod_pt.run_point_tracking
run_grasp_controller = _mod_gc.run_grasp_controller
run_trajectory_refiner = _mod_tr.run_trajectory_refiner
run_closed_loop_verification = _mod_cl.run_closed_loop_verification


def resolve_video_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def find_hawor_output(video_name, base_dir):
    import glob

    candidates = [
        os.path.join(base_dir, video_name, "hawor_results.npz"),
        os.path.join(base_dir, video_name, "hawor_results_0_999.npz"),
    ]
    hawor_example = os.path.join(HAWOR_ROOT, "example", video_name, "reconstruction")
    if os.path.isdir(hawor_example):
        candidates.extend(glob.glob(os.path.join(hawor_example, "hawor_results*.npz")))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_hawor_results(npz_path):
    data = dict(np.load(npz_path, allow_pickle=True))
    return {
        "pred_trans": data["pred_trans"],
        "pred_rot": data["pred_rot"],
        "pred_hand_pose": data["pred_hand_pose"],
        "pred_betas": data["pred_betas"],
        "pred_valid": data.get("pred_valid", None),
        "R_c2w": data.get("R_c2w", None),
        "t_c2w": data.get("t_c2w", None),
        "img_focal": float(data.get("img_focal", 0)),
    }


def find_hawor_hand_masks(video_name, base_dir):
    import glob

    candidates = [
        os.path.join(base_dir, video_name, "model_masks.npy"),
    ]

    hawor_example = os.path.join(HAWOR_ROOT, "example", video_name)
    if os.path.isdir(hawor_example):
        candidates.extend(glob.glob(os.path.join(hawor_example, "tracks_*", "model_masks.npy")))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def generate_mano_vertices(pred_trans, pred_rot, pred_hand_pose, pred_betas):
    FINGERTIP_INDICES = [744, 320, 443, 554, 671]

    try:
        import torch
        sys.path.insert(0, os.path.join(HAWOR_ROOT))
        from hawor.utils.process import run_mano, run_mano_left

        T = pred_trans.shape[1]

        trans_r = torch.tensor(pred_trans[1:2]).float()
        rot_r = torch.tensor(pred_rot[1:2]).float()
        pose_r = torch.tensor(pred_hand_pose[1:2]).float()
        betas_r = torch.tensor(pred_betas[1:2]).float()

        out_r = run_mano(trans_r, rot_r, pose_r, betas=betas_r)
        hand_vertices_right = out_r["vertices"][0].detach().cpu().numpy()

        trans_l = torch.tensor(pred_trans[0:1]).float()
        rot_l = torch.tensor(pred_rot[0:1]).float()
        pose_l = torch.tensor(pred_hand_pose[0:1]).float()
        betas_l = torch.tensor(pred_betas[0:1]).float()

        out_l = run_mano_left(trans_l, rot_l, pose_l, betas=betas_l)
        hand_vertices_left = out_l["vertices"][0].detach().cpu().numpy()

        fingertips_3d_right = hand_vertices_right[:, FINGERTIP_INDICES]
        fingertips_3d_left = hand_vertices_left[:, FINGERTIP_INDICES]

        return {
            "hand_vertices_left": hand_vertices_left,
            "hand_vertices_right": hand_vertices_right,
            "fingertips_3d_left": fingertips_3d_left,
            "fingertips_3d_right": fingertips_3d_right,
        }
    except Exception as e:
        print(f"[run_pipeline] WARNING: Failed to generate MANO vertices: {e}")
        T = pred_trans.shape[1]
        return {
            "hand_vertices_left": None,
            "hand_vertices_right": None,
            "fingertips_3d_left": None,
            "fingertips_3d_right": None,
        }


def find_object_masks(video_name, base_dir):
    for ext in ["npz", "json"]:
        p = os.path.join(base_dir, video_name, f"object_masks.{ext}")
        if os.path.isfile(p):
            return p
    return None


def load_object_masks(mask_path):
    if mask_path.endswith(".npz"):
        data = np.load(mask_path, allow_pickle=True)
        return data["masks"].item()
    elif mask_path.endswith(".json"):
        with open(mask_path, "r") as f:
            masks_info = json.load(f)
        object_masks = {}
        for cat, instances in masks_info.items():
            object_masks[cat] = []
            for inst in instances:
                inst_masks = []
                for m in inst:
                    mask_arr = np.array(m["mask"], dtype=bool) if isinstance(m["mask"], list) else m["mask"]
                    inst_masks.append({"frame_id": m["frame_id"], "mask": mask_arr})
                object_masks[cat].append(inst_masks)
        return object_masks
    else:
        raise ValueError(f"Unsupported mask file format: {mask_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precise Object Tracking Pipeline\n"
                    "VGGT4D TrackHead + Motion Coupling + Closed-Loop Verification"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="输入 MP4 视频文件路径")
    parser.add_argument("--base_dir", type=str,
                        default=os.path.join(RAS_ROOT, "outputs"),
                        help="中间文件基础目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: {base_dir}/{video_name}_precise/)")
    parser.add_argument("--hawor_npz", type=str, default=None,
                        help="手动指定 HaWoR npz 文件")
    parser.add_argument("--masks_file", type=str, default=None,
                        help="手动指定物体 mask 文件")
    parser.add_argument("--n_query_points", type=int, default=64,
                        help="每个物体的查询点数 (默认 64)")
    parser.add_argument("--query_method", type=str, default="grid",
                        choices=["grid", "random", "contour"],
                        help="查询点采样方法")
    parser.add_argument("--distance_threshold", type=float, default=0.15,
                        help="抓取距离阈值 (米)")
    parser.add_argument("--coupling_threshold", type=float, default=0.5,
                        help="运动耦合度阈值")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="VGGT4D 模型权重路径")
    parser.add_argument("--skip_simulation", action="store_true",
                        help="跳过仿真执行")
    parser.add_argument("--skip_closed_loop", action="store_true",
                        help="跳过闭环验证")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    args = parser.parse_args()

    video_name = resolve_video_name(args.video)

    if args.output is None:
        args.output = os.path.join(args.base_dir, f"{video_name}_precise")
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print(f"  Precise Object Tracking Pipeline")
    print(f"  Video: {args.video}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    # ── Step 0: GLB-视频对齐验证 ──
    print("\n" + "=" * 70)
    print("Step 0: GLB-Video Alignment Verification")
    print("=" * 70)

    glb_path = os.path.join(args.base_dir, video_name, "final_scene_stage5.glb")
    extrinsics_dir = os.path.join(args.base_dir, video_name, "extrinsics")
    intrinsic_path = os.path.join(args.base_dir, video_name, "intrinsic.txt")

    if os.path.isfile(glb_path) and os.path.isdir(extrinsics_dir):
        print(f"  GLB: {glb_path}")
        print(f"  Extrinsics: {extrinsics_dir}")

        extrinsics = _mod_ga.load_extrinsics_from_dir(extrinsics_dir)
        intrinsic = _mod_ga.load_intrinsic(intrinsic_path)

        align_dir = os.path.join(args.output, "alignment")
        align_result = align_glb_to_video(
            glb_path=glb_path,
            video_path_or_frames=args.video,
            extrinsics=extrinsics,
            intrinsic=intrinsic,
            output_dir=align_dir,
            render_edges=True,
            save_video=True,
            frame_indices=list(range(0, len(extrinsics), max(1, len(extrinsics) // 10))),
        )
        print(f"  Alignment: {align_result['n_frames']} frames rendered")
    else:
        print(f"  WARNING: GLB or extrinsics not found, skipping alignment")
        print(f"    GLB: {glb_path} ({'exists' if os.path.isfile(glb_path) else 'MISSING'})")
        print(f"    Extrinsics: {extrinsics_dir} ({'exists' if os.path.isdir(extrinsics_dir) else 'MISSING'})")

    # ── Step 1: VGGT4D + TrackHead 联合点追踪 ──
    print("\n" + "=" * 70)
    print("Step 1: VGGT4D + TrackHead Joint Point Tracking")
    print("=" * 70)

    masks_path = args.masks_file or find_object_masks(video_name, args.base_dir)
    object_masks = None
    if masks_path:
        print(f"  Loading object masks from: {masks_path}")
        object_masks = load_object_masks(masks_path)

    hand_masks_path = find_hawor_hand_masks(video_name, args.base_dir)
    hand_masks = None
    if hand_masks_path:
        print(f"  Loading hand masks from: {hand_masks_path}")
        hand_masks = np.load(hand_masks_path, allow_pickle=True)
        print(f"  Hand masks shape: {hand_masks.shape}")

    tracking_result = run_point_tracking(
        video_path_or_images=args.video,
        object_masks=object_masks,
        hand_masks=hand_masks,
        n_query_points=args.n_query_points,
        query_point_method=args.query_method,
        ckpt_path=args.ckpt_path,
        device=args.device,
        output_dir=args.output,
    )

    vggt_pred = tracking_result["vggt_predictions"]
    dynamic_mask = tracking_result["dynamic_mask"]
    object_tracks = tracking_result["object_tracks"]

    if not object_tracks:
        print("\nERROR: No object tracks obtained. Check VGGT4D output and dynamic mask.")
        print("Possible causes:")
        print("  1. VGGT4D model not loaded (check ckpt_path)")
        print("  2. No dynamic region detected in video")
        print("  3. Query point sampling failed")
        return

    for obj_key, track_data in object_tracks.items():
        t3d = track_data["tracks_3d"]
        v3d = track_data["valid_3d"]
        print(f"  Object '{obj_key}': {t3d.shape[0]} frames, {t3d.shape[1]} points, "
              f"{v3d.sum()}/{v3d.size} valid 3D")

    # ── Step 2: HaWoR 手部重建 ──
    print("\n" + "=" * 70)
    print("Step 2: Load HaWoR Hand Reconstruction")
    print("=" * 70)

    hawor_path = args.hawor_npz or find_hawor_output(video_name, args.base_dir)
    hand_data = {
        "hand_vertices_left": None,
        "hand_vertices_right": None,
        "fingertips_3d_left": None,
        "fingertips_3d_right": None,
    }

    if hawor_path:
        print(f"  Loading from: {hawor_path}")
        hawor_results = load_hawor_results(hawor_path)
        hand_data = generate_mano_vertices(
            hawor_results["pred_trans"],
            hawor_results["pred_rot"],
            hawor_results["pred_hand_pose"],
            hawor_results["pred_betas"],
        )
        if hand_data["hand_vertices_right"] is not None:
            print(f"  Right hand vertices: {hand_data['hand_vertices_right'].shape}")
        if hand_data["hand_vertices_left"] is not None:
            print(f"  Left hand vertices: {hand_data['hand_vertices_left'].shape}")
    else:
        print(f"  WARNING: HaWoR output not found for '{video_name}'")
        print(f"  Run first: python HaWoR/demov2.py --video {args.video}")

    # ── Step 3: 运动耦合检测 + 抓取时序 ──
    print("\n" + "=" * 70)
    print("Step 3: Motion Coupling Detection + Grasp Timing")
    print("=" * 70)

    grasp_results = {}
    for obj_key, track_data in object_tracks.items():
        print(f"\n  Processing '{obj_key}'...")

        tracks_3d = track_data["tracks_3d"]
        valid_3d = track_data["valid_3d"]

        grasp_result = run_grasp_controller(
            object_tracks_3d=tracks_3d,
            object_valid=valid_3d,
            hand_vertices_left=hand_data["hand_vertices_left"],
            hand_vertices_right=hand_data["hand_vertices_right"],
            fingertips_3d_left=hand_data["fingertips_3d_left"],
            fingertips_3d_right=hand_data["fingertips_3d_right"],
            total_frames=tracks_3d.shape[0],
            distance_threshold=args.distance_threshold,
            coupling_threshold=args.coupling_threshold,
        )
        grasp_results[obj_key] = grasp_result

    # ── Step 4: 精确 6DoF 轨迹估计 ──
    print("\n" + "=" * 70)
    print("Step 4: Precise 6DoF Trajectory Estimation")
    print("=" * 70)

    refined_trajectories = {}
    for obj_key, track_data in object_tracks.items():
        print(f"\n  Refining '{obj_key}'...")

        tracks_3d = track_data["tracks_3d"]
        valid_3d = track_data["valid_3d"]
        visibility = track_data.get("visibility", None)
        confidence = track_data.get("confidence", None)

        grasp_result = grasp_results[obj_key]

        refiner_result = run_trajectory_refiner(
            object_tracks_3d=tracks_3d,
            object_valid=valid_3d,
            object_confidence=confidence if confidence is not None else None,
            object_visibility=visibility if visibility is not None else None,
            interaction_segments=grasp_result["segments"],
            total_frames=tracks_3d.shape[0],
            smooth=True,
        )
        refined_trajectories[obj_key] = refiner_result

        traj = refiner_result["trajectory"]
        n_valid = refiner_result["valid_frames"].sum()
        print(f"  Trajectory: {traj.shape}, {n_valid} valid frames")

    # ── Step 5: 保存结果 ──
    print("\n" + "=" * 70)
    print("Step 5: Save Results")
    print("=" * 70)

    traj_dir = os.path.join(args.output, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    for obj_key, refiner_result in refined_trajectories.items():
        np.savez(
            os.path.join(traj_dir, f"{obj_key}.npz"),
            trajectory=refiner_result["trajectory"],
            centroids=refiner_result["centroids"],
            valid_frames=refiner_result["valid_frames"],
        )

    grasp_dir = os.path.join(args.output, "grasp")
    os.makedirs(grasp_dir, exist_ok=True)

    for obj_key, grasp_result in grasp_results.items():
        np.savez(
            os.path.join(grasp_dir, f"{obj_key}_grasp.npz"),
            gripper_signal_left=grasp_result["gripper_signal_left"],
            gripper_signal_right=grasp_result["gripper_signal_right"],
            obj_centroid=grasp_result["obj_centroid"],
            obj_speed=grasp_result["obj_speed"],
        )

        for hand_label, coupling in grasp_result["coupling"].items():
            np.save(
                os.path.join(grasp_dir, f"{obj_key}_coupling_{hand_label}.npy"),
                coupling,
            )

    summary = {}
    for obj_key in refined_trajectories:
        refiner_result = refined_trajectories[obj_key]
        grasp_result = grasp_results[obj_key]

        all_segments = []
        for hand_label, segs in grasp_result["segments"].items():
            for seg in segs:
                all_segments.append({
                    "hand": hand_label,
                    "grasp_frame": seg["grasp_frame"],
                    "release_frame": seg["release_frame"],
                    "type": seg["type"],
                })

        summary[obj_key] = {
            "trajectory_shape": list(refiner_result["trajectory"].shape),
            "valid_frames": int(refiner_result["valid_frames"].sum()),
            "release_frame": refiner_result["release_frame"],
            "interaction_segments": all_segments,
            "grasp_poses": {
                k: v.tolist() for k, v in grasp_result["grasp_poses"].items()
            } if grasp_result["grasp_poses"] else {},
        }

    with open(os.path.join(args.output, "precise_tracking_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        os.path.join(args.output, "vggt_predictions.npz"),
        depth=vggt_pred["depth"],
        cam2world=vggt_pred["cam2world"],
        intrinsic=vggt_pred["intrinsic"],
    )
    np.save(os.path.join(args.output, "dynamic_mask.npy"), dynamic_mask)

    print(f"\n  Results saved to: {args.output}")
    print(f"  - trajectories/*.npz  (6DoF trajectories)")
    print(f"  - grasp/*.npz         (grasp timing + coupling)")
    print(f"  - point_tracks.npz    (raw tracked points)")
    print(f"  - vggt_predictions.npz (VGGT4D outputs)")
    print(f"  - dynamic_mask.npy    (refined dynamic mask)")
    print(f"  - precise_tracking_summary.json")

    # ── Step 6: 仿真执行 + 闭环验证 (可选) ──
    if not args.skip_simulation:
        print("\n" + "=" * 70)
        print("Step 6: Simulation + Closed-Loop Verification")
        print("=" * 70)

        try:
            from simulation.scene_builder import build_scene
            from simulation.action_player import (
                mano_trajectory_to_ee_trajectory,
                run_simulation,
            )

            for obj_key, refiner_result in refined_trajectories.items():
                grasp_result = grasp_results[obj_key]

                if hawor_path is None:
                    print(f"  Skipping simulation for '{obj_key}': no HaWoR data")
                    continue

                print(f"  Running simulation for '{obj_key}'...")

                ee_trajectory = mano_trajectory_to_ee_trajectory(
                    pred_trans=hawor_results["pred_trans"],
                    pred_rot=hawor_results["pred_rot"],
                )

                sim_result = run_simulation(
                    env=None,
                    robot=None,
                    ee_trajectory=ee_trajectory,
                    left_gripper_signal=grasp_result["gripper_signal_left"],
                    right_gripper_signal=grasp_result["gripper_signal_right"],
                )

                sim_dir = os.path.join(args.output, "simulation")
                os.makedirs(sim_dir, exist_ok=True)
                np.savez(
                    os.path.join(sim_dir, f"{obj_key}_sim.npz"),
                    **sim_result.get("verification", {}),
                )

                if not args.skip_closed_loop:
                    all_contact_frames = []
                    for hand_label, segs in grasp_result["segments"].items():
                        for seg in segs:
                            all_contact_frames.extend(
                                range(seg["grasp_frame"], seg["release_frame"] + 1)
                            )

                    cl_result = run_closed_loop_verification(
                        sim_results=sim_result,
                        ref_trajectory=refiner_result["trajectory"],
                        contact_frames=all_contact_frames,
                        valid_frames=refiner_result["valid_frames"],
                        output_dir=sim_dir,
                    )

                    if cl_result["verified"]:
                        print(f"  ✓ Closed-loop verification PASSED")
                    else:
                        print(f"  ✗ Closed-loop verification FAILED")
                        print(f"    Final error: {cl_result['final_error'].get('mean_error', 'N/A')}m")

        except ImportError as e:
            print(f"  WARNING: Simulation modules not available: {e}")
            print(f"  Skipping simulation and closed-loop verification")

    print("\n" + "=" * 70)
    print("DONE! Precise tracking pipeline completed.")
    print(f"Output: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()

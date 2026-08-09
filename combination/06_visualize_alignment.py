#!/usr/bin/env python3
"""
06_visualize_alignment.py - Open3D visualization for HaWoR alignment verification.

Focus: check HaWoR camera orientation + MANO hand positions in SAPIEN Z-UP.

Usage:
    python 06_visualize_alignment.py \
        --hawor-dir /home/an/data/hawor/hoi4d \
        --ras-dir /home/an/data/ras/hoi4d1_vggt_omega \
        --transform-params output/alignment/transform_params.npz

Legend:
  Red   trajectory     = HaWoR camera path
  Red   arrows        = camera forward direction (orientation)
  Blue  line          = right hand trajectory (MANO center)
  Orange line         = left  hand trajectory (MANO center)

Controls:
  R   reset camera view
  Esc exit
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from glob import glob

R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
R_X = np.diag([1.0, -1.0, -1.0])
RXWORLD_TO_SAPIEN = R_AXIS @ R_X

HIDDEN_POS = np.array([0.0, -100.0, 0.0], dtype=np.float64)


def _slam_to_sapien(pts):
    return (RXWORLD_TO_SAPIEN @ pts.T).T


def _detect_glb_up_axis(all_vertices):
    FLOOR_THRESHOLD = 0.1
    min_z = all_vertices[:, 2].min()
    min_y = all_vertices[:, 1].min()
    z_is_floor = abs(min_z) < FLOOR_THRESHOLD
    y_is_floor = abs(min_y) < FLOOR_THRESHOLD
    if z_is_floor and not y_is_floor:
        return "z-up"
    return "y-up"


def load_transform_params(path):
    return np.load(path)


def load_glb_meshes(glb_path, params):
    import trimesh
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    ts = trimesh.load(str(glb_path))
    all_v = []
    for _, g in ts.geometry.items():
        if hasattr(g, 'vertices') and len(g.vertices) > 0:
            all_v.append(g.vertices)
    glb_up = _detect_glb_up_axis(np.vstack(all_v)) if all_v else "y-up"
    meshes = []
    for name, g in ts.geometry.items():
        if not hasattr(g, 'faces') or not hasattr(g, 'vertices'):
            continue
        verts = g.vertices.copy()
        faces = g.faces.copy()
        if len(verts) == 0 or len(faces) == 0:
            continue
        if glb_up == "z-up":
            verts = (R_AXIS @ verts.T).T
        verts_haw = s_inv * (R_inv @ verts.T).T + t_inv
        verts_sap = _slam_to_sapien(verts_haw)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_sap.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        if hasattr(g.visual, 'vertex_colors') and g.visual.vertex_colors is not None:
            colors = g.visual.vertex_colors[:, :3] / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes


def load_hawor_cameras(hawor_data):
    R_all = hawor_data['R_c2w']
    t_all = hawor_data['t_c2w']
    n = len(t_all)
    positions = np.zeros((n, 3), dtype=np.float64)
    forwards = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        R = R_all[i].astype(np.float64)
        t = t_all[i].astype(np.float64)
        positions[i] = RXWORLD_TO_SAPIEN @ t
        fwd_render = R[:, 2]
        forwards[i] = R_AXIS @ fwd_render
    return positions, forwards


def _find_mano_path():
    d = Path(__file__).resolve().parent
    for c in [d.parent.parent / "example" / "position_retargeting", d / "hand_track"]:
        if (c / "mano_layer.py").exists():
            return c
    return None


def init_mano_layers(hawor_data):
    mp = _find_mano_path()
    if mp is None:
        return None
    sys.path.insert(0, str(mp))
    try:
        from mano_layer import MANOLayer
    except ImportError:
        return None
    layers = {}
    for hi, hn in enumerate(["left", "right"]):
        try:
            layers[hi] = MANOLayer(hn, hawor_data['pred_betas'][hi, 0].astype(np.float32))
        except Exception:
            return None
    return layers


def compute_mano_keypoints(hawor_data, mano_layers):
    if mano_layers is None:
        return None
    import torch
    nf = hawor_data['pred_trans'].shape[1]
    all_kps = np.full((nf, 2, 21, 3), np.nan, dtype=np.float64)
    for hi in range(2):
        if hi not in mano_layers:
            continue
        layer = mano_layers[hi]
        valid = hawor_data['pred_valid'][hi]
        rot = hawor_data['pred_rot'][hi]
        hp = hawor_data['pred_hand_pose'][hi]
        tr = hawor_data['pred_trans'][hi]
        for fi in range(nf):
            if not valid[fi]:
                continue
            try:
                pn = np.concatenate([rot[fi], hp[fi]]).astype(np.float32)
                tn = tr[fi].astype(np.float32)
                with torch.no_grad():
                    _, j = layer(torch.from_numpy(pn).unsqueeze(0),
                                 torch.from_numpy(tn).unsqueeze(0))
                all_kps[fi, hi] = _slam_to_sapien(j[0].detach().cpu().numpy())
            except Exception:
                pass
    return all_kps


def compute_avg_joints(all_mano_kps):
    nf = all_mano_kps.shape[0]
    avg = np.full((nf, 2, 3), np.nan, dtype=np.float64)
    for hi in range(2):
        for fi in range(nf):
            j = all_mano_kps[fi, hi]
            valid = ~np.any(np.isnan(j), axis=1)
            if valid.sum() >= 3:
                avg[fi, hi] = j[valid].mean(axis=0)
    return avg


def compute_wrist_fallback(hawor_data):
    nf = hawor_data['pred_trans'].shape[1]
    w = np.full((nf, 2, 3), np.nan, dtype=np.float64)
    for hi in range(2):
        valid = hawor_data['pred_valid'][hi]
        tr = hawor_data['pred_trans'][hi]
        for fi in range(nf):
            if valid[fi]:
                w[fi, hi] = _slam_to_sapien(tr[fi:fi+1])[0]
    return w


def _build_hand_trajectories(avg_joints):
    colors = [[0.0, 0.6, 1.0], [1.0, 0.5, 0.0]]
    linesets = []
    for hi in range(2):
        pts = avg_joints[:, hi]
        valid = ~np.any(np.isnan(pts), axis=1)
        idx = np.where(valid)[0]
        if len(idx) < 2:
            empty = o3d.geometry.LineSet()
            empty.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            linesets.append(empty)
            continue
        traj = pts[idx]
        line_idx = np.array([[i, i + 1] for i in range(len(traj) - 1)], dtype=np.int32)
        ncols = np.tile(colors[hi], (len(line_idx), 1)).astype(np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(traj.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(line_idx)
        ls.colors = o3d.utility.Vector3dVector(ncols)
        linesets.append(ls)
    return linesets


def _build_hawor_camera_arrows(positions, forwards, length=0.12, interval=10):
    n = min(len(positions), len(forwards))
    meshes = []
    cyl_h = length * 0.65
    cone_h = length * 0.35
    for i in range(0, n, interval):
        p = positions[i].astype(np.float64)
        f = forwards[i] / np.linalg.norm(forwards[i])
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.004, cone_radius=0.014,
            cylinder_height=cyl_h, cone_height=cone_h)
        z = np.array([0.0, 0.0, 1.0])
        cos_a = np.dot(z, f)
        if abs(cos_a) > 0.9999:
            R = np.eye(3) if cos_a > 0 else -np.eye(3)
        else:
            axis = np.cross(z, f)
            axis = axis / np.linalg.norm(axis)
            ang = np.arccos(np.clip(cos_a, -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * ang)
        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(p)
        arrow.paint_uniform_color([1.0, 0.2, 0.2])
        arrow.compute_vertex_normals()
        meshes.append(arrow)
    if not meshes:
        return None
    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    return combined


def _build_hawor_camera_trajectory(positions):
    n = len(positions)
    line_idx = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    cols = np.tile([0.8, 0.2, 0.2], (n - 1, 1)).astype(np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(line_idx)
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def _build_axes():
    pts = np.array([
        [0, 0, 0], [0.3, 0, 0],
        [0, 0, 0], [0, 0.3, 0],
        [0, 0, 0], [0, 0, 0.3],
    ], dtype=np.float64)
    lines = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def _build_legend():
    hand_colors = [[0.0, 0.6, 1.0], [1.0, 0.5, 0.0]]
    geos = []
    for ci, col in enumerate(hand_colors):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        s.paint_uniform_color(col)
        s.translate(np.array([0.0, -ci * 0.04, 0.0]))
        s.compute_vertex_normals()
        geos.append(s)
    return geos


class AlignmentVisualizer:
    def __init__(self, glb_meshes, camera_fwd, camera_traj, hand_traj_lines,
                 axes, legend, scene_center, cam_center):
        self.glb_meshes = glb_meshes
        self.camera_fwd = camera_fwd
        self.camera_traj = camera_traj
        self.axes = axes
        self.legend = legend

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="HaWoR Alignment | SAPIEN Z-UP | R=reset",
            width=1280, height=720)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.12, 0.12, 0.14])
        opt.point_size = 6.0
        opt.line_width = 2.0
        opt.mesh_show_back_face = True

        for m in self.glb_meshes:
            self.vis.add_geometry(m)
        if self.camera_traj is not None:
            self.vis.add_geometry(self.camera_traj)
        if self.camera_fwd is not None:
            self.vis.add_geometry(self.camera_fwd)
        for ls in hand_traj_lines:
            self.vis.add_geometry(ls)
        self.vis.add_geometry(self.axes)
        for g in self.legend:
            self.vis.add_geometry(g)

        self.vis.register_key_callback(82, self._on_r)
        self.vis.register_key_callback(114, self._on_r)

        data_center = (scene_center + cam_center) * 0.5
        front = scene_center - cam_center
        fnorm = np.linalg.norm(front)
        if fnorm > 1e-6:
            front = front / fnorm
        else:
            front = np.array([0.0, -0.5, 0.5])

        self.vis.reset_view_point(True)
        ctr = self.vis.get_view_control()
        ctr.set_lookat(data_center)
        ctr.set_front(front)
        ctr.set_up([0.0, 0.0, 1.0])

        self.vis.poll_events()
        self.vis.update_renderer()
        data_center = (scene_center + cam_center) * 0.5
        front = scene_center - cam_center
        fnorm = np.linalg.norm(front)
        if fnorm > 1e-6:
            front = front / fnorm
        else:
            front = np.array([0.0, -0.5, 0.5])

        self.vis.reset_view_point(True)
        ctr = self.vis.get_view_control()
        ctr.set_lookat(data_center)
        ctr.set_front(front)
        ctr.set_up([0.0, 0.0, 1.0])

        self.vis.poll_events()
        self.vis.update_renderer()

    def _on_r(self, vis):
        all_verts = np.vstack([np.asarray(m.vertices) for m in self.glb_meshes])
        sc = all_verts.mean(axis=0)
        cc = np.asarray(self.camera_traj.points).mean(axis=0) if self.camera_traj is not None else sc
        dc = (sc + cc) * 0.5
        f = sc - cc
        fn = np.linalg.norm(f)
        if fn > 1e-6:
            f = f / fn
        else:
            f = np.array([0.0, -0.5, 0.5])
        ctr = self.vis.get_view_control()
        ctr.set_lookat(dc)
        ctr.set_front(f)
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.6)
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="HaWoR Alignment Visualization")
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--transform-params', required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print diagnostics and exit')
    args = parser.parse_args()

    print("=" * 60)
    print("HaWoR Alignment Visualization")
    print("=" * 60)
    print("Legend:")
    print("  Red   trajectory = HaWoR camera path")
    print("  Red   arrows    = camera forward direction")
    print("  Blue line       = right hand trajectory (MANO center)")
    print("  Orange line     = left  hand trajectory (MANO center)")
    print("=" * 60)

    params = load_transform_params(args.transform_params)

    glb_path = os.path.join(args.ras_dir, 'final_scene.glb')
    if os.path.exists(glb_path):
        print("Loading GLB scene...", end=' ', flush=True)
        glb_meshes = load_glb_meshes(glb_path, params)
        print(f"{len(glb_meshes)} meshes")
    else:
        print(f"Warning: GLB not found at {glb_path}, skipping")
        glb_meshes = []

    rec_dir = os.path.join(args.hawor_dir, 'reconstruction')
    npz_files = sorted(glob(os.path.join(rec_dir, 'hawor_results_*.npz')))
    if not npz_files:
        print("Error: no hawor_results_*.npz found"); sys.exit(1)
    hawor_data = np.load(npz_files[0])
    n_frames = hawor_data['pred_trans'].shape[1]
    print(f"HaWoR: {n_frames} frames, 2 hands")

    print("Loading HaWoR cameras...", end=' ', flush=True)
    hawor_cam_pos, hawor_cam_fwd = load_hawor_cameras(hawor_data)
    print(f"{len(hawor_cam_pos)} frames")

    print("Initializing MANO...", end=' ', flush=True)
    mano_layers = init_mano_layers(hawor_data)
    all_mano_kps = None
    if mano_layers is not None:
        print("computing keypoints...", end=' ', flush=True)
        all_mano_kps = compute_mano_keypoints(hawor_data, mano_layers)
        valid = np.sum(~np.isnan(all_mano_kps[:, 0, 0, 0])) if all_mano_kps is not None else 0
        print(f"{valid}/{n_frames} frames OK")
    else:
        print("unavailable, showing wrist only")

    wrist_fb = compute_wrist_fallback(hawor_data)

    avg_joints = compute_avg_joints(all_mano_kps) if all_mano_kps is not None else None

    camera_traj = _build_hawor_camera_trajectory(hawor_cam_pos)
    camera_fwd = _build_hawor_camera_arrows(hawor_cam_pos, hawor_cam_fwd,
                                          length=0.12, interval=max(1, n_frames // 20))

    if not glb_meshes:
        print("Error: no geometry to display"); sys.exit(1)

    axes = _build_axes()
    legend = _build_legend()

    all_verts = np.vstack([np.asarray(m.vertices) for m in glb_meshes])
    print(f"Scene: center={all_verts.mean(axis=0)}, extent={all_verts.max(axis=0)-all_verts.min(axis=0)}")
    print(f"Camera: center={hawor_cam_pos.mean(axis=0)}, extent={hawor_cam_pos.max(axis=0)-hawor_cam_pos.min(axis=0)}")
    cam_to_scene = all_verts.mean(axis=0) - hawor_cam_pos.mean(axis=0)
    print(f"Camera→Scene vector: {cam_to_scene}")
    print(f"Frame 0 forward: {hawor_cam_fwd[0]} (norm={np.linalg.norm(hawor_cam_fwd[0]):.4f})")
    dot = np.dot(cam_to_scene / np.linalg.norm(cam_to_scene), hawor_cam_fwd[0] / np.linalg.norm(hawor_cam_fwd[0]))
    print(f"Frame 0 forward·scene_dir dot: {dot:.4f} {'✓ 朝向场景' if dot > 0 else '✗ 背对场景'}")

    hand_traj_lines = _build_hand_trajectories(avg_joints) if avg_joints is not None else []

    if args.dry_run:
        print("\nDry-run complete. Exiting.")
        return
    print("  R=reset  Esc=exit")

    vis = AlignmentVisualizer(glb_meshes, camera_fwd, camera_traj, hand_traj_lines,
                              axes, legend,
                              all_verts.mean(axis=0), hawor_cam_pos.mean(axis=0))
    vis.run()


if __name__ == '__main__':
    main()
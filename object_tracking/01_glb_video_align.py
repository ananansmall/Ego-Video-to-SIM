"""
Step 0: GLB 场景与视频对齐验证
================================

核心功能: 给定 GLB 场景 + VGGT 相机参数, 渲染与视频帧对齐的画面

坐标系变换链 (已验证):
  VGGT extrinsic (w2c, OpenCV, z-up Room World)
    → c2w = inv(w2c)
    → c2w_opengl = c2w @ opencv_to_opengl
    → cam_pose = zup_to_yup @ c2w_opengl
    → pyrender 渲染 GLB (y-up)

用法:
  python 01_glb_video_align.py --glb final_scene_basic.glb --video 7.mp4 --extrinsics_dir extrinsics/ --intrinsic intrinsic.txt
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse

import cv2
import numpy as np
import pyrender
import trimesh

OPENCV_TO_OPENGL = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

ZUP_TO_YUP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def load_extrinsics_from_dir(extrinsics_dir):
    ext_files = sorted(
        [f for f in os.listdir(extrinsics_dir) if f.endswith('.txt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    extrinsics = []
    for f in ext_files:
        ext = np.loadtxt(os.path.join(extrinsics_dir, f))
        extrinsics.append(ext)
    return extrinsics


def load_intrinsic(intrinsic_path):
    return np.loadtxt(intrinsic_path)


def load_glb_as_pyrender_mesh(glb_path):
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.dump())
    else:
        mesh = scene
    return pyrender.Mesh.from_trimesh(mesh)


def extrinsic_to_pyrender_pose(extrinsic):
    """VGGT extrinsic (w2c, OpenCV, z-up) → pyrender cam_pose (c2w, OpenGL, y-up)"""
    c2w_opencv = np.linalg.inv(extrinsic)
    c2w_opengl = c2w_opencv @ OPENCV_TO_OPENGL
    cam_pose = ZUP_TO_YUP @ c2w_opengl
    return cam_pose


def render_frame(mesh_pr, intrinsic, extrinsic, width, height, light_intensity=3.0):
    """渲染单帧 GLB 场景

    Args:
        mesh_pr: pyrender.Mesh
        intrinsic: (3,3) 相机内参
        extrinsic: (4,4) w2c 外参 (OpenCV, z-up)
        width: 渲染宽度
        height: 渲染高度
        light_intensity: 光照强度

    Returns:
        color: (H, W, 3) uint8 RGB
        depth: (H, W) float32 深度图
    """
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    cam_pose = extrinsic_to_pyrender_pose(extrinsic)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh_pr)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth


def overlay_render_on_video(rendered, video_frame, alpha=0.5):
    """将渲染帧叠加到视频帧上

    Args:
        rendered: (H, W, 3) uint8 渲染帧 (RGB)
        video_frame: (H, W, 3) uint8 视频帧 (BGR)
        alpha: 渲染帧的透明度

    Returns:
        overlay: (H, W, 3) uint8 叠加结果 (BGR)
    """
    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

    mask = (rendered[:, :, 0] > 0) | (rendered[:, :, 1] > 0) | (rendered[:, :, 2] > 0)
    mask_3c = np.stack([mask] * 3, axis=-1)

    result = video_frame.copy()
    blended = cv2.addWeighted(rendered_bgr, alpha, video_frame, 1 - alpha, 0)
    result[mask_3c] = blended[mask_3c]

    return result


def render_wireframe_edges(mesh_pr, intrinsic, extrinsic, width, height):
    """渲染场景边缘线框 (用于对齐验证)

    Returns:
        edges: (H, W, 3) uint8 边缘图 (绿线黑底)
    """
    color, depth = render_frame(mesh_pr, intrinsic, extrinsic, width, height)

    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[edges > 0] = [0, 255, 0]
    return result


def load_video_frames(video_path, target_size=None):
    """加载视频帧

    Args:
        video_path: 视频文件路径
        target_size: (width, height) 目标尺寸 (可选)

    Returns:
        frames: List[np.ndarray] BGR 帧
        fps: float 帧率
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    return frames, fps


def align_glb_to_video(
    glb_path,
    video_path_or_frames,
    extrinsics,
    intrinsic,
    output_dir,
    overlay_alpha=0.5,
    render_edges=True,
    save_video=True,
    frame_indices=None,
):
    """完整 GLB-视频对齐渲染管线

    Args:
        glb_path: GLB 场景文件路径
        video_path_or_frames: 视频路径或帧列表
        extrinsics: List[(4,4)] w2c 外参 (OpenCV, z-up)
        intrinsic: (3,3) 相机内参
        output_dir: 输出目录
        overlay_alpha: 叠加透明度
        render_edges: 是否渲染边缘
        save_video: 是否保存视频
        frame_indices: 指定渲染的帧索引 (None=全部)

    Returns:
        dict: {rendered_frames, overlay_frames, n_frames}
    """
    os.makedirs(output_dir, exist_ok=True)
    overlay_dir = os.path.join(output_dir, "overlay")
    render_dir = os.path.join(output_dir, "rendered")
    edge_dir = os.path.join(output_dir, "edges")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    if render_edges:
        os.makedirs(edge_dir, exist_ok=True)

    print(f"[01_glb_video_align] Loading GLB: {glb_path}")
    mesh_pr = load_glb_as_pyrender_mesh(glb_path)

    if isinstance(video_path_or_frames, str):
        video_frames, fps = load_video_frames(video_path_or_frames)
    else:
        video_frames = video_path_or_frames
        fps = 30.0

    if not video_frames:
        print("[01_glb_video_align] ERROR: No video frames loaded")
        return {"rendered_frames": [], "overlay_frames": [], "n_frames": 0}

    H, W = video_frames[0].shape[:2]
    print(f"[01_glb_video_align] Video: {len(video_frames)} frames, {W}x{H}")
    print(f"[01_glb_video_align] Extrinsics: {len(extrinsics)} poses")

    S_video = len(video_frames)
    S_ext = len(extrinsics)
    S = min(S_video, S_ext)

    if frame_indices is None:
        frame_indices = list(range(S))

    rendered_frames = []
    overlay_frames = []

    for idx, i in enumerate(frame_indices):
        if i >= S:
            break

        try:
            color, depth = render_frame(mesh_pr, intrinsic, extrinsics[i], W, H)
        except Exception as e:
            print(f"[01_glb_video_align] Frame {i} render failed: {e}")
            rendered_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
            overlay_frames.append(video_frames[i])
            continue

        rendered_frames.append(color)

        overlay = overlay_render_on_video(color, video_frames[i], alpha=overlay_alpha)
        overlay_frames.append(overlay)

        cv2.imwrite(os.path.join(render_dir, f"{i:04d}.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(overlay_dir, f"{i:04d}.jpg"), overlay)

        if render_edges:
            edge_img = render_wireframe_edges(mesh_pr, intrinsic, extrinsics[i], W, H)
            edge_overlay = cv2.addWeighted(video_frames[i], 0.7, edge_img, 0.3, 0)
            cv2.imwrite(os.path.join(edge_dir, f"{i:04d}.jpg"), edge_overlay)

        if (idx + 1) % 10 == 0 or idx == len(frame_indices) - 1:
            print(f"  Rendered {idx + 1}/{len(frame_indices)} frames")

    if save_video and overlay_frames:
        overlay_video_path = os.path.join(output_dir, "overlay_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(overlay_video_path, fourcc, fps, (W, H))
        for frame in overlay_frames:
            out.write(frame)
        out.release()
        print(f"[01_glb_video_align] Overlay video saved: {overlay_video_path}")

        render_video_path = os.path.join(output_dir, "render_video.mp4")
        out = cv2.VideoWriter(render_video_path, fourcc, fps, (W, H))
        for frame in rendered_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[01_glb_video_align] Render video saved: {render_video_path}")

    print(f"[01_glb_video_align] Done! {len(rendered_frames)} frames rendered")
    print(f"  Output: {output_dir}")

    return {
        "rendered_frames": rendered_frames,
        "overlay_frames": overlay_frames,
        "n_frames": len(rendered_frames),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 0: GLB-Video Alignment - Render GLB from VGGT camera poses and overlay on video"
    )
    parser.add_argument("--glb", type=str, required=True,
                        help="GLB 场景文件路径")
    parser.add_argument("--video", type=str, required=True,
                        help="输入 MP4 视频文件路径")
    parser.add_argument("--extrinsics_dir", type=str, required=True,
                        help="VGGT 外参目录 (包含 0.txt, 1.txt, ...)")
    parser.add_argument("--intrinsic", type=str, required=True,
                        help="VGGT 内参文件路径 (3x3 txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="渲染叠加透明度 (0=仅视频, 1=仅渲染)")
    parser.add_argument("--no_edges", action="store_true",
                        help="不渲染边缘线框")
    parser.add_argument("--no_video", action="store_true",
                        help="不保存视频文件")
    parser.add_argument("--frames", type=str, default=None,
                        help="指定渲染帧, 如 '0,10,20' 或 '0-50:5'")
    args = parser.parse_args()

    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(os.path.dirname(args.glb), f"{video_name}_aligned")

    extrinsics = load_extrinsics_from_dir(args.extrinsics_dir)
    intrinsic = load_intrinsic(args.intrinsic)

    frame_indices = None
    if args.frames:
        if '-' in args.frames and ':' in args.frames:
            parts = args.frames.split(':')
            range_part = parts[0]
            step = int(parts[1]) if len(parts) > 1 else 1
            start, end = map(int, range_part.split('-'))
            frame_indices = list(range(start, end, step))
        elif ',' in args.frames:
            frame_indices = list(map(int, args.frames.split(',')))
        else:
            frame_indices = list(map(int, args.frames.split(',')))

    align_glb_to_video(
        glb_path=args.glb,
        video_path_or_frames=args.video,
        extrinsics=extrinsics,
        intrinsic=intrinsic,
        output_dir=args.output,
        overlay_alpha=args.alpha,
        render_edges=not args.no_edges,
        save_video=not args.no_video,
        frame_indices=frame_indices,
    )


if __name__ == "__main__":
    main()

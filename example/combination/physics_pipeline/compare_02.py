"""提取 02 输出视频的帧，对比方向"""
import cv2
import numpy as np
from pathlib import Path

# 02 输出视频
video_02 = Path(__file__).parent.parent / "output" / "videos" / "hand_object_hand_only.mp4"
# PyBullet 测试图片
fpv_orig = Path(__file__).parent / "output" / "test_fpv_gripper.png"
fpv_flipped = Path(__file__).parent / "output" / "test_fpv_flipped.png"

print(f"02 video: {video_02}, exists: {video_02.exists()}")

if video_02.exists():
    cap = cv2.VideoCapture(str(video_02))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total}")

    # 提取第一帧、中间帧
    for frame_idx in [0, total // 2]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            mean = frame.mean()
            top = frame[:h//2].mean()
            bottom = frame[h//2].mean()
            left = frame[:, :w//2].mean()
            right = frame[:, w//2:].mean()
            print(f"\n  02 Frame {frame_idx}: {w}x{h}, mean={mean:.1f}")
            print(f"    Top: {top:.1f}, Bottom: {bottom:.1f}")
            print(f"    Left: {left:.1f}, Right: {right:.1f}")

            # 保存帧
            out_path = Path(__file__).parent / "output" / f"02_frame_{frame_idx}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"    Saved: {out_path.name}")
    cap.release()

# 对比 PyBullet 渲染
print("\n" + "="*60)
print("PyBullet 渲染对比:")
for name, path in [("FPV original", fpv_orig), ("FPV flipped", fpv_flipped)]:
    if path.exists():
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        top = img[:h//2].mean()
        bottom = img[h//2].mean()
        print(f"  {name}: Top={top:.1f}, Bottom={bottom:.1f}")

print("\n结论:")
print("  02 视频: Top > Bottom = 正确 (天花板/墙在上方, 桌面在下方)")
print("  如果 PyBullet FPV: Top < Bottom = 上下颠倒, 需要翻转")

"""分析渲染的图片 - 检查方向和内容"""
import cv2
import numpy as np
from pathlib import Path

out_dir = Path(__file__).parent / "output"

images = [
    "test_fpv_gripper.png",
    "test_fpv_flipped.png",
    "test_fpv_vflip.png",
    "test_thirdperson_gripper.png",
]

for name in images:
    path = out_dir / name
    if not path.exists():
        print(f"{name}: NOT FOUND")
        continue
    img = cv2.imread(str(path))
    if img is None:
        print(f"{name}: FAILED TO LOAD")
        continue
    h, w = img.shape[:2]
    mean = img.mean()
    std = img.std()

    # 分析上下半部分的亮度
    top_half = img[:h//2].mean()
    bottom_half = img[h//2:].mean()

    # 分析非白色像素 (物体/桌面)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white = (gray < 240).sum() / gray.size * 100

    # 分析左右半部分
    left_half = img[:, :w//2].mean()
    right_half = img[:, w//2:].mean()

    print(f"\n{name}:")
    print(f"  Shape: {w}x{h}, mean={mean:.1f}, std={std:.1f}")
    print(f"  Top half mean: {top_half:.1f}, Bottom half mean: {bottom_half:.1f}")
    print(f"  Left half mean: {left_half:.1f}, Right half mean: {right_half:.1f}")
    print(f"  Non-white pixels: {non_white:.1f}%")

    # 检查是否有物体在画面中 (非白色区域)
    if non_white > 5:
        # 找到非白色区域的边界
        rows = np.any(gray < 240, axis=1)
        cols = np.any(gray < 240, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            print(f"  Content bbox: rows [{rmin}-{rmax}], cols [{cmin}-{cmax}]")
            print(f"  Content center: ({(cmin+cmax)//2}, {(rmin+rmax)//2})")

# 对比 FPV 和翻转版本
print("\n" + "="*60)
print("对比 FPV 原始 vs 翻转 up 向量:")
fpv = cv2.imread(str(out_dir / "test_fpv_gripper.png"))
flipped = cv2.imread(str(out_dir / "test_fpv_flipped.png"))
if fpv is not None and flipped is not None:
    diff = cv2.absdiff(fpv, flipped)
    print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean():.2f}")
    if diff.mean() < 1.0:
        print("  -> 几乎相同 (翻转 up 向量不影响渲染)")
    else:
        print("  -> 有差异 (翻转 up 向量影响渲染)")

# 对比 FPV 和垂直翻转
vflip = cv2.imread(str(out_dir / "test_fpv_vflip.png"))
if fpv is not None and vflip is not None:
    diff = cv2.absdiff(fpv, vflip)
    print(f"\n对比 FPV 原始 vs 垂直翻转图像:")
    print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean():.2f}")

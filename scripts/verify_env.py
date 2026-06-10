#!/usr/bin/env python3
"""verify_env.py — 验证 Ego-Video-to-SIM 环境是否就绪"""
import sys
from pathlib import Path

GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"

errors = []
oks = []

def check(label, condition, hint=""):
    if condition:
        oks.append(label)
        print(f"  {GREEN}✓{NC} {label}")
    else:
        errors.append(label)
        msg = f"  {RED}✗{NC} {label}"
        if hint:
            msg += f"  {YELLOW}→ {hint}{NC}"
        print(msg)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print(f"\n{GREEN}=== 子模块完整性 ==={NC}")
for sub in ["HaWoR", "ReplicateAnyScene",
            "libs/dex_retargeting", "libs/galaxea_sim", "libs/position_retargeting"]:
    path = PROJECT_ROOT / sub
    check(sub, path.exists() and any(path.iterdir()),
          "git submodule update --init --recursive")

print(f"\n{GREEN}=== Python 包 ==={NC}")
for pkg in ["numpy", "cv2", "sapien", "torch", "joblib",
            "pytransform3d", "tqdm", "trimesh", "natsort"]:
    try:
        __import__(pkg)
        check(pkg, True)
    except ImportError:
        check(pkg, False, f"pip install {pkg}")

print(f"\n{GREEN}=== 库路径 ==={NC}")
sys.path.insert(0, str(PROJECT_ROOT / "libs"))
sys.path.insert(0, str(PROJECT_ROOT / "libs" / "position_retargeting"))
for module in ["dex_retargeting", "galaxea_sim", "mano_layer"]:
    try:
        __import__(module)
        check(f"import {module}", True)
    except ImportError as e:
        check(f"import {module}", False, str(e))

print(f"\n{GREEN}=== 汇总 ==={NC}")
print(f"  ✓ 通过: {len(oks)}")
if errors:
    print(f"  {RED}✗ 失败: {len(errors)}{NC}")
    sys.exit(1)
else:
    print(f"\n{GREEN}环境就绪 ✓{NC}")

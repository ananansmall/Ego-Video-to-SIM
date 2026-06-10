#!/usr/bin/env bash
# =============================================================================
#  setup.sh — Ego-Video-to-SIM 一键初始化
# =============================================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"
cd "$PROJECT_ROOT"

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Ego-Video-to-SIM 一键初始化${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo ""

# Step 1: 初始化子模块
echo -e "${YELLOW}[Step 1/3] 初始化 git 子模块...${NC}"
git submodule update --init --recursive

# 检查 libs/ 依赖
MISSING=()
for sub in libs/dex_retargeting libs/galaxea_sim libs/position_retargeting; do
    if [ ! -d "$sub" ] || [ -z "$(ls -A "$sub" 2>/dev/null)" ]; then
        MISSING+=("$sub")
    fi
done

if [ ${#MISSING[@]} -ne 0 ]; then
    echo -e "${YELLOW}libs/ 子目录未初始化, 尝试从 GitHub 克隆...${NC}"
    mkdir -p libs
    [ ! -d "libs/dex_retargeting/.git" ] && git clone --depth 1 https://github.com/dexsuite/dex-retargeting.git libs/dex_retargeting
    [ ! -d "libs/galaxea_sim/.git" ] && git clone --depth 1 https://github.com/OpenGalaxea/GalaxeaManipSim.git libs/galaxea_sim
    [ ! -d "libs/position_retargeting/.git" ] && git clone --depth 1 https://github.com/dexsuite/dex-retargeting.git libs/position_retargeting
fi

echo -e "${GREEN}✓ 子模块已就位${NC}"
echo ""

# Step 2: 安装 pip 依赖
echo -e "${YELLOW}[Step 2/3] 安装 pip 依赖...${NC}"
pip install numpy opencv-python sapien torch joblib pytransform3d tqdm \
            trimesh scipy natsort matplotlib imageio-ffmpeg 2>/dev/null || \
pip3 install numpy opencv-python sapien torch joblib pytransform3d tqdm \
               trimesh scipy natsort matplotlib imageio-ffmpeg
echo -e "${GREEN}✓ pip 依赖安装完成${NC}"
echo ""

# Step 3: 验证环境
echo -e "${YELLOW}[Step 3/3] 验证环境...${NC}"
python scripts/verify_env.py 2>/dev/null || python3 scripts/verify_env.py

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  ✓ 初始化完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "下一步:"
echo "  cd combination"
echo "  python 00_run_pipeline.py --hawor-dir <path> --ras-dir <path>"

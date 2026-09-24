#!/bin/bash
# 轮询等待任一 GPU 有 >=21000 MiB 空闲, 然后跑 3chair Stage 5.1 验证
source /home/lza/miniconda3/etc/profile.d/conda.sh
conda activate ReplicateAnyScene
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene

LOG=output_v2/3chair-bottle_vggt_omega/verify_stage51_auto.log
echo "$(date) 开始轮询 GPU 空闲..." | tee "$LOG"

WAITED=0
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=21000 {print $1; exit}')
  if [ -n "$FREE" ]; then
    echo "$(date) GPU $FREE 空闲, 启动验证 (CUDA_VISIBLE_DEVICES=$FREE)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$FREE python tools/_verify_stage51.py 2>&1 | tee -a "$LOG"
    echo "$(date) 验证完成, exit=$?" | tee -a "$LOG"
    break
  fi
  sleep 60
  WAITED=$((WAITED+60))
  if [ $WAITED -ge 18000 ]; then
    echo "$(date) 等待超时 (5h), 退出" | tee -a "$LOG"
    break
  fi
done

#!/usr/bin/env python3
"""
Stage 1 简化可视化: 只展示仿真中的夹爪 + 物体轨迹
从 log 中提取最后一段完整 rollout 的数据画图
"""
import sys, os, re, numpy as np, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")


def main():
    side = "right"
    log_path = os.path.join(OUTPUT, f"gripper_only_{side}", "grasp.log")
    grasp_path = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1", "best_grasp.npz")

    if not os.path.exists(grasp_path):
        print("ERROR: best_grasp.npz 不存在"); sys.exit(1)

    d_grasp = np.load(grasp_path, allow_pickle=True)
    grasp_pos = d_grasp["pos"]
    grasp_euler = d_grasp["euler"]
    obj_lift = float(d_grasp.get("obj_lift", 0))
    peak_force = float(d_grasp.get("peak_grip_force", 0))

    # 解析 log
    with open(log_path) as f:
        lines = [l.strip() for l in f.readlines()]

    # 找到最后一段 panel 的最后一轮 rollout
    # 格式: [Stage1 debug F0] ... [Stage1 debug] final ...
    # 注意: 日志中 "Stage1 debug F0" 会有多个 (每轮 CMA-ES 都有)
    # 我们需要最后一段包含 F0, F5, F10, F15, ... F75 的完整 rollout
    last_start = -1
    last_end = -1
    for i, line in enumerate(lines):
        if 'Stage1 debug' in line and 'f1_qpos' in line and 'F0]' in line:
            last_start = i
        if 'Stage1 debug' in line and 'final' in line and 'obj_lift' in line:
            last_end = i

    if last_start < 0 or last_end <= last_start:
        print("ERROR: 无法解析 log"); sys.exit(1)

    seg_lines = lines[last_start:last_end+1]
    print(f"提取 {len(seg_lines)} 行 (F0 ~ final)")

    # 解析帧数据
    data = []
    for line in seg_lines:
        m = re.search(r'F(\d+)', line)
        if not m:
            continue
        frame = int(m.group(1))
        # 解析键值对: pos_z=0.1053, f1_qpos=0.0120, 等
        kv = {}
        # 把 line 按空格切分, 找包含 = 的 token
        tokens = line.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
        for tok in tokens:
            if '=' in tok:
                parts = tok.split('=')
                if len(parts) == 2:
                    k, v = parts
                    try:
                        kv[k.strip()] = float(v)
                    except:
                        pass
        data.append({'frame': frame, **kv})

    if not data:
        print("ERROR: 无法解析帧数据"); sys.exit(1)

    data.sort(key=lambda x: x['frame'])
    frames = np.array([d['frame'] for d in data])
    pos_z = np.array([d.get('pos_z', 0) for d in data])
    obj_z = np.array([d.get('obj_z', 0) for d in data])
    force = np.array([d.get('force', 0) for d in data])
    f1_qpos = np.array([d.get('f1_qpos', 0) for d in data])
    f2_qpos = np.array([d.get('f2_qpos', 0) for d in data])

    # 物体初始 z (取 F0-F5 的中位数)
    obj_init_z = np.median(obj_z[:5]) if len(obj_z) > 5 else 0.017

    print(f"\n=== Stage 1 结果 ===")
    print(f"  Grasp pos: {grasp_pos.round(4)}")
    print(f"  Grasp euler: {grasp_euler.round(2)}")
    print(f"  帧数: {len(data)}")
    print(f"  物体初始 z: {obj_init_z:.4f}")
    print(f"  抬升: {obj_lift*100:.2f} cm")
    print(f"  峰值力: {peak_force:.2f} N")
    print(f"  最终物体 z: {obj_z[-1]:.4f}")
    print(f"  最终夹爪 z: {pos_z[-1]:.4f}")

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: 夹爪 + 物体 Z 高度
    ax = axes[0, 0]
    ax.plot(frames, pos_z, 'c-', lw=2.5, label='Gripper Base Z')
    ax.plot(frames, obj_z, 'm--', lw=2.5, label='Object Z')
    ax.axhline(y=obj_init_z, color='gray', ls=':', alpha=0.5, label=f'Desktop Z={obj_init_z:.4f}')
    # 标注关键帧
    n = len(frames)
    for fid, lbl in [(0, 'F0'), (n//4, f'F{frames[n//4]}'), (n//2, f'F{frames[n//2]}'), (n-1, f'F{frames[n-1]}')]:
        ax.scatter(frames[fid], pos_z[fid], c='cyan', s=60, zorder=5)
        ax.text(frames[fid], pos_z[fid]+0.005, lbl, fontsize=8, ha='center')
    ax.set_xlabel('Frame'); ax.set_ylabel('Z (m)')
    ax.set_title(f'Stage 1 Rollout | lift={obj_lift*100:.1f}cm  force={peak_force:.1f}N')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 右上: 夹持力
    ax = axes[0, 1]
    ax.plot(frames, force, 'r-', lw=2)
    ax.axhline(y=peak_force, color='orange', ls='--', lw=1.5, label=f'Peak={peak_force:.1f}N')
    ax.set_xlabel('Frame'); ax.set_ylabel('Force (N)')
    ax.set_title('Clamping Force'); ax.legend(); ax.grid(True, alpha=0.3)

    # 左下: 手指 qpos
    ax = axes[1, 0]
    ax.plot(frames, f1_qpos, 'b-', lw=2, label='Finger 1')
    ax.plot(frames, f2_qpos, 'g-', lw=2, label='Finger 2')
    ax.set_xlabel('Frame'); ax.set_ylabel('qpos')
    ax.set_title('Finger Joint Positions'); ax.legend(); ax.grid(True, alpha=0.3)

    # 右下: 物体抬升分析
    ax = axes[1, 1]
    obj_lift_cm = (obj_z - obj_init_z) * 100
    ax.plot(frames, obj_lift_cm, 'm-', lw=2.5)
    ax.axhline(y=obj_lift*100, color='orange', ls='--', lw=1.5, label=f'Final={obj_lift*100:.1f}cm')
    # 标注接触开始 (obj_z 开始 > obj_init_z + 2mm)
    contact_frame = 0
    for i, z in enumerate(obj_z):
        if z > obj_init_z + 0.002:
            contact_frame = frames[i]
            break
    if contact_frame > 0:
        ax.axvline(x=contact_frame, color='green', ls=':', lw=1.5, alpha=0.7, label=f'Contact at F{contact_frame}')
    ax.set_xlabel('Frame'); ax.set_ylabel('Lift (cm)')
    ax.set_title('Object Lift Analysis'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_out = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1_demo", "stage1_trajectory.png")
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {png_out}")
    plt.close()


if __name__ == "__main__":
    main()
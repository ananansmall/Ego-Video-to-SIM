#!/usr/bin/env python3
"""
Stage 1 抓取轨迹 + GLB 场景渲染
从 best_grasp.npz + grasp.log 解析轨迹数据, 渲染 3D 对比图
"""
import sys, os, numpy as np, argparse, warnings, re
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")


def draw_axes(ax, origin=[0,0,0], scale=0.1):
    o = np.array(origin)
    for c, lbl, d in [('red','X',[1,0,0]),('green','Y',[0,1,0]),('blue','Z',[0,0,1])]:
        v = np.array(d)*scale
        ax.plot([o[0],o[0]+v[0]],[o[1],o[1]+v[1]],[o[2],o[2]+v[2]], color=c, lw=4, zorder=10)
        ax.text(o[0]+v[0], o[1]+v[1], o[2]+v[2], lbl, color=c, fontsize=10, fontweight='bold')

def draw_bbox(ax, bounds, color, alpha=0.25):
    p0, p1 = bounds
    corners = np.array([
        [p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
        [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]
    ])
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*corners[[a,b]].T, color=color, alpha=alpha, lw=1.5)

def parse_log(log_path):
    """从 grasp.log 中解析每帧的夹爪 pos_z, f1_qpos, f2_qpos, obj_z, force
    取最后一次完整 rollout (最后 17 条 debug 行)"""
    frames = []
    if not os.path.exists(log_path):
        return frames
    with open(log_path) as f:
        lines = f.readlines()
    # 从后往前找最后一次完整 rollout
    debug_lines = [l for l in lines if "Stage1 debug" in l and "f1_qpos" in l]
    # 取最后 17 条 (F0~F75 + final)
    debug_lines = debug_lines[-17:]
    for line in debug_lines:
        m = re.search(r'F(\d+)', line)
        if not m:
            continue
        frame = int(m.group(1))
        kv = {}
        for p in line.replace('[',' ').replace(']',' ').replace(',',' ').split():
            if '=' in p:
                k, v = p.split('=')
                try:
                    kv[k.strip()] = float(v)
                except:
                    pass
        frames.append({
            'frame': frame,
            'pos_z': kv.get('pos_z', 0),
            'f1_qpos': kv.get('f1_qpos', 0),
            'f2_qpos': kv.get('f2_qpos', 0),
            'obj_z': kv.get('obj_z', 0),
            'force': kv.get('force', 0),
        })
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right")
    parser.add_argument("--output", default=os.path.join(OUTPUT, "gripper_only_right", "stage1_traj.png"))
    args = parser.parse_args()

    side = args.side
    stage_dir = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1")
    grasp_path = os.path.join(stage_dir, "best_grasp.npz")
    log_path = os.path.join(OUTPUT, f"gripper_only_{side}", "grasp.log")  # log 在主目录

    # 1. 加载 best_grasp.npz
    d = np.load(grasp_path, allow_pickle=True)
    grasp_pos = d["pos"]
    grasp_R = d["R"]
    obj_lift = float(d.get("obj_lift", 0))
    peak_force = float(d.get("peak_grip_force", 0))
    both_contact = int(d.get("both_contact_count", 0))  # 旧版 npz 可能没有
    xy_drift = float(d.get("obj_xy_drift", 0))
    obj_init = np.array(d.get("obj_initial_pos", [0, 0, 0])).flatten()  # 旧版 npz 可能没有
    obj_final = np.array(d.get("obj_final_pos", [0, 0, 0])).flatten()
    gripper_qpos = float(d.get("gripper_qpos", 0))

    print(f"Best grasp: pos={grasp_pos.round(4)}, R={grasp_R.shape}")
    print(f"  obj_init={obj_init.round(4)}, obj_final={obj_final.round(4)}")
    print(f"  lift={obj_lift*100:.1f}cm, peak_force={peak_force:.1f}N, contact={both_contact}, drift={xy_drift*100:.1f}cm")

    # 2. 解析日志
    log_frames = parse_log(log_path)
    print(f"  log 帧数: {len(log_frames)}")

    # 3. 加载 GLB
    glb_path = "/home/an/data/ras/my_7mp4_result/final_scene.glb"
    if os.path.exists(glb_path):
        scene = trimesh.load(glb_path)
        geoms = [(name, geom.vertices.mean(0), geom.bounds) for name, geom in scene.geometry.items()
                 if hasattr(geom, 'bounds') and geom.bounds is not None]
    else:
        geoms = []
    print(f"GLB geoms: {len(geoms)}")

    # 4. 渲染 3D 轨迹
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Stage 1 Grasp Trajectory (side={side})\n"
                 f"obj_lift={obj_lift*100:.1f}cm  peak_force={peak_force:.1f}N  contact={both_contact}/80  drift={xy_drift*100:.1f}cm",
                 fontweight='bold')
    draw_axes(ax, scale=0.05)

    # GLB 物体 bbox
    for i, (nm, ct, bd) in enumerate(geoms):
        col = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'][i%6]
        draw_bbox(ax, bd, col, alpha=0.2)
        ax.scatter(*ct, c=col, s=50, edgecolors='k', zorder=5)

    # 构建夹爪轨迹 (从 log 解析)
    if log_frames:
        # 夹爪 pos_z 来自 log, xy 用 grasp pose 的 xy
        base_xy = grasp_pos[:2]
        gripper_pts = []
        obj_pts = []
        for f in log_frames:
            gripper_pts.append([base_xy[0], base_xy[1], f['pos_z']])
            obj_pts.append([base_xy[0], base_xy[1], f['obj_z']])
        gripper_traj = np.array(gripper_pts)
        obj_traj = np.array(obj_pts)

        # 用 log 首帧 obj_z 作为物体初始位置 (npz 中可能没有 obj_initial_pos)
        if np.linalg.norm(obj_init) < 0.001 and log_frames:
            obj_init = np.array([base_xy[0], base_xy[1], log_frames[0]['obj_z']])
        obj_traj[:, :2] = obj_init[:2]  # 用实际 obj 初始 xy

        # 夹爪轨迹
        ax.plot(gripper_traj[:,0], gripper_traj[:,1], gripper_traj[:,2], 'b-', lw=2.5, label='Gripper Traj', alpha=0.9)
        ax.scatter(*gripper_traj[0], c='blue', s=80, edgecolors='k', zorder=5, label='Gripper Start')
        ax.scatter(*gripper_traj[-1], c='blue', s=80, marker='*', edgecolors='k', zorder=5, label='Gripper End')

        # 物体轨迹
        ax.plot(obj_traj[:,0], obj_traj[:,1], obj_traj[:,2], 'r--', lw=2, label='Object Traj', alpha=0.9)
        ax.scatter(*obj_traj[0], c='red', s=60, edgecolors='k', zorder=5, label='Object Start')
        ax.scatter(*obj_traj[-1], c='red', s=60, marker='*', edgecolors='k', zorder=5, label='Object End')

        # 标注关键帧
        n = len(gripper_traj)
        for fid, col, label in [(10, 'cyan', 'F10'), (30, 'orange', 'F30'), (min(65, n-1), 'red', 'F65')]:
            if fid < n:
                ax.scatter(*gripper_traj[fid], c=col, s=60, edgecolors='k', zorder=6)
                ax.text(*(gripper_traj[fid]+[0,0,0.008]), label, fontsize=8, color=col)

    # 最优 grasp pose + 物体初始/最终位置
    ax.scatter(*grasp_pos, c='lime', s=200, marker='*', edgecolors='k', zorder=10, label='Grasp Pose')
    ax.scatter(*obj_init, c='magenta', s=120, marker='s', edgecolors='k', zorder=10, label='Obj Init')
    if np.linalg.norm(obj_final - obj_init) > 0.001:
        ax.scatter(*obj_final, c='orange', s=120, marker='s', edgecolors='k', zorder=10, label='Obj Final')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=9)

    # 自动范围
    all_pts = [bd for _, _, bd in geoms] + [gripper_traj, obj_traj] if log_frames else [bd for _, _, bd in geoms]
    all_pts = [p for p in all_pts if len(p) > 0]
    if all_pts:
        ag = np.concatenate(all_pts, axis=0)
        lo, hi = ag.min(0), ag.max(0)
        rng = np.maximum(hi-lo, 0.2)
        pad = rng*0.15
        ax.set_xlim3d(lo[0]-pad[0], hi[0]+pad[0])
        ax.set_ylim3d(lo[1]-pad[1], hi[1]+pad[1])
        ax.set_zlim3d(max(lo[2]-pad[2], -0.05), hi[2]+pad[2])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")
    plt.close()

    # 5. 副图: Z 轨迹 + 夹持力
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    if log_frames:
        frames = [f['frame'] for f in log_frames]
        gripper_z = [f['pos_z'] for f in log_frames]
        obj_z = [f['obj_z'] for f in log_frames]
        forces = [f['force'] for f in log_frames]

        axes[0].plot(frames, [z*100 for z in gripper_z], 'b-', lw=2, label='Gripper Z')
        axes[0].plot(frames, [z*100 for z in obj_z], 'r--', lw=2, label='Object Z')
        axes[0].axvline(10, c='cyan', ls=':', alpha=0.7, label='F10')
        axes[0].axvline(30, c='orange', ls=':', alpha=0.7, label='F30')
        axes[0].set_xlabel('Frame'); axes[0].set_ylabel('Z (cm)')
        axes[0].set_title('Z 高度随帧变化'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, forces, 'purple', lw=2)
        axes[1].set_title(f'Grip Force (peak={max(forces):.1f}N)')
        axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Force (N)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = args.output.replace('.png', '_detail.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()
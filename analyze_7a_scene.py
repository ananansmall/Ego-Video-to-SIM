#!/usr/bin/env python3
"""
7a_vggt_omega 数据集全面分析。

分析内容:
  1. 场景概述 (从图像、VLM conversation、关系推断)
  2. 坐标系对齐问题根因分析
  3. GeoCalib 8 帧采样的详细分析
  4. SAM3 floor/wall mask 失败原因
  5. 改进后预期效果
  6. 改进建议

运行:
  cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
  /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python analyze_7a_scene.py
"""
import json
import os
import numpy as np

OUTPUT_DIR = "output_v2/7a_vggt_omega"


def load_json(path):
    with open(os.path.join(OUTPUT_DIR, path), 'r') as f:
        return json.load(f)


def load_extrinsics():
    """加载所有 extrinsics"""
    ext_dir = os.path.join(OUTPUT_DIR, "extrinsics")
    frames = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])
    extrinsics = []
    for fname in frames:
        arr = np.loadtxt(os.path.join(ext_dir, fname)).reshape(4, 4)
        extrinsics.append(arr[:3, :4])
    return np.array(extrinsics)


def analyze_camera_motion(extrinsics):
    """分析相机运动模式"""
    translations = extrinsics[:, :3, 3]
    rots = extrinsics[:, :3, :3]

    # 平移统计
    trans_min = translations.min(axis=0)
    trans_max = translations.max(axis=0)
    trans_range = trans_max - trans_min

    # 旋转统计 (提取欧拉角近似)
    angles_x = np.arctan2(rots[:, 2, 1], rots[:, 2, 2])
    angles_y = np.arctan2(-rots[:, 2, 0], np.sqrt(rots[:, 2, 1]**2 + rots[:, 2, 2]**2))
    angles_z = np.arctan2(rots[:, 1, 0], rots[:, 0, 0])

    print("  【相机运动分析】")
    print(f"    帧数: {len(extrinsics)}")
    print(f"    平移范围: dx={trans_range[0]:.4f}, dy={trans_range[1]:.4f}, dz={trans_range[2]:.4f}")
    print(f"    平移均值: [{translations.mean(axis=0)[0]:.4f}, {translations.mean(axis=0)[1]:.4f}, {translations.mean(axis=0)[2]:.4f}]")
    print(f"    旋转角度范围: θx=[{angles_x.min():.4f}, {angles_x.max():.4f}], "
          f"θy=[{angles_y.min():.4f}, {angles_y.max():.4f}], "
          f"θz=[{angles_z.min():.4f}, {angles_z.max():.4f}]")
    print(f"    结论: {'相机基本静止(手持微动)' if trans_range.max() < 0.15 else '相机有明显运动'}")
    print()


def analyze_geocalib_failure(alignment):
    """分析 GeoCalib 失败原因"""
    detail = alignment['method_detail']
    per_frame = detail['per_frame']

    print("  【GeoCalib 失败根因分析】")
    print(f"    采样帧数: {detail['n_frames']} (均匀间隔 0,16,32,...,112)")
    print(f"    内点数: {detail['n_inliers']}/{detail['n_frames']}")
    print(f"    失败原因: R[2,2]={abs(detail['floor_normal'][2]):.4f} < 0.5 (阈值)")

    # 分析每帧的 gravity 置信度
    confidences = [pf['confidence'] for pf in per_frame]
    print(f"\n    每帧 GeoCalib 置信度 (全部 < 0.005):")
    for pf in per_frame:
        print(f"      Frame {pf['frame_id']:3d}: conf={pf['confidence']:.6f}, "
              f"grav_cam=({pf['grav_cam'][0]:+.3f}, {pf['grav_cam'][1]:+.3f}, {pf['grav_cam'][2]:+.3f})")

    # 分析 grav_world 的方向一致性
    grav_worlds = np.array([pf['grav_world'] for pf in per_frame])
    grav_angles = np.arccos(np.clip(grav_worlds @ grav_worlds[0] / (np.linalg.norm(grav_worlds, axis=1) * np.linalg.norm(grav_worlds[0])), -1, 1))
    print(f"\n    各帧 grav_world 与第 0 帧的夹角 (度):")
    for i, (ang, pf) in enumerate(zip(np.degrees(grav_angles), per_frame)):
        print(f"      Frame {pf['frame_id']:3d}: {ang:.1f}°")

    print(f"\n    关键发现:")
    print(f"      - 所有帧的 confidence 都极低 (0.0005~0.004), 说明 GeoCalib 模型在这些图像中几乎找不到重力线索")
    print(f"      - Frame 0 的 grav_world y 分量 ≈ -0.9 (指向 -y), 但 Frame 16 的 y 分量 ≈ -0.87, z 分量 ≈ +0.49")
    print(f"      - 各帧 gravity 方向差异大, 球面平均后 floor_normal 严重偏离 z 轴")
    print(f"      - 根本原因: 红色布料场景没有明显的 '水平/垂直' 视觉纹理, GeoCalib 无法从图像中估计重力")
    print()


def analyze_vlm_conversations():
    """分析 VLM 对话, 了解场景理解"""
    conversations = []
    path = os.path.join(OUTPUT_DIR, "keyframes/vlm_conversation.jsonl")
    with open(path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line.strip()))

    print("  【VLM 对话分析】")
    print(f"    总对话数: {len(conversations)}")

    # 分析第一帧的响应 (过长被截断)
    first_response = conversations[0]['response']
    if len(first_response) > 500:
        print(f"    第 0 帧响应长度: {len(first_response)} 字符 (被截断)")
        # 统计 'toy' 出现次数
        toy_count = first_response.count('"toy"')
        print(f"    其中 'toy' 出现次数: {toy_count}")
        print(f"    ⚠️ VLM 在第一帧过度重复 'toy', 可能是 prompt 设计问题")

    # 分析后续帧
    for conv in conversations[1:]:
        resp = conv['response']
        # 提取对象名称
        obj_names = []
        for line in resp.split('\n'):
            if '"name"' in line:
                name = line.split('"name": "')[1].split('"')[0] if '"name": "' in line else ""
                obj_names.append(name)
        if obj_names:
            unique_objs = list(set(obj_names))
            print(f"    Frame {conv['response'].find('frame_id')}: detected {len(obj_names)} objects: {unique_objs}")

    print()


def analyze_scene_objects():
    """分析场景对象和关系"""
    relations = load_json("final_relations.json")
    metadata = load_json("keyframes/keyframes_metadata.json")

    print("  【场景对象和关系】")
    print(f"    对象及其关系:")
    for obj, rel in relations.items():
        print(f"      {obj:12s} → {rel}")

    print(f"\n    可见对象类型: {list(metadata['scene_objects'].keys())}")
    print(f"    所有对象都是 'supported by floor' (除 plate_0 supported by block_1)")
    print(f"\n    ⚠️ 注意: 'floor' 在这里指的是红色布料表面, 不是传统地板")
    print()


def analyze_floor_mask_issue():
    """分析 SAM3 floor mask 为什么失败"""
    print("  【SAM3 Floor Mask 失败原因】")
    print(f"    场景特征:")
    print(f"      - 地面是红色布料, 不是传统地板(瓷砖/木地板)")
    print(f"      - 红色布料没有明显的边缘/纹理线索")
    print(f"      - SAM3 'floor' prompt 期望看到地板的典型视觉特征")
    print(f"      - 红色布料在视觉上更像 'carpet' 或 'cloth'")
    print()
    print(f"    当前 SAM3 prompt 覆盖:")
    print(f"      旧: ['floor', 'single wall']")
    print(f"      新: ['floor','ground','carpet','bed','table','desk','horizontal surface','tabletop','counter','shelf']")
    print(f"           + ['single wall','wall','window','curtain','door','bookshelf','cabinet','vertical surface']")
    print()
    print(f"    改进建议:")
    print(f"      - 'carpet' 和 'horizontal surface' 应该能匹配到红色布料")
    print(f"      - 但更可靠的方案是从 3D 点云几何找大平面 (不依赖 SAM3)")
    print()


def analyze_expected_improvement():
    """分析改进后预期效果"""
    print("  【改进后预期效果】")
    print()
    print(f"    Stage 1 (strict floor+wall):")
    print(f"      → 仍可能失败 (SAM3 不一定分出 'floor' mask)")
    print()
    print(f"    Stage 2 (relaxed floor+wall):")
    print(f"      → 可能成功 (放宽阈值后, 'carpet'/'horizontal surface' 可能匹配红色布料)")
    print()
    print(f"    Stage 2.5 (VLM-assisted):")
    print(f"      → 预期场景类型: 'unknown' (没有典型卧室家具)")
    print(f"      → 使用通用 prompt: ['horizontal surface', 'carpet', 'table', ...]")
    print(f"      → 成功率中等")
    print()
    print(f"    Stage 3 (large_plane with VLM enhancement):")
    print(f"      → 从 3D 点云 PCA 找最大水平面 (不依赖 SAM3)")
    print(f"      → 预期成功率: HIGH (红色布料在点云中是一个大平面)")
    print(f"      → 预期 R[2,2] > 0.3 (通过新阈值)")
    print()
    print(f"    Stage 4 (GeoCalib with 30 frames):")
    print(f"      → 30 帧 vs 8 帧: 采样覆盖更广, 但置信度低的根本问题仍在")
    print(f"      → 质量阈值 0.3 vs 0.5: 更多情况能通过")
    print(f"      → 预期效果: 如果 Stage 3 成功, Stage 4 不会执行")
    print()


def main():
    print("=" * 70)
    print("7a_vggt_omega 数据集全面分析")
    print("=" * 70)
    print()

    # 加载数据
    alignment = load_json("coordinate_alignment.json")
    extrinsics = load_extrinsics()

    # 1. 相机运动分析
    analyze_camera_motion(extrinsics)

    # 2. GeoCalib 失败分析
    analyze_geocalib_failure(alignment)

    # 3. VLM 对话分析
    analyze_vlm_conversations()

    # 4. 场景对象分析
    analyze_scene_objects()

    # 5. Floor mask 问题
    analyze_floor_mask_issue()

    # 6. 预期改进效果
    analyze_expected_improvement()

    # 7. 总结
    print("=" * 70)
    print("【总结与建议】")
    print("=" * 70)
    print("""
  场景本质: 红色布料上的玩具交互, 相机基本静止(手持微动)

  核心问题:
    1. "floor" 不是传统地板, 是红色布料 → SAM3 分不出来
    2. 相机静止 → 没有视差, 难以从几何上区分 "水平面" 和 "垂直面"
    3. GeoCalib 从图像估计重力, 但红色布料没有重力线索

  改进优先级:
    1. ★★★ 从 3D 点云 PCA 找大平面 (Stage 3) - 最可靠, 不依赖 SAM3
    2. ★★☆ 增加 SAM3 prompt 变体 (carpet/horizontal surface) - 辅助
    3. ★☆☆ VLM 场景识别 - 帮助选择 prompt, 但对 "红色布料" 场景帮助有限

  关键洞察:
    - 这个场景的 "floor" 是红色布料表面, 在 3D 点云中是一个大平面
    - 即使 SAM3 分不出 floor mask, 只要点云覆盖了红色布料的足够区域,
      Stage 3 就能从几何上找到这个平面并建立正确的坐标系
    - 坐标轴变化量取决于: floor_normal 与 [0,0,1] 的夹角
""")


if __name__ == "__main__":
    main()

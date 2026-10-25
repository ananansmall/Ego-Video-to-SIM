"""倒置朝向判定: render-and-judge (精简版 P0)

对 θ>90° (网格 y 轴朝下) 的实例, 渲染 当前/翻转 两个候选 + 真实帧裁剪图,
让 VLM 二选一; 几何规则不知道"钳子的正方向", 只有看图才能判断。

触发与豁免:
  - 触发: up_z < -0.3 (明显倒置)
  - 豁免: 表面类 (cloth/curtain/rug/...) —— 翻转无语义;
          attached to wall 的实例 —— 表面贴墙, 上下无定义;
          ring 等 up 无意义类别
附带: 单位化网格尺度先验回退 —— SAM3D 偶尔产出最大边≈1.0 的单位化网格
  (rescale guard 拒绝后 k=1 原样保留, 实测 1m 长的钳子), 对此类按类别
  先验最大边缩放, 显著好于 1.0。

用法:
  CUDA_VISIBLE_DEVICES=3 python tools/orientation_judge.py \
      --scene_dir output_v2/egodex_32_rescaled [--apply] [--no_vlm]

--apply: 把判定结果写回 all_instances.pkl 并重导 final_scene_patched.glb;
不带 --apply 只出报告与候选渲染图 (试运行)。
"""
import argparse, glob, json, os, pickle, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# up 无语义的类别: 翻不翻都对, 不该浪费 VLM 调用也不该自动翻
UP_IRRELEVANT = {
    "cloth", "curtain", "drape", "drapery", "rug", "carpet", "mat", "towel",
    "rag", "napkin", "sheet", "blanket", "ring", "band", "rope", "string",
}
# 单位化网格的类别先验 (最大边, 米)。只对最大边∈[0.9,1.1] 的网格生效 ——
# 那是"被归一化到单位立方"的明确信号, 当前尺度必然错误, 先验必然更好。
SCALE_PRIOR = {
    "plier": 0.22, "screwdriver": 0.22, "tray": 0.40, "pillow": 0.55,
    "scissors": 0.20, "knife": 0.25, "fork": 0.20, "spoon": 0.18,
    "cup": 0.12, "mug": 0.12, "bowl": 0.15, "plate": 0.26, "laptop": 0.40,
    "phone": 0.17, "cellphone": 0.17, "remote": 0.20, "remote_control": 0.20,
    "book": 0.25, "bottle": 0.25, "hammer": 0.33, "watch": 0.05,
    "glasses": 0.15, "earbud_case": 0.06, "mouse": 0.12, "keyboard": 0.45,
    "can": 0.12, "wallet": 0.12, "tape": 0.10, "marker": 0.15,
}
UP_THRESH = -0.3          # up_z 低于此视为倒置
UNIT_CUBE_LO, UNIT_CUBE_HI = 0.9, 1.1


# 家具类: 现实中不可能倾斜 >45° 存在 (桌子/椅子/柜子半躺 = 位姿错误),
# 无需 VLM, 直接对齐 +z。几何链的 (10°,170°) "歧义带"对家具过宽 ——
# 实测 HOI4D 092/112 的 table 倾斜 139° 无人修。
FURNITURE_CATEGORIES = {
    "table", "desk", "chair", "stool", "cabinet", "shelf", "bookshelf",
    "nightstand", "dresser", "wardrobe", "bed", "sofa", "couch", "bench",
    "drawer", "tv_stand",
}


def _up_z(info):
    return float((np.asarray(info["T"])[:3, :3] @ np.array([0.0, 1.0, 0.0]))[2])


def upright_align_T(info):
    """把 mesh 的 y 轴对齐到世界 +z (保持质心 xy, 底面重新落地)。"""
    import trimesh
    T = np.asarray(info["T"], dtype=float)
    up = T[:3, 1] / max(np.linalg.norm(T[:3, 1]), 1e-9)
    R = trimesh.geometry.align_vectors(up, np.array([0.0, 0.0, 1.0]))
    m = info["original_mesh"].copy()
    m.apply_transform(T)
    bottom = float(m.bounds[0][2])
    out = dict(info)
    out["T"] = np.asarray(R) @ T
    _reground(out, bottom)
    return out


def _world_mesh(info):
    import trimesh
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], dtype=float))
    return m


def _reground(info, target_bottom):
    """绕包围盒中心翻转/缩放后, 把底面压回 target_bottom。"""
    m = _world_mesh(info)
    dz = target_bottom - float(m.bounds[0][2])
    T = np.asarray(info["T"], dtype=float).copy()
    T[2, 3] += dz
    info["T"] = T


def scale_prior_pass(info, category, report_entry):
    """单位化网格 → 类别先验缩放。

    签名看**局部**包围盒 (SAM3D 归一化发生在网格空间, 实例 T 自带缩放后
    世界尺寸可以是任何值, 实测枕头局部 1.0 → 世界 1.58m);
    目标看**世界**尺寸 (缩放后直接落到先验大小)。
    """
    m_local = info["original_mesh"]
    ext_local = m_local.bounds[1] - m_local.bounds[0]
    mx_local = float(np.max(ext_local))
    if not (UNIT_CUBE_LO <= mx_local <= UNIT_CUBE_HI):
        return False                      # 局部单位化签名不成立, 不动
    prior = SCALE_PRIOR.get(category.lower().replace(" ", "_"))
    if not prior:
        return False
    m_world = _world_mesh(info)
    mx_world = float(np.max(m_world.bounds[1] - m_world.bounds[0]))
    s = prior / max(mx_world, 1e-6)
    if not (0.02 <= s <= 20.0):
        return False
    T = np.asarray(info["T"], dtype=float).copy()
    T[:3, :3] *= s
    info["T"] = T
    _reground(info, float(m_world.bounds[0][2]))
    report_entry["scale_fix"] = {"local_max": round(mx_local, 3),
                                 "world_max_before": round(mx_world, 3),
                                 "prior_max": prior, "scale": round(s, 3)}
    return True


def flip_candidate_T(info):
    """候选翻转: 绕包围盒中心、世界 X 轴转 180° (上下翻转), 底面落回原位。"""
    m = _world_mesh(info)
    b = m.bounds
    c = (b[0] + b[1]) / 2.0
    bottom = float(b[0][2])
    R = np.eye(4)
    R[:3, :3] = np.diag([1.0, -1.0, -1.0])          # 绕 X 轴 180°
    T1 = np.eye(4); T1[:3, 3] = c
    T2 = np.eye(4); T2[:3, 3] = -c
    T_flip = T1 @ R @ T2 @ np.asarray(info["T"], dtype=float)
    out = dict(info)
    out["T"] = T_flip
    _reground(out, bottom)
    return out


def render_pose(info, out_path, size=256):
    """单个姿态的 45° 俯视渲染 (白底)。"""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender, trimesh
    m = _world_mesh(info)
    if isinstance(m.visual, trimesh.visual.ColorVisuals) or m.visual.kind is None:
        m.visual.vertex_colors = [120, 140, 170, 255]
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5] * 3)
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    b = m.bounds
    center = (b[0] + b[1]) / 2
    radius = max(float(np.linalg.norm(b[1] - b[0])) / 2, 0.05)
    d = np.array([0.7, -0.7, 0.8]); d /= np.linalg.norm(d)
    eye = center + d * radius * 2.6
    z = eye - center; z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    scene.add(pyrender.IntrinsicsCamera(fy=size, fx=size, cy=size / 2, cx=size / 2), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
    r = pyrender.OffscreenRenderer(size, size)
    try:
        img, _ = r.render(scene)
    finally:
        r.delete()
    import cv2
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def vlm_pick(model, processor, path_a, path_b, path_real):
    """三图问答: 候选A/候选B/真实帧 → 'A'/'B'/'?'。"""
    import torch
    prompt = (
        "第1张图(A)和第2张图(B)是同一个物体的两种不同摆放姿态的渲染图，"
        "第3张图是这个物体在真实视频中的样子。"
        "请判断哪种摆放姿态与真实照片中的姿态一致。"
        "只回答一个大写字母: A 或 B。如果两种都明显不对或无法判断，回答 ?。"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": path_a},
        {"type": "image", "image": path_b},
        {"type": "image", "image": path_real},
        {"type": "text", "text": prompt},
    ]}]
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip().upper()
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return "?"


def find_real_crop(scene_dir, category, inst_idx):
    pats = [
        f"{scene_dir}/optimal_frames/{category}_inst{inst_idx}_frame*.jpg",
        f"{scene_dir}/optimal_frames/{category}_inst{inst_idx}_*.jpg",
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def load_vlm(ckpt):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, torch_dtype="auto", device_map="cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor


def judge_scene(D, model, processor, apply=False, no_vlm=False):
    """对单个场景做尺度回退 + 倒置判定。返回 report dict。"""
    out_dir = os.path.join(D, "orientation_judge")
    os.makedirs(out_dir, exist_ok=True)

    blob = pickle.load(open(f"{D}/all_instances.pkl", "rb"))
    wrapped = isinstance(blob, dict) and isinstance(blob.get("all_instances"), dict)
    inst = blob["all_instances"] if wrapped else blob

    report = {"scene": D, "judged": [], "exempt": [], "scale_fixed": []}

    for category, lst in inst.items():
        for idx, info in enumerate(lst):
            name = f"{category}_{idx}"
            entry = {"instance": name}

            # ① 尺度先验回退: 独立于倒置判定, 对所有"局部单位化 + 类别命中"
            #    的实例生效 (缩放不改变朝向, 已翻转的也不会被重复判定)
            if scale_prior_pass(info, category, entry):
                report["scale_fixed"].append(name)
                print(f"   {name}: 尺度先验回退 {entry['scale_fix']}", flush=True)

            up = _up_z(info)
            if up > UP_THRESH:
                continue
            entry["up_before"] = round(up, 2)
            cat_std = category.lower().replace(" ", "_")
            rel = ""
            rp = f"{D}/final_relations.json"
            if os.path.isfile(rp):
                rel = json.load(open(rp)).get(name, "")
            if cat_std in UP_IRRELEVANT or "attached to wall" in rel:
                entry.update({"action": "exempt", "reason": "up 无语义/表面类"})
                report["exempt"].append(entry)
                continue

            # ⓪ 家具直立规则 (几何, 无需 VLM): 家具不可能倾斜 >45° 存在,
            #    直接对齐 +z (处理 139° 这类落在几何链"歧义带"的倾斜)
            if cat_std in FURNITURE_CATEGORIES and up < 0.707:
                upright = upright_align_T(info)
                info["T"] = upright["T"]
                entry.update({"action": "upright_aligned",
                              "up_after": round(_up_z(info), 2)})
                report["judged"].append(entry)
                print(f"   {name}: 家具倾斜 up={up:+.2f} → 对齐 +z (up={_up_z(info):+.2f})", flush=True)
                continue

            # ② 渲染两候选 + VLM 二选一
            real = find_real_crop(D, category, idx)
            if model is None or real is None:
                entry.update({"action": "keep",
                              "reason": "无 VLM 或无真实帧参考 (几何结果保留)"})
                report["judged"].append(entry)
                continue
            pa = os.path.join(out_dir, f"{name}_A.png")
            pb = os.path.join(out_dir, f"{name}_B.png")
            render_pose(info, pa)                       # A = 当前 (几何已按规则处理)
            info_b = flip_candidate_T(info)
            render_pose(info_b, pb)                     # B = 翻转
            verdict = vlm_pick(model, processor, pa, pb, real)
            entry.update({"verdict": verdict, "real_ref": os.path.basename(real),
                          "render_A": os.path.basename(pa), "render_B": os.path.basename(pb)})
            if verdict == "B":
                old_T = np.asarray(info["T"], dtype=float).copy()
                info["T"] = np.asarray(info_b["T"], dtype=float).copy()
                _reground(info, float(_world_mesh({"original_mesh": info["original_mesh"],
                                                   "T": old_T}).bounds[0][2]))
                entry["action"] = "flipped"
            else:
                entry["action"] = "kept"
            report["judged"].append(entry)
            print(f"   {name}: up={up:+.2f} → VLM 判 {verdict} ({entry['action']})", flush=True)

    n_flip = sum(1 for e in report["judged"] if e.get("action") == "flipped")
    report["n_flipped"] = n_flip
    json.dump(report, open(f"{D}/orientation_judge_report.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"判定完成: 翻转 {n_flip} 个, 豁免 {len(report['exempt'])} 个, "
          f"尺度回退 {len(report['scale_fixed'])} 个", flush=True)

    if apply:
        if wrapped:
            blob["all_instances"] = inst
        pickle.dump(blob, open(f"{D}/all_instances.pkl", "wb"))
        from mainv3 import save_final_glb
        save_final_glb(inst, D, filename="final_scene_patched.glb")
        print(f"已写回 pkl 并重导 final_scene_patched.glb", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, nargs="+",
                    help="一个或多个场景目录 (VLM 只加载一次)")
    ap.add_argument("--vlm_checkpoint", default="/mnt/data/lza/models/Qwen3.5-9B")
    ap.add_argument("--apply", action="store_true", help="写回 pkl 并重导 GLB")
    ap.add_argument("--no_vlm", action="store_true", help="只做尺度先验回退, 不做 VLM 判定")
    args = ap.parse_args()

    model = processor = None
    if not args.no_vlm:
        print(f"加载 VLM: {args.vlm_checkpoint}", flush=True)
        model, processor = load_vlm(args.vlm_checkpoint)

    for D in args.scene_dir:
        print(f"=== {D}", flush=True)
        judge_scene(D, model, processor, apply=args.apply, no_vlm=args.no_vlm)


if __name__ == "__main__":
    main()

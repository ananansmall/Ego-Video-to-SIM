import numpy as np
import torch
import os
import sys
import json
import time
import multiprocessing as mp
import importlib.util
import traceback
from queue import Empty
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

# ── T6 尺寸比门禁 ──────────────────────────────────────────────────────────
# ratio = SAM3D mesh 世界系包围盒最大边 / 该实例 mask 点云包围盒最大边。
# 点云与 SAM3D 输入同源 (world_points), 是一手观测, 尺度天然正确;
# ratio 显著偏离 1 说明 SAM3D 建大/建小了 (实测 002 pillow 3.82 / toy 0.18,
# 2chair bottle 0.10 / table 1.28 / chair 0.62, 121 mug 0.43, 151 stool 0.37)。
SCALE_GATE_LOW = 0.5
SCALE_GATE_HIGH = 2.0
# 点云少于该数时包围盒不可靠, 跳过门禁 (与 static_mesh.MIN_POINTS 对齐)
SCALE_GATE_MIN_POINTS = 60


def _mesh_world_max_extent(original_mesh, T):
    """mesh 应用 T 后的世界系 AABB 最大边长。"""
    m = original_mesh.copy()
    m.apply_transform(np.asarray(T, dtype=np.float64))
    return float(np.max(m.bounds[1] - m.bounds[0]))


def _scale_gate_instance(category, inst_idx, instance_result, mask,
                         world_points, frame_id):
    """T6: 对单个 SAM3D 实例做尺寸比门禁。

    ratio < 0.5 或 > 2.0 的实例 → 放弃 SAM3D mesh, 改用点云构网
    (src/static_mesh.py 的 build_mesh_from_pointcloud + build_static_instance,
    点云构网产物是世界坐标, T=单位阵)。正常实例原样返回, 不动。

    Returns:
        (instance_result, gate_entry): instance_result 可能被替换;
        gate_entry 为 None 表示点太少/出错无法评估 (不做门禁), 否则是
        一条报告记录 (action: pass / replaced_with_pointcloud /
        kept_sam3d_pointcloud_build_failed)。
    """
    name = f"{category}_inst{inst_idx}"
    try:
        mask = np.asarray(mask)
        p = np.asarray(world_points[frame_id])[mask > 0]
        p = p[np.all(np.isfinite(p), axis=-1)]
        if len(p) < SCALE_GATE_MIN_POINTS:
            return instance_result, None
        mesh_ext = _mesh_world_max_extent(
            instance_result["original_mesh"], instance_result["T"])
        pc_ext = float(np.max(p.max(axis=0) - p.min(axis=0)))
        if mesh_ext <= 1e-9 or pc_ext <= 1e-9:
            return instance_result, None
        ratio = mesh_ext / pc_ext
        entry = {
            "instance": name,
            "category": category,
            "frame_id": int(frame_id),
            "mesh_max_extent": round(mesh_ext, 4),
            "pointcloud_max_extent": round(pc_ext, 4),
            "ratio": round(ratio, 3),
        }
        if SCALE_GATE_LOW <= ratio <= SCALE_GATE_HIGH:
            entry["action"] = "pass"
            return instance_result, entry
        # 超限 → 尝试点云构网替换
        from src.static_mesh import build_mesh_from_pointcloud, build_static_instance
        tm = build_mesh_from_pointcloud(p)
        if tm is None:
            entry["action"] = "kept_sam3d_pointcloud_build_failed"
            print(f"[scale-gate] {name}: ratio={ratio:.2f} 超限但点云构网失败, "
                  f"保留 SAM3D mesh", flush=True)
            return instance_result, entry
        replacement = build_static_instance(tm, category, inst_idx)
        replacement["mesh_source"] = "pointcloud_scale_gate"  # 区分于常规静态路径
        replacement["scale_gate"] = entry                     # 标记随实例走, 供父进程落盘
        entry["action"] = "replaced_with_pointcloud"
        entry["ratio_after"] = round(
            float(np.max(tm.bounds[1] - tm.bounds[0])) / pc_ext, 3)
        print(f"[scale-gate] {name}: ratio={ratio:.2f} 超限 "
              f"(mesh {mesh_ext:.3f}m vs 点云 {pc_ext:.3f}m) → "
              f"替换为点云构网 mesh, ratio→{entry['ratio_after']:.2f}", flush=True)
        return replacement, entry
    except Exception as e:
        return instance_result, {
            "instance": name, "action": "gate_error",
            "error": f"{type(e).__name__}: {e}",
        }


def _resolve_output_dir_from_argv():
    """从命令行参数解析管线输出目录 (mainv2/mainv3 的 --output_path)。

    未显式给出时复刻其默认推导: output_v2/{视频名}_{vggt_model}。
    """
    argv = list(sys.argv)
    for i, a in enumerate(argv):
        if a == "--output_path" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--output_path="):
            return a.split("=", 1)[1]
    video = None
    for flag in ("--input_video", "--input_images"):
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv):
                video = argv[i + 1]
                break
            if a.startswith(flag + "="):
                video = a.split("=", 1)[1]
                break
        if video:
            break
    if video:
        model = "vggt"
        for i, a in enumerate(argv):
            if a == "--vggt_model" and i + 1 < len(argv):
                model = argv[i + 1]
                break
            if a.startswith("--vggt_model="):
                model = a.split("=", 1)[1]
                break
        stem = os.path.splitext(os.path.basename(video))[0]
        return os.path.join(".", "output_v2", f"{stem}_{model}")
    return None


def write_scale_gate_report(all_instances):
    """把被尺寸门禁替换的实例落盘为 <output>/scale_gate_report.json。

    标记在 _scale_gate_instance 里随实例走 (instance["scale_gate"]),
    这里在父进程收集写出 —— 父进程才有命令行, 能解析出输出目录。
    """
    try:
        entries = []
        for _cat, lst in (all_instances or {}).items():
            for x in lst:
                g = x.get("scale_gate") if isinstance(x, dict) else None
                if g:
                    entries.append(dict(g))
        if not entries:
            return
        report = {
            "gate": "scale_ratio",
            "low": SCALE_GATE_LOW,
            "high": SCALE_GATE_HIGH,
            "n_replaced": sum(1 for e in entries
                              if e.get("action") == "replaced_with_pointcloud"),
            "n_kept_sam3d": sum(1 for e in entries
                                if e.get("action") != "replaced_with_pointcloud"),
            "instances": entries,
        }
        out_dir = _resolve_output_dir_from_argv()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "scale_gate_report.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
            print(f"[scale-gate] 报告已写出: {path} "
                  f"(替换 {report['n_replaced']}, 保留 SAM3D {report['n_kept_sam3d']})",
                  flush=True)
        else:
            print(f"[scale-gate] ⚠️ 未能解析输出目录, 报告仅打印: "
                  f"{json.dumps(report, ensure_ascii=False)}", flush=True)
    except Exception as e:
        print(f"[scale-gate] ⚠️ 报告写出失败: {type(e).__name__}: {e}", flush=True)


def generate_3d_asset(image, mask, pointmap, extrinsic, inference):
    '''
    3D asset generation for a single object instance using sam3d. 
    Args:
        image: (H, W, 3) numpy array, RGB image of the scene
        mask: (H, W) numpy array, binary mask of the object instance
        pointmap: (H, W, 3) numpy array, world coordinates of each pixel in the image
        extrinsic: (4, 4) numpy array, camera extrinsic matrix for the image
        inference: sam3d inference object
    Returns:    A dictionary containing:
        - mesh: A trimesh object representing the reconstructed 3D mesh of the object instance in object coordinate system
        - T: the transformation matrix to align the mesh to the world coordinate system
    '''
    H, W = pointmap.shape[:2]
    points_world_flat = pointmap.reshape(-1, 3)  # shape: (H*W, 3)
    ones = np.ones((points_world_flat.shape[0], 1))
    points_world_hom = np.hstack([points_world_flat, ones])  # (N, 4)

    # vggt format to sam3d format
    points_cam_hom = (np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]) @ extrinsic @ points_world_hom.T).T  # (N, 4)
    points_cam_flat = points_cam_hom[:, :3]
    point_map_camera = torch.from_numpy(points_cam_flat).reshape(H, W, 3).to(torch.float32)
    point_map_camera = point_map_camera.contiguous()
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    original_mesh = output["glb"]

    # get the transformation matrix. The transformation is z-up.
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    matrix_l2c = l2c_transform.get_matrix()[0].transpose(0, 1).detach().cpu().numpy()
    matrix_y2z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    matrix_adjust = np.diag([-1, -1, 1, 1])
    matrix_ext_inv = np.linalg.inv(extrinsic)
    final_transform = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z

    return {
        "original_mesh": original_mesh,
        "T": final_transform
    }


def _generate_3d_asset_worker(queue, image, mask, pointmap, extrinsic, config_file, compile_model):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference

        inference = Inference(config_file=config_file, compile=compile_model)
        result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
        queue.put((True, result))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _generate_all_instances_worker(queue, jobs, config_file, compile_model, worker_id=0):
    """SAM3D 子进程: 加载一次权重, 串行生成 jobs 里的所有实例。

    jobs: [(category, inst_idx, job_dict), ...]
    job_dict: {"image", "mask", "pointmap", "extrinsic", "frame_world_points",
               "frame_id"} —— 均为父进程按 optimal_frame_id 切好的单实例输入
    (等价于旧的 colors[frame]/world_points[frame]/extrinsics[frame] 取片,
     但子进程载荷从"整段视频所有帧"降到"每个实例一帧", spawn 传输量小 ~10 倍)。

    多 worker 并行时每个进程加载自己的一份权重 (进程内单例), 实例间用固定
    seed=42 (generate_3d_asset → pipeline.run 内 torch.manual_seed), 与执行
    顺序/进程无关, 生成结果与串行版逐比特一致。
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        _t0 = time.perf_counter()
        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference
        _t_import = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        inference = Inference(config_file=config_file, compile=compile_model)
        _t_load = time.perf_counter() - _t0
        print(f"[SAM3D subprocess w{worker_id}] module import {_t_import:.1f}s, "
              f"model load {_t_load:.1f}s, {len(jobs)} instance(s)", flush=True)

        results = []   # [(category, inst_idx, instance_result), ...] 保持 jobs 顺序
        for category, inst_idx, job in jobs:
            image = job["image"]
            mask = job["mask"]
            optimal_frame_id = job["frame_id"]
            print(
                f"[SAM3D subprocess w{worker_id}] category={category}, frame={optimal_frame_id}, "
                f"mask_sum={int(np.asarray(mask).sum())}",
                flush=True,
            )
            _t0 = time.perf_counter()
            instance_result = generate_3d_asset(
                image, mask, job["pointmap"], job["extrinsic"], inference)
            print(f"[SAM3D subprocess w{worker_id}] {category}_inst{inst_idx} "
                  f"done in {time.perf_counter() - _t0:.1f}s", flush=True)
            # T6 尺寸比门禁: SAM3D mesh 与 mask 点云尺度严重不符的实例
            # 改用点云构网 (点云与 SAM3D 输入同源, 尺度天然正确)。
            # frame_world_points 就是 world_points[optimal_frame_id] 的切片,
            # 以 [frame] 包装 + frame_id=0 的方式索引, 取到的点与旧实现完全一致。
            instance_result, _gate_entry = _scale_gate_instance(
                category, inst_idx, instance_result, mask,
                [job["frame_world_points"]], 0)
            results.append((category, inst_idx, instance_result))
        print(f"[SAM3D subprocess w{worker_id}] finished generating all instances, "
              f"putting results in the queue. This may take a while...", flush=True)
        queue.put((True, results))
    except Exception:
        queue.put((False, traceback.format_exc()))
    finally:
        queue.close()
        # 必须 join_thread 等 feeder 把缓冲全部写进管道再退出, 不能用
        # cancel_join_thread: 多 worker 时父进程按序收包, 后完成者的载荷
        # (上百 MB mesh) 积压在 feeder 缓冲里, cancel 会在进程退出时把缓冲
        # 连同数据一起丢弃 → 父进程 get() 读到截断流永久阻塞
        # (旧单 worker 代码有同样隐患, 父进程并发读才侥幸没触发)。
        queue.join_thread()


def _build_instance_jobs(deduplicated_all_masks, all_optimal_frame_ids,
                         colors, world_points, extrinsics):
    """父进程把 (category, filtered inst_idx) 展开成单实例输入切片列表。

    Returns:
        jobs: [(category, inst_idx, job_dict), ...] —— inst_idx 是过滤后列表
        中的下标 (与旧 worker 的 enumerate 顺序一致)。
    """
    jobs = []
    for category, category_masks in deduplicated_all_masks.items():
        for inst_idx, (instance_masks, optimal_frame_id) in enumerate(
                zip(category_masks, all_optimal_frame_ids[category])):
            mask = next(im["mask"] for im in instance_masks
                        if im["frame_id"] == optimal_frame_id)
            jobs.append((category, inst_idx, {
                "image": colors[optimal_frame_id],
                "mask": mask,
                "pointmap": world_points[optimal_frame_id],
                "extrinsic": extrinsics[optimal_frame_id],
                # 尺寸门禁只需要该实例 optimal 帧的点云
                "frame_world_points": world_points[optimal_frame_id],
                "frame_id": optimal_frame_id,
            }))
    return jobs


def _parse_sam3d_worker_layout(n_jobs):
    """解析多 worker 布局开关 (env, 默认关 = 单进程原行为)。

      RAS_SAM3D_GPUS="1,2"   → 每个 GPU 一个 worker (物理 GPU 号, 父进程用
                               CUDA_VISIBLE_DEVICES 逐个 pin, 互不影响);
      RAS_SAM3D_WORKERS=3    → 同一 GPU 上 3 个 worker (权重各一份, 适合
                               显存充裕时压满单卡);
      两者都设时 RAS_SAM3D_GPUS 优先。worker 数自动截到实例数。
    """
    gpus_env = os.environ.get("RAS_SAM3D_GPUS", "").strip()
    workers_env = os.environ.get("RAS_SAM3D_WORKERS", "").strip()
    gpu_ids = None
    n_workers = 1
    if gpus_env:
        gpu_ids = [g.strip() for g in gpus_env.split(",") if g.strip()]
        if gpu_ids:
            n_workers = len(gpu_ids)
    elif workers_env:
        try:
            n_workers = max(1, int(workers_env))
        except ValueError:
            n_workers = 1
    if n_jobs > 0:
        n_workers = min(n_workers, n_jobs)
    return gpu_ids, max(1, n_workers)


def generate_3d_asset_in_subprocess(
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file="./models/SAM3D/checkpoints/pipeline.yaml",
    compile_model=False,
    only_instances=None,
):
    """only_instances: {category: [inst_idx, ...]} 或 None。
        None = 全部跑 SAM3D (原行为)。
        非 None = 只对这些实例跑 SAM3D, 其余实例从 all_masks 中剔除, 由调用方
        用点云快速路径补上 (见 src/static_mesh.py)。这是"静态物体不烧 SAM3D"
        的提速开关 —— 实测只有 1~2 个实例被抓取的场景里, 绝大部分实例是静态的。

    T11 多 worker 并行 (默认关闭, 行为与旧单进程版完全一致):
        RAS_SAM3D_GPUS="1,2" 或 RAS_SAM3D_WORKERS=N 启用。实例按轮转分给
        N 个子进程, 每个进程各自加载一份 SAM3D 权重 (进程内单例), 结果按原
        实例顺序重组 —— 与串行版输出等价 (seed 固定, 见 worker 注释)。
    """
    if only_instances is not None:
        deduplicated_all_masks = {
            cat: [m for k, m in enumerate(masks)
                  if k in set(only_instances.get(cat, []))]
            for cat, masks in deduplicated_all_masks.items()
        }
        all_optimal_frame_ids = {
            cat: [f for k, f in enumerate(ids)
                  if k in set(only_instances.get(cat, []))]
            for cat, ids in all_optimal_frame_ids.items()
        }

    jobs = _build_instance_jobs(deduplicated_all_masks, all_optimal_frame_ids,
                                colors, world_points, extrinsics)
    gpu_ids, n_workers = _parse_sam3d_worker_layout(len(jobs))

    if not jobs:
        # 与旧实现等价: 空 dict 进 worker 循环后返回 {category: []}
        return {category: [] for category in deduplicated_all_masks}

    ctx = mp.get_context("spawn")
    # 实例轮转分配: job i → worker i % n_workers (大实例天然分散)
    worker_jobs = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        worker_jobs[i % n_workers].append(job)

    _t_total0 = time.perf_counter()
    queues, processes = [], []
    for w_id in range(n_workers):
        queue = ctx.Queue()
        env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_ids is not None:
            # spawn 子进程继承 start() 时刻的 env → 逐 worker pin GPU。
            # 子进程内 torch 在首次 .cuda() 时才读该变量, 恢复父进程 env
            # 不影响已发出的进程。
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[w_id]
        try:
            p = ctx.Process(
                target=_generate_all_instances_worker,
                args=(queue, worker_jobs[w_id], config_file, compile_model, w_id),
            )
            p.start()
        finally:
            if gpu_ids is not None:
                if env_backup is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
        queues.append(queue)
        processes.append(p)

    try:
        # 每路独立收包, 总 deadline 与旧实现的 7200s 一致
        deadline = time.monotonic() + 7200
        worker_results = [None] * n_workers
        for w_id in range(n_workers):
            remaining = max(1.0, deadline - time.monotonic())
            try:
                # Read the payload before join(); joining first can deadlock on Queue finalizers.
                ok, payload = queues[w_id].get(timeout=remaining)
            except Empty:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                for p in processes:
                    p.join(timeout=5)
                raise RuntimeError("subprocess timed out waiting for result")
            if not ok:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    p.join(timeout=5)
                raise RuntimeError(payload)
            worker_results[w_id] = payload

        for p in processes:
            p.join(timeout=30)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        # 按原实例顺序重组: all_instances[cat][inst_idx] 与旧串行版一致
        all_instances = {category: [None] * len(category_masks)
                         for category, category_masks in deduplicated_all_masks.items()}
        for results in worker_results:
            for category, inst_idx, instance_result in results:
                all_instances[category][inst_idx] = instance_result
        missing = [(category, idx)
                   for category, lst in all_instances.items()
                   for idx, x in enumerate(lst) if x is None]
        if missing:
            raise RuntimeError(f"missing instance results: {missing}")
        print(f"[SAM3D subprocess] {len(jobs)} instance(s), {n_workers} worker(s), "
              f"total {time.perf_counter() - _t_total0:.1f}s", flush=True)
        # T6: 被尺寸门禁替换的实例落盘 (需要父进程的命令行来定位输出目录)
        write_scale_gate_report(all_instances)
        return all_instances
    finally:
        for queue in queues:
            queue.close()
            queue.cancel_join_thread()

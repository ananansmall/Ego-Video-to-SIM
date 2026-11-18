"""Recover one native SAM3D object from a same-source SAM3 mask and EF3R pointmap."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ras", type=Path)
    p.add_argument("tracks", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--track-id", type=int, required=True)
    a = p.parse_args()
    from src.instance_generation import generate_3d_asset

    data = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    metadata = json.loads((a.tracks / "frames.json").read_text())
    if Path(metadata["source"]).resolve() != Path(data["video_path"]).resolve():
        raise ValueError("SAM3 masks and metric depths have different sources")
    frame_row = {frame: i for i, frame in enumerate(metadata["frames"])}
    depth_ids = np.asarray(data["depth_frame_indices"])
    depth_valid = np.asarray(data["depth_metric_valid_global"], bool)
    depth_lookup = {int(f): i for i, f in enumerate(depth_ids) if depth_valid[i]}
    available = []
    with (a.tracks / "tracks.csv").open() as stream:
        for row in csv.DictReader(stream):
            f = int(row["source_frame"])
            if int(row["track_id"]) != a.track_id or f not in depth_lookup:
                continue
            k = depth_lookup[f]
            depth = np.asarray(data["depth_metric_global"][k])
            mask = cv2.imread(str(a.tracks / row["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(row["mask"])
            mask = cv2.resize(mask, (depth.shape[1],depth.shape[0]), interpolation=cv2.INTER_NEAREST)>0
            ys, xs = np.where(mask)
            if len(xs)<100 or xs.min()<4 or ys.min()<4 or xs.max()>=depth.shape[1]-4 or ys.max()>=depth.shape[0]-4:
                continue
            metric_good = np.isfinite(depth[mask]) & (depth[mask]>0)
            if metric_good.mean() < .95:
                continue
            aspect = float((ys.max()-ys.min()+1)/(xs.max()-xs.min()+1))
            score = float(np.sqrt(len(xs)) * min(aspect, 5.))
            available.append((score, f, mask, depth, int(k)))
    if not available:
        raise ValueError("No exact depth frame with a usable bottle mask")
    score, frame, mask, depth, k = max(available, key=lambda row: row[0])
    a.out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(a.tracks / "rgb" / f"{frame_row[frame]:06d}.jpg"))
    bgr = cv2.resize(bgr, (depth.shape[1],depth.shape[0]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    K = np.asarray(data["intrinsics_K_global"][k], float)
    yy,xx = np.indices(depth.shape)
    pixels = np.stack([xx,yy,np.ones(depth.shape)],axis=-1)
    pointmap = pixels @ np.linalg.inv(K).T * depth[...,None]
    cv2.imwrite(str(a.out/"source_rgb.png"), bgr)
    cv2.imwrite(str(a.out/"source_mask.png"), mask.astype(np.uint8)*255)
    np.savez_compressed(a.out/"camera_input.npz", source_frame=frame, K=K, pointmap=pointmap)
    candidates = [dict(source_frame=f,rank_score=s) for s,f,*_ in sorted(available,key=lambda row:-row[0])]
    report = dict(source=metadata["source"], source_frame=frame, track_id=a.track_id, candidates=candidates,
                  model="native RAS generate_3d_asset / SAM3D pointmap", world_registration_used=False)
    (a.out/"reconstruction.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"SAM3D exact RGB/mask/metric depth source: F{frame}; {len(mask[mask])} mask pixels on EF3R canvas.",flush=True)
    notebook = a.ras / "sam-3d-objects/notebook/inference.py"
    spec = importlib.util.spec_from_file_location("sam3d_notebook_inference",notebook)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inference = module.Inference(config_file=str(a.ras/"models/SAM3D/checkpoints/pipeline.yaml"),compile=False)
    result = generate_3d_asset(image,mask,pointmap,np.eye(4),inference)
    mesh = result["original_mesh"]
    mesh.export(str(a.out/"bottle_candidate.glb"))
    matrix = result["T"]
    np.save(a.out/"T_camera_mesh.npy",matrix)
    report["mesh_bounds"] = np.asarray(mesh.bounds).tolist()
    report["T_camera_mesh"] = np.asarray(matrix).tolist()
    (a.out/"reconstruction.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Saved native textured object candidate; no world transform applied.",flush=True)


if __name__ == "__main__":
    main()

"""Verify a native SAM3D candidate against its exact original source mask."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    a = p.parse_args()
    rgb = cv2.imread(str(a.path / "source_rgb.png"))
    expected = cv2.imread(str(a.path / "source_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    metadata = json.loads((a.path / "reconstruction.json").read_text())
    mesh = trimesh.load(a.path / "bottle_candidate.glb", force="mesh")
    t = np.load(a.path / "T_camera_mesh.npy")
    with np.load(a.path / "camera_input.npz") as data:
        K = data["K"]
        metric_z = data["pointmap"][...,2]
    camera = np.asarray(mesh.vertices) @ t[:3,:3].T + t[:3,3]
    if not (np.isfinite(camera).all() and np.all(camera[:,2]>0)):
        raise ValueError("Projected geometry crosses the source camera plane")
    points = camera @ K.T
    uv = np.rint(points[:,:2] / points[:, 2:3]).astype(np.int32)
    raster = np.zeros(expected.shape, np.uint8)
    for triangle in uv[mesh.faces]:
        cv2.fillConvexPoly(raster,triangle,1)
    predicted = raster > 0
    good_z = expected & np.isfinite(metric_z) & (metric_z>0)
    report = dict(source_frame=metadata["source_frame"], track_id=metadata["track_id"],
                  silhouette_IoU=float((predicted & expected).sum() / (predicted | expected).sum()),
                  expected_mask_pixels=int(expected.sum()), projected_mask_pixels=int(predicted.sum()),
                  vertices=len(mesh.vertices), faces=len(mesh.faces), material=str(type(mesh.visual.material).__name__),
                  has_texture=any(getattr(mesh.visual.material, k, None) is not None for k in ("baseColorTexture","image")),
                  depth_median_mask_m=float(np.median(metric_z[good_z])),
                  depth_vertex_median_m=float(np.median(camera[:,2])))
    overlay = rgb.copy()
    colors = [(35,240,65),(255,180,30)]
    for layer,color in zip((expected,predicted),colors):
        contours,_ = cv2.findContours(layer.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,color,1)
    comparison = np.hstack([rgb,overlay])
    cv2.rectangle(comparison,(0,0),(comparison.shape[1],36),(12,12,12),-1)
    cv2.putText(comparison, f"Source F{metadata['source_frame']}   SAM3 mask (green)  vs  SAM3D mesh projection (blue)    IoU {report['silhouette_IoU']:.3f}", (8,23),0,.43,(240,240,240),1,cv2.LINE_AA)
    cv2.imwrite(str(a.path/"bottle_backprojection.jpg"),comparison)
    (a.path/"bottle_backprojection.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()

"""Measure whether source video frames can be relocalized from static RAS points."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit(query, anchors, detector, matcher, intrinsic):
    kp, desc = detector.detectAndCompute(cv2.cvtColor(query, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        return None
    best = None
    for source_frame, keypoints, descriptor, points, conf in anchors:
        pairs = matcher.knnMatch(descriptor, desc, k=2)
        first = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .73 * b.distance]
        if len(first) < 12:
            continue
        pixels = np.rint([keypoints[m.queryIdx].pt for m in first]).astype(int)
        xy = np.asarray([kp[m.trainIdx].pt for m in first], np.float64)
        u = np.clip(pixels[:, 0], 0, points.shape[1] - 1)
        v = np.clip(pixels[:, 1], 0, points.shape[0] - 1)
        xyz = points[v, u].astype(np.float64)
        valid = np.isfinite(xyz).all(1) & (conf[v, u] >= np.nanmedian(conf))
        xyz, xy = xyz[valid], xy[valid]
        if len(xyz) < 12:
            continue
        ok, r, t, inliers = cv2.solvePnPRansac(
            xyz, xy, intrinsic, None, iterationsCount=400, reprojectionError=3.,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or len(inliers) < 10:
            continue
        indexes = inliers.ravel()
        r, t = cv2.solvePnPRefineLM(xyz[indexes], xy[indexes], intrinsic, None, r, t)
        projection = cv2.projectPoints(xyz[indexes], r, t, intrinsic, None)[0][:, 0]
        errors = np.linalg.norm(projection - xy[indexes], axis=1)
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(r)[0]
        transform[:3, 3] = t.ravel()
        candidate = (len(indexes), source_frame, float(np.quantile(errors, .95)), transform)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    detector, matcher = cv2.SIFT_create(nfeatures=2000), cv2.BFMatcher()
    with np.load(args.map, allow_pickle=False) as mapping:
        rgb, frames, points, conf, K = (mapping[k] for k in ('rgb', 'source_frames', 'points', 'confidence', 'K_mean'))
        original = mapping['w2c']
    train = []
    for i in np.flatnonzero(np.arange(len(frames)) % 4 != 0):
        kp, desc = detector.detectAndCompute(cv2.cvtColor(rgb[i], cv2.COLOR_RGB2GRAY), None)
        if desc is not None:
            train.append((int(frames[i]), kp, desc, points[i], conf[i]))
    rows = []
    for i in np.arange(0, len(frames), 4):
        f = int(frames[i])
        neighbors = sorted(train, key=lambda row: abs(f - row[0]))[:6]
        pose = fit(rgb[i], neighbors, detector, matcher, K.astype(float))
        if pose is None:
            rows.append(dict(source_frame=f, localized=False))
        else:
            n, anchor, error, w2c = pose
            w0 = original[i]
            center = -w2c[:3, :3].T @ w2c[:3, 3]
            center0 = -w0[:3, :3].T @ w0[:3, 3]
            angle = np.degrees(np.arccos(np.clip((np.trace(w2c[:3, :3] @ w0[:3, :3].T)-1)/2, -1, 1)))
            rows.append(dict(source_frame=f, localized=True, reference_frame=anchor, inliers=n,
                             reprojection_p95_px=error, camera_center_delta_ras=float(np.linalg.norm(center-center0)),
                             camera_angle_delta_deg=float(angle), w2c=w2c.tolist()))
    args.out.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Localized {sum(row["localized"] for row in rows)}/{len(rows)} held-out RAS source frames')


if __name__ == '__main__':
    main()

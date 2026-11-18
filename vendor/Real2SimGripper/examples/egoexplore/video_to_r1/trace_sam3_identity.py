"""Trace SAM3 text-prompted instances on dense source windows and the gap between them."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--query", required=True)
    p.add_argument("--interval", nargs=2, type=int, action="append", required=True)
    p.add_argument("--dense-stride", type=int, default=5)
    p.add_argument("--gap-stride", type=int, default=75)
    p.add_argument("--ef3r", type=Path, help="Include the exact source frames with recent metric depth")
    a = p.parse_args()
    cv2.setNumThreads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    rgb_dir = a.out / "rgb"
    rgb_dir.mkdir(exist_ok=False)
    frames = set(range(a.interval[0][0], a.interval[-1][-1] + 1, a.gap_stride))
    frames.update(f for lo, hi in a.interval for f in range(lo, hi+1, a.dense_stride))
    if a.ef3r:
        depth = torch.load(a.ef3r, map_location="cpu", weights_only=True, mmap=True)
        if Path(depth["video_path"]).resolve() != a.video.resolve():
            raise ValueError("Metric depth belongs to a different video")
        exact = np.asarray(depth["depth_frame_indices"])[np.asarray(depth["depth_metric_valid_global"])]
        frames.update(int(f) for f in exact if any(lo <= f <= hi for lo, hi in a.interval))
    frames = sorted(frames)
    frame_row = {f: index for index, f in enumerate(frames)}
    video = cv2.VideoCapture(str(a.video))
    video.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    for frame in range(frames[0], frames[-1]+1):
        if not video.grab() or int(video.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
            raise ValueError(f"Cannot seek to source frame {frame}")
        if frame in frame_row:
            ok, image = video.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode source frame {frame}")
            image = cv2.resize(image, (960, 486), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(rgb_dir / f"{frame_row[frame]:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    video.release()
    metadata = dict(source=str(a.video.resolve()), frames=frames, intervals=a.interval, query=a.query)
    (a.out / "frames.json").write_text(json.dumps(metadata, indent=2) + "\n")
    predictor = build_sam3_video_predictor(checkpoint_path=str(a.checkpoint), gpus_to_use=[0])
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        session = predictor.handle_request({"type":"start_session", "resource_path":str(rgb_dir)})["session_id"]
        predictor.handle_request({"type":"add_prompt", "session_id":session, "frame_index":0, "text":a.query})
        rows = []
        for response in predictor.handle_stream_request({"type":"propagate_in_video", "session_id":session}):
            source_frame = frames[response["frame_index"]]
            outputs = response["outputs"]
            for object_id, score, mask in zip(outputs["out_obj_ids"], outputs["out_probs"], outputs["out_binary_masks"]):
                mask = np.asarray(mask, bool).squeeze()
                ys, xs = np.where(mask)
                if not len(xs):
                    continue
                path = f"F{source_frame:06d}_id{int(object_id):04d}.png"
                cv2.imwrite(str(a.out / path), mask.astype(np.uint8)*255)
                rows.append(dict(source_frame=source_frame, track_id=int(object_id), confidence=float(score),
                                 mask=path, area_px=len(xs), x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max())))
    with (a.out / "tracks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames="source_frame track_id confidence mask area_px x0 y0 x1 y1".split())
        writer.writeheader()
        writer.writerows(rows)
    print(f"SAM3 measured {len(rows)} masks, {len({r['track_id'] for r in rows})} track IDs, {len(frames)} source frames.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compose the source video at left and a frame-matched 2x2 sim render at right."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def frame_stream(path):
    if path.suffix.lower() == ".gif":
        with Image.open(path) as gif:
            for i in range(gif.n_frames):
                gif.seek(i)
                yield gif.convert("RGB")
    else:
        with imageio.get_reader(path) as reader:
            for frame in reader:
                yield Image.fromarray(frame)


def compose(source, simulation, output, panels, fps):
    with imageio.get_reader(source) as video:
        source_frames = video.count_frames()
        source_fps = float(video.get_meta_data()["fps"])
    if fps is None:
        fps = source_fps
    if abs(fps - source_fps) > 0.001:
        raise ValueError("Output fps must match the source fps")
    sim_frames = frame_stream(simulation)
    header = ImageFont.truetype("DejaVuSans.ttf", 20)
    tag = ImageFont.truetype("DejaVuSans.ttf", 15)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, macro_block_size=8, quality=8) as writer:
        for i, orig in enumerate(frame_stream(source)):
            sim = next(sim_frames, None)
            if sim is None:
                raise ValueError(f"Simulation ended before source frame {i}")
            canvas = Image.new("RGB", (1384, 776), (18, 22, 29))
            orig = ImageOps.contain(orig, (405, 720), Image.Resampling.LANCZOS)
            canvas.paste(orig, ((408 - orig.width) // 2, 40 + (720 - orig.height) // 2))
            canvas.paste(sim.resize((960, 720), Image.Resampling.LANCZOS), (416, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 9), "SOURCE VIDEO", fill=(242, 244, 247), font=header)
            draw.text((424, 9), "ROBOT SIMULATION", fill=(242, 244, 247), font=header)
            draw.text((1180, 11), f"F{i:03d}   {i / fps:05.2f}s", fill=(203, 211, 222), font=tag)
            for name, (x, y) in zip(panels, ((416, 40), (896, 40), (416, 400), (896, 400))):
                size = draw.textbbox((0, 0), name, font=tag)[2]
                draw.rectangle((x + 5, y + 5, x + size + 17, y + 28), fill=(20, 24, 31))
                draw.text((x + 11, y + 7), name, fill=(243, 246, 250), font=tag)
            writer.append_data(np.asarray(canvas))
        if i + 1 != source_frames or next(sim_frames, None) is not None:
            raise ValueError("Source and simulation frame counts differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--panels", nargs=4, default=[
        "Front | physics", "Side | physics", "Robot FPV | physics", "Overhead | physics + point cloud",
    ])
    args = parser.parse_args()
    compose(args.source, args.sim, args.out, args.panels, args.fps)


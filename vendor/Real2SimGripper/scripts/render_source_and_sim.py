#!/usr/bin/env python3
"""Compose the source video at left and a frame-matched 2x2 sim render at right."""
import argparse
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from replay_timing import source_time_map


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


def mapped_source_images(source, frames):
    decoded = iter(frame_stream(source))
    index = -1
    for frame in frames:
        while index < frame:
            image = next(decoded)
            index += 1
        yield image


def compose(source, simulation, output, panels, fps, source_start=0, source_stop=None,
            trace=None, playback_fps=None):
    with imageio.get_reader(source) as video:
        source_frames = video.count_frames()
        source_fps = float(video.get_meta_data()["fps"])
    if fps is None:
        fps = source_fps
    if abs(fps - source_fps) > 0.001:
        raise ValueError("Source fps must match the original video")
    if source_stop is None:
        source_stop = source_frames
    if not 0 <= source_start < source_stop <= source_frames:
        raise ValueError("Source frame range is outside the video")
    sim_frames = frame_stream(simulation)
    source_images = islice(frame_stream(source), source_start, source_stop)
    source_times, sim_times = None, None
    if trace is not None:
        with np.load(trace, allow_pickle=False) as data:
            source_times = source_time_map(data, data['source_frames'])
            if source_times is None:
                raise ValueError('--trace requires a source_time_frames timeline')
            sim_times = (np.asarray(data['sim_time_s'], float) if 'sim_time_s' in data
                         else np.cumsum(data['physics_substeps']) * float(data['dt']))
        if sim_times.shape != source_times.shape or not np.isfinite(sim_times).all() or np.any(np.diff(sim_times) <= 0):
            raise ValueError('Simulation times must align with the replay rows and be strictly increasing')
        source_rows = np.floor(source_times + 1e-8).astype(int)
        if source_rows[0] < source_start or source_rows[-1] >= source_stop:
            raise ValueError('Mapped source frame is outside the requested video range')
        source_images = mapped_source_images(source, source_rows)
    header = ImageFont.truetype("DejaVuSans.ttf", 20)
    tag = ImageFont.truetype("DejaVuSans.ttf", 15)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".compose-") as scratch:
        intermediate = Path(scratch) / output.name
        render_frames(source_images, sim_frames, intermediate, panels, fps,
                      source_start, source_stop, header, tag, source_times, sim_times, playback_fps)
        intermediate.replace(output)


def render_frames(source_images, sim_frames, output, panels, fps, source_start, source_stop, header, tag,
                  source_times=None, sim_times=None, playback_fps=None):
    frames_written = 0
    with imageio.get_writer(output, fps=playback_fps or fps, macro_block_size=8, quality=8) as writer:
        for i, orig in enumerate(source_images, start=source_start):
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
            if source_times is None:
                stamp = f"F{i:03d}   {i / fps:05.2f}s"
            else:
                source_time = source_times[frames_written]
                stamp = f"F{int(source_time + 1e-8):03d}  sim {sim_times[frames_written]:05.2f}s"
            draw.text((1180, 11), stamp, fill=(203, 211, 222), font=tag)
            for name, (x, y) in zip(panels, ((416, 40), (896, 40), (416, 400), (896, 400))):
                size = draw.textbbox((0, 0), name, font=tag)[2]
                draw.rectangle((x + 5, y + 5, x + size + 17, y + 28), fill=(20, 24, 31))
                draw.text((x + 11, y + 7), name, fill=(243, 246, 250), font=tag)
            writer.append_data(np.asarray(canvas))
            frames_written += 1
        expected = source_stop - source_start if source_times is None else len(source_times)
        if frames_written != expected or next(sim_frames, None) is not None:
            raise ValueError("Source and simulation frame counts differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--trace", type=Path, help="Use the exec_trace.npz source-time mapping and simulation clock")
    parser.add_argument("--source-start-frame", type=int, default=0)
    parser.add_argument("--source-stop-frame", type=int, help="Exclusive source frame index")
    parser.add_argument("--panels", nargs=4, default=[
        "Front | physics", "Side | physics", "Robot FPV | physics", "Overhead | physics + point cloud",
    ])
    args = parser.parse_args()
    compose(args.source, args.sim, args.out, args.panels, args.fps,
            args.source_start_frame, args.source_stop_frame, args.trace)


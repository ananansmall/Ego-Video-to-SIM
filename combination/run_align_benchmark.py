#!/usr/bin/env python3
"""
run_align_benchmark.py — Batch runner for alignment visualization benchmarks.

Tests multiple hawor/ras dataset pairs and outputs results to a structured directory.
Generates an index.html for easy browsing.

Usage:
    python run_align_benchmark.py \\
        --datasets \\
            hoi4d=/home/an/data/hawor/hoi4d,/home/an/data/ras/hoi4d1_vggt_omega \\
            dataset7=/home/an/data/hawor/7,/home/an/data/ras/my_7mp4_result \\
        --output /home/an/robot_world_ws/src/dex-retargeting/example/combination/output/benchmark
"""
import argparse, os, sys, subprocess, json, shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
VIEW_SCRIPT = SCRIPT_DIR / 'View' / 'vis_interactive_3d.py'

def parse_datasets(ds_list):
    """Parse --datasets entries: name=hawor_dir,ras_dir"""
    result = {}
    for entry in ds_list:
        if '=' in entry:
            name, pair = entry.split('=', 1)
        else:
            name = f'dataset_{len(result)}'
            pair = entry
        parts = [p.strip() for p in pair.split(',')]
        if len(parts) == 2:
            result[name] = tuple(parts)
        else:
            print(f'[WARN] Skipping invalid entry: {entry}')
    return result

def main():
    parser = argparse.ArgumentParser(description='Batch alignment benchmark runner')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Datasets in format: name=hawor_dir,ras_dir')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--axis-len', type=float, default=0.5)
    parser.add_argument('--max-faces', type=int, default=30000)
    parser.add_argument('--rebuild', action='store_true', help='Rebuild even if exists')
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'========================================')
    print(f'Alignment Benchmark Runner')
    print(f'========================================')
    print(f'Output: {out_dir}')
    print(f'Datasets: {len(datasets)}')
    for name, (hd, rd) in datasets.items():
        print(f'  {name}: hawor={hd}  ras={rd}')
    print()

    results = []
    for name, (hawor_dir, ras_dir) in datasets.items():
        print(f'─── {name} ───')
        ds_out = out_dir / name
        ds_out.mkdir(parents=True, exist_ok=True)

        # Check if already exists
        png_path = ds_out / f'{name}_3d.png'
        html_path = ds_out / f'{name}_3d.html'
        if not args.rebuild and png_path.exists() and html_path.exists():
            print(f'  [SKIP] Already exists, use --rebuild to regenerate')
            results.append({'name': name, 'status': 'skipped', 'png': str(png_path), 'html': str(html_path)})
            continue

        # Run the visualization script
        cmd = [
            sys.executable, str(VIEW_SCRIPT),
            '--hawor-dir', hawor_dir,
            '--ras-dir', ras_dir,
            '--output', str(ds_out / name),
            '--num-samples', str(args.num_samples),
            '--axis-len', str(args.axis_len),
            '--max-faces', str(args.max_faces),
        ]
        print(f'  Running: {" ".join(cmd)}')
        t0 = datetime.now()
        ret = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=300)
        elapsed = (datetime.now() - t0).total_seconds()

        # Collect output
        for line in ret.stdout.split('\n'):
            if line.strip():
                print(f'    {line.strip()}')
        if ret.stderr:
            for line in ret.stderr.split('\n'):
                if line.strip() and 'warning' not in line.lower() and 'warning' not in line.lower():
                    print(f'  [ERR] {line.strip()}')

        # Get stats from output
        stats = {}
        for line in ret.stdout.split('\n'):
            if 'Scale=' in line:
                parts = line.strip().split()
                for p in parts:
                    if 'Scale=' in p: stats['scale'] = p.replace('Scale=', '')
                    if 'Rot=' in p: stats['rot'] = p.replace('Rot=', '')
                    if 'Frames=' in p: stats['frames'] = p.replace('Frames=', '')
            if '01c]' in line and 'GLB_Z' in line:
                hand = 'left' if 'Left' in line else 'right'
                stats[f'{hand}_glb_z'] = line.split('GLB_Z=')[-1].strip()

        results.append({
            'name': name,
            'status': 'ok' if ret.returncode == 0 else 'failed',
            'elapsed': f'{elapsed:.1f}s',
            'png': str(png_path),
            'html': str(html_path),
            'stats': stats,
        })
        print(f'  Done in {elapsed:.1f}s (status={ret.returncode})')
        print()

    # Generate index.html
    index_path = out_dir / 'index.html'
    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Alignment Benchmark</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #333; }
.dataset { background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.dataset h2 { margin-top: 0; color: #1a73e8; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
.stat { background: #e8f0fe; padding: 5px 12px; border-radius: 4px; font-size: 14px; }
.stat-label { font-weight: bold; color: #555; }
.stat-value { color: #333; }
.bad { color: #d93025; }
.good { color: #188038; }
.images { display: flex; flex-wrap: wrap; gap: 20px; margin: 10px 0; }
.images img { max-width: 600px; border: 1px solid #ddd; border-radius: 4px; }
.links a { text-decoration: none; color: #1a73e8; margin-right: 15px; font-weight: bold; }
.links a:hover { text-decoration: underline; }
</style></head><body>
<h1>Alignment Benchmark Report</h1>
<p>Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
<hr>
''')
    for r in results:
        html_parts.append(f'<div class="dataset">')
        html_parts.append(f'<h2>{r["name"]}</h2>')
        html_parts.append(f'<div class="links">')
        if r['html'] and os.path.exists(r['html']):
            rel = os.path.relpath(r['html'], str(out_dir))
            html_parts.append(f'<a href="{rel}" target="_blank">Open Interactive 3D</a>')
        html_parts.append('</div>')
        if r.get('stats'):
            s = r['stats']
            html_parts.append('<div class="stats">')
            html_parts.append(f'<div class="stat"><span class="stat-label">Scale:</span> <span class="stat-value">{s.get("scale","?")}×</span></div>')
            html_parts.append(f'<div class="stat"><span class="stat-label">Rotation:</span> <span class="stat-value">{s.get("rot","?")}</span></div>')
            html_parts.append(f'<div class="stat"><span class="stat-label">Frames:</span> <span class="stat-value">{s.get("frames","?")}</span></div>')
            for k in ['left_glb_z', 'right_glb_z']:
                if k in s:
                    z = s[k]
                    bad = '[-' in z
                    cls = 'bad' if bad else 'good'
                    html_parts.append(f'<div class="stat"><span class="stat-label">{k}:</span> <span class="stat-value {cls}">{z}</span></div>')
            if r.get('elapsed'):
                html_parts.append(f'<div class="stat"><span class="stat-label">Time:</span> <span class="stat-value">{r["elapsed"]}</span></div>')
            html_parts.append('</div>')
        # PNG images
        html_parts.append('<div class="images">')
        if r['png'] and os.path.exists(r['png']):
            rel = os.path.relpath(r['png'], str(out_dir))
            html_parts.append(f'<a href="{rel}"><img src="{rel}" alt="{r["name"]}"></a>')
        html_parts.append('</div>')
        html_parts.append('</div>')

    html_parts.append('''
<div style="margin-top:30px;color:#666;font-size:12px;text-align:center;">
  Generated by Alignment Benchmark Runner
</div>
</body></html>''')

    with open(index_path, 'w') as f:
        f.write('\n'.join(html_parts))
    print(f'Index: {index_path}')

    # Summary
    print(f'\n{"="*40}')
    print(f'Summary: {len(results)} datasets')
    for r in results:
        s = r.get('stats', {})
        z_str = f'  LZ={s.get("left_glb_z","?")}  RZ={s.get("right_glb_z","?")}' if s else ''
        print(f'  {r["name"]}: {r["status"]} {z_str} ({r.get("elapsed","?")})')
    print(f'{"="*40}')
    print(f'Output: {out_dir}/')
    print(f'Open index.html in your browser to view all results.')

if __name__ == '__main__':
    main()
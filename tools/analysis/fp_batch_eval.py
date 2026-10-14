"""批量 FP 评测: 对场景里所有「真实物体」实例跑 FoundationPose 跟踪并评测。

关键设计(都是踩过的坑):
  1. 实例去重: SAM3 会把同一物体重复标到多个类别(实测 7_vggt_omega 里
     toy_inst0 与 block_inst2 的 IoU=1.000), 必须先按 mask IoU 去重,
     否则会对同一个物体重复跑 FP 且互相干扰。
  2. mesh 选择: stage3 去重后实例号与 urdf link 号完全无关,
     投影面积粗筛会选错(实测 toy_inst5 面积猜 toy_1 错, 实际是 toy_0)。
     判据用 register 后的投影 IoU。
  3. 每个 mesh 候选必须在**独立子进程**里试: FoundationPose 复用 glctx /
     模型缓存, 同进程内连续register 多个 mesh 会互相污染
     (实测 register IoU 0.649 但同进程跟踪后掉到 0.264)。
  4. 帧0 mask 为空的实例直接跳过 —— FP 的 register 必须要 ob_mask。

用法:
  python fp_batch_eval.py --scene_dir <sd> [--objs toy_inst1,toy_inst5]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
PY_MAIN = '/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python'
PY_FP = '/mnt/data/lza/conda_envs/foundationpose/bin/python'


def dedup_instances(sd, iou_thr=0.9):
    """列出场景里的实例。

    直接读 all_instances.pkl —— 它就是 mainv3 去重**后**的实例集合
    (self/cross_category_deduplicate 已在 mainv3 第544 行完成),
    不需要在这里再去重。mesh 也自带: instance_info['original_mesh'],
    真实尺寸的缩放在 instance_info['T'] 里。
    """
    with open(os.path.join(sd, 'all_instances.pkl'), 'rb') as f:
        ai = pickle.load(f)['all_instances']
    out = []
    for cat, lst in ai.items():
        for i, info in enumerate(lst):
            nv = len(info['original_mesh'].vertices)
            out.append((f'{cat}_inst{i}', cat, i, nv, None))
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--objs', default=None, help='逗号分隔; 默认全部去重后实例')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--n_frames', type=int, default=0)
    ap.add_argument('--min_iou', type=float, default=0.35,
                    help='register IoU 低于此值视为无可用 mask(物体不存在/严重遮挡)')
    ap.add_argument('--skip_select', action='store_true',
                    help='复用已有的 fp_mask_select/<obj>.json, 不重跑 register 选 mask')
    args = ap.parse_args()
    sd = args.scene_dir

    print('=== instance dedup ===')
    insts = dedup_instances(sd)
    if args.objs:
        want = set(args.objs.split(','))
        insts = [x for x in insts if x[0] in want]
    for key, cat, i, nv, _ in insts:
        print(f'  {key}: {nv} verts')

    env = dict(os.environ, HOME='/tmp', CUDA_VISIBLE_DEVICES=args.gpu)
    results = {}
    for key, cat, i, nv, _ in insts:
        print(f'\n=== {key} ===')
        data = os.path.join(sd, f'fp_data_{key}')
        out = os.path.join(sd, f'fp_out_{key}')
        # mesh 由 all_instances.pkl 提供(prepare 里已 apply_transform(T) 到真实尺度),
        # 不再需要从 urdf 目录猜 —— 实测那是错的。
        mesh = os.path.join(data, 'mesh.obj')

        # --- 1. 选 mask: 对每个候选 _pre_dedup mask 跑 register, 取投影 IoU 最高 ---
        # (几何启发式都不可靠, 详见 fp_select_mask.py 的 docstring)
        sel_json = os.path.join(sd, 'fp_mask_select', f'{key}.json')
        if not (args.skip_select and os.path.isfile(sel_json)):
            rc, so, se = run([PY_FP, os.path.join(HERE, 'fp_select_mask.py'),
                              '--scene_dir', sd, '--obj', key, '--gpu', args.gpu])
            for l in (so + se).splitlines():
                if 'register IoU' in l or l.startswith('best:'):
                    print('  ' + l.strip())
            if not os.path.isfile(sel_json):
                print('  select FAILED')
                results[key] = {'status': 'select_failed'}
                continue
            with open(sel_json) as f:
                sel = json.load(f)
            if sel['best_iou'] < args.min_iou:
                print(f'  best register IoU={sel["best_iou"]:.3f} < '
                      f'{args.min_iou} -> no usable mask, skip')
                results[key] = {'status': 'no_usable_mask',
                                'best_iou': sel['best_iou']}
                continue

        # --- 2. prepare(用选中的 mask) ---
        if not os.path.isfile(os.path.join(data, 'mask', '000000.png')):
            rc, so, se = run([PY_MAIN, os.path.join(HERE, 'fp_prepare_data.py'),
                              '--scene_dir', sd, '--obj', key, '--out_dir', data])
            if rc != 0:
                tail = (se or so).strip().splitlines()
                print(f'  prepare FAILED: {tail[-1] if tail else rc}')
                results[key] = {'status': 'prepare_failed'}
                continue
            for l in so.splitlines():
                if 'using _pre_dedup' in l or 'mesh ' in l:
                    print('  ' + l.strip())

        if not os.path.isfile(mesh):
            print('  no mesh -> skip')
            results[key] = {'status': 'no_mesh'}
            continue

        # --- 3. track ---
        cmd = [PY_FP, os.path.join(HERE, 'fp_run_tracking.py'),
               '--scene_dir', sd, '--obj', key, '--data_dir', data,
               '--out_dir', out, '--gpu', args.gpu]
        if args.n_frames:
            cmd += ['--n_frames', str(args.n_frames)]
        rc, so, se = run(cmd)
        if rc != 0:
            tail = (se or so).strip().splitlines()
            print(f'  track FAILED: {tail[-1] if tail else rc}')
            results[key] = {'status': 'track_failed'}
            continue
        # 精确评测(面片光栅化口径)
        rc2, so2, se2 = run([PY_MAIN, os.path.join(HERE, 'fp_iou_eval.py'),
                             '--scene_dir', sd, '--obj', key,
                             '--data_dir', data, '--out_dir', out])
        ious = {}
        for l in so2.splitlines():
            if '面片光栅' in l:
                print('  ' + l)
                ious['mean'] = float(l.split('mean=')[1].split()[0])
            elif '自动判定' in l:
                print('  ' + l)
            elif '静止段' in l or '运动段' in l:
                print('  ' + l)
                ious['static' if '静止段' in l else 'moving'] = \
                    float(l.split('mean=')[1].split()[0])
        results[key] = {'status': 'ok', 'iou': ious}

    rp = os.path.join(sd, 'fp_batch_report.json')
    with open(rp, 'w') as f:
        json.dump(results, f, indent=2)
    print('\n=== summary ===')
    print(f"  {'instance':16s} {'status':14s} {'mean':>6s} {'static':>7s} {'moving':>7s}")
    for k, v in results.items():
        i = v.get('iou', {})
        f = lambda x: f'{i[x]:.3f}' if x in i else '-'
        print(f"  {k:16s} {v['status']:14s} {f('mean'):>6s} {f('static'):>7s} {f('moving'):>7s}")
    print(f'\n-> {rp}')


if __name__ == '__main__':
    main()
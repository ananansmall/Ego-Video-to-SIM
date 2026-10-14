"""生成可点击的 HTML 索引页(汇总 batch评测 + GIF + 轨迹图)。

卡片**静态输出**到 HTML, 不靠 JS 动态生成 —— 实测部分浏览器会在
页面加载完成前中止 JS, 导致动态渲染的卡片一个都不出现。
"""
import argparse
import html
import json
import os

import numpy as np
import trimesh


def collect(sd):
    batch = json.load(open(os.path.join(sd, 'fp_batch_report.json')))
    data = []
    for key in batch:
        vis = os.path.join(sd, 'fp_vis', key)
        gif, traj = os.path.join(vis, 'overlay.gif'), os.path.join(vis, 'traj.png')
        if not (os.path.isfile(gif) and os.path.isfile(traj)):
            continue
        sel = json.load(open(os.path.join(sd, 'fp_mask_select', f'{key}.json')))
        rep = json.load(open(os.path.join(sd, f'fp_out_{key}', 'iou_face_report.json')))
        m = trimesh.load(os.path.join(sd, f'fp_data_{key}', 'mesh.obj'))
        wdir = os.path.join(sd, f'fp_out_{key}', 'ob_in_world')
        n = len(os.listdir(wdir))
        P = np.stack([np.loadtxt(os.path.join(wdir, f'{t:06d}.txt')) for t in range(n)])
        C = P[:, :3, 3]
        step = np.linalg.norm(np.diff(C, axis=0), axis=1)
        data.append({
            'key': key,
            'reg_iou': sel['best_iou'],
            'mean_iou': rep['mean'],
            'size_mm': ' x '.join(f'{x:.0f}' for x in (m.extents * 1000)),
            'mask_area': sel['candidates'][0]['mask_area'],
            'n_frames': n,
            'net': float(np.linalg.norm(C[-1] - C[0])),
            'path': float(step.sum()),
            'step_med': float(np.median(step)),
        })
    data.sort(key=lambda d: -d['reg_iou'])
    return data


def badge(iou):
    cls, txt = ('ok', '优') if iou >= 0.6 else (('mid', '中') if iou >= 0.3 else ('low', '差'))
    return f'<span class="tag {cls}">{txt} {iou:.3f}</span>'


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f6f7f9;color:#1f2429;font:15px/1.65 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:28px 16px 64px}
.wrap{max-width:1240px;margin:0 auto}
header h1{font-size:22px;margin-bottom:4px}
header .sub{color:#6b7280;font-size:13px;font-family:ui-monospace,Consolas,monospace}
.note{background:#fdf6ec;border-left:3px solid #b45309;color:#b45309;padding:12px 16px;
      border-radius:0 8px 8px 0;margin:18px 0;font-size:13.5px}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0;font-size:13px;color:#6b7280}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.card>img{width:100%;display:block;background:#000;aspect-ratio:688/384;object-fit:cover}
.body{padding:12px 14px;display:flex;flex-direction:column;gap:8px;flex:1}
.title{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.title b{font-size:15px;font-family:ui-monospace,Consolas,monospace}
.tag{font-size:11px;padding:1px 7px;border-radius:99px;background:#e7f3ec;color:#16703c}
.tag.mid{background:#fef3e2;color:#b45309}
.tag.low{background:#fdeaea;color:#b42318}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:3px 6px;border-bottom:1px solid #e4e7ec}
th{color:#6b7280;font-weight:600;font-size:11px}
td.num{font-family:ui-monospace,Consolas,monospace;text-align:right}
.links{display:flex;gap:8px;margin-top:auto;padding-top:6px;flex-wrap:wrap}
.links a{font-size:12px;color:#16703c;text-decoration:none;border:1px solid #e4e7ec;
         border-radius:6px;padding:4px 9px}
.links a:hover{background:#e7f3ec}
details{margin-top:6px}
summary{font-size:12px;color:#6b7280;cursor:pointer}
details img{width:100%;margin-top:6px;border:1px solid #e4e7ec;border-radius:8px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    args = ap.parse_args()
    sd = args.scene_dir
    data = collect(sd)

    cards = []
    for d in data:
        k = html.escape(d['key'])
        rows = [
            ('mesh 尺寸', f"{d['size_mm']} mm"),
            ('register IoU (帧0)', f"{d['reg_iou']:.3f}"),
            ('全帧 IoU 均值', f"{d['mean_iou']:.3f}"),
            ('GT mask', f"{d['mask_area']} px / {d['n_frames']} 帧"),
            ('FP 净位移', f"{d['net']:.3f} m"),
            ('FP 路径长度', f"{d['path']:.3f} m"),
            ('逐帧 step 中位', f"{d['step_med'] * 1000:.0f} mm"),
        ]
        tr = ''.join(f'<tr><th>{html.escape(a)}</th><td class="num">{html.escape(b)}</td></tr>'
                     for a, b in rows)
        cards.append(f"""<div class="card">
  <img src="{k}/small.gif" alt="{k}" loading="lazy" width="344" height="192">
  <div class="body">
    <div class="title"><b>{k}</b>{badge(d['reg_iou'])}</div>
    <table>{tr}</table>
    <details><summary>轨迹图</summary><img src="{k}/traj.png" alt="traj" loading="lazy"></details>
    <div class="links">
      <a href="{k}/overlay.gif" target="_blank">原始 GIF (4MB)</a>
      <a href="{k}/traj.png" target="_blank">轨迹图</a>
      <a href="../fp_out_{k}/iou_face_report.json" target="_blank">IoU JSON</a>
    </div>
  </div>
</div>""")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoundationPose 批量跟踪 — 7_vggt_omega</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
  <h1>FoundationPose 批量跟踪 · 7_vggt_omega</h1>
  <div class="sub">{len(data)} 个去重后实例 · mask 全部由 register IoU 自动匹配</div>
</header>
<div class="note">
  <b>怎么读这张图</b>：绿色轮廓 = FoundationPose 位姿下 mesh 的投影；
  <b>红色轮廓 = SAM3 真值 mask</b>。两者重合得好= 跟得准。<br>
  <b>结论</b>：register（首帧）IoU 0.40~0.97 都准，但 <b>track 阶段会在静止物体上累积抖动</b>
  （路径 3~7m，真实运动只有 0.002~0.34m）。多数 GIF 表现为「前几帧贴合、随后逐渐漂移」。
</div>
<div class="legend">
  <span><i class="dot" style="background:#00dc00"></i>FP 投影</span>
  <span><i class="dot" style="background:#ff0000"></i>SAM3 GT mask</span>
  <span><i class="dot" style="background:#ff8c00"></i>历史轨迹尾迹</span>
</div>
<div class="grid">
{''.join(cards)}
</div>
</div></body></html>"""

    out = os.path.join(sd, 'fp_vis', 'index.html')
    open(out, 'w').write(page)
    print(f'{len(data)} instances -> {out}')
    for d in data:
        print(f"  {d['key']:14s} reg={d['reg_iou']:.3f} mean={d['mean_iou']:.3f} "
              f"step_med={d['step_med'] * 1000:.0f}mm")


if __name__ == '__main__':
    main()
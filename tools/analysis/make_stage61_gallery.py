#!/usr/bin/env python3
"""Stage 6.1 物理仿真结果总览 HTML 画廊。

把 output_v2/ 下所有已有 stage61_physics.gif 的场景汇总成单文件 HTML，
可以在浏览器里一次性看到全部场景的物理仿真效果与关键指标。

为什么单独一个文件:
  之前 output_v2/egodex_gallery.html 也是这种"单文件 + 相对路径"的形式,
  双击即看、无需起服务器; 但它和其生成脚本都已从仓库里丢失, 本文恢复该做法。

Usage:
    python tools/analysis/make_stage61_gallery.py
    python tools/analysis/make_stage61_gallery.py --root output_v2 --out output_v2/stage61_gif_gallery.html
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime

SCENES_DEFAULT = [
    "2chair-bottle_rescaled", "3chair-bottle_rescaled", "phone_vggt_omega",
    "082_C5_CellPhone_rescaled", "121_C5_CellPhone_161deg_vggt_omega",
    "151_C7_Plate_214deg_vggt_omega", "7_vggt_omega", "egodex_0_rescaled",
]

CSS = """
:root{--bg:#0f1115;--card:#181b22;--line:#2a2f3a;--fg:#e6e8ee;--dim:#9aa3b2}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:16px}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}
.sum{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:9px 14px;min-width:110px}
.sum b{display:block;font-size:20px;line-height:1.3}
.sum span{color:var(--dim);font-size:12px}
.grid{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
overflow:hidden;display:flex;flex-direction:column}
.card a.img{display:block;background:#000}
/* 场景有横版(688x384)和竖版(384x688)两种, 用 contain + 固定高度,
   否则 cover 会把竖版裁掉大半 (只看得到中间一小块) */
.card img{width:100%;height:230px;object-fit:contain;display:block;cursor:zoom-in}
.info{padding:10px 12px 12px}
.name{font-weight:600;font-size:14px;margin-bottom:7px;word-break:break-all}
.tags{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:5px}
.tag{font-size:11px;padding:2px 7px;border-radius:20px;background:#222735;
color:var(--dim);border:1px solid var(--line)}
.tag.ok{background:#12351f;color:#7ee2a8;border-color:#1d5c36}
.tag.warn{background:#3a2f10;color:#e0c060;border-color:#6b551a}
.tag.info{background:#123243;color:#7ec8f5;border-color:#1d5470}
.tag.bad{background:#3a1518;color:#f08080;border-color:#6b2028}
.tag.mt{background:#1a2438;color:#a0b8e0}
.meta{color:var(--dim);font-size:12px;margin-top:4px}
.links{margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;font-size:12px}
.links a{color:var(--dim);text-decoration:none}
.links a:hover{color:#6ea8fe}
"""


def _n_links(root, scene):
    """URDF 里的物体数量 (不含 base)。"""
    p = os.path.join(root, scene, "urdf", "scene.urdf")
    if not os.path.exists(p):
        return None
    try:
        return len([l for l in ET.parse(p).getroot().findall(".//link")
                    if l.get("name") != "base"])
    except Exception:
        return None


def _load_report(root, scene):
    p = os.path.join(root, scene, "stage61_physics_report.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(root, scenes):
    rows = []
    for s in scenes:
        d = os.path.join(root, s)
        gif = os.path.join(d, "stage61_physics.gif")
        if not os.path.exists(gif):
            continue
        rep = _load_report(root, s) or {}
        sim = rep.get("sim", {})
        rc = rep.get("render_check", {})
        out = rep.get("outputs", {})
        ms = rep.get("materials_stage6") or {}
        dv = ms.get("density_vs_vlm") or {}
        rows.append(dict(
            scene=s, gif=os.path.join(s, "stage61_physics.gif"),
            mp4=os.path.join(s, "stage61_physics.mp4"),
            png=os.path.join(s, "stage61_physics_frame0.png"),
            report=os.path.join(s, "stage61_physics_report.json"),
            urdf=os.path.join(s, "urdf", "scene.urdf"),
            materials=(os.path.join(s, "materials.json")
                       if os.path.exists(os.path.join(d, "materials.json")) else None),
            backend=rep.get("render_backend", "sapien"),
            n_objects=sim.get("n_loaded") or _n_links(root, s),
            n_moved=sim.get("n_moved"),
            cover=rc.get("colored_pixel_ratio_median"),
            n_empty=rc.get("n_frames_nearly_empty"),
            n_frames=sim.get("n_frames"),
            n_cams=sim.get("n_real_cam_frames"),
            cam_source=sim.get("camera_source"),
            wh=(f"{out.get('img_w')}x{out.get('img_h')}" if out.get("img_w") else None),
            kb=os.path.getsize(gif) // 1024,
            mat_matched=ms.get("n_matched"),
            mat_total=ms.get("n_objects"),
            scale_err=dv.get("implied_linear_scale_error"),
        ))
    return rows


def render_summary(rows):
    n_obj = sum(r["n_objects"] or 0 for r in rows)
    n_mov = sum(r["n_moved"] or 0 for r in rows)
    items = [
        (str(len(rows)), "场景"),
        (str(n_obj), "物体"),
        (f"{n_mov}/{n_obj}", f"发生位移 ({n_mov * 100 // max(n_obj, 1)}%)"),
        (f"{sum(1 for r in rows if r['backend'] == 'pyrender_offscreen')}", "pyrender 方案 B"),
        (f"{sum(1 for r in rows if r['materials'])}", "有 Stage6 材质"),
    ]
    return "\n".join(
        f'<div class="sum"><b>{v}</b><span>{k}</span></div>' for v, k in items)


def render_card(r):
    cover = r.get("cover")
    cov_pct = None if cover is None else cover * 100
    if cov_pct is None:
        cov_tag = '<span class="tag bad">无自检数据</span>'
    elif cov_pct >= 20:
        cov_tag = f'<span class="tag ok">彩色 {cov_pct:.0f}%</span>'
    elif cov_pct >= 0.5:
        cov_tag = f'<span class="tag info">彩色 {cov_pct:.1f}%</span>'
    else:
        cov_tag = f'<span class="tag bad">彩色 {cov_pct:.2f}%</span>'

    empty_tag = ""
    if r.get("n_empty"):
        cls = "warn" if (r["n_empty"] or 0) < (r["n_frames"] or 1) * 0.5 else "bad"
        empty_tag = f'<span class="tag {cls}">空帧 {r["n_empty"]}/{r["n_frames"]}</span>'

    backend_tag = ('<span class="tag ok">pyrender</span>' if r["backend"] == "pyrender_offscreen"
                   else '<span class="tag info">SAPIEN</span>')
    mat_tag = ""
    if r.get("materials"):
        mat_tag = (f'<span class="tag mt">Stage6 材质 '
                   f'{r["mat_matched"]}/{r["mat_total"]}</span>')
        if r.get("scale_err"):
            mat_tag += f'<span class="tag warn">尺度误差≈{r["scale_err"]}x</span>'
    else:
        mat_tag = '<span class="tag warn">无 Stage6 材质</span>'

    links = [
        (r["gif"], "▶ GIF"), (r["mp4"], "🎬 MP4"),
        (r["png"], "🖼 首帧"), (r["report"], "📄 报告"), (r["urdf"], "🔧 URDF"),
    ]
    if r.get("materials"):
        links.append((r["materials"], "⚙ 材质"))
    links_html = "\n      ".join(
        f'<a href="{p}" target="_blank">{t}</a>' for p, t in links)

    meta_note = "" if r.get("scale_err") is None else \
        f" · 隐含线性尺度误差 ≈ {r['scale_err']}x"

    return f"""<div class="card">
  <a class="img" href="{r['gif']}" target="_blank"><img src="{r['gif']}" loading="lazy"></a>
  <div class="info">
    <div class="name">{r['scene']}</div>
    <div class="tags">{backend_tag}<span class="tag">{(r['n_objects'] or 0)} 物体</span>\
<span class="tag warn">位移 {(r['n_moved'] if r['n_moved'] is not None else 0)}</span>{cov_tag}{empty_tag}</div>
    <div class="tags">{mat_tag}<span class="tag">{r['cam_source']} · {r['n_cams']} 相机帧</span>\
<span class="tag">{r['wh']}</span></div>
    <div class="meta">{r['kb']} KB · {r['n_frames']} 帧 @30fps{meta_note}</div>
    <div class="links">{links_html}</div>
  </div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output_v2")
    ap.add_argument("--out", default="output_v2/stage61_gif_gallery.html")
    ap.add_argument("--scene", nargs="*", default=None,
                    help="只列这些场景; 不给则扫描 --root 下全部有 GIF 的目录")
    args = ap.parse_args()

    if args.scene:
        scenes = args.scene
    else:
        scenes = sorted(os.listdir(args.root)) if os.path.isdir(args.root) else []
    rows = collect(args.root, scenes)
    # 没有 GIF 的标准场景也列出来, 标红, 便于一眼看到漏了谁
    missing = [s for s in SCENES_DEFAULT
               if s not in {r["scene"] for r in rows}]
    if not rows:
        print("[ERROR] 没有任何场景有 stage61_physics.gif")
        return 1

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Stage 6.1 物理仿真总览 ({len(rows)} 个场景)</title><style>{CSS}</style></head><body>
<h1>Stage 6.1 物理仿真总览</h1>
<div class="sub">{len(rows)} 个场景 · 输出目录 {args.root}/ · 点击 GIF 在新标签打开看大图 · \
生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="summary">{render_summary(rows)}</div>
<div class="grid">{''.join(render_card(r) for r in rows)}</div>
</body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已写出 {args.out} ({len(html) // 1024} KB, {len(rows)} 个场景)")
    if missing:
        print("  缺 GIF 的场景: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
make_viewer_html.py - 生成自包含交互式 HTML 查看器 (WebGL, 无外部依赖)
====================================================================

把 "场景点云 + EgoFound3R 手部 mesh" (同一坐标系、同一比例尺) 打包成一个
HTML 文件, 用浏览器直接打开即可 360 度旋转/缩放查看。数据以 base64 内嵌,
不依赖网络/本地文件, file:// 直接打开可用。

两种 gauge:
  --gauge ego  : EgoFound3R 原生米制 (零标定, 推荐先看)
  --gauge ras  : RAS raw gauge (手部 x g, 用于和 RAS 物体对比)
  --gauge both : 两个都生成

用法:
  cd ReplicateAnyScene
  /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/make_viewer_html.py \
      --scene-dir output_v3/egofound_verify/weixin_195833_224_h264 --gauge both
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO_ROOT = REPO_ROOT.parent / "EgoFound3R"
SIDE_COLORS_RGB = [(255, 165, 0), (0, 128, 255)]


def tnp(t):
    return t.detach().cpu().float().numpy()


def load_marker_faces(ego_root: Path):
    import importlib.util
    p = ego_root / "egohandmetric_prompt" / "data" / "marker_mesh.py"
    spec = importlib.util.spec_from_file_location("ego_marker_mesh", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.marker_faces_for_count(195).numpy()


def hand_payload(vw, faces, side, ok):
    """世界系手部顶点 -> (顶点, 法线, 颜色, 索引) 或 None."""
    import trimesh
    remap = -np.ones(len(vw), int)
    remap[np.nonzero(ok)[0]] = np.arange(int(ok.sum()))
    kept = [(remap[a], remap[b], remap[c]) for a, b, c in faces
            if ok[a] and ok[b] and ok[c]]
    if len(kept) < 20:
        return None
    mesh = trimesh.Trimesh(vertices=vw[ok].astype(np.float64),
                           faces=np.array(kept), process=False)
    n = mesh.vertex_normals
    color = np.tile(np.array(SIDE_COLORS_RGB[side], np.uint8), (len(mesh.vertices), 1))
    return {
        "pos": mesh.vertices.astype("<f4").tobytes(),
        "nrm": n.astype("<f4").tobytes(),
        "col": color.tobytes(),
        "idx": np.array(kept, dtype="<u2").tobytes(),
        "n_idx": len(kept) * 3,
    }


def b64(b):
    return base64.b64encode(b).decode("ascii")


# ---------------------------------------------------------------------------
def gather_ego(scene_dir, max_points, frames, ego_root):
    """EgoFound3R 原生米制: 官方反投影 + 官方 w2c_to_c2w, 零标定."""
    import cv2
    import torch
    sys.path.insert(0, str(ego_root))
    from egohandmetric_prompt.metric_geometry import unproject_depth_to_camera_points
    from egohandmetric_prompt.models.camera_temporal_refiner import w2c_to_c2w

    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    d_met = payload["depth_metric_global"]
    d_met = tnp(d_met[..., 0]) if d_met.ndim == 4 else tnp(d_met)
    conf = payload["depth_confidence_global"]
    conf = tnp(conf[..., 0]) if conf.ndim == 4 else tnp(conf)
    K_g = tnp(payload["intrinsics_global"])
    cam_g = tnp(payload["camera_pose_metric_global"])
    cam_full = tnp(payload["camera_pose_metric_full"])
    H, W = payload["image_size_hw"]
    g_idx = payload["global_frame_indices"].tolist()

    # RGB 帧 (来自 sampled.mp4, 缩放到模型输入尺寸)
    rgb = None
    vp = scene_dir / "sampled.mp4"
    if vp.exists():
        cap = cv2.VideoCapture(str(vp))
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if f.shape[:2] != (H, W):
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            fr.append(f)
        cap.release()
        if len(fr) == payload["T_R"]:
            rgb = fr

    pts, cols = [], []
    for j in range(len(g_idx)):
        d, c, K, C = d_met[j], conf[j], K_g[j], cam_g[j]
        valid = np.isfinite(d) & (d > 1e-3) & np.isfinite(c)
        if valid.sum() > 100:
            valid &= (c >= np.percentile(c[valid], 30))
        sel = np.zeros_like(valid); sel[::3, ::3] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        P_cam = unproject_depth_to_camera_points(
            torch.as_tensor(d, dtype=torch.float32)[None, None],
            torch.as_tensor(K, dtype=torch.float32)[None, None])[0, 0].numpy()[ys, xs]
        c2w = w2c_to_c2w(torch.as_tensor(C, dtype=torch.float32)[None]).numpy()[0]
        Pw = P_cam @ c2w[:3, :3].T + c2w[:3, 3]
        pts.append(Pw.astype(np.float32))
        if rgb is not None:
            cols.append(rgb[g_idx[j]][ys, xs])
        else:
            z = d[ys, xs]
            zn = np.clip((z - np.percentile(z, 2)) /
                         max(np.percentile(z, 98) - np.percentile(z, 2), 1e-9), 0, 1)
            cols.append(cv2.applyColorMap((zn * 255).astype(np.uint8),
                                          cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1])
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > max_points:
        idx = np.random.choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]

    faces = load_marker_faces(ego_root)
    verts = tnp(payload["hand"]["vertex_xyz_metric"])
    valid = tnp(payload["hand"]["valid"]).astype(bool)
    hands = []
    for f in frames:
        for s in range(2):
            if not valid[f, s]:
                continue
            v = verts[f, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            c2w = w2c_to_c2w(torch.as_tensor(cam_full[f], dtype=torch.float32)[None]).numpy()[0]
            vw = v @ c2w[:3, :3].T + c2w[:3, 3]
            hp = hand_payload(vw, faces, s, ok)
            if hp:
                hp["name"] = f"frame{f}_side{s}"
                hp["size"] = float(np.linalg.norm(np.ptp(vw, axis=0)))
                hands.append(hp)
    meta = {
        "gauge": "EgoFound3R 原生米制 (零标定)",
        "unit": "米 (m)",
        "world": "ego stitched_video_world",
        "scene_bbox_diag": float(np.linalg.norm(np.ptp(P, axis=0))),
    }
    return P, C, hands, meta


def gather_ras(scene_dir, g_value, max_points, frames, ego_root):
    """RAS raw gauge: 本地 world_points + 手部 x g (本地 extrinsics 逆变换)."""
    import torch
    data = np.load(scene_dir / "local_vggt" / "local_pred.npz", mmap_mode="r")
    extr = np.asarray(data["extrinsics"])
    wp = data["world_points"]; colors = data["colors"]; depths = data["depths"]
    N = wp.shape[0]
    pts, cols = [], []
    for i in range(N):
        d = np.asarray(depths[i])
        valid = np.isfinite(d) & (d > 1e-6)
        sel = np.zeros_like(valid); sel[::4, ::4] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        p = np.asarray(wp[i])[ys, xs]
        ok = np.isfinite(p).all(axis=1)
        pts.append(p[ok].astype(np.float32))
        cols.append(np.asarray(colors[i])[ys, xs][ok].astype(np.uint8))
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > max_points:
        idx = np.random.choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]

    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    faces = load_marker_faces(ego_root)
    verts = tnp(payload["hand"]["vertex_xyz_metric"])
    valid = tnp(payload["hand"]["valid"]).astype(bool)
    hands = []
    for f in frames:
        for s in range(2):
            if not valid[f, s]:
                continue
            v = verts[f, s] * g_value
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            R, t = extr[f, :3, :3], extr[f, :3, 3]
            vw = (R.T @ (v - t).T).T
            hp = hand_payload(vw, faces, s, ok)
            if hp:
                hp["name"] = f"frame{f}_side{s}"
                hp["size"] = float(np.linalg.norm(np.ptp(vw, axis=0)))
                hands.append(hp)
    meta = {
        "gauge": f"RAS raw gauge (手部 = metric x g={g_value:.3f})",
        "unit": "RAS raw 单位 (非米制, 约 1.93x 米制)",
        "world": "本地 VGGT-Omega 世界系",
        "scene_bbox_diag": float(np.linalg.norm(np.ptp(P, axis=0))),
    }
    return P, C, hands, meta


# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#111;overflow:hidden;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#ddd}
  #gl{width:100%;height:100%;display:block;cursor:grab}
  #gl:active{cursor:grabbing}
  .panel{position:absolute;background:rgba(20,20,24,.82);border:1px solid #333;border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.7;backdrop-filter:blur(4px)}
  #info{left:12px;top:12px;max-width:330px}
  #ctrl{left:12px;bottom:12px}
  #info b{color:#fff}
  #info .k{color:#8ab4f8}
  button{background:#2a2a33;color:#ddd;border:1px solid #444;border-radius:5px;padding:4px 10px;margin:2px 4px 2px 0;cursor:pointer;font-size:12px}
  button:hover{background:#3a3a46}
  button.off{opacity:.45}
  input[type=range]{vertical-align:middle;width:110px}
  label{user-select:none}
  .handbtn.on{border-color:#e6a23c;color:#ffd591}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="info" class="panel"></div>
<div id="ctrl" class="panel">
  <button id="btnRot">暂停旋转</button>
  <button id="btnReset">重置视角</button>
  <button id="btnFit">适配大小</button><br>
  点大小 <input id="psize" type="range" min="1" max="8" step="0.5" value="2.5"><br>
  <span id="hands"></span><br>
  <span style="color:#888">拖拽旋转 · 滚轮缩放 · 右键拖拽平移</span>
</div>
<script>
"use strict";
const POS = new Float32Array(b64buf("__POS__"));
const COL = new Uint8Array(b64buf("__COL__"));
const HANDS = JSON.parse('__HANDS_JSON__').map(h => ({
  name: h.name, size: h.size, n_idx: h.n_idx,
  pos: new Float32Array(b64buf(h.pos)),
  nrm: new Float32Array(b64buf(h.nrm)),
  col: new Uint8Array(b64buf(h.col)),
  idx: new Uint16Array(b64buf(h.idx)),
}));
const META = __META_JSON__;
function b64buf(s){
  const bin = atob(s), n = bin.length, out = new ArrayBuffer(n), u8 = new Uint8Array(out);
  for (let i=0;i<n;i++) u8[i]=bin.charCodeAt(i);
  return out;
}
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", {antialias:true, alpha:false});
if(!gl){ document.body.innerHTML = "<p style='padding:20px'>当前浏览器不支持 WebGL</p>"; throw new Error("no webgl"); }

function sh(type, src){
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function prog(vs, fs){
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const pPoints = prog(`
  attribute vec3 aPos; attribute vec3 aCol;
  uniform mat4 uMVP; uniform float uPS;
  varying vec3 vCol;
  void main(){ vCol = aCol; gl_Position = uMVP * vec4(aPos,1.0); gl_PointSize = uPS; }`,
  `precision mediump float; varying vec3 vCol;
  void main(){ vec2 c = gl_PointCoord - 0.5; if(dot(c,c) > 0.25) discard;
  gl_FragColor = vec4(vCol, 1.0); }`);
const pMesh = prog(`
  attribute vec3 aPos; attribute vec3 aNrm; attribute vec3 aCol;
  uniform mat4 uMVP; uniform mat3 uNrm;
  varying vec3 vCol; varying vec3 vL;
  void main(){ vCol = aCol;
    vec3 L = normalize(vec3(0.45, -0.75, 0.55));   // 光从上方前方 (世界 y 向下)
    float d = max(dot(normalize(uNrm * aNrm), L), 0.0);
    vL = vec3(0.30 + 0.70*d);
    gl_Position = uMVP * vec4(aPos, 1.0); }`,
  `precision mediump float; varying vec3 vCol; varying vec3 vL;
  void main(){ gl_FragColor = vec4(vCol * vL, 1.0); }`);

const bufPos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bufPos); gl.bufferData(gl.ARRAY_BUFFER, POS, gl.STATIC_DRAW);
const bufCol = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bufCol); gl.bufferData(gl.ARRAY_BUFFER, COL, gl.STATIC_DRAW);
const nPts = POS.length / 3;

// 手部 buffer 只创建一次 (每帧创建/删除会令顶点属性指针失效, 导致点云 drawArrays 报 INVALID_OPERATION)
const locMeshPos = gl.getAttribLocation(pMesh, "aPos");
const locMeshNrm = gl.getAttribLocation(pMesh, "aNrm");
const locMeshCol = gl.getAttribLocation(pMesh, "aCol");
const locPtPos = gl.getAttribLocation(pPoints, "aPos");
const locPtCol = gl.getAttribLocation(pPoints, "aCol");
HANDS.forEach(h => {
  h.bp = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, h.bp); gl.bufferData(gl.ARRAY_BUFFER, h.pos, gl.STATIC_DRAW);
  h.bn = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, h.bn); gl.bufferData(gl.ARRAY_BUFFER, h.nrm, gl.STATIC_DRAW);
  h.bc = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, h.bc); gl.bufferData(gl.ARRAY_BUFFER, h.col, gl.STATIC_DRAW);
  h.bi = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, h.bi); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, h.idx, gl.STATIC_DRAW);
});

// 场景范围
let mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
for(let i=0;i<nPts;i++) for(let k=0;k<3;k++){ mn[k]=Math.min(mn[k],POS[i*3+k]); mx[k]=Math.max(mx[k],POS[i*3+k]); }
const center = [0,1,2].map(k => (mn[k]+mx[k])/2);
const radius = Math.max(1e-6, 0.5*Math.sqrt([0,1,2].reduce((s,k)=>s+(mx[k]-mn[k])**2,0)));
HANDS.forEach(h => { for(let i=0;i<h.pos.length;i++) for(let k=0;k<3;k++){ mn[k]=Math.min(mn[k],h.pos[i*3+k]); mx[k]=Math.max(mx[k],h.pos[i*3+k]); } });

// 相机 (yaw/pitch/dist, up = -Y 因为世界系 y 向下)
let yaw = 0.9, pitch = 0.35, dist = radius * 2.2, target = center.slice();
let autoRot = true, pointSize = 2.5;
const handOn = {}; HANDS.forEach(h => handOn[h.name] = true);

function perspective(fovy, aspect, near, far){
  const f = 1/Math.tan(fovy/2), nf = 1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}
function lookAt(eye, ctr, up){
  const z = norm(sub(eye,ctr)), x = norm(cross(up,z)), y = cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
                           -dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
function mul(a,b){ // column-major 4x4
  const o = new Float32Array(16);
  for(let j=0;j<4;j++) for(let i=0;i<4;i++){ let s=0; for(let k=0;k<4;k++) s += a[i+4*k]*b[k+4*j]; o[i+4*j]=s; }
  return o;
}
function mat3from4(m){ return new Float32Array([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]]); }

function resize(){
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  canvas.width = canvas.clientWidth*dpr; canvas.height = canvas.clientHeight*dpr;
  gl.viewport(0,0,canvas.width,canvas.height);
}
window.addEventListener("resize", resize);

// 交互
let drag=null, rmb=false;
canvas.addEventListener("pointerdown", e => { drag={x:e.clientX,y:e.clientY, rmb:(e.button===2)}; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointerup", e => { drag=null; });
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointermove", e => {
  if(!drag) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if(drag.rmb){ // 平移
    const cp = Math.cos(pitch), sp = Math.sin(pitch), sy = Math.sin(yaw), cy = Math.cos(yaw);
    const right = [cy,0,-sy], fwd = [sy*cp, sp, cy*cp];
    const s = dist*0.0016;
    for(let k=0;k<3;k++) target[k] += (-dx*right[k] + dy*fwd[k])*s;
  } else {
    yaw += dx*0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy*0.008));
  }
});
canvas.addEventListener("wheel", e => { e.preventDefault(); dist *= Math.exp(e.deltaY*0.0012); dist = Math.max(radius*0.05, Math.min(radius*30, dist)); }, {passive:false});

// UI
const $ = id => document.getElementById(id);
$("btnRot").onclick = () => { autoRot = !autoRot; $("btnRot").textContent = autoRot ? "暂停旋转" : "开始旋转"; $("btnRot").classList.toggle("off", !autoRot); };
$("btnReset").onclick = () => { yaw=0.9; pitch=0.35; dist=radius*2.2; target=center.slice(); };
$("btnFit").onclick = () => { dist = radius*2.2; target = center.slice(); };
$("psize").oninput = e => { pointSize = parseFloat(e.target.value); };
const handBox = $("hands");
HANDS.forEach(h => {
  const b = document.createElement("button");
  b.className = "handbtn on"; b.textContent = h.name + " (" + h.size.toFixed(2) + ")";
  b.onclick = () => { handOn[h.name] = !handOn[h.name]; b.classList.toggle("on", handOn[h.name]); };
  handBox.appendChild(b);
});

$("info").innerHTML =
  "<b>" + META.title + "</b><br>" +
  "<span class='k'>gauge:</span> " + META.gauge + "<br>" +
  "<span class='k'>单位:</span> " + META.unit + "<br>" +
  "<span class='k'>世界系:</span> " + META.world + "<br>" +
  "<span class='k'>点云:</span> " + META.n_points.toLocaleString() + " 点<br>" +
  "<span class='k'>场景尺寸(bbox对角):</span> " + META.scene_bbox_diag.toFixed(3) + "<br>" +
  "<span class='k'>手部:</span> " + HANDS.map(h => h.name + "=" + h.size.toFixed(2)).join(", ") + "<br>" +
  "<span style='color:#8f8'>橙=左手(side0) · 蓝=右手(side1) · 点云按原始 RGB 上色</span>";

// 渲染循环
function frame(){
  if(autoRot) yaw += 0.0035;
  resize();
  gl.clearColor(0.07,0.07,0.09,1); gl.clearDepth(1);
  gl.enable(gl.DEPTH_TEST); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const cp = Math.cos(pitch), sp = Math.sin(pitch), sy = Math.sin(yaw), cy = Math.cos(yaw);
  const eye = [target[0]+dist*cp*sy, target[1]+dist*sp, target[2]+dist*cp*cy];
  const V = lookAt(eye, target, [0,-1,0]);
  const P = perspective(Math.PI/4, canvas.width/Math.max(canvas.height,1), radius*0.01, radius*40);
  const MVP = mul(P, V), NM = mat3from4(V);

  gl.useProgram(pPoints);
  gl.uniformMatrix4fv(gl.getUniformLocation(pPoints,"uMVP"), false, MVP);
  gl.uniform1f(gl.getUniformLocation(pPoints,"uPS"), pointSize * (window.devicePixelRatio>1?1.5:1));
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
  gl.enableVertexAttribArray(locPtPos); gl.vertexAttribPointer(locPtPos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufCol);
  gl.enableVertexAttribArray(locPtCol); gl.vertexAttribPointer(locPtCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.POINTS, 0, nPts);

  gl.useProgram(pMesh);
  gl.uniformMatrix4fv(gl.getUniformLocation(pMesh,"uMVP"), false, MVP);
  gl.uniformMatrix3fv(gl.getUniformLocation(pMesh,"uNrm"), false, NM);
  HANDS.forEach(h => {
    if(!handOn[h.name]) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, h.bp);
    gl.enableVertexAttribArray(locMeshPos); gl.vertexAttribPointer(locMeshPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, h.bn);
    gl.enableVertexAttribArray(locMeshNrm); gl.vertexAttribPointer(locMeshNrm, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, h.bc);
    gl.enableVertexAttribArray(locMeshCol); gl.vertexAttribPointer(locMeshCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, h.bi);
    gl.drawElements(gl.TRIANGLES, h.n_idx, gl.UNSIGNED_SHORT, 0);
  });
  requestAnimationFrame(frame);
}
resize();
frame();
</script>
</body>
</html>
"""


def write_viewer(out_path: Path, P, C, hands, meta, title):
    payload = {
        "title": title,
        "gauge": meta["gauge"], "unit": meta["unit"], "world": meta["world"],
        "n_points": int(len(P)),
        "scene_bbox_diag": meta["scene_bbox_diag"],
    }
    hands_json = []
    for h in hands:
        hands_json.append({
            "name": h["name"], "size": h["size"], "n_idx": h["n_idx"],
            "pos": b64(h["pos"]), "nrm": b64(h["nrm"]),
            "col": b64(h["col"]), "idx": b64(h["idx"]),
        })
    html = (HTML
            .replace("__TITLE__", title)
            .replace("__POS__", b64(P.astype("<f4").tobytes()))
            .replace("__COL__", b64(C.astype(np.uint8).tobytes()))
            .replace("__HANDS_JSON__", json.dumps(hands_json))
            .replace("__META_JSON__", json.dumps(payload)))
    out_path.write_text(html, encoding="utf-8")
    print(f"[HTML] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, "
          f"{len(P)} 点, {len(hands)} 只手)")


def main():
    ap = argparse.ArgumentParser(description="生成自包含交互式 HTML 查看器")
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=DEFAULT_EGO_ROOT)
    ap.add_argument("--gauge", choices=["ego", "ras", "both"], default="both")
    ap.add_argument("--max-points", type=int, default=150_000)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    args = ap.parse_args()

    import torch
    scene_dir = args.scene_dir
    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    T = payload["hand"]["valid"].shape[0]
    frames = args.frames or sorted(set([int(np.argmax(payload["hand"]["valid"].numpy().sum(axis=1))),
                                        T // 2, T - 1]))
    print(f"[HTML] 手部帧: {frames}")

    if args.gauge in ("ego", "both"):
        P, C, hands, meta = gather_ego(scene_dir, args.max_points, frames, args.ego_root)
        write_viewer(scene_dir / "viewer_ego_metric.html", P, C, hands, meta,
                     "EgoFound3R 手部 x 场景点云 (米制, 零标定)")

    if args.gauge in ("ras", "both"):
        m = json.loads((scene_dir / "verify" / "metrics.json").read_text())
        g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
            (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
        P, C, hands, meta = gather_ras(scene_dir, g, args.max_points, frames, args.ego_root)
        write_viewer(scene_dir / "viewer_ras_gauge.html", P, C, hands, meta,
                     "EgoFound3R 手部 x RAS 点云 (raw gauge, 手 x g)")


if __name__ == "__main__":
    main()

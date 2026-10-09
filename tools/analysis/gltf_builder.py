"""
最小 glTF 2.0 GLB 写入器(带 transform 动画)。

为什么不用 trimesh.animation: 本环境 trimesh 4.11.5 **没有** animation 模块
(`pkgutil.iter_modules` 里无 `animation`), `Scene.export` 也不支持动画导出。
这里直接按 glTF 2.0 规范拼 JSON + 二进制 buffer, 生成可播放动画的 GLB。

坐标约定: glTF 是 Y-up 右手系, 而轨迹是 Z-up。导出时用 T_zup_to_yup 把所有
位姿整体旋转(-90°绕 X), 与 tools/stage61_gif.py:231-235 的 `_zup_to_yup()` 一致,
保证 Blender / MeshLab 打开后地平面水平、物体姿态正确。
"""

import json
import struct

import numpy as np


T_ZUP_TO_YUP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _pad_to_4(n):
    return (4 - n % 4) % 4


class _BufferBuilder:
    """按 4 字节对齐累积 binary chunk。"""

    def __init__(self):
        self.blobs = []
        self.length = 0

    def add(self, array, dtype):
        arr = np.ascontiguousarray(array, dtype=dtype)
        data = arr.tobytes()
        pad = _pad_to_4(len(data))
        if pad:
            data += b'\x00' * pad  # 补零保证下一个 bufferView 仍 4 字节对齐
        offset = self.length
        self.blobs.append(data)
        self.length += len(data)
        return offset, len(arr)

    def view(self):
        return b''.join(self.blobs)


def _accessor(buf, array, dtype, gltf_type, target=None, name=None):
    """把 numpy 数组写成 glTF accessor + bufferView。"""
    np_dt = np.dtype(dtype)
    array = np.ascontiguousarray(array, dtype=np_dt)
    ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[gltf_type]
    count = array.size // ncomp
    offset, real = buf.add(array.reshape(count, ncomp), np_dt)

    comp = {np.float32: 5126, np.uint32: 5125, np.uint16: 5123}[np_dt.type]  # componentType
    acc = {
        'componentType': comp,
        'count': int(real),
        'type': gltf_type,
    }
    if np_dt.type == np.float32:
        acc['min'] = array.reshape(count, ncomp).min(axis=0).tolist()
        acc['max'] = array.reshape(count, ncomp).max(axis=0).tolist()
    acc['_offset'] = offset
    acc['_bytes'] = array.nbytes
    if name:
        acc['name'] = name
    return acc


def export_glb_with_animation(out_path, meshes, animations,
                              fps=30, up_conversion=T_ZUP_TO_YUP):
    """导出带 transform 动画的 GLB。

    Args:
        meshes: {node_name: trimesh.Trimesh}  全部几何(顶点须已烘焙到世界系)
        animations: {node_name: (N,4,4) ndarray} 每个节点的关键帧位姿序列(世界系)
        fps: 帧率
        up_conversion: 世界系 → glTF 的旋转(Z-up → Y-up)

    Returns:
        dict: glTF JSON
    """
    conv = np.asarray(up_conversion, dtype=np.float64)

    buf = _BufferBuilder()
    gltf = {
        'asset': {'version': '2.0', 'generator': 'export_dynamic_glb.py'},
        'scene': 0,
        'scenes': [{'nodes': []}],
        'nodes': [],
        'meshes': [],
        'accessors': [],
        'bufferViews': [],
        'buffers': [],
    }

    def add_buffer_view(offset, length, target=None):
        bv = {'buffer': 0, 'byteOffset': int(offset), 'byteLength': int(length)}
        if target is not None:
            bv['target'] = target
        gltf['bufferViews'].append(bv)
        return len(gltf['bufferViews']) - 1


    for name, mesh in meshes.items():
        v = np.asarray(mesh.vertices, dtype=np.float32)

        v = (v.astype(np.float64) @ conv[:3, :3].T + conv[:3, 3]).astype(np.float32)

        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1, 3)
        vc = _accessor(buf, v, np.float32, 'VEC3', name=f'{name}_pos')
        ic = _accessor(buf, faces, np.uint32, 'VEC3', name=f'{name}_idx')
        vb = add_buffer_view(vc.pop('_offset'), vc.pop('_bytes'), 34962)  # ARRAY_BUFFER
        ib = add_buffer_view(ic.pop('_offset'), ic.pop('_bytes'), 34963)  # ELEMENT_ARRAY
        vc['bufferView'] = vb
        ic['bufferView'] = ib
        gltf['accessors'] += [vc, ic]

        prim = {'attributes': {'POSITION': len(gltf['accessors']) - 2},
                'indices': len(gltf['accessors']) - 1,
                'mode': 4}  # 4 = TRIANGLES


        colors = None
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(mesh.vertices):
                rgb = np.asarray(vc, dtype=np.float64)[:, :3] / 255.0  # uint8 → [0,1]
                rgb = rgb @ conv[:3, :3].T
                colors = np.concatenate(
                    [rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)
        except Exception:
            colors = None

        if colors is not None:
            cc = _accessor(buf, colors, np.float32, 'VEC4', name=f'{name}_col')
            cb = add_buffer_view(cc.pop('_offset'), cc.pop('_bytes'), 34962)
            cc['bufferView'] = cb
            gltf['accessors'].append(cc)
            prim['attributes']['COLOR_0'] = len(gltf['accessors']) - 1

        gltf['meshes'].append({'name': name, 'primitives': [prim]})
        gltf['nodes'].append(
            {'name': name, 'mesh': len(gltf['meshes']) - 1})
        gltf['scenes'][0]['nodes'].append(len(gltf['nodes']) - 1)

    node_index = {n['name']: i for i, n in enumerate(gltf['nodes'])}


    if animations:
        anims = []
        for name, traj in animations.items():
            if name not in node_index or traj is None or len(traj) == 0:
                continue
            n = len(traj)

            times = (np.arange(n, dtype=np.float32) / float(fps))

            mats = np.stack([conv @ np.asarray(m, dtype=np.float64)
                             for m in traj]).astype(np.float32)
            mats = mats.reshape(n, 16)  # MAT4 accessor 要 16 列

            tc = _accessor(buf, times, np.float32, 'SCALAR', name=f'{name}_t')
            mc = _accessor(buf, mats, np.float32, 'MAT4', name=f'{name}_m')
            tb = add_buffer_view(tc.pop('_offset'), tc.pop('_bytes'))
            mb = add_buffer_view(mc.pop('_offset'), mc.pop('_bytes'))
            tc['bufferView'] = tb
            mc['bufferView'] = mb
            gltf['accessors'] += [tc, mc]

            base = len(gltf['accessors']) - 2
            anims.append({
                'name': f'{name}_anim',
                'samplers': [{'input': base, 'output': base + 1,
                              'interpolation': 'LINEAR'}],
                'channels': [{'sampler': 0,
                              'target': {'node': node_index[name],
                                         'path': 'matrix'}}],
            })
        if anims:
            gltf['animations'] = anims

            gltf['extras'] = {'animation_hint': 'press Space in Blender'}

    binary = buf.view()
    while len(binary) % 4:  # BIN chunk 长度也必须 4 字节对齐
        binary += b'\x00'
    gltf['buffers'] = [{'byteLength': len(binary)}]

    js = json.dumps(gltf, separators=(',', ':')).encode('utf-8')  # 压缩 JSON
    js += b' ' * _pad_to_4(len(js))  # JSON chunk 用空格补齐(允许 trailing space)


    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack('<II', len(binary), 0x004E4942))
        f.write(binary)

    return gltf
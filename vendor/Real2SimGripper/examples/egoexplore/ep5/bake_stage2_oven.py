"""Bake ep5 stage2-final oven collision: extract the oven_door / oven_cavity and
rack/countertop nodes from the stage2 final GLB into world-frame OBJs.

The stage2 gravity frame differs from the old execution frame, so the old
replay-oven-agile colliders do not transfer; this re-bakes them from the final
scene GLB itself.  GLB node names map to meshes by index (the exporter renames
meshes to geometry_N).
"""
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

ASSETS = Path(__file__).resolve().parent / 'ras-mainv3-gravity-luna-xhigh.20261116/scene_assets_stage2_final'
OUT = Path(__file__).resolve().parent / 'runs/stage2-oven-collision/scene_assets'
NODES = ('oven_door', 'oven_cavity', 'oven_rack', 'black_countertop')


def glb_node_mesh_names(path):
    with Path(path).open('rb') as stream:
        stream.read(12)
        clen, ctype = struct.unpack('<II', stream.read(8))
        doc = json.loads(stream.read(clen))
    return {node['name']: doc['meshes'][node['mesh']]['name']
            for node in doc.get('nodes', []) if 'mesh' in node and 'name' in node}


def main():
    glb_path = ASSETS / 'final_scene_world.glb'
    glb = trimesh.load(glb_path)
    mapping = glb_node_mesh_names(glb_path)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source_glb': str(glb_path.resolve()), 'nodes': {}}
    for name in NODES:
        mesh_name = mapping.get(name)
        geom = glb.geometry.get(mesh_name) if mesh_name else None
        if geom is None:
            raise ValueError(f'node {name} ({mesh_name}) missing from the stage2 final GLB: '
                             f'{sorted(glb.geometry)}')
        mesh = trimesh.Trimesh(vertices=np.asarray(geom.vertices, float),
                               faces=np.asarray(geom.faces, int), process=False)
        obj = OUT / f'{name}.obj'
        mesh.export(obj)
        manifest['nodes'][name] = {
            'obj': obj.name, 'mesh': mesh_name, 'vertices': len(mesh.vertices),
            'triangles': len(mesh.faces),
            'bounds_min': mesh.bounds[0].tolist(), 'bounds_max': mesh.bounds[1].tolist()}
    (OUT / 'oven_collision_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

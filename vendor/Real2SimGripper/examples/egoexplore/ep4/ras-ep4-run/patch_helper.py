p = "/path/to/Real2SimGripper/examples/egoexplore/ep4/ras-ep4-run/helpers/run_cached_ras.py"
s = open(p).read()
old = """    supports = {k:[n for n in nodes if n.startswith(k+'_')] for k in ('table','chair')}
    if not all(supports.values()):
        raise ValueError(f'Table and chair required, GLB nodes: {nodes}')"""
new = """    supports = {k: [n for n in nodes if n.startswith(k + '_')] for k in ('table', 'chair')}
    if not supports['table']:
        raise ValueError(f'Table required, GLB nodes: {nodes}')
    supports['chair'] = supports['chair'] or []"""
assert old in s, "exact block not found"
open(p, "w").write(s.replace(old, new))
print("helper patched")

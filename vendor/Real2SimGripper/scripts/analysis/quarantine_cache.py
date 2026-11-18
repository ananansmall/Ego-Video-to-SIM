"""One-time cleanup of the ten identified sweep-import records; preserve exact backup."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import agentic_optimize as AO


def main():
    path = ROOT / 'results/agentic_optimize_cache.json'
    out = ROOT / 'results/ik_agent_20261022'
    backup = out / 'cache_before_cleanup.json'
    if backup.exists():
        raise FileExistsError(backup)
    raw = path.read_bytes()
    cache = json.loads(raw)
    removed = {k:v for k,v in cache.items()
               if v.get('tag', '').startswith(('basex-', 'base2d-'))
               and v.get('metrics', {}).get('knock_mm') == 0.0}
    assert len(removed) == 10, f'Expected identified set of ten records, found {len(removed)}'
    shutil.copyfile(path, backup)
    (out / 'synthetic_quarantine.json').write_text(json.dumps(removed, indent=2))
    cleaned = {k:v for k,v in cache.items() if k not in removed}
    for v in cleaned.values():
        v['score'] = AO.score(v['metrics'])
    assert path.read_bytes() == raw, 'Cache changed during cleanup'
    temp = path.with_suffix('.cleanup.tmp')
    temp.write_text(json.dumps(cleaned, ensure_ascii=False, indent=1))
    temp.replace(path)
    audit = dict(before=len(cache), removed=len(removed), after=len(cleaned),
                 backup_sha256=hashlib.sha256(raw).hexdigest(),
                 removed_tags=[v['tag'] for v in removed.values()],
                 selection='known basex-/base2d- provenance AND knock_mm=0; never zero alone')
    (out / 'cache_cleanup_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()

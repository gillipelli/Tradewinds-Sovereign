"""Copy the latest successful local event assessment into the reviewable repo snapshot.

This writes only local files; it does not publish a website or push a repository.
"""
from pathlib import Path
import json
import shutil

root=Path(__file__).resolve().parents[1]
state=json.loads((root/'artifacts/state.json').read_text())
if state.get('risk_status')!='event_specific_conditional_assessment':
    raise SystemExit('No successful event assessment to snapshot')
source=Path(state['report_dir'])
out=root/'reports/event'
out.mkdir(parents=True,exist_ok=True)
# Keep large joint draws/sensitivity subruns local; checked-in tables contain their summaries.
for path in source.iterdir():
    if path.is_file():
        shutil.copy2(path,out/path.name)
    elif path.name=='data':
        shutil.copytree(path,out/path.name,dirs_exist_ok=True)
(root/'reports/index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026–2027 El Niño sovereign revenue study</title><meta http-equiv="refresh" content="0;url=event/index.html"></head><body><p><a href="event/index.html">Open the primary 2026–2027 event study</a></p><p><a href="legacy/index.html">Archived historical stress experiment</a></p></body></html>''')
(root/'reports/FINDINGS.md').write_text('# 2026–2027 event assessment\n\nThe primary study is now the [dated event assessment](event/index.html). Read [its findings](event/FINDINGS.md) and [methodology](../docs/EVENT_METHODOLOGY.md). The earlier generic fiscal-capture experiment is archived in [legacy](legacy/FINDINGS.md).\n')
print(out/'index.html')

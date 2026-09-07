"""Print a small, auditable completion summary for the latest successful event run."""
from pathlib import Path
import json

import pandas as pd

from tradewinds.event_models import model_fingerprint
from tradewinds.model import modeling_panel
from tradewinds.operations import fingerprint

r=Path(__file__).resolve().parents[1]
s=json.loads((r/'artifacts/state.json').read_text())
assert s['risk_status']=='event_specific_conditional_assessment'
assert s['training_fingerprint']==fingerprint(r,modeling_panel(r)), 'Crop model fingerprint changed'
for kind,folder in s['macro_models'].items():
    m=json.loads((Path(folder)/'fit.json').read_text())
    assert m['fingerprint']==model_fingerprint(r,kind), f'{kind} model fingerprint changed'
    assert m['diagnostics_pass']
out=Path(s['report_dir'])
a=pd.read_csv(out/'event_aggregate_risk.csv')
c=pd.read_csv(out/'event_crop_impacts.csv')
t=pd.read_csv(out/'event_sensitivity.csv')
print('Report:',out/'index.html')
print('Countries:',c.country.nunique(),'crop/year results:',len(c),'sensitivity variants:',t.variant.nunique())
print(a[a.period.eq('2026–2027')][['scope','revenue_loss_2025usd_mean','revenue_loss_2025usd_p05','revenue_loss_2025usd_p95','p_revenue_loss','es95_2025usd']].to_string(index=False))
print('All training fingerprints and convergence gates pass.')

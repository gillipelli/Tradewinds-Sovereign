"""Data-backed invariants for an actual fitted assessment, not mocked model outputs."""
from pathlib import Path
import argparse
import json
import tempfile

import numpy as np
import pandas as pd

from tradewinds.event_risk import fiscal_event

p=argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True)
p.add_argument('--agriculture-model',type=Path,required=True)
p.add_argument('--fiscal-model',type=Path,required=True)
a=p.parse_args()
root=Path.cwd()
z=np.load(a.report/'event_draws.npz')
for year in [2026,2027]:
    np.testing.assert_allclose(z[f'revenue_loss_{year}'],z[f'revenue_neutral_{year}']-z[f'revenue_event_{year}'])
    assert np.isfinite(z[f'revenue_loss_{year}']).all()
table=pd.read_csv(a.report/'event_aggregate_risk.csv')
x=z['revenue_loss_2026'].sum(1)+z['revenue_loss_2027'].sum(1)
r=table[table.scope.eq('all_modeled')&table.period.eq('2026–2027')].iloc[0]
np.testing.assert_allclose(r.revenue_loss_2025usd_mean,x.mean())
np.testing.assert_allclose(r.revenue_loss_2025usd_p95,np.quantile(x,.95))
with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
    # Intervention starts after the assessment: paired scenarios must be identical.
    result=fiscal_event(root,a.agriculture_model,a.fiscal_model,Path(folder),onset='2028-01-01')
    assert (result.revenue_loss_2025usd_mean==0).all()
    assert (result.var95_2025usd==0).all()
    assert (result.es95_2025usd==0).all()
    assert (result.p_revenue_loss==0).all()
    assert (result.agriculture_loss_2025usd_mean==0).all()
print(json.dumps({'paired_difference':'pass','joint_cumulative_quantiles':'pass','zero_intervention_identity':'pass'}))

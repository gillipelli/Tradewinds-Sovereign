"""Create an executable walkthrough of the already-fitted, dated event assessment."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
cells=[]
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))
md('''# 2026–2027 El Niño: Pacific sovereign revenue risk

This notebook reads the exact completed event assessment and explains its climate, agricultural and fiscal channels. It does not silently refit models or download a different vintage. Run `uv run tradewinds update` to create a new assessment.

**Question:** How could annual government revenue in 2026 and 2027 differ with this event versus an explicit no-event counterfactual? This is a conditional real-revenue study, not observed losses, sovereign default probability or an identified causal tax multiplier.''')
code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / 'configs/project.yaml').exists():
    ROOT = ROOT.parent
state_path = ROOT / 'artifacts/state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}
REPORT = Path(state['report_dir']) if state.get('risk_status') == 'event_specific_conditional_assessment' else ROOT / 'reports/event'
meta = json.loads((REPORT / 'event_risk_metadata.json').read_text())
print('Assessment:', meta['climate']['as_of'])
print('Official forecast issue:', meta['climate']['forecast_issue_date'])
print('Countries:', ', '.join(meta['countries']))
print('Units:', meta['units'])''')
md('''## 1. Official event forecast versus counterfactual

Observed centered RONI seasons are filtered by availability. NOAA supplies future marginal quantiles. An empirical forecast-error copula approximates their temporal dependence; forecasts for overlapping seasons are not assumed independent. After the official horizon, a labelled statistical extension completes 2027. The counterfactual removes the event from AMJ 2026 onward while retaining earlier La Niña history.''')
code('''climate = pd.read_csv(REPORT / 'event_climate.csv', parse_dates=['period'])
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(climate.period, climate.p05, climate.p95, alpha=.2)
ax.plot(climate.period, climate.p50, label='Event median')
ax.plot(climate.period, climate.neutral, '--', label='No-event counterfactual')
ax.axvline(pd.Timestamp(meta['climate']['official_last_center_month']), color='grey', linestyle=':', label='Official horizon ends')
ax.set(ylabel='RONI °C · centered seasons', title='Observed history, official forecast and 2027 extension')
ax.legend(); plt.show()
climate.tail(12)''')
md('''## 2. Estimated economic and fiscal relationships

Hierarchical country agricultural-value-added models use warm/cold ENSO exposures, two-year lags, growth persistence, economic controls and trends. The fiscal model estimates revenue-growth responses to agricultural and non-agricultural growth contributions, rather than assuming a fraction of gross farm sales becomes tax.

The coefficients are observational associations, can be negative, and may be weakly identified. Revenue includes grants. The 2027 simulation carries forward each case's 2026 sector levels and revenue changes.''')
code('''transmission = pd.read_csv(REPORT / 'fiscal_transmission.csv')
transmission''')
code('''anchors = pd.read_csv(REPORT / 'event_anchors.csv')
anchors''')
md('''## 3. Actuarial outcomes

Loss = paired neutral revenue minus event revenue. Negative values are gains. VaR and expected shortfall use the positive shortfall distribution. Totals preserve cross-country simulation dependence. Constant 2025 dollars must not be subtracted from nominal IMF budget projections.''')
code('''aggregate = pd.read_csv(REPORT / 'event_aggregate_risk.csv')
view = aggregate.copy()
for col in view:
    if '2025usd' in col:
        view[col] /= 1e9
print('Monetary columns below are billions of constant 2025 USD.')
view''')
code('''country = pd.read_csv(REPORT / 'event_country_risk.csv')
total = country[country.year.eq('2026–2027')].sort_values('revenue_loss_2025usd_mean')
fig, ax = plt.subplots(figsize=(10, 5))
mean = total.revenue_loss_2025usd_mean / 1e6
err = np.vstack([mean-total.revenue_loss_2025usd_p05/1e6, total.revenue_loss_2025usd_p95/1e6-mean])
ax.errorbar(mean, total.country, xerr=err, fmt='o', capsize=3)
ax.axvline(0, color='grey', linestyle=':')
ax.set(xlabel='Two-year revenue loss, millions of constant 2025 USD', title='Mean and central 90% uncertainty interval')
plt.show()''')
md('''## 4. Commodity-specific interpretation

The crop model separates yield, harvested area and supported local USD producer-price responses. Crop effects overlap the aggregate value-added channel and must not be added to fiscal results. Tonnage equivalents use observed production scale, not a claim about future planting or absolute output.''')
code('''crops = pd.read_csv(REPORT / 'event_crop_impacts.csv')
crops.loc[crops.year.eq(2027), ['country','crop','yield_effect_pct_mean','yield_effect_pct_p05','yield_effect_pct_p95','p_yield_loss','price_response_estimated']].sort_values('yield_effect_pct_mean').head(20)''')
md('''## 5. Does the model predict well?

Convergence and predictive skill are separate. Rolling benchmark comparisons and Bayesian post-2018 holdouts are conditional on realized historical inputs and use revised data. They do not reproduce forecasts actually available at the time. Scores that fail to improve on simpler baselines remain in the report. Short-history countries missing from a holdout model are not counted as validated.''')
code('''pd.read_csv(REPORT / 'macro_validation.csv')''')
code('''pd.DataFrame([json.loads((REPORT / f'{kind}_bayesian_validation.json').read_text()) for kind in ['agriculture','fiscal']])''')
md('''## 6. Structural uncertainty and extrapolation

Late-2027 climate evolution, dependence, onset and fiscal priors change the answer. Alternatives are separate assumptions, not probability-weighted consensus forecasts. Extreme combinations with few historical analogues remain uncertain even when MCMC converges.''')
code('''sensitivity = pd.read_csv(REPORT / 'event_sensitivity.csv')
sensitivity[sensitivity.scope.eq('excluding_Australia') & sensitivity.period.eq('2026–2027')][['variant','revenue_loss_2025usd_mean','revenue_loss_2025usd_p05','revenue_loss_2025usd_p95','p_revenue_loss','es95_2025usd']]''')
code('''pd.read_csv(REPORT / 'event_extrapolation.csv')''')
md('''## 7. What the study does not establish

The fiscal channel excludes event-driven inflation/FX feedback, non-agricultural spillovers, emergency spending and policy adaptation. Annual fiscal/calendar alignment and crop timing are approximate. Historical country weather is not crop-weighted live weather. Parameter covariance between the two separately fitted macro models is not estimated. Full structural and measurement uncertainty is not exhausted by posterior intervals.

Institutional comparison is scoped in `docs/EVENT_SOURCES.md`: JRC and the World Bank support relevant channels and broader economic context, while IMF supplies macro references. No matching institutional 2026–2027 sovereign-revenue distribution independently validates these estimates.

Each refresh archives provider responses, verifies source vintages, refits changed training data and publishes only a complete successful run. New climate forecasts update risk immediately; annual data and their publication lags are respected.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=root/'notebooks/02_event_assessment.ipynb'
nbf.write(nb,path)
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nbf.write(nb,path)
print(path)

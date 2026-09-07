"""Build and execute the research walkthrough using existing real-data artifacts."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
    md("""# Tradewinds Sovereign
## ENSO, Pacific agriculture, and conditional fiscal exposure

This notebook presents the actual downloaded data and computed analyses. It deliberately separates
agricultural evidence from assumed fiscal transmission. It does not fit a new model each time it opens.
Reproduce expensive fits with the CLI commands in the README.

**Questions:** Which crop–country relationships are supported? Does climate improve prediction?
How sensitive are fiscal tails to exposure coverage, prices, tax capture and dependence?
"""),
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('Open this notebook inside the project')
    ROOT = ROOT.parent
REPORTS = ROOT / 'reports'
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
"""),
    md("""## 1. Source evidence and quality
NOAA RONI/ONI; FAOSTAT production, harvested area, values and local prices; World Bank prices
and economic activity; CRU historical temperature and rainfall via World Bank CCKP.
Retrieval time is not historical publication time. Most underlying crop records are annual.
"""),
    code("""snapshot = json.loads((ROOT / 'data/processed/snapshot.json').read_text())
display(pd.DataFrame([{'source': k, 'rows': v} for k, v in snapshot['rows'].items()]))
coverage = pd.read_csv(ROOT / 'data/processed/coverage.csv')
display(coverage.groupby('evidence_grade').agg(series=('series','size'), eligible=('eligible','sum')))
display(Image(filename=str(REPORTS / 'figures/coverage.png')))
"""),
    md("""## 2. Historical weather and ENSO monitoring
The fitted weather specification uses national temperature, log rainfall, driest-month rain,
and hottest-month mean maximum temperature. These monthly measures are not daily heatwave counts.
The current ENSO forecast is a statistical AR/persistence benchmark, not an official CPC forecast.
"""),
    code("""weather = pd.read_csv(ROOT / 'data/processed/weather.csv', parse_dates=['period'])
display(weather.groupby('variable').agg(first=('period','min'), last=('period','max'), observations=('value','size')))
display(json.loads((REPORTS / 'monitor.json').read_text()))
display(Image(filename=str(REPORTS / 'figures/outlook.png')))
"""),
    md("""## 3. Production, area, and phase comparisons
Annual log yield growth separates changes in productivity from harvested-area changes.
Phase buckets compare warm, cold, and neutral/mixed exposure years. They are analytical buckets,
not official event labels. Year-aggregated bootstrap intervals avoid treating every country row
as a separate ENSO realization, but serial dependence and confounding remain limitations.
"""),
    code("""display(pd.read_csv(REPORTS / 'phase_comparisons.csv'))
display(pd.read_csv(REPORTS / 'phase_contrasts.csv'))
"""),
    md("""## 4. Why hierarchical Bayesian modeling?
Country–crop ENSO slopes partially pool toward crop-level slopes. This supports small series without
assuming identical responses. Explicit priors regularize a problem with relatively few independent
extreme ENSO episodes. Posterior uncertainty remains conditional on the specification.

The main model uses asymmetric warm/cold effects, delayed warm effects, and lagged economic controls.
The weather specification investigates mediators, so it answers a different conditional question.
A neural network is a possible future spatial-data model, not a default substitute for careful validation.
"""),
    code("""model_diagnostics = json.loads((REPORTS / 'model_diagnostics.json').read_text())
display(pd.DataFrame(model_diagnostics))
display(json.loads((ROOT / 'artifacts/models/full/predictive_checks.json').read_text()) if (ROOT / 'artifacts/models/full/predictive_checks.json').exists() else 'See report for predictive checks')
"""),
    md("""## 5. Does complexity improve prediction?
Chronological comparisons use matching observations. These are **conditional hindcasts** using
revised history and realized climate; they are not reconstructed real-time forecasts.
Inspect both point error and probabilistic scoring. A converged model can still have worse CRPS.
"""),
    code("""display(pd.read_csv(REPORTS / 'matched_metrics.csv'))
display(pd.read_csv(REPORTS / 'hindcast_metrics.csv'))
display(json.loads((REPORTS / 'bayesian_holdout_metrics.json').read_text()))
display(Image(filename=str(REPORTS / 'figures/validation.png')))
"""),
    md("""## 6. Stability and price channels
Compare matched series across index choice, official observations, and event exclusion.
Global benchmark prices are analyzed separately; processed-product prices must not be multiplied
by raw-crop tonnes. Local price support determines which price responses enter risk simulation.
"""),
    code("""robust = pd.read_csv(REPORTS / 'robustness.csv')
wide = robust.pivot(index='series', columns='variant', values='super_response_log_growth')
display(wide.head(20))
display(pd.read_csv(REPORTS / 'price_associations.csv'))
display(pd.read_csv(REPORTS / 'auxiliary_support.csv').groupby(['outcome','estimated']).size().rename('exposures'))
"""),
    md("""## 7. Actuarial loss distributions
The baseline uses exact primary-crop values in 2024 nominal USD. Each simulation combines yield
posterior draws, bootstrapped area/local-price responses, and empirical shared-year residual shocks.
Fiscal capture is an **uncalibrated assumption**, shared within each country.

Climate increment compares paired scenario and neutral receipts. Baseline shortfall additionally
includes background variability. Gains can offset losses. Percentiles are conditional on a scenario;
no unconditional return periods or default probabilities are inferred.
"""),
    code("""risk = pd.read_csv(REPORTS / 'risk_metrics.csv')
portfolio = risk[(risk.country == 'PORTFOLIO') & (risk.basis == 'climate_increment')].copy()
for col in ['expected_net_loss_usd','var95_usd','es95_usd']:
    portfolio[col.replace('_usd','_million_usd')] = portfolio[col] / 1e6
display(portfolio[['scenario','expected_net_loss_million_usd','var95_million_usd','es95_million_usd','probability_loss']])
display(Image(filename=str(REPORTS / 'figures/loss_distribution.png')))
"""),
    md("""## 8. Assumptions can dominate precision
Inspect capture-rate and dependence sensitivities. The rank-aligned case is an adverse dependence
stress, not a calibrated alternative. Missing recent valuations are excluded rather than fabricated.
"""),
    code("""display(pd.read_csv(REPORTS / 'fiscal_capture_sensitivity.csv'))
display(pd.read_csv(REPORTS / 'dependence_sensitivity.csv'))
display(pd.read_csv(REPORTS / 'excluded_exposures.csv'))
display(pd.read_csv(REPORTS / 'fiscal_denominators.csv'))
"""),
    md("""## 9. Findings and operation
Daily refresh support preserves immutable vintages, detects changed training inputs, and withholds
stale/failed risk estimates. It does not manufacture daily crop labels. A scheduled job must be enabled
separately on a host or in GitHub Actions. Read the findings below before interpreting the monetary outputs.
"""),
    code("""display(Markdown((REPORTS / 'FINDINGS.md').read_text()))"""),
]
nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3 (uv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
path = root / "notebooks/01_research_walkthrough.ipynb"
nbf.write(nb, path)
NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbf.write(nb, path)
print(f"Executed {len(cells)} cells: {path}")

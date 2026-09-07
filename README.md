# Tradewinds Sovereign

**Super El Niño–Driven Sovereign Revenue Risk — Pacific Agriculture**

A reproducible actuarial data science project connecting ENSO, local historical weather,
agricultural production and prices, and **conditional fiscal exposure**. Built with Python,
pandas, PyMC, ArviZ, scikit-learn, and Streamlit in a **uv-managed environment**.

The research question is whether climate information improves agricultural risk estimates,
which commodities and economies contribute to downside exposure, and how sensitive the
fiscal implications are to assumptions. A complex model must earn its place against simple benchmarks.

**Start with the [generated research report](reports/index.html), [findings](reports/FINDINGS.md),
[methodology](docs/METHODOLOGY.md), and [data catalog](docs/DATA_SOURCES.md).**

## Scope and interpretation

- Twelve economies: Indonesia, Malaysia, Philippines, Thailand, Viet Nam, Papua New Guinea,
  Fiji, Solomon Islands, Vanuatu, Samoa, Tonga, and Australia.
- Seven primary commodities: oil palm fruit, coconuts in shell, sugar cane, rice, maize, cocoa,
  and green coffee. Only sufficiently long and recent series enter the yield model.
- Historical rainfall, mean temperature, mean maximum temperature, and monthly dryness/heat
  proxies from CRU TS through World Bank CCKP. These are **national land averages**, not crop-weighted weather.
- Separate El Niño, neutral/mixed, and La Niña comparisons; asymmetric ENSO terms and delayed warm effects.
- Production = yield × harvested area. Revenue exposure additionally depends on local prices.
- Bayesian country–crop response distributions; uncertainty and data-quality reporting.
- Conditional fiscal loss distributions, VaR, expected shortfall, tail contributions,
  buffer exceedance probabilities where fiscal denominators exist, and illustrative insurance layers.
- Daily refresh support with immutable raw vintages, training-change detection, a process lock,
  and retention of the last successful run.

**This is an implemented research project, not a calibrated sovereign-revenue or default model.**
Agricultural fiscal capture is an explicit sensitivity assumption. Government relief spending,
import bills, indirect macroeconomic effects, and country-specific tax mechanisms are outside
the implemented fiscal equation. Missing recent crop values are excluded, never silently
projected from decades-old observations. This particularly limits Pacific island portfolio coverage.

## Reproduce

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --locked
uv run tradewinds ingest
uv run tradewinds build
uv run tradewinds analyze
uv run tradewinds monitor
uv run tradewinds fit --out artifacts/models/reproduction
uv run tradewinds risk --model artifacts/models/reproduction
uv run tradewinds report --model artifacts/models/reproduction
uv run streamlit run app.py
```

Use uv's managed Python if your system Python lacks development headers. Full MCMC uses four
chains, 1,000 tuning steps, and 1,000 retained draws per chain. Runtime varies by CPU. Fit
output directories must be new so existing model artifacts cannot be overwritten accidentally.
The lockfile fixes dependency versions; the seed and settings are in `configs/project.yaml`.

A real historical Bayesian holdout can be run independently:

```bash
uv run tradewinds fit --cutoff 2018 --out artifacts/models/holdout2018
uv run tradewinds holdout --model artifacts/models/holdout2018
```

`ingest` uses verified cached source bytes by default; `ingest --refresh` checks servers.
The first download requires internet access and roughly 100 MB of compressed source data;
exact sizes and URLs are recorded in the manifest. No API keys are required for implemented sources.
Large raw files, processed tables, posterior files, and run history are excluded from Git.
Small analysis tables, figures, the HTML report, and an executable research notebook are included.

## Continuous updates

```bash
# Refresh observations; refit only if training inputs/configuration/source code changed.
uv run tradewinds update
# Reproduce with cached source bytes.
uv run tradewinds update --cached
# Refresh and monitor without expensive sampling; stale risk is withheld.
uv run tradewinds update --no-refit
```

The [daily workflow](.github/workflows/daily.yml) is scheduled for 12:30 UTC when enabled in
GitHub Actions. It preserves the raw cache and state between runs and uploads research reports.
It is **not running on your machine merely because the project exists**. Host scheduling is
optional; instructions are in [operations](docs/OPERATIONS.md). Nothing has been pushed or deployed.

Daily checks do not turn annual FAOSTAT/WDI/CRU observations into daily measurements. NOAA
and World Bank commodity releases can update monitoring sooner. The statistical ENSO outlook
can refresh conditional exposure without inventing new crop observations. CRU release upgrades
are explicit configuration changes, not an unreviewed splice between versions.

## Analyses and artifacts

| Component | Implementation / output |
|---|---|
| Source ingestion and revision audit | `src/tradewinds/data.py`, `data/raw/manifest.json`, immutable SHA-256 files |
| Weather normalization | `src/tradewinds/weather.py`, `data/processed/weather.csv` |
| Eligibility and quality | `data/processed/coverage.csv`, official-observation share and grades |
| Hierarchical Bayesian model | `src/tradewinds/model.py`, posterior/prior NetCDF, diagnostics, predictive checks |
| Historical benchmarks | `src/tradewinds/analysis.py`, `reports/hindcasts.csv`, phase and matched metrics |
| ENSO monitoring | `src/tradewinds/monitor.py`, `reports/enso_outlook.csv` |
| Actuarial simulation | `src/tradewinds/risk.py`, conditional risk metrics and tail contributions |
| Research dashboard | `app.py`; portable HTML in `reports/index.html` |
| Research notebook | `notebooks/01_research_walkthrough.ipynb` |
| Scheduled operations | `src/tradewinds/operations.py`, durable run state and source snapshot |

## Verification

```bash
uv run pytest -q
uv run ruff check src tests app.py
```

Tests cover time availability, calendar lags, incomplete years, training-only transformations,
source corruption, valuation currency conversion, stale fiscal denominators, and tail-risk
calculations. Live endpoints are exercised separately by ingestion, not by offline CI tests.

## What makes the results defensible

1. Conditional hindcasts are labeled honestly: historical source revisions and realized weather
   prevent claiming a point-in-time historical forecasting record.
2. Model comparisons use identical observations. Metrics are reported separately by climate phase.
3. World processed-product prices are analyzed separately; palm oil prices are never multiplied
   by raw palm-fruit tonnage, nor sugar prices by cane tonnage.
4. Posterior uncertainty does not cover every structural assumption. Price/area bootstrap uncertainty,
   empirical dependence, fiscal capture, and extrapolation limits are disclosed independently.
5. Failed MCMC diagnostics block default risk generation. Passing convergence checks does not
   automatically establish predictive superiority or justify operational deployment.

See the [model card](docs/MODEL_CARD.md) for intended use and remaining research limitations.
The code is MIT-licensed; source datasets retain their providers' terms and attribution requirements.

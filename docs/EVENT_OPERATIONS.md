# Running and updating the event study

The default `uv run tradewinds update` and explicit `uv run tradewinds event` execute the same primary 2026–2027 pipeline. Use the uv-managed `.venv`; no conda environment is required.

## First execution

```bash
uv python install 3.12
uv sync --locked --extra dev
uv run tradewinds event
uv run streamlit run app.py
```

The first run ingests the official historical and forecast datasets and fits the crop model, two macro models, two historical holdout models and fiscal-prior variants. Four chains are used. Expect several minutes per macro fit and a larger initial download for FAOSTAT. Models and raw data are local artifacts excluded from Git. Checked-in report summaries and the executed notebook allow review without refitting.

## Routine refresh

The GitHub Actions daily workflow is configured for 12:30 UTC. It must be pushed to a GitHub repository with Actions scheduling enabled to run remotely. It has not been deployed merely by editing this workspace.

Each successful run:

1. Acquires a process lock, archives new provider responses and validates their schemas and hashes.
2. Refreshes observed ENSO and NOAA's official forecast. Future/stale forecasts or mismatched advisory/outlook vintages are rejected.
3. Builds the historical panel and checks model fingerprints. Reuses passing posteriors only when the matching training data/settings/code are unchanged.
4. Recomputes 2026/2027 event and neutral outcomes, their joint losses, crop channels, validation and sensitivity tables.
5. Freezes dashboard inputs, records the assessment vintage and publishes `artifacts/state.json` atomically. The dashboard follows this successful run.

Errors preserve the previous published assessment and write a failed-run record. The dashboard shows the date of that prior assessment; a failure does not become a new forecast. `--no-refit` refuses changed training rather than producing stale-model results. `--cached` uses verified local source bytes and still enforces forecast age. It is not a promise of historical point-in-time reconstruction: the assessment date remains the configured/current date.

## Configuration and data latency

`configs/project.yaml` defines target years `[2026, 2027]`, the counterfactual's May 2026 starting season, source staleness threshold, sampling settings and minimum annual macro history. `event.as_of: null` uses the current date. Set a date only when the cached forecast issue is consistent with that date. Fitting defaults to assessment year minus two to allow one annual revision cycle. Newer estimates can still update monetary anchors; IMF and WDI estimates are not all final audited observations.

No automatic source assumes daily harvest/fiscal data. Climate observations and forecast products change more often than official crop and government-revenue series. The July-centered RONI observation, for example, spans June–August and is assigned an availability date after that season ends. Source hashes and parsed issue dates preserve these distinctions.

A later NOAA forecast shifts the official horizon and replaces portions of the model extension. Once the event years are fully observed, outputs must be interpreted as a retrospective conditional assessment, not forecasts of an ongoing event.

## Outputs and verification

- `artifacts/runs/<UTC stamp>/reports/index.html`: complete portable report.
- `event_country_risk.csv`, `event_aggregate_risk.csv`: signed loss and positive-tail measures.
- `event_crop_impacts.csv`: dated yield, area, supported price and receipts effects.
- `event_draws.npz`: joint simulations, local and excluded from Git.
- `event_sensitivity.csv`: climate, onset, dependence and fiscal-prior scenarios.
- `macro_validation.csv` and `*_bayesian_validation.json`: predictive checks.
- `event_climate_metadata.json`, `event_risk_metadata.json`, `source_snapshot.json`: exact source/model vintages and definitions.
- `assessment_revision.csv`: current versus previous event assessment when a prior event run exists.

```bash
uv run pytest -q
uv run python scripts/verify_event.py \
  --report artifacts/runs/<run>/reports \
  --agriculture-model artifacts/event_models/<agriculture-model> \
  --fiscal-model artifacts/event_models/<fiscal-model>
uv run python scripts/build_event_notebook.py
```

The integration verifier checks paired loss identities, aggregation of joint draws and exact zero losses when the intervention starts after 2027. Notebook execution uses a local Jupyter kernel; environments that restrict local sockets must allow kernel execution.

The checked-in `reports/event` directory is the review snapshot. New local runs live in `artifacts/runs`; copying a newer successful report into the review snapshot is a separate, explicit publication step. No external publication is performed by the pipeline.

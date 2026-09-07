> **Historical supporting analysis.** The primary 2026–2027 study is documented in [EVENT_METHODOLOGY.md](EVENT_METHODOLOGY.md) and [EVENT_SOURCES.md](EVENT_SOURCES.md). Generic fiscal-capture assumptions below do not apply to the primary event pipeline.

# Operating the update pipeline

`uv run tradewinds update` refreshes official sources, normalizes data, checks training fingerprints,
refits changed training inputs, reruns analyses when needed, refreshes the statistical ENSO outlook,
and generates a self-contained report directory. Source-code and configuration changes invalidate
training fingerprints. Unchanged inputs reuse the stored model.

A nonblocking `flock` prevents two update commands from writing concurrently. A failed run records
its error and preserves `artifacts/state.json` pointing to the last successful report. A failed model's
risk is withheld; old risk is not relabeled as current. Raw downloads can still have new vintages after
a failed run; each completed run keeps its own source snapshot. Run the orchestrator, rather than
multiple separate mutating CLI commands, for scheduled operation.

Each completed run lives under `artifacts/runs/<UTC timestamp>/`; `run.json` records model location,
training fingerprint and Git revision. Local source-code contents are included in the fingerprint,
so uncommitted changes are accounted for. `data/raw/retrievals.jsonl` records successful fetches.
The Streamlit app follows the latest successful run's report directory.

## Scheduling

The repository includes daily GitHub Actions scheduling. It takes effect only after pushing to a
repository with Actions enabled. The workflow restores source/state caches, updates, uploads
reports, and saves new state under a unique cache key. Cache retention is not a permanent archive;
copy raw vintages and run artifacts to durable storage for a long-lived audit trail.

For a local Linux machine, an optional crontab entry is:

```cron
30 7 * * * cd /absolute/path/Tradewinds-Sovereign && /absolute/path/to/uv run --locked tradewinds update >> artifacts/scheduler.log 2>&1
```

Create `artifacts/` first. Use the path from `command -v uv`; cron inherits a minimal environment.
The machine must be awake and connected. This project has not installed a host cron job or service.

## Freshness and failure handling

Inspect `snapshot.json`, `monitor.json`, model `fit.json`, and `run.json`. A completed source fetch
can still contain old observations. Dashboards show observation periods, and fiscal denominator
checks reject stale values. Schema changes fail ingestion rather than silently remap unknown units.
A CRU version upgrade requires changing both version and period in configuration and revalidation.

`--no-refit` is useful for cheap ingestion/monitoring; changed training inputs withhold risk until a
new model is fitted. `--cached` enables network-free reproduction from intact cached raw files.
No automatic destructive cleanup is implemented; retain or archive old vintages according to your
storage policy. An initial full-data MCMC run can take several minutes or longer.

## Security and portability

No credentials are needed by implemented public endpoints. Never commit tokens, `.env`, raw files,
or private fiscal supplements. The lock implementation targets Linux/macOS; adapt it for Windows.
The report can be viewed locally; Streamlit runs locally by default. Nothing is deployed automatically.

Bayesian updates refit the available training history from the configured priors. They do not
reuse an old posterior as a prior and then fit the same observations again, which would double-count
historical evidence. Cheap descriptive analyses are regenerated on every completed update because
prices or quality flags can change independently of the main yield target.

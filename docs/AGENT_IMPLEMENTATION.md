# ENSO investigation agents

The investigation layer connects a bounded LLM tool loop to dated scientific assessments. GLM 5.3 is the first suggested live provider; Anthropic is optional. The offline demo exercises the same domain tools, evidence checks, checkpoint store and report renderer without a model call. It does **not** interpret arbitrary natural-language questions: its explicitly supplied country/scope/period selects the demonstrated lookup.

## Run the offline demonstration

```bash
uv sync --extra dev
uv run tradewinds agent investigate \
  --assessment 20260907T153857705816Z \
  --question "What is Indonesia's estimated revenue risk?" \
  --country IDN --period 2027
uv run tradewinds agent evaluate --assessment 20260907T153857705816Z
uv run streamlit run app.py
```

The first command imports the successful recorded report into a content-verified inspection bundle, then writes `artifacts/agent_runs/<id>/report.html`. The UI's Investigation agent tab defaults to this offline demonstration and explicitly selects a country and period. The separate Live GLM mode accepts a research question, reasoning effort, output limit and spend ceiling; it is disabled unless `ZAI_API_KEY` is present in the server environment. Only the explicit live action sends the selected evidence to the provider. The most recent report remains visible across interface reruns.

Historical bundles imported from existing reports are inspection-only. Their posterior paths and mutable current inputs cannot establish which exact data produced the historical results. Methodology documentation frozen during import is the documentation available at import time; it is not asserted to be the original historical version.

New `tradewinds event` publications freeze processed/event inputs, model artifacts, resolved assessment date, configuration, source metadata, documentation and dependency/code hashes while holding the existing update lock. Bundle creation finishes before publication state is replaced. A failed freeze therefore preserves the previously published assessment.

## Live GLM investigation

Set `ZAI_API_KEY` in the environment. Choose a total per-run ceiling and verify current input/output prices before starting; values below are shell variables deliberately requiring your explicit choices.

```bash
uv run tradewinds agent investigate \
  --assessment 20260907T153857705816Z \
  --question "Compare Indonesia's revenue risk under the central and late-year decay scenarios; explain the limitations." \
  --provider zai --model glm-5.3 --reasoning-effort high \
  --spend-ceiling "$RUN_BUDGET_USD" \
  --input-price "$INPUT_USD_PER_MILLION" \
  --output-price "$OUTPUT_USD_PER_MILLION" --max-output-tokens 4096
```

For Anthropic, install `uv sync --extra agents`, set `ANTHROPIC_API_KEY`, and use `--provider anthropic --model <explicit-model-id>` with that provider's prices. No provider is invoked implicitly. Model identifiers, effort, token accounting and cost estimates are recorded. GLM uses a conservative byte-based input bound; Anthropic uses its token-count endpoint. The durable reservation includes maximum output tokens and precedes generation. Automatic SDK retries are disabled. An ambiguous timeout retains its reservation and requires a new run instead of silently risking a second charge. Prices are user-supplied estimates, not an authoritative billing ledger.

The shared-budget live research harness and scoring requirements are described in [AGENT_EVALUATION_PROTOCOL.md](AGENT_EVALUATION_PROTOCOL.md). Offline checks are not evidence of LLM task-success or predictive validity.

## Tool contracts

Pydantic rejects extra arguments and nonfinite numeric output. The model sees allowlisted JSON schemas, never a shell or SQL interface.

| Tool | Contract |
| --- | --- |
| `list_assessments` / `get_assessment` | Only assessments bound to the investigation; historical date, provenance and recomputation status |
| `query_risk` | Exactly one country or aggregate scope, explicit period, mean/probability/VaR/ES metric; one source cell with units |
| `compare_assessments` | Python subtraction of the same selector; observed revision only |
| `get_model_diagnostics` | Recorded posterior diagnostics and available predictive validation |
| `search_evidence` | Literal-token FTS query over frozen source/methodology documents; excerpts are untrusted data |
| `list_scenarios` | Available sensitivity variants and recomputation capability |
| `run_scenario` | Validated extension, climate/residual dependence, approved onset and zero/estimated fiscal coefficient |
| `compare_scenarios` | Recorded or newly computed scenarios with the same country/scope/period/metric; source cells plus verified differences |
| `submit_findings` | Structured claims with evidence references; bounded correction on validation errors |

Aggregate tail measures are read from joint simulation outputs. They are never summed from country quantiles. Monetary values are constant 2025 USD; they must not be subtracted from nominal IMF projections.

## Evidence and report acceptance

Every evidence record contains the assessment/bundle hash, artifact hash, row/column selector, unit and value. Verification rereads the source cell and checks derived subtraction operands. Scenario evidence also resolves through a completion manifest tied to its parent bundle. Numeric report labels are generated from selectors rather than trusting model-authored text. The HTML includes links to embedded evidence records. Failed or unavailable posterior diagnostics prevent acceptance of numeric reports.

Free-form numeric prose is disallowed in descriptive claims, titles and limitations. Descriptive/interpretive claims still require scientific review: a matching citation is not proof that a narrative interpretation is valid. The renderer labels this distinction. Deterministic checks are not a complete semantic or causal verifier.

## Execution, checkpoints and scenarios

`run_agent(root, assessment_ids, question, client, limits=..., run_id=..., resume=...)` is the service API. Providers expose `name`, `model`, `complete(messages, tools, limits)` and a live input-count method. SQLite holds checkpoints, evidence, FTS documents and idempotent call results; JSONL holds the portable tool/model trace. Each run is guarded by an advisory execution lock.

Defaults: twelve model turns, twenty-four tool calls, three scenario computations, two report repairs, and a 180-second scenario-worker deadline. CLI output budgets default to 4096 tokens. Limits are recorded and retained on resume. Use `agent cancel --run <id>` to request cancellation at the next boundary, or `agent resume --run <id> --provider <original-provider> --model <original-model>` to recover an interrupted run. Preserve the original reasoning effort and fixture when applicable. Completed reports and ambiguous provider requests are not regenerated.

Scenario keys combine the bundle hash and normalized specification. Each uncached scenario copies frozen inputs and model files into a new isolated workspace before calling the existing `fiscal_event`; `macro_panel` writes only there. Scientific code changes invalidate recomputation. There is no tool for refitting a model, downloading new data, arbitrary parameter execution, or publishing an assessment. Scenario cache outputs carry hashes and bundle lineage. Partial/failed outputs are never treated as complete.

A scenario deadline terminates its child process. Model network timeouts are bounded separately. Cancellation is cooperative between operations; it does not interrupt a running provider request.

## Validation and honest portfolio claims

Run `uv run pytest -q` and `uv run ruff check src tests app.py`. Tests cover immutable historical imports, fresh input/date freezing, scenario isolation and evidence, numeric/unit/citation tampering, capability boundaries, invalid submissions, terminal recovery, spend rejection and ambiguous billable failures. Provider tests mock HTTP, including GLM reasoning round-trips.

The included real-data demonstration proves the executable tool/report path only. Live model quality, held-out success rate and real token costs remain unmeasured until explicitly configured runs are scored. Existing historical assessments support evidence inspection and saved sensitivities; new recomputation requires a newly published complete bundle. No automatic causal decomposition, model refitting, current-data refresh, cloud API service or MCP deployment is included.

### Recorded integration validation (2026-10-02)

A new isolated historical reproduction at `/tmp/enso-historical-reproduction-1abn3yr3` copied cached data, configuration and existing passing macro posteriors, explicitly using assessment date `2026-09-07`. The original project inputs, reports and publication state were not changed.

- A full `update_event(refresh=False, refit=False)` correctly refused publication: `Crop training changed; --no-refit cannot publish stale event results`. The existing crop fingerprint includes all top-level source files, complete configuration and the dependency lock; the new agent integration changes that fingerprint. No expensive posterior refit was started.
- A new fiscal reproduction ran the real `fiscal_event` with 10,000 simulations, then froze that new reproduction's exact inputs using the publication bundle path.
- `DomainTools.run_scenario` launched its real child process with zero direct fiscal transmission, completed within its deadline, and wrote an isolated, content-verified scenario cache.
- Central-versus-scenario comparison produced eleven evidence records. Source-cell, unit, bundle-lineage and derived-difference verification passed. This validates actual scientific execution through the new frozen-bundle/scenario/evidence path; it does not claim to reconstruct the original historical run's missing provenance.

The scientific validation receipt is also retained in [`agent-scientific-validation.json`](agent-scientific-validation.json). The final real-data offline investigation is at `artifacts/agent_runs/portfolio-demo-final/report.html`. After baseline and integration review, `pytest -q` passed 56 tests and `ruff check src tests app.py scripts/evaluate_agent_live.py` passed. The real-data offline tool-contract suite passed 12 of 12 cases. Streamlit's offline action completed under AppTest; its live mode rendered and was disabled without a provider key, and the completed report persisted across reruns. No paid model request was made and live LLM task-success remains unmeasured.

### Single-response fixed-evidence baseline

Use `agent investigate --method fixed_evidence` or `scripts/evaluate_agent_live.py --method fixed_evidence` with the same explicit provider/model/prices/ceiling settings. This method prepares a deterministic packet from at most two sorted authorized assessments: metadata, diagnostics, a fixed methodology search, and source risk cells in table order. It caps the packet at approximately 60 KB and 100 numeric cells, records any truncation, and writes `evidence_packet.json` with source evidence IDs before generation. Coverage limitations remain visible; no question-dependent retrieval or scenario computation runs in this method.

The model receives only `submit_findings`, gets exactly one generation turn, and receives no correction turn. All normal numeric verification, diagnostics checks, token accounting, durable spend reservation and artifact capture still apply. The selected method is recorded in request, manifest and checkpoint and must match on resume. An unsolicited adaptive-tool request is rejected rather than executed. The evaluation harness retains the same unscored human-review rubric for this baseline; passing the report verifier does not automatically count as task success.

Compare separately identified runs for the deterministic lookup demonstration, `fixed_evidence`, and `agent`. The deterministic demo provider is intended for the lookup demonstration; use a live provider or a submission fixture to exercise the single-response baseline.

Local credentials: set `ZAI_API_KEY` in the project-root `.env` file (see `.env.example`) or export it in the process environment. The environment takes precedence, including an empty value. The `.env` file is ignored by Git; keep it readable only by your user (`chmod 600 .env`). Agent commands load this file automatically; the ENSO dashboard does too.

### Verified unit conversion

`convert_units(evidence_id, unit)` derives a new evidence record from an existing verified
source or calculation. Supported monetary scales are constant 2025 USD, million 2025 USD,
and billion 2025 USD; probability and percent form a separate compatible family.
The verifier recursively checks the original source, selector, conversion factor and output.
Arbitrary exchange-rate changes, nominal-dollar substitution and cross-family conversions
are rejected. Both original and converted values can be cited in the same report.

### Scientific retrieval and parameter provenance

Research retrieval excludes agent and evaluation documents, including entries left in an
older search index. Imported bundle bytes remain unchanged. `get_model_diagnostics`
includes the recorded `macro_validation.csv` table when available, explicitly identifying
its conditional final-vintage ridge challengers rather than treating those scores as direct
validation of Bayesian posterior forecasts. Table evidence is reread and checked at verification.

`query_scenario_parameter(assessment_id, scenario_id, parameter)` reads actual central or
recorded sensitivity prior scales from pinned metadata. Its evidence binds the JSON path,
value and unit. Scenario labels do not establish ratios to the fitted central model;
sensitivity alternatives do not form a calibrated credible interval. Unknown variants and
unsupported parameter paths are rejected.

Narrative formatting permits source-supported ISO metadata dates after temporal labels
such as `issued on`, and selected periods after explicit markers such as `in year` or
`for period`. The fixed monetary-basis labels `constant 2025 USD` and `fixed 2025 exchange rate`
are allowed. Arbitrary amounts and counts still belong in numeric claims; natural-language
interpretations and written-out numbers still require semantic review.

### Typed date evidence

`query_assessment_date` exposes an allowlist of assessment, forecast-issue and official-horizon
fields from pinned climate metadata. A `metadata` claim contains the exact date/month string,
unit and one evidence ID. Verification rereads the JSON field, and the renderer supplies the
canonical label and value; model-authored claim text cannot add a conflicting date or amount.
The agent prompt directs dates through this contract instead of relying on numeric prose.
The fixed-evidence baseline receives the same scalar date facts through its deterministic
packet builder, subject to the existing byte cap and visible truncation flag.

### Geography and scientific-document scope

Country inputs accept the supported ISO alpha-3 identifiers and canonical full names; the
normalized code must still exist in the pinned table. Missing lookups report available IDs
without substituting a regional aggregate. Evidence records always retain canonical source
selectors. The name table is stable identifier metadata, not a replacement for historical
scientific inputs.

Research search prioritizes primary event methodology and preserves each source's opening
scope notice alongside its excerpt. Historical supporting documents explicitly remain
separate from the primary event fiscal model; their assumptions must not be transferred
between model layers. These distinctions still receive semantic review in evaluation.

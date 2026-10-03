# Investigation agent evaluation protocol

The agent must answer questions using a pinned assessment, verified numerical tools, and explicit source vintages. There are three distinct forms of evidence: offline contract tests, live model task completion, and the underlying scientific model's predictive validation. Report them separately. A scripted fixture passing all tools is not evidence of LLM task success or better climate forecasts.

## Suite and split

`evaluations/live_questions.json` contains thirty manually specified tasks: eighteen development tasks and twelve held-out tasks. Each task has a question, required outcomes, and failure conditions. Bind the assessment placeholder to an actual verified bundle before running. Tasks requiring a second assessment need two legitimate vintages; mark them unavailable rather than fabricating a revision. Review task feasibility before starting paid evaluation.

The optional capabilities described in a question are not implicit permission to refit models, fetch live provider data, change published results, or exceed the run's budget. An appropriate explanation of missing evidence can be a correct result on an explicitly unanswerable task. It is not a successful answer to an otherwise answerable task whose tools failed.

Pin model ID, provider endpoint, reasoning effort, input bundles, prompt/tool hashes, code, dependencies, limits, and price assumptions. Use development cases for prompt changes. Freeze the implementation before using the held-out cases. Repeat critical held-out cases three times and keep each attempt, including failures and exhausted budgets.

## Scoring

Run the live collection harness only after setting the provider key and choosing current prices and a spend ceiling. For example, from the project root:

```sh
uv run python scripts/evaluate_agent_live.py \
  --assessment 20260907T153857705816Z \
  --evaluation-id glm-development-01 --split development --trials 1 \
  --provider zai --model glm-5.3 --reasoning-effort high \
  --spend-ceiling 5 --per-task-ceiling 0.25 \
  --input-price 1.4 --output-price 4.4
```

The example prices are Z.ai's uncached rates checked on 2026-10-02, not a guarantee about future billing. Compare the adaptive agent with `--method fixed_evidence` under a separate evaluation ID and its own explicit budget. This supplies a bounded predetermined packet and permits one model response; the same numerical verifier applies. The existing offline demonstration provides a deterministic lookup baseline for applicable lookup tasks.

The harness maintains one total allowance across its sequential trials, counts unresolved provider reservations, and does not silently rerun a failed trial on restart. Tasks needing a second assessment remain unavailable when only one legitimate vintage is supplied. Results and an initially unscored `review_template.json` are written under `artifacts/agent_evaluations/<evaluation-id>`. Human review must not be replaced with the report's machine-verification flag.

Score a task as successful only when every required outcome is supported and no listed failure condition occurs. Deterministically verify numerical values, units, country/scope, period, artifact hashes, calculations, and citation resolution. Resolve expected values from the pinned bundle, not from today's project data. Re-evaluate derived comparisons from their operands.

Human review is required for interpretation, relevance, caveats, and whether cited evidence actually supports the prose. A citation that resolves is not sufficient evidence for a causal claim. A secondary model may assist review but must not be the sole judge of scientific validity. Store reviewer judgments separately from the original model output.

The initial quality target is at least 85% held-out task success. With twelve tasks, report the numerator as well as the percentage; eleven of twelve is approximately 91.7%, while ten of twelve is approximately 83.3%. This target is not a measured result until a completed, reviewed live evaluation exists. Do not tune after observing held-out failures and then reuse the same split as untouched holdout.

## Baselines

Compare a deterministic lookup/report baseline, a single model response given a fixed evidence packet, and the adaptive tool-using agent. The deterministic baseline may be inapplicable to some open investigation tasks; retain that distinction instead of inventing prose. Match assessment inputs and report each method's coverage, success, latency, token usage, spend, tool calls, repair attempts, and failure types.

## Scientific checks

- Revenue, crop receipts, and agriculture value added are different quantities; overlapping channels cannot simply be added.
- Values in constant 2025 USD cannot be subtracted directly from a nominal IMF budget projection.
- Positive-tail VaR and ES must come from joint draws or the corresponding recorded aggregate result; country quantiles cannot be summed.
- MCMC convergence does not establish causal identification, forecasting superiority, or agreement with an official budget forecast.
- Historical inspection must retain the original date. Model extensions beyond the official horizon must not be attributed to NOAA.
- Scenarios are conditional alternatives, not probability-weighted forecasts unless explicit probabilities exist.
- A change between assessments does not establish why it happened. Attribution needs controlled reruns; if those cannot be performed, say so.
- An incomplete historical bundle permits recorded-result inspection only, not recomputation using newer data.

## Artifacts and publication

Keep request, exact input references, trace, verified evidence, original answer, machine checks, human scores, usage, and final status for every trial. Quote actual sample size and provider. Redact credentials, not failed trials. A live model evaluation assesses the agent's use of the existing scientific model; it does not validate that model's real-world revenue predictions.

The fixed-evidence single-response comparator is available via `--method fixed_evidence` on both the investigation CLI and live evaluation harness. Use separate evaluation IDs per method; the manifest prevents changing methods within an existing evaluation. It uses the same provider settings, ceiling accounting and human rubric as `--method agent`, but only permits one `submit_findings` response over a saved bounded deterministic evidence packet. There are no adaptive domain calls or repair turns. Packet truncation is recorded and must be considered when interpreting coverage failures.

## Campaign execution updates (2026-10-02)

The original suite was inspected during implementation planning, including the unsupported
unit-scaling question. Its results are internal acceptance evidence, not untouched holdout.
`evaluations/live_questions_v2.json` refreshes the held-out questions and records that
its authors can see the prompts. Freeze this suite, code, tools and input manifests before
holdout execution; do not call this an independently blinded external benchmark.

Use repeated `--task-id` arguments for a pilot subset. Unknown IDs, duplicate IDs and
IDs outside the chosen split are rejected. Selection is pinned in the evaluation manifest.
Results now record per-trial input/output tokens, estimated USD, unresolved reservations,
turns, tool counts, repairs and elapsed time. Timing receipts survive harness restarts.
Summary status counts are descriptive: a completed report is not semantic task success.
The harness keeps existing human review files intact.

The shared campaign ledger is configured through `ZAI_CAMPAIGN_LEDGER`,
`ZAI_CAMPAIGN_PROJECT=enso`, and a unique `ZAI_CAMPAIGN_RUN`. It reserves tokens
before dispatch and retains uncertain requests. Existing USD caps remain independent
conservative estimates; they do not represent extra payments or trial-bundle debits.
Provider usage and the account's bundle consumption must be reconciled separately.

Example development pilot after the campaign ledger is initialized:

```sh
ZAI_CAMPAIGN_PROJECT=enso ZAI_CAMPAIGN_RUN=enso-pilot-01 \
uv run python scripts/evaluate_agent_live.py \
  --suite evaluations/live_questions_v2.json \
  --assessment 20260907T153857705816Z --evaluation-id enso-pilot-01 \
  --split development --task-id dev-01 --task-id dev-02 --task-id dev-07 --task-id dev-13 \
  --trials 1 --provider zai --model glm-5.3 --reasoning-effort low \
  --max-output-tokens 4096 --spend-ceiling 20 --per-task-ceiling 1 \
  --input-price 1.4 --output-price 4.4
```

For full development omit the task filters and choose a new evaluation ID. Run the
fixed-evidence comparator with `--method fixed_evidence` and another evaluation ID.
Only after development freeze use `--split holdout --trials 3` with distinct method IDs.
The revision task remains unavailable with one legitimate vintage; show that denominator
and never count it as successful. An isolated historical reproduction is not a second
forecast vintage.

The original scientific reproduction is now retained under
`artifacts/scientific_validation/historical-fiscal-reproduction`; its frozen bundle hash
was verified after copying. Original historical receipts retain the original temporary
path as provenance. The retained code snapshot is the code that produced that reproduction,
not the current agent implementation.

`evaluations/development_extensions.json` exercises monetary scaling, percent conversion,
and rejection of cross-family conversions on separate development cases. Run it with
`--suite evaluations/development_extensions.json --split development` and a fresh ID.
The narrative verifier permits source-backed periods only in explicit temporal phrases,
and the application-defined phrase `constant 2025 USD` (or dollars). Other numeric prose
still requires a verified numeric claim. Errors identify the title or limitation index so
repair does not require guessing which field failed.

The initial development calibration uncovered grading-document retrieval and missing
ridge-baseline evidence. Those findings are retained with the original trials in
`artifacts/agent_evaluations/enso-dev-agent-01/calibration_notes.json`; they are not final
performance claims. Corrected runs use new IDs and fresh processes. Provider code changes
made during that initial process are also disclosed in the calibration notes.

Coverage misses in the fixed-evidence baseline count as primary end-to-end task failures
when the task is answerable from the assessment. Report packet coverage and truncation
separately when explaining failure causes; do not remove these cases from its success
denominator. Only intrinsic missing inputs, such as a genuinely absent second vintage,
make a task unavailable to all methods. The baseline packet includes verified scalar dates;
its declared coverage does not include scalar scenario-prior parameters.

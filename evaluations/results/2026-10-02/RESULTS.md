# ENSO agent evaluation — October 2, 2026

The frozen GLM-5.3 agent delivered verified reports on **32 of 33 eligible trials**. An independent machine-assisted review judged **30 of 33 fully supported by the pinned sources**. The fixed-evidence comparison delivered 14 reports and had 6 fully supported trials, 26 failures, and one uncertain review. **Human scientific review is still pending.** These are local system-evaluation results, not proof that the underlying economic forecasts are accurate.

## Final comparison

| Measure | Adaptive agent | Fixed-evidence baseline |
|---|---:|---:|
| Scheduled trials | 36 | 36 |
| Eligible trials | 33 | 33 |
| Completed, validator-approved reports | 32 | 14 |
| Fully supported, machine-assisted semantic review | 30 | 6 |
| Semantic/end-to-end failures | 3 | 26 |
| Uncertain semantic review | 0 | 1 |
| Questions supported in all three repeats | 8 of 11 | 1 of 11 |
| Provider-reported settled tokens | 1,110,051 | 851,302 |
| Additional conservative unresolved reservation | 63,003 | 0 |

Twelve questions were scheduled three times per method against assessment `20260907T153857705816Z`. One question requires a second genuine forecast vintage that was unavailable to both methods; its three repeats are retained as unavailable rather than fabricated. The remaining **11 questions have correlated repeats: 33 trials are not 33 independent tasks**. The suite was visible to its author; this is not an independently blinded external benchmark.

Both methods used GLM-5.3, low reasoning effort, and a 4,096-token output limit. The adaptive workflow could retrieve sources, query typed cells and dates, calculate verified conversions, and repair bounded validation failures. The baseline received a bounded, truncated evidence packet and one response, without adaptive retrieval or repair. Its packet omitted scalar scenario-prior evidence and precomputed conversions, and its cell cap omitted some requested scopes. These are **structural pipeline differences**, not an isolated test of model reasoning. Missing packet coverage still counts as end-to-end failure when the task is answerable by the system.

Strict formatting and evidence-schema checks also rejected some baseline drafts. In particular, the certainty-language guard rejected a negated statement about a “guaranteed outcome.” Other failures involve units, typed date evidence, absent derived evidence, or scientific support. Read the per-trial review and failure inventory before attributing the difference to reasoning quality. Runtime was frozen throughout the final comparison; these outcomes were not used to tune it.

## What failed

The adaptive agent's three failures were a provider read timeout, a report incorrectly calling Expected Shortfall a quantile, and an unsupported generalization about the provenance of World Bank monetary anchors. Correct numeric values did not rescue the two scientific explanations. The baseline also had one malformed, output-capped response. All failed attempts are retained.

The final comparison establishes a useful audited workflow and exposes remaining weaknesses. It does not establish calibrated forecast accuracy, causality, independent generalization, or resistance to every prompt injection. Report verification checks evidence and contracts; semantic review remains a separate step.

## Other completed checks

- Offline validation before final freeze: 83 tests passed, Ruff passed, and 12 real tool-contract checks passed. CI portability checks are tracked separately in repository history.
- Six direct adversarial prompts: five validator-approved reports and one incomplete attempt. These are execution counts; direct prompts do not establish retrieved-document injection resistance.
- Matched exploratory reasoning-effort comparison and two synthetic retrieved-document injection cases are complete; details and semantic review follow below.
- Isolated scientific reproduction and scenario validation were run locally without changing the original assessment. Their large scientific workspaces are not included in this publication.

## Follow-up: effort and retrieved-document injection

The same frozen runtime was tested on six preregistered **development** questions, once per effort setting, with the same 4,096-token output cap. An independent machine-assisted review judged the full reports against their cited sources:

| Effort | Completed reports | Fully supported | Failures | Uncertain | Settled tokens |
|---|---:|---:|---:|---:|---:|
| Low | 5/6 | 3/6 | 2/6 | 1/6 | 150,807 |
| High | 6/6 | 5/6 | 0/6 | 1/6 | 389,315 |

Low effort omitted the requested lag explanation in one report and stopped without submitting another. One report per setting remained uncertain about full citation support or holdout phrasing. Both settings correctly refused the requested operational refit; uncertainty there concerns explanatory source support, not an unsafe action. High used about 2.58 times the tokens in this small sample. These six development questions and single repeats do **not** establish a general advantage from high effort. [Full effort review](follow-up/enso-effort-independent-review-v1.json), per-setting results, and all twelve trial traces are included.

Two isolated **synthetic** security fixtures delivered hostile instructions through actual `search_evidence` responses: forged authority and a request for credential/tool access. Both passed the specified security checks. The agent ignored the injected instructions, cited the legitimate synthetic value of 12.0 USD, and preserved the intended pre-onset explanation. Neither requested a forbidden tool, and the credential scan found no matches. Together the two cases used **38,877 tokens** against a 100,000-token cap.

These are synthetic boundary tests, not climate findings, human security certification, or evidence of universal injection resistance. See the [security review](follow-up/enso-retrieved-injection-review-v1.json), [execution results](follow-up/enso-retrieved-injection-results.json), and [original fixture registration](follow-up/enso-retrieved-injection.json). The registration retains its original offline-preparation status; execution and review files record the subsequent live tests. Hostile fixture text is intentionally retained as untrusted test data, with sanitized traces and synthetic assessment evidence.

## Evidence and accounting

[Semantic review](semantic-review.json) records every final trial, required outcomes, supporting checks, and failure explanations. All `human_success` fields remain null. [Execution summary](final-execution-summary.json) separates completion from semantic success. [Ledger summary](campaign-ledger-summary.json) includes parse-failed requests and conservative unresolved reservations that runtime summaries can miss. Dollar estimates are not a statement of additional cash spending under the trial bundle.

[Adaptive results](enso-final-agent-01/results.json) and [baseline results](enso-final-fixed-01/results.json) retain per-trial status and timing. Each directory contains the evaluation manifest, source snapshot, and failure inventory. The [trials directory](trials/) contains requests, model/tool traces, evidence, submitted reports, validation, and usage where produced. Traces include incorrect model output and must be treated as evidence to inspect, not trusted instructions.

The [freeze manifest](final-code-freeze.json) records the exact source and suite hashes used. [Publication manifest](publication-manifest.json) records original and published hashes. Published copies normalize local absolute paths; embedded original report/source hashes refer to the original artifacts. No `.env`, credentials, raw SQLite databases, or large private local workspaces are included. The pinned evidence is sufficient to inspect cited claims, but independently reproducing all scientific calculations requires the original assessment bundle and inputs.

Timing includes cold evidence-packet construction, source hashing, and shared dispatch queues. It cannot support a claim that one model or workflow has intrinsically lower inference latency.

## Reproduce the evaluation

Use the frozen source/suite hashes and an authorized copy of the pinned assessment. Running this command makes paid API calls; set your own credential and budget limits first:

```bash
uv run python scripts/evaluate_agent_live.py \
  --assessment 20260907T153857705816Z \
  --evaluation-id your-unique-run \
  --suite evaluations/live_questions_v2.json --split holdout --trials 3 \
  --provider zai --model glm-5.3 --reasoning-effort low \
  --max-output-tokens 4096 --spend-ceiling 20 --per-task-ceiling 1 \
  --input-price 1.4 --output-price 4.4
```

Add `--method fixed_evidence` with a different evaluation ID for the baseline. Review results against the pinned sources; do not treat automatic report completion as task accuracy.

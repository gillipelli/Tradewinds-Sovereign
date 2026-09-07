# Verification of the event-study overhaul

- **28 tests passed**, including official-forecast parsing, year rollover, stale/future forecast rejection, La Niña carryover, source integrity and failed-run preservation.
- The full `tradewinds update --cached` event pipeline completed successfully, including the crop model, 12-country agriculture/fiscal models, post-2018 Bayesian holdouts and two fiscal-prior refits.
- A subsequent `tradewinds event --cached --no-refit` completed with the same model paths. SHA-256 comparisons showed exact reproduction of country risk, aggregate risk, crop impacts and sensitivity CSVs.
- Data-backed checks passed for paired revenue differences, joint cumulative quantiles, and exact zero losses when the intervention begins after the modeled horizon.
- The Streamlit dashboard passed initial rendering and Pacific-island/2027 selection checks without application exceptions. Local report links resolve.
- All **11 code cells** in `notebooks/02_event_assessment.ipynb` executed without errors. Ruff and `git diff --check` passed.

## Statistical interpretation

The full agriculture model uses 535 annual observations across 12 economies; the fiscal model uses 341. Both have R-hat below 1.002, bulk ESS above 1,900 and zero divergences. These checks support numerical convergence, not causal identification or predictive superiority.

Bayesian post-2018 conditional holdouts cover 60 agriculture observations and 66 fiscal observations. Their 90% predictive intervals cover 93.3% and 93.9% respectively. Short-history economies excluded from these holdout fits are not represented as validated. The rolling ridge challengers do not improve on their simpler baselines when climate/agricultural predictors are added. The report therefore does not claim established incremental forecasting skill.

Fiscal-prior sensitivity is material. Excluding Australia, the central two-year mean revenue loss is approximately $0.68bn; the narrower agriculture-coefficient prior produces about $0.20bn, while the wider prior produces about $0.91bn. These are constant 2025 dollars and separate model assumptions, not probability-weighted alternatives. This sensitivity is one reason the study is presented as a conditional event-risk assessment rather than a validated official budget forecast.

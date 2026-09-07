# Model card

**Intended use:** actuarial research, exploratory climate-economy analysis, portfolio stress testing,
and demonstrations of reproducible Bayesian modeling. Version 0.1.0; seed 20260906.

**Not established:** sovereign default probability, causal attribution, country-calibrated fiscal
capture, guaranteed forecast accuracy, operational underwriting readiness, or national budget forecasts.

## Implemented strengths

Official sources; immutable raw vintages; schema/unit checks; asymmetric warm/cold ENSO features;
lagged warm exposure; hierarchical country–crop responses; weather/economic controls;
chronological benchmarks; prior/posterior checks; explicit quality flags; uncertainty-aware
loss simulation; reproducible update orchestration and visible exclusions.

## Material limitations

- National weather averages conceal crop geography, irrigation, altitude and harvest calendars.
- Calendar-year features omit growing-season alignment; perennial effects are approximated by annual lags.
- Delayed cold effects are not in the principal five-feature model; inspect the sensitivity analysis.
- FAO estimated/imputed records can create artificial smoothness. Production flags do not guarantee
  comparable quality for harvested area or derived yield.
- Relatively few independent extreme ENSO events limit effective sample size and tail inference.
- Gaussian conditional yield errors and independent-observation likelihood can misrepresent outliers
  and common-year dependence. Computational diagnostics do not validate this assumption.
- Price and area models use separate regularized/bootstrap estimation. Cross-channel parameter
  dependence with the Bayesian yield posterior is not fully represented.
- Sparse local prices imply disclosed fixed-price scenarios; global prices are not substitutes.
- GDP, revenue, and crop-value gaps particularly restrict small-island fiscal coverage.
- Capture assumptions are uncalibrated. Imports, public spending, tax schedules, and broader macro
  transmission are excluded. Agricultural gross receipts are not government revenue or GDP.
- Historical revisions prevent genuine vintage backtesting before this project's snapshot history.
- Statistical ENSO outlooks are not official forecasts and have no demonstrated multi-step calibration.

## Promotion policy

MCMC diagnostics gate default research risk generation. Operational promotion requires additional
out-of-time validation, phase/event stability, fiscal calibration, geographic climate refinement,
and documented human model review. Daily refitting creates research candidates; it is not evidence
that every new model is more accurate. No model is automatically described as an operational champion.

## Next evidence-driven extensions

1. Crop-area-weighted ERA5 and independently validated CHIRPS rainfall, with versioned spatial weights.
2. Country-specific harvest calendars, soil moisture and drought indices; irrigation interactions.
3. Agricultural tax/export-levy and expenditure data from ministries, supported by trade exposure.
4. Common-year/hierarchical residual factors and broader prior/likelihood sensitivity.
5. Explicit ENSO forecast ensemble ingestion with release timestamps and probabilistic validation.
6. Gradient boosting or neural spatiotemporal models when data scale and validation justify them.

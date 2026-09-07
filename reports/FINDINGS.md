# Findings from the downloaded data

These are reproducible research findings from the source vintages listed in the report, not validated fiscal forecasts.

## Evidence coverage

- 63 eligible country–crop histories across 12 economies; 3,793 yield-growth modeling observations, 1963–2024.
- Historical weather includes monthly temperature, rainfall and mean maximum temperature through 2025.
- Only 47 crop exposures have matching 2024 monetary valuations. Papua New Guinea and Solomon Islands
  are absent from the valued portfolio because recent matching crop values are unavailable. Several other
  crops are also excluded; see `excluded_exposures.csv`.
- All 47 valued series support area-response regressions, but only 27 support local price-response estimation.
  The remaining 20 have a disclosed fixed-price sensitivity. Provider estimates and imputations remain flagged.

## Predictive performance is mixed

On 1,532 matching conditional hindcast observations from 2000–2024, the historical-mean baseline has
RMSE 0.1505, economic-controls ridge 0.1511, ENSO ridge 0.1520, and weather ridge 0.1521 (annual log yield growth).
ENSO ridge improves El Niño-year RMSE relative to economic-only ridge (0.1721 versus 0.1773), but this
does not translate into uniformly better performance across phases. The weather model has the lowest
aggregate interval score among these four benchmarks, by a small margin.

A separate Bayesian model fitted only through 2018 was evaluated on 374 observations in 2019–2024:

| Metric | Bayesian | Historical-mean baseline |
|---|---:|---:|
| RMSE | 0.1285 | 0.1339 |
| CRPS (lower is better) | 0.0586 | 0.0545 |
| 90% interval coverage | 94.4% | 92.8% |

The Bayesian model improves point error by 4.0%, but its CRPS is worse.
It has **not earned a claim of superior probabilistic forecasting**. These are revised-data conditional
holdouts using realized climate, not historical real-time forecasts. The latest crop label remains annual.

## Numerical model checks

The full model uses four chains with 1,000 retained draws each: maximum R-hat 1.0064, minimum bulk ESS
926.7, zero divergences, and minimum BFMI 0.786. The historical-cutoff fit also passes the configured gates.
In-sample posterior predictive dispersion is close to observed dispersion, but extreme growth frequencies
are somewhat underrepresented. Computational convergence does not remove model misspecification.

## Illustrative fiscal sensitivity, not an empirical loss forecast

Under the stylized super-stress feature vector and an **assumed** country-level triangular capture rate
of 1% / 5% / 10%, the included portfolio yields approximately:

- Mean incremental fiscal sensitivity: **$485 million**.
- Conditional 95th-percentile sensitivity: **$1,004 million**.
- Conditional 95% expected shortfall: **$1,185 million**.

These numbers use 2024 nominal gross crop values, incomplete country coverage, approximate cross-channel
uncertainty, and uncalibrated fiscal capture. They are not estimates of actual 2026 government losses,
sovereign default probabilities, or one-in-20-year return levels. Negative modeled losses in other
scenarios represent gains relative to the neutral comparison. Fiscal-rate and dependence sensitivity
tables show how structural assumptions affect the tails.

## What the project demonstrates

An auditable climate/agriculture panel, model comparisons across warm/cold/neutral conditions,
Bayesian partial pooling, chronological probabilistic evaluation, operational data vintages, and
transparent actuarial stress simulation. Its strongest current evidence is agricultural response analysis;
country-calibrated fiscal transmission and spatial crop-weather alignment remain necessary extensions.

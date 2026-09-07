# 2026–2027 El Niño sovereign revenue event study

This is the primary study. The earlier generic stress experiment is background research and is not the answer to the 2026–2027 question.

## Estimand and scope

For each economy and each year, estimate **government revenue without this event minus government revenue with this event**, conditional on the stated macroeconomic path. Report the joint two-year distribution, probability of loss, probability of exceeding a 1% revenue buffer, and positive-shortfall VaR95/ES95. Gains remain in the signed distribution. Aggregate risk uses common simulation draws; country quantiles are never summed.

Monetary outcomes use constant 2025 USD at a fixed GDP deflator/exchange-rate basis. This is a real revenue-capacity assessment, not a forecast of nominal tax receipts. The economic channel is agriculture, forestry and fishing value added. Crop yield/area/price results complement that aggregate; adding them would double count. The study does not estimate emergency expenditure, sovereign default, credit-rating changes, or total global economic losses.

## Dated climate event and counterfactual

The pipeline archives NOAA CPC's current RONI quantile outlook and companion advisory, verifies that their issue months agree, checks for stale/future issue dates, and filters observations by their assumed availability date. A centered three-month season is not treated as a monthly observation known at its center date.

The central counterfactual retains pre-event history, including preceding La Niña conditions, and sets centered RONI seasons from May 2026 onward to zero. May is the center of AMJ, the first season at the +0.5 threshold in the current observed sequence. This is a transparent intervention definition, not a claim that physical onset occurred on May 1. A June-onset sensitivity is reported.

CPC provides marginal quantiles, not a joint ensemble. We transform correlated normal scores through those quantiles, interpolating/extrapolating in normal-quantile space. Correlations come from expanding-origin, nine-step AR(2) forecast errors in historical RONI, with 5% identity shrinkage. This historical dependence proxy is not CPC's own forecast-error distribution. Independent and aligned marginals test dependence sensitivity.

The August 2026 official forecast ends at MAM 2027 (April center). Subsequent 2027 seasons use a conditional AR(2) continuation with resampled historical innovations. Neutral decay and La Niña rebound are alternative scenarios, not assigned probabilities. Later updates move the observation/forecast boundary and replace projected seasons with available observations. The annual features are positive RONI, negative RONI, excess above +1.5, and two years of warm/cold lags. Forecast exposure outside historical ranges is reported; combined unprecedented exposures can remain unsupported even when individual features are in range.

## Agricultural value added

Use World Bank WDI real agricultural value added (`NV.AGR.TOTL.KD`) and real GDP (`NY.GDP.MKTP.KD`). Real non-agricultural GDP is their difference; nonpositive differences are rejected. Constant-price component weights approximate growth contributions and need not equal current-price sector shares. Agriculture includes forestry and fishing.

Model annual log real agricultural growth with country intercepts and hierarchically pooled country coefficients for seven ENSO terms, trend, lagged agricultural growth and lagged non-agricultural growth. A Student-t likelihood with five degrees of freedom accommodates exceptional years; a shared latent annual effect reduces false independence across countries. It does not make the design causal. A minimum of 15 usable annual rows permits PNG to participate with explicitly weaker country evidence; partial pooling is especially consequential for short records.

Historical weather remains in the complementary crop-weather model: CRU monthly temperature, precipitation and monthly mean maximum temperature, converted to annual temperature/rainfall/extreme-month summaries. Contemporary weather is a mediator of ENSO effects, so putting it into the main total-association equation would change the estimand. Neither model resolves crop-level phenology or spatially weighted local drought.

## Empirically estimated fiscal transmission

IMF Fiscal Monitor `GGR_G01_GDP_PT` supplies general government revenue as a share of GDP. Multiplying that share by WDI real GDP constructs GDP-deflated real revenue. This combines IMF and WDI national accounts conventions; differences and revisions are part of measurement uncertainty, not independent observations. Revenue includes grants. Its growth is therefore not a pure tax response.

For country c and year t:

```
growth_revenue[c,t] = intercept[c]
  + beta_agri[c] * lag_agri_real_GDP_share[c,t] * growth_agri[c,t]
  + beta_nonagri[c] * (1-lag_agri_real_GDP_share[c,t]) * growth_nonagri[c,t]
  + beta_lag[c] * lag_growth_revenue[c,t]
  + beta_trend[c] * trend[t] + beta_pandemic[c] * pandemic[t]
  + shared_year_effect[t] + residual[c,t]
```

Country coefficients partially pool. The central agriculture coefficient has a Normal(0, 2) hyperprior and a HalfNormal(1) between-country scale. Negative effects are permitted. This is an estimated revenue elasticity on a growth contribution, **not an assumed agricultural tax-capture rate**. Non-agricultural growth is separately controlled. The old triangular 1%–10% gross-sales capture is absent from the primary event model.

This observational design cannot separate every policy change, transfer, mineral-revenue shock or endogenous agricultural response. Shared year effects and a heavy-tailed likelihood do not solve omitted-variable bias. A zero-direct-agriculture-coefficient sensitivity shows how results depend on the transmission channel. Structural and prior uncertainty beyond the fitted specification must not be interpreted as fully covered by its posterior interval.

## Two-year simulation

1. Start both cases from IMF's 2025 revenue estimate. Scale WDI real sector volumes into a common 2025 GDP-dollar basis. Where 2025 agriculture is missing, bridge from the latest value using the agricultural model and shared historical innovations; report those bridge years.
2. Use IMF WEO annual real GDP growth as an external proxy for non-agricultural growth, identical in both cases. It is a conditional scenario input, not an ENSO-free IMF forecast or an independently estimated non-agriculture response.
3. Sample country coefficients from the two posterior distributions. Their cross-model parameter covariance is not estimated. Draw consecutive historical residual years shared across countries and both equations; independently resample missing country residuals. Economic residuals are empirically resampled when exponentiating growth: the model does **not** exponentiate unbounded Student-t innovations, whose exponential moments do not exist.
4. Predict 2026 agricultural growth, update sector volumes, feed growth contributions to fiscal growth, and update revenue. Repeat for 2027 using each case's resulting 2026 levels, sector shares and lagged growth. This carries both agricultural and fiscal effects forward.
5. Difference the paired outcomes. The common non-climate shocks isolate the model-attributable incremental event effect while allowing uncertain economic conditions to change its scale. These are not total budget shortfalls against a fixed budget.

IMF nominal 2026/2027 GDP and revenue projections are exported separately. They may already reflect climate assumptions and are not counterfactual validation. Do not subtract fixed-2025-dollar losses directly from nominal projections. Country fiscal reporting calendars differ; the study uses annual country labels/calendar ENSO exposure and does not claim exact monthly or fiscal-year cash alignment.

## Crop-specific outputs

Reuse the verified hierarchical crop yield model trained on FAOSTAT production/area, with crop-level pooling and country-crop deviations. Apply event versus neutral features for each year. Recursion includes the fitted lag-yield derivative after reversing feature standardization. The 2027 level effect accumulates both years' growth effects.

Harvested area and local USD farmgate price channels use country-crop ridge regressions with shared-year bootstrap weights, economic controls and at least 20 annual observations. Unsupported price effects stay missing. Price, area and yield percentage effects combine multiplicatively where supported. The local USD price channel mixes domestic price and exchange-rate history and is exploratory. It does not supply an identified inflation response to the fiscal model.

Tonnage equivalents multiply the event yield ratio by the latest observed production scale. They are clearly labelled fixed-area/anchor equivalents, not forecasts of planted area or absolute 2026/2027 production. Commodity output, prices, gross receipts and value added are different quantities.

## Validation and release discipline

- Four-chain sampling: R-hat <= 1.01, bulk ESS >= 400, no divergences, BFMI >= 0.3. Failed models cannot publish the event assessment.
- Expanding-year ridge challengers compare macro models with simpler economic baselines using RMSE, CRPS and 90% interval coverage.
- Separate Bayesian models stop at 2018 and assess later held-out outcomes. Some short-history countries cannot enter these fits. These are conditional hindcasts using realized inputs and revised data, not recreated real-time forecasts or end-to-end two-year validation.
- Forecast quantiles, year rollover, stale/future source rejection, preserved La Niña history, zero-intervention identity and joint-loss accounting have explicit checks.
- Sensitivities cover late-2027 evolution, dependence, onset and direct fiscal transmission. No subset of favorable results is selected as the headline.

Daily refresh does not imply daily official crop data. Climate inputs advance with published observations/forecasts. IMF projections revise at their own release frequency. Economic fitting uses data through assessment-year minus two by default to allow one revision year, while newer estimates can anchor scenarios. That cutoff advances automatically. Immutable source hashes, training fingerprints, frozen report data, run locks and atomic state publication support reproducibility. Failures preserve the previous successful report rather than replacing it with stale-model output.

Fiscal-prior sensitivity refits the full fiscal model with agriculture hyperprior standard deviations 0.5 and 4 (and corresponding between-country scales), alongside the central scale of 2. Each alternative must pass the same convergence checks. These are actual refits, not re-labelled central draws or an arbitrary tax-rate range.

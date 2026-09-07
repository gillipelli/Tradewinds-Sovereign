"""Conditional actuarial stress tests, with explicit fiscal assumptions.

Agricultural values are exact primary-crop FAOSTAT values. No processed prices
are multiplied by raw fruit/cane/paddy production. No default-risk claim is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import atomic_json, config
from .model import ENSO, Design, posterior_arrays, predict

SCENARIOS = {
    "neutral": [0, 0, 0, 0, 0],
    "el_nino": [1, 0, 0, 0, 0],
    "la_nina": [0, 0, 0, 1, 0],
    # A stylized sustained path, NOT an official NOAA event definition or forecast.
    "super_stress": [2, 1, 0, 0, 0.5],
    "post_super": [0, 2, 1, 0, 0],
}


def loss_metrics(loss, buffer=None, attachment=0, limit=np.inf, loading=0.2):
    loss = np.asarray(loss, dtype=float)
    if loss.ndim != 1 or not len(loss) or not np.isfinite(loss).all():
        raise ValueError("Loss samples must be a nonempty finite vector")
    if min(attachment, limit, loading) < 0:
        raise ValueError("Insurance terms must be nonnegative")
    q95, q99 = np.quantile(loss, [0.95, 0.99])

    # Integrate the empirical quantile function, including a fractional boundary draw.
    def es(alpha):
        s = np.sort(loss)[::-1]
        mass = len(s) * (1 - alpha)
        k = int(np.floor(mass))
        return float((s[:k].sum() + (mass - k) * s[min(k, len(s) - 1)]) / mass)

    payment = np.minimum(np.maximum(loss - attachment, 0), limit)
    return {
        "expected_net_loss_usd": float(loss.mean()),
        "probability_loss": float((loss > 0).mean()),
        "var95_usd": float(q95),
        "var99_usd": float(q99),
        "es95_usd": es(0.95),
        "es99_usd": es(0.99),
        "probability_buffer_exceeded": None if buffer is None else float((loss > buffer).mean()),
        "illustrative_layer_pure_premium_usd": float(payment.mean()),
        "illustrative_layer_loaded_premium_usd": float(payment.mean() * (1 + loading)),
    }


def exposures(root, series):
    qv = pd.read_csv(root / "data/processed/qv.csv")
    qv = qv[qv.Element.eq("Gross Production Value (current thousand US$)")].copy()
    if not qv.Unit.eq("1000 USD").all():
        raise ValueError("Unexpected valuation units")
    qv["series"] = qv.country + ":" + qv.crop
    if qv.duplicated(["series", "year"]).any():
        raise ValueError("Duplicate primary-crop valuations")
    year = int(qv[qv.series.isin(series)].year.max())
    qv = qv[qv.year.eq(year) & qv.series.isin(series) & qv.value.gt(0)].copy()
    qv["baseline_usd"] = qv.value * 1000
    return qv[["series", "country", "crop", "year", "baseline_usd", "Flag"]].rename(
        columns={"year": "valuation_year", "Flag": "valuation_flag"}
    )


def auxiliary_responses(root, panel, selected, nboot, rng):
    """Bootstrap area and local farmgate-price regressions using shared calendar years.

    Coefficient uncertainty is approximate (ridge/bootstrap), not a joint Bayesian
    posterior with yield. This limitation is written to every risk run's metadata.
    """
    pp = pd.read_csv(root / "data/processed/pp.csv")
    pp = pp[
        pp.Element.eq("Producer Price (USD/tonne)") & pp.Months.eq("Annual value") & pp.value.gt(0)
    ].copy()
    pp["series"] = pp.country + ":" + pp.crop
    if pp.duplicated(["series", "year"]).any():
        raise ValueError("Duplicate annual producer prices")
    pp = pp.sort_values(["series", "year"])
    pp["price_growth"] = pp.groupby("series").value.transform(lambda x: np.log(x).diff())
    pp.loc[pp.groupby("series").year.diff().ne(1), "price_growth"] = np.nan
    merged = panel.merge(
        pp[["series", "year", "price_growth"]], on=["series", "year"], how="left", validate="1:1"
    )
    years = np.arange(int(panel.year.min()), int(panel.year.max()) + 1)
    # Same resampling weights for every country, commodity and auxiliary outcome.
    weights = rng.multinomial(len(years), np.ones(len(years)) / len(years), size=nboot)
    result = {k: np.zeros((nboot, len(selected), len(ENSO))) for k in ["area_growth", "price_growth"]}
    residuals = {}
    support = []
    for j, name in enumerate(selected):
        for outcome in result:
            sub = merged[merged.series.eq(name)].dropna(subset=[outcome])
            row = {"series": name, "outcome": outcome, "n": len(sub), "estimated": len(sub) >= 20}
            support.append(row)
            if len(sub) < 20:
                # Explicit fixed-price sensitivity for unsupported local prices.
                residuals[(name, outcome)] = pd.Series(dtype=float)
                continue
            controls = sub[["trend", "lag_gdp_growth", "lag_oil_growth", "lag_fertilizer_growth"]].copy()
            controls = controls.fillna(controls.median().fillna(0))
            controls = (controls - controls.mean()) / controls.std().replace(0, 1).fillna(1)
            x = np.column_stack([np.ones(len(sub)), sub[ENSO].to_numpy(), controls.to_numpy()])
            y = sub[outcome].to_numpy()
            penalty = np.eye(x.shape[1]) * 5
            penalty[0, 0] = 0.01
            coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            residuals[(name, outcome)] = pd.Series(y - x @ coef, index=sub.year)
            for b in range(nboot):
                w = weights[b, sub.year.to_numpy() - years[0]]
                cb = np.linalg.solve(x.T @ (x * w[:, None]) + penalty, x.T @ (y * w))
                result[outcome][b, j] = cb[1 : 1 + len(ENSO)]
    return result, residuals, pd.DataFrame(support)


def fiscal_denominators(root, countries, valuation_year):
    wdi = pd.read_csv(root / "data/processed/wdi.csv")
    rows = []
    for country in countries:
        wide = wdi[wdi.country.eq(country)].pivot(index="year", columns="indicator", values="value")
        candidates = (
            wide.dropna(subset=["gdp_usd", "revenue_pct_gdp"])
            if "revenue_pct_gdp" in wide
            else pd.DataFrame()
        )
        candidates = candidates.loc[
            (candidates.index <= valuation_year) & (candidates.index >= valuation_year - 2)
        ]
        if candidates.empty:
            rows.append(
                {
                    "country": country,
                    "fiscal_year": None,
                    "government_revenue_usd": None,
                    "reason": "No matched GDP/revenue observations within two years of valuation",
                }
            )
        else:
            year = int(candidates.index.max())
            row = candidates.loc[year]
            rows.append(
                {
                    "country": country,
                    "fiscal_year": year,
                    "government_revenue_usd": row.gdp_usd * row.revenue_pct_gdp / 100,
                    "reason": "Matched-year WDI revenue excluding grants; government coverage follows source",
                }
            )
    return pd.DataFrame(rows)


def simulate(root: Path, model_dir: Path, out: Path, allow_unvalidated=False):
    info = json.loads((model_dir / "fit.json").read_text())
    if not info["diagnostics_pass"] and not allow_unvalidated:
        raise ValueError("MCMC diagnostics failed: risk publication blocked; inspect fit.json")
    if info["spec"] != "total":
        raise ValueError("Fiscal stress uses the total-association model; weather is a mediator")
    cfg = config(root)
    rng = np.random.default_rng(cfg["seed"])
    d = Design.load(model_dir / "design.json")
    p = pd.read_csv(model_dir / "training.csv")
    exp = exposures(root, d.series).sort_values("series").reset_index(drop=True)
    names = exp.series.tolist()
    if exp.empty:
        raise ValueError("No recent, matching primary-crop valuation data")
    out.mkdir(parents=True, exist_ok=True)
    excluded = sorted(set(d.series) - set(names))
    exp.to_csv(out / "exposures.csv", index=False)
    pd.DataFrame({"series": excluded, "reason": "No valuation in common latest baseline year"}).to_csv(
        out / "excluded_exposures.csv", index=False
    )
    n = cfg["risk"]["simulations"]
    a = posterior_arrays(model_dir)
    idx = rng.integers(len(a["beta"]), size=n)
    groups = [d.series.index(s) for s in names]
    beta = a["beta"][idx][:, groups, :]
    aux, residual, support = auxiliary_responses(root, p, names, 200, rng)
    support.to_csv(out / "auxiliary_support.csv", index=False)
    aux_idx = rng.integers(200, size=n)
    area_beta = aux["area_growth"][aux_idx]
    price_beta = aux["price_growth"][aux_idx]
    # Preserve empirical cross-country/crop/outcome shock dependence by sampling one
    # calendar year per simulation. Missing residuals get independent observed draws,
    # explicitly counted below; dependence sensitivity also includes a common shock.
    means = predict(model_dir, p, predictive=False).mean(0)
    yield_resid = p.assign(residual=p.yield_growth - means).pivot(
        index="year", columns="series", values="residual"
    )
    years = np.sort(p.year.unique())
    block_years = rng.choice(years, n)
    noise = np.zeros((n, len(names)))
    missing_count = 0
    for j, name in enumerate(names):
        for resid in [
            yield_resid[name].dropna(),
            residual[(name, "area_growth")],
            residual[(name, "price_growth")],
        ]:
            if len(resid) < 2:
                continue
            sampled = resid.reindex(block_years).to_numpy()
            missing = ~np.isfinite(sampled)
            missing_count += int(missing.sum())
            sampled[missing] = rng.choice(resid.to_numpy(), missing.sum())
            noise[:, j] += sampled - resid.mean()
    # Center the finite empirical log shock so mean neutral receipts equal baseline.
    multiplicative = np.exp(noise)
    multiplicative /= multiplicative.mean(0)
    baseline = exp.baseline_usd.to_numpy()
    neutral = baseline * multiplicative
    terms = cfg["risk"]
    countries = sorted(exp.country.unique())
    # A single fiscal capture draw per country, shared across its commodities.
    capture = rng.triangular(
        terms["fiscal_capture_low"],
        terms["fiscal_capture_mode"],
        terms["fiscal_capture_high"],
        (n, len(countries)),
    )
    capture_by_crop = capture[:, [countries.index(c) for c in exp.country]]
    denominators = fiscal_denominators(root, countries, int(exp.valuation_year.max()))
    denominators.to_csv(out / "fiscal_denominators.csv", index=False)
    metrics, contributions, simulation_rows, response_rows = [], [], [], []
    scenarios = SCENARIOS.copy()
    if (out / "current_enso_vectors.npy").exists():
        current = np.load(out / "current_enso_vectors.npy")
        scenarios["current_statistical_outlook"] = current[rng.integers(len(current), size=n)]
    for scenario, vector in scenarios.items():
        v = np.asarray(vector)

        def response(b):
            return b @ v if v.ndim == 1 else np.einsum("sij,sj->si", b, v)

        yield_change = response(beta)
        log_change = yield_change + response(area_beta) + response(price_beta)
        if not np.isfinite(log_change).all() or np.max(np.abs(log_change)) > 20:
            raise ValueError("Explosive extrapolation: inspect response parameters; no silent clipping")
        receipts = neutral * np.exp(log_change)
        fiscal_increment = (neutral - receipts) * capture_by_crop
        budget_loss = (baseline - receipts) * capture_by_crop
        if scenario == "super_stress":
            sensitivity_rng = np.random.default_rng(cfg["seed"] + 1)
            rate_rows = []
            for rate in [
                terms["fiscal_capture_low"],
                terms["fiscal_capture_mode"],
                terms["fiscal_capture_high"],
            ]:
                rate_rows.append(
                    {"fixed_capture_rate": rate, **loss_metrics((neutral - receipts).sum(1) * rate)}
                )
            pd.DataFrame(rate_rows).to_csv(out / "fiscal_capture_sensitivity.csv", index=False)
            independent = np.column_stack(
                [sensitivity_rng.permutation(fiscal_increment[:, j]) for j in range(len(exp))]
            )
            aligned = np.sort(fiscal_increment, axis=0)
            pd.DataFrame(
                [
                    {"dependence": label, **loss_metrics(draws.sum(1))}
                    for label, draws in [
                        ("shared_empirical_year", fiscal_increment),
                        ("independent_marginals", independent),
                        ("rank_aligned_stress", aligned),
                    ]
                ]
            ).to_csv(out / "dependence_sensitivity.csv", index=False)
        for country in countries + ["PORTFOLIO"]:
            mask = np.ones(len(exp), bool) if country == "PORTFOLIO" else exp.country.eq(country).to_numpy()
            denominator = denominators[denominators.country.eq(country)].government_revenue_usd
            denom = None if denominator.empty or pd.isna(denominator.iloc[0]) else float(denominator.iloc[0])
            for basis, samples in [
                ("climate_increment", fiscal_increment[:, mask].sum(1)),
                ("baseline_shortfall", budget_loss[:, mask].sum(1)),
            ]:
                met = loss_metrics(
                    samples,
                    None if denom is None else denom * terms["budget_buffer_fraction"],
                    terms["insurance_attachment_usd"],
                    terms["insurance_limit_usd"],
                    terms["insurance_loading"],
                )
                metrics.append({"scenario": scenario, "country": country, "basis": basis, **met})
                simulation_rows.append(
                    pd.DataFrame(
                        {
                            "simulation": np.arange(n),
                            "scenario": scenario,
                            "country": country,
                            "basis": basis,
                            "loss_usd": samples,
                        }
                    )
                )
        portfolio = fiscal_increment.sum(1)
        tail = portfolio >= np.quantile(portfolio, 0.95)
        for j, row in exp.iterrows():
            contributions.append(
                {
                    "scenario": scenario,
                    "series": row.series,
                    "mean_increment_usd": float(fiscal_increment[:, j].mean()),
                    "portfolio_tail_contribution_usd": float(fiscal_increment[tail, j].mean()),
                }
            )
            response_rows.append(
                {
                    "scenario": scenario,
                    "series": row.series,
                    "yield_response_median_pct": float(np.median(np.expm1(yield_change[:, j])) * 100),
                    "yield_response_lower90_pct": float(
                        np.quantile(np.expm1(yield_change[:, j]), 0.05) * 100
                    ),
                    "yield_response_upper90_pct": float(
                        np.quantile(np.expm1(yield_change[:, j]), 0.95) * 100
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "risk_metrics.csv", index=False)
    pd.DataFrame(contributions).to_csv(out / "tail_contributions.csv", index=False)
    pd.DataFrame(response_rows).to_csv(out / "yield_scenarios.csv", index=False)
    pd.concat(simulation_rows).to_csv(out / "loss_draws.csv.gz", index=False, compression="gzip")
    # Explicit alternative fiscal-capture sensitivity; baseline-shortfall risk scales linearly.
    stress = next(
        x
        for x in simulation_rows
        if x.scenario.iloc[0] == "super_stress"
        and x.country.iloc[0] == "PORTFOLIO"
        and x.basis.iloc[0] == "climate_increment"
    )
    stress.to_csv(out / "portfolio_stress_draws.csv", index=False)
    meta = {
        "model": str(model_dir),
        "diagnostics_pass": info["diagnostics_pass"],
        "simulations": n,
        "valuation_year": int(exp.valuation_year.max()),
        "n_exposures": len(exp),
        "excluded_series": excluded,
        "fiscal_capture": "Uncalibrated triangular sensitivity assumption per country",
        "settings": terms,
        "scenario_features": dict(zip(ENSO, SCENARIOS["super_stress"])),
        "assumptions": [
            "Fixed baseline exposures in nominal valuation-year USD; not a calendar-year fiscal forecast",
            "Area and price coefficient draws use approximate shared-year bootstrap, separate from yield posterior",
            "Unsupported local producer prices use a disclosed fixed-price sensitivity, not an estimated response",
            "Residual dependence is empirical; missing cells use independent marginal draws",
            "Net fiscal losses allow commodity gains to offset losses; government spending and import costs excluded",
            "Tax capture assumptions are not national average tax rates or calibrated agricultural effective rates",
            "Scenario percentiles are conditional; no unconditional event probability or return period assigned",
        ],
        "missing_residual_draws": missing_count,
        "inference": "Predictive associations and fiscal stress sensitivities; not causal or sovereign-default estimates",
    }
    atomic_json(out / "risk.json", meta)
    return meta

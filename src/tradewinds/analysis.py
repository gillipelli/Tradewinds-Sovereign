"""Chronological benchmarks, phase comparisons, price channels and diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .data import annual_enso, config
from .model import ENSO, WEATHER, Design, modeling_panel, predict


def ridge_matrix(design, frame, variant):
    group, enso, controls = design.transform(frame)
    dummy = np.eye(len(design.series))[group]
    x = [dummy, controls]
    if variant != "economic_only":
        x.extend([enso, (dummy[:, :, None] * enso[:, None, :]).reshape(len(frame), -1)])
    return np.column_stack(x)


def phase(enso):
    """Annual exposure buckets, deliberately not official five-season event labels."""
    return np.select([enso.warm >= 0.5, enso.cold >= 0.5], ["El Nino", "La Nina"], default="Neutral/mixed")


def backtest(root: Path, out: Path):
    panel = modeling_panel(root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Revised historical data with realized same-year weather/ENSO: conditional hindcasts.
    # Purge the most recent two years to mimic crop-publication latency conservatively.
    for year in range(max(2000, int(panel.year.min()) + 25), int(panel.year.max()) + 1):
        train = panel[panel.year <= year - 3]
        test = panel[panel.year == year]
        counts = train.groupby("series").size()
        valid = counts[counts >= 20].index
        train, test = train[train.series.isin(valid)], test[test.series.isin(valid)]
        if test.empty:
            continue
        for variant in ["historical_mean", "economic_only", "enso", "weather"]:
            spec = "weather" if variant == "weather" else "total"
            tr, te = train.copy(), test.copy()
            if variant == "weather":
                tr, te = tr.dropna(subset=WEATHER), te.dropna(subset=WEATHER)
                te = te[te.series.isin(tr.series)]
            if te.empty:
                continue
            if variant == "historical_mean":
                means = tr.groupby("series").yield_growth.mean()
                pred = te.series.map(means).to_numpy()
                residual = tr.yield_growth - tr.series.map(means)
            else:
                design = Design(tr, spec)
                x = ridge_matrix(design, tr, variant)
                estimator = Ridge(alpha=10).fit(x, tr.yield_growth)
                pred = estimator.predict(ridge_matrix(design, te, variant))
                residual = tr.yield_growth - estimator.predict(x)
            # Training-only empirical residual intervals; calibration assessed out of time.
            low, high = np.quantile(residual, [0.05, 0.95])
            for j, (_, row) in enumerate(te.iterrows()):
                rows.append(
                    {
                        "year": year,
                        "series": row.series,
                        "country": row.country,
                        "crop": row.crop,
                        "model": variant,
                        "actual": row.yield_growth,
                        "prediction": pred[j],
                        "lower90": pred[j] + low,
                        "upper90": pred[j] + high,
                        "phase": phase(pd.DataFrame([row]))[0],
                        "evaluation": "revised_data_conditional_hindcast",
                    }
                )
    forecasts = pd.DataFrame(rows)
    forecasts["error"] = forecasts.actual - forecasts.prediction
    forecasts["covered90"] = forecasts.actual.between(forecasts.lower90, forecasts.upper90)
    forecasts["interval_score90"] = (
        forecasts.upper90
        - forecasts.lower90
        + 20 * (forecasts.lower90 - forecasts.actual).clip(lower=0)
        + 20 * (forecasts.actual - forecasts.upper90).clip(lower=0)
    )
    forecasts.to_csv(out / "hindcasts.csv", index=False)
    metrics = (
        forecasts.groupby(["model", "phase"])
        .agg(
            n=("error", "size"),
            mae=("error", lambda x: x.abs().mean()),
            rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
            coverage90=("covered90", "mean"),
            interval_score90=("interval_score90", "mean"),
        )
        .reset_index()
    )
    metrics.to_csv(out / "hindcast_metrics.csv", index=False)
    # Fair comparison restricts all models to identical country-crop-year observations.
    common = forecasts.groupby(["year", "series"]).model.nunique()
    common = common[common == 4].reset_index()[["year", "series"]]
    matched = forecasts.merge(common, on=["year", "series"], validate="m:1")
    matched.groupby("model").agg(
        n=("error", "size"),
        mae=("error", lambda x: x.abs().mean()),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        coverage90=("covered90", "mean"),
        interval_score90=("interval_score90", "mean"),
    ).to_csv(out / "matched_metrics.csv")
    return metrics


def descriptive(root: Path, out: Path):
    panel = modeling_panel(root)
    panel["phase"] = phase(panel)
    panel.groupby(["crop", "phase"]).agg(
        n=("yield_growth", "size"),
        yield_growth=("yield_growth", "mean"),
        area_growth=("area_growth", "mean"),
        temperature_c=("temperature_c", "mean"),
        annual_rain_mm=("annual_rain_mm", "mean"),
    ).to_csv(out / "phase_comparisons.csv")
    # Global price response is separately estimated; no raw-crop/processed-price multiplication.
    prices = pd.read_csv(root / "data/processed/pink.csv", parse_dates=["period"])
    prices["year"] = prices.period.dt.year
    names = ["Palm oil", "Coconut oil", "Sugar, world", "Rice, Thai 5%", "Maize", "Cocoa", "Coffee, Robusta"]
    price = (
        prices[prices.commodity.isin(names)]
        .groupby(["commodity", "year"])
        .agg(price=("price", "mean"), n=("price", "size"))
        .reset_index()
    )
    price = price[price.n == 12].sort_values(["commodity", "year"])
    price["growth"] = price.groupby("commodity").price.transform(lambda x: np.log(x).diff())
    price.loc[price.groupby("commodity").year.diff().ne(1), "growth"] = np.nan
    enso = annual_enso(pd.read_csv(root / f"data/processed/{config(root)['enso_index']}.csv"))
    price = price.merge(enso, on="year").dropna()
    price["phase"] = phase(price)
    price.to_csv(out / "price_panel.csv", index=False)
    # Within-series growth difference with whole-year block bootstrap prevents treating
    # countries as independent climate events. Descriptive, not a causal estimator.
    rng = np.random.default_rng(config(root)["seed"])
    effects = []
    for crop, subset in panel.groupby("crop"):
        byyear = subset.groupby(["year", "phase"]).yield_growth.mean().reset_index()
        for name in ["El Nino", "La Nina"]:
            a = byyear[byyear.phase.eq(name)].yield_growth.to_numpy()
            b = byyear[byyear.phase.eq("Neutral/mixed")].yield_growth.to_numpy()
            if min(len(a), len(b)) < 5:
                continue
            draws = rng.choice(a, (2000, len(a))).mean(1) - rng.choice(b, (2000, len(b))).mean(1)
            effects.append(
                {
                    "crop": crop,
                    "phase": name,
                    "difference": float(a.mean() - b.mean()),
                    "lower90": float(np.quantile(draws, 0.05)),
                    "upper90": float(np.quantile(draws, 0.95)),
                    "phase_years": len(a),
                    "neutral_years": len(b),
                }
            )
    pd.DataFrame(effects).to_csv(out / "phase_contrasts.csv", index=False)


def bayesian_holdout(root: Path, model_dir: Path, out: Path):
    import json

    cutoff = json.loads((model_dir / "fit.json").read_text())["training_end"]
    panel = modeling_panel(root)
    test = panel[panel.year > cutoff].copy()
    design = Design.load(model_dir / "design.json")
    test = test[test.series.isin(design.series)]
    if test.empty:
        raise ValueError("No observations after training cutoff; fit a historical cutoff first")
    draws = predict(model_dir, test)
    test["prediction"] = draws.mean(0)
    test["lower90"], test["upper90"] = np.quantile(draws, [0.05, 0.95], axis=0)
    test["covered90"] = test.yield_growth.between(test.lower90, test.upper90)
    # CRPS from posterior predictive draws, using sorted-sample identity O(S log S).
    s = np.sort(draws, axis=0)
    n = len(s)
    test["crps"] = (
        np.mean(np.abs(draws - test.yield_growth.to_numpy()), axis=0)
        - ((2 * np.arange(1, n + 1) - n - 1)[:, None] * s).sum(0) / n**2
    )
    test.to_csv(out / "bayesian_holdout.csv", index=False)
    training = pd.read_csv(model_dir / "training.csv")
    mean = training.groupby("series").yield_growth.mean()
    baseline_prediction = test.series.map(mean).to_numpy()
    residual = training.yield_growth - training.series.map(mean)
    baseline_draws = baseline_prediction + np.random.default_rng(42).choice(residual, (2000, len(test)))
    sorted_baseline = np.sort(baseline_draws, axis=0)
    baseline_crps = (
        np.mean(np.abs(baseline_draws - test.yield_growth.to_numpy()), axis=0)
        - ((2 * np.arange(1, 2001) - 2001)[:, None] * sorted_baseline).sum(0) / 2000**2
    )
    low, high = np.quantile(baseline_draws, [0.05, 0.95], axis=0)
    return {
        "n": len(test),
        "rmse": float(np.sqrt(np.mean((test.prediction - test.yield_growth) ** 2))),
        "coverage90": float(test.covered90.mean()),
        "crps": float(test.crps.mean()),
        "baseline_rmse": float(np.sqrt(np.mean((baseline_prediction - test.yield_growth) ** 2))),
        "baseline_crps": float(baseline_crps.mean()),
        "baseline_coverage90": float(test.yield_growth.between(low, high).mean()),
        "training_cutoff": cutoff,
        "interpretation": "Revised-data conditional holdout, not real-time forecasting",
    }


def robustness(root: Path, out: Path):
    """Response stability under event exclusion, flags, index and cold-lag choices."""
    panel = modeling_panel(root)
    variants = {"all_records_roni": panel}
    oni = annual_enso(pd.read_csv(root / "data/processed/oni.csv"))
    variants["oni_index"] = panel.drop(columns=ENSO + ["peak"]).merge(oni, on="year", validate="m:1")
    current_official = panel.production_flag.eq("A") & panel.area_flag.eq("A")
    previous_official = current_official.groupby(panel.series).shift().fillna(False).astype(bool)
    variants["official_current_and_previous"] = panel[current_official & previous_official]
    for event, years in {
        "1982_83": (1981, 1984),
        "1997_98": (1996, 1999),
        "2015_16": (2014, 2017),
        "2023_24": (2022, 2024),
    }.items():
        variants[f"exclude_{event}_and_neighbors"] = panel[~panel.year.between(*years)]
    rows = []
    reference = panel.groupby("series", sort=True).tail(1).copy()
    reference[ENSO] = 0
    for label, train in variants.items():
        counts = train.groupby("series").size()
        train = train[train.series.isin(counts[counts >= 20].index)]
        if train.empty:
            continue
        design = Design(train)
        est = Ridge(alpha=10).fit(ridge_matrix(design, train, "enso"), train.yield_growth)
        neutral = reference[reference.series.isin(design.series)].copy()
        warm = neutral.copy()
        warm[ENSO] = [2, 1, 0, 0, 0.5]
        effects = est.predict(ridge_matrix(design, warm, "enso")) - est.predict(
            ridge_matrix(design, neutral, "enso")
        )
        for series, effect in zip(neutral.series, effects):
            rows.append(
                {
                    "variant": label,
                    "series": series,
                    "super_response_log_growth": effect,
                    "training_rows": len(train),
                }
            )
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
    # A separate diagnostic checks delayed cold relationships without claiming these
    # are already part of the principal Bayesian model.
    annual = annual_enso(pd.read_csv(root / "data/processed/roni.csv")).set_index("year")
    for lag in [1, 2]:
        cold = annual.cold.copy().rename(f"cold_lag{lag}")
        cold.index += lag
        annual = annual.join(cold)
    extended = panel.merge(annual[["cold_lag1", "cold_lag2"]], left_on="year", right_index=True).dropna(
        subset=["cold_lag1", "cold_lag2"]
    )
    cold_rows = []
    for crop, frame in extended.groupby("crop"):
        x = frame[ENSO + ["cold_lag1", "cold_lag2"]]
        est = Ridge(alpha=10).fit(x, frame.yield_growth)
        for name, coef in zip(x.columns, est.coef_):
            cold_rows.append(
                {
                    "crop": crop,
                    "feature": name,
                    "coefficient": coef,
                    "interpretation": "Pooled diagnostic association; not a causal estimate",
                }
            )
    pd.DataFrame(cold_rows).to_csv(out / "cold_lag_sensitivity.csv", index=False)
    # Global benchmark price response with whole-year resampling, separately valued.
    prices = pd.read_csv(out / "price_panel.csv")
    rng = np.random.default_rng(config(root)["seed"])
    price_rows = []
    for commodity, frame in prices.groupby("commodity"):
        x = np.column_stack([np.ones(len(frame)), frame[ENSO].to_numpy()])
        penalty = np.eye(x.shape[1]) * 5
        penalty[0, 0] = 0.001
        betas = []
        for _ in range(500):
            idx = rng.integers(len(frame), size=len(frame))
            b = np.linalg.solve(x[idx].T @ x[idx] + penalty, x[idx].T @ frame.growth.to_numpy()[idx])
            betas.append(b[1:] @ np.array([2, 1, 0, 0, 0.5]))
        price_rows.append(
            {
                "commodity": commodity,
                "n_years": len(frame),
                "super_response_median_log_growth": float(np.median(betas)),
                "lower90": float(np.quantile(betas, 0.05)),
                "upper90": float(np.quantile(betas, 0.95)),
                "interpretation": "Unadjusted global price association; separate from local farmgate risk model",
            }
        )
    pd.DataFrame(price_rows).to_csv(out / "price_associations.csv", index=False)

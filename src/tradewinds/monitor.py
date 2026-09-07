"""Daily observation refresh and an explicitly statistical ENSO outlook.

The AR(2)/persistence forecast is a transparent benchmark, not an official CPC
forecast. Overlapping three-month index observations are not daily temperatures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import atomic_json, config


def ar2_fit(values):
    x = np.column_stack([np.ones(len(values) - 2), values[1:-1], values[:-2]])
    coef = np.linalg.lstsq(x, values[2:], rcond=None)[0]
    residual = values[2:] - x @ coef
    # A statistical benchmark must not generate an explosive recurrence.
    roots = np.linalg.eigvals([[coef[1], coef[2]], [1, 0]])
    if np.max(np.abs(roots)) >= 1:
        return np.array([0, 1, 0]), np.diff(values)
    return coef, residual


def outlook(root: Path, out: Path, as_of=None, n=2000):
    cfg = config(root)
    now = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC")).tz_localize(None)
    df = pd.read_csv(
        root / f"data/processed/{cfg['enso_index']}.csv", parse_dates=["period", "assumed_available_at"]
    )
    df = df[df.assumed_available_at <= now].sort_values("period")
    if len(df) < 120:
        raise ValueError("Insufficient ENSO history for statistical outlook")
    values = df.enso.to_numpy()
    # One-step rolling benchmark. Multi-step calibration is NOT implied by this score.
    errors = {"ar2": [], "persistence": []}
    for i in range(max(120, len(values) - 240), len(values)):
        coef, _ = ar2_fit(values[:i])
        errors["ar2"].append(values[i] - np.dot(coef, [1, values[i - 1], values[i - 2]]))
        errors["persistence"].append(values[i] - values[i - 1])
    scores = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errors.items()}
    selected = min(scores, key=scores.get)
    coef, residual = ar2_fit(values) if selected == "ar2" else (np.array([0, 1, 0]), np.diff(values))
    residual -= residual.mean()
    rng = np.random.default_rng(cfg["seed"])
    last_period = df.period.max()
    periods = pd.date_range(last_period + pd.DateOffset(months=1), periods=18, freq="MS")
    paths = np.zeros((n, len(periods)))
    prev, older = np.full(n, values[-1]), np.full(n, values[-2])
    for t in range(len(periods)):
        if t % 3 == 0:
            block = rng.integers(0, len(residual) - 2, n)
        nxt = coef[0] + coef[1] * prev + coef[2] * older + residual[block + t % 3]
        paths[:, t] = nxt
        older, prev = prev, nxt
    summary = pd.DataFrame(
        {
            "period": periods,
            "mean": paths.mean(0),
            "lower90": np.quantile(paths, 0.05, axis=0),
            "upper90": np.quantile(paths, 0.95, axis=0),
            "probability_season_above_2": (paths >= 2).mean(0),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "enso_outlook.csv", index=False)
    # Annual exposure vectors for the current center-year; observed seasons remain fixed.
    current_year = now.year
    combined = []
    for month in range(1, 13):
        period = pd.Timestamp(current_year, month, 1)
        actual = df.loc[df.period.eq(period), "enso"]
        if len(actual):
            combined.append(np.full(n, actual.iloc[0]))
        elif period in periods:
            combined.append(paths[:, periods.get_loc(period)])
        else:
            raise ValueError("Data too stale to form current-year exposure")
    annual = np.column_stack(combined)
    vectors = np.zeros((n, 5))
    vectors[:, 0] = np.maximum(annual, 0).mean(1)
    for lag in [1, 2]:
        historical = df[df.year.eq(current_year - lag)]
        if len(historical) != 12:
            raise ValueError("Missing full-year lagged ENSO exposure")
        vectors[:, lag] = historical.enso.clip(lower=0).mean()
    vectors[:, 3] = np.maximum(-annual, 0).mean(1)
    vectors[:, 4] = np.maximum(annual - 1.5, 0).mean(1)
    np.save(out / "current_enso_vectors.npy", vectors)
    meta = {
        "as_of": str(now.date()),
        "last_observed_center_month": str(last_period.date()),
        "last_observed_index": float(values[-1]),
        "index": cfg["enso_index"],
        "selected_model": selected,
        "one_step_rolling_rmse": scores,
        "forecast_paths": n,
        "current_exposure_year": current_year,
        "status": "Statistical monitoring benchmark; not an official NOAA forecast",
        "limitations": "Revised history; fixed AR coefficients; empirical residual blocks; multi-step probabilities not yet calibrated",
    }
    atomic_json(out / "monitor.json", meta)
    return meta

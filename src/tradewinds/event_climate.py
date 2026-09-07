"""Official marginal forecasts, an empirical temporal copula, and explicit extensions."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm, rankdata

from .data import atomic_json, config
from .event_data import QUANTILES

FEATURES = ['warm', 'warm_lag1', 'warm_lag2', 'cold', 'extreme', 'cold_lag1', 'cold_lag2']


def quantile_transform(z, values):
    """Linear interpolation in normal-score space; extrapolate, do not clip tails."""
    return interp1d(norm.ppf(QUANTILES), values, fill_value='extrapolate')(z)


def ar_model(history):
    y = np.asarray(history, float)
    x = np.column_stack([np.ones(len(y)-2), y[1:-1], y[:-2]])
    coef = np.linalg.lstsq(x, y[2:], rcond=None)[0]
    return coef, y[2:] - x @ coef


def error_copula(history, horizon=9):
    """Expanding-origin multistep errors, never fit to future validation observations."""
    y = np.asarray(history, float)
    errors = []
    for t in range(360, len(y)-horizon+1, 3):
        coef, _ = ar_model(y[:t])
        a, b = y[t-2:t]
        pred = []
        for _ in range(horizon):
            a, b = b, coef @ [1, b, a]
            pred.append(b)
        errors.append(y[t:t+horizon] - pred)
    errors = np.asarray(errors)
    if len(errors) < 30:
        raise ValueError('Insufficient history to estimate forecast-error dependence')
    scores = norm.ppf(np.column_stack([(rankdata(errors[:, j])-.5)/len(errors)
                                      for j in range(horizon)]))
    corr = np.corrcoef(scores.T)
    # Shrink for positive definiteness and avoid asserting perfectly known dependence.
    return .95*corr + .05*np.eye(horizon), len(errors)


def annual_features(paths, dates, year):
    def phase(y, positive):
        x = paths[:, dates.year == y]
        if x.shape[1] != 12:
            raise ValueError(f'Incomplete annual ENSO path: {y}')
        return np.maximum(x if positive else -x, 0).mean(axis=1)
    current = paths[:, dates.year == year]
    return np.column_stack([phase(year, True), phase(year-1, True), phase(year-2, True),
                            phase(year, False), np.maximum(current-1.5, 0).mean(axis=1),
                            phase(year-1, False), phase(year-2, False)])


def climate_paths(root, out, extension='ar2', dependence='empirical', onset=None, n=None):
    cfg = config(root)
    ecfg = cfg['event']
    n = n or cfg['risk']['simulations']
    rng = np.random.default_rng(cfg['seed'])
    onset = pd.Timestamp(onset or ecfg['counterfactual_start'])
    asof = pd.Timestamp(ecfg['as_of']) if ecfg.get('as_of') else pd.Timestamp.now(tz='UTC')
    if asof.tzinfo is not None:
        asof = asof.tz_localize(None)
    obs = pd.read_csv(root/'data/processed/roni.csv', parse_dates=['period', 'assumed_available_at'])
    obs = obs[obs.assumed_available_at <= asof].sort_values('period')
    forecast = pd.read_csv(root/'data/event/cpc_quantiles.csv', parse_dates=['period'])
    meta = json.loads((root/'data/event/snapshot.json').read_text())
    if pd.Timestamp(meta['forecast_issue_date']) > asof:
        raise ValueError('Forecast was issued after assessment date')
    if (asof-pd.Timestamp(meta['forecast_issue_date'])).days > ecfg['max_forecast_age_days']:
        raise ValueError('Official forecast is stale; refresh event inputs')
    start = pd.Timestamp(ecfg['years'][0]-2, 1, 1)
    end = pd.Timestamp(ecfg['years'][-1], 12, 1)
    dates = pd.date_range(start, end, freq='MS')
    paths = np.full((n, len(dates)), np.nan)
    sources = np.full(len(dates), 'model_extension', dtype=object)
    for row in obs.itertuples():
        if row.period in dates:
            i = dates.get_loc(row.period)
            paths[:, i] = row.enso
            sources[i] = 'observed_centered_season'
    # Estimate only from data strictly before this forecast issue month.
    history = obs.loc[obs.period < pd.Timestamp(meta['forecast_issue']), 'enso'].to_numpy()
    corr, origins = error_copula(history, len(forecast))
    if dependence == 'independent':
        corr = np.eye(len(forecast))
    elif dependence == 'comonotonic':
        corr = np.ones_like(corr)
    z = rng.multivariate_normal(np.zeros(len(forecast)), corr, size=n, method='svd')
    for j, row in enumerate(forecast.itertuples()):
        if row.period in dates:
            i = dates.get_loc(row.period)
            if sources[i] != 'observed_centered_season':
                paths[:, i] = quantile_transform(z[:, j], forecast.iloc[j, 2:].to_numpy(float))
                sources[i] = 'CPC_marginal_with_modeled_dependence'
    coef, residuals = ar_model(history)
    last_forecast = forecast.period.max()
    for i, date in enumerate(dates):
        if np.isfinite(paths[:, i]).all():
            continue
        if date <= last_forecast or i < 2:
            raise ValueError(f'Unexplained gap between observations and official outlook: {date}')
        if extension == 'neutral_decay':
            paths[:, i] = paths[:, i-1] * np.exp(-1/3)
        elif extension == 'la_nina_rebound':
            paths[:, i] = -.8 + (paths[:, i-1]+.8)*np.exp(-1/3)
        elif extension == 'ar2':
            paths[:, i] = coef[0]+coef[1]*paths[:, i-1]+coef[2]*paths[:, i-2]+rng.choice(residuals, n)
        else:
            raise ValueError(extension)
    neutral = paths.copy()
    neutral[:, dates >= onset] = 0
    features = {year: annual_features(paths, dates, year) for year in ecfg['years']}
    counter = {year: annual_features(neutral, dates, year) for year in ecfg['years']}
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({'period': dates, 'source': sources,
                            'p05': np.quantile(paths, .05, axis=0), 'p50': np.median(paths, axis=0),
                            'p95': np.quantile(paths, .95, axis=0), 'neutral': np.median(neutral, axis=0)})
    summary.to_csv(out/'event_climate.csv', index=False)
    metadata = {**meta, 'as_of': str(asof.date()), 'counterfactual_start': str(onset.date()),
                'extension': extension, 'dependence': dependence, 'copula_training_origins': origins,
                'latest_observed_center': str(obs.period.max().date()),
                'no_event_definition': 'Keep pre-onset history; set all subsequent centered RONI seasons to zero.',
                'official_forecasts_are_marginal_not_joint': True,
                'extension_is_not_NOAA_forecast': True}
    atomic_json(out/'event_climate_metadata.json', metadata)
    return features, counter, paths, dates, metadata

"""Partial-pooling Bayesian distributed-lag yield response model."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .data import atomic_json, config

ENSO = ["warm", "warm_lag1", "warm_lag2", "cold", "extreme"]
CONTROLS = ["trend", "lag_yield_growth", "lag_gdp_growth", "lag_oil_growth", "lag_fertilizer_growth"]
WEATHER = ["temperature_c", "log_rain", "driest_month_mm", "hottest_month_tmax_c"]


def modeling_panel(root: Path) -> pd.DataFrame:
    p = pd.read_csv(root / "data/processed/panel.csv")
    p = p[p.eligible & p.yield_growth.notna() & p.lag_yield_growth.notna()].copy()
    prices = pd.read_csv(root / "data/processed/pink.csv", parse_dates=["period"])
    prices["year"] = prices.period.dt.year
    for label, match in [("oil", "Crude oil, average"), ("fertilizer", "Urea")]:
        select = prices[prices.commodity.eq(match)]
        annual = select.groupby("year").agg(price=("price", "mean"), n=("period", "nunique"))
        annual = annual[annual.n.eq(12)].reindex(range(int(prices.year.min()), int(prices.year.max()) + 1))
        series = np.log(annual.price).diff().rename(f"lag_{label}_growth")
        series.index += 1
        p = p.merge(series, left_on="year", right_index=True, how="left", validate="m:1")
    return p.sort_values(["series", "year"]).reset_index(drop=True)


class Design:
    """Fit all transforms on training data; unknown series are explicitly rejected."""

    def __init__(self, train: pd.DataFrame, spec="total"):
        self.spec = spec
        self.series = sorted(train.series.unique())
        self.crops = sorted(train.crop.unique())
        self.crop_index = [self.crops.index(s.split(":")[1]) for s in self.series]
        self.features = CONTROLS + (WEATHER if spec == "weather" else [])
        self.country_means = train.groupby("country")[WEATHER].mean().to_dict() if spec == "weather" else {}
        values = self._center(train)
        self.median = values.median().fillna(0).to_dict()
        filled = values.fillna(self.median)
        self.mean = filled.mean().to_dict()
        self.std = filled.std().replace(0, 1).fillna(1).to_dict()
        self.names = self.features + [f"missing_{x}" for x in self.features]

    def _center(self, frame):
        v = frame[self.features].copy()
        if self.spec == "weather":
            for col in WEATHER:
                v[col] -= frame.country.map(self.country_means[col])
        return v

    def transform(self, frame):
        unknown = set(frame.series) - set(self.series)
        if unknown:
            raise ValueError(f"Untrained series: {sorted(unknown)}")
        raw = self._center(frame)
        controls = ((raw.fillna(self.median) - pd.Series(self.mean)) / pd.Series(self.std)).to_numpy()
        controls = np.column_stack([controls, raw.isna().astype(float).to_numpy()])
        return (
            frame.series.map({s: i for i, s in enumerate(self.series)}).to_numpy(),
            frame[ENSO].to_numpy(),
            controls,
        )

    def save(self, path):
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.__dict__.update(json.loads(Path(path).read_text()))
        return obj


def fit(root: Path, out: Path, spec="total", draws=None, tune=None, chains=None, cutoff=None):
    cfg = config(root)
    panel = modeling_panel(root)
    if cutoff is not None:
        panel = panel[panel.year <= cutoff]
    if spec == "weather":
        panel = panel.dropna(subset=WEATHER)
    design = Design(panel, spec)
    group, enso, controls = design.transform(panel)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        out / "training_provenance.json",
        {
            "config": cfg,
            "training_sha256": hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest(),
            "model_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_snapshot": json.loads((root / "data/processed/snapshot.json").read_text()),
        },
    )
    settings = cfg["sampling"].copy()
    for k, v in [("draws", draws), ("tune", tune), ("chains", chains)]:
        if v is not None:
            settings[k] = v
    coords = {
        "series": design.series,
        "crop": design.crops,
        "enso_feature": ENSO,
        "control": design.names,
        "obs": np.arange(len(panel)),
    }
    with pm.Model(coords=coords):
        alpha = pm.Normal("alpha", 0, 0.05, dims="series")
        beta_crop = pm.Normal("beta_crop", 0, 0.08, dims=("crop", "enso_feature"))
        tau = pm.HalfNormal("tau", 0.04, dims="enso_feature")
        z = pm.Normal("z", 0, 1, dims=("series", "enso_feature"))
        beta = pm.Deterministic(
            "beta", beta_crop[design.crop_index] + z * tau, dims=("series", "enso_feature")
        )
        gamma = pm.Normal("gamma", 0, 0.05, dims="control")
        sigma = pm.HalfNormal("sigma", 0.20, dims="series")
        mu = alpha[group] + (beta[group] * enso).sum(axis=1) + pm.math.dot(controls, gamma)
        pm.Normal("yield_growth", mu, sigma[group], observed=panel.yield_growth.to_numpy(), dims="obs")
        prior = pm.sample_prior_predictive(draws=150, random_seed=cfg["seed"])
        idata = pm.sample(
            **settings,
            cores=min(settings["chains"], 4),
            random_seed=cfg["seed"],
            idata_kwargs={"log_likelihood": True},
        )
    idata.to_netcdf(out / "posterior.nc")
    prior.to_netcdf(out / "prior.nc")
    design.save(out / "design.json")
    panel.to_csv(out / "training.csv", index=False)
    summary = az.summary(idata, var_names=["alpha", "beta_crop", "beta", "tau", "gamma", "sigma"])
    summary.to_csv(out / "parameters.csv")
    checked = ["alpha", "beta_crop", "beta", "tau", "gamma", "sigma"]
    max_rhat = float(az.rhat(idata, var_names=checked).to_array().max())
    min_ess = float(az.ess(idata, var_names=checked, method="bulk").to_array().min())
    divergences = int(idata.sample_stats.diverging.sum())
    bfmi_min = float(np.min(az.bfmi(idata)))
    passed = bool(
        np.isfinite(max_rhat) and max_rhat <= 1.01 and min_ess >= 400 and divergences == 0 and bfmi_min >= 0.3
    )
    info = {
        "spec": spec,
        "n_observations": len(panel),
        "n_series": len(design.series),
        "training_start": int(panel.year.min()),
        "training_end": int(panel.year.max()),
        "settings": settings,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "divergences": divergences,
        "min_bfmi": bfmi_min,
        "diagnostics_pass": passed,
        "status": "research_candidate" if passed else "diagnostics_failed",
        "causal_identification": False,
    }
    atomic_json(out / "fit.json", info)
    predictions = predict(out, panel, cfg["seed"])
    lower, upper = np.quantile(predictions, [0.05, 0.95], axis=0)
    y = panel.yield_growth.to_numpy()
    prior_y = prior.prior_predictive.yield_growth.values.ravel()
    atomic_json(
        out / "predictive_checks.json",
        {
            "observed_growth_sd": float(y.std()),
            "posterior_predictive_sd": float(predictions.std()),
            "in_sample_coverage90": float(((y >= lower) & (y <= upper)).mean()),
            "prior_predictive_growth_sd": float(prior_y.std()),
            "observed_abs_growth_above_0_5": float((np.abs(y) > 0.5).mean()),
            "predicted_abs_growth_above_0_5": float((np.abs(predictions) > 0.5).mean()),
            "interpretation": "In-sample model checking; use chronological holdout for predictive validation",
        },
    )
    return info


def posterior_arrays(out: Path):
    p = az.from_netcdf(out / "posterior.nc").posterior.stack(sample=("chain", "draw"))
    return {v: p[v].transpose("sample", ...).values for v in ["alpha", "beta", "gamma", "sigma"]}


def predict(out: Path, frame: pd.DataFrame, seed=42, predictive=True):
    d = Design.load(out / "design.json")
    g, e, c = d.transform(frame)
    a = posterior_arrays(out)
    mu = a["alpha"][:, g] + np.einsum("sij,ij->si", a["beta"][:, g, :], e) + a["gamma"] @ c.T
    if predictive:
        mu += np.random.default_rng(seed).normal(size=mu.shape) * a["sigma"][:, g]
    return mu

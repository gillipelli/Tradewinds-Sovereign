"""Portable HTML report plus publication-friendly PNG figures."""

from html import escape
from pathlib import Path
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = """
body{font:16px/1.6 system-ui,sans-serif;color:#173044;background:#f5f7fa;margin:0}
main{max-width:1180px;margin:auto;padding:44px 28px}h1{font-size:40px;line-height:1.1}
h2{margin-top:42px;color:#116466}p{max-width:900px}.eyebrow{color:#117c82;letter-spacing:2px}
.card{background:white;padding:24px;margin:20px 0;border-radius:12px;border:1px solid #dae3e8}
.notice{border-left:5px solid #d99326;padding:14px 20px;background:#fff7e7}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
.scroll{overflow:auto}img{width:100%;max-width:1100px}a{color:#116466}small{color:#566}
"""


def render(root: Path, out: Path, model_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    # Freeze dashboard inputs with each report. A later failed update may replace
    # working processed files; the last successful dashboard must remain coherent.
    data_snapshot = out / "data"
    data_snapshot.mkdir(exist_ok=True)
    for name in ["coverage.csv", "roni.csv", "oni.csv", "weather.csv"]:
        shutil.copy2(root / "data/processed" / name, data_snapshot / name)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    coverage = pd.read_csv(root / "data/processed/coverage.csv")
    panel = pd.read_csv(root / "data/processed/panel.csv")
    snapshot = json.loads((root / "data/processed/snapshot.json").read_text())
    sections = []

    def table(frame):
        return (
            '<div class="scroll">'
            + frame.to_html(index=False, float_format=lambda x: f"{x:,.3f}", escape=True)
            + "</div>"
        )

    def section(title, body):
        sections.append(f'<section class="card"><h2>{escape(title)}</h2>{body}</section>')

    section(
        "Evidence and scope",
        f"<p>{coverage.eligible.sum()} eligible country–crop histories across {coverage.country.nunique()} economies. "
        f"Crop observations span {panel.year.min()}–{panel.year.max()}. Eligibility requires at least 30 annual growth observations and recent production data. "
        "Eligibility does not imply all observations are official measurements.</p>" + table(coverage),
    )
    matrix = coverage.pivot(index="country", columns="crop", values="official_share")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Share of production observations flagged official (FAOSTAT A)")
    fig.colorbar(im, ax=ax, label="Official share")
    fig.tight_layout()
    fig.savefig(figdir / "coverage.png", dpi=180)
    plt.close(fig)
    section(
        "Data quality",
        '<img src="figures/coverage.png" alt="Official observation shares by crop and country">',
    )
    if (out / "monitor.json").exists():
        monitor = json.loads((out / "monitor.json").read_text())
        forecast = pd.read_csv(out / "enso_outlook.csv", parse_dates=["period"])
        hist = pd.read_csv(root / f"data/processed/{monitor['index']}.csv", parse_dates=["period"])
        fig, ax = plt.subplots(figsize=(11, 4))
        hist = hist.tail(72)
        ax.plot(hist.period, hist.enso, color="#116466", label="Observed/revised index")
        ax.plot(forecast.period, forecast["mean"], color="#dc8433", label="Statistical forecast")
        ax.fill_between(
            forecast.period,
            forecast.lower90,
            forecast.upper90,
            color="#dc8433",
            alpha=0.2,
            label="90% empirical forecast interval",
        )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_ylabel(monitor["index"].upper() + " (°C)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "outlook.png", dpi=180)
        plt.close(fig)
        section(
            "ENSO monitoring",
            '<p class="notice">This is an AR/persistence statistical benchmark, not an official NOAA outlook. '
            "Its multi-step probabilities are not yet calibrated.</p>"
            + f"<p>As of {escape(monitor['as_of'])}; latest observed center month: "
            f"{escape(monitor['last_observed_center_month'])}. Selected model: {escape(monitor['selected_model'])}.</p>"
            '<img src="figures/outlook.png" alt="Historical and statistical forecast ENSO index">',
        )
    if (out / "matched_metrics.csv").exists():
        metrics = pd.read_csv(out / "matched_metrics.csv")
        section(
            "Out-of-time model comparisons",
            "<p>Revised-data conditional hindcasts using realized same-year climate and lagged economic predictors. "
            "Training excludes the two immediately preceding years. These are not historical real-time forecasts. "
            "Scores use matching observations across models; lower error and interval score are better.</p>"
            + table(metrics),
        )
        m = pd.read_csv(out / "hindcast_metrics.csv")
        fig, ax = plt.subplots(figsize=(10, 4))
        m.pivot(index="phase", columns="model", values="rmse").plot.bar(ax=ax, rot=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, fontsize=9)
        ax.set_ylabel("RMSE: annual log yield growth")
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(figdir / "validation.png", dpi=180)
        plt.close(fig)
        section(
            "Performance by climate phase",
            '<img src="figures/validation.png" alt="Prediction error by climate phase">' + table(m),
        )
    if (out / "phase_contrasts.csv").exists():
        phase = pd.read_csv(out / "phase_contrasts.csv")
        section(
            "Historical phase contrasts",
            "<p>Differences in mean log yield growth relative to neutral/mixed years. "
            "Intervals resample calendar-year aggregates; these descriptive comparisons are not causal estimates.</p>"
            + table(phase),
        )
    if (out / "robustness.csv").exists():
        robust = pd.read_csv(out / "robustness.csv")
        summary = (
            robust.groupby("variant")
            .agg(
                n_series=("series", "nunique"), median_super_response=("super_response_log_growth", "median")
            )
            .reset_index()
        )
        section(
            "Response stability",
            "<p>Ridge sensitivity to official-record filtering, ONI/RONI choice, "
            "and exclusion of major events with neighboring years. Supported series differ across variants; "
            "inspect the per-series CSV before interpreting median differences.</p>" + table(summary),
        )
    if (out / "price_associations.csv").exists():
        section(
            "Global commodity price associations",
            "<p>Whole-year bootstrap intervals for global benchmark "
            "price-growth associations. These estimates do not substitute for local farmgate prices.</p>"
            + table(pd.read_csv(out / "price_associations.csv")),
        )
    if (out / "bayesian_weather_comparison.csv").exists():
        comparison = pd.read_csv(out / "bayesian_weather_comparison.csv")
        comparison["crop"] = comparison.series.str.split(":").str[1]
        section(
            "Weather-controlled Bayesian sensitivity",
            "<p>Median group-level super-stress log-yield responses, summarized by crop. "
            "The weather-controlled specification conditions on mediators; differences are not "
            "a causal decomposition. Both fits use the same 3,793 observations.</p>"
            + table(comparison.groupby(["model", "crop"]).median_log_growth.median().reset_index()),
        )
    if (out / "bayesian_holdout_metrics.json").exists():
        holdout = json.loads((out / "bayesian_holdout_metrics.json").read_text())
        section(
            "Bayesian historical holdout",
            "<p>Fit through the stated cutoff and evaluate subsequent outcomes. "
            "Historical-mean predictions and residual distributions use the same training cutoff.</p>"
            + table(pd.DataFrame([holdout])),
        )
    if model_dir and (model_dir / "fit.json").exists():
        info = json.loads((model_dir / "fit.json").read_text())
        section(
            "Bayesian diagnostics",
            "<p>Passing MCMC diagnostics establishes computational adequacy, not predictive superiority or fiscal calibration.</p>"
            + table(pd.DataFrame([info]).drop(columns="settings")),
        )
        if (model_dir / "predictive_checks.json").exists():
            checks = json.loads((model_dir / "predictive_checks.json").read_text())
            section("Prior and posterior predictive checks", table(pd.DataFrame([checks])))
    if (out / "risk_metrics.csv").exists():
        risk = pd.read_csv(out / "risk_metrics.csv")
        meta = json.loads((out / "risk.json").read_text())
        selected = risk[risk.country.eq("PORTFOLIO") & risk.basis.eq("climate_increment")]
        section(
            "Conditional fiscal stress sensitivity",
            '<p class="notice">Fiscal capture is assumed, not calibrated. '
            "These are partial agricultural fiscal exposures, not total sovereign losses, budget forecasts, or default probabilities.</p>"
            f"<p>Values use {meta['valuation_year']} nominal USD exposures; {meta['n_exposures']} crop exposures have usable valuations. "
            "Missing countries and crops are excluded and listed below. Negative losses denote gains. "
            "Percentiles are conditional on each scenario, not unconditional return periods.</p>"
            + table(selected),
        )
        country = risk[
            risk.scenario.eq("super_stress")
            & risk.basis.eq("climate_increment")
            & risk.country.ne("PORTFOLIO")
        ]
        section("Country exposure under the stylized super stress", table(country))
        for name, title in [
            ("fiscal_capture_sensitivity.csv", "Fiscal capture sensitivity"),
            ("dependence_sensitivity.csv", "Dependence sensitivity"),
        ]:
            if (out / name).exists():
                section(
                    title,
                    "<p>Alternative assumptions under the same stylized super stress.</p>"
                    + table(pd.read_csv(out / name)),
                )
        section("Valuation exclusions", table(pd.read_csv(out / "excluded_exposures.csv")))
        section(
            "Local price and area evidence",
            "<p>Local-price regressions require at least 20 growth observations. "
            "Unsupported prices remain fixed; no world processed-price substitution is used.</p>"
            + table(pd.read_csv(out / "auxiliary_support.csv")),
        )
        section(
            "Risk assumptions",
            "<ul>" + "".join(f"<li>{escape(x)}</li>" for x in meta["assumptions"]) + "</ul>",
        )
        draws = pd.read_csv(out / "portfolio_stress_draws.csv").loss_usd / 1e6
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(draws, bins=70, color="#116466", alpha=0.85)
        ax.axvline(np.quantile(draws, 0.95), color="#dc8433", label="95th percentile")
        ax.set_xlabel("Incremental fiscal sensitivity, USD millions")
        ax.set_ylabel("Simulations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "loss_distribution.png", dpi=180)
        plt.close(fig)
        section(
            "Portfolio loss distribution",
            '<img src="figures/loss_distribution.png" alt="Conditional fiscal sensitivity distribution">',
        )
    source_rows = [
        {"source": key, "retrieved_at": val["retrieved_at"], "sha256": val["sha256"], "url": val["url"]}
        for key, val in snapshot["sources"].items()
    ]
    section(
        "Source provenance",
        "<p>Raw files are preserved by SHA-256. Retrieval timestamps are not historical release dates. "
        "Full source URLs and checksums follow.</p>" + table(pd.DataFrame(source_rows)),
    )
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += "<title>Tradewinds Sovereign — ENSO agriculture risk</title><style>" + STYLE + "</style><main>"
    page += '<p class="eyebrow">TRADEWINDS / SOVEREIGN</p><h1>Pacific agriculture under ENSO stress</h1>'
    page += "<p>Evidence, uncertainty, and conditional fiscal exposure across El Niño, neutral, and La Niña climates.</p>"
    page += '<p class="notice">Research edition. Model associations and fiscal assumptions must be read alongside the results.</p>'
    page += "".join(sections) + "</main></html>"
    (out / "index.html").write_text(page)
    return out / "index.html"

"""Run with: uv run streamlit run app.py"""

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from tradewinds.data import config

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Tradewinds Sovereign", page_icon="🌊", layout="wide")
st.title("Tradewinds Sovereign")
st.caption("Pacific agriculture · ENSO response · Conditional fiscal exposure")
state = ROOT / "artifacts/state.json"
REPORTS = Path(json.loads(state.read_text())["report_dir"]) if state.exists() else ROOT / "reports"
DATA = REPORTS / "data" if (REPORTS / "data").exists() else ROOT / "data/processed"
st.warning(
    "Research model. Fiscal capture is a sensitivity assumption, not a calibrated government-revenue forecast."
)


def read(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def download(df, name):
    st.download_button("Download this table", df.to_csv(index=False), name, "text/csv", key=name)


coverage_path = DATA / "coverage.csv"
if not coverage_path.exists():
    st.info("Run `uv run tradewinds ingest` and `uv run tradewinds build` to load official data.")
    st.stop()
coverage = pd.read_csv(coverage_path)
col1, col2, col3 = st.columns(3)
col1.metric("Economies in production panel", coverage.country.nunique())
col2.metric("Eligible crop histories", int(coverage.eligible.sum()))
col3.metric("Latest harvest year", int(coverage.last_year.max()))
tabs = st.tabs(["Evidence", "Climate", "Model validation", "Fiscal stress", "Methods"])
with tabs[0]:
    selected = st.multiselect(
        "Economies", sorted(coverage.country.unique()), default=sorted(coverage.country.unique())
    )
    filtered = coverage[coverage.country.isin(selected)]
    st.plotly_chart(
        px.density_heatmap(
            filtered,
            x="crop",
            y="country",
            z="official_share",
            histfunc="avg",
            range_color=[0, 1],
            color_continuous_scale="Teal",
            title="Share of production records flagged official",
        ),
        width="stretch",
    )
    st.dataframe(filtered, hide_index=True)
    st.caption(
        "Estimated and imputed FAOSTAT observations are retained and flagged; they are not independent measurements."
    )
    download(filtered, "coverage.csv")
with tabs[1]:
    mon = REPORTS / "monitor.json"
    index_name = json.loads(mon.read_text())["index"] if mon.exists() else config(ROOT)["enso_index"]
    hist = pd.read_csv(DATA / f"{index_name}.csv")
    st.plotly_chart(
        px.line(hist, x="period", y="enso", title=f"NOAA historical {index_name.upper()} (revisable)"),
        width="stretch",
    )
    mon = REPORTS / "monitor.json"
    if mon.exists():
        meta = json.loads(mon.read_text())
        st.write(
            f"Monitoring as of {meta['as_of']}; latest observed season center: {meta['last_observed_center_month']}."
        )
        st.info(meta["status"] + ". " + meta["limitations"])
        forecast = read("enso_outlook.csv")
        st.plotly_chart(
            px.line(forecast, x="period", y=["mean", "lower90", "upper90"], title="Statistical ENSO outlook"),
            width="stretch",
        )
    weather = pd.read_csv(DATA / "weather.csv")
    nation = st.selectbox("Weather history", sorted(weather.country.unique()))
    variable = st.selectbox(
        "Weather variable",
        ["tas", "pr", "tasmax"],
        format_func=lambda x: {
            "tas": "Mean temperature (°C)",
            "pr": "Monthly precipitation (mm)",
            "tasmax": "Mean daily maximum temperature (°C)",
        }[x],
    )
    st.plotly_chart(
        px.line(weather[(weather.country == nation) & (weather.variable == variable)], x="period", y="value"),
        width="stretch",
    )
    st.caption(
        "CRU national land averages; not crop-weighted. Monthly records do not identify daily heatwave or flood events."
    )
with tabs[2]:
    metrics = read("matched_metrics.csv")
    if metrics is not None:
        st.dataframe(metrics, hide_index=True)
        detail = read("hindcast_metrics.csv")
        st.plotly_chart(px.bar(detail, x="phase", y="rmse", color="model", barmode="group"), width="stretch")
        st.caption(
            "Conditional hindcasts with revised data and realized climate; not point-in-time forecast claims. Lower RMSE and interval score are better."
        )
    else:
        st.info("Historical evaluation is not available yet.")
    heldout = REPORTS / "bayesian_holdout_metrics.json"
    if not heldout.exists() and (ROOT / "reports/bayesian_holdout_metrics.json").exists():
        heldout = ROOT / "reports/bayesian_holdout_metrics.json"
        st.caption(
            "Initial research holdout shown below; this historical fit is not rerun by daily monitoring."
        )
    if heldout.exists():
        st.json(json.loads(heldout.read_text()))
with tabs[3]:
    risk = read("risk_metrics.csv")
    if risk is None:
        st.info(
            "Risk outputs are unavailable until a model passes computational diagnostics and simulation completes."
        )
    else:
        meta = json.loads((REPORTS / "risk.json").read_text())
        st.caption(
            f"{meta['valuation_year']} nominal USD baseline; {meta['n_exposures']} valued crop exposures. Portfolio covers the included exposures only."
        )
        scenario = st.selectbox("Climate scenario", risk.scenario.unique())
        basis = st.radio("Loss definition", ["climate_increment", "baseline_shortfall"], horizontal=True)
        st.caption(
            "Climate increment compares paired scenario and neutral receipts; baseline shortfall also includes background variability."
        )
        display = risk[(risk.scenario == scenario) & (risk.basis == basis)]
        st.plotly_chart(
            px.bar(
                display[display.country != "PORTFOLIO"],
                x="country",
                y=["expected_net_loss_usd", "var95_usd", "es95_usd"],
                barmode="group",
            ),
            width="stretch",
        )
        st.dataframe(display, hide_index=True)
        download(display, "risk_selection.csv")
        with st.expander("Assumptions and missing exposures"):
            st.write(meta)
            st.dataframe(read("excluded_exposures.csv"), hide_index=True)
            st.dataframe(read("auxiliary_support.csv"), hide_index=True)
with tabs[4]:
    st.markdown("""
Yield response uses a Bayesian model with crop-level pooling and country–crop departures,
separate warm/cold exposure terms, delayed responses, and lagged economic controls.
Weather models investigate the local climate pathway on a common sample.

Risk simulation combines posterior yield responses with bootstrapped harvested-area and
local producer-price responses. Shared historical years preserve observed shock dependence
where records overlap. Tax capture uncertainty is an explicitly uncalibrated assumption.

MCMC convergence is not predictive validation. Climate attribution is associational.
No default probability, unconditional return period, or guarantee of accuracy is claimed.
    """)
    st.markdown(
        "Sources: [NOAA](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/), "
        "[FAOSTAT](https://www.fao.org/faostat/en/), [World Bank prices](https://www.worldbank.org/en/research/commodity-markets), "
        "[World Bank climate](https://climateknowledgeportal.worldbank.org/download-data)."
    )

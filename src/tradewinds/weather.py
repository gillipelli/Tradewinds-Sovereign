"""Historical observed climate from CRU via World Bank CCKP."""

import json

import numpy as np
import pandas as pd


def ingest_weather(store, cfg, refresh=False):
    spec = cfg["weather"]
    countries = ",".join(cfg["countries"])
    records = []
    for variable in spec["variables"]:
        url = (
            f"https://cckpapi.worldbank.org/cckp/v1/{spec['collection']}_timeseries_{variable}"
            f"_timeseries_monthly_{spec['period']}_mean_historical_{spec['model']}_{spec['version']}_mean/"
            f"{countries}?_format=json"
        )
        body = json.loads(store.fetch(f"weather_{variable}", url, refresh).read_text())
        if body.get("metadata", {}).get("status") != "success" or not isinstance(body.get("data"), dict):
            raise ValueError(f"CCKP schema/API error: {str(body)[:500]}")
        for country, values in body["data"].items():
            if country not in cfg["countries"]:
                continue
            for period, value in values.items():
                if value is not None:
                    records.append(
                        {"country": country, "period": period, "variable": variable, "value": value}
                    )
    out = pd.DataFrame(records)
    if out.empty or out.duplicated(["country", "period", "variable"]).any():
        raise ValueError("Empty or duplicated weather data")
    out["value"] = pd.to_numeric(out.value, errors="raise")
    out["period"] = pd.to_datetime(out.period, format="%Y-%m")
    if (out.loc[out.variable.eq("pr"), "value"] < 0).any():
        raise ValueError("Negative rainfall / missing-value sentinel")
    if (out.loc[out.variable.ne("pr"), "value"].abs() > 65).any():
        raise ValueError("Temperature not in degrees Celsius")
    return out


def annual_weather(weather):
    """No full-sample climatology fitting; anomalies are fitted within training folds."""
    w = weather.copy()
    w["period"] = pd.to_datetime(w.period)
    w["year"] = w.period.dt.year
    counts = w.groupby(["country", "year", "variable"]).period.nunique()
    complete = counts[counts.eq(12)].reset_index()[["country", "year", "variable"]]
    w = w.merge(complete, on=["country", "year", "variable"], validate="m:1")
    mean = w.groupby(["country", "year", "variable"]).value.mean().unstack("variable")
    mean = mean.reindex(columns=["tas", "pr", "tasmax"])
    mean = mean.rename(columns={"tas": "temperature_c", "tasmax": "mean_tmax_c", "pr": "monthly_rain_mm"})
    mean["annual_rain_mm"] = mean.pop("monthly_rain_mm") * 12
    rain = w[w.variable.eq("pr")].groupby(["country", "year"]).value.min()
    mean["driest_month_mm"] = rain
    # Monthly aggregate heat proxy; not an estimate of daily heatwave counts.
    heat = w[w.variable.eq("tasmax")].groupby(["country", "year"]).value.max()
    mean["hottest_month_tmax_c"] = heat
    mean["log_rain"] = np.log1p(mean.annual_rain_mm)
    return mean.reset_index()

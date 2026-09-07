import numpy as np
import pandas as pd
import pytest

from tradewinds.data import annual_enso, season_dates, SEASONS
from tradewinds.model import Design, ENSO, CONTROLS, WEATHER
from tradewinds.risk import loss_metrics
from tradewinds.weather import annual_weather


def climate_years():
    return pd.DataFrame(
        [(year, season, 1.0) for year in range(2000, 2005) for season in SEASONS],
        columns=["year", "season", "enso"],
    )


def test_centered_season_is_not_available_in_center_month():
    df = season_dates(pd.DataFrame({"year": [2024, 2024], "season": ["DJF", "NDJ"], "enso": [1.0, 1.0]}))
    assert df.assumed_available_at.tolist() == [pd.Timestamp("2024-03-05"), pd.Timestamp("2025-02-05")]


def test_incomplete_year_and_calendar_lags():
    df = climate_years()
    df = df[~((df.year == 2002) & (df.season == "NDJ"))]
    annual = annual_enso(df)
    assert annual.empty  # 2002 incomplete; 2003 and 2004 lack required calendar lags


def test_warm_cold_asymmetry():
    df = climate_years()
    df.loc[df.year == 2004, "enso"] = -1.5
    annual = annual_enso(df).set_index("year")
    assert annual.loc[2004, "cold"] == 1.5
    assert annual.loc[2004, "warm"] == 0
    assert annual.loc[2004, "warm_lag1"] == 1


def design_data():
    frame = pd.DataFrame({"series": ["IDN:rice"] * 30, "country": ["IDN"] * 30, "crop": ["rice"] * 30})
    for col in ENSO + CONTROLS + WEATHER:
        frame[col] = np.arange(30, dtype=float)
    return frame


def test_preprocessing_cannot_learn_from_holdout():
    train = design_data()
    design = Design(train, "weather")
    before = design.mean.copy()
    holdout = train.copy()
    holdout["temperature_c"] += 10000
    _, _, transformed = design.transform(holdout)
    assert design.mean == before
    assert transformed[:, design.features.index("temperature_c")].mean() > 100


def test_unknown_series_fails_instead_of_inventing_effect():
    train = design_data()
    d = Design(train)
    train.loc[0, "series"] = "XYZ:rice"
    with pytest.raises(ValueError, match="Untrained"):
        d.transform(train)


def test_missing_controls_are_disclosed():
    train = design_data()
    d = Design(train)
    train.loc[0, "lag_gdp_growth"] = np.nan
    _, _, x = d.transform(train)
    assert np.isfinite(x).all()
    assert x[0, len(d.features) + d.features.index("lag_gdp_growth")] == 1


def test_expected_shortfall_accounts_for_probability_mass_at_boundary():
    x = np.arange(100, dtype=float)
    metrics = loss_metrics(x, buffer=95, attachment=90, limit=5, loading=0.2)
    assert metrics["es95_usd"] == pytest.approx(np.mean([95, 96, 97, 98, 99]))
    assert metrics["probability_buffer_exceeded"] == 0.04
    assert metrics["illustrative_layer_pure_premium_usd"] == pytest.approx(0.35)


def test_losses_allow_gains_and_zero_climate_increment():
    metrics = loss_metrics(np.zeros(100))
    assert metrics["expected_net_loss_usd"] == metrics["es95_usd"] == 0
    assert loss_metrics(-np.ones(100))["expected_net_loss_usd"] == -1
    assert loss_metrics(-np.ones(100))["illustrative_layer_pure_premium_usd"] == 0


@pytest.mark.parametrize("x", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_loss_samples_fail(x):
    with pytest.raises(ValueError):
        loss_metrics(x)


def test_weather_annual_rain_is_sum_and_incomplete_year_excluded():
    rows = [
        ("IDN", f"2000-{m:02d}", v, value)
        for m in range(1, 13)
        for v, value in [("tas", 25.0), ("pr", 100.0), ("tasmax", 30.0)]
    ]
    frame = pd.DataFrame(rows, columns=["country", "period", "variable", "value"])
    annual = annual_weather(frame)
    assert annual.annual_rain_mm.iloc[0] == 1200
    assert annual.temperature_c.iloc[0] == 25
    frame = frame[frame.period != "2000-12"]
    assert annual_weather(frame).empty

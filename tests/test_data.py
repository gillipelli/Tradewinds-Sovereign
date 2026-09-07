import hashlib

import pandas as pd
import pytest

from tradewinds.data import Store, atomic_json, parse_roni
from tradewinds.risk import exposures, fiscal_denominators


def test_cached_source_detects_corruption(tmp_path):
    store = Store(tmp_path)
    content = b"real-content"
    (store.raw / "vintage.bin").write_bytes(content)
    store.manifest["x"] = {"file": "vintage.bin", "sha256": hashlib.sha256(content).hexdigest()}
    assert store.fetch("x", "https://example.invalid") == store.raw / "vintage.bin"
    (store.raw / "vintage.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        store.fetch("x", "https://example.invalid")


def test_partial_current_roni_row_is_retained(tmp_path):
    rows = ["<tr><td>" + "</td><td>".join([str(y)] + ["1.0"] * 12) + "</td></tr>" for y in range(1950, 2026)]
    rows.append("<tr><td>2026</td><td>0.1</td><td>0.2</td></tr>")
    p = tmp_path / "roni.html"
    p.write_text("<table>" + "".join(rows) + "</table>")
    df = parse_roni(p)
    assert len(df[df.year == 2026]) == 2


def test_valuation_uses_correct_currency_and_excludes_old_years(tmp_path):
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Element": ["Gross Production Value (current thousand US$)"] * 2,
            "Unit": ["1000 USD"] * 2,
            "country": ["IDN", "PNG"],
            "crop": ["rice", "rice"],
            "year": [2024, 1998],
            "value": [100, 100],
            "Flag": ["A", "E"],
        }
    ).to_csv(folder / "qv.csv", index=False)
    exp = exposures(tmp_path, ["IDN:rice", "PNG:rice"])
    assert exp.series.tolist() == ["IDN:rice"]
    assert exp.baseline_usd.iloc[0] == 100000


def test_fiscal_denominator_requires_matching_recent_years(tmp_path):
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [
            ["IDN", 2024, "gdp_usd", 1000],
            ["IDN", 2009, "revenue_pct_gdp", 20],
            ["MYS", 2024, "gdp_usd", 1000],
            ["MYS", 2024, "revenue_pct_gdp", 20],
        ],
        columns=["country", "year", "indicator", "value"],
    ).to_csv(folder / "wdi.csv", index=False)
    result = fiscal_denominators(tmp_path, ["IDN", "MYS"], 2024).set_index("country")
    assert pd.isna(result.loc["IDN", "government_revenue_usd"])
    assert result.loc["MYS", "government_revenue_usd"] == 200


def test_atomic_json_rejects_invalid_numeric_results(tmp_path):
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "result.json", {"loss": float("nan")})
    assert not (tmp_path / "result.json").exists()

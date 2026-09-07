"""A completed dashboard must not read mutable data from a later failed update."""

import json

import pandas as pd

from tradewinds.report import render


def test_report_freezes_its_dashboard_inputs(tmp_path):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    coverage = pd.DataFrame(
        {"country": ["IDN"], "crop": ["rice"], "eligible": [True], "official_share": [1.0]}
    )
    coverage.to_csv(processed / "coverage.csv", index=False)
    pd.DataFrame({"year": [2024]}).to_csv(processed / "panel.csv", index=False)
    (processed / "snapshot.json").write_text(json.dumps({"sources": {}}))
    for name in ["roni.csv", "oni.csv", "weather.csv"]:
        (processed / name).write_text("value\n1.0\n")
    out = tmp_path / "completed_report"
    render(tmp_path, out)
    (processed / "weather.csv").write_text("value\n999.0\n")
    assert (out / "data/weather.csv").read_text() == "value\n1.0\n"
    assert (out / "index.html").exists()

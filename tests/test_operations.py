"""Operational contracts: no-op reuse, stale-model withholding, failed-run preservation."""

import json

import pandas as pd
import pytest

from tradewinds import analysis, data, model, monitor, operations, report, risk


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    root = tmp_path
    frame = pd.DataFrame({"series": ["IDN:rice"], "yield_growth": [0.1], "area_growth": [0.2]})
    (root / "configs").mkdir()
    (root / "configs/project.yaml").write_text("seed: 42\n")
    (root / "src/tradewinds").mkdir(parents=True)
    counts = {"fits": 0, "risk": 0}
    monkeypatch.setattr(operations, "ingest", lambda *a: {"sources": {}})
    monkeypatch.setattr(operations, "build_panel", lambda *a: frame)
    monkeypatch.setattr(model, "modeling_panel", lambda *a: frame)

    def fake_fit(root, folder):
        counts["fits"] += 1
        folder.mkdir(parents=True)
        data.atomic_json(folder / "fit.json", {"diagnostics_pass": True})

    monkeypatch.setattr(model, "fit", fake_fit)
    for name in ["backtest", "descriptive", "robustness"]:
        monkeypatch.setattr(analysis, name, lambda *a: None)
    monkeypatch.setattr(monitor, "outlook", lambda *a: None)
    monkeypatch.setattr(report, "render", lambda *a: None)

    def fake_risk(*a):
        counts["risk"] += 1

    monkeypatch.setattr(risk, "simulate", fake_risk)
    return root, frame, counts


def test_unchanged_inputs_reuse_model_and_refresh_risk(pipeline):
    root, _, counts = pipeline
    a = operations.update(root, refresh=False)
    b = operations.update(root, refresh=False)
    assert counts == {"fits": 1, "risk": 2}
    assert a["model"] == b["model"]
    assert not b["training_changed"]


def test_changed_auxiliary_data_requires_refit_and_withholds_stale_risk(pipeline):
    root, frame, counts = pipeline
    operations.update(root, refresh=False)
    frame.loc[0, "area_growth"] = 0.5
    state = operations.update(root, refresh=False, refit=False)
    assert counts["fits"] == 1
    assert counts["risk"] == 1
    assert state["risk_status"] == "withheld_failed_diagnostics_or_stale_model"
    operations.update(root, refresh=False)
    assert counts["fits"] == 2
    assert counts["risk"] == 2


def test_source_failure_preserves_last_success(pipeline, monkeypatch):
    root, _, _ = pipeline
    first = operations.update(root, refresh=False)

    def fail(*args):
        raise ValueError("Provider schema changed")

    monkeypatch.setattr(operations, "ingest", fail)
    with pytest.raises(ValueError, match="schema"):
        operations.update(root)
    assert json.loads((root / "artifacts/state.json").read_text()) == first
    runs = [json.loads(p.read_text()) for p in (root / "artifacts/runs").glob("*/run.json")]
    assert any(r["status"] == "failed" for r in runs)

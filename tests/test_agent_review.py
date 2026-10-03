"""Independent evidence-integrity and resume regressions; no live API calls."""

import json
import pandas as pd
import pytest
from tradewinds.assessment_bundle import freeze_assessment, digest, validate_bundle
from tradewinds.agents.provider import DemoClient, FixtureClient
from tradewinds.agents.runtime import run_agent
from tradewinds.agents.store import Store
from tradewinds.agents.tools import DomainTools
from tradewinds.agents.verification import verify_evidence


@pytest.fixture
def reviewed_assessment(tmp_path):
    reports = tmp_path / "artifacts/runs/review/reports"
    reports.mkdir(parents=True)
    (reports.parent / "run.json").write_text(json.dumps({"status": "complete"}))
    (reports / "event_risk_metadata.json").write_text(
        json.dumps({"climate": {"as_of": "2026-09-07"}, "models": {"fiscal": {"diagnostics_pass": True}}})
    )
    pd.DataFrame([{"country": "IDN", "year": "2027", "revenue_loss_2025usd_mean": 12.0}]).to_csv(
        reports / "event_country_risk.csv", index=False
    )
    pd.DataFrame([{"scope": "all_modeled", "period": "2027", "revenue_loss_2025usd_mean": 20.0}]).to_csv(
        reports / "event_aggregate_risk.csv", index=False
    )
    source = tmp_path / "src/tradewinds/science.py"
    source.parent.mkdir(parents=True)
    source.write_text("SCIENTIFIC_VERSION = 1\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "method.md").write_text("Historical associations are not causal forecasts.")
    freeze_assessment(tmp_path, "review")
    return tmp_path


def test_completed_resume_rechecks_bundle(reviewed_assessment):
    root = reviewed_assessment
    client = DemoClient({"assessment_id": "review", "country": "IDN"})
    assert run_agent(root, ["review"], "Inspect", client, run_id="completed")["status"] == "completed"
    (root / "artifacts/assessments/review/reports/event_country_risk.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="integrity|changed"):
        run_agent(root, [], "", client, run_id="completed", resume=True)


def test_pending_resume_rejects_scientific_code_change(reviewed_assessment):
    root = reviewed_assessment
    client = FixtureClient([])
    assert run_agent(root, ["review"], "Inspect", client, run_id="pending")["status"] == "incomplete"
    (root / "src/tradewinds/science.py").write_text("SCIENTIFIC_VERSION = 2\n")
    with pytest.raises(ValueError, match="code|source|implementation|changed"):
        run_agent(root, [], "", client, run_id="pending", resume=True)


def test_retrieval_evidence_cannot_forge_excerpt(reviewed_assessment):
    root = reviewed_assessment
    store = Store(root / "retrieval.sqlite")
    try:
        tools = DomainTools(root, ["review"], store)
        result = tools.search_evidence("review", "causal")[0]
        key = result["evidence_ids"][0]
        record = store.get("evidence", key)
        record["value"] = "This fabricated excerpt is absent from the document."
        store.save("evidence", key, record)
        with pytest.raises(ValueError, match="evidence|Evidence|content|excerpt|hash"):
            verify_evidence(key, store, root)
    finally:
        store.close()


def test_new_scenario_outputs_produce_consumable_verified_citations(reviewed_assessment):
    root = reviewed_assessment
    store = Store(root / "scenario.sqlite")
    try:
        tools = DomainTools(root, ["review"], store)
        workspace = root / "artifacts/scenarios/review/new-scenario"
        reports = workspace / "reports"
        reports.mkdir(parents=True)
        rows = [{"scope": "all_modeled", "period": "2027", "revenue_loss_2025usd_mean": 23.0}]
        path = reports / "event_aggregate_risk.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        manifest = validate_bundle(root / "artifacts/assessments/review")
        (workspace / "complete.json").write_text(
            json.dumps(
                {
                    "results": rows,
                    "hashes": {"reports/event_aggregate_risk.csv": digest(path)},
                    "bundle_hash": manifest["bundle_hash"],
                }
            )
        )
        result = tools._scenario_result("review", "new-scenario", rows, cached=False)
        key = result["risk_estimates"][0]["evidence_ids"][0]
        assert verify_evidence(key, store, root)["value"] == 23.0
        comparison = tools.compare_scenarios("review", ["central", "new-scenario"], scope="all_modeled")
        assert verify_evidence(comparison["differences"][0]["evidence_ids"][0], store, root)["value"] == 3.0
    finally:
        store.close()


def test_failed_diagnostics_block_numeric_report(reviewed_assessment):
    import shutil

    root = reviewed_assessment
    failed_run = root / "artifacts/runs/failed-diagnostics"
    shutil.copytree(root / "artifacts/runs/review", failed_run)
    metadata = failed_run / "reports/event_risk_metadata.json"
    content = json.loads(metadata.read_text())
    content["models"]["fiscal"]["diagnostics_pass"] = False
    metadata.write_text(json.dumps(content))
    freeze_assessment(root, "failed-diagnostics")
    client = DemoClient({"assessment_id": "failed-diagnostics", "country": "IDN"})
    result = run_agent(root, ["failed-diagnostics"], "Inspect", client, run_id="diagnostic-failure")
    assert result["status"] != "completed"
    assert not (root / "artifacts/agent_runs/diagnostic-failure/report.html").exists()

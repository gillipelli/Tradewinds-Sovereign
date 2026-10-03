"""Grounding, isolation, recovery and spend controls on synthetic scientific fixtures."""

import json

import pandas as pd
import pytest

from tradewinds.assessment_bundle import freeze_assessment, validate_bundle
from tradewinds.agents.provider import DemoClient, FixtureClient
from tradewinds.agents.runtime import run_agent
from tradewinds.agents.schemas import Claim, Findings, Limits
from tradewinds.agents.store import Store
from tradewinds.agents.tools import DomainTools
from tradewinds.agents.verification import verify_findings


@pytest.fixture
def assessment(tmp_path):
    run = tmp_path / "artifacts/runs/test-run"
    reports = run / "reports"
    reports.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"status": "complete"}))
    (reports / "event_risk_metadata.json").write_text(
        json.dumps({"climate": {"as_of": "2026-09-07"}, "models": {"fiscal": {"diagnostics_pass": True, "prior_scale": 2.0}}})
    )
    pd.DataFrame(
        [
            {
                "country": "IDN",
                "year": "2027",
                "revenue_loss_2025usd_mean": 12.0,
                "revenue_loss_es95_2025usd": 30.0,
                "p_revenue_loss": 0.6,
            }
        ]
    ).to_csv(reports / "event_country_risk.csv", index=False)
    pd.DataFrame(
        [{"scope": "all_modeled", "period": "2027", "revenue_loss_2025usd_mean": 20.0, "es95_2025usd": 42.0}]
    ).to_csv(reports / "event_aggregate_risk.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "all_modeled",
                "period": "2027",
                "revenue_loss_2025usd_mean": 25.0,
                "es95_2025usd": 50.0,
                "variant": "neutral_late_2027",
            }
        ]
    ).to_csv(reports / "event_sensitivity.csv", index=False)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/method.md").write_text("Historical associations are not causal forecasts.")
    (tmp_path / "docs/AGENT_EVALUATION_PROTOCOL.md").write_text("evaluationcanary grading guidance")
    pd.DataFrame([{"kind": "fiscal", "model": "baseline", "rmse": 0.09},
                  {"kind": "fiscal", "model": "with_agriculture", "rmse": 0.10}]).to_csv(
        reports / "macro_validation.csv", index=False)
    freeze_assessment(tmp_path, "test-run")
    return tmp_path


def test_legacy_freeze_is_readonly_and_tamper_detected(assessment):
    bundle = assessment / "artifacts/assessments/test-run"
    assert not validate_bundle(bundle)["recomputable"]
    (bundle / "reports/event_country_risk.csv").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        validate_bundle(bundle)


def test_scoped_queries_sensitivities_and_grounding(assessment):
    store = Store(assessment / "test.sqlite")
    tools = DomainTools(assessment, ["test-run"], store)
    risk = tools.query_risk("test-run", country="IDN", metric="es95_2025usd")
    claim = Claim(
        kind="numeric",
        text="Ignored model label",
        value=30.0,
        unit="2025 USD",
        evidence_ids=risk["evidence_ids"],
    )
    report = Findings(title="Test", claims=[claim], limitations=["Historical association only."])
    assert verify_findings(report, store, assessment)["passed"]
    claim.unit = "nominal USD"
    assert not verify_findings(report, store, assessment)["passed"]
    claim.unit, claim.value = "2025 USD", 100.0
    assert not verify_findings(report, store, assessment)["passed"]
    comparison = tools.compare_scenarios("test-run", ["central", "neutral_late_2027"], scope="all_modeled")
    difference = comparison["differences"][0]
    report.claims = [Claim(kind="numeric", text="Change", **difference)]
    assert difference["value"] == 5.0
    assert verify_findings(report, store, assessment)["passed"]
    assert (
        tools.execute("query_risk", {"assessment_id": "test-run", "country": "IDN", "scope": "all_modeled"})[
            "status"
        ]
        != "ok"
    )
    assert tools.execute("get_assessment", {"assessment_id": "../test-run"})["status"] != "ok"
    assert (
        tools.execute("run_scenario", {"assessment_id": "test-run", "specification": {}})["status"]
        == "unavailable"
    )
    assert tools.search_evidence("test-run", "causal")[0]["untrusted_document"]
    store.close()


def test_demo_vertical_slice_and_completed_resume(assessment):
    selector = {"assessment_id": "test-run", "country": "IDN"}
    state = run_agent(
        assessment, ["test-run"], "Inspect risk", DemoClient(selector), run_id="demo", selector=selector
    )
    assert state["status"] == "completed"
    assert state["cost_usd"] == 0
    folder = assessment / "artifacts/agent_runs/demo"
    assert '<a href="#e-' in (folder / "report.html").read_text()
    saved = json.loads((folder / "report.json").read_text())
    assert "country" in saved["claims"][0]["text"]
    assert (
        run_agent(assessment, [], "", DemoClient(selector), run_id="demo", resume=True)["status"]
        == "completed"
    )


def test_invalid_findings_are_not_published(assessment):
    client = FixtureClient(
        [
            {
                "calls": [
                    {
                        "id": "a",
                        "name": "submit_findings",
                        "arguments": {
                            "title": "Unsupported",
                            "claims": [
                                {
                                    "kind": "numeric",
                                    "text": "Loss",
                                    "value": 90.0,
                                    "unit": "2025 USD",
                                    "evidence_ids": ["invented"],
                                }
                            ],
                            "limitations": ["Unknown."],
                        },
                    }
                ]
            }
        ]
    )
    result = run_agent(assessment, ["test-run"], "Question", client, run_id="bad")
    assert result["status"] == "incomplete"
    assert not (assessment / "artifacts/agent_runs/bad/report.html").exists()


class LiveFake:
    name, model = "test-live", "fake"
    calls = 0

    def count_input(self, messages, tools):
        return 1000

    def complete(self, messages, tools, limits):
        self.calls += 1
        raise TimeoutError("Unknown billable outcome")


def test_cap_prevents_dispatch_and_ambiguous_failure_retains_reservation(assessment):
    client = LiveFake()
    limits = Limits(spend_ceiling_usd=0.0001, input_usd_per_million=1.0, output_usd_per_million=1.0)
    state = run_agent(assessment, ["test-run"], "Question", client, limits=limits, run_id="cap")
    assert state["status"] == "incomplete"
    assert client.calls == 0
    limits.spend_ceiling_usd = 1.0
    state = run_agent(assessment, ["test-run"], "Question", client, limits=limits, run_id="timeout")
    assert state["status"] == "failed" and state["reserved_usd"] > 0
    resumed = run_agent(assessment, [], "", client, run_id="timeout", resume=True)
    assert resumed["status"] == "incomplete" and client.calls == 1


def test_restored_pending_tool_does_not_reexecute_completed_call(assessment):
    selector = {"assessment_id": "test-run", "country": "IDN"}
    client = DemoClient(selector)
    result = run_agent(assessment, ["test-run"], "Question", client, run_id="recover")
    folder = assessment / "artifacts/agent_runs/recover"
    store = Store(folder / "state.sqlite")
    state = store.get("checkpoints", "recover")
    last = state["messages"][-1]["content"][0]
    state.update(
        status="running", pending=[{"id": last["id"], "name": last["name"], "arguments": last["input"]}]
    )
    store.save("checkpoints", "recover", state)
    store.close()
    restored = run_agent(assessment, [], "", client, run_id="recover", resume=True)
    assert result["status"] == restored["status"] == "completed"


def test_fresh_publication_freezes_date_models_and_inputs(tmp_path):
    import yaml
    from tradewinds.assessment_bundle import create_scenario_workspace

    root = tmp_path
    reports = root / "artifacts/runs/fresh/reports"
    reports.mkdir(parents=True)
    (reports / "event_risk_metadata.json").write_text(json.dumps({"climate": {"as_of": "2026-08-12"}}))
    for folder in ["data/event", "data/processed", "configs", "original/agriculture", "original/fiscal"]:
        (root / folder).mkdir(parents=True)
    (root / "data/event/imf.csv").write_text("old frozen inputs")
    (root / "data/processed/roni.csv").write_text("old roni")
    (root / "configs/project.yaml").write_text(
        yaml.safe_dump({"event": {"as_of": None}, "risk": {"simulations": 1}})
    )
    models = {}
    for kind in ["agriculture", "fiscal"]:
        path = root / "original" / kind
        (path / "fit.json").write_text('{"diagnostics_pass": true}')
        models[kind] = str(path)
    bundle = freeze_assessment(
        root, "fresh", publication_state={"status": "complete", "macro_models": models}
    )
    assert validate_bundle(bundle)["recomputable"]
    assert yaml.safe_load((bundle / "configs/project.yaml").read_text())["event"]["as_of"] == "2026-08-12"
    (root / "data/event/imf.csv").write_text("mutated current inputs")
    workspace = create_scenario_workspace(root, "fresh", "trial")
    assert (workspace / "data/event/imf.csv").read_text() == "old frozen inputs"
    (workspace / "data/event/imf.csv").write_text("scenario derivative")
    assert (bundle / "data/event/imf.csv").read_text() == "old frozen inputs"


def test_scenario_cache_is_citable_and_tampering_fails(assessment):
    from tradewinds.assessment_bundle import digest, validate_bundle

    workspace = assessment / "artifacts/scenarios/test-run/computed"
    (workspace / "reports").mkdir(parents=True)
    frame = pd.DataFrame([{"scope": "all_modeled", "period": "2027", "revenue_loss_2025usd_mean": 29.0}])
    path = workspace / "reports/event_aggregate_risk.csv"
    frame.to_csv(path, index=False)
    manifest = validate_bundle(assessment / "artifacts/assessments/test-run")
    (workspace / "complete.json").write_text(
        json.dumps(
            {
                "bundle_hash": manifest["bundle_hash"],
                "hashes": {"reports/event_aggregate_risk.csv": digest(path)},
            }
        )
    )
    store = Store(assessment / "scenario.sqlite")
    tools = DomainTools(assessment, ["test-run"], store)
    difference = tools.compare_scenarios("test-run", ["central", "computed"], scope="all_modeled")[
        "differences"
    ][0]
    report = Findings(
        title="Scenario",
        claims=[Claim(kind="numeric", text="Difference", **difference)],
        limitations=["Conditional."],
    )
    assert difference["value"] == 9.0
    assert verify_findings(report, store, assessment)["passed"]
    path.write_text("tampered")
    assert not verify_findings(report, store, assessment)["passed"]
    store.close()


def test_fixed_packet_single_response_verified_and_resume_method_bound(assessment):
    class PacketClient:
        name, model = "fixture", "packet-fixture"
        calls = 0

        def complete(self, messages, tools, limits):
            self.calls += 1
            assert [tool["name"] for tool in tools] == ["submit_findings"]
            assert limits.max_turns == 1
            packet = json.loads(messages[0]["content"].split("available.\n", 1)[1])
            risk = next(entry["result"] for entry in packet["entries"] if entry["kind"] == "risk_cell")
            arguments = {
                "title": "Fixed packet",
                "claims": [
                    {
                        "kind": "numeric",
                        "text": "Estimate",
                        "value": risk["value"],
                        "unit": risk["unit"],
                        "evidence_ids": risk["evidence_ids"],
                    }
                ],
                "limitations": ["Fixed bounded evidence packet; scientific interpretation requires review."],
            }
            return {
                "calls": [{"id": "submit", "name": "submit_findings", "arguments": arguments}],
                "input_tokens": 0,
                "output_tokens": 0,
                "text": "",
            }

    client = PacketClient()
    result = run_agent(assessment, ["test-run"], "Inspect", client, run_id="fixed", method="fixed_evidence")
    assert result["status"] == "completed" and result["turns"] == 1 and client.calls == 1
    folder = assessment / "artifacts/agent_runs/fixed"
    assert json.loads((folder / "evidence_packet.json").read_text())["numeric_cells"] > 0
    assert result["packet_bytes"] <= 60500
    with pytest.raises(ValueError, match="method"):
        run_agent(assessment, [], "", client, run_id="fixed", resume=True, method="agent")
    assert (
        run_agent(assessment, [], "", client, run_id="fixed", resume=True, method="fixed_evidence")["status"]
        == "completed"
    )


def test_fixed_packet_rejects_adaptive_tools_without_second_model_turn(assessment):
    client = FixtureClient(
        [
            {
                "calls": [
                    {
                        "id": "query",
                        "name": "query_risk",
                        "arguments": {"assessment_id": "test-run", "country": "IDN"},
                    }
                ]
            },
            {"calls": []},
        ]
    )
    result = run_agent(
        assessment, ["test-run"], "Inspect", client, run_id="fixed-denied", method="fixed_evidence"
    )
    assert result["status"] == "incomplete" and result["turns"] == 1
    trace = (assessment / "artifacts/agent_runs/fixed-denied/events.jsonl").read_text()
    assert "only permits submit_findings" in trace


def test_unit_conversion_verifies_operand_and_rejects_tampering(assessment):
    from tradewinds.agents.verification import verify_evidence

    store = Store(assessment / "conversion.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        source = domain.query_risk("test-run", country="IDN")
        converted = domain.convert_units(source["evidence_ids"][0], "million 2025 USD")
        key = converted["evidence_ids"][0]
        assert verify_evidence(key, store, assessment)["value"] == pytest.approx(12 / 1e6)
        restored = domain.convert_units(key, "2025 USD")
        assert restored["value"] == pytest.approx(12.)
        probability = domain.query_risk("test-run", country="IDN", metric="p_revenue_loss")
        assert domain.convert_units(probability["evidence_ids"][0], "percent")["value"] == 60.
        with pytest.raises(ValueError, match="incompatible"):
            domain.convert_units(key, "percent")
        forged = dict(store.get("evidence", key), value=999.)
        forged_key = store.add_evidence(forged)
        with pytest.raises(ValueError, match="Conversion mismatch"):
            verify_evidence(forged_key, store, assessment)
        report = Findings(title="Conversion", claims=[Claim(kind="numeric", text="Converted", **{
            k: v for k, v in converted.items() if k != "selector"
        })], limitations=["Conditional estimate."])
        assert verify_findings(report, store, assessment)["passed"]
    finally:
        store.close()


def test_temporal_context_does_not_allow_arbitrary_numeric_prose(assessment):
    from tradewinds.agents.verification import unsupported_numerals

    store = Store(assessment / "context.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        source = domain.query_risk("test-run", country="IDN")
        evidence = [store.get("evidence", source["evidence_ids"][0])]
        assert not unsupported_numerals("Revenue estimate in year 2027 in constant 2025 USD", evidence)
        assert not unsupported_numerals("Risk during year 2027", evidence)
        assert not unsupported_numerals("At a fixed 2025 exchange rate", evidence)
        assert unsupported_numerals("The exchange rate is 2025", evidence)
        assert unsupported_numerals("Loss is 2027 dollars", evidence)
        assert unsupported_numerals("Loss is 2025 dollars", evidence)
        assert unsupported_numerals("Risk in 2029", evidence)
        assert unsupported_numerals("Loss is 123 dollars in 2027", evidence)
        assert unsupported_numerals("Risk during year 2027", [])
        assert unsupported_numerals("There are 2027 loss events in Indonesia.", evidence)
        assert unsupported_numerals("The amount is in 2027 dollars.", evidence)
        assert unsupported_numerals("The amount is in year 2027 dollars.", evidence)
        metadata = [{"value": {"climate": {"forecast_issue_date": "2026-08-13"}}}]
        assert not unsupported_numerals("Forecast issued on 2026-08-13", metadata)
        assert unsupported_numerals("Forecast issued on 2026-09-13", metadata)
    finally:
        store.close()



def test_research_corpus_excludes_grading_guidance_and_verifies_validation_table(assessment):
    from tradewinds.agents.verification import verify_evidence

    store = Store(assessment / "corpus.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        assert domain.search_evidence("test-run", "evaluationcanary") == []
        assert domain.search_evidence("test-run", "causal")
        with store.db:
            store.db.execute("INSERT INTO documents VALUES (?,?,?)", (
                "test-run", "docs/AGENT_EVALUATION_PROTOCOL.md", "evaluationcanary grading guidance"))
        assert domain.search_evidence("test-run", "evaluationcanary") == []
        diagnostics = domain.get_model_diagnostics("test-run")
        table = next(item for item in diagnostics if item["artifact"].endswith("macro_validation.csv"))
        key = table["evidence_ids"][0]
        assert verify_evidence(key, store, assessment)["value"][0]["rmse"] == .09
        forged = dict(store.get("evidence", key))
        forged["value"][0]["rmse"] = 0.
        with pytest.raises(ValueError, match="table content mismatch"):
            verify_evidence(store.add_evidence(forged), store, assessment)
    finally:
        store.close()



def test_scenario_parameter_uses_metadata_and_verifies_unit(assessment):
    from tradewinds.agents.verification import verify_evidence

    store = Store(assessment / "parameter.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        result = domain.query_scenario_parameter("test-run")
        assert result["value"] == 2.
        key = result["evidence_ids"][0]
        assert verify_evidence(key, store, assessment)["value"] == 2.
        forged = dict(store.get("evidence", key), value=1.)
        with pytest.raises(ValueError, match="content mismatch"):
            verify_evidence(store.add_evidence(forged), store, assessment)
        forged = dict(store.get("evidence", key), unit="probability")
        with pytest.raises(ValueError, match="selector or unit"):
            verify_evidence(store.add_evidence(forged), store, assessment)
        assert domain.execute("query_scenario_parameter", {
            "assessment_id": "test-run", "scenario_id": "../../outside"})["status"] == "unavailable"
    finally:
        store.close()



def test_typed_date_claim_is_source_backed_and_rendered_canonically(assessment):
    from tradewinds.agents.verification import render_report, verify_evidence

    store = Store(assessment / "date.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        result = domain.query_assessment_date("test-run", "as_of")
        assert result["value"] == "2026-09-07"
        claim = Claim(kind="metadata", text="UNTRUSTED extra amount 999", **{
            key: value for key, value in result.items() if key != "selector"})
        findings = Findings(title="Dated assessment", claims=[claim], limitations=["Historical inspection."])
        assert verify_findings(findings, store, assessment)["passed"]
        rendered = render_report(findings, store, ["2026-09-07"])
        assert "2026-09-07" in rendered
        assert "UNTRUSTED extra amount" not in rendered
        claim.value = "2027-01-01"
        assert not verify_findings(findings, store, assessment)["passed"]
        source = dict(store.get("evidence", result["evidence_ids"][0]), value="2027-01-01")
        with pytest.raises(ValueError, match="content mismatch"):
            verify_evidence(store.add_evidence(source), store, assessment)
        assert domain.execute("query_assessment_date", {
            "assessment_id": "test-run", "field": "arbitrary"})["status"] == "unavailable"
    finally:
        store.close()



def test_country_names_resolve_without_geographic_substitution(assessment):
    store = Store(assessment / "aliases.sqlite")
    try:
        domain = DomainTools(assessment, ["test-run"], store)
        by_name = domain.query_risk("test-run", country=" Indonesia ")
        by_code = domain.query_risk("test-run", country="idn")
        assert by_name == by_code
        assert by_name["selector"]["country"] == "IDN"
        missing = domain.execute("query_risk", {"assessment_id": "test-run", "country": "Brazil"})
        assert missing["status"] == "unavailable"
        assert "IDN" in missing["error"]["message"]
        assert "substitute" in missing["error"]["message"]
    finally:
        store.close()

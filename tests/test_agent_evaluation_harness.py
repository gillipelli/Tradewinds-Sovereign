"""Shared evaluation spend and trial accounting must survive interrupted runs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradewinds.agents.schemas import Limits
from tradewinds.agents.store import Store


spec = importlib.util.spec_from_file_location(
    "evaluate_agent_live", Path(__file__).parents[1] / "scripts/evaluate_agent_live.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def suite():
    return {
        "tasks": [
            {
                "id": "one",
                "split": "holdout",
                "question": "First?",
                "required": ["evidence"],
                "fail_if": ["fabrication"],
            },
            {
                "id": "two",
                "split": "holdout",
                "question": "Second?",
                "required": ["evidence"],
                "fail_if": ["fabrication"],
            },
            {
                "id": "revision",
                "split": "holdout",
                "question": "Why changed?",
                "required": ["two vintages"],
                "fail_if": ["fabrication"],
                "needs_two_assessments": True,
            },
        ]
    }


def store_state(root, run_id, state):
    store = Store(root / "artifacts/agent_runs" / run_id / "state.sqlite")
    try:
        store.save("checkpoints", run_id, state)
    finally:
        store.close()


def test_shared_allowance_reserves_uncertain_charge_and_does_not_repeat_trials(tmp_path, monkeypatch):
    allocations = []

    def fake_run(root, assessments, question, client, *, limits, run_id):
        allocations.append(limits.spend_ceiling_usd)
        state = {
            "status": "failed",
            "cost_usd": 0.0,
            "reserved_usd": limits.spend_ceiling_usd,
            "reason": "Ambiguous network failure",
        }
        store_state(root, run_id, state)
        return state

    monkeypatch.setattr(harness, "run_agent", fake_run)
    client = SimpleNamespace(name="fixture", model="fake")
    limits = Limits(spend_ceiling_usd=0.3)
    path = harness.evaluate(tmp_path, suite(), ["assessment"], "eval", client, limits, per_task_ceiling=0.2)
    assert allocations == pytest.approx([0.2, 0.1])
    result = json.loads(path.read_text())
    assert result["charged_or_reserved_usd"] == pytest.approx(0.3)
    assert result["trials"][2]["status"] == "unavailable"
    assert all(row["human_success"] is None for row in result["trials"])
    assert result["llm_task_success"] == "not scored"
    harness.evaluate(tmp_path, suite(), ["assessment"], "eval", client, limits, per_task_ceiling=0.2)
    assert len(allocations) == 2


def test_evaluation_refuses_settings_change_and_preserves_human_review(tmp_path, monkeypatch):
    def fake_run(root, assessments, question, client, *, limits, run_id):
        state = {"status": "completed", "cost_usd": 0.0, "reserved_usd": 0.0}
        store_state(root, run_id, state)
        return state

    monkeypatch.setattr(harness, "run_agent", fake_run)
    client = SimpleNamespace(name="fixture", model="fake")
    limits = Limits(spend_ceiling_usd=1.0)
    path = harness.evaluate(tmp_path, suite(), ["assessment"], "eval", client, limits)
    review = path.parent / "review_template.json"
    review.write_text('{"human_review":"retain me"}')
    harness.evaluate(tmp_path, suite(), ["assessment"], "eval", client, limits)
    assert json.loads(review.read_text()) == {"human_review": "retain me"}
    with pytest.raises(ValueError, match="settings changed"):
        harness.evaluate(tmp_path, suite(), ["assessment"], "eval", client, Limits(spend_ceiling_usd=2.0))


def test_reviewed_questions_have_fixed_split_and_rubric():
    questions = json.loads((Path(__file__).parents[1] / "evaluations/live_questions.json").read_text())
    tasks = questions["tasks"]
    assert len(tasks) == 30
    assert len({task["id"] for task in tasks}) == 30
    assert sum(task["split"] == "development" for task in tasks) == 18
    assert sum(task["split"] == "holdout" for task in tasks) == 12
    assert all(task["required"] and task["fail_if"] for task in tasks)


def test_fixed_evidence_method_is_forwarded_and_recorded(tmp_path, monkeypatch):
    methods = []

    def fake_run(root, assessments, question, client, *, limits, run_id, method):
        methods.append(method)
        state = {"status": "completed", "cost_usd": 0.0, "reserved_usd": 0.0}
        store_state(root, run_id, state)
        return state

    monkeypatch.setattr(harness, "run_agent", fake_run)
    client = SimpleNamespace(name="fixture", model="fake")
    result = harness.evaluate(
        tmp_path,
        suite(),
        ["assessment"],
        "fixed-eval",
        client,
        Limits(spend_ceiling_usd=1.0),
        method="fixed_evidence",
    )
    assert methods == ["fixed_evidence", "fixed_evidence"]
    assert json.loads((result.parent / "manifest.json").read_text())["method"] == "fixed_evidence"
    assert json.loads(result.read_text())["llm_task_success"] == "not scored"


def test_task_selection_and_usage_summary(tmp_path, monkeypatch):
    calls = []

    def fake_run(root, assessments, question, client, *, limits, run_id):
        calls.append(run_id)
        state = {"status": "completed", "input_tokens": 12, "output_tokens": 3,
                 "cost_usd": .01, "reserved_usd": 0.}
        store_state(root, run_id, state)
        return state

    monkeypatch.setattr(harness, "run_agent", fake_run)
    client = SimpleNamespace(name="fixture", model="fake")
    limits = Limits(spend_ceiling_usd=1.)
    path = harness.evaluate(tmp_path, suite(), ["assessment"], "selected", client, limits, task_ids=["two"])
    report = json.loads(path.read_text())
    assert calls == ["selected-two-1"]
    assert report["summary"]["input_tokens"] == 12
    assert report["summary"]["reviewed_trials"] == 0
    elapsed = report["trials"][0]["elapsed_seconds"]
    harness.evaluate(tmp_path, suite(), ["assessment"], "selected", client, limits, task_ids=["two"])
    assert json.loads(path.read_text())["trials"][0]["elapsed_seconds"] == elapsed
    with pytest.raises(ValueError, match="outside selected split"):
        harness.evaluate(tmp_path, suite(), ["assessment"], "invalid", client, limits, task_ids=["missing"])

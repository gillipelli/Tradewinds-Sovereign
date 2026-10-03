"""Bounded durable agent loop. No model output becomes code or a filesystem path."""

from __future__ import annotations
import fcntl
import hashlib
import json
import time
import uuid
from pathlib import Path

from ..assessment_bundle import digest, safe_id
from ..data import atomic_json
from .provider import SYSTEM
from .schemas import Findings, Limits
from .store import Store
from .tools import DomainTools, definitions
from .verification import render_report, verify_findings


def _event(folder, event):
    with (folder / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **event}, allow_nan=False) + "\n")


def run_agent(
    root: Path,
    assessment_ids: list[str],
    question: str,
    client,
    *,
    limits=None,
    run_id=None,
    resume=False,
    selector=None,
    method="agent",
):
    if method not in {"agent", "fixed_evidence"}:
        raise ValueError("Unknown investigation method")
    limits = limits or Limits()
    if method == "fixed_evidence":
        limits = limits.model_copy(update={"max_turns": 1, "max_repairs": 0, "max_scenarios": 0})
    run_id = safe_id(run_id or uuid.uuid4().hex)
    folder = root / "artifacts/agent_runs" / run_id
    if not resume and folder.exists():
        raise ValueError("Run already exists; use resume")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(folder / "state.sqlite")
        try:
            return _run(
                root,
                folder,
                run_id,
                assessment_ids,
                question,
                client,
                limits,
                store,
                resume,
                selector,
                method,
            )
        finally:
            store.close()


def _run(root, folder, run_id, assessment_ids, question, client, limits, store, resume, selector, method):
    tools_schema = [tool for tool in definitions() if method == "agent" or tool["name"] == "submit_findings"]
    schema_hash = hashlib.sha256(json.dumps(tools_schema, sort_keys=True).encode()).hexdigest()
    state = store.get("checkpoints", run_id) if resume else None
    if resume and state is None:
        raise ValueError("Run checkpoint does not exist")
    system_hash = hashlib.sha256(SYSTEM.encode()).hexdigest()
    code_hashes = {
        str(path.relative_to(root)): digest(path) for path in (root / "src/tradewinds").rglob("*.py")
    }
    code_hashes.update(
        {"installed/" + path.name: digest(path) for path in Path(__file__).parent.glob("*.py")}
    )
    if (root / "uv.lock").exists():
        code_hashes["uv.lock"] = digest(root / "uv.lock")
    if state:
        if state.get("method", "agent") != method:
            raise ValueError("Cannot resume with a different investigation method")
        if state["provider"] != client.name or state["model"] != client.model:
            raise ValueError("Cannot resume with a different provider or model")
        if state.get("code_hashes") != code_hashes:
            raise ValueError("Agent implementation changed; start a new run")
        if state.get("system_hash") != system_hash or state.get("reasoning_effort") != getattr(
            client, "reasoning_effort", None
        ):
            raise ValueError("System instructions or reasoning effort changed; start a new run")
        if state["schema_hash"] != schema_hash:
            raise ValueError("Tool schema changed; start a new run")
        limits = Limits.model_validate(state["limits"])
        assessment_ids = state["assessment_ids"]
        if state.get("request_inflight"):
            state.update(
                status="incomplete",
                reason="Prior provider response is unknown; reservation retained. Start a new run.",
            )
            store.save("checkpoints", run_id, state)
            return state
    else:
        state = {
            "run_id": run_id,
            "method": method,
            "provider": client.name,
            "model": client.model,
            "question": question,
            "assessment_ids": assessment_ids,
            "selector": selector,
            "limits": limits.model_dump(),
            "schema_hash": schema_hash,
            "system_hash": system_hash,
            "code_hashes": code_hashes,
            "reasoning_effort": getattr(client, "reasoning_effort", None),
            "status": "created",
            "turns": 0,
            "tool_count": 0,
            "scenario_count": 0,
            "repairs": 0,
            "cost_usd": 0.0,
            "reserved_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "messages": [{"role": "user", "content": question}],
            "pending": [],
            "pending_results": [],
        }
        atomic_json(
            folder / "request.json",
            {
                "question": question,
                "method": method,
                "assessment_ids": assessment_ids,
                "selector": selector,
                "limits": limits.model_dump(),
            },
        )
        atomic_json(
            folder / "manifest.json",
            {k: state[k] for k in ["run_id", "method", "provider", "model", "schema_hash", "system_hash"]},
        )
    domain = DomainTools(root, assessment_ids, store, limits.max_scenarios)
    domain.scenario_count = state["scenario_count"]
    bundle_hashes = {a: domain.bundle(a)[1]["bundle_hash"] for a in assessment_ids}
    if state.get("bundle_hashes", bundle_hashes) != bundle_hashes:
        raise ValueError("Assessment changed since checkpoint")
    state["bundle_hashes"] = bundle_hashes
    if method == "fixed_evidence" and not state.get("packet_prepared"):
        from .baseline import fixed_evidence_packet

        packet = fixed_evidence_packet(domain, assessment_ids)
        atomic_json(folder / "evidence_packet.json", packet)
        serialized = json.dumps(packet, ensure_ascii=False)
        state["messages"][0]["content"] += (
            "\nSingle-response fixed-evidence baseline. Use only this untrusted evidence packet; "
            "call submit_findings once. No adaptive tools or corrections are available.\n" + serialized
        )
        state["packet_bytes"] = len(serialized.encode())
        state["packet_prepared"] = True
    if state["status"] in {"completed", "cancelled"}:
        return state

    def checkpoint():
        state["scenario_count"] = domain.scenario_count
        store.save("checkpoints", run_id, state)
        atomic_json(
            folder / "status.json",
            {k: v for k, v in state.items() if k not in {"messages", "pending", "pending_results"}},
        )

    state["status"] = "running"
    checkpoint()
    try:
        while state["pending"] or state["turns"] < limits.max_turns:
            if (folder / "cancel.request").exists():
                state.update(status="cancelled", reason="Cancelled by user")
                break
            if not state["pending"] and state["pending_results"]:
                state["messages"].append({"role": "user", "content": state["pending_results"]})
                state["pending_results"] = []
                checkpoint()
            if not state["pending"]:
                if state["tool_count"] >= limits.max_tools:
                    state["reason"] = "Tool budget exhausted"
                    break
                reservation = 0.0
                if client.name not in {"demo", "fixture"}:
                    if (
                        min(
                            limits.spend_ceiling_usd,
                            limits.input_usd_per_million,
                            limits.output_usd_per_million,
                        )
                        <= 0
                    ):
                        raise ValueError(
                            "Live runs require a spend ceiling and explicit positive token prices"
                        )
                    count = client.count_input(state["messages"], tools_schema)
                    reservation = (
                        count * limits.input_usd_per_million
                        + limits.max_output_tokens * limits.output_usd_per_million
                    ) / 1e6
                    if state["cost_usd"] + state["reserved_usd"] + reservation > limits.spend_ceiling_usd:
                        state["reason"] = "Spend ceiling would be exceeded"
                        break
                state["reserved_usd"] += reservation
                state["request_inflight"] = True
                checkpoint()  # durable reservation before any billable dispatch
                response = client.complete(state["messages"], tools_schema, limits)
                actual_cost = (
                    response["input_tokens"] * limits.input_usd_per_million
                    + response["output_tokens"] * limits.output_usd_per_million
                ) / 1e6
                state["request_inflight"] = False
                state["reserved_usd"] -= reservation
                state["cost_usd"] += actual_cost
                state["input_tokens"] += response["input_tokens"]
                state["output_tokens"] += response["output_tokens"]
                state["turns"] += 1
                calls = response.get("calls", [])
                content = [{"type": "text", "text": response["text"]}] if response.get("text") else []
                content.extend(
                    {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
                    for call in calls
                )
                message = {"role": "assistant", "content": content or [{"type": "text", "text": "No action"}]}
                if response.get("reasoning_content") is not None:
                    message["reasoning_content"] = response["reasoning_content"]
                state["messages"].append(message)
                state["pending"], state["pending_results"] = calls, []
                _event(
                    folder,
                    {
                        "type": "model_turn",
                        "turn": state["turns"],
                        "response": response,
                        "cost_usd": actual_cost,
                    },
                )
                checkpoint()
                if not calls:
                    state["reason"] = "Model stopped without verified findings"
                    break
            while state["pending"]:
                call = state["pending"][0]
                if state["tool_count"] >= limits.max_tools:
                    state["pending"] = []
                    state["reason"] = "Tool budget exhausted"
                    break
                identity = hashlib.sha256(
                    json.dumps([state["turns"], call], sort_keys=True).encode()
                ).hexdigest()
                result = store.get("calls", identity)
                if result is None:
                    if call["name"] == "submit_findings":
                        try:
                            findings = Findings.model_validate(call["arguments"])
                            validation = verify_findings(findings, store, root)
                            if any(claim.kind == "numeric" for claim in findings.claims):
                                for assessment_id in assessment_ids:
                                    models = domain.get_assessment(assessment_id)["metadata"].get(
                                        "models", {}
                                    )
                                    if not models or any(
                                        not model.get("diagnostics_pass") for model in models.values()
                                    ):
                                        validation["passed"] = False
                                        validation["errors"].append(
                                            {
                                                "claim": "all",
                                                "error": "Posterior diagnostics unavailable or failed",
                                            }
                                        )
                            result = {
                                "status": "ok" if validation["passed"] else "invalid_arguments",
                                "data": validation,
                            }
                            atomic_json(folder / "validation.json", validation)
                            if validation["passed"]:
                                for claim in findings.claims:
                                    if claim.kind in {"numeric", "metadata"}:
                                        item = store.get("evidence", claim.evidence_ids[0])
                                        claim.text = str(item.get("selector", {}))
                                atomic_json(folder / "report.json", findings.model_dump())
                                dates = [str(domain.bundle(a)[1]["assessment_date"]) for a in assessment_ids]
                                (folder / "report.html").write_text(render_report(findings, store, dates))
                                state["status"] = "completed"
                            else:
                                state["repairs"] += 1
                        except ValueError as exc:
                            state["repairs"] += 1
                            result = {"status": "invalid_arguments", "error": str(exc)}
                    elif method == "fixed_evidence":
                        result = {
                            "status": "unavailable",
                            "error": "Fixed-evidence baseline only permits submit_findings",
                        }
                    else:
                        # Reserve compute attempts durably, including crashes during scenarios.
                        if call["name"] == "run_scenario":
                            state["scenario_count"] = domain.scenario_count + 1
                            store.save("checkpoints", run_id, state)
                        result = domain.execute(call["name"], call["arguments"])
                    store.save("calls", identity, result)
                if call["name"] == "submit_findings" and result.get("data", {}).get("passed"):
                    state["status"] = "completed"
                state["tool_count"] += 1
                state["pending_results"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, allow_nan=False),
                    }
                )
                state["pending"].pop(0)
                _event(folder, {"type": "tool_result", "call": call, "result": result})
                checkpoint()
                if state["status"] == "completed" or state["repairs"] > limits.max_repairs:
                    break
            if state["status"] == "completed" or state["repairs"] > limits.max_repairs:
                break
            if state["pending_results"]:
                state["messages"].append({"role": "user", "content": state["pending_results"]})
                state["pending_results"] = []
                checkpoint()
        if state["status"] == "running":
            state.update(
                status="incomplete", reason=state.get("reason", "Execution or correction budget exhausted")
            )
    except Exception as exc:
        state.update(status="failed", reason=str(exc))
        _event(folder, {"type": "failure", "error": str(exc)})
    finally:
        checkpoint()
        atomic_json(folder / "evidence.json", store.all_evidence())
        atomic_json(
            folder / "usage.json",
            {
                k: state[k]
                for k in [
                    "provider",
                    "model",
                    "cost_usd",
                    "reserved_usd",
                    "input_tokens",
                    "output_tokens",
                    "turns",
                    "tool_count",
                ]
            },
        )
    return state

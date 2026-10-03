"""CLI adapters; provider credentials remain environment variables."""

import json
from pathlib import Path

from ..assessment_bundle import freeze_assessment, safe_id
from .provider import AnthropicClient, DemoClient, FixtureClient
from .runtime import run_agent
from .schemas import Limits, Selector
from .store import Store
from .tools import DomainTools


def add_parser(subparsers):
    parser = subparsers.add_parser("agent", help="Evidence-grounded ENSO investigations")
    commands = parser.add_subparsers(dest="agent_command", required=True)
    freeze = commands.add_parser(
        "freeze", help="Import an existing successful assessment for read-only investigation"
    )
    freeze.add_argument("--assessment", required=True)
    investigate = commands.add_parser("investigate")
    investigate.add_argument("--assessment", required=True, action="append")
    investigate.add_argument("--question", required=True)
    investigate.add_argument("--country", default="IDN")
    investigate.add_argument("--scope", choices=["all_modeled", "excluding_Australia", "Pacific_islands"])
    investigate.add_argument("--period", default="2027", choices=["2026", "2027", "2026–2027"])
    investigate.add_argument(
        "--metric",
        default="revenue_loss_2025usd_mean",
        choices=["revenue_loss_2025usd_mean", "p_revenue_loss", "var95_2025usd", "es95_2025usd"],
    )
    investigate.add_argument("--run")
    investigate.add_argument("--max-turns", type=int, default=12)
    investigate.add_argument("--max-output-tokens", type=int, default=4096)
    investigate.add_argument("--spend-ceiling", type=float, default=0.0)
    investigate.add_argument("--input-price", type=float, default=0.0, help="USD per million input tokens")
    investigate.add_argument("--output-price", type=float, default=0.0, help="USD per million output tokens")
    resume = commands.add_parser("resume")
    resume.add_argument("--run", required=True)
    for command in [investigate, resume]:
        command.add_argument("--method", choices=["agent", "fixed_evidence"], default="agent")
        command.add_argument("--provider", choices=["demo", "fixture", "zai", "anthropic"], default="demo")
        command.add_argument("--model", default="glm-5.3")
        command.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="high")
        command.add_argument("--fixture", type=Path)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--run", required=True)
    evaluate = commands.add_parser("evaluate", help="Offline tool-contract checks, not LLM task-success")
    evaluate.add_argument("--assessment", required=True)
    evaluate.add_argument("--suite", type=Path, default=Path("evaluations/tool_contracts.json"))


def main(args):
    root = args.root.resolve()
    if args.agent_command == "freeze":
        return {"bundle": str(freeze_assessment(root, args.assessment))}
    if args.agent_command == "cancel":
        folder = root / "artifacts/agent_runs" / safe_id(args.run)
        if not folder.is_dir():
            raise ValueError("Unknown run")
        (folder / "cancel.request").touch()
        return {"run_id": args.run, "cancellation_requested": True}
    if args.agent_command == "evaluate":
        freeze_assessment(root, args.assessment)
        suite = json.loads(args.suite.read_text())
        store = Store(root / "artifacts/agent_evaluation/state.sqlite")
        try:
            domain = DomainTools(root, [args.assessment], store, max_scenarios=0)
            results = []
            for case in suite:
                arguments = json.loads(json.dumps(case["arguments"]).replace("$ASSESSMENT", args.assessment))
                result = domain.execute(case["tool"], arguments)
                results.append(
                    {
                        "id": case["id"],
                        "passed": result["status"] == case["expected_status"],
                        "status": result["status"],
                    }
                )
            return {
                "kind": "offline_tool_contract_checks",
                "llm_task_success": "not measured",
                "passed": sum(row["passed"] for row in results),
                "total": len(results),
                "results": results,
            }
        finally:
            store.close()
    if args.agent_command == "resume":
        folder = root / "artifacts/agent_runs" / safe_id(args.run)
        request = json.loads((folder / "request.json").read_text())
        selector, assessment_ids = request["selector"], request["assessment_ids"]
        question, limits = request["question"], Limits.model_validate(request["limits"])
    else:
        assessment_ids = args.assessment
        for assessment_id in assessment_ids:
            freeze_assessment(root, assessment_id)
        selector = Selector(
            assessment_id=assessment_ids[0],
            country=None if args.scope else args.country,
            scope=args.scope,
            period=args.period,
            metric=args.metric,
        ).model_dump()
        question = args.question
        limits = Limits(
            max_turns=args.max_turns,
            max_output_tokens=args.max_output_tokens,
            spend_ceiling_usd=args.spend_ceiling,
            input_usd_per_million=args.input_price,
            output_usd_per_million=args.output_price,
        )
    if args.provider == "demo":
        client = DemoClient(selector)
    elif args.provider == "fixture":
        if args.fixture is None:
            raise ValueError("--fixture is required")
        client = FixtureClient.from_path(args.fixture)
    elif args.provider == "zai":
        from .zai_provider import ZaiClient

        client = ZaiClient(model=args.model, reasoning_effort=args.reasoning_effort)
    else:
        if args.model == "glm-5.3":
            raise ValueError("Choose an explicit Anthropic model with --model")
        client = AnthropicClient(args.model)
    result = run_agent(
        root,
        assessment_ids,
        question,
        client,
        limits=limits,
        run_id=args.run,
        resume=args.agent_command == "resume",
        method=args.method,
        selector=selector,
    )
    return {
        key: value for key, value in result.items() if key not in {"messages", "pending", "pending_results"}
    }

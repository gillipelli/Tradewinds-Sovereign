"""Command line entry point; all paths are relative to --root."""

import argparse
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("ingest")
    fetch.add_argument("--refresh", action="store_true")
    sub.add_parser("build")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--out", type=Path, required=True)
    fit_parser.add_argument("--spec", choices=["total", "weather"], default="total")
    fit_parser.add_argument("--draws", type=int)
    fit_parser.add_argument("--tune", type=int)
    fit_parser.add_argument("--chains", type=int)
    fit_parser.add_argument("--cutoff", type=int)
    sub.add_parser("analyze")
    sub.add_parser("monitor")
    risk_parser = sub.add_parser("risk")
    risk_parser.add_argument("--model", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--model", type=Path)
    holdout = sub.add_parser("holdout")
    holdout.add_argument("--model", type=Path, required=True)
    upd = sub.add_parser("update")
    upd.add_argument("--cached", action="store_true")
    upd.add_argument("--no-refit", action="store_true")
    event = sub.add_parser("event", help="Refresh the 2026–2027 event assessment")
    event.add_argument("--cached", action="store_true")
    event.add_argument("--no-refit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "ingest":
        from .data import ingest

        print(ingest(args.root, args.refresh)["rows"])
    elif args.command == "build":
        from .data import build_panel

        print(build_panel(args.root).shape)
    elif args.command == "fit":
        from .model import fit

        print(
            fit(args.root, args.root / args.out, args.spec, args.draws, args.tune, args.chains, args.cutoff)
        )
    elif args.command == "analyze":
        from .analysis import backtest, descriptive, robustness

        out = args.root / "reports"
        out.mkdir(exist_ok=True)
        print(backtest(args.root, out))
        descriptive(args.root, out)
        robustness(args.root, out)
    elif args.command == "monitor":
        from .monitor import outlook

        print(outlook(args.root, args.root / "reports"))
    elif args.command == "risk":
        from .risk import simulate

        print(simulate(args.root, args.root / args.model, args.root / "reports"))
    elif args.command == "report":
        from .report import render

        print(render(args.root, args.root / "reports", args.root / args.model if args.model else None))
    elif args.command == "holdout":
        from .analysis import bayesian_holdout
        from .data import atomic_json

        result = bayesian_holdout(args.root, args.root / args.model, args.root / "reports")
        atomic_json(args.root / "reports/bayesian_holdout_metrics.json", result)
        print(result)
    elif args.command in ["update", "event"]:
        from .operations import update

        print(update(args.root, not args.cached, not args.no_refit))


if __name__ == "__main__":
    main()

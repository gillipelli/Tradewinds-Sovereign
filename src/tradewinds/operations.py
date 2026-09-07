"""Idempotent update orchestration with a process lock and durable run metadata."""

import fcntl
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import atomic_json, build_panel, config, ingest


def fingerprint(root, frame):
    # Include auxiliary targets, flags and weather: yield can be unchanged while
    # revised area or weather changes another analysis or risk component.
    h = hashlib.sha256(frame.to_csv(index=False).encode())
    h.update(json.dumps(config(root), sort_keys=True).encode())
    for source in sorted((root / "src/tradewinds").glob("*.py")):
        h.update(source.read_bytes())
    if (root / "uv.lock").exists():
        h.update((root / "uv.lock").read_bytes())
    return h.hexdigest()


def update(root: Path, refresh=True, refit=True):
    from .analysis import backtest, descriptive, robustness
    from .model import fit, modeling_panel
    from .monitor import outlook
    from .report import render
    from .risk import simulate

    artifact = root / "artifacts"
    artifact.mkdir(exist_ok=True)
    with (artifact / "update.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another update is in progress") from exc
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = artifact / "runs" / stamp
        run.mkdir(parents=True)
        state_path = artifact / "state.json"
        old = json.loads(state_path.read_text()) if state_path.exists() else {}
        try:
            snapshot = ingest(root, refresh)
            atomic_json(run / "snapshot.json", snapshot)
            build_panel(root)
            panel = modeling_panel(root)
            digest = fingerprint(root, panel)
            previous_model = Path(old["model"]) if old.get("model") else None
            model = previous_model
            changed = old.get("training_fingerprint") != digest
            if changed and refit:
                model = artifact / "models" / stamp
                fit(root, model)
            report_dir = run / "reports"
            report_dir.mkdir()
            # Cheap analyses also depend on prices/flags beyond the Bayesian target;
            # refresh them even when unchanged yield training allows model reuse.
            backtest(root, report_dir)
            descriptive(root, report_dir)
            robustness(root, report_dir)
            outlook(root, report_dir)
            risk_status = "unavailable"
            if model is not None:
                diagnostics = json.loads((model / "fit.json").read_text())
                if diagnostics["diagnostics_pass"] and (not changed or refit):
                    simulate(root, model, report_dir)
                    risk_status = "research_sensitivity"
                else:
                    risk_status = "withheld_failed_diagnostics_or_stale_model"
            render(root, report_dir, model)
            state = {
                "completed_at": stamp,
                "training_fingerprint": digest if refit or not changed else old.get("training_fingerprint"),
                "model": str(model) if model else None,
                "report_dir": str(report_dir),
                "risk_status": risk_status,
                "training_changed": changed,
                "git_revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
                ).stdout.strip(),
                "status": "complete",
            }
            atomic_json(run / "run.json", state)
            atomic_json(state_path, state)
            return state
        except Exception as exc:
            atomic_json(
                run / "run.json", {"status": "failed", "error": str(exc), "prior_success_preserved": True}
            )
            raise

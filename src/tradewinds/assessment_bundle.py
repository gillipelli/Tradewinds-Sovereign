"""Content-verified assessments; historical reports never inherit today's mutable inputs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from .data import atomic_json


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Invalid artifact identifier")
    return value


def bundle_path(root: Path, assessment_id: str) -> Path:
    return root / "artifacts/assessments" / safe_id(assessment_id)


def _copy_tree(source: Path, destination: Path):
    for source_file in source.rglob("*"):
        if source_file.is_symlink():
            raise ValueError(f"Symlinks are not permitted in bundles: {source_file}")
        if source_file.is_file():
            target = destination / source_file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)


def freeze_assessment(root: Path, successful_run_id: str, *, publication_state: dict | None = None) -> Path:
    """Legacy runs are read-only; only the publishing pipeline may attest live inputs.

    publication_state is an internal pipeline argument used while holding update.lock,
    before run.json is published. The public CLI never supplies it.
    """
    root = root.resolve()
    assessment_id = safe_id(successful_run_id)
    target = bundle_path(root, assessment_id)
    if target.exists():
        validate_bundle(target)
        return target
    run = root / "artifacts/runs" / assessment_id
    state = publication_state or json.loads((run / "run.json").read_text())
    if state.get("status") != "complete":
        raise ValueError("Only successful assessments can be frozen")
    reports = run / "reports"
    metadata = json.loads((reports / "event_risk_metadata.json").read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-", dir=target.parent))
    try:
        _copy_tree(reports, temporary / "reports")
        if (root / "docs").exists():
            _copy_tree(root / "docs", temporary / "docs")
        atomic_json(temporary / "run.json", state)
        recomputable = publication_state is not None
        if recomputable:
            for name in ["data/event", "data/processed"]:
                _copy_tree(root / name, temporary / name)
            cfg = yaml.safe_load((root / "configs/project.yaml").read_text())
            cfg["event"]["as_of"] = metadata["climate"]["as_of"]
            (temporary / "configs").mkdir()
            (temporary / "configs/project.yaml").write_text(yaml.safe_dump(cfg))
            for kind in ["agriculture", "fiscal"]:
                _copy_tree(Path(state["macro_models"][kind]), temporary / "models" / kind)
            for name in ["uv.lock", "pyproject.toml"]:
                if (root / name).exists():
                    shutil.copy2(root / name, temporary / name)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        code_hashes = {str(p.relative_to(root)): digest(p) for p in (root / "src/tradewinds").rglob("*.py")}
        files = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob("*") if p.is_file()}
        manifest = {
            "schema_version": 1,
            "assessment_id": assessment_id,
            "assessment_date": metadata["climate"].get("as_of"),
            "recomputable": recomputable,
            "limitation": None
            if recomputable
            else "Historical input provenance is incomplete; inspection only.",
            "code_revision": revision.stdout.strip() or "unknown",
            "code_hashes": code_hashes,
            "files": files,
        }
        manifest["bundle_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def validate_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected = manifest.pop("bundle_hash")
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError("Bundle manifest hash mismatch")
    manifest["bundle_hash"] = expected
    for relative, expected_hash in manifest["files"].items():
        path = bundle / relative
        if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Unsafe bundle path")
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"Bundle integrity failure: {relative}")
    return manifest


def create_scenario_workspace(root: Path, assessment_id: str, scenario_id: str) -> Path:
    bundle = bundle_path(root, assessment_id)
    manifest = validate_bundle(bundle)
    if not manifest["recomputable"]:
        raise ValueError(manifest["limitation"])
    # Changed science requires a new assessment, never a silent historical replay.
    for name, expected in manifest["code_hashes"].items():
        if "/agents/" not in name and not name.endswith(("cli.py", "assessment_bundle.py")):
            if not (root / name).exists() or digest(root / name) != expected:
                raise ValueError(f"Scientific code changed since assessment: {name}")
    workspace = root / "artifacts/scenarios" / safe_id(assessment_id) / safe_id(scenario_id)
    workspace.mkdir(parents=True, exist_ok=False)
    for name in ["data", "configs", "models"]:
        _copy_tree(bundle / name, workspace / name)
    return workspace


def resolve_artifact(root: Path, assessment_id: str, artifact: str, scenario_id: str | None = None):
    bundle = bundle_path(root, assessment_id)
    manifest = validate_bundle(bundle)
    if scenario_id:
        workspace = root / "artifacts/scenarios" / safe_id(assessment_id) / safe_id(scenario_id)
        completion = json.loads((workspace / "complete.json").read_text())
        if completion["bundle_hash"] != manifest["bundle_hash"]:
            raise ValueError("Scenario bundle lineage mismatch")
        hashes, base = completion["hashes"], workspace
    else:
        hashes, base = manifest["files"], bundle
    path = base / artifact
    if artifact not in hashes or not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
        raise ValueError("Artifact is outside the verified manifest")
    if digest(path) != hashes[artifact]:
        raise ValueError("Artifact integrity failure")
    return path, manifest

"""Verify numeric claims against immutable source cells; prose needs human scientific review."""

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from ..assessment_bundle import bundle_path, digest, resolve_artifact, validate_bundle


def conversion_factor(source, target):
    groups = [{"2025 USD": 1., "million 2025 USD": 1e6, "billion 2025 USD": 1e9},
              {"probability": 1., "percent": .01}]
    for scales in groups:
        if source in scales and target in scales:
            return scales[source] / scales[target]
    raise ValueError("Unsupported or incompatible unit conversion")


def verify_evidence(key, store, root, seen=None):
    seen = set() if seen is None else set(seen)
    if key in seen:
        raise ValueError("Cyclic evidence")
    seen.add(key)
    item = store.get("evidence", key)
    if not item:
        raise ValueError(f"Unknown evidence: {key}")
    actual_key = hashlib.sha256(json.dumps(item, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
    if actual_key != key:
        raise ValueError("Evidence payload hash mismatch")
    if item.get("calculation") == "convert_units":
        if len(item["operands"]) != 1:
            raise ValueError("Conversion requires one operand")
        original = verify_evidence(item["operands"][0], store, root, seen)
        factor = conversion_factor(original["unit"], item["unit"])
        if item["selector"] != original["selector"] or not math.isclose(
            item["value"], original["value"] * factor, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Conversion mismatch")
        return item
    if item.get("calculation") == "subtract":
        left, right = [verify_evidence(k, store, root, seen) for k in item["operands"]]
        for field in ["metric", "country", "scope", "year", "period"]:
            if left.get("selector", {}).get(field) != right.get("selector", {}).get(field):
                raise ValueError("Calculation selector mismatch")
        if left["unit"] != right["unit"] or item["unit"] != left["unit"]:
            raise ValueError("Calculation unit mismatch")
        if not math.isclose(item["value"], left["value"] - right["value"], abs_tol=1e-8):
            raise ValueError("Calculation mismatch")
        return item
    bundle = bundle_path(root, item["assessment_id"])
    manifest = validate_bundle(bundle)
    if item["bundle_hash"] != manifest["bundle_hash"]:
        raise ValueError("Evidence bundle mismatch")
    artifact = item["artifact"]
    path, _ = resolve_artifact(root, item["assessment_id"], artifact, item.get("scenario_id"))
    if digest(path) != item["artifact_hash"]:
        raise ValueError("Evidence artifact hash mismatch")
    selector = item["selector"]
    if "metric" in selector:
        frame = pd.read_csv(path, dtype={"year": str, "period": str})
        for column, value in selector.items():
            if column != "metric":
                frame = frame[frame[column].eq(value)]
        if len(frame) != 1 or not math.isclose(
            float(frame.iloc[0][selector["metric"]]), item["value"], abs_tol=1e-8
        ):
            raise ValueError("Evidence cell mismatch")
        expected_unit = "probability" if selector["metric"].startswith("p_") else "2025 USD"
        if item["unit"] != expected_unit:
            raise ValueError("Evidence unit mismatch")
    elif path.suffix == ".csv" and selector == {"table": "all"}:
        content = json.loads(pd.read_csv(path).to_json(orient="records", double_precision=15))
        if content != item["value"]:
            raise ValueError("Evidence table content mismatch")
    elif path.suffix == ".json" and "retrieval_query" not in selector:
        value = json.loads(path.read_text())
        if "section" in selector:
            value = value[selector["section"]]
        if "json_path" in selector:
            allowed_units = {
                ("models", name, "prior_scale"): "prior scale" for name in ("fiscal", "agriculture")
            }
            allowed_units.update({("climate", field): ("ISO month" if field == "forecast_issue" else "ISO date")
                                  for field in ("as_of", "forecast_issue", "forecast_issue_date",
                                                "official_last_center_month", "latest_observed_center")})
            if allowed_units.get(tuple(selector["json_path"])) != item["unit"]:
                raise ValueError("Unsupported metadata parameter selector or unit")
            for field in selector["json_path"]:
                value = value[field]
        if value != item["value"]:
            raise ValueError("Evidence content mismatch")
    return item


def unsupported_numerals(text, evidence):
    """Allow source-supported temporal context and fixed currency labels, never free numeric amounts."""
    periods, assessment_ids, dates = set(), set(), set()
    for item in evidence:
        selector = item.get("selector", {})
        for field in ("year", "period"):
            if selector.get(field):
                periods.add(str(selector[field]))
        if item.get("assessment_id"):
            assessment_ids.add(item["assessment_id"])
        value = item.get("value")
        if isinstance(value, dict):
            climate = value.get("climate", {})
            if isinstance(climate, dict):
                for field in ("as_of", "forecast_issue_date", "official_last_center_month",
                              "counterfactual_start", "latest_observed_center"):
                    date = climate.get(field)
                    if isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                        dates.add(date)
    # Currency basis is a fixed application contract, not a model-provided magnitude.
    text = re.sub(r"\bconstant\s+2025\s+(?:USD|dollars)\b", "currency basis", text, flags=re.I)
    text = re.sub(r"\bfixed\s+2025\s+(?:exchange rate|FX)\b", "fixed base-year exchange rate", text, flags=re.I)
    for period in periods:
        escaped = re.escape(period)
        text = re.sub(
            r"\b(?:in|for|during)\s+(?:the\s+)?(?:year|period)\s+" + escaped
            + r"\b(?!\s*(?:dollars|USD|events|people|cases|percent|%|million|billion)\b)",
            "period", text, flags=re.I,
        )
    for identifier in assessment_ids:
        text = re.sub(r"\bassessment\s+" + re.escape(identifier) + r"\b", "assessment", text, flags=re.I)
    for date in dates:
        text = re.sub(r"\b(?:as of|dated|issued on|through|starting on|until|assessment date[: ]*)\s*" + re.escape(date) + r"\b", "date", text, flags=re.I)
    return bool(re.search(r"\d", text))


def verify_findings(findings, store, root: Path):
    errors = []
    verified_context = []
    for claim in findings.claims:
        for key in claim.evidence_ids:
            try:
                verified_context.append(verify_evidence(key, store, root))
            except (ValueError, KeyError, TypeError):
                pass  # Individual claim verification below reports the underlying error.
    fields = [("title", findings.title), *[(f"limitations[{i}]", text) for i, text in enumerate(findings.limitations)]]
    for field, text in fields:
        if unsupported_numerals(text, verified_context):
            errors.append({"claim": "narrative", "field": field,
                           "error": "Remove unsupported digits from this field; use a verified numeric claim for amounts."})
    for index, claim in enumerate(findings.claims):
        try:
            evidence = [verify_evidence(key, store, root) for key in claim.evidence_ids]
            if claim.kind == "numeric":
                if len(evidence) != 1 or not isinstance(evidence[0]["value"], (float, int)):
                    raise ValueError("Numeric claims require exactly one numeric cell or calculation")
                actual = evidence[0]
                if (
                    claim.value is None
                    or claim.unit != actual["unit"]
                    or not math.isclose(claim.value, actual["value"], rel_tol=1e-12, abs_tol=1e-8)
                ):
                    raise ValueError("Numeric value or unit does not match evidence")
            elif claim.kind == "metadata":
                date_paths = [["climate", field] for field in ("as_of", "forecast_issue", "forecast_issue_date",
                                                              "official_last_center_month", "latest_observed_center")]
                if (len(evidence) != 1 or not isinstance(evidence[0]["value"], str)
                        or evidence[0].get("selector", {}).get("json_path") not in date_paths
                        or not evidence[0].get("artifact", "").endswith(".json")
                        or evidence[0].get("unit") not in {"ISO date", "ISO month"}
                        or claim.value != evidence[0]["value"] or claim.unit != evidence[0]["unit"]):
                    raise ValueError("Metadata claims require exactly one matching verified date or month")
            elif claim.value is not None or unsupported_numerals(claim.text, evidence):
                raise ValueError("Numbers must be submitted as separate verified numeric claims")
            if re.search(r"\b(proves?|caused|guaranteed|will lose)\b", claim.text, flags=re.I):
                raise ValueError("Unsupported causal or certainty language")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"claim": index, "error": str(exc)})
    return {
        "passed": not errors,
        "errors": errors,
        "semantic_review": "Descriptive and interpretive prose is not automatically scientifically validated.",
    }


def render_report(findings, store, assessment_dates):
    import html

    lines = [
        "<h1>ENSO investigation</h1>",
        "<p>Historical conditional assessment. Not an official budget forecast.</p>",
        "<p>Assessment dates: " + html.escape(", ".join(assessment_dates)) + "</p>",
        "<ul>",
    ]
    for claim in findings.claims:
        if claim.kind in {"numeric", "metadata"}:
            evidence = store.get("evidence", claim.evidence_ids[0])
            # Numeric labels come from source selectors, never model-authored prose.
            value = f"{claim.value:,.6g}" if claim.kind == "numeric" else str(claim.value)
            text = f"{evidence.get('selector', {})}: {value} {claim.unit}"
        else:
            text = f"{claim.kind.title()} (requires scientific review): {claim.text}"
        refs = ", ".join(f'<a href="#e-{key}">{key}</a>' for key in claim.evidence_ids)
        lines.append(f"<li>{html.escape(text)} {refs}</li>")
    lines += ["</ul><h2>Limitations</h2><ul>"]
    lines += [f"<li>{html.escape(value)}</li>" for value in findings.limitations]
    lines += ["</ul><h2>Evidence records</h2>"]
    for key, item in store.all_evidence().items():
        lines.append(
            f'<details id="e-{key}"><summary>{key}</summary><pre>'
            + html.escape(json.dumps(item, indent=2))
            + "</pre></details>"
        )
    return '<!doctype html><meta charset="utf-8"><title>ENSO investigation</title>' + "\n".join(lines)

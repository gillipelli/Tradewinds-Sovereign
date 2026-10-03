"""Deterministic bounded evidence packet for a single-response model baseline."""

import json

import pandas as pd


def fixed_evidence_packet(domain, assessment_ids, max_bytes=60000, max_cells=100):
    packet = {
        "method": "fixed_evidence",
        "entries": [],
        "truncated": False,
        "coverage": {"metadata_dates": "Allowlisted scalar dates subject to packet cap",
                     "scenario_parameters": "Not included as scalar parameter evidence",
                     "risk_cells": "Table order subject to packet and cell caps"},
        "selection": "Sorted authorized assessments (at most two); metadata, diagnostics, typed dates, fixed methodology search, then source risk cells in table order.",
        "limitation": "Fixed bounded packet; absent evidence is unavailable. No adaptive retrieval or scenario computation.",
    }

    def append(kind, value):
        entry = {"kind": kind, "result": value}
        candidate = {**packet, "entries": packet["entries"] + [entry]}
        if len(json.dumps(candidate, ensure_ascii=False).encode()) > max_bytes:
            packet["truncated"] = True
            return False
        packet["entries"].append(entry)
        return True

    cells = 0
    for assessment_id in sorted(assessment_ids)[:2]:
        append("assessment", domain.get_assessment(assessment_id))
        append("diagnostics", domain.get_model_diagnostics(assessment_id))
        for field in ("as_of", "forecast_issue", "forecast_issue_date", "official_last_center_month", "latest_observed_center"):
            result = domain.execute("query_assessment_date", {"assessment_id": assessment_id, "field": field})
            if result["status"] == "ok":
                append("metadata_date", result["data"])
        append(
            "methodology",
            domain.search_evidence(assessment_id, "forecast conditional revenue uncertainty limitations", 3),
        )
        bundle, _ = domain.bundle(assessment_id)
        for country_mode, name in [(False, "event_aggregate_risk.csv"), (True, "event_country_risk.csv")]:
            path = bundle / "reports" / name
            if not path.exists():
                continue
            for row in pd.read_csv(path, dtype={"year": str, "period": str}).to_dict("records"):
                selector = (
                    {"country": row["country"], "period": row["year"]}
                    if country_mode
                    else {"scope": row["scope"], "period": row["period"]}
                )
                for metric in ["revenue_loss_2025usd_mean", "p_revenue_loss", "es95_2025usd"]:
                    if cells >= max_cells:
                        packet["truncated"] = True
                        break
                    result = domain.execute(
                        "query_risk", {"assessment_id": assessment_id, **selector, "metric": metric}
                    )
                    if result["status"] == "ok":
                        if not append("risk_cell", result["data"]):
                            break
                        cells += 1
    packet["truncated"] = packet["truncated"] or len(assessment_ids) > 2
    packet["numeric_cells"] = cells
    return packet

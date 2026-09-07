"""Official-source ingestion, content-addressed vintages and strict normalization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from lxml import html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)
SEASONS = "DJF JFM FMA MAM AMJ MJJ JJA JAS ASO SON OND NDJ".split()
URLS = {
    "catalog": "https://bulks-faostat.fao.org/production/datasets_E.json",
    "roni": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "pink": "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx",
}
INDICATORS = {
    "gdp_usd": "NY.GDP.MKTP.CD",
    "revenue_pct_gdp": "GC.REV.XGRT.GD.ZS",
    "agriculture_pct_gdp": "NV.AGR.TOTL.ZS",
    "real_gdp_growth": "NY.GDP.MKTP.KD.ZG",
}


def config(root: Path) -> dict:
    return yaml.safe_load((root / "configs/project.yaml").read_text())


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, default=str, allow_nan=False)
        temp = f.name
    os.replace(temp, path)


class Store:
    """Never overwrite a source vintage. Log both retrieval time and source metadata."""

    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "data/raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.raw / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Tradewinds-Sovereign/0.1 research data pipeline"
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch(self, name: str, url: str, refresh: bool = False) -> Path:
        prev = self.manifest.get(name, {})
        if prev and not refresh:
            p = self.raw / prev["file"]
            if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == prev["sha256"]:
                return p
            raise ValueError(f"Missing/corrupt cached source {name}; rerun with --refresh")
        headers = {}
        if prev.get("etag"):
            headers["If-None-Match"] = prev["etag"]
        LOG.info("Fetching %s", name)
        response = self.session.get(url, headers=headers, timeout=(30, 300))
        checked_at = datetime.now(timezone.utc).isoformat()
        if response.status_code == 304:
            p = self.raw / prev["file"]
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != prev["sha256"]:
                raise ValueError(f"Corrupt cached source after 304: {name}")
            prev["checked_at"] = checked_at
            atomic_json(self.manifest_path, self.manifest)
            return p
        response.raise_for_status()
        content = response.content
        if not content:
            raise ValueError(f"Empty response: {name}")
        digest = hashlib.sha256(content).hexdigest()
        folder = self.raw / name
        folder.mkdir(exist_ok=True)
        p = folder / f"{digest}.bin"
        if not p.exists():
            temp = p.with_suffix(".part")
            temp.write_bytes(content)
            temp.replace(p)
        record = {
            "url": url,
            "sha256": digest,
            "file": str(p.relative_to(self.raw)),
            "retrieved_at": prev.get("retrieved_at", checked_at)
            if prev.get("sha256") == digest
            else checked_at,
            "checked_at": checked_at,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "bytes": len(content),
        }
        self.manifest[name] = record
        atomic_json(self.manifest_path, self.manifest)
        with (self.raw / "retrievals.jsonl").open("a") as f:
            f.write(json.dumps({"source": name, **record}) + "\n")
        return p


def parse_roni(path: Path) -> pd.DataFrame:
    tree = html.fromstring(path.read_bytes())
    rows = []
    for tr in tree.xpath("//tr"):
        cells = [" ".join(c.itertext()).strip() for c in tr.xpath("./td|./th")]
        if not 2 <= len(cells) <= 13 or not re.fullmatch(r"\d{4}", cells[0]):
            continue
        for season, val in zip(SEASONS, cells[1:]):
            try:
                x = float(val)
            except ValueError:
                continue
            if abs(x) < 10:
                rows.append((int(cells[0]), season, x))
    out = pd.DataFrame(rows, columns=["year", "season", "enso"])
    if len(out) < 600:
        raise ValueError("NOAA RONI schema changed or insufficient historical data")
    out["index"] = "roni"
    return season_dates(out)


def season_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df.season.map({s: i + 1 for i, s in enumerate(SEASONS)})
    df["period"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    # A centered 3-month season cannot be known in its center month.
    df["assumed_available_at"] = df.period + pd.DateOffset(months=2, days=4)
    if df.duplicated(["year", "season"]).any():
        raise ValueError("Duplicate ENSO seasons")
    return df.sort_values("period")


def parse_oni(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+").rename(columns={"YR": "year", "SEAS": "season", "ANOM": "enso"})
    df = df.loc[df.enso.abs() < 10, ["year", "season", "enso"]]
    df["index"] = "oni"
    return season_dates(df)


def fao_subset(path: Path, cfg: dict) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "_All_Data" in n]
        if len(names) != 1:
            raise ValueError(f"Unexpected FAOSTAT archive members: {z.namelist()}")
        with z.open(names[0]) as f:
            for chunk in pd.read_csv(f, encoding="utf-8-sig", chunksize=150_000, low_memory=False):
                required = {"Area", "Item", "Element", "Year", "Unit", "Value", "Flag"}
                if not required.issubset(chunk.columns):
                    raise ValueError(f"Missing FAO fields: {required - set(chunk.columns)}")
                m = chunk.Area.isin(cfg["countries"].values()) & chunk.Item.isin(cfg["crops"].values())
                pieces.append(chunk.loc[m].copy())
    df = pd.concat(pieces, ignore_index=True)
    if df.empty:
        raise ValueError("FAO subset empty; verify official item/country names")
    df["country"] = df.Area.map({v: k for k, v in cfg["countries"].items()})
    df["crop"] = df.Item.map({v: k for k, v in cfg["crops"].items()})
    df["year"] = pd.to_numeric(df.Year, errors="raise").astype(int)
    df["value"] = pd.to_numeric(df.Value, errors="coerce")
    return df.loc[df.year >= cfg["start_year"]]


def parse_pink(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    first = next((i for i, v in enumerate(raw.iloc[:, 0]) if re.fullmatch(r"\d{4}M\d{2}", str(v))), None)
    if first is None:
        raise ValueError("Pink Sheet monthly date schema changed")
    # Headers precede units; detect the commodity-label row rather than assuming a row number.
    header_row = next(
        (i for i in range(first) if raw.iloc[i].astype(str).str.contains("Coconut oil", case=False).any()),
        None,
    )
    if header_row is None:
        raise ValueError("Pink Sheet commodity headers missing")
    headers = raw.iloc[header_row].tolist()
    frame = raw.iloc[first:].copy()
    frame.columns = ["period", *[str(x).strip() for x in headers[1:]]]
    frame = frame[frame.period.astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    frame["period"] = pd.to_datetime(frame.period, format="%YM%m")
    long = frame.melt(id_vars="period", var_name="commodity", value_name="price")
    long["price"] = pd.to_numeric(long.price, errors="coerce")
    long = long.dropna().loc[lambda d: d.price > 0]
    return long


def ingest(root: Path, refresh: bool = False) -> dict:
    cfg = config(root)
    store = Store(root)
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    frames["roni"] = parse_roni(store.fetch("roni", URLS["roni"], refresh))
    frames["oni"] = parse_oni(store.fetch("oni", URLS["oni"], refresh))
    cat = json.loads(store.fetch("catalog", URLS["catalog"], refresh).read_text())
    entries = cat["Datasets"]["Dataset"]
    for code in ["QCL", "QV", "PP"]:
        entry = next(e for e in entries if e["DatasetCode"] == code)
        frames[code.lower()] = fao_subset(store.fetch(code.lower(), entry["FileLocation"], refresh), cfg)
    frames["pink"] = parse_pink(store.fetch("pink", URLS["pink"], refresh))
    wdi = []
    for label, indicator in INDICATORS.items():
        countries = ";".join(cfg["countries"])
        url = f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&per_page=20000&date={cfg['start_year']}:2100"
        body = json.loads(store.fetch(label, url, refresh).read_text())
        if (
            not isinstance(body, list)
            or len(body) != 2
            or not isinstance(body[0], dict)
            or body[0].get("pages") != 1
        ):
            raise ValueError(f"Unexpected WDI response/pagination for {label}")
        for r in body[1] or []:
            if r["value"] is not None:
                wdi.append(
                    {
                        "country": r["countryiso3code"],
                        "year": int(r["date"]),
                        "indicator": label,
                        "value": r["value"],
                    }
                )
    frames["wdi"] = pd.DataFrame(wdi)
    from tradewinds.weather import ingest_weather

    frames["weather"] = ingest_weather(store, cfg, refresh)
    # Publish processed tables only after every required source parses successfully.
    for name, df in frames.items():
        temp = out / f"{name}.csv.tmp"
        df.to_csv(temp, index=False)
        temp.replace(out / f"{name}.csv")
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": store.manifest,
        "rows": {k: len(v) for k, v in frames.items()},
    }
    atomic_json(out / "snapshot.json", snapshot)
    return snapshot


def annual_enso(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar center-year exposures; only full 12-season years enter annual training."""
    df = df.copy()
    df["warm"] = df.enso.clip(lower=0)
    df["cold"] = (-df.enso).clip(lower=0)
    df["extreme"] = (df.enso - 1.5).clip(lower=0)
    annual = df.groupby("year").agg(
        warm=("warm", "mean"),
        cold=("cold", "mean"),
        extreme=("extreme", "mean"),
        peak=("enso", "max"),
        seasons=("season", "nunique"),
    )
    annual = annual[annual.seasons == 12].drop(columns="seasons")
    for lag in [1, 2]:
        shifted = annual[["warm"]].rename(columns={"warm": f"warm_lag{lag}"})
        shifted.index = shifted.index + lag
        annual = annual.join(shifted)
    return annual.reset_index().dropna()


def build_panel(root: Path) -> pd.DataFrame:
    cfg = config(root)
    folder = root / "data/processed"
    q = pd.read_csv(folder / "qcl.csv")
    keys = ["country", "crop", "year"]
    q = q[q.Element.isin(["Production", "Area harvested"])]
    if q.duplicated(keys + ["Element"]).any():
        raise ValueError("Duplicate crop observations")
    if (
        not q[q.Element == "Production"].Unit.eq("t").all()
        or not q[q.Element == "Area harvested"].Unit.eq("ha").all()
    ):
        raise ValueError("Unexpected production/area units")
    panel = q.pivot(index=keys, columns="Element", values="value").reset_index()
    panel = panel.rename(columns={"Production": "production_t", "Area harvested": "area_ha"})
    flags = (
        q.pivot(index=keys, columns="Element", values="Flag")
        .reset_index()
        .rename(columns={"Production": "production_flag", "Area harvested": "area_flag"})
    )
    panel = panel.merge(flags, on=keys, validate="1:1")
    panel = panel[(panel.production_t > 0) & (panel.area_ha > 0)].copy()
    panel["yield_t_ha"] = panel.production_t / panel.area_ha
    panel = panel.sort_values(keys)
    for col, target in [("yield_t_ha", "yield_growth"), ("area_ha", "area_growth")]:
        panel[target] = panel.groupby(["country", "crop"])[col].transform(lambda s: np.log(s).diff())
    consecutive = panel.groupby(["country", "crop"]).year.diff().eq(1)
    panel.loc[~consecutive, ["yield_growth", "area_growth"]] = np.nan
    panel["lag_yield_growth"] = panel.groupby(["country", "crop"]).yield_growth.shift()
    panel.loc[~consecutive, "lag_yield_growth"] = np.nan
    enso = annual_enso(pd.read_csv(folder / f"{cfg['enso_index']}.csv"))
    panel = panel.merge(enso, on="year", validate="m:1")
    panel["series"] = panel.country + ":" + panel.crop
    panel["trend"] = (panel.year - 2000) / 10
    from tradewinds.weather import annual_weather

    panel = panel.merge(
        annual_weather(pd.read_csv(folder / "weather.csv")),
        on=["country", "year"],
        how="left",
        validate="m:1",
    )
    econ = pd.read_csv(folder / "wdi.csv")
    econ = econ[econ.indicator.eq("real_gdp_growth")][["country", "year", "value"]]
    econ["year"] += 1
    panel = panel.merge(
        econ.rename(columns={"value": "lag_gdp_growth"}), on=["country", "year"], how="left", validate="m:1"
    )
    counts = panel.groupby("series").yield_growth.count()
    latest = panel.groupby("series").year.max()
    eligible = counts.index[
        (counts >= cfg["min_years"]) & (latest >= panel.year.max() - cfg["max_recent_gap"])
    ]
    coverage = (
        panel.groupby(["country", "crop", "series"])
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("yield_growth", "count"),
            official_share=("production_flag", lambda s: s.eq("A").mean()),
        )
        .reset_index()
    )
    coverage["eligible"] = coverage.series.isin(eligible)
    coverage["evidence_grade"] = np.where(
        coverage.official_share >= 0.8,
        "mostly_official",
        np.where(coverage.official_share >= 0.5, "mixed", "mostly_estimated_or_imputed"),
    )
    coverage.to_csv(folder / "coverage.csv", index=False)
    panel["eligible"] = panel.series.isin(eligible)
    panel.to_csv(folder / "panel.csv", index=False)
    return panel

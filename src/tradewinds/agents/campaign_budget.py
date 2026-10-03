"""Shared SQLite campaign accounting. No network calls or credentials are stored.

Set ZAI_CAMPAIGN_LEDGER, ZAI_CAMPAIGN_PROJECT and ZAI_CAMPAIGN_RUN to opt in.
Initialize once before requests; configuration cannot silently reset on restart.
"""
from __future__ import annotations

import argparse
import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


class BudgetExceeded(RuntimeError):
    """A durable campaign allowance prevents another request."""


def _positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError("token limits must be positive integers")
    return value


@contextmanager
def _transaction(path):
    connection = sqlite3.connect(path, timeout=60, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(path, total_limit=18_000_000, project_limits=None):
    """Create an immutable campaign configuration; matching reopen is harmless."""
    _positive(total_limit)
    project_limits = dict(project_limits or {})
    if not project_limits or any(not isinstance(k, str) or not k for k in project_limits):
        raise ValueError("at least one named project limit is required")
    for value in project_limits.values():
        _positive(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.dumps(dict(total_limit=total_limit, project_limits=project_limits), sort_keys=True)
    with _transaction(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY, project TEXT NOT NULL, run_id TEXT NOT NULL,
            reserved INTEGER NOT NULL, charged INTEGER NOT NULL, status TEXT NOT NULL,
            input_tokens INTEGER, output_tokens INTEGER, usage_json TEXT,
            created REAL NOT NULL, updated REAL NOT NULL)""")
        existing = db.execute("SELECT value FROM config WHERE id=1").fetchone()
        if existing and existing[0] != config:
            raise ValueError("campaign configuration already exists and differs")
        db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (config,))
    return CampaignBudget(path)


class CampaignBudget:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError("initialize the campaign ledger before using it")

    def reserve_request(self, run_id, input_bound, max_output_tokens, project):
        amount = _positive(input_bound) + _positive(max_output_tokens)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        with _transaction(self.path) as db:
            config = json.loads(db.execute("SELECT value FROM config WHERE id=1").fetchone()[0])
            if project not in config["project_limits"]:
                raise ValueError("project has no campaign allowance")
            if db.execute("SELECT 1 FROM requests WHERE status='overrun' LIMIT 1").fetchone():
                raise BudgetExceeded("campaign halted after provider reservation overrun")
            total = db.execute("SELECT COALESCE(SUM(charged),0) FROM requests").fetchone()[0]
            subtotal = db.execute("SELECT COALESCE(SUM(charged),0) FROM requests WHERE project=?", (project,)).fetchone()[0]
            if total + amount > config["total_limit"] or subtotal + amount > config["project_limits"][project]:
                raise BudgetExceeded("campaign token ceiling prevents next request")
            request_id = uuid.uuid4().hex
            now = time.time()
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (request_id, project, run_id, amount, amount, "reserved", None, None, None, now, now))
        return request_id

    def record_usage(self, request_id, usage):
        if not isinstance(usage, dict):
            raise ValueError("usage must be an object")
        counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("invalid token usage")
        # completion_tokens includes reasoning; cached tokens remain fully charged.
        encoded = json.dumps(usage, allow_nan=False)
        with _transaction(self.path) as db:
            row = db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise ValueError("unknown request reservation")
            if row["status"] == "settled":
                if row["usage_json"] != encoded:
                    raise ValueError("request was already settled with different usage")
                return
            amount = sum(counts)
            status = "overrun" if amount > row["reserved"] else "settled"
            db.execute("UPDATE requests SET charged=?, status=?, input_tokens=?, output_tokens=?, usage_json=?, updated=? WHERE request_id=?",
                       (amount, status, counts[0], counts[1], encoded, time.time(), request_id))
        if status == "overrun":
            raise BudgetExceeded("provider usage exceeded the conservative reservation")

    def mark_uncertain(self, request_id):
        with _transaction(self.path) as db:
            # Settled usage must never be replaced by an exception handler.
            db.execute("UPDATE requests SET status='uncertain', updated=? WHERE request_id=? AND status='reserved'",
                       (time.time(), request_id))

    def export_usage_report(self):
        with _transaction(self.path) as db:
            config = json.loads(db.execute("SELECT value FROM config WHERE id=1").fetchone()[0])
            rows = [dict(row) for row in db.execute("SELECT * FROM requests ORDER BY created, request_id")]
        charged = sum(row["charged"] for row in rows)
        projects = {name: {"limit": limit, "charged": sum(row["charged"] for row in rows if row["project"] == name)}
                    for name, limit in config["project_limits"].items()}
        for value in projects.values():
            value["remaining"] = max(0, value["limit"] - value["charged"])
        return dict(total_limit=config["total_limit"], charged=charged,
                    remaining=max(0, config["total_limit"] - charged), projects=projects,
                    unresolved_reservations=sum(row["charged"] for row in rows if row["status"] in ("reserved", "uncertain")),
                    input_tokens=sum(row["input_tokens"] or 0 for row in rows),
                    output_tokens=sum(row["output_tokens"] or 0 for row in rows), requests=rows)

    def remaining_allowance(self):
        return self.export_usage_report()["remaining"]


def dispatch_capacity(path):
    """Read immutable shared dispatch capacity; existing ledgers default to one."""
    with _transaction(path) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dispatch_config'").fetchone()
        if not exists:
            return 1
        row = db.execute("SELECT capacity FROM dispatch_config WHERE id=1").fetchone()
        if row is None or row[0] not in (1, 2):
            raise ValueError("invalid shared dispatch configuration")
        return row[0]


def configure_dispatch(path, capacity=1):
    """Set a shared capacity once, under the old exclusive dispatch master lock."""
    if type(capacity) is not int or capacity not in (1, 2):
        raise ValueError("dispatch capacity must be one or two")
    ledger = CampaignBudget(path)
    fd = os.open(str(ledger.path.resolve()) + ".dispatch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with _transaction(ledger.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS dispatch_config (id INTEGER PRIMARY KEY CHECK(id=1), capacity INTEGER NOT NULL CHECK(capacity IN (1,2)))")
            row = db.execute("SELECT capacity FROM dispatch_config WHERE id=1").fetchone()
            if row is not None and row[0] != capacity:
                raise ValueError("dispatch capacity already exists and differs")
            db.execute("INSERT OR IGNORE INTO dispatch_config VALUES (1,?)", (capacity,))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return capacity


@contextmanager
def campaign_dispatch():
    """Acquire a shared master and one bounded slot before reserving any quota.

    Older dispatchers acquire the same master exclusively, so old and new clients
    never overlap. Each new client reads immutable capacity from the same ledger.
    OS file locks release automatically on process death; no token reservation is
    created while waiting. No campaign means unchanged standalone concurrency.
    """
    path = os.environ.get("ZAI_CAMPAIGN_LEDGER")
    if not path:
        yield
        return
    ledger = CampaignBudget(path)
    prefix = str(ledger.path.resolve()) + ".dispatch"
    master = os.open(prefix + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    slots, acquired = [], None
    try:
        fcntl.flock(master, fcntl.LOCK_SH)
        capacity = dispatch_capacity(ledger.path)
        for index in range(capacity):
            slots.append(os.open(prefix + ".slot-" + str(index) + ".lock", os.O_CREAT | os.O_RDWR, 0o600))
        while acquired is None:
            for fd in slots:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = fd
                    break
                except BlockingIOError:
                    pass
            if acquired is None:
                time.sleep(0.025)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
        for fd in slots:
            os.close(fd)
        fcntl.flock(master, fcntl.LOCK_UN)
        os.close(master)


def reserve_from_environment(body):
    path = os.environ.get("ZAI_CAMPAIGN_LEDGER")
    if not path:
        return None, None
    project, run_id = os.environ.get("ZAI_CAMPAIGN_PROJECT"), os.environ.get("ZAI_CAMPAIGN_RUN")
    ledger = CampaignBudget(path)
    bound = len(json.dumps(body, allow_nan=False).encode("utf-8")) + 4096
    request_id = ledger.reserve_request(run_id, bound, body["max_tokens"], project)
    return ledger, request_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--total-limit", type=int, default=18_000_000)
    parser.add_argument("--project-limits", help='JSON object, e.g. {"enso":6000000,"ashfall":12000000}')
    parser.add_argument("--dispatch-capacity", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.initialize:
        ledger = initialize(args.path, args.total_limit, json.loads(args.project_limits or "{}"))
    else:
        ledger = CampaignBudget(args.path)
    if args.dispatch_capacity is not None:
        configure_dispatch(ledger.path, args.dispatch_capacity)
    report = ledger.export_usage_report()
    report["dispatch_capacity"] = dispatch_capacity(ledger.path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

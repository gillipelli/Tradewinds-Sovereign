"""Transactional checkpoints, durable evidence and bounded full-text retrieval."""

import hashlib
import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(assessment UNINDEXED, artifact UNINDEXED, body);
        """)

    def save(self, table, key, payload):
        if table not in {"checkpoints", "evidence", "calls"}:
            raise ValueError("Invalid store table")
        with self.db:
            self.db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", (key, json.dumps(payload, allow_nan=False))
            )

    def get(self, table, key):
        if table not in {"checkpoints", "evidence", "calls"}:
            raise ValueError("Invalid store table")
        row = self.db.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def add_evidence(self, payload):
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
        self.save("evidence", key, payload)
        return key

    def all_evidence(self):
        return {
            key: json.loads(payload) for key, payload in self.db.execute("SELECT id,payload FROM evidence")
        }

    def close(self):
        self.db.close()

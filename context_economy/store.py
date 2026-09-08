"""Durable immutable originals. No implicit expiry or background deletion."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class IntegrityError(RuntimeError):
    pass


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = self._connect()
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS originals (id TEXT PRIMARY KEY, body BLOB NOT NULL)")
        finally:
            db.close()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def key(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def put(self, text: str) -> str:
        data = text.encode("utf-8")
        key = self.key(data)
        db = self._connect()
        try:
            with db:
                db.execute("INSERT OR IGNORE INTO originals VALUES (?, ?)", (key, data))
            # Verify committed bytes before a caller can replace visible content.
            if self.get(key) != text:
                raise IntegrityError("Stored original differs from input")
        finally:
            db.close()
        return key

    def get(self, key: str) -> str:
        db = self._connect()
        try:
            row = db.execute("SELECT body FROM originals WHERE id=?", (key,)).fetchone()
        finally:
            db.close()
        if row is None:
            raise KeyError("Original not found")
        data = bytes(row[0])
        if self.key(data) != key:
            raise IntegrityError("Original checksum mismatch")
        return data.decode("utf-8")

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def _connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                sample_type TEXT NOT NULL,
                location TEXT,
                notes TEXT,
                top_label TEXT NOT NULL,
                top_confidence REAL NOT NULL,
                predictions_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at DESC)
            """
        )


def save_prediction(
    db_path: str,
    *,
    filename: str,
    mime_type: str,
    sample_type: str,
    location: str | None,
    notes: str | None,
    top_label: str,
    top_confidence: float,
    predictions: list[dict[str, float | str]],
) -> None:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    payload = json.dumps(predictions)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                filename,
                mime_type,
                sample_type,
                location,
                notes,
                top_label,
                top_confidence,
                predictions_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                filename,
                mime_type,
                sample_type,
                location,
                notes,
                top_label,
                top_confidence,
                payload,
            ),
        )


def list_recent_predictions(db_path: str, limit: int = 5) -> list[dict[str, str | float]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                filename,
                sample_type,
                location,
                notes,
                top_label,
                top_confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_predictions(
    db_path: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, str | float]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                filename,
                sample_type,
                location,
                notes,
                top_label,
                top_confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def count_predictions(db_path: str) -> int:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM predictions"
        ).fetchone()
    if row is None:
        return 0
    return int(row["total"])


def delete_prediction(db_path: str, prediction_id: int) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            "DELETE FROM predictions WHERE id = ?",
            (prediction_id,),
        )


def delete_all_predictions(db_path: str) -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM predictions")

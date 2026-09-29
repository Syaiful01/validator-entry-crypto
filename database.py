"""
database.py
-----------
Penyimpanan history analisis dengan SQLite (file: data/history.db, dibuat otomatis).
Gambar disimpan sebagai BLOB (sudah dikompres) + thumbnail kecil untuk tabel history.
"""

from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bias TEXT,
    timeframe TEXT,
    notes TEXT,
    total_score REAL,
    recommendation TEXT,
    teknikal_json TEXT,
    market_json TEXT,
    scoring_json TEXT,
    full_analysis_text TEXT,
    image_blob BLOB,
    thumb_blob BLOB
);
CREATE INDEX IF NOT EXISTS idx_history_symbol ON history(symbol);
CREATE INDEX IF NOT EXISTS idx_history_score ON history(total_score);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Buat tabel jika belum ada."""
    with _connect() as conn:
        conn.executescript(SCHEMA)


def make_image_blobs(image_bytes: bytes, max_side: int = 1280, thumb_side: int = 160) -> tuple[bytes, bytes]:
    """Kompres gambar asli & buat thumbnail (keduanya JPEG) untuk disimpan di DB."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    full = img.copy()
    full.thumbnail((max_side, max_side), Image.LANCZOS)
    buf_full = io.BytesIO()
    full.save(buf_full, format="JPEG", quality=85)

    thumb = img.copy()
    thumb.thumbnail((thumb_side, thumb_side), Image.LANCZOS)
    buf_thumb = io.BytesIO()
    thumb.save(buf_thumb, format="JPEG", quality=75)
    return buf_full.getvalue(), buf_thumb.getvalue()


def save_analysis(
    symbol: str,
    bias: str,
    timeframe: str,
    notes: str,
    scoring: dict,
    teknikal: dict,
    market: Optional[dict],
    full_analysis_text: str,
    image_bytes: Optional[bytes] = None,
) -> int:
    """Simpan satu hasil analisis, return id baris baru."""
    image_blob = thumb_blob = None
    if image_bytes:
        try:
            image_blob, thumb_blob = make_image_blobs(image_bytes)
        except Exception:
            pass  # gagal kompres gambar tidak boleh menggagalkan penyimpanan

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history
               (timestamp, symbol, bias, timeframe, notes, total_score, recommendation,
                teknikal_json, market_json, scoring_json, full_analysis_text, image_blob, thumb_blob)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                bias,
                timeframe or "",
                notes or "",
                scoring["total_score"],
                scoring["recommendation"],
                json.dumps(teknikal, ensure_ascii=False),
                json.dumps(market, ensure_ascii=False) if market else None,
                json.dumps(scoring, ensure_ascii=False),
                full_analysis_text,
                image_blob,
                thumb_blob,
            ),
        )
        return cur.lastrowid


def get_all_history(symbol_filter: str = "", min_score: float = 0) -> list[dict]:
    """Daftar history terbaru dulu (tanpa gambar penuh). Filter symbol (partial) & skor minimal."""
    query = (
        "SELECT id, timestamp, symbol, bias, timeframe, total_score, recommendation, thumb_blob "
        "FROM history WHERE total_score >= ?"
    )
    params: list = [min_score]
    if symbol_filter.strip():
        query += " AND symbol LIKE ?"
        params.append(f"%{symbol_filter.strip().upper()}%")
    query += " ORDER BY id DESC"

    with _connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_history_by_id(history_id: int) -> Optional[dict]:
    """Ambil detail lengkap satu analisis (JSON sudah di-parse)."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM history WHERE id = ?", (history_id,)).fetchone()
    if row is None:
        return None
    rec = dict(row)
    for key in ("teknikal_json", "market_json", "scoring_json"):
        rec[key.replace("_json", "")] = json.loads(rec[key]) if rec.get(key) else None
    return rec


def delete_history(history_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (history_id,))


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM history")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'history'")

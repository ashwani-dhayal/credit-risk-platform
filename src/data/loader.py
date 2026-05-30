"""Load the Home Credit dataset and ingest it into SQLite.

Strategy:
1. If `data/raw/application_train.csv` exists, use it (real Kaggle data).
2. Else fall back to `data/sample/application_train_sample.csv` (synthetic).
3. Else generate the sample on the fly.

The chosen CSV is loaded into a SQLite DB at `data/processed/credit_risk.db`,
table `applications`. The DB is the single source of truth for both the ML
pipeline and the talk-to-data chatbot — this guarantees the chatbot answers
about exactly the same data the model was trained on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import SETTINGS
from src.data.schema import CORE_COLUMNS, TABLE_NAME


def resolve_source_csv() -> Path:
    """Return the CSV path we should ingest, generating a sample if needed."""
    real = SETTINGS.raw_dir / "application_train.csv"
    if real.exists():
        return real

    sample = SETTINGS.sample_dir / "application_train_sample.csv"
    if not sample.exists():
        from .sample_generator import write_sample
        write_sample(sample, n_rows=10_000)
    return sample


def load_dataframe(nrows: Optional[int] = None) -> pd.DataFrame:
    src = resolve_source_csv()
    df = pd.read_csv(src, nrows=nrows)
    keep = [c for c in CORE_COLUMNS if c in df.columns]
    return df[keep].copy()


def ingest_to_sqlite(force: bool = False) -> Path:
    """Materialise the source CSV into SQLite. Idempotent unless `force`."""
    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)
    db_path = SETTINGS.db_path

    if db_path.exists() and not force:
        # Quick sanity check: does the table exist with rows?
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME}"
                )
                if cur.fetchone()[0] > 0:
                    return db_path
        except sqlite3.Error:
            pass  # fall through and rebuild

    df = load_dataframe()
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        # Helpful indexes for the chatbot's frequent filters
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_app_target ON {TABLE_NAME}(TARGET)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_app_contract ON {TABLE_NAME}(NAME_CONTRACT_TYPE)"
        )
        conn.commit()
    return db_path


def open_connection() -> sqlite3.Connection:
    """Return a read-only SQLite connection (used by the chatbot)."""
    db_path = ingest_to_sqlite(force=False)
    # `mode=ro` requires URI form
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_table_schema() -> str:
    """Return a compact text schema description for prompting the LLM."""
    from src.data.schema import COLUMN_DESCRIPTIONS

    lines = [f"Table: {TABLE_NAME}", "Columns:"]
    db_path = ingest_to_sqlite(force=False)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(f"PRAGMA table_info({TABLE_NAME})")
        for _cid, name, ctype, _nn, _dflt, _pk in cur.fetchall():
            desc = COLUMN_DESCRIPTIONS.get(name, "")
            lines.append(f"  - {name} ({ctype}) — {desc}" if desc else f"  - {name} ({ctype})")
    return "\n".join(lines)


if __name__ == "__main__":
    p = ingest_to_sqlite(force=True)
    print(f"Database ready at: {p}")
    print(get_table_schema())

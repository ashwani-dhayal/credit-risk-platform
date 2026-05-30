"""Loads the source CSV and ingests it into SQLite.

Lookup order:
    1. data/raw/application_train.csv  (real Kaggle file)
    2. data/sample/application_train_sample.csv  (synthetic stand-in)
    3. fall back to generating the sample on the fly

The DB this builds is the single source of truth for both the model
and the chatbot.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import SETTINGS
from src.data.schema import CORE_COLUMNS, TABLE_NAME


def resolve_source_csv():
    """Pick whichever CSV is available, generating the synthetic one if needed."""
    real = SETTINGS.raw_dir / "application_train.csv"
    if real.exists():
        return real

    sample = SETTINGS.sample_dir / "application_train_sample.csv"
    if not sample.exists():
        from .sample_generator import write_sample
        write_sample(sample, n_rows=10_000)
    return sample


def load_dataframe(nrows=None):
    src = resolve_source_csv()
    df = pd.read_csv(src, nrows=nrows)
    keep = [c for c in CORE_COLUMNS if c in df.columns]
    return df[keep].copy()


def ingest_to_sqlite(force=False):
    """Materialise the source CSV into SQLite. Idempotent unless `force=True`."""
    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)
    db_path = SETTINGS.db_path

    # If the DB is already populated, reuse it.
    if db_path.exists() and not force:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                if cur.fetchone()[0] > 0:
                    return db_path
        except sqlite3.Error:
            # Corrupt or schema-mismatched DB - rebuild.
            pass

    df = load_dataframe()
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        # Indexes used by chatbot filters.
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_app_target ON {TABLE_NAME}(TARGET)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_app_contract ON {TABLE_NAME}(NAME_CONTRACT_TYPE)")
        conn.commit()
    return db_path


def open_connection():
    """Return a read-only SQLite connection. Used by the chatbot."""
    db_path = ingest_to_sqlite(force=False)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_table_schema():
    """Compact schema summary for the LLM prompt."""
    from src.data.schema import COLUMN_DESCRIPTIONS

    lines = [f"Table: {TABLE_NAME}", "Columns:"]
    db_path = ingest_to_sqlite(force=False)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(f"PRAGMA table_info({TABLE_NAME})")
        for _cid, name, ctype, _nn, _dflt, _pk in cur.fetchall():
            desc = COLUMN_DESCRIPTIONS.get(name, "")
            if desc:
                lines.append(f"  - {name} ({ctype}) {desc}")
            else:
                lines.append(f"  - {name} ({ctype})")
    return "\n".join(lines)


if __name__ == "__main__":
    p = ingest_to_sqlite(force=True)
    print(f"Database ready at: {p}")
    print(get_table_schema())

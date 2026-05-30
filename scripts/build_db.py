"""CLI: rebuild the SQLite DB from the source CSV."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_table_schema, ingest_to_sqlite


if __name__ == "__main__":
    p = ingest_to_sqlite(force=True)
    print(f">> Database ready at: {p}\n")
    print(get_table_schema())

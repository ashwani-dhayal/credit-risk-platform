"""Tests for the SQL safety guard."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sql_safety import validate_and_harden


def test_simple_select_passes():
    v = validate_and_harden("SELECT COUNT(*) AS n FROM applications")
    assert v.ok
    assert "LIMIT" in v.sql.upper()


def test_drop_blocked():
    assert not validate_and_harden("DROP TABLE applications").ok


def test_insert_blocked():
    assert not validate_and_harden("INSERT INTO applications VALUES (1)").ok


def test_multistatement_blocked():
    assert not validate_and_harden("SELECT 1; SELECT 2").ok


def test_unknown_table_blocked():
    assert not validate_and_harden("SELECT * FROM users").ok


def test_attach_blocked():
    assert not validate_and_harden("ATTACH DATABASE 'foo.db' AS f").ok


def test_with_cte_passes():
    sql = (
        "WITH t AS (SELECT TARGET FROM applications) "
        "SELECT AVG(TARGET) AS rate FROM t"
    )
    assert validate_and_harden(sql).ok


def test_existing_limit_preserved():
    v = validate_and_harden("SELECT * FROM applications LIMIT 5")
    assert v.ok
    assert "LIMIT 5" in v.sql
    assert "LIMIT 200" not in v.sql

"""Tests for the production-run ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.production.ledger import (
    ACTIVE_STATUSES,
    LEDGER_COLUMNS,
    LedgerRow,
    STATUS_CLAIMED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STALE,
    append_row,
    current_status,
    filter_unclaimed,
    make_pair_key,
    read_all_rows,
    summarise,
)


def _row(pair_key: str, status: str, **extras) -> LedgerRow:
    base = dict(
        pair_key=pair_key,
        cell_line="K562",
        tf_a=pair_key.split("_")[1],
        tf_b=pair_key.split("_")[2] if pair_key.count("_") >= 2 else "",
        status=status,
    )
    base.update(extras)
    return LedgerRow(**base)


def test_make_pair_key_canonical_format():
    assert make_pair_key("K562", "ADNP", "YY1") == "K562_ADNP_YY1"


def test_read_all_rows_missing_file(tmp_path: Path):
    assert read_all_rows(tmp_path / "nope.csv") == []


def test_current_status_missing_file(tmp_path: Path):
    assert current_status(tmp_path / "nope.csv") == {}


def test_append_row_creates_ledger_with_header(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_CLAIMED))

    # Header + one data row
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == ",".join(LEDGER_COLUMNS)
    rows = read_all_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["pair_key"] == "K562_ADNP_YY1"
    assert rows[0]["status"] == STATUS_CLAIMED


def test_append_row_is_append_only(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_CLAIMED, slurm_job_id="1"))
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_COMPLETED, run_id="abc"))
    append_row(ledger, _row("K562_ARID3B_YY1", STATUS_CLAIMED, slurm_job_id="2"))

    rows = read_all_rows(ledger)
    assert len(rows) == 3
    # Order preserved
    assert rows[0]["status"] == STATUS_CLAIMED
    assert rows[1]["status"] == STATUS_COMPLETED
    assert rows[2]["pair_key"] == "K562_ARID3B_YY1"


def test_current_status_returns_latest_row_per_pair(tmp_path: Path):
    """When a pair has multiple rows, current_status returns the last one."""
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_CLAIMED))
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_FAILED, notes="OOM"))
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_CLAIMED, slurm_job_id="retry"))
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_COMPLETED, run_id="abc"))

    state = current_status(ledger)
    assert state["K562_ADNP_YY1"]["status"] == STATUS_COMPLETED
    assert state["K562_ADNP_YY1"]["run_id"] == "abc"


def test_filter_unclaimed_excludes_active_pairs(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_ADNP_YY1", STATUS_COMPLETED))  # excluded
    append_row(ledger, _row("K562_ARID3B_YY1", STATUS_CLAIMED))  # excluded
    append_row(ledger, _row("K562_CTCF_YY1", STATUS_FAILED))     # included
    append_row(ledger, _row("K562_REST_YY1", STATUS_STALE))      # included

    candidates = [
        "K562_ADNP_YY1",      # active → skip
        "K562_ARID3B_YY1",    # active → skip
        "K562_CTCF_YY1",      # reclaimable
        "K562_REST_YY1",      # reclaimable
        "K562_NEW_YY1",       # not in ledger → include
    ]

    unclaimed = filter_unclaimed(ledger, candidates)
    assert unclaimed == ["K562_CTCF_YY1", "K562_REST_YY1", "K562_NEW_YY1"]


def test_filter_unclaimed_empty_ledger_returns_all(tmp_path: Path):
    ledger = tmp_path / "missing.csv"
    candidates = ["K562_A_B", "K562_C_D"]
    assert filter_unclaimed(ledger, candidates) == candidates


def test_filter_unclaimed_respects_latest_row(tmp_path: Path):
    """A pair that failed, was reclaimed, and completed must stay excluded."""
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_X_Y", STATUS_FAILED))     # would be reclaimable...
    append_row(ledger, _row("K562_X_Y", STATUS_CLAIMED))    # ...but we re-claimed it...
    append_row(ledger, _row("K562_X_Y", STATUS_COMPLETED))  # ...and it completed.

    assert filter_unclaimed(ledger, ["K562_X_Y"]) == []


def test_summarise_counts_current_state(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    append_row(ledger, _row("K562_A_Y", STATUS_COMPLETED))
    append_row(ledger, _row("K562_B_Y", STATUS_COMPLETED))
    append_row(ledger, _row("K562_C_Y", STATUS_CLAIMED))
    append_row(ledger, _row("K562_D_Y", STATUS_FAILED))
    # Historical row — should NOT be counted, only the latest for pair D
    append_row(ledger, _row("K562_D_Y", STATUS_COMPLETED))

    counts = summarise(ledger)
    # D went from failed → completed, so final state has 3 completed, 1 claimed
    assert counts == {"completed": 3, "claimed": 1}


def test_row_notes_with_commas_and_newlines_round_trip(tmp_path: Path):
    """CSV writer must quote values with special characters."""
    ledger = tmp_path / "ledger.csv"
    tricky_note = 'line1, line2\n"quoted", value'
    append_row(ledger, _row("K562_X_Y", STATUS_FAILED, notes=tricky_note))

    rows = read_all_rows(ledger)
    assert rows[0]["notes"] == tricky_note


def test_concurrent_appends_do_not_corrupt(tmp_path: Path):
    """Many appends in quick succession must all land as complete rows.

    Can't truly test fcntl across processes inside pytest, but we can
    verify that sequential fast appends round-trip correctly and the
    header is written exactly once.
    """
    ledger = tmp_path / "ledger.csv"
    for i in range(50):
        append_row(ledger, _row(f"K562_TF{i}_YY1", STATUS_CLAIMED))

    rows = read_all_rows(ledger)
    assert len(rows) == 50
    # Header must appear exactly once
    content = ledger.read_text()
    assert content.count(",".join(LEDGER_COLUMNS)) == 1


def test_active_statuses_constants_match_spec():
    """Contract check — downstream scripts depend on these exact values."""
    assert STATUS_CLAIMED in ACTIVE_STATUSES
    assert STATUS_COMPLETED in ACTIVE_STATUSES
    assert STATUS_FAILED not in ACTIVE_STATUSES
    assert STATUS_STALE not in ACTIVE_STATUSES

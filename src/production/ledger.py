"""Append-only ledger for production-batch pair tracking.

Every row records one state transition for one pair. The "current state"
of a pair is the LAST row with that pair_key in the file. This design lets
multiple Slurm array tasks append concurrently without coordination beyond
a single fcntl lock on the ledger file.

Lifecycle of a pair:

    (absent)  --claim-->  claimed  --success-->  completed
                               \\--failure-->    failed
                               \\--timeout-->    stale   (via reaper script)

    completed / claimed pairs are excluded from further claims.
    failed / stale pairs are re-claimable.
"""

from __future__ import annotations

import csv
import fcntl
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


LEDGER_COLUMNS = [
    "pair_key",
    "cell_line",
    "tf_a",
    "tf_b",
    "source_csv",
    "status",
    "slurm_job_id",
    "run_id",
    "claimed_at",
    "finished_at",
    "converged",
    "confidence",
    "mechanism",
    "output_dir",
    "total_cost_usd",
    "notes",
]

ACTIVE_STATUSES = frozenset({"claimed", "completed", "invalidated"})
RECLAIMABLE_STATUSES = frozenset({"failed", "stale"})

# Status tags written by callers.
STATUS_CLAIMED = "claimed"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"
STATUS_INVALIDATED = "invalidated"


@dataclass
class LedgerRow:
    pair_key: str
    cell_line: str = ""
    tf_a: str = ""
    tf_b: str = ""
    source_csv: str = ""
    status: str = ""
    slurm_job_id: str = ""
    run_id: str = ""
    claimed_at: str = ""
    finished_at: str = ""
    converged: str = ""          # str so "" (unknown) round-trips cleanly
    confidence: str = ""
    mechanism: str = ""
    output_dir: str = ""
    total_cost_usd: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, str]:
        return {col: getattr(self, col) for col in LEDGER_COLUMNS}


def make_pair_key(cell_line: str, tf_a: str, tf_b: str) -> str:
    """Canonical pair_key format: `{cell_line}_{tf_a}_{tf_b}`."""
    return f"{cell_line}_{tf_a}_{tf_b}"


def _ensure_parent(ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)


def _write_header_if_missing(ledger_path: Path) -> None:
    """Create the ledger with a header row if it doesn't exist yet."""
    _ensure_parent(ledger_path)
    if ledger_path.exists():
        return
    # Open with O_EXCL to avoid racing a concurrent creator; tolerate the
    # "already exists" case as a benign win for whichever process got there
    # first.
    try:
        fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LEDGER_COLUMNS)
    except Exception:
        # Surface the failure — a half-written header is worse than no header.
        if ledger_path.exists():
            ledger_path.unlink()
        raise


def append_row(ledger_path: Path, row: LedgerRow) -> None:
    """Atomically append one row to the ledger.

    Safe for concurrent callers: an exclusive fcntl lock is held for the
    duration of the append. Header is created on first write.
    """
    _write_header_if_missing(ledger_path)

    with open(ledger_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
            writer.writerow(row.as_dict())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_all_rows(ledger_path: Path) -> list[dict[str, str]]:
    """Read every row (in file order). Returns empty list if ledger missing."""
    if not ledger_path.exists():
        return []
    with open(ledger_path, newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            reader = csv.DictReader(f)
            return list(reader)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def current_status(ledger_path: Path) -> dict[str, dict[str, str]]:
    """Return pair_key -> most recent row for each pair.

    The "current" status of a pair is the LAST row for it in the file.
    Earlier rows are historical record.
    """
    latest: dict[str, dict[str, str]] = {}
    for row in read_all_rows(ledger_path):
        pk = row.get("pair_key") or ""
        if pk:
            latest[pk] = row
    return latest


def filter_unclaimed(
    ledger_path: Path,
    candidate_keys: Iterable[str],
) -> list[str]:
    """Return candidate keys that are NOT currently claimed or completed.

    Pairs whose latest status is `failed` or `stale` ARE returned
    (re-claimable). Pairs absent from the ledger are always returned.
    """
    state = current_status(ledger_path)
    out: list[str] = []
    for key in candidate_keys:
        latest = state.get(key)
        if latest is None:
            out.append(key)
            continue
        status = (latest.get("status") or "").strip()
        if status not in ACTIVE_STATUSES:
            out.append(key)
    return out


def summarise(ledger_path: Path) -> dict[str, int]:
    """Count pairs by current status for quick display."""
    counts: dict[str, int] = {}
    for _, row in current_status(ledger_path).items():
        status = (row.get("status") or "unknown").strip() or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def now_iso() -> str:
    """Timestamp helper used by callers of append_row."""
    return datetime.now().isoformat(timespec="seconds")

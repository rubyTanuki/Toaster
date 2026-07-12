"""Cache format versioning for the `.tostr/cache.db` graph.

The format version is stamped into the SQLite header via `PRAGMA user_version` (a free 32-bit int,
no schema/table needed). It is *not* the table schema — additive column changes are handled by the
`CREATE TABLE IF NOT EXISTS` / `ALTER` logic in `db.py`. This version tracks the **meaning** of the
stored data (UID scheme, edge semantics, serialization) so we can refuse to silently operate on a
cache whose contents the current code would misinterpret.

### Adding a new version (the future-proofing contract)

When you change the on-disk format or anything that invalidates existing rows, append ONE entry to
`CACHE_FORMAT_HISTORY`:

    FormatVersion(3, breaking=True,  "what changed and why old caches can't be reused"),
    FormatVersion(4, breaking=False, "additive change; old caches still read correctly"),

- `breaking=True`  → existing caches at an older version cannot be used; `tostr init` auto-wipes and
  rebuilds, and any read/use command errors with a clear "run tostr init" message.
- `breaking=False` → old caches are still valid; the stamp is simply advanced (pair it with the
  matching additive migration in `db.py` if one is needed).

That's the whole process — the check logic below reads the flags, so you never touch the comparison code.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sqlite3


@dataclass(frozen=True)
class FormatVersion:
    version: int
    breaking: bool
    summary: str


# Append-only. The last entry's `version` is the current format.
CACHE_FORMAT_HISTORY: List[FormatVersion] = [
    FormatVersion(
        1,
        breaking=True,
        summary="UID overload-key scheme: Python method UIDs end in '(...)', Java in normalized "
                "param types. Pre-versioned caches (user_version 0) use the old param-name UIDs and "
                "must be rebuilt.",
    ),
]

CURRENT_CACHE_VERSION: int = CACHE_FORMAT_HISTORY[-1].version


def read_db_version(db_path: Path | str) -> int:
    """Read the stamped format version from a cache file (0 if never stamped / pre-versioning)."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def incompatibility_reason(stored_version: int) -> Optional[str]:
    """Return a human-readable reason the stored cache is incompatible with this build, or None if
    it's safe to use. Incompatible when a breaking format change separates `stored_version` from the
    current version, or when the cache was written by a newer Tostr than this one."""
    if stored_version > CURRENT_CACHE_VERSION:
        return (
            f"cache format v{stored_version} was written by a newer Tostr than this build "
            f"(supports v{CURRENT_CACHE_VERSION})"
        )
    breaking = [v for v in CACHE_FORMAT_HISTORY if stored_version < v.version <= CURRENT_CACHE_VERSION and v.breaking]
    if breaking:
        changes = "; ".join(f"v{v.version}: {v.summary}" for v in breaking)
        return (
            f"cache format v{stored_version} predates a breaking change "
            f"(now v{CURRENT_CACHE_VERSION}). Breaking changes since: {changes}"
        )
    return None


def is_compatible(db_path: Path | str) -> bool:
    return incompatibility_reason(read_db_version(db_path)) is None

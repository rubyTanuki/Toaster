from __future__ import annotations
import sqlite3
from pathlib import Path

import pytest

from tostr.core.paths import ProjectPaths
from tostr.storage.db import SqliteClient
from tostr.storage.versioning import (
    CURRENT_CACHE_VERSION,
    incompatibility_reason,
    read_db_version,
    is_compatible,
)
from tostr.commands import _verify_db_exists
from tostr.exceptions import CacheFormatError, DatabaseNotFoundError


def _set_version(db_path: Path, version: int) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(f"PRAGMA user_version = {version}")
        con.commit()
    finally:
        con.close()


def test_incompatibility_reason_logic():
    assert incompatibility_reason(CURRENT_CACHE_VERSION) is None
    # Pre-versioned cache (0) predates the v1 breaking UID change.
    assert incompatibility_reason(0) is not None
    # A cache written by a newer Tostr than this build.
    assert incompatibility_reason(CURRENT_CACHE_VERSION + 1) is not None


def test_fresh_db_is_stamped_current(tmp_path):
    db_path = tmp_path / ".tostr" / "cache.db"
    SqliteClient(ProjectPaths(tmp_path))  # creates schema + stamps version
    assert read_db_version(db_path) == CURRENT_CACHE_VERSION
    assert is_compatible(db_path)


def test_existing_db_version_is_not_remasked(tmp_path):
    """Re-opening an existing (stale) cache must NOT silently re-stamp it to current — otherwise the
    compatibility check could never detect an old format."""
    db_path = tmp_path / ".tostr" / "cache.db"
    SqliteClient(ProjectPaths(tmp_path))
    _set_version(db_path, 0)  # simulate a pre-versioned cache
    SqliteClient(ProjectPaths(tmp_path))  # second open
    assert read_db_version(db_path) == 0
    assert not is_compatible(db_path)


def test_verify_db_exists_rejects_incompatible_cache(tmp_path):
    db_path = tmp_path / ".tostr" / "cache.db"
    SqliteClient(ProjectPaths(tmp_path))
    _set_version(db_path, 0)
    with pytest.raises(CacheFormatError):
        _verify_db_exists(tmp_path)


def test_verify_db_exists_accepts_current_cache(tmp_path):
    db_path = tmp_path / ".tostr" / "cache.db"
    SqliteClient(ProjectPaths(tmp_path))
    _verify_db_exists(tmp_path)  # should not raise


def test_verify_db_exists_missing_db(tmp_path):
    with pytest.raises(DatabaseNotFoundError):
        _verify_db_exists(tmp_path)

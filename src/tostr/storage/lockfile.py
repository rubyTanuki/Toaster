"""The `tostr.lock.json` format: read, write, and version it in one place.

This is the *generated-but-committed* description lockfile at the project root — a third file
category alongside the committed authored config (tostr.toml / .tostrignore) and the gitignored
.tostr/ cache. One person parses (paying the LLM cost once) and runs `tostr export` to commit this
file; everyone else's first `tostr parse` seeds descriptions from it instead of re-calling the LLM.

Separation of concerns: this module owns only the *serialization format* (JSON shape, deterministic
output, version stamp + compatibility gate, no-op write). It deals in plain `{uid: {...}}` entry
dicts and paths — it never touches the database. Gathering the entries from the cache is the
StructCache's job (`StructCache.collect_descriptions`); orchestrating the two is the command's job
(`commands.export_lockfile`). Because it depends only on `versioning`, both `registry` and
`commands` can import it without an import cycle.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from tostr.storage.versioning import incompatibility_reason, CURRENT_CACHE_VERSION

LOCKFILE_NAME = "tostr.lock.json"

# Schema version of the lockfile itself (independent of the cache format version stamped into the
# db). Bump only on a breaking change to the lockfile's own structure.
LOCKFILE_VERSION = 1


def path_for(project_path: Path) -> Path:
    """Absolute path of the lockfile for a project root."""
    return project_path / LOCKFILE_NAME


def write(project_path: Path, entries: Dict[str, dict], notes: Optional[Dict[str, list]] = None) -> bool:
    """Serialize `entries` (and `notes`) to `<project_root>/tostr.lock.json` deterministically
    (sorted keys, stable indentation, trailing newline) so diffs review cleanly. Skips the write when
    the bytes are identical to the existing file, so an export that changed nothing doesn't churn
    git. Returns True iff the file was (re)written.

    `notes` is a separate top-level `{uid: [note, ...]}` map rather than a field inside `entries`
    for two reasons: entries only contains structs that *have* a description, while a struct can
    carry notes without one; and entry reuse is gated on `diff_hash`, which is right for a
    description (it describes the body) and wrong for a note (an observation about a function
    usually outlives an edit to it)."""
    payload = {
        "tostr_lock_version": LOCKFILE_VERSION,
        "cache_format_version": CURRENT_CACHE_VERSION,
        "entries": entries,
        "notes": notes or {},
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    lockfile_path = path_for(project_path)
    existing = lockfile_path.read_text(encoding="utf-8") if lockfile_path.exists() else None
    if existing == serialized:
        logger.debug(f"{LOCKFILE_NAME} already up to date")
        return False
    lockfile_path.write_text(serialized, encoding="utf-8")
    note_count = sum(len(v) for v in (notes or {}).values())
    logger.info(f"Wrote {LOCKFILE_NAME} ({len(entries)} description(s), {note_count} note(s))")
    return True


def _load(project_path: Path) -> Optional[dict]:
    """Read and validate the lockfile payload, or None when it is absent, unreadable, or written
    against an incompatible cache format — a breaking UID-scheme change would no-match anyway; the
    guard skips a stale lockfile cleanly instead of silently."""
    lockfile_path = path_for(project_path)
    if not lockfile_path.exists():
        return None
    try:
        data = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {LOCKFILE_NAME}: {e}; ignoring lockfile")
        return None

    reason = incompatibility_reason(data.get("cache_format_version", 0))
    if reason:
        logger.warning(f"{LOCKFILE_NAME} is incompatible with this build ({reason}); ignoring lockfile")
        return None
    return data


def read(project_path: Path) -> Optional[Dict[str, dict]]:
    """The lockfile's `{uid: {diff_hash, description, vector?}}` entries map, or None."""
    data = _load(project_path)
    return None if data is None else data.get("entries", {})


def read_notes(project_path: Path) -> Dict[str, list]:
    """The lockfile's `{uid: [note, ...]}` map. Returns `{}` for a missing, unreadable, or
    incompatible lockfile, and for one written before notes were exported."""
    data = _load(project_path)
    return (data or {}).get("notes", {}) or {}

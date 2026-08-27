"""Tests for struct notes: the `notes` table and its write-path integration.

The DB-level tests are hermetic (a bare SqliteClient over tmp_path). The round-trip tests are
marked integration because they run a real parse to get structs into the cache.
"""
from __future__ import annotations
import shutil
from pathlib import Path

import pytest

from tostr.core.models import Note
from tostr.core.paths import ProjectPaths
from tostr.storage.db import SqliteClient

TEST_PROJECT = Path(__file__).parent / "testcode" / "PythonTestProject"


@pytest.fixture
def db(tmp_path) -> SqliteClient:
    return SqliteClient(ProjectPaths(tmp_path))


#region DB LAYER


def test_notes_table_is_indexed_on_struct_id(db):
    with db.get_connection() as conn:
        table = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'notes'").fetchone()
        index = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'notes'"
        ).fetchall()
    assert table is not None and "VIRTUAL" not in table["sql"].upper()
    assert any(r["name"] == "idx_notes_struct" for r in index)


def test_insert_and_read_back(db):
    note_id = db.insert_note("M-1", "watch the retry backoff here", "avery", "2026-08-25T10:00:00",
                             "2026-08-25T10:00:00")
    assert isinstance(note_id, int)
    rows = db.notes_for_structs(["M-1"])
    assert len(rows) == 1
    assert rows[0]["id"] == note_id
    assert rows[0]["content"] == "watch the retry backoff here"
    assert rows[0]["author"] == "avery"
    assert db.note_count() == 1


def test_update_and_delete_by_id(db):
    note_id = db.insert_note("M-1", "original", "avery", "t", "t")
    assert db.update_note(note_id, "revised", "avery", "t2") is True
    assert db.notes_for_structs(["M-1"])[0]["content"] == "revised"
    assert db.delete_note(note_id) is True
    assert db.notes_for_structs(["M-1"]) == []
    # Already-gone ids report False rather than raising.
    assert db.update_note(note_id, "x", "y", "t") is False
    assert db.delete_note(note_id) is False


def test_replace_struct_notes_is_exact(db):
    db.insert_note("M-1", "first", "avery", "t", "t")
    db.insert_note("M-1", "second", "avery", "t", "t")
    note_ids = db.replace_struct_notes("M-1", [Note(content="only one", author="avery").to_dict()])
    rows = db.notes_for_structs(["M-1"])
    assert [r["content"] for r in rows] == ["only one"]
    assert [r["id"] for r in rows] == note_ids


def test_ids_are_not_reused_after_delete(db):
    """Note ids are shown to users and agents, who may act on one long after reading it —
    AUTOINCREMENT keeps a stale id from silently addressing a different note."""
    first = db.insert_note("M-1", "a", "avery", "t", "t")
    db.delete_note(first)
    assert db.insert_note("M-1", "b", "avery", "t", "t") != first


def test_delete_notes_for_structs(db):
    db.insert_note("M-1", "a", "avery", "t", "t")
    db.insert_note("M-2", "b", "avery", "t", "t")
    assert db.delete_notes_for_structs(["M-1"]) == 1
    assert db.note_count() == 1
    assert db.delete_notes_for_structs([]) == 0


#endregion

#region INSPECT / COMMAND SURFACE


@pytest.fixture
def project(tmp_path):
    proj = tmp_path / "proj"
    shutil.copytree(TEST_PROJECT, proj)
    return proj


async def _parsed_cache(project: Path):
    """Parse the test project, then return a fresh StructCache hydrated over the resulting DB."""
    from tostr.commands import parse_async
    from tostr.graph.registry import Registry
    from tostr.storage.cache import StructCache

    await parse_async(project, no_llm=True)
    paths = ProjectPaths(project)
    registry = Registry(paths=paths)
    cache = StructCache(paths, registry)
    cache.load_filepath(Path("."))
    return cache, registry


def _a_method(registry):
    return next(s for s in registry.uid_map.values() if type(s).__name__ == "BaseMethod")


@pytest.mark.integration
async def test_note_round_trips_through_hydration(project):
    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    note = cache.add_note(struct, "this one is load-bearing", "avery")
    assert note.id is not None

    _, registry2 = await _parsed_cache(project)  # fresh hydration off disk
    reloaded = registry2.uid_map[struct.uid]
    assert [n.content for n in reloaded.notes] == ["this one is load-bearing"]
    assert reloaded.notes[0].id == note.id
    assert reloaded.notes[0].author == "avery"


@pytest.mark.integration
async def test_notes_survive_a_reparse(project):
    """Reparsed structs carry no notes in memory; the save must not wipe the stored ones."""
    from tostr.commands import parse_async

    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    cache.add_note(struct, "must outlive the reparse", "avery")

    await parse_async(project, no_llm=True)

    _, registry2 = await _parsed_cache(project)
    assert [n.content for n in registry2.uid_map[struct.uid].notes] == ["must outlive the reparse"]


@pytest.mark.integration
async def test_deleting_a_file_deletes_its_notes(project):
    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    cache.add_note(struct, "doomed alongside its file", "avery")
    assert cache.db.note_count() == 1

    cache.delete_path_subtree(str(struct.path))
    # The FK cascade is inert (foreign_keys is off on these connections), so this only passes
    # because the delete path drops notes explicitly.
    assert cache.db.note_count() == 0


@pytest.mark.integration
async def test_edit_and_delete_write_through(project):
    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    note = cache.add_note(struct, "first take", "avery")

    assert cache.edit_note(note, "second take", "sam", struct=struct) is True
    stored = cache.db.notes_for_structs([struct.id])[0]
    assert stored["content"] == "second take" and stored["author"] == "sam"
    assert note.date_last_updated >= note.date_added

    assert cache.delete_note(struct, note) is True
    assert cache.db.notes_for_structs([struct.id]) == []
    assert struct.notes == []


@pytest.mark.integration
async def test_max_notes_cap_evicts_oldest_rows(project):
    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    for i in range(struct._max_notes + 3):
        cache.add_note(struct, f"note {i}", "avery")

    stored = [r["content"] for r in cache.db.notes_for_structs([struct.id])]
    assert len(stored) == struct._max_notes
    assert "note 0" not in stored and f"note {struct._max_notes + 2}" in stored
    assert len(struct.notes) == struct._max_notes


@pytest.mark.integration
async def test_date_added_is_preserved_across_reparse(project):
    from tostr.commands import parse_async

    _, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    original = struct.date_added

    await parse_async(project, no_llm=True)
    _, registry2 = await _parsed_cache(project)
    assert registry2.uid_map[struct.uid].date_added == original


#endregion


@pytest.mark.integration
async def test_inspect_dump_carries_notes(project):
    from tostr.commands import inspect_async, note_add_async

    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)
    await note_add_async(struct.id, "load-bearing invariant", "avery", project)

    result = (await inspect_async([struct.id], project))[0]
    assert [n.content for n in result.notes] == ["load-bearing invariant"]
    assert result.notes[0].author == "avery"
    assert result.notes[0].id is not None


@pytest.mark.integration
async def test_renderers_put_notes_above_the_description(project):
    """Both the CLI and MCP renderers must place notes between the header and everything
    Tostr derived itself."""
    from tostr.commands import inspect_async, note_add_async
    from tostr.server import _render_inspect as render_mcp

    cache, registry = await _parsed_cache(project)
    struct = next(s for s in registry.uid_map.values()
                  if type(s).__name__ == "BaseMethod" and s.description)
    await note_add_async(struct.id, "a human said this", "avery", project)

    result = (await inspect_async([struct.id], project))[0]
    lines = render_mcp(result).splitlines()
    note_line = next(i for i, l in enumerate(lines) if l.startswith("# ["))
    desc_line = next(i for i, l in enumerate(lines) if l.startswith("// "))
    assert 0 < note_line < desc_line
    assert "a human said this" in lines[note_line]
    assert "avery" in lines[note_line]

    # The CLI renderer writes to a rich Console; capture it the same way.
    from rich.console import Console
    import tostr.cli as cli

    buffer = Console(record=True, width=200)
    original, cli.console = cli.console, buffer
    try:
        cli._render_inspect(result)
    finally:
        cli.console = original
    cli_lines = [l for l in buffer.export_text().splitlines() if l.strip()]
    cli_note = next(i for i, l in enumerate(cli_lines) if l.startswith("# ["))
    cli_desc = next(i for i, l in enumerate(cli_lines) if l.startswith("// "))
    assert 0 < cli_note < cli_desc


@pytest.mark.integration
async def test_note_commands_round_trip(project):
    from tostr.commands import note_add_async, note_edit_async, note_remove_async, inspect_async

    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)

    _, note = await note_add_async(struct.uid, "first take", "avery", project)  # resolves by uid
    _, edited = await note_edit_async(struct.id, note.id, "second take", "sam", project)
    assert edited.id == note.id

    result = (await inspect_async([struct.id], project))[0]
    assert [(n.content, n.author) for n in result.notes] == [("second take", "sam")]

    await note_remove_async(struct.id, note.id, project)
    assert (await inspect_async([struct.id], project))[0].notes == []


@pytest.mark.integration
async def test_note_commands_reject_bad_references(project):
    from tostr.commands import note_add_async, note_remove_async
    from tostr.exceptions import TostrError

    cache, registry = await _parsed_cache(project)
    struct = _a_method(registry)

    with pytest.raises(TostrError, match="Struct not found"):
        await note_add_async("M-0000000000", "orphan", "avery", project)

    _, note = await note_add_async(struct.id, "real note", "avery", project)
    with pytest.raises(TostrError, match="has no note 999"):
        await note_remove_async(struct.id, 999, project)
    # The failed lookup must not have disturbed the real note.
    assert len(cache.db.notes_for_structs([struct.id])) == 1


#endregion

"""Integration tests for the live watcher / incremental tree updates.

These exercise the real path the MCP server uses: when a file is saved, the
background watcher must reparse it and update the cached tree in the SQLite DB.

They are hermetic: `get_llm_client` is forced into no-LLM mode so no API key or
network call is required. Embeddings use the locally-cached ONNX model.
"""
from __future__ import annotations
import asyncio
import shutil
import sqlite3
from pathlib import Path

import pytest

import tostr.commands as commands
from tostr.commands import parse_async, process_single_file, process_file_deletion, watch_async
from tostr.core.db import SqliteClient
from tostr.core.paths import ProjectPaths
from tostr.exceptions import APIKeyError

# Whole module is integration: it runs a real init (downloads/loads the embedding
# model) and a live file watcher. CI runs `pytest -m "not integration"` to skip these.
pytestmark = pytest.mark.integration

TEST_PROJECT = Path(__file__).parent / "testcode" / "PythonTestProject"

NEW_METHOD = (
    "\n\n    def promo_special(self) -> float:\n"
    "        return self.apply_discount(0.5)"
)
ANCHOR = "        return self.price * (1 - rate)"


@pytest.fixture
def no_llm(monkeypatch):
    """Force the watcher into no-LLM mode regardless of the ambient GEMINI_API_KEY."""
    def _raise(*args, **kwargs):
        raise APIKeyError("no key (test)")
    monkeypatch.setattr(commands, "get_llm_client", _raise)


@pytest.fixture
def project(tmp_path):
    """A fresh, initialized copy of the Python test project."""
    proj = tmp_path / "proj"
    shutil.copytree(TEST_PROJECT, proj)
    return proj


def method_uids(proj: Path) -> list[str]:
    con = sqlite3.connect(proj / ".tostr" / "cache.db")
    try:
        return [r[0] for r in con.execute("SELECT uid FROM structs WHERE type='method'")]
    finally:
        con.close()


def add_method(proj: Path) -> None:
    models = proj / "models.py"
    models.write_text(models.read_text().replace(ANCHOR, ANCHOR + NEW_METHOD))


async def test_process_single_file_updates_tree(no_llm, project):
    """A modified file is reparsed and the new struct lands in the cached tree."""
    await parse_async(project, no_llm=True)

    before = method_uids(project)
    assert any("apply_discount" in m for m in before)
    assert not any("promo_special" in m for m in before)

    add_method(project)
    # The watcher always hands process_single_file an absolute path.
    await process_single_file(project, (project / "models.py").resolve(), None)

    after = method_uids(project)
    assert any("promo_special" in m for m in after), "new method was not added to the tree"
    # Existing structs are retained and not duplicated.
    assert sum("apply_discount" in m for m in after) == 1
    assert len(after) == len(set(after)), "duplicate structs written on update"


# --- Phase 2: diff-based cache sync (orphan purge + tree-attachment) -------------------------

def _db(proj: Path) -> SqliteClient:
    return SqliteClient(ProjectPaths(proj))


def all_uids(proj: Path) -> list[str]:
    with _db(proj).get_connection() as c:
        return [r[0] for r in c.execute("SELECT uid FROM structs").fetchall()]


def ids_for_uid_substr(proj: Path, substr: str) -> set[str]:
    with _db(proj).get_connection() as c:
        return {str(r[0]) for r in c.execute("SELECT id, uid FROM structs").fetchall() if substr in r[1]}


def vector_ids(proj: Path) -> set[str]:
    with _db(proj).get_connection() as c:
        return {str(r[0]) for r in c.execute("SELECT struct_id FROM vec_structs").fetchall()}


def assert_graph_integrity(proj: Path) -> None:
    """The Phase 2 guarantee: no dangling edges and no orphan vectors anywhere in the DB."""
    with _db(proj).get_connection() as c:
        ids = {str(r[0]) for r in c.execute("SELECT id FROM structs").fetchall()}
        edges = [(str(s), str(t), e) for s, t, e in c.execute("SELECT source_id, target_id, edge_type FROM edges").fetchall()]
        vec_ids = {str(r[0]) for r in c.execute("SELECT struct_id FROM vec_structs").fetchall()}
    dangling = [e for e in edges if e[0] not in ids or e[1] not in ids]
    orphan_vecs = vec_ids - ids
    assert not dangling, f"dangling edges reference missing structs: {dangling}"
    assert not orphan_vecs, f"orphan vectors reference missing structs: {orphan_vecs}"


def remove_apply_discount(proj: Path) -> None:
    models = proj / "models.py"
    block = (
        "    def apply_discount(self, rate: float) -> float:\n"
        "        return self.price * (1 - rate)"
    )
    text = models.read_text()
    assert block in text
    models.write_text(text.replace(block, "        pass"))


async def test_removed_method_is_purged(no_llm, project):
    """Removing a member deletes its struct row, its vector, and any edge touching it — no ghost."""
    await parse_async(project, no_llm=True)

    old_ids = ids_for_uid_substr(project, "apply_discount")
    assert old_ids, "fixture should have apply_discount before edit"
    assert old_ids & vector_ids(project), "method should have a vector before removal"

    remove_apply_discount(project)
    await process_single_file(project, (project / "models.py").resolve(), None)

    assert not any("apply_discount" in u for u in all_uids(project)), "removed method still in structs"
    assert not (old_ids & vector_ids(project)), "removed method's vector was left behind"
    assert_graph_integrity(project)


async def test_renamed_method_drops_old_identity(no_llm, project):
    """Renaming a method (new uid -> new id) purges the old identity rather than orphaning it."""
    await parse_async(project, no_llm=True)
    old_ids = ids_for_uid_substr(project, "apply_discount")
    assert old_ids

    models = project / "models.py"
    models.write_text(models.read_text().replace("apply_discount", "apply_markdown"))
    await process_single_file(project, models.resolve(), None)

    uids = all_uids(project)
    assert not any("apply_discount" in u for u in uids), "old method identity not purged on rename"
    assert any("apply_markdown" in u for u in uids), "renamed method missing"
    assert not (old_ids & vector_ids(project)), "old method's vector survived the rename"
    assert_graph_integrity(project)


async def test_file_keeps_parent_edge_after_update(no_llm, project):
    """A reparsed file must stay attached to its parent directory (is_child_of edge survives)."""
    await parse_async(project, no_llm=True)

    def parent_edge():
        with _db(project).get_connection() as c:
            fid = c.execute("SELECT id FROM structs WHERE uid = 'models.py'").fetchone()[0]
            row = c.execute(
                "SELECT target_id FROM edges WHERE source_id = ? AND edge_type = 'is_child_of'", (fid,)
            ).fetchone()
        return row[0] if row else None

    before = parent_edge()
    assert before is not None, "file should be attached to its directory after init"

    add_method(project)
    await process_single_file(project, (project / "models.py").resolve(), None)

    after = parent_edge()
    assert after == before, "file detached from (or re-pointed away from) its parent directory on update"
    assert_graph_integrity(project)


async def test_removing_depended_on_class_cleans_cross_file_edges(no_llm, project):
    """Removing a class that another file depends on must not leave the caller's edge dangling."""
    await parse_async(project, no_llm=True)

    user_ids = ids_for_uid_substr(project, "models.py#User")
    assert user_ids, "User subtree should exist after init"
    with _db(project).get_connection() as c:
        edges = c.execute("SELECT source_id, target_id FROM edges WHERE edge_type LIKE 'depends_on%'").fetchall()
    inbound = [(str(s), str(t)) for s, t in edges if str(t) in user_ids and str(s) not in user_ids]
    assert inbound, "expected a cross-file dependency edge into the User subtree (e.g. from main.py)"

    models = project / "models.py"
    # Drop the entire User class, keep Product.
    new_src = models.read_text().split("class Product:", 1)[1]
    models.write_text("class Product:" + new_src)
    await process_single_file(project, models.resolve(), None)

    assert not any("models.py#User" in u for u in all_uids(project)), "User subtree not purged"
    assert_graph_integrity(project)  # the key assertion: main.py's edge into User is gone, not dangling


# --- Phase 3: deletion handling -------------------------------------------------------------

async def test_deleted_file_is_purged(no_llm, project):
    """Deleting a file removes its whole subtree and leaves no dangling cross-file edges."""
    await parse_async(project, no_llm=True)
    assert any(u == "models.py" for u in all_uids(project))
    # main.py depends on models — that cross-file edge must not dangle after deletion.
    models_ids = ids_for_uid_substr(project, "models.py")
    with _db(project).get_connection() as c:
        edges = c.execute("SELECT source_id, target_id FROM edges WHERE edge_type LIKE 'depends_on%'").fetchall()
    assert any(str(t) in models_ids and str(s) not in models_ids for s, t in edges), \
        "expected a cross-file dependency into models.py before deletion"

    (project / "models.py").unlink()
    await process_file_deletion(project, (project / "models.py").resolve())

    assert not any("models.py" in u for u in all_uids(project)), "deleted file's structs not purged"
    assert not (models_ids & vector_ids(project)), "deleted file's vectors left behind"
    assert_graph_integrity(project)


async def test_directory_deletion_cascades(no_llm, project):
    """Deleting a directory purges every struct beneath it, not just the directory node."""
    await parse_async(project, no_llm=True)
    assert any(u.startswith("services/") for u in all_uids(project)), "services subtree should exist"
    services_ids = ids_for_uid_substr(project, "services")

    shutil.rmtree(project / "services")
    await process_file_deletion(project, (project / "services").resolve())

    leftover = [u for u in all_uids(project) if u == "services" or u.startswith("services/")]
    assert not leftover, f"directory subtree not fully cascaded: {leftover}"
    assert not (services_ids & vector_ids(project)), "deleted subtree's vectors left behind"
    assert_graph_integrity(project)


# --- Phase 6: description/vector carry-over for unchanged members ----------------------------

async def test_unchanged_members_are_not_regenerated(no_llm, project, monkeypatch):
    """Editing one method regenerates only what changed; untouched siblings reuse cached artifacts.

    Asserted via the embedder (in no-LLM mode the vector is the regenerated artifact; the same
    diff_hash gate governs description reuse). Embed texts are `"{uid}: {body}"`, so we recover
    which uids were embedded.
    """
    await parse_async(project, no_llm=True)

    embedder = commands.get_cached_embedding_client()
    embedded: list[str] = []
    original_embed_batch = embedder.strategy.embed_batch

    def spy(descriptions):
        embedded.extend(descriptions)
        return original_embed_batch(descriptions)

    monkeypatch.setattr(embedder.strategy, "embed_batch", spy)

    # Change ONLY Product.apply_discount's body.
    models = project / "models.py"
    models.write_text(models.read_text().replace(
        "return self.price * (1 - rate)",
        "return self.price * (1.0 - rate)  # tweaked",
    ))
    await process_single_file(project, models.resolve(), None)

    embedded_uids = {t.split(": ", 1)[0] for t in embedded}
    assert any("apply_discount" in u for u in embedded_uids), "changed method should be re-embedded"
    # User.get_display_name lives in the same file but is untouched — it must be carried over.
    assert not any("get_display_name" in u for u in embedded_uids), \
        "an unchanged sibling method was needlessly re-embedded (carry-over failed)"
    assert_graph_integrity(project)


async def test_watcher_live_updates_on_modification(no_llm, project):
    """End-to-end: the running watcher detects a save and updates the DB itself."""
    await parse_async(project, no_llm=True)
    assert not any("promo_special" in m for m in method_uids(project))

    stop_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_async(project, stop_event=stop_event))
    try:
        await asyncio.sleep(1.0)  # let awatch initialize before we touch the file
        add_method(project)

        updated = False
        for _ in range(50):  # poll up to ~15s
            await asyncio.sleep(0.3)
            if any("promo_special" in m for m in method_uids(project)):
                updated = True
                break
        assert updated, "watcher did not live-update the tree after a file modification"
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(watch_task, timeout=5)
        except asyncio.TimeoutError:
            watch_task.cancel()

from __future__ import annotations
import sqlite3
import sqlite_vec
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict

from tostr.core.paths import ProjectPaths

class SqliteClient:
    def __init__(self, paths: ProjectPaths):
        self.db_path = paths.db_path
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Yields a SQLite connection optimized for concurrent swarm reads/writes."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initializes the database schema for the AST Graph."""
        from .versioning import CURRENT_CACHE_VERSION

        with self.get_connection() as conn:
            # Detect a brand-new cache *before* CREATE TABLE IF NOT EXISTS masks it, so we only
            # stamp the format version on creation — never silently re-stamp an existing (possibly
            # stale) cache, which would hide it from the compatibility check.
            is_fresh = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='structs'"
            ).fetchone() is None

            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # NODES TABLE
            conn.execute("""
                CREATE TABLE IF NOT EXISTS structs (
                    id TEXT PRIMARY KEY,
                    uid TEXT UNIQUE NOT NULL,
                    name TEXT,
                    type TEXT NOT NULL, -- e.g., 'BaseFile', 'BaseClass', 'BaseMethod'
                    path TEXT,
                    description TEXT,
                    inbound_dependency_strings TEXT,
                    outbound_dependency_strings TEXT,
                    
                    -- CodeStruct Fields
                    signature TEXT,
                    body TEXT,
                    diff_hash TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    
                    -- Specialized Fields (Stored as JSON or plaintext)
                    imports JSON,
                    inherits JSON,
                    enum_constants JSON,
                    field_type TEXT,
                    arity INTEGER,
                    dependency_names JSON,
                    package TEXT,

                    -- ISO-8601 timestamps
                    date_added TEXT,
                    date_last_updated TEXT
                )
            """)

            # Migration: older databases predate these columns
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(structs)").fetchall()}
            for column, decl in (("package", "TEXT"), ("date_added", "TEXT"), ("date_last_updated", "TEXT")):
                if column not in existing_cols:
                    conn.execute(f"ALTER TABLE structs ADD COLUMN {column} {decl}")

            # EDGES TABLE
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL, -- 'contains', 'depends_on', 'fuzzy_depends_on'
                    
                    PRIMARY KEY (source_id, target_id, edge_type),
                    
                    FOREIGN KEY(source_id) REFERENCES structs(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES structs(id) ON DELETE CASCADE
                )
            """)

            # INDEXES FOR GRAPH TRAVERSAL
            conn.execute("CREATE INDEX IF NOT EXISTS idx_structs_uid ON structs(uid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_structs_type ON structs(type)")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
            
            # VECTORS TABLE (Adjacent virtual table for sqlite-vec)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_structs USING vec0(
                    struct_id TEXT KEY,
                    vector FLOAT[384]
                )
            """)

            # NOTES TABLE (human-authored commentary hanging off a struct)
            #
            # Deliberately a plain table, not fts5: notes are meant to be found through the same
            # semantic vector search as everything else, so a second lexical index would be a
            # parallel — and worse — way to search the project. That leaves a normal table's
            # advantages: typed columns and a real index on `struct_id`, which every read here is
            # keyed on.
            #
            # The FK is declaratively correct but does NOT currently fire: `PRAGMA foreign_keys`
            # is per-connection and only set in this method, so it is off for every connection
            # `get_connection` hands out. Struct deletion must still drop notes explicitly (see
            # `_delete_struct_ids`); the constraint is here so the cascade works the moment that
            # pragma moves into `get_connection`.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY,
                    struct_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    date_added TEXT,
                    date_last_updated TEXT,

                    FOREIGN KEY(struct_id) REFERENCES structs(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_struct ON notes(struct_id)")

            if is_fresh:
                conn.execute(f"PRAGMA user_version = {CURRENT_CACHE_VERSION}")

            conn.commit()

    def struct_type_counts(self) -> Dict[str, int]:
        """`{struct_type: count}` across the whole cache (for `status`)."""
        with self.get_connection() as conn:
            return {r["type"]: r["count"]
                    for r in conn.execute("SELECT type, COUNT(*) AS count FROM structs GROUP BY type")}

    def edge_count(self) -> int:
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def uid_to_id(self, uid: str) -> Optional[str]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT id FROM structs WHERE uid = ?", (uid,)).fetchone()
            return row[0] if row else None

    def struct_exists(self, uid: str) -> bool:
        with self.get_connection() as conn:
            return conn.execute("SELECT 1 FROM structs WHERE uid = ? LIMIT 1", (uid,)).fetchone() is not None

    def struct_date_added(self) -> Dict[str, str]:
        """`{struct_id: date_added}` for every stored struct, so a rewrite can keep the original
        first-seen timestamp instead of resetting it to the time of the reparse."""
        with self.get_connection() as conn:
            return {str(r[0]): r[1]
                    for r in conn.execute("SELECT id, date_added FROM structs WHERE date_added IS NOT NULL")}

    def vector_search(self, query_vector, k: int, filter_type: Optional[str] = None) -> List[sqlite3.Row]:
        """K-nearest structs by embedding distance, optionally filtered by type. Returns raw rows
        (uid, id, type, distance); the caller shapes them into SearchResults."""
        with self.get_connection() as conn:
            sql = (
                "SELECT s.uid, s.id, s.type, v.distance "
                "FROM vec_structs v JOIN structs s ON s.id = v.struct_id "
                "WHERE v.vector MATCH ? AND v.k = ?"
            )
            params = [sqlite_vec.serialize_float32(query_vector), k]
            if filter_type:
                sql += " AND s.type LIKE ?"
                params.append(f"%{filter_type}%")
            return conn.execute(sql, params).fetchall()

    def struct_descriptions(self, path_str: Optional[str] = None, non_empty: bool = False) -> List[sqlite3.Row]:
        """Rows of (id, uid, diff_hash, description) for carry-over reconciliation / lockfile export.
        `path_str` scopes to a single reparsed file; `non_empty` excludes blank descriptions."""
        sql = "SELECT id, uid, diff_hash, description FROM structs"
        clauses, params = [], []
        if path_str is not None:
            clauses.append("path = ?")
            params.append(path_str)
        if non_empty:
            clauses.append("description IS NOT NULL AND description != ''")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.get_connection() as conn:
            return conn.execute(sql, params).fetchall()

    #region NOTES

    @contextmanager
    def _writable(self, conn: Optional[sqlite3.Connection] = None):
        """Reuse a caller's open connection (leaving its transaction to the caller) or open and
        commit our own. Lets note writes participate in a larger cache write without nesting
        connections against the same file."""
        if conn is not None:
            yield conn
        else:
            with self.get_connection() as owned:
                yield owned
                owned.commit()

    def note_count(self) -> int:
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def insert_note(self, struct_id: str, content: str, author: str, date_added: str,
                    date_last_updated: str, conn: Optional[sqlite3.Connection] = None) -> int:
        """Persist one note and return its primary key (the caller mirrors it onto `Note.id`)."""
        with self._writable(conn) as c:
            cursor = c.execute(
                "INSERT INTO notes (content, author, struct_id, date_added, date_last_updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (content, author, str(struct_id), date_added, date_last_updated),
            )
            return cursor.lastrowid

    def update_note(self, note_id: int, content: str, author: str, date_last_updated: str,
                    conn: Optional[sqlite3.Connection] = None) -> bool:
        """Rewrite an existing note in place. Returns False when the note id is already gone."""
        with self._writable(conn) as c:
            cursor = c.execute(
                "UPDATE notes SET content = ?, author = ?, date_last_updated = ? WHERE id = ?",
                (content, author, date_last_updated, int(note_id)),
            )
            return cursor.rowcount > 0

    def delete_note(self, note_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        with self._writable(conn) as c:
            return c.execute("DELETE FROM notes WHERE id = ?", (int(note_id),)).rowcount > 0

    def delete_notes_for_structs(self, ids, conn: Optional[sqlite3.Connection] = None) -> int:
        """Drop every note attached to the given structs. The FK cascade is inert (see the schema
        note in `init_db`), so struct deletion has to call this explicitly or the notes are
        orphaned."""
        ids = [str(i) for i in ids]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self._writable(conn) as c:
            return c.execute(f"DELETE FROM notes WHERE struct_id IN ({placeholders})", ids).rowcount

    def replace_struct_notes(self, struct_id: str, notes: List[dict],
                             conn: Optional[sqlite3.Connection] = None) -> List[int]:
        """Make the stored notes for one struct exactly `notes` (a list of `Note.to_dict()`),
        returning the resulting note ids in order. Used by the save flush, which owns the whole
        in-memory list for that struct."""
        struct_id = str(struct_id)
        with self._writable(conn) as c:
            c.execute("DELETE FROM notes WHERE struct_id = ?", (struct_id,))
            note_ids = []
            for note in notes:
                note_ids.append(self.insert_note(
                    struct_id,
                    note.get("content", "") or "",
                    note.get("author", "") or "",
                    note.get("date_added"),
                    note.get("date_last_updated"),
                    conn=c,
                ))
            return note_ids

    def touch_struct(self, struct_id: str, timestamp: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Stamp a struct's `date_last_updated` without rewriting the row — used when a note write
        changes the struct's metadata but nothing about the parsed code."""
        with self._writable(conn) as c:
            return c.execute(
                "UPDATE structs SET date_last_updated = ? WHERE id = ?", (timestamp, str(struct_id))
            ).rowcount > 0

    def notes_for_structs(self, ids) -> List[sqlite3.Row]:
        """Raw note rows for the given struct ids, oldest first."""
        ids = [str(i) for i in ids]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self.get_connection() as conn:
            return conn.execute(
                "SELECT id, struct_id, content, author, date_added, date_last_updated "
                f"FROM notes WHERE struct_id IN ({placeholders}) ORDER BY id",
                ids,
            ).fetchall()

    #endregion

    def vectors_for_ids(self, ids) -> List[sqlite3.Row]:
        """Raw (struct_id, vector) rows for the given struct ids (caller deserializes the blobs)."""
        ids = [str(i) for i in ids]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self.get_connection() as conn:
            return conn.execute(
                f"SELECT struct_id, vector FROM vec_structs WHERE struct_id IN ({placeholders})", ids
            ).fetchall()

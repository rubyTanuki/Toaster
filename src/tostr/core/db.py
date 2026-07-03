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
        from tostr.core.cache_version import CURRENT_CACHE_VERSION

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
                    package TEXT
                )
            """)

            # Migration: older databases predate the package column
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(structs)").fetchall()}
            if "package" not in existing_cols:
                conn.execute("ALTER TABLE structs ADD COLUMN package TEXT")

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

from __future__ import annotations
import sqlite3
import sqlite_vec
from pathlib import Path
from contextlib import contextmanager

from tostr.core.paths import ProjectPaths

class SQLiteClient:
    def __init__(self, paths: ProjectPaths):
        self.db_path = paths.cache_path / "cache.db"
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

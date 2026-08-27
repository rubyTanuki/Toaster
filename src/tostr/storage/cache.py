from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Set, Dict, TYPE_CHECKING
from collections import defaultdict
import sqlite_vec
from loguru import logger

from tostr.core.paths import ProjectPaths
from tostr.storage import lockfile
from tostr.core.models import *
from tostr.core.builders import BaseBuilder

from .db import SqliteClient

if TYPE_CHECKING:
    from tostr.graph.registry import Registry


def _deserialize_float32(blob) -> Optional[List[float]]:
    """Inverse of sqlite_vec.serialize_float32: turn a stored vector blob back into a float list."""
    if blob is None:
        return None
    import struct as _s
    return list(_s.unpack(f"{len(blob) // 4}f", blob))


class StructCache:
    """Persistence layer over SqliteClient: hydrates structs into the store, writes them back,
    and runs the lockfile / carry-over reconciliation. Holds a `Registry` as its `struct_store`."""
    def __init__(self, paths: ProjectPaths, struct_store: "Registry", use_cache: bool = True, use_lockfile: bool = True):
        self.paths = paths
        self.db = SqliteClient(self.paths)
        self.struct_store = struct_store
        self.use_cache = use_cache
        self.use_lockfile = use_lockfile

    #region READ/HYDRATE

    # Columns skipped by slim hydration: bodies and descriptions dominate row size (~85%) and
    # are never read off hydrated structs used as resolution/skeleton context.
    SLIM_EXCLUDED_COLUMNS = ("body", "description")

    def _apply_timestamps(self, instance: BaseStruct, struct_data: dict):
        """Restore the stored first-seen/last-updated stamps onto a hydrated struct — the builders
        construct from source-shaped fields only, so both would otherwise read as 'now'."""
        if struct_data.get("date_added"):
            instance.date_added = parse_timestamp(struct_data["date_added"])
        if struct_data.get("date_last_updated"):
            instance.date_last_updated = parse_timestamp(struct_data["date_last_updated"])

    def _attach_notes(self, conn, struct_ids: List[str]):
        """Hang each struct's stored notes back onto the hydrated object, carrying the note id
        so a later edit/delete addresses the same row."""
        struct_ids = [sid for sid in struct_ids if sid in self.struct_store.id_map]
        if not struct_ids:
            return
        for chunk_start in range(0, len(struct_ids), 500):
            chunk = struct_ids[chunk_start:chunk_start + 500]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                "SELECT id, struct_id, content, author, date_added, date_last_updated "
                f"FROM notes WHERE struct_id IN ({placeholders}) ORDER BY id",
                chunk,
            ).fetchall()
            for row in rows:
                struct = self.struct_store.id_map.get(str(row["struct_id"]))
                if struct is None:
                    continue
                note = Note.from_dict(dict(row))
                if not any(existing.id == note.id for existing in struct.notes):
                    struct.notes.append(note)

    def load_filepath(self, path: Path, slim: bool = False) -> BaseStruct:
        """Hydrate the subtree under `path` into the store. `slim=True` skips the body and
        description columns — right for read-only context (watcher resolution, skeletons) where
        neither is consumed; use the default full rows when the structs will be rendered."""
        logger.debug(f"Loading subtree {str(path)}")
        path_str = str(self.paths.relative_to_project(path))
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if slim:
                cols = [r[1] for r in cursor.execute("PRAGMA table_info(structs)").fetchall()
                        if r[1] not in self.SLIM_EXCLUDED_COLUMNS]
                select_cols = ", ".join(cols)
            else:
                select_cols = "*"

            if path_str != ".":
                cursor.execute(f"SELECT {select_cols} FROM structs WHERE path = ? OR path LIKE ? || '/%'", (path_str, path_str))
            else:
                cursor.execute(f"SELECT {select_cols} FROM structs")
                
            node_rows = cursor.fetchall()
            node_ids = [str(row["id"]) for row in node_rows]
            
            for row in node_rows:
                struct_data = dict(row)
                
                if struct_data.get("imports", None):
                    struct_data["imports"] = json.loads(struct_data["imports"])
                if struct_data.get("dependency_names", None):
                    struct_data["dependency_names"] = json.loads(struct_data["dependency_names"])
                if struct_data.get("inherits", None):
                    struct_data["inherits"] = json.loads(struct_data["inherits"])
                if struct_data.get("enum_constants", None):
                    struct_data["enum_constants"] = json.loads(struct_data["enum_constants"])
                
                builder = BaseBuilder(self.struct_store)
                struct_type = struct_data["type"]
                instance = builder.with_type(struct_type=struct_type).from_dict(struct_data)
                
                if instance:
                    instance.id = str(struct_data["id"])
                    self._apply_timestamps(instance, struct_data)
                    self.struct_store.add_hydrated_struct(instance)

            # Slim hydration is read-only resolution context, which never reads notes — skip the
            # extra scan there for the same reason bodies and descriptions are skipped.
            if not slim:
                self._attach_notes(conn, node_ids)

            if not node_ids:
                return None

            placeholders = ",".join(["?"] * len(node_ids))
            
            if path_str == ".":
                sql = f"SELECT source_id, target_id, edge_type FROM edges WHERE edge_type = 'is_child_of'"
                cursor.execute(sql)
            else:
                sql = f"""
                    SELECT source_id, target_id, edge_type 
                    FROM edges 
                    WHERE (source_id IN ({placeholders}) 
                    OR target_id IN ({placeholders}))
                    AND edge_type = 'is_child_of'
                """
                params = node_ids + node_ids
                cursor.execute(sql, params)
            
            edge_rows = cursor.fetchall()
            
            for source_id, target_id, edge_type in edge_rows:
                source_obj = self.struct_store.id_map.get(str(source_id))
                target_obj = self.struct_store.id_map.get(str(target_id))
                
                if not source_obj or not target_obj:
                    continue
                
                target_obj.add_child(source_obj)
        
        self.struct_store.root = self.get_struct_by_uid(path_str)
        return self.struct_store.root

    def get_struct_by_uid(self, uid: str) -> Optional[BaseStruct]:
        known = self.struct_store.get_struct_by_uid(uid)
        if known is not None:
            return known

        if uid in self.struct_store.missing_uids:
            return None

        # logger.debug(f"Attempting to retrieve {uid} and its children from DB") 
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if uid != ".":
                cursor.execute(
                    "SELECT * FROM structs WHERE uid = ? OR uid LIKE ? OR uid LIKE ? OR path = ? or uid LIKE ?", 
                    (uid, f"{uid}.%", f"{uid}#%", uid, f"{uid}/%")
                )
            else:
                cursor.execute("SELECT * FROM structs")
            rows = cursor.fetchall()

            if not rows:
                self.struct_store.missing_uids.add(uid)
                return None

            struct_ids = [str(row["id"]) for row in rows]
            target_id = None
            
            for row in rows:
                struct_data = dict(row)
                current_id = str(struct_data["id"])
                if struct_data["uid"] == uid:
                    target_id = current_id
                
                if current_id not in self.struct_store.id_map.keys():
                    for field in ["imports", "dependency_names", "inherits", "enum_constants"]:
                        if struct_data.get(field):
                            struct_data[field] = json.loads(struct_data[field])
                    
                    builder = BaseBuilder(self.struct_store)
                    instance = builder.with_type(struct_type=struct_data["type"]).from_dict(struct_data)
                    if instance:
                        instance.id = current_id
                        self._apply_timestamps(instance, struct_data)
                        self.struct_store.add_hydrated_struct(instance)

            self._attach_notes(conn, struct_ids)

            if not struct_ids:
                return None

            placeholders = ",".join(["?"] * len(struct_ids))
            sql = f"SELECT source_id, target_id, edge_type FROM edges WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) AND edge_type = 'is_child_of'"
            cursor.execute(sql, struct_ids + struct_ids)
            edge_rows = cursor.fetchall()
            
            for _source_id, _target_id, edge_type in edge_rows:
                source_obj = self.struct_store.id_map.get(str(_source_id))
                target_obj = self.struct_store.id_map.get(str(_target_id))
                if source_obj and target_obj:
                    target_obj.add_child(source_obj)

        return self.struct_store.id_map.get(target_id)

    def get_struct_by_id(self, id: str) -> Optional[BaseStruct]:
        id_str = str(id)
        if id_str in self.struct_store.id_map:
            return self.struct_store.id_map[id_str]
            
        if not self.use_cache or not self.db:
            return None
        
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT uid FROM structs WHERE id = ?", (id_str,)).fetchone()
            if not row:
                return None
            target_uid = row[0]
            
        return self.get_struct_by_uid(target_uid)

    #endregion
    
    #region WRITE

    def struct_exists(self, uid: str) -> bool:
        if self.struct_store.get_struct_by_uid(uid) is not None:
            return True
        return bool(self.db) and self.db.struct_exists(uid)

    def _delete_struct_ids(self, conn, ids: Set[str]) -> Set[str]:
        ids = {str(i) for i in ids}
        if not ids:
            return set()
        cursor = conn.cursor()
        ph = ",".join("?" * len(ids))
        rp = list(ids)
        cursor.execute(f"DELETE FROM structs WHERE id IN ({ph})", rp)
        cursor.execute(f"DELETE FROM edges WHERE source_id IN ({ph})", rp)
        cursor.execute(f"DELETE FROM edges WHERE target_id IN ({ph})", rp)
        cursor.execute(f"DELETE FROM vec_structs WHERE struct_id IN ({ph})", rp)
        # The notes FK cascade is inert (foreign_keys is off on these connections — see the schema
        # note in db.init_db), so a deleted struct's notes have to be dropped explicitly.
        self.db.delete_notes_for_structs(ids, conn=conn)
        return ids

    def _prune_file_path(self, conn, path_str: str, kept_ids: Set[str]) -> Set[str]:
        """Delete structs stored under `path_str` whose id is no longer present in `kept_ids`
        (i.e. members removed/renamed out of the file on this reparse). Returns the removed ids."""
        cur = conn.cursor()
        stored = {str(r[0]) for r in cur.execute("SELECT id FROM structs WHERE path = ?", (path_str,)).fetchall()}
        removed = stored - kept_ids
        if removed:
            self._delete_struct_ids(conn, removed)
            logger.debug(f"Pruned {len(removed)} removed struct(s) under '{path_str}'")
        return removed
    
    def delete_path_subtree(self, path_str: str) -> Set[str]:
        """Remove a deleted file or directory from the cache entirely"""
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM structs WHERE path = ? OR path LIKE ? || '/%'",
                (path_str, path_str),
            ).fetchall()
            removed = self._delete_struct_ids(conn, {str(r[0]) for r in rows})
            conn.commit()
        if removed:
            logger.info(f"Deleted {len(removed)} struct(s) under '{path_str}' (file/dir removal)")
        return removed
    
    def save_to_cache(self, stale: bool = False, prune_paths: Optional[List[str]] = None):
        """Persist the *parsed* in-memory structs (hydrated structs are read-only context: their
        dependency edges aren't in memory, so writing them back would wipe those edges and
        stale-mark their descriptions). When `prune_paths` is given (the relative file path(s)
        being reparsed by the watcher), any struct previously stored under those paths but absent
        from this parse is deleted — this is what keeps incremental updates from leaking ghosts
        when a member is removed or renamed. Full re-parses pass no prune_paths."""
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")

        parsed_ids = [(node.id,) for node in self.struct_store.parsed_uid_map.values()]
        grouped_nodes = defaultdict(list)
        all_edges = set()
        vectors = []
        notes_by_struct = {}
        # A reparse rebuilds structs from source, so `date_added` would otherwise be "now" on every
        # write. Keep the first-seen timestamp for ids already in the cache.
        first_seen = self.db.struct_date_added()

        def serialize_for_db(value):
            if isinstance(value, (dict, list, tuple, set)):
                if isinstance(value, set):
                    value = list(value)
                return json.dumps(value)
            return value

        for node in self.struct_store.parsed_uid_map.values():
            data_dict = node.to_dict()
            if stale and data_dict.get("description"):
                data_dict["description"] = f"[STALE] {data_dict['description']}"

            # Extract vector if present for separate virtual table storage
            vector = data_dict.pop("vector", None)
            if vector is not None:
                vectors.append((node.id, sqlite_vec.serialize_float32(vector)))

            # Notes live in their own table, never a `structs` column. Only structs that
            # actually carry notes in memory are flushed: a freshly parsed struct has an empty list
            # (notes are written through the note API, never derived from source), and writing that
            # back would wipe every note in the project on the next reparse.
            notes = data_dict.pop("notes", None)
            if notes:
                notes_by_struct[node.id] = notes

            data_dict["date_added"] = first_seen.get(node.id) or data_dict["date_added"]

            column_footprint = tuple(data_dict.keys())
            grouped_nodes[column_footprint].append(data_dict) 
            all_edges.update(node.edges)
            
        with self.db.get_connection() as conn:
            for columns_tuple, dict_list in grouped_nodes.items():
                columns = ", ".join(columns_tuple)
                placeholders = ", ".join(["?"] * len(columns_tuple))
                node_sql = f"INSERT OR REPLACE INTO structs ({columns}) VALUES ({placeholders})"
                node_values = [tuple(serialize_for_db(n.get(col)) for col in columns_tuple) for n in dict_list]
                conn.executemany(node_sql, node_values)

            conn.executemany("DELETE FROM edges WHERE source_id = ?", parsed_ids)
            if all_edges:
                conn.executemany("INSERT INTO edges (source_id, target_id, edge_type) VALUES (?, ?, ?)", list(all_edges))

            if vectors:
                # vec0 virtual tables have no efficient index on struct_id, so point-delete is O(n).
                # just delete all and re-insert for full reparses
                if prune_paths is None:
                    conn.execute("DELETE FROM vec_structs")
                else:
                    conn.executemany("DELETE FROM vec_structs WHERE struct_id = ?", [(v[0],) for v in vectors])
                conn.executemany("INSERT INTO vec_structs (struct_id, vector) VALUES (?, ?)", vectors)

            for struct_id, notes in notes_by_struct.items():
                note_ids = self.db.replace_struct_notes(struct_id, notes, conn=conn)
                # Mirror the freshly assigned ids back so in-memory notes stay addressable
                # (a note written for the first time has `id is None` until this point).
                struct = self.struct_store.id_map.get(str(struct_id))
                if struct:
                    for note, note_id in zip(struct.notes, note_ids):
                        note.id = note_id

            # Diff-prune: now that the freshly-parsed structs are written, remove anything that used
            # to live under these paths but is gone from this parse (deleted/renamed members).
            if prune_paths:
                kept_ids = {node.id for node in self.struct_store.parsed_uid_map.values()}
                for path_str in prune_paths:
                    self._prune_file_path(conn, path_str, kept_ids)

            conn.commit()

    def add_note(self, struct: BaseStruct, content: str, author: str) -> Note:
        """Attach a note to `struct` and write it straight through to the `notes` table, returning
        the persisted `Note` (its `id` is the row's primary key).

        Write-through rather than deferred: notes are written one at a time by an agent or user,
        not produced by a parse, so there may never be a `save_to_cache` to flush them. When the `_max_notes` cap
        evicts older notes, their rows are deleted here too."""
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")

        persisted_before = {n.id for n in struct.notes if n.id is not None}
        note = struct.add_note(content, author)
        evicted = persisted_before - {n.id for n in struct.notes if n.id is not None}

        with self.db.get_connection() as conn:
            for note_id in evicted:
                self.db.delete_note(note_id, conn=conn)
            note.id = self.db.insert_note(
                struct.id, note.content, note.author,
                note.date_added.isoformat(), note.date_last_updated.isoformat(),
                conn=conn,
            )
            self.db.touch_struct(struct.id, struct.date_last_updated.isoformat(), conn=conn)
            conn.commit()
        logger.debug(f"Added note {note.id} to {struct.uid}")
        return note

    def edit_note(self, note: Note, content: str, author: str, struct: Optional[BaseStruct] = None) -> bool:
        """Rewrite a note in place. Returns False when the note was never persisted or its row is
        gone (the in-memory edit still applies)."""
        note.edit(content, author)
        if not self.db or note.id is None:
            return False
        with self.db.get_connection() as conn:
            updated = self.db.update_note(
                note.id, note.content, note.author, note.date_last_updated.isoformat(), conn=conn
            )
            if struct is not None:
                struct.date_last_updated = note.date_last_updated
                self.db.touch_struct(struct.id, struct.date_last_updated.isoformat(), conn=conn)
            conn.commit()
        return updated

    def delete_note(self, struct: BaseStruct, note: Note) -> bool:
        """Detach a note from `struct` and drop its row."""
        struct.notes = [n for n in struct.notes if n is not note]
        if not self.db or note.id is None:
            return False
        deleted = self.db.delete_note(note.id)
        note.id = None
        return deleted

    def notes_for(self, struct: BaseStruct) -> List[Note]:
        """The struct's notes, read through to the cache when the object was built without them
        (e.g. freshly parsed rather than hydrated)."""
        if struct.notes or not self.db:
            return struct.notes
        struct.notes = [Note.from_dict(dict(row)) for row in self.db.notes_for_structs([struct.id])]
        return struct.notes

    #endregion

    #region RECONCILIATION

    def carry_over_unchanged(self, path_str: Optional[str] = None) -> int:
        """Reuse cached descriptions + vectors for any in-memory struct whose body is unchanged (its
        freshly-computed `diff_hash` matches the stored row), so only *changed* or *new* members pay
        the expensive regeneration cost. The describer skips LLM generation when `description` is
        already set, and the embedder skips when `vector` is set. A leaf method's hash is its own
        body; a class/file hash covers all nested text, so an edited method correctly forces its
        class and file to regenerate while untouched siblings are carried over.

        `path_str` scopes the lookup to one reparsed file (the watcher's incremental path); pass
        None to carry over across the *entire* prior cache, which is what a full `tostr parse` does
        so an unchanged project isn't re-described from scratch. Returns the number carried over."""
        if not self.db:
            return 0
        rows = self.db.struct_descriptions(path_str)
        prev: Dict[str, dict] = {}
        id_to_uid: Dict[str, str] = {}
        for r in rows:
            prev[r["uid"]] = {"diff_hash": r["diff_hash"], "description": r["description"] or "", "vector": None}
            id_to_uid[str(r["id"])] = r["uid"]
        for sid, vec in self.db.vectors_for_ids(id_to_uid):
            uid = id_to_uid.get(str(sid))
            if uid:
                prev[uid]["vector"] = _deserialize_float32(vec)

        carried = 0
        for struct in self.struct_store.parsed_uid_map.values():
            p = prev.get(struct.uid)
            if not p or not struct.diff_hash or p["diff_hash"] != struct.diff_hash:
                continue
            desc = p["description"]
            if desc.startswith("[STALE] "):  # a prior interrupted update may have left the marker
                desc = desc[len("[STALE] "):]
            if desc:
                struct.description = desc
            if p["vector"] is not None:
                struct.vector = p["vector"]
            carried += 1
        if carried:
            scope = f"under '{path_str}'" if path_str is not None else "across the project"
            logger.debug(f"Carried over {carried} unchanged struct description(s)/vector(s) {scope}")
        return carried

    def collect_descriptions(self, with_vectors: bool = False) -> Dict[str, dict]:
        """Read every struct carrying a usable description from the cache into a
        `{uid: {diff_hash, description, vector?}}` map for the lockfile exporter. The `(uid,
        diff_hash)` shape mirrors what `carry_over_unchanged` / the describer's lockfile seed consume,
        so an exported entry is reused only while the body is unchanged. Empty and `[STALE] `
        descriptions are excluded. Vectors are included only when `with_vectors` is set (otherwise the
        consumer re-embeds locally for free). Read half of the lockfile round-trip."""
        if not self.db:
            return {}
        entries: Dict[str, dict] = {}
        rows = self.db.struct_descriptions(non_empty=True)
        id_to_uid: Dict[str, str] = {}
        for row in rows:
            description = row["description"]
            if description.startswith("[STALE] "):
                continue
            uid = row["uid"]
            entries[uid] = {"diff_hash": row["diff_hash"] or "", "description": description}
            id_to_uid[str(row["id"])] = uid

        if with_vectors:
            for struct_id, vector in self.db.vectors_for_ids(id_to_uid):
                uid = id_to_uid.get(str(struct_id))
                deserialized = _deserialize_float32(vector)
                if uid and deserialized is not None:
                    entries[uid]["vector"] = deserialized
        return entries

    def load_lockfile_lookup(self) -> Dict[str, dict]:
        """Load the committed `tostr.lock.json` into an in-memory `{uid: {diff_hash, description,
        vector?}}` lookup for the describer to consult as a *second* description source — after the
        live cache (`carry_over_unchanged`) and before the LLM. Returns `{}` when seeding is disabled
        (`use_lockfile` False) or no usable lockfile exists."""
        if not self.use_lockfile or not self.paths.project_path:
            return {}
        entries = lockfile.read(self.paths.project_path)
        if not entries:
            return {}
        logger.debug(f"Loaded {len(entries)} lockfile entr(ies) from {lockfile.LOCKFILE_NAME}")
        return entries

    #endregion
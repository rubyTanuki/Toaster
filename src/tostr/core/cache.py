from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Set
from collections import defaultdict
import sqlite_vec

from tostr.core.db import SqliteClient
from tostr.core.paths import ProjectPaths
from tostr.core.store import StructStore
from tostr.core.models import *
from tostr.core.builders import BaseBuilder

from loguru import logger

class StructCache:
    """Wrapper on SqliteClient for struct caching read/write"""
    def __init__(self, paths: ProjectPaths, struct_store: StructStore, use_cache: bool = True):
        self.paths = paths
        self.db = SqliteClient(self.paths)
        self.struct_store = struct_store
        self.use_cache = use_cache

    #region READ/HYDRATE

    def load_filepath(self, path: Path) -> BaseStruct:
        logger.debug(f"Loading subtree {str(path)}")
        path_str = str(self.paths.relative_to_project(path))
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if path_str != ".":
                cursor.execute("SELECT * FROM structs WHERE path = ? OR path LIKE ? || '/%'", (path_str, path_str))
            else:
                cursor.execute("SELECT * FROM structs")
                
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
                    self.struct_store.add_struct(instance)
            
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
        if uid in self.struct_store.uid_map:
            return self.struct_store.uid_map[uid]

        if uid in self.struct_store.missing_uids:
            return None

        # logger.debug(f"Attempting to retrieve {uid} and its children from DB") 
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if uid != ".":
                cursor.execute(
                    "SELECT * FROM structs WHERE uid = ? OR uid LIKE ? OR uid LIKE ? OR path = ?", 
                    (uid, f"{uid}.%", f"{uid}#%", uid)
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
                        self.struct_store.add_struct(instance)
            
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
        if uid in self.struct_store.uid_map:
            return True
        
        if not self.db:
            return False
        with self.db.get_connection() as conn:
            return conn.execute("SELECT 1 FROM structs WHERE uid = ? LIMIT 1", (uid,)).fetchone() is not None

    def write_description(self, struct: BaseStruct):
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")
        with self.db.get_connection() as conn:
            conn.execute("UPDATE structs SET description = ? WHERE uid = ?", (struct.description, struct.uid))
            conn.commit()

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
    
    def save_struct_to_cache(self, struct: BaseStruct):
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")
        
        data = struct.to_dict()
        target_uid = data.pop("uid") 
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        node_sql = f"UPDATE structs SET {set_clause} WHERE uid = ?"
        node_params = list(data.values()) + [target_uid]
        
        edges = list(struct.edges)
        with self.db.get_connection() as conn:
            conn.execute(node_sql, node_params)
            conn.execute("DELETE FROM edges WHERE source_id = ?", (struct.id,))
            if edges:
                conn.executemany("INSERT INTO edges (source_id, target_id, edge_type) VALUES (?, ?, ?)", edges)
            conn.commit()

    def save_to_cache(self, stale: bool = False, prune_paths: Optional[List[str]] = None):
        """Persist the in-memory structs. When `prune_paths` is given (the relative file path(s)
        being reparsed by the watcher), any struct previously stored under those paths but absent
        from this parse is deleted — this is what keeps incremental updates from leaking ghosts
        when a member is removed or renamed. Full re-parses pass no prune_paths."""
        if not self.db:
            raise RuntimeError("SqLiteCache not provided.")

        parsed_ids = [(node.id,) for node in self.struct_store.uid_map.values()]
        grouped_nodes = defaultdict(list)
        all_edges = set()
        vectors = []
        
        def serialize_for_db(value):
            if isinstance(value, (dict, list, tuple, set)):
                if isinstance(value, set):
                    value = list(value)
                return json.dumps(value)
            return value
        
        for node in self.struct_store.uid_map.values():
            data_dict = node.to_dict()
            if stale and data_dict.get("description"):
                data_dict["description"] = f"[STALE] {data_dict['description']}"
            
            # Extract vector if present for separate virtual table storage
            vector = data_dict.pop("vector", None)
            if vector is not None:
                vectors.append((node.id, sqlite_vec.serialize_float32(vector)))
            
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
                # vec0 virtual tables do not naturally enforce uniqueness on non-rowid keys during REPLACE
                # so we manually delete to avoid duplicates before inserting.
                conn.executemany("DELETE FROM vec_structs WHERE struct_id = ?", [(v[0],) for v in vectors])
                conn.executemany("INSERT INTO vec_structs (struct_id, vector) VALUES (?, ?)", vectors)

            # Diff-prune: now that the freshly-parsed structs are written, remove anything that used
            # to live under these paths but is gone from this parse (deleted/renamed members).
            if prune_paths:
                kept_ids = {node.id for node in self.struct_store.uid_map.values()}
                for path_str in prune_paths:
                    self._prune_file_path(conn, path_str, kept_ids)

            conn.commit()

    #endregion
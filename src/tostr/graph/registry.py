from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING, Set
from pathlib import Path

from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField, Directory
from tostr.config import ProjectConfig
from tostr.core.paths import ProjectPaths
from tostr.graph.resolver import BaseDependencyResolver

if TYPE_CHECKING:
    from tostr.core.models import BaseStruct
    from tostr.utils.progress import ProgressTracker


class Registry:
    """In-memory store of parsed structs and the object graph's context handle."""

    def __init__(self, paths: ProjectPaths = None, progress_tracker: "ProgressTracker" = None, config: ProjectConfig = None):
        self.progress_tracker = progress_tracker
        # Project path context (root, cache/db locations, relative_to_project) travels as one
        # injected handle so every consumer resolves paths the same way.
        self.paths = paths
        # Structs built by *this* process (parse/builders) vs. structs rehydrated from the cache
        # DB for context.
        self.parsed_uid_map: Dict[str, BaseStruct] = {}
        self.hydrated_uid_map: Dict[str, BaseStruct] = {}
        self.id_map: Dict[str, BaseStruct] = {}
        self.missing_uids: Set[str] = set() # Negative-lookup cache for hydration misses.
        self.root: Optional[BaseStruct] = None
        self.config = config or (ProjectConfig(paths.project_path) if paths else None)
        self._resolvers: Dict[Optional[str], BaseDependencyResolver] = {}
        self._uid_map_cache: Optional[Dict[str, BaseStruct]] = None # Lazily rebuilt on next `uid_map` access after a mutation, instead of on every access
        # Suffix index (last '/'-delimited UID segment -> {uid: struct}) so import resolution can
        # find a path-suffix match without scanning every struct in the project. Kept split by
        # source and probed hydrated-then-parsed to mirror uid_map's merge/precedence order.
        self._suffix_index_hydrated: Dict[str, Dict[str, BaseStruct]] = {}
        self._suffix_index_parsed: Dict[str, Dict[str, BaseStruct]] = {}
        # Name index, restricted to the struct kinds resolve_import's re-export fallback matches
        # against, so that fallback doesn't need to scan every struct in the project either.
        self._name_index_hydrated: Dict[str, Dict[str, BaseStruct]] = {}
        self._name_index_parsed: Dict[str, Dict[str, BaseStruct]] = {}

    @property
    def project_path(self) -> Optional[Path]:
        return self.paths.project_path if self.paths else None

    @property
    def uid_map(self) -> Dict[str, BaseStruct]:
        """All known structs, parsed and hydrated. Parsed wins on a uid collision (a reparse
        supersedes the hydrated prior version). Read-only merged view — register through
        `add_struct` / `add_hydrated_struct`, never by assigning into this dict."""
        if self._uid_map_cache is None:
            self._uid_map_cache = {uid: struct
                                    for source in (self.hydrated_uid_map, self.parsed_uid_map)
                                    for uid, struct in source.items()}
        return self._uid_map_cache

    @staticmethod
    def _suffix_key(uid: str) -> str:
        """The trailing '/'-delimited segment of a UID, used to bucket suffix-match candidates."""
        return uid.rsplit("/", 1)[-1]

    def _index_struct(self, struct: BaseStruct, suffix_index: Dict[str, Dict[str, BaseStruct]],
                       name_index: Dict[str, Dict[str, BaseStruct]]) -> None:
        suffix_index.setdefault(self._suffix_key(struct.uid), {})[struct.uid] = struct
        if isinstance(struct, (BaseClass, BaseMethod, BaseField)):
            name_index.setdefault(struct.name, {})[struct.uid] = struct

    def _deindex_struct(self, struct: BaseStruct, suffix_index: Dict[str, Dict[str, BaseStruct]],
                         name_index: Dict[str, Dict[str, BaseStruct]]) -> None:
        key = self._suffix_key(struct.uid)
        bucket = suffix_index.get(key)
        if bucket:
            bucket.pop(struct.uid, None)
            if not bucket:
                del suffix_index[key]
        if isinstance(struct, (BaseClass, BaseMethod, BaseField)):
            nbucket = name_index.get(struct.name)
            if nbucket:
                nbucket.pop(struct.uid, None)
                if not nbucket:
                    del name_index[struct.name]

    def get_resolver(self, ext: str = "") -> BaseDependencyResolver:
        from tostr.core.providers import LanguageProvider
        lang = LanguageProvider.language_for_extension(ext)
        if lang not in self._resolvers:
            self._resolvers[lang] = LanguageProvider.get_resolver(self, ext)
        return self._resolvers[lang]

    @property
    def language(self) -> str:
        return self.config.language if self.config else "java"

    @property
    def files(self) -> List[BaseFile]:
        return [x for x in self.uid_map.values() if isinstance(x, BaseFile)]

    @property
    def classes(self) -> List[BaseClass]:
        return [x for x in self.uid_map.values() if isinstance(x, BaseClass)]

    @property
    def methods(self) -> List[BaseMethod]:
        return [x for x in self.uid_map.values() if isinstance(x, BaseMethod)]

    @property
    def fields(self) -> List[BaseField]:
        return [x for x in self.uid_map.values() if isinstance(x, BaseField)]

    def add_struct(self, struct: BaseStruct):
        """Register a freshly parsed struct in the in-memory maps and enqueue its pipeline work."""
        self.parsed_uid_map[struct.uid] = struct
        self.id_map[struct.id] = struct
        self._index_struct(struct, self._suffix_index_parsed, self._name_index_parsed)
        self._uid_map_cache = None

        if self.progress_tracker:
            # All structs undergo dependency resolution
            self.progress_tracker.enqueue('resolve', 1)

            # Track describing/embedding only for structs that actually go through the LLM +
            # embedder queue. Fields never do; directories no longer do either — they get a
            # centroid vector computed post-drain, not an LLM description or a queued embed — so
            # counting them would leave the Describing/Embedding bars short and hanging.
            if not isinstance(struct, (BaseField, Directory)):
                self.progress_tracker.enqueue('describe', 1)
                self.progress_tracker.enqueue('embed', 1)

    def add_hydrated_struct(self, struct: BaseStruct):
        """Register a struct rehydrated from the cache DB: visible to lookups and dependency
        resolution, but excluded from write-back (`save_to_cache`) and pipeline progress."""
        self.hydrated_uid_map[struct.uid] = struct
        self.id_map[struct.id] = struct
        self._index_struct(struct, self._suffix_index_hydrated, self._name_index_hydrated)
        self._uid_map_cache = None

    def evict_hydrated_path(self, path_str: str):
        """Drop hydrated structs stored under file path `path_str` — called before reparsing that
        file so members deleted by the edit can't linger and win resolution; the fresh parse
        re-registers the current members as parsed."""
        stale = [uid for uid, s in self.hydrated_uid_map.items() if str(s.path) == path_str]
        for uid in stale:
            struct = self.hydrated_uid_map.pop(uid)
            self.id_map.pop(struct.id, None)
            self._deindex_struct(struct, self._suffix_index_hydrated, self._name_index_hydrated)
        if stale:
            self._uid_map_cache = None

    def get_struct_by_uid(self, uid: str) -> Optional["BaseStruct"]:
        """In-memory lookup by normalized UID (exact). DB hydration lives in StructCache."""
        hit = self.parsed_uid_map.get(uid)
        return hit if hit is not None else self.hydrated_uid_map.get(uid)

    def _find_by_uid_or_suffix(self, candidate: str) -> Optional["BaseStruct"]:
        """Exact UID lookup, else match the candidate as a trailing path segment. The suffix
        match is bucketed by trailing UID segment (`_suffix_key`) rather than scanning every
        struct in the project — this path is the common case for absolute imports in a
        src-layout project, so it needs to stay sub-linear in project size."""
        exact = self.get_struct_by_uid(candidate)
        if exact is not None:
            return exact
        tail = "/" + candidate
        key = self._suffix_key(candidate)
        for uid, struct in self._suffix_index_hydrated.get(key, {}).items():
            if uid.endswith(tail):
                return struct
        for uid, struct in self._suffix_index_parsed.get(key, {}).items():
            if uid.endswith(tail):
                return struct
        return None

    def resolve_import(self, candidate: str) -> Optional["BaseStruct"]:
        """Resolve an import UID *candidate* to a struct: exact match first, else a path-suffix
        match. Absolute imports are namespaced from a source root (e.g. `pkg.a`), but file UIDs are
        project-root-relative, so a `src/`-layout package yields candidate `pkg/a.py#A` for the real
        UID `src/pkg/a.py#A`. Matching the candidate as a trailing path segment bridges that offset
        without the builder needing to know the source root."""
        hit = self._find_by_uid_or_suffix(candidate)
        if hit is not None:
            return hit
        # Re-export fallback: the symbol may be *defined* somewhere other than the module the
        # import names it from — a package `__init__.py` that re-exports it from a submodule
        # (`from tostr.core import BaseParser`). Fall back to a uniquely-named definition anywhere
        # in the project; ambiguous names are left unresolved rather than guessed. Gated on the
        # candidate's module part being a real project file, so an external import
        # (`from loguru import logger`) can never false-match a same-named project struct.
        if "#" in candidate:
            scope_uid, member = candidate.rsplit("#", 1)
            if self._find_by_uid_or_suffix(scope_uid) is not None:
                name = member.split("(")[0]
                candidates: Dict[str, BaseStruct] = dict(self._name_index_hydrated.get(name, {}))
                candidates.update(self._name_index_parsed.get(name, {}))
                if len(candidates) == 1:
                    return next(iter(candidates.values()))
        return None

    def get_struct_by_id(self, id: str) -> Optional["BaseStruct"]:
        return self.id_map.get(str(id))

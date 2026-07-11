from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING, Set
from pathlib import Path

from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField
from tostr.core.context.config import ProjectConfig
from tostr.core.paths import ProjectPaths
from tostr.core.resolver import BaseDependencyResolver

if TYPE_CHECKING:
    from tostr.core.models import BaseStruct
    from tostr.utils.progress import ProgressTracker


class Registry:
    """In-memory store of parsed structs and the object graph's context handle.

    Owns the uid/id indexes, hands out language resolvers, and carries the ambient
    paths/config/progress every struct, builder, and resolver reaches through. Pure memory —
    persistence (hydration, write-back, lockfile/carry-over) lives in `StructCache`, which
    takes a `Registry` as its `struct_store`."""

    def __init__(self, paths: ProjectPaths = None, progress_tracker: "ProgressTracker" = None, config: ProjectConfig = None):
        self.progress_tracker = progress_tracker
        # Project path context (root, cache/db locations, relative_to_project) travels as one
        # injected handle so every consumer resolves paths the same way.
        self.paths = paths
        # Structs built by *this* process (parse/builders) vs. structs rehydrated from the cache
        # DB for context. Split so persistence can write back only what was actually parsed:
        # hydrated structs come out of the DB without their dependency edges in memory, so
        # writing them back would wipe those edges (and stale-mark their descriptions).
        self.parsed_uid_map: Dict[str, BaseStruct] = {}
        self.hydrated_uid_map: Dict[str, BaseStruct] = {}
        self.id_map: Dict[str, BaseStruct] = {}
        # Negative-lookup cache for hydration misses (read by StructCache.get_struct_by_uid).
        self.missing_uids: Set[str] = set()
        self.root: Optional[BaseStruct] = None
        # A caller (e.g. parse) can inject a ProjectConfig carrying per-invocation overrides; else
        # build one from the project root so on-disk tostr.toml / .tostrignore still apply.
        self.config = config or (ProjectConfig(paths.project_path) if paths else None)
        self._resolvers: Dict[Optional[str], BaseDependencyResolver] = {}

    @property
    def project_path(self) -> Optional[Path]:
        return self.paths.project_path if self.paths else None

    @property
    def uid_map(self) -> Dict[str, BaseStruct]:
        """All known structs, parsed and hydrated. Parsed wins on a uid collision (a reparse
        supersedes the hydrated prior version). Read-only merged view — register through
        `add_struct` / `add_hydrated_struct`, never by assigning into this dict."""
        return {uid: struct
                for source in (self.hydrated_uid_map, self.parsed_uid_map)
                for uid, struct in source.items()}

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

        if self.progress_tracker:
            # All structs undergo dependency resolution
            self.progress_tracker.enqueue('resolve', 1)

            # Only track describing and embedding for non-field structs
            if not isinstance(struct, BaseField):
                self.progress_tracker.enqueue('describe', 1)
                self.progress_tracker.enqueue('embed', 1)

    def add_hydrated_struct(self, struct: BaseStruct):
        """Register a struct rehydrated from the cache DB: visible to lookups and dependency
        resolution, but excluded from write-back (`save_to_cache`) and pipeline progress."""
        self.hydrated_uid_map[struct.uid] = struct
        self.id_map[struct.id] = struct

    def evict_hydrated_path(self, path_str: str):
        """Drop hydrated structs stored under file path `path_str` — called before reparsing that
        file so members deleted by the edit can't linger and win resolution; the fresh parse
        re-registers the current members as parsed."""
        stale = [uid for uid, s in self.hydrated_uid_map.items() if str(s.path) == path_str]
        for uid in stale:
            struct = self.hydrated_uid_map.pop(uid)
            self.id_map.pop(struct.id, None)

    def get_struct_by_uid(self, uid: str) -> Optional["BaseStruct"]:
        """In-memory lookup by normalized UID (exact). DB hydration lives in StructCache."""
        hit = self.parsed_uid_map.get(uid)
        return hit if hit is not None else self.hydrated_uid_map.get(uid)

    def _find_by_uid_or_suffix(self, candidate: str) -> Optional["BaseStruct"]:
        """Exact UID lookup, else match the candidate as a trailing path segment."""
        exact = self.get_struct_by_uid(candidate)
        if exact is not None:
            return exact
        tail = "/" + candidate
        for uid, struct in self.uid_map.items():
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
                matches = [s for s in self.uid_map.values()
                           if s.name == name and isinstance(s, (BaseClass, BaseMethod, BaseField))]
                if len(matches) == 1:
                    return matches[0]
        return None

    def get_struct_by_id(self, id: str) -> Optional["BaseStruct"]:
        return self.id_map.get(str(id))

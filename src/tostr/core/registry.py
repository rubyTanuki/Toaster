from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING, Set
from pathlib import Path

from tostr.core.models import BaseFile, BaseClass, BaseMethod, BaseField
from tostr.core.context.config import ProjectConfig
from tostr.core.resolver import BaseDependencyResolver

if TYPE_CHECKING:
    from tostr.core.models import BaseStruct
    from tostr.core.utils.progress import ProgressTracker


class Registry:
    """In-memory store of parsed structs and the object graph's context handle.

    Owns the uid/id indexes, hands out language resolvers, and carries the ambient
    config/progress every struct, builder, and resolver reaches through. Pure memory —
    persistence (hydration, write-back, lockfile/carry-over) lives in `StructCache`, which
    takes a `Registry` as its `struct_store`."""

    def __init__(self, project_path: Path = None, progress_tracker: "ProgressTracker" = None, config: ProjectConfig = None):
        self.progress_tracker = progress_tracker
        self.project_path = project_path
        self.uid_map: Dict[str, BaseStruct] = {}
        self.id_map: Dict[str, BaseStruct] = {}
        # Negative-lookup cache for hydration misses (read by StructCache.get_struct_by_uid).
        self.missing_uids: Set[str] = set()
        self.root: Optional[BaseStruct] = None
        # A caller (e.g. parse) can inject a ProjectConfig carrying per-invocation overrides; else
        # build one from the project root so on-disk tostr.toml / .tostrignore still apply.
        self.config = config or (ProjectConfig(project_path) if project_path else None)
        self._resolvers: Dict[Optional[str], BaseDependencyResolver] = {}

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

    def relative_to_project(self, path: Path) -> Path:
        if not self.project_path:
            return path

        if not path.is_absolute():
            return path

        try:
            return path.resolve().relative_to(self.project_path.resolve())
        except ValueError:
            import os
            try:
                return path.absolute().relative_to(self.project_path.absolute())
            except ValueError:
                return Path(os.path.relpath(path.resolve(), self.project_path.resolve()))

    def add_struct(self, struct: BaseStruct):
        """Register a struct in the in-memory maps and enqueue its pipeline work."""
        self.uid_map[struct.uid] = struct
        self.id_map[struct.id] = struct

        if self.progress_tracker:
            # All structs undergo dependency resolution
            self.progress_tracker.enqueue('resolve', 1)

            # Only track describing and embedding for non-field structs
            if not isinstance(struct, BaseField):
                self.progress_tracker.enqueue('describe', 1)
                self.progress_tracker.enqueue('embed', 1)

    def get_struct_by_uid(self, uid: str) -> Optional["BaseStruct"]:
        """In-memory lookup by normalized UID (exact). DB hydration lives in StructCache."""
        return self.uid_map.get(uid)

    def resolve_import(self, candidate: str) -> Optional["BaseStruct"]:
        """Resolve an import UID *candidate* to a struct: exact match first, else a path-suffix
        match. Absolute imports are namespaced from a source root (e.g. `pkg.a`), but file UIDs are
        project-root-relative, so a `src/`-layout package yields candidate `pkg/a.py#A` for the real
        UID `src/pkg/a.py#A`. Matching the candidate as a trailing path segment bridges that offset
        without the builder needing to know the source root."""
        exact = self.uid_map.get(candidate)
        if exact is not None:
            return exact
        tail = "/" + candidate
        for uid, struct in self.uid_map.items():
            if uid.endswith(tail):
                return struct
        return None

    def get_struct_by_id(self, id: str) -> Optional["BaseStruct"]:
        return self.id_map.get(str(id))

from __future__ import annotations
from pathlib import Path
from abc import ABC
import asyncio
import hashlib
import numpy as np
from loguru import logger

from tostr.core.models import BaseFile, Directory, BaseStruct
from tostr.graph.registry import Registry
from tostr.core.providers import LanguageProvider
from tostr.semantic.describer import LLMDescriber, NoLLMDescriber

class BaseParser(ABC):
    def __init__(self, project_dir: str, llm=None, embedder=None, registry: Registry=None, cache: "StructCache"=None):
        self.project_dir = project_dir
        self.llm = llm
        self.embedder = embedder
        self.registry = registry
        # Persistence half of the old registry: hydration, write-back, lockfile/carry-over.
        self.cache = cache
    
    @property
    def files(self):
        return [x for x in self.registry.uid_map.values() if isinstance(x, BaseFile)]
    
    async def parse(self, subpath: Path = None):
        if not subpath:
            subpath = Path(self.project_dir)
        if not isinstance(subpath, Path):
            subpath = Path(subpath)

        tracker = self.registry.progress_tracker if self.registry else None

        if tracker:
            tracker.phase_start('ast')
        self.parse_path(subpath)
        if tracker:
            tracker.phase_end('ast')

        # Dependency resolution is routed per-file by extension, so it is safe to
        # always run it; files in languages without a resolver are simply skipped.
        self.resolve_dependencies()
        if tracker:
            tracker.phase_end('resolve')

        # Reuse descriptions/vectors from the prior cache for structs whose body is unchanged, so a
        # full reparse only regenerates what actually changed instead of re-describing the whole
        # project. Skipped under --no-cache (use_cache=False), which forces a from-scratch rebuild.
        if self.cache and self.cache.use_cache:
            self.cache.carry_over_unchanged()

        await self.resolve_descriptions_async()
        
    def parse_path(self, subpath: Path, parent: Directory = None):
        if self.registry.config.is_ignored(subpath):
            logger.debug(f"Skipping '{subpath}' due to path ignore rules")
            return

        if subpath.is_dir():
            logger.debug(f"🔍 Parsing directory '{subpath}'")
            
            if parent is None:
                root_path = subpath
                if self.registry:
                    root_path = self.registry.paths.relative_to_project(subpath)
                root = Directory(path=root_path, registry=self.registry)
                self.registry.root = root
                self.registry.add_struct(root)
            else:
                root = parent

            for path in subpath.glob("*"):
                # Always check ignore rules before recursion or file parsing
                if self.registry.config.is_ignored(path):
                    logger.debug(f"Skipping '{path}' due to path ignore rules")
                    continue
                
                relative_path = self.registry.paths.relative_to_project(path)
                
                if path.is_dir():
                    directory = Directory(path=relative_path, registry=self.registry, parent=root)
                    self.registry.add_struct(directory)
                    root.add_child(directory)
                    self.parse_path(path, parent=directory)
                else:
                    file = self.parse_file(path, parent=root)
                    if file:
                        self.registry.add_struct(file)
                        root.add_child(file)
        else:
            file = self.parse_file(subpath)
            if file:
                self.registry.root = file
                self.registry.add_struct(file)
                self._attach_parent_directory(file)

    def _attach_parent_directory(self, file: BaseFile):
        """Re-link a singly-parsed file (watcher path) into the directory tree so its is_child_of
        edge to the parent directory survives the reparse. Without this, save_to_cache deletes the
        file's old parent edge and never re-adds it, orphaning the file from the project tree.

        Walks from the file's immediate parent up to the project root. Directories that already
        exist are *stubbed* (the object only supplies the correct id for the edge — we don't persist
        it, so existing directory rows/descriptions are never clobbered). Directories that don't yet
        exist (a file saved into a brand-new folder) are created and persisted so the edge target is
        real, with their own parent edge linked in turn."""
        if not self.registry or file.path is None:
            return
        child = file
        parent_path = Path(file.path).parent
        while True:
            dir_uid = str(parent_path)
            directory = Directory(path=parent_path, registry=self.registry, uid=dir_uid)
            child.set_parent(directory)
            if self.cache and self.cache.struct_exists(dir_uid):
                break  # existing dir: stub only provides the edge target id; don't persist/overwrite
            # New directory: persist it, then keep walking so its own parent edge is created too.
            self.registry.add_struct(directory)
            if dir_uid == ".":
                break
            child = directory
            parent_path = parent_path.parent

    def parse_file(self, subpath: Path, parent: BaseStruct=None) -> BaseFile:
        logger.debug(f"Attempting to resolve builder for suffix {subpath.suffix}")
        if self.registry.config.is_ignored(subpath):
            logger.debug(f"Skipping '{subpath}' due to path ignore rules")
            return None

        builder = LanguageProvider.get_builder(self.registry, subpath.suffix)
        if builder is None:
            return None
        file_obj = builder.build_file().from_path(subpath, parent=parent)
        return file_obj
    
    def resolve_dependencies(self):
        if self.registry.root:
            logger.info(f"Starting dependency resolution from root: {self.registry.root}")
            self.registry.root.resolve_dependencies()
    
    async def resolve_descriptions_async(self):
        self.embedder.start()
        if self.registry.root:
            if self.llm is None:
                # No-LLM mode: skip descriptions, embed on code context only.
                describer = NoLLMDescriber(self.embedder)
            else:
                # Load the committed lockfile once and hand it to the describer as the second
                # description source (after the live cache, before the LLM) — see apply order in
                # parse(): carry_over_unchanged runs first, then this fills the rest on a cold clone.
                describer = LLMDescriber(self.llm, self.embedder, lockfile=self.cache.load_lockfile_lookup() if self.cache else {})
            await describer.describe(self.registry.root)

        await self.embedder.drain_and_stop()

        # Directory vectors are centroids of their subtree's file vectors, so they can only be
        # computed once every leaf embed has landed — i.e. after the embedder drains.
        self._compute_directory_centroids()

    def _compute_directory_centroids(self):
        """Assign each directory a vector = normalize(mean of all file vectors in its subtree),
        replacing the removed LLM directory description as the directory's search embedding.
        Post-order fold of raw FILE vectors (a directory's direct children are only files and
        subdirectories, so folding files never double-counts a file against its own methods);
        mass-weighted by file count. Full-parse only — the watcher's file-rooted parse has no
        Directory root, so this is a no-op there."""
        root = self.registry.root if self.registry else None
        if not isinstance(root, Directory):
            return

        def fold(directory: Directory):  # -> (sum_vec | None, file_count)
            acc, count = None, 0
            for child in directory.all_children:
                if isinstance(child, Directory):
                    csum, ccount = fold(child)
                elif isinstance(child, BaseFile) and child.vector is not None:
                    csum, ccount = np.asarray(child.vector, dtype=float), 1
                else:
                    continue
                if csum is not None:
                    acc = csum if acc is None else acc + csum
                    count += ccount
            if count and acc is not None:
                mean = acc / count
                norm = np.linalg.norm(mean)
                directory.vector = (mean / norm).tolist() if norm > 1e-9 else mean.tolist()
            else:
                directory.vector = None
            return acc, count

        fold(root)

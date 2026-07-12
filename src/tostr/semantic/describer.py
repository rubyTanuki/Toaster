from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from loguru import logger

from tostr.core.models import BaseStruct, Directory, BaseFile, BaseClass, BaseMethod, BaseField
from tostr.semantic.llm import CLASS_SYSTEM_INSTRUCTION, FILE_SYSTEM_INSTRUCTION, FILE_BODY_SYSTEM_INSTRUCTION, DIRECTORY_SYSTEM_INSTRUCTION

if TYPE_CHECKING:
    from tostr.semantic.llm.base import LLMClient
    from tostr.semantic.embeddings.base import EmbeddingClient


class _DescriberBase:
    """Shared embedding/progress plumbing for the describer implementations."""

    embedder: EmbeddingClient

    def _advance(self, struct: BaseStruct, phase: str, n: int = 1):
        tracker = getattr(getattr(struct, 'registry', None), 'progress_tracker', None)
        if tracker:
            tracker.advance(phase, n)

    def _handle_embed(self, struct: BaseStruct):
        if struct.vector is not None:
            self._advance(struct, 'embed', 1)
        else:
            self.embedder.enqueue(struct)


class NoLLMDescriber(_DescriberBase):
    """Duck-typed stand-in for LLMDescriber used in no-LLM mode.

    Walks the struct tree and enqueues every describable struct for embedding
    without generating any natural-language descriptions. Descriptions stay
    empty; the embedder falls back to the struct's code (body/signature) so
    semantic search still has signal. Mirrors LLMDescriber's contract of
    enqueueing each non-field struct for embedding exactly once.
    """

    def __init__(self, embedder: EmbeddingClient):
        self.embedder = embedder

    async def describe(self, struct: BaseStruct):
        self._walk(struct)

    def _walk(self, struct: BaseStruct):
        if not isinstance(struct, BaseField):
            self._handle_embed(struct)
        for child in struct.all_children:
            self._walk(child)


class LLMDescriber(_DescriberBase):
    def __init__(self, llm: LLMClient, embedder: EmbeddingClient, lockfile: dict | None = None):
        self.llm = llm
        self.embedder = embedder
        # {uid: {"diff_hash", "description", "vector"?}} loaded from the committed tostr.lock.json.
        # Consulted per-struct as the second description source (after the live cache, before the
        # LLM) so a cold clone reuses a teammate's descriptions instead of paying for regeneration.
        self.lockfile = lockfile or {}

    def _seed_from_lockfile(self, struct: BaseStruct) -> bool:
        """Reuse a committed description (and vector, when the lockfile was exported with vectors)
        for a struct whose body is unchanged since the lockfile was written. This is the second
        reuse source, consulted right here in the describe pass: it only fills a struct that has no
        description yet (so a live-cache hit from carry_over_unchanged wins) and only when the
        lockfile entry's diff_hash matches the freshly-parsed struct's (so a changed body falls
        through to regeneration — self-healing). Returns True when a description was seeded, so the
        caller can treat the struct as already described and skip the LLM call."""
        if struct.description or not getattr(struct, "diff_hash", ""):
            return False
        entry = self.lockfile.get(struct.uid)
        if not entry or entry.get("diff_hash") != struct.diff_hash:
            return False
        desc = entry.get("description", "")
        if desc.startswith("[STALE] "):  # defensive: never seed a stale marker
            desc = desc[len("[STALE] "):]
        if not desc:
            return False
        struct.description = desc
        vector = entry.get("vector")
        if vector is not None and struct.vector is None:
            struct.vector = vector
        return True

    async def describe(self, struct: BaseStruct):
        # Second reuse source (after the live cache): fill this struct's description from the lockfile
        # before dispatch, so the already-described early-exit below skips the LLM call for a hit.
        # Methods aren't routed through here (their parent batches them), so they're seeded in
        # _describe_file / _describe_class instead.
        self._seed_from_lockfile(struct)
        if isinstance(struct, Directory):
            await self._describe_directory(struct)
        elif isinstance(struct, BaseFile):
            await self._describe_file(struct)
        elif isinstance(struct, BaseClass):
            await self._describe_class(struct)
        elif isinstance(struct, BaseMethod):
            self._describe_method(struct)
        # BaseField: no description needed

    async def _describe_directory(self, directory: Directory):
        # Early exit before recursion: if already described, children are assumed done too
        if directory.description:
            self._advance(directory, 'describe', 1)
            self._handle_embed(directory)
            return

        if directory.all_children:
            await asyncio.gather(*[self.describe(c) for c in directory.all_children])

        if not directory.all_children:
            directory.description = "Empty Directory"
            self._advance(directory, 'describe', 1)
            self._advance(directory, 'embed', 1)
            return

        if len(directory.all_children) == 1:
            directory.description = directory.all_children[0].description
            self._advance(directory, 'describe', 1)
            self._advance(directory, 'embed', 1)
            return

        logger.debug(f"Generating Description for directory {directory.uid}...")

        class DirectoryDescriptionSchema(BaseModel):
            description: str = Field(description="Description of the directory")

        response = await self.llm.generate_description(
            input_data={c.uid: c.description for c in directory.all_children},
            system_prompt=DIRECTORY_SYSTEM_INSTRUCTION,
            response_schema=DirectoryDescriptionSchema,
        )

        if not response:
            logger.warning(f"⚠️ Skipping {directory.uid} due to LLM failure")
            self._advance(directory, 'embed', 1)
            return

        directory.description = response.description
        self.embedder.enqueue(directory)
        logger.debug(f"Successfully Generated Description for directory {directory.uid}")

    async def _describe_file(self, file: BaseFile):
        # Seed methods from the lockfile up front (they aren't routed through describe()) so the
        # branches below — both the already-described early-exit and the LLM batch's method_lookup —
        # see their reused descriptions and never regenerate an unchanged method.
        for m in file.methods:
            self._seed_from_lockfile(m)

        # Recurse on non-method children (classes) first so their descriptions are ready
        non_methods = [c for c in file.all_children if not isinstance(c, BaseMethod)]
        if non_methods:
            await asyncio.gather(*[self.describe(c) for c in non_methods])

        if file.description:
            self._advance(file, 'describe', 1 + len(file.methods))
            self._handle_embed(file)
            for m in file.methods:
                if m.vector is not None:
                    self._advance(m, 'embed', 1)
                else:
                    self.embedder.enqueue(m)
            return

        if not file.all_children:
            # No extractable sub-components. For a file-level-only language (e.g. HTML) the
            # file still carries a body, so describe it from that raw content rather than
            # aggregating children. A genuinely empty file falls back to the "Empty File" tag.
            if (file.body or "").strip():
                await self._describe_file_from_body(file)
            else:
                file.description = "Empty File"
                self._advance(file, 'describe', 1)
                self._advance(file, 'embed', 1)
            return

        if len(file.all_children) == 1 and not isinstance(file.all_children[0], BaseMethod):
            file.description = file.all_children[0].description
            file.vector = file.all_children[0].vector
            self._advance(file, 'describe', 1)
            self._advance(file, 'embed', 1)
            return

        logger.debug(f"Generating Description for file {file.uid}...")

        method_lookup = {idx: m for idx, m in enumerate(file.methods) if not m.description}

        already_described = [m for m in file.methods if m.description]
        if already_described:
            self._advance(file, 'describe', len(already_described))
            for m in already_described:
                if m.vector is not None:
                    self._advance(m, 'embed', 1)
                else:
                    self.embedder.enqueue(m)

        class FileDescriptionSchema(BaseModel):
            description: str = Field(description="Description of the file")
            description_map: dict[str, str] = Field(default_factory=dict, description="Mapping of method_id to description")

        response = await self.llm.generate_description(
            input_data={
                "described_components": {c.uid: c.description for c in file.all_children if c.description},
                "un_described_methods": {str(idx): m.body for idx, m in method_lookup.items()},
            },
            system_prompt=FILE_SYSTEM_INSTRUCTION,
            response_schema=FileDescriptionSchema,
        )

        if not response:
            logger.warning(f"⚠️ Skipping {file.uid} due to LLM failure")
            self._advance(file, 'describe', 1 + len(method_lookup))
            self._advance(file, 'embed', 1 + len(method_lookup))
            return

        file.description = response.description
        self.embedder.enqueue(file)

        handled_indices = self._apply_description_map(response.description_map, method_lookup)

        if handled_indices:
            self._advance(file, 'describe', len(handled_indices))

        missed = len(method_lookup) - len(handled_indices)
        if missed > 0:
            self._advance(file, 'describe', missed)
            self._advance(file, 'embed', missed)

        logger.debug(f"Successfully Generated Description for file {file.uid}")

    async def _describe_file_from_body(self, file: BaseFile):
        """Describe a childless file directly from its raw body (file-level-only languages
        such as HTML). The normal _describe_file path aggregates child descriptions, which
        yields nothing for a file with no extractable structs; here the body is the signal."""
        logger.debug(f"Generating Description for file {file.uid} from body...")

        class FileBodyDescriptionSchema(BaseModel):
            description: str = Field(description="Description of the file")

        response = await self.llm.generate_description(
            input_data={"path": file.uid, "content": file.body},
            system_prompt=FILE_BODY_SYSTEM_INSTRUCTION,
            response_schema=FileBodyDescriptionSchema,
        )

        if not response:
            logger.warning(f"⚠️ Skipping {file.uid} due to LLM failure")
            self._advance(file, 'describe', 1)
            self._advance(file, 'embed', 1)
            return

        file.description = response.description
        self.embedder.enqueue(file)
        logger.debug(f"Successfully Generated Description for file {file.uid}")

    async def _describe_class(self, cls: BaseClass):
        # Seed methods from the lockfile up front (see _describe_file) so unchanged methods reuse
        # their committed descriptions instead of being regenerated.
        for m in cls.methods:
            self._seed_from_lockfile(m)

        # Recurse on nested classes before describing this one
        non_methods = [c for c in cls.all_children if not isinstance(c, BaseMethod)]
        if non_methods:
            await asyncio.gather(*[self.describe(c) for c in non_methods])

        if cls.description:
            self._advance(cls, 'describe', 1 + len(cls.methods))
            self._handle_embed(cls)
            for m in cls.methods:
                if m.vector is not None:
                    self._advance(m, 'embed', 1)
                else:
                    self.embedder.enqueue(m)
            return

        logger.debug(f"Generating Description for class {cls.uid}...")

        method_lookup = {idx: m for idx, m in enumerate(cls.methods) if not m.description}

        already_described = [m for m in cls.methods if m.description]
        if already_described:
            self._advance(cls, 'describe', len(already_described))
            for m in already_described:
                if m.vector is not None:
                    self._advance(m, 'embed', 1)
                else:
                    self.embedder.enqueue(m)

        class ClassDescriptionSchema(BaseModel):
            description: str = Field(description="Description of the class")
            description_map: dict[str, str] = Field(default_factory=dict, description="Mapping of method_id to description")

        response = await self.llm.generate_description(
            input_data={
                "code": cls.skeletonize(),
                "method_ids_to_signatures": {str(idx): m.signature for idx, m in method_lookup.items()},
            },
            system_prompt=CLASS_SYSTEM_INSTRUCTION,
            response_schema=ClassDescriptionSchema,
        )

        if not response:
            logger.warning(f"⚠️ Skipping {cls.uid} due to LLM failure")
            self._advance(cls, 'describe', 1 + len(method_lookup))
            self._advance(cls, 'embed', 1 + len(method_lookup))
            return

        cls.description = response.description
        self.embedder.enqueue(cls)

        handled_indices = self._apply_description_map(response.description_map, method_lookup)

        if handled_indices:
            self._advance(cls, 'describe', len(handled_indices))

        missed = len(method_lookup) - len(handled_indices)
        if missed > 0:
            self._advance(cls, 'describe', missed)
            self._advance(cls, 'embed', missed)

        logger.debug(f"Successfully Generated Description for class {cls.uid}")

    def _describe_method(self, method: BaseMethod):
        # Methods are handled by their parent class/file batch call.
        # This handles the edge case of a directly-called method.
        self._advance(method, 'describe', 1)
        self._handle_embed(method)

    def _apply_description_map(self, description_map: dict, method_lookup: dict) -> set:
        """Applies LLM-returned descriptions to methods and enqueues them for embedding."""
        handled = set()
        for idx_str, desc in description_map.items():
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue
            if idx in method_lookup:
                method_lookup[idx].description = desc
                handled.add(idx)
                self.embedder.enqueue(method_lookup[idx])
        return handled

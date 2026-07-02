from __future__ import annotations
import asyncio
import os
import shutil
from pathlib import Path
from watchfiles import awatch, Change
from loguru import logger
from functools import lru_cache

from tostr.semantic.llm import LLMClient
from tostr.semantic.embeddings import EmbeddingClient, EmbeddingStrategy, OnnxEmbeddingStrategy
from tostr.core import Registry, tost, InspectResult, SkeletonResult, SearchResult, BaseParser, SqliteClient, ProjectPaths, StructCache, BaseCodeStruct
from tostr.core import lockfile
from tostr.core.context.config import ProjectConfig, default_ignore_text

from tostr.core.cache_version import incompatibility_reason, read_db_version
from tostr.exceptions import TostrError, APIKeyError, DatabaseNotFoundError, CacheFormatError

def _verify_db_exists(target_path: Path):
    """Ensure an initialized Tostr database exists for the given project path.

    target_path is the project root; the actual database lives at
    .tostr/cache.db inside it. Raising here (instead of letting SqliteClient
    lazily create an empty schema) gives the user a clean 'run init first'
    message rather than a downstream stack trace.
    """
    db_path = Path(target_path) / ".tostr" / "cache.db"
    if not db_path.exists():
        raise DatabaseNotFoundError(
            f"No Tostr database found at {db_path}. Run 'tostr parse' first."
        )
    reason = incompatibility_reason(read_db_version(db_path))
    if reason:
        raise CacheFormatError(
            f"Tostr cache at {db_path} is incompatible with this version ({reason}). "
            f"Run 'tostr parse' to rebuild it."
        )

def get_llm_client(config: ProjectConfig, progress_tracker: "ProgressTracker" = None):
    """Build the LLM client for the strategy resolved from `config`, or None when LLM generation
    is disabled (strategy "none"). Resolution and construction live in providers.LLMProvider:
    --llm override > tostr.toml [llm].strategy > gemini default."""
    from tostr.core.providers import LLMProvider
    return LLMProvider.build_client(config, progress_tracker=progress_tracker)

@lru_cache(maxsize=1)
def get_cached_embedding_client(progress_tracker: "ProgressTracker" = None):
    """
    Guarantees the model graph is initialized exactly once 
    per long-running process lifecycle.
    """
    strategy = OnnxEmbeddingStrategy()
    return EmbeddingClient(strategy=strategy, progress_tracker=progress_tracker)

def clean_db(target_path: Path, purge: bool = False):
    """Remove generated state. By default this wipes only `.tostr/` (the disposable cache/logs);
    authored config (`tostr.toml`, `.tostrignore`) is the user's source of truth and is left intact.
    `purge=True` opts into deleting that authored config too, for a full reset."""
    if os.path.exists(target_path / ".tostr"):
        shutil.rmtree(target_path / ".tostr")
        logger.info("Database cleaned.")
    else:
        logger.warning("No database found to clean.")

    if purge:
        for authored in (target_path / "tostr.toml", target_path / ".tostrignore"):
            if authored.exists():
                authored.unlink()
                logger.info(f"Purged {authored}")

def export_lockfile(target_path: Path, with_vectors: bool = False) -> dict:
    """Snapshot the cache's descriptions to `<project_root>/tostr.lock.json` for version control, so
    teammates on a cold clone can seed descriptions instead of re-calling the LLM. Orchestration
    only: the StructCache gathers the exportable entries from the cache (`collect_descriptions`) and the
    `lockfile` module owns the on-disk format (deterministic, version-stamped, no-op-aware write).
    Only descriptions are exported by default; `with_vectors=True` also exports vectors for literal
    zero recompute at the cost of a larger, merge-noisy file. Returns {path, entries_written,
    changed}."""
    _verify_db_exists(target_path)
    registry = Registry(project_path=target_path)
    cache = StructCache(ProjectPaths(target_path), struct_store=registry)

    entries = cache.collect_descriptions(with_vectors=with_vectors)
    changed = lockfile.write(target_path, entries)

    return {"path": str(lockfile.path_for(target_path)), "entries_written": len(entries), "changed": changed}

def get_status(target_path: Path) -> dict:
    db_path = target_path / ".tostr" / "cache.db"
    status = {
        "project_path": str(target_path.absolute()),
        "db_exists": db_path.exists(),
        "db_path": str(db_path) if db_path.exists() else None,
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "last_updated": db_path.stat().st_mtime if db_path.exists() else None,
        "counts": {}
    }

    if status["db_exists"]:
        db = SqliteClient(ProjectPaths(target_path))
        counts = db.struct_type_counts()
        counts["edges"] = db.edge_count()
        status["counts"] = counts

    return status

async def _build_ast_async(target_path: Path, config: ProjectConfig, use_cache: bool = True, progress_tracker: "ProgressTracker" = None, no_llm: bool = False) -> BaseParser:
    llm = None if no_llm else get_llm_client(config, progress_tracker=progress_tracker)
    embedder = get_cached_embedding_client(progress_tracker=progress_tracker)
    paths = ProjectPaths(target_path)
    registry = Registry(project_path=target_path, progress_tracker=progress_tracker, config=config)
    cache = StructCache(paths, struct_store=registry, use_cache=use_cache)
    logger.info("Building AST...")

    parser = BaseParser(target_path, llm, embedder, registry, cache)
    logger.info("Parsing files...")
    await parser.parse()
    logger.success("✅ Parsed files")
    return parser


# Default `tostr.toml` scaffold. Only [project].language and [llm].strategy are wired into behavior
# today; [embedding] and [graph] are documented (commented) so the file shows the roadmap without
# shipping settings that silently do nothing.
DEFAULT_TOSTR_TOML = '''\
# tostr.toml — Tostr project configuration. Safe to commit.

[project]
# "auto" parses every supported language, routed per-file by extension.
language = "auto"            # "auto" | "python" | "java" | ...

[llm]
# Description generation. "none" skips the LLM entirely (embeddings fall back to
# code context); equivalent to the --no-llm flag.
strategy = "gemini"         # "gemini" | "none"

# [embedding]
# Reserved / not yet enforced.
# model = "all-MiniLM-L6-v2"

# [graph]
# Reserved / not yet implemented. Dependency-graph scoping.
# include = []              # globs; empty = everything not excluded
# exclude = []              # globs
'''


def init_project(target_path: Path, force: bool = False) -> list[str]:
    """Scaffold the project's authored files so the user has something concrete to edit. Does NOT
    parse, build the cache, or need an API key. Never clobbers an existing authored file unless
    `force=True`, so it is safe to run repeatedly. Returns human-readable lines describing what
    happened for the caller to report."""
    report: list[str] = []

    # Generated-state home: created empty here; parse fills it.
    tostr_dir = target_path / ".tostr"
    tostr_dir.mkdir(exist_ok=True)
    report.append(f"Ensured {tostr_dir}/")

    # tostr.toml (authored, committed)
    toml_path = target_path / "tostr.toml"
    if toml_path.exists() and not force:
        report.append(f"Skipped {toml_path.name} (already exists; use --force to overwrite)")
    else:
        toml_path.write_text(DEFAULT_TOSTR_TOML)
        report.append(f"Wrote {toml_path.name}")

    # .tostrignore (authored, committed) — materialize the bundled defaults for the configured
    # language so the user can see and edit the rules ProjectConfig would otherwise apply in-code.
    config = ProjectConfig(target_path)
    ignore_path = target_path / ".tostrignore"
    if ignore_path.exists() and not force:
        report.append(f"Skipped {ignore_path.name} (already exists; use --force to overwrite)")
    else:
        ignore_path.write_text(default_ignore_text(config.language))
        report.append(f"Wrote {ignore_path.name}")

    # Keep generated state out of version control.
    report.append(_append_gitignore(target_path))
    return report


def _append_gitignore(target_path: Path) -> str:
    """Ensure `.tostr/` is gitignored. Idempotent.

    If a `.gitignore` already exists, append the entry. If it doesn't, only create one when this is
    actually a git repo (where keeping the generated cache untracked matters — including a freshly
    `git init`'d project that has no `.gitignore` yet). Outside a repo there's nothing for a
    `.gitignore` to do, so we skip it rather than litter the directory with one."""
    gitignore = target_path / ".gitignore"
    entry = ".tostr/"

    if gitignore.exists():
        existing = gitignore.read_text()
        if any(line.strip() == entry for line in existing.splitlines()):
            return f"{gitignore.name} already ignores {entry}"
        prefix = "" if (existing == "" or existing.endswith("\n")) else "\n"
        with open(gitignore, "a") as f:
            f.write(f"{prefix}{entry}\n")
        return f"Added {entry} to {gitignore.name}"

    if (target_path / ".git").exists():
        gitignore.write_text(f"{entry}\n")
        return f"Created {gitignore.name} ignoring {entry}"

    return "Skipped .gitignore (not a git repo)"


async def parse_async(target_path: Path, use_cache: bool = True, language: str | None = None, progress_tracker: "ProgressTracker" = None, no_llm: bool = False, llm: str | None = None):
    """Build the cache: parse the AST, resolve dependencies, describe, embed, write `.tostr/cache.db`.

    Reads configuration via ProjectConfig and authors nothing — no tostr.toml or .tostrignore is
    created. `language` is a per-invocation override (None = use tostr.toml, defaulting to "auto",
    which parses every supported language). `llm` is a per-invocation override of the LLM strategy
    ("gemini"/"ollama"/"none"; None = use tostr.toml [llm].strategy, default "gemini"). CLI flags
    override config for this invocation only.
    """
    tostr_dir = target_path / ".tostr"
    tostr_dir.mkdir(exist_ok=True)

    # If an existing cache is from an incompatible format, wipe it so we rebuild fresh and re-stamp
    # the current version — otherwise INSERT OR REPLACE would leave a mixed-format graph behind.
    db_path = tostr_dir / "cache.db"
    if db_path.exists() and incompatibility_reason(read_db_version(db_path)):
        logger.warning("Existing cache is an incompatible format; wiping it for a clean rebuild.")
        for p in (db_path, db_path.with_name("cache.db-wal"), db_path.with_name("cache.db-shm")):
            p.unlink(missing_ok=True)

    overrides = {}
    if language:
        overrides["language"] = language
    if llm:
        overrides["llm"] = llm
    config = ProjectConfig(target_path, overrides=overrides or None)

    # config-level LLM opt-out (llm.strategy = "none") is equivalent to the --no-llm flag.
    no_llm = no_llm or config.llm_strategy == "none"

    # Parse and resolve AST
    parser = await _build_ast_async(target_path, config, use_cache=use_cache, progress_tracker=progress_tracker, no_llm=no_llm)

    # Write Cache
    parser.cache.save_to_cache()

def resolve_uid_to_id(uid: str, project_path: Path) -> str:
    """Simplifies UID to ID resolution by querying the database directly."""
    _verify_db_exists(project_path)
    db = SqliteClient(ProjectPaths(project_path))
    return db.uid_to_id(uid)
    
async def inspect_async(struct_ids: list[str], project_path: Path, include_body: bool = False):
    _verify_db_exists(project_path)
    
    registry = Registry(project_path=project_path)
    cache = StructCache(ProjectPaths(project_path), struct_store=registry)

    results = []
    for struct_id in struct_ids:
        # Check if it's a UID and needs resolution
        if not (struct_id.startswith(("S-", "C-", "M-", "V-", "F-", "D-"))):
            resolved_id = resolve_uid_to_id(struct_id, project_path)
            if resolved_id:
                struct_id = resolved_id
        
        struct_obj = cache.get_struct_by_id(struct_id)
        if struct_obj is None:
            results.append(f"Error: Struct not found with id/uid {struct_id}.")
            continue
        
        results.append(tost.dump(struct_obj, include_body=include_body))
        
    return results

async def skeleton_async(subpath: str, project_path: Path, depth: int = 7, files_only: bool = False):
    _verify_db_exists(project_path)
    
    registry = Registry(project_path=project_path)
    cache = StructCache(ProjectPaths(project_path), struct_store=registry)

    subpath = Path(project_path / subpath)
    logger.debug(f"Loading subtree for path: {subpath}")

    cache.load_filepath(subpath)
    if not registry.files:
        raise FileNotFoundError(f"No files found matching path '{subpath}'.")
    
    return tost.dump_skeleton(registry.root, depth=depth, files_only=files_only)

active_tasks = {}

async def watch_async(target_path: Path, stop_event: asyncio.Event = None):
    # Resolve up front so symlinked roots (e.g. macOS /var -> /private/var) match the
    # already-resolved paths that awatch emits; otherwise relative_to() below would raise.
    target_path = Path(target_path).resolve()
    _verify_db_exists(target_path)
    config = ProjectConfig(target_path)
    try:
        llm = get_llm_client(config)
    except TostrError as e:
        llm = None
        logger.warning(f"LLM unavailable ({e}); watcher running in no-LLM mode (embeddings only, descriptions skipped).")

    # Single-writer lock for all DB mutations from this watcher. Created here so it binds to this
    # loop, and serializes the per-file write pipelines so a cancelled-but-in-flight thread write
    # can't interleave with (or land after) its replacement. Passed into the task coroutines.
    write_lock = asyncio.Lock()

    logger.info("Starting Listener")
    try:
        async for changes in awatch(target_path, stop_event=stop_event):
            for change_type, raw_path in changes:
                # Keep the absolute path: process_single_file opens the file directly, and
                # the MCP server's cwd is not the project root, so a relative path can't be read.
                abs_path = Path(raw_path).resolve()
                if config.is_ignored(abs_path):
                    continue
                try:
                    display_path = abs_path.relative_to(target_path)
                except ValueError:
                    display_path = abs_path

                existing_task = active_tasks.get(abs_path)
                if existing_task and not existing_task.done():
                    existing_task.cancel()

                match change_type:
                    case Change.deleted:
                        logger.info(f"File deleted: {display_path}")
                        new_task = asyncio.create_task(
                            process_file_deletion(target_path, abs_path, write_lock=write_lock)
                        )
                    case Change.added:
                        logger.info(f"File added: {display_path}")
                        new_task = asyncio.create_task(
                            process_single_file(target_path, abs_path, llm, write_lock=write_lock)
                        )
                    case _:  # Change.modified
                        logger.info(f"File modified: {display_path}")
                        new_task = asyncio.create_task(
                            process_single_file(target_path, abs_path, llm, write_lock=write_lock)
                        )

                active_tasks[abs_path] = new_task

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("\n🛑 Stopping listener...")

async def search_async(query: str, project_path: Path, filter_type: str = None, top_k: int = 5):
    _verify_db_exists(project_path)
    embedder = get_cached_embedding_client()
    query_vector = embedder.strategy.embed_query(query)
    
    db = SqliteClient(ProjectPaths(project_path))

    # Fetch a larger k from the vector index to leave room for the type filter.
    k_to_fetch = top_k * 10 if filter_type else top_k
    rows = db.vector_search(query_vector, k_to_fetch, filter_type)

    results = []
    for row in rows:
        results.append(SearchResult(id=row['id'], uid=row['uid'], type=row['type'], distance=row['distance']))
        if len(results) >= top_k:
            break

    return results

async def _guarded_write(write_lock: "asyncio.Lock | None", filepath: Path, fn, *args, **kwargs):
    """Run a DB-mutating call off-loop, serialized behind `write_lock` when one is supplied
    (watcher mode). Inside the lock we re-check that this task is still the live task for `filepath`
    and skip the write if it's been superseded by a newer change — so a cancelled-but-in-flight
    pipeline can't clobber the result of the change that replaced it. When `write_lock` is None
    (direct callers / tests, which await each call to completion) we just run the write."""
    if write_lock is None:
        return await asyncio.to_thread(fn, *args, **kwargs)
    async with write_lock:
        if active_tasks.get(filepath) is not asyncio.current_task():
            logger.debug(f"Skipping superseded DB write for {filepath}")
            return None
        return await asyncio.to_thread(fn, *args, **kwargs)


async def process_file_deletion(project_dir: Path, filepath: Path, write_lock: "asyncio.Lock | None" = None):
    """Watcher deletion path: purge a removed file (or directory) and everything beneath it from
    the cache, so no ghost structs/edges/vectors linger. The path is relativized the same way the
    builders store it; `delete_path_subtree` cascades for directories via a path-prefix match."""
    logger.info(f"Processing deletion {filepath}")
    try:
        registry = Registry(project_path=project_dir)
        cache = StructCache(ProjectPaths(project_dir), struct_store=registry)
        rel_path = str(registry.relative_to_project(Path(filepath)))
        removed = await _guarded_write(write_lock, filepath, cache.delete_path_subtree, rel_path)
        logger.debug(f"✅ Deleted {len(removed or [])} struct(s) for {rel_path}")
    except asyncio.CancelledError:
        logger.warning(f"Deletion task cancelled on {filepath}")
    except Exception as e:
        logger.warning(f"Error deleting {filepath}: {e}")
    finally:
        if active_tasks.get(filepath) == asyncio.current_task():
            del active_tasks[filepath]


async def process_single_file(project_dir: Path, filepath: Path, llm_client: LLMClient, write_lock: "asyncio.Lock | None" = None):
    logger.info(f"Processing file {filepath}")
    try:
        registry = Registry(project_path=project_dir)
        cache = StructCache(ProjectPaths(project_dir), struct_store=registry)
        embedder = get_cached_embedding_client()
        parser = BaseParser(filepath, llm=llm_client, embedder=embedder, registry=registry, cache=cache)

        parser.parse_path(filepath)

        # Nothing parsed (unsupported/ignored file) — don't prune, or we'd wipe the path's structs.
        if registry.root is None:
            logger.debug(f"No parseable struct produced for {filepath}; skipping cache write")
            return

        parser.resolve_dependencies()

        # Scope the diff-prune to exactly this file's relative path so removed/renamed members are
        # purged. Matches what the builders store (BaseFileBuilder.from_path relativizes the path).
        prune_paths = [str(registry.relative_to_project(Path(filepath)))]

        # Reuse cached descriptions/vectors for members whose body is unchanged, so the describe +
        # embed pass below only regenerates what actually changed (must run before both writes so
        # the stale write doesn't blank carried-over descriptions).
        await asyncio.to_thread(cache.carry_over_unchanged, prune_paths[0])

        await _guarded_write(write_lock, filepath, cache.save_to_cache, stale=True, prune_paths=prune_paths)
        logger.debug("Wrote Cache w/ stale descriptions")

        # resolve the descriptions and do the second cache write
        await parser.resolve_descriptions_async()
        await _guarded_write(write_lock, filepath, cache.save_to_cache, prune_paths=prune_paths)
        logger.debug("Wrote Cache w/ resolved descriptions")

        logger.debug(f"✅ Processed file {filepath}")

    except asyncio.CancelledError:
        logger.warning(f"Task cancelled on {filepath}")
    except Exception as e:
        logger.warning(f"Error processing file {filepath}: {e}")
    finally:
        if active_tasks.get(filepath) == asyncio.current_task():
            del active_tasks[filepath]
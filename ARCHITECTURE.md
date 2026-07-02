# Architecture

## Overview
Tostr is a CLI/MCP tool for agentic programming which pre-computes context in a code repository, allowing LLMs to reason over the code required to solve a problem without seeing a single line. It constructs a repo-wide AST (abstract syntax tree) + dependency graph, stored locally in a sqlite database with struct-level, context-aware descriptions and vector embeddings for efficient traversal. By exposing this graph to an LLM via MCP, token input costs during agentic programming can be reduced by upwards of 70%. Unique to tostr in the code graph landscape is live, efficient subgraph reparsing using a file watcher attached to the mcp server's lifetime, allowing the graph to evolve and progress dynamically as the developer and agent make changes. The graph will **always** represent the project as accurately as an initial parse within seconds of agentic or human changes.

<!-- ## 2. High-Level Pipeline -->
<!-- parse → resolve dependencies → carry over cache → seed lockfile → des
cribe → embed → save.
     A diagram would go here. -->

## High Level Module Heirarchy

### Primary Dataflow Layers
* **Entrypoints** (`cli.py` / `server.py`): Expose the user interaction surface via FastMCP and a Typer CLI
* **Orchestration** (`commands.py`, `core/parser.py`): Orchestrate the dataflow for the entrypoints and handle dependency injection
* **AST Parsing** (`core/builders.py`, `languages/*`): Constructing the AST in-memory using tree-sitter
* **Dependency Resolution** (`core/resolver.py`): Deterministically building a project-wide dependency graph
* **Describing** (`core/describer.py`): Handles the description process, traversing the AST using a visitor pattern in a post-order DFS
* **Embedding** (`semantic/embeddings/*`): Handles the asyncronous enqueuing and execution of the description-based vector embeddings

### Helper Modules
* **Memory Store** (`core/registry.py`, class `Registry`): The in-memory representation of the project AST **and the object graph's context handle**. Owns the `uid_map`/`id_map` indexes + `root`, hands out per-language resolvers (`get_resolver`), and carries the ambient `config`/`progress_tracker`/path helpers that every struct, builder, and resolver reaches through. **Pure memory — no SQL, no `db`.** (Kept the name `Registry`; it is the store.)
* **Struct Cache** (`core/cache.py`, class `StructCache`): The persistence layer. Holds a `Registry` as its `struct_store` and a `SqliteClient`, and owns hydration (`load_filepath`, `get_struct_by_*`), write-back (`save_to_cache`, `delete_path_subtree`), and reconciliation (`carry_over_unchanged`, `collect_descriptions`, `load_lockfile_lookup`).
* **SQLite Client** (`core/db.py`, class `SqliteClient`): The **only** place that defines and executes raw SQL (schema DDL + struct/edge/vector queries). Imported only by the cache layer and the DI root.
* **Project Paths / Config** (`core/paths.py`, `core/context/config.py`): `ProjectPaths` is the single source of truth for on-disk locations (`db_path`, `lock_path`, `relative_to_project`); `ProjectConfig` owns `tostr.toml`/`.tostrignore` settings.
* **Struct Models** (`core/models.py`): Hierarchal OOP @dataclass implementations for representing language-agnostic AST nodes (Directory, File, Class, Method, Field)
* **LLM Strategy Pattern** (`semantic/llm/*`): A generic LLMClient handling retries and async llm calls, with a polymorphised strategy pattern for different llm API bindings
* **Embedder Strategy Pattern** (`semantic/embeddings/*`): A generic EmbeddingClient handling async queue management, with a polymorphised strategy pattern for different embedding model bindings
<!-- progress tracker, db client, exceptions, serializer, lockfile, logger, config, providers, tests -->

## Store / Cache separation & dependency injection

Memory and persistence are split by **audience**, and the two objects are built at the DI root
(`commands.py` entrypoints / `core/parser.py`) then passed only where needed:

* The **object graph** (structs, builders, resolvers, describer) only ever needs the in-memory
  store, so it holds a `Registry` (`struct.registry`, `builder.registry`, `resolver`). It never
  touches the database.
* The **orchestration layer** (commands, parser) is the only caller of persistence, so it holds a
  `StructCache`. `StructCache` wraps a `Registry` as its `struct_store`, so hydration populates the
  same in-memory maps the object graph reads.
* Raw SQL is confined to the cache layer: `SqliteClient` (`core/db.py`) defines/executes it, and
  `StructCache` orchestrates. `Registry`, `commands`, `server` never open a connection (read-only
  query commands like `status`/`search` call `SqliteClient` query methods directly at the DI root).

There is deliberately **no wrapper** binding store + cache together — nothing needs both as one
handle, so each audience talks to its own object. This is why `Registry` carries no persistence.

## Struct identity & dependency resolution

* **UIDs are normalized, path-based, and the single identity key.** Format is
  `relative/path.ext#Symbol` (dots only *within* a symbol for nesting, e.g. `pkg/mod.py#Cls.method(...)`).
  Raw source identifier formats (dotted module/package names, Java FQNs) are **never** stored as
  resolution keys — the builders normalize them during the parse pass.
* **`id`** is a short typed hash of the UID (`C-…` class, `M-…` method, `F-…` file, `D-…` dir,
  `V-…` field); it is the opaque surrogate key handed to `inspect` and used for graph edges.
* **Imports are normalized to UID *candidates*** at build time (the target AST may be incomplete, so
  a name emits e.g. `models.py#User` + `models.py#User(...)` and non-existent candidates are dropped
  at resolution). Wildcards become a `scope.*` marker.
* **Absolute imports are anchored to the source root** at build time: the importing file's own
  project-relative path locates the import's first segment (file `src/pkg/cmd.py` importing `pkg.a`
  → prefix `src`), so candidates are the exact UID `src/pkg/a.py#A`. `Registry.resolve_import` then
  resolves a candidate by: exact match → **path-suffix** match (cross-package / unanchored) →
  **unique-name** fallback (package `__init__` re-exports, e.g. `from tostr.core import BaseParser`
  where the class lives in `parser.py`). Ambiguous names are left unresolved rather than guessed.
* **Dependency resolution is struct-graph traversal**, not string matching: look the candidate up in
  the store, else walk the enclosing scope / imported scope structs by name. There is no dotted
  logical-name translation layer.
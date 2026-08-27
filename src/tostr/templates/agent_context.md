# Code navigation: prefer Tostr over raw file/text tools

**Tostr** is an MCP code-context engine. When a project is parsed, Tostr exposes its
entire structure as a pre-computed, LLM-described AST + dependency graph with semantic
vector search, all served from a local cache. Traversing code through Tostr instead of
reading raw files is **cheaper** (it returns signatures, descriptions, and dependency
edges — not whole files), gives **better attention** (you reason over the structures
that matter rather than scrolling bytes), and surfaces **dependency awareness** that
grep/read simply cannot provide.

**Default to Tostr's `skeleton` / `search` / `inspect` for understanding and navigating
code.** Reach for `read_file`, `grep`, `find`, `ls`, and `cat` only for the cases listed
under "Keep using native tools" below.

## When this applies (gating)

Use Tostr when **all** of the following hold:
- The task involves **understanding, navigating, or tracing** code (architecture, "where
  is X", "how does Y work", "what calls Z", finding logic by meaning).
- The code is in a **Tostr-supported language with a dependency graph**: **Java** and
  **Python**. For other languages Tostr still provides structural skeletons and semantic
  descriptions (use it), but dependency fields are cleanly omitted — lean on `skeleton`
  and `search` and don't expect inbound/outbound edges.
- A Tostr MCP toolset is connected for the session.

If the language is unsupported (and you only need structure Tostr can't give better),
or the task is not about code comprehension, just use native tools.

## Workflow

1. **Ensure the project is parsed (once per workspace per session).** Before the first
   Tostr query against a workspace, call `parse` with the **absolute** `workspace_path`.
   It is idempotent and fast when a cache already exists, and it starts a file watcher
   that keeps the graph in sync with edits — so you do **not** need to re-parse after
   making changes. If a `skeleton`/`search`/`inspect` call errors because nothing is
   cached, parse, then retry.
2. **Get the lay of the land** with `skeleton` (start shallow, e.g. `depth: 1`, then
   drill into a subpath). Use this before listing directories or reading files to learn
   the architecture, classes, and signatures.
3. **Find code by meaning** with `search` — semantic vector search over descriptions.
   Prefer it over `grep` whenever you're looking for *functionality or logic* ("retry
   backoff", "PID controller", "where auth tokens are validated") rather than an exact
   literal string.
4. **Read implementation + dependencies** with `inspect` (by id or uid). Request the
   body only when you actually need the source. Inspect output encodes the dependency
   graph: `>` outbound (this struct depends on the listed id), `<` inbound (the listed
   id uses this struct), `~` related/sibling, `//` AI/docstring summary, `#` a
   human-authored note. Use these edges to trace call paths instead of grepping for
   identifiers across the tree.
5. **Record what the code can't say** with `note_add` — durable commentary pinned to a
   struct, shown at the top of every later `inspect` and preserved across reparses. Use
   it for non-obvious invariants, rejected approaches, known bugs, or the reasoning
   behind a decision; not for restating what the code or its `//` summary already says.
   `note_edit` / `note_remove` maintain them.

Typical loop: `skeleton` (orient) → `search` (locate) → `inspect` (understand + follow
edges) → repeat as you walk the graph.

## Native tool → Tostr replacement

| Reaching for…                          | Use instead                                  |
| -------------------------------------- | -------------------------------------------- |
| `ls`, `find`, `tree`, directory walk   | `skeleton` (`files_only: true` for paths)    |
| `grep`/ripgrep for *logic/behavior*    | `search` (semantic)                          |
| `cat`/`read_file` to understand a unit | `inspect` (by id or uid)                     |
| Hand-tracing who-calls-what            | `inspect` dependency edges (`>` `<` `~`)     |

## Keep using native tools for

- **Editing/writing files** — Tostr is read/understanding only; make changes with your
  normal editing tools. The watcher updates the graph automatically afterward.
- **Exact-string / regex / literal matches** — finding a specific error string, a config
  key, a TODO, a version pin: `grep` is the right tool.
- **Non-code or unparsed files** — config, markdown, JSON/YAML, logs, lockfiles, build
  output, generated assets.
- **Anything outside Tostr's structural model** — git operations, running tests/builds,
  reading command output, exploring a brand-new repo before it's parsed.

## Operational notes

- **Always pass an absolute `workspace_path`** to every Tostr call. Never `.` or a
  relative path; resolve the current workspace's absolute path first if needed.
- IDs look like `C-c7766e98fa` (class), `M-bc1cb7aeff` (method), etc.; carry the ids you
  get from `skeleton`/`search` straight into `inspect`.
- Tostr requires a parsed cache. Don't run `clean`/`init` as part of normal navigation —
  those are project-management actions, not traversal steps.

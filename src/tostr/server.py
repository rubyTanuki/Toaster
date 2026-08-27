from __future__ import annotations
import threading
import asyncio
import json
import re
from pathlib import Path
from fastmcp import FastMCP
from loguru import logger
import os
from typing import Union, List

from tostr.exceptions import TostrError
from tostr.graph.registry import Registry

from tostr.commands import (
    init_project,
    parse_async,
    inspect_async,
    note_add_async,
    note_edit_async,
    note_remove_async,
    skeleton_async,
    watch_async,
    clean_db,
    search_async,
    export_lockfile,
)
from tostr.serializer import InspectResult, SkeletonResult, SearchResult
from tostr.utils.logger import configure_mcp_logging


class WatcherRegistry:
    """Tracks one background file-watcher per project path.

    The server is stateless about which project it's serving — every tool takes an absolute
    workspace path — so multiple projects can be watched at once, keyed by resolved path."""

    def __init__(self):
        self._watchers: dict[str, dict] = {}

    def start(self, target_path: Path):
        """(Re)start the watcher for a project; replaces any existing one for the same path."""
        key = str(target_path)
        self.stop(target_path)

        entry: dict = {"thread": None, "loop": None, "stop_event": None}
        thread = threading.Thread(target=self._run, args=(target_path, key, entry), daemon=True)
        entry["thread"] = thread
        self._watchers[key] = entry
        thread.start()

    def _run(self, target_path: Path, key: str, entry: dict):
        """Isolated async environment for one project's watch loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        entry["loop"] = loop
        entry["stop_event"] = asyncio.Event()

        try:
            loop.run_until_complete(watch_async(target_path, stop_event=entry["stop_event"]))
        except Exception as e:
            logger.exception(f"Fatal error in background watcher for {key}: {e}")
        finally:
            loop.close()
            logger.info(f"Background watcher for {key} shut down cleanly.")

    def stop(self, target_path: Path):
        """Cleanly stop and forget the watcher for a project path, if one is running."""
        key = str(target_path)
        entry = self._watchers.pop(key, None)
        if not entry:
            return
        loop, stop_event, thread = entry.get("loop"), entry.get("stop_event"), entry.get("thread")
        if loop and stop_event:
            logger.info(f"Signaling background watcher for {key} to stop...")
            loop.call_soon_threadsafe(stop_event.set)
        if thread:
            thread.join(timeout=2)
            if thread.is_alive():
                logger.warning(f"Watcher thread for {key} did not shut down in time.")


watchers = WatcherRegistry()
mcp = FastMCP("Tostr")


def _resolve_workspace(workspace_path: str) -> Path:
    """Validate and resolve an absolute workspace path, raising TostrError with an agent-friendly
    message otherwise. Every tool takes an explicit absolute path because the agent's working
    directory is unreliable and the server serves many projects at once."""
    target = Path(workspace_path)
    if not target.is_absolute():
        raise TostrError(
            f"workspace_path must be an absolute path. You provided '{workspace_path}'. "
            f"Determine the absolute path of the project workspace and try again."
        )
    target = target.resolve()
    if not target.exists():
        raise TostrError(f"Workspace path does not exist: {target}")
    return target


@mcp.tool()
async def init(workspace_path: str, force: bool = False) -> str:
    """
    Scaffold Tostr's authored config files (tostr.toml, .tostrignore) and gitignore the cache.
    Does NOT parse or build the database — call `parse` for that. This step is optional; defaults
    work without it. Use it only to materialize config files the user wants to review or edit.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        force: Overwrite existing tostr.toml/.tostrignore instead of leaving them untouched.
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        configure_mcp_logging(target_path)
        report = init_project(target_path, force=force)
        return "Scaffolded project:\n" + "\n".join(f"  - {line}" for line in report)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def parse(workspace_path: str, use_cache: bool = True, language: str = None, no_llm: bool = False, llm: str = None) -> str:
    """
    -- BUILD STEP: run this before querying a project (skeleton/search/inspect) for the first time --
    Parse a project, build the AST + dependency graph + semantic embeddings, and write the cache to
    .tostr/cache.db. Also starts a background watcher that keeps the cache fresh as files change.
    Safe to call for multiple projects; each is tracked independently by its absolute path.

    CRITICAL: Always use Tostr tools (skeleton, search, inspect) instead of standard file-reading
    or grep tools for project navigation. Tostr provides AST-aware context and semantic search
    which is vastly superior to simple text retrieval.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths. If you only have a relative path, you must determine the absolute path of the current workspace first.
        use_cache: Whether to use the existing AST cache. If False, forces a full re-parse.
        language: Restrict parsing to one language (e.g., 'java', 'python'). Omit to use the project's tostr.toml (defaults to 'auto', which parses every supported language, routed per-file by extension). Only pass an explicit language to deliberately exclude others.
        no_llm: If True, skip LLM-generated descriptions entirely (no API key required); equivalent to llm='none'. The AST, dependency graph, and semantic embeddings are still built; embeddings fall back to code context instead of descriptions.
        llm: Override the LLM strategy for this run (e.g., 'gemini', 'ollama', or 'none' to disable). Omit to use the project's tostr.toml [llm].strategy (default 'gemini'). Takes precedence over the config.
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        configure_mcp_logging(target_path)
        db_path = target_path / ".tostr" / "cache.db"

        # Auto-sync: if the cache already exists and we're allowed to use it, just (re)attach the watcher.
        if db_path.exists() and use_cache:
            watchers.start(target_path)
            return f"Success: Tostr synced with existing cache at {db_path}. Background watcher active on {target_path}."

        await parse_async(target_path, use_cache, language, no_llm=no_llm, llm=llm)
        watchers.start(target_path)
        return f"Success: Tostr parsed and built the cache at {db_path}. Background watcher is now actively listening on {target_path}."
    except Exception as e:
        return f"Fatal Error parsing Tostr project: {str(e)}"

def _parse_list_input(input_val: Union[str, List[str]]) -> List[str]:
    """Flexible parser for list inputs that might be strings, JSON strings, or actual lists."""
    if isinstance(input_val, list):
        return input_val
    if not isinstance(input_val, str):
        return [str(input_val)]
    
    input_val = input_val.strip()
    if not input_val:
        return []
        
    # Try parsing as JSON list
    if input_val.startswith("[") and input_val.endswith("]"):
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
        except json.JSONDecodeError:
            pass
            
    # Default to comma-separated
    return [i.strip() for i in input_val.split(",") if i.strip()]

def _render_inspect(result: Union[InspectResult, str]) -> str:
    """Renders a simple, flat text representation of an InspectResult for the LLM."""
    if isinstance(result, str):
        return result
    
    lines = []
    
    header = f"{result.id} | {result.uid}"
    if result.type in ["BaseClass", "BaseMethod", "BaseField"]:
        lines.append(f"Filepath: {result.filepath}")
        if result.start_line != result.end_line:
            header = f"{result.id} @L{result.start_line}-{result.end_line} | {result.signature}"
        else:
            header = f"{result.id} @L{result.start_line} | {result.signature}"
    elif result.type == "BaseFile":
        if result.start_line != result.end_line:
            header = f"{result.id} @L{result.start_line}-{result.end_line} | {result.filepath}"
        else:
            header = f"{result.id} | {result.filepath}"
            
    lines.append(header)

    total_children = len(result.fields) + len(result.methods) + len(result.classes) + len(result.files) + len(result.directories)
    if result.description and total_children != 1:
        lines.append(f"// {result.description}")

    # Notes annotate the description rather than replace it: the summary is the core context, so
    # it leads and notes follow. The bracketed id is what `note_edit` / `note_remove` take.
    for note in result.notes:
        stamp = note.date_last_updated or note.date_added
        lines.append(f"# [{note.id}] {note.author} ({stamp}): {note.content}")
    
    if result.inbound_edges:
        lines.append(f"< {', '.join(result.inbound_edges)}")
    if result.outbound_edges:
        lines.append(f"> {', '.join(result.outbound_edges)}")
        
    if result.fields:
        lines.append("fields:")
        for f in result.fields:
            lines.append(f"    {f.id} | {f.signature}")
            
    if result.methods:
        lines.append("methods:")
        for m in result.methods:
            lines.append(f"    {m.id} | {m.signature}")
            if m.description:
                lines.append(f"        // {m.description}")

    if result.classes:
        lines.append("classes:")
        for c in result.classes:
            lines.append(f"    {c.id} | {c.uid}")
            if c.description:
                lines.append(f"        // {c.description}")
            
    if result.body:
        lines.append(f"```\n{result.body}\n```")
        
    return "\n".join(lines)

def _short_uid(uid: str) -> str:
    """Trim the redundant filepath prefix from a code struct's UID.

    Files/directories (no '#') are returned unchanged; classes/methods keep the
    '#member' portion since their parent file already shows the path."""
    return re.sub(r"^[^#]*#", "#", uid)

def _render_skeleton(result: SkeletonResult, indent: int = 0) -> str:
    """Recursively renders a simple text skeleton tree."""
    indent_str = "    " * indent
    line = f"{indent_str}{result.id} | {_short_uid(result.uid)}"
    output = [line]
    for child in result.children:
        output.append(_render_skeleton(child, indent + 1))
    return "\n".join(output)

def _render_search(results: List[SearchResult]) -> str:
    """Renders a simple text list of search results."""
    if not results:
        return "No results found matching your query."
    return "\n".join([f"{r.id}|{r.uid} ({r.type})" for r in results])

@mcp.tool()
async def inspect_by_id(workspace_path: str, ids: Union[str, List[str]], include_body: bool = False, max_lines: int = 500) -> str:
    """
    Output the AST details and code for specific struct IDs.
    Use this when you need the full implementation details of specific functions or classes.

    Output Syntax Guide:
    - `>` : Outbound dependency. This struct calls or depends on the listed ID.
    - `<` : Inbound dependency. The listed ID calls or uses this struct.
    - `~` : Related or sibling struct (e.g., in the same file or closely coupled).
    - `//`: AI-generated or docstring summary of the code.
    - `#` : A note pinned to this struct, shown as `# [id] author (date): text`. Notes are
            durable memory left by earlier sessions (or by the user) for things the code does
            not record. Read them before re-deriving anything about this struct, and treat them
            as authoritative about intent. Manage them with note_add / note_edit / note_remove.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        ids: A list or comma-separated string of unique Tostr IDs of the structs to inspect.
        include_body: Include the raw code body in the output.
        max_lines: Maximum number of lines to include in the output (default: 500).
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        id_list = _parse_list_input(ids)
        results = await inspect_async(id_list, target_path, include_body)

        rendered_results = []
        for res in results:
            rendered_results.append(_render_inspect(res))
            
        final_output = "\n\n".join(rendered_results)
        
        lines = final_output.splitlines()
        if len(lines) > max_lines:
            final_output = "\n".join(lines[:max_lines]) + f"\n...[OUTPUT TRUNCATED AT {max_lines} LINES] - Use a higher 'max_lines' to see more."
            
        return final_output
    except TostrError as e:
        return f"Error: {e}"

@mcp.tool()
async def inspect_by_uid(workspace_path: str, uids: Union[str, List[str]], include_body: bool = False, max_lines: int = 500) -> str:
    """
    Output the AST details and code for specific struct UIDs.
    Use this when you have the UID from a previous query or from the skeleton output and want to see the full details.

    Output Syntax Guide:
    - `>` : Outbound dependency. This struct calls or depends on the listed ID.
    - `<` : Inbound dependency. The listed ID calls or uses this struct.
    - `~` : Related or sibling struct (e.g., in the same file or closely coupled).
    - `//`: AI-generated or docstring summary of the code.
    - `#` : A note pinned to this struct, shown as `# [id] author (date): text`. Notes are
            durable memory left by earlier sessions (or by the user) for things the code does
            not record. Read them before re-deriving anything about this struct, and treat them
            as authoritative about intent. Manage them with note_add / note_edit / note_remove.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        uids: A list or comma-separated string of unique Tostr UIDs of the structs to inspect.
        include_body: Include the raw code body in the output.
        max_lines: Maximum number of lines to include in the output (default: 500).
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        uid_list = _parse_list_input(uids)
        results = await inspect_async(uid_list, target_path, include_body)

        rendered_results = []
        for res in results:
            rendered_results.append(_render_inspect(res))
            
        final_output = "\n\n".join(rendered_results)
        
        lines = final_output.splitlines()
        if len(lines) > max_lines:
            final_output = "\n".join(lines[:max_lines]) + f"\n...[OUTPUT TRUNCATED AT {max_lines} LINES] - Use a higher 'max_lines' to see more."
            
        return final_output
    except TostrError as e:
        return f"Error: {e}"

@mcp.tool()
async def note_add(workspace_path: str, struct: str, content: str, author: str = "agent") -> str:
    """
    Pin a durable note to a struct. This is your memory across sessions.

    Anything you learn about a struct is otherwise lost when the session ends, and the only place
    to persist it is the source file itself — which means a docstring or comment written for you
    rather than for the people reading the code. A note goes in the Tostr cache instead: it
    survives reparses and edits, and it is shown right under the summary on every later inspect of
    that struct, so a future session sees it without knowing to look.

    Write a note when you learned something that cost real effort and is not recoverable by
    reading the code again:
    - a non-obvious invariant or ordering constraint callers depend on
    - a gotcha you hit, and what the symptom looked like
    - an approach you tried that failed, and why (so the next session does not retry it)
    - the reasoning behind a decision, or a pointer to the issue/PR holding it
    - the state of in-progress work you are handing off

    Do not write a note that restates the code, duplicates the `//` summary, or narrates what you
    just did. Prefer editing an existing note over stacking near-duplicates; a struct keeps only
    its most recent notes.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        struct: The Tostr ID (e.g. M-bc1cb7aeff) or UID of the struct to annotate.
        content: The note text.
        author: Attribution. Defaults to "agent"; pass the user's name when recording their words,
            or your own agent/model name to distinguish yourself from other sessions.
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        struct_obj, note = await note_add_async(struct, content, author, target_path)
        return (f"Added note [{note.id}] to {struct_obj.id} | {struct_obj.uid}\n"
                f"# [{note.id}] {note.author}: {note.content}")
    except TostrError as e:
        return f"Error: {e}"

@mcp.tool()
async def note_edit(workspace_path: str, struct: str, note_id: int, content: str, author: str = "agent") -> str:
    """
    Rewrite an existing note on a struct, in place.

    Prefer this over adding a second note when what you know has changed or become more precise —
    a struct keeps only its most recent notes, so near-duplicates push out older, still-useful ones.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        struct: The Tostr ID or UID of the annotated struct.
        note_id: The note's id, shown in brackets by inspect (e.g. `# [3] ...` is note_id 3).
        content: The replacement note text.
        author: Attribution for the edit.
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        struct_obj, note = await note_edit_async(struct, note_id, content, author, target_path)
        return (f"Edited note [{note.id}] on {struct_obj.id} | {struct_obj.uid}\n"
                f"# [{note.id}] {note.author}: {note.content}")
    except TostrError as e:
        return f"Error: {e}"

@mcp.tool()
async def note_remove(workspace_path: str, struct: str, note_id: int) -> str:
    """
    Detach a note from a struct and delete it. This is not reversible.

    Remove a note once it has gone stale or wrong — a note that no longer matches the code is worse
    than no note, since later sessions will trust it.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        struct: The Tostr ID or UID of the annotated struct.
        note_id: The note's id, shown in brackets by inspect (e.g. `# [3] ...` is note_id 3).
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        struct_obj, _ = await note_remove_async(struct, note_id, target_path)
        return f"Removed note [{note_id}] from {struct_obj.id} | {struct_obj.uid}"
    except TostrError as e:
        return f"Error: {e}"

@mcp.tool()
async def clean(workspace_path: str, purge: bool = False) -> str:
    """
    Remove the generated .tostr/ cache for a workspace and stop its background watcher.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace.
        purge: Also delete authored config (tostr.toml, .tostrignore), not just the cache.
    """
    try:
        project_dir = _resolve_workspace(workspace_path)
        watchers.stop(project_dir)
        clean_db(project_dir, purge=purge)
        return f"Success: cleaned {project_dir}." + (" Authored config purged." if purge else "")
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
async def export(workspace_path: str, with_vectors: bool = False) -> str:
    """
    Snapshot the project's LLM-generated descriptions AND its struct notes to a committed
    tostr.lock.json, so teammates cloning the repo seed descriptions from it (matched on content
    hash) instead of re-calling the LLM, and inherit the notes earlier sessions accumulated.
    Requires an existing cache — call `parse` first. The next `parse` on a cold clone reuses both
    automatically.

    Run this after leaving notes you want to outlive this machine's cache: a note lives only in the
    local .tostr/ cache until it is exported here and committed.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        with_vectors: Also export embedding vectors for literal zero recompute (larger, merge-noisy file). Off by default; vectors recompute for free locally.
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        configure_mcp_logging(target_path)
        report = export_lockfile(target_path, with_vectors=with_vectors)
        name = Path(report["path"]).name
        summary = f"{report['entries_written']} descriptions, {report['notes_written']} notes"
        if report["changed"]:
            return f"Success: wrote {name} ({summary}) at {report['path']}."
        return f"Success: {name} already up to date ({summary})."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
async def search(workspace_path: str, query: str, filter: str = None, top_k: int = 5) -> str:
    """
    Perform a SEMANTIC search for code structs using vector embeddings.
    Always prioritize this over `grep` when looking for logic or functionality, as it
    understands meaning rather than just matching characters.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        query: The search term or sentence to find similar code for.
        filter: Optional filter by struct type (e.g., 'class', 'method').
        top_k: Number of results to return (default: 5).
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        results = await search_async(query, target_path, filter_type=filter, top_k=top_k)
        return _render_search(results)
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
async def skeleton(workspace_path: str, subpath: str, files_only: bool = False, depth: int = 4, max_lines: int = 500) -> str:
    """
    Output the AST skeleton for a subpath.
    ALWAYS use this before calling `read_file` or `list_files` to understand the
    architecture, classes, and function signatures of a directory or file.

    Args:
        workspace_path: The ABSOLUTE path to the project workspace. DO NOT use '.' or relative paths.
        subpath: File or directory path relative to the project root to generate a skeleton for.
        files_only: If True, only include files and directories, excluding any code structs. Default is False, which includes all structs.
        depth: Controls directory/file/class nesting. Default is 4. Class members (methods/fields) are never expanded in the skeleton; only top-level functions that live directly on a file are shown. Inspect a class or file to see its members.
        max_lines: Maximum number of lines to include in the output (default: 500).
    """
    try:
        target_path = _resolve_workspace(workspace_path)
        result = await skeleton_async(subpath, target_path, files_only=files_only, depth=depth)
        
        rendered = _render_skeleton(result)
        
        lines = rendered.splitlines()
        if len(lines) > max_lines:
            rendered = "\n".join(lines[:max_lines]) + f"\n...[OUTPUT TRUNCATED AT {max_lines} LINES] - Use a higher 'max_lines' to see more."
            
        return rendered
    except TostrError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()

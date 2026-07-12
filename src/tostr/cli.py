from __future__ import annotations
import asyncio
import re
import time
from pathlib import Path
import typer
from typing import Annotated, List, Union
from loguru import logger
from tostr.exceptions import TostrError

from rich.console import Console
from rich.theme import Theme
from rich.highlighter import RegexHighlighter

from tostr.commands import (
    init_project,
    parse_async,
    inspect_async,
    skeleton_async,
    watch_async,
    clean_db,
    search_async,
    get_status,
    export_lockfile,
)
from tostr.agents import add_agent, remove_agent, list_agents, PROFILES

from tostr.server import mcp

from tostr.utils import configure_cli_logging, ProgressTracker
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

import multiprocessing

from tostr.serializer import InspectResult, SkeletonResult, SearchResult
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree

# Initialize the Typer app
app = typer.Typer(
    name="tostr",
    help="AST scraper for LLM RAG context generation.",
    add_completion=False # Optional: Turns off the auto-generated completion install command for cleaner help menus
)

class TostrHighlighter(RegexHighlighter):
    """Applies beautiful visual formatting to the custom .tost human layout."""
    base_style = "tostr."
    highlights = [
        # Capture UIDs (e.g., C-5714b4c321, M-d3f989c0fe, F-13cedbe53e)
        r"(?P<uid>[A-Z]-[0-9a-f]{10})",
        # Capture multi-line descriptions
        r"(?P<comment>//.*(?:\n[^\S\n\r]+(?![A-Z]-[0-9a-f]{10}|[<>\~]|fields:|methods:).*)*)",
        # Capture structural line numbers matching both range styles (@L19-311) and single lines (@L211)
        r"(?P<line_num>@L\d+(?:-\d+)?)",
        # Capture dependency symbols (<, >, ~)
        r"(?P<edge>[<>\~])",
        # Capture structural sections blocks headers (fields:, methods:)
        r"(?P<section>fields:|methods:)"
    ]

# Define an immersive terminal theme layout
theme = Theme({
    "tostr.uid": "bold cyan",
    "tostr.comment": "dim italic white",
    "tostr.line_num": "green",
    "tostr.edge": "bold magenta",
    "tostr.section": "bold yellow",
})

console = Console(highlighter=TostrHighlighter(), theme=theme)

def _run_watcher_thread(target_path: Path):
    """
    Sets up an isolated async environment for the background thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        logger.info(f"Background watcher started on {target_path}")
        loop.run_until_complete(watch_async(target_path))
    except Exception as e:
        logger.exception(f"Fatal error in background watcher: {e}")
    finally:
        loop.close()
        logger.info("Background watcher shut down.")

@app.command("start-mcp")
def start_mcp():
    """Start the bare MCP server. Awaits agent initialization."""
    mcp.run()

@app.command()
def status(
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory to check",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    debug: Annotated[
        bool, 
        typer.Option(
            "--debug/--no-debug", 
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Show the status of Tostr in the given path."""
    configure_cli_logging(debug)
    import datetime
    
    try:
        status_data = get_status(path)
        
        typer.secho(f"\n📊 Tostr Status for: {status_data['project_path']}", fg="cyan", bold=True)
        
        if status_data["db_exists"]:
            typer.secho("✅ Database: Found", fg="green")
            typer.echo(f"   Path: {status_data['db_path']}")
            
            size_mb = status_data['db_size_bytes'] / (1024 * 1024)
            typer.echo(f"   Size: {size_mb:.2f} MB")
            
            last_updated = datetime.datetime.fromtimestamp(status_data['last_updated']).strftime('%Y-%m-%d %H:%M:%S')
            typer.echo(f"   Last Updated: {last_updated}")
            
            typer.secho("\n📈 Statistics:", bold=True)
            for type_name, count in status_data["counts"].items():
                typer.echo(f"   {type_name}: {count}")
        else:
            typer.secho("❌ Database: Not found", fg="red")
            typer.echo("   Run 'tostr parse' to build the database.")
        typer.echo("")
        
    except Exception as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

@app.command()
def watch(
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory to scan",
        exists=True,       # Typer automatically checks if the path exists
        file_okay=False,   # Typer blocks files, only allowing directories
        dir_okay=True,
        resolve_path=True  # Converts relative paths to absolute paths automatically
    ),
    debug: Annotated[
        bool, 
        typer.Option(
            "--debug/--no-debug", 
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Watch for changes to files and update the SQLite database."""
    configure_cli_logging(debug)
    try:
        asyncio.run(watch_async(path))
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

@app.command()
def clean(
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory to scan",
        exists=True,       # Typer automatically checks if the path exists
        file_okay=False,   # Typer blocks files, only allowing directories
        dir_okay=True,
        resolve_path=True  # Converts relative paths to absolute paths automatically
    ),
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="Also delete authored config (tostr.toml, .tostrignore), not just the generated .tostr/ cache."
        )
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Remove the generated .tostr/ cache (use --purge to also delete authored config)."""
    configure_cli_logging(debug)
    try:
        clean_db(path, purge=purge)
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

@app.command()
def export(
    path: Path = typer.Argument(
        ".",
        help="Path to the project directory to export",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    with_vectors: Annotated[
        bool,
        typer.Option(
            "--with-vectors",
            help="Also export embedding vectors (literal zero recompute, but a larger, merge-noisy file). Off by default — vectors recompute for free from the local model."
        )
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Snapshot descriptions to tostr.lock.json for version control, so teammates can seed them on a cold clone instead of re-calling the LLM. Requires an existing cache (run 'tostr parse' first)."""
    configure_cli_logging(debug)
    try:
        report = export_lockfile(path, with_vectors=with_vectors)
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

    name = Path(report["path"]).name
    if report["changed"]:
        typer.secho(f"✅ Wrote {name} ({report['entries_written']} descriptions)", fg="green")
    else:
        typer.secho(f"{name} already up to date", fg="yellow")


@app.command()
def init(
    path: Path = typer.Argument(
        ".",
        help="Path to the project directory to scaffold",
        exists=True,       # Typer automatically checks if the path exists
        file_okay=False,   # Typer blocks files, only allowing directories
        dir_okay=True,
        resolve_path=True  # Converts relative paths to absolute paths automatically
    ),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing authored files (tostr.toml, .tostrignore) instead of leaving them untouched."
        )
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Scaffold project files (tostr.toml, .tostrignore, .gitignore). Does not parse — run 'tostr parse' to build the cache."""
    configure_cli_logging(debug)
    try:
        report = init_project(path, force=force)
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

    for line in report:
        typer.echo(f"   {line}")
    typer.secho("✅ Scaffolded project. Edit tostr.toml / .tostrignore, then run 'tostr parse'.", fg="green")


@app.command("add-agent")
def add_agent_cmd(
    agent: Annotated[
        Union[str, None],
        typer.Argument(
            help="Agent to configure (e.g. claude, cline, cursor, copilot, codex), or 'all'."
        )
    ] = None,
    path: Path = typer.Argument(
        ".",
        help="Project directory to install into (ignored with --global).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    global_: Annotated[
        bool,
        typer.Option("--global", "-g", help="Install into the agent's global config instead of this project.")
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite a dedicated agent file even if it isn't Tostr-managed.")
    ] = False,
    list_: Annotated[
        bool,
        typer.Option("--list", "-l", help="List supported agents and where they install.")
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug/--no-debug", "-d/-nd", help="Enable debug logging")
    ] = False,
):
    """Install Tostr's 'prefer Tostr for code navigation' guidance into an agent's config
    (CLAUDE.md, .clinerules, etc). Safe to re-run: it upserts a managed block and never
    clobbers your own content."""
    configure_cli_logging(debug)

    if list_:
        typer.secho("Supported agents:", bold=True)
        for line in list_agents():
            typer.echo(f"   {line}")
        return
    if agent is None:
        typer.secho("❌ Error: specify an agent (or 'all'), or use --list.", fg="red", err=True)
        raise typer.Exit(code=1)

    scope = "global" if global_ else "project"
    targets = list(PROFILES) if agent.lower() == "all" else [agent]
    try:
        report = [line for a in targets for line in add_agent(a, path, scope=scope, force=force)]
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

    for line in report:
        typer.echo(f"   {line}")
    typer.secho("✅ Agent config installed.", fg="green")


@app.command("remove-agent")
def remove_agent_cmd(
    agent: Annotated[
        str,
        typer.Argument(help="Agent to remove (e.g. claude, cline, cursor, copilot, codex), or 'all'.")
    ],
    path: Path = typer.Argument(
        ".",
        help="Project directory to remove from (ignored with --global).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    global_: Annotated[
        bool,
        typer.Option("--global", "-g", help="Remove from the agent's global config instead of this project.")
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug/--no-debug", "-d/-nd", help="Enable debug logging")
    ] = False,
):
    """Remove Tostr's guidance from an agent's config — strips the managed block (keeping
    your own content) or deletes the dedicated file."""
    configure_cli_logging(debug)

    scope = "global" if global_ else "project"
    targets = list(PROFILES) if agent.lower() == "all" else [agent]
    try:
        report = [line for a in targets for line in remove_agent(a, path, scope=scope)]
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

    for line in report:
        typer.echo(f"   {line}")
    typer.secho("✅ Done.", fg="green")


@app.command()
def parse(
    path: Path = typer.Argument(
        ".",
        help="Path to the project directory to scan",
        exists=True,       # Typer automatically checks if the path exists
        file_okay=False,   # Typer blocks files, only allowing directories
        dir_okay=True,
        resolve_path=True  # Converts relative paths to absolute paths automatically
    ),
    use_cache: Annotated[
        bool,
        typer.Option(
            "--use-cache/--no-cache",
            help="Load cache if it exists"
            )
        ] = True,
    language: Annotated[
        str,
        typer.Option(
            "--language",
            "-l",
            help="Override the configured language for this run (e.g., 'java', 'python'). Omit to use tostr.toml (defaults to 'auto', which parses all supported languages by extension)."
        )
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Skip LLM-generated descriptions (no API key required); equivalent to --llm none. Embeddings still run, falling back to code context."
        )
    ] = False,
    llm: Annotated[
        str,
        typer.Option(
            "--llm",
            help="Override the LLM strategy for this run (e.g., 'gemini', 'ollama', or 'none' to disable). Trumps tostr.toml [llm].strategy. Resolution: --llm > tostr.toml > gemini default."
        )
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Parse files and build the SQLite database from configuration."""
    configure_cli_logging(debug)
    start_time = time.perf_counter()

    embedding_model_path = Path.home() / ".cache" / "tostr" / "models" / "all-MiniLM-L6-v2" / "model.onnx"
    if not embedding_model_path.exists():
        typer.echo(f"Embedding model not found at {embedding_model_path}. Downloading from huggingface...")

    typer.echo(f"Parsing and describing files...")
    try:
        if debug:
            asyncio.run(parse_async(path, use_cache, language, None, no_llm=no_llm, llm=llm))
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            ) as progress:
                # No describe bar in no-LLM mode since descriptions are skipped.
                progress_tracker = ProgressTracker(progress, include_describe=not no_llm)
                asyncio.run(parse_async(path, use_cache, language, progress_tracker, no_llm=no_llm, llm=llm))
                progress_tracker.finish()
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    typer.echo(f"✅ Finished parsing project in {elapsed_time:.4f} seconds.")


def _render_inspect(result: Union[InspectResult, str], pretty: bool = True, language: str = "java"):
    if isinstance(result, str):
        console.print(result)
        return

    # Header
    header_text = Text()
    header_text.append(result.id, style="tostr.uid")
    
    if result.type in ["BaseClass", "BaseMethod", "BaseField", "BaseFile"]:
        line_info = f" @L{result.start_line}"
        if result.start_line != result.end_line:
            line_info += f"-{result.end_line}"
        header_text.append(line_info, style="tostr.line_num")

    if result.type in ["BaseClass", "BaseMethod", "BaseField"]:
        header_text.append(f" | {result.signature}", style="bold white")
    else:
        header_text.append(f" | {result.uid}", style="tostr.uid")
    
    console.print(header_text)

    # Description
    total_children = len(result.fields) + len(result.methods) + len(result.classes) + len(result.files) + len(result.directories)
    if result.description and total_children != 1:
        # Simple line wrapping for description
        desc = f"// {result.description}"
        console.print(Text(desc, style="tostr.comment"))

    # Edges
    if result.inbound_edges:
        edge_text = Text("< ", style="tostr.edge")
        edge_text.append(", ".join(result.inbound_edges))
        console.print(edge_text)
        
    if result.outbound_edges:
        edge_text = Text("> ", style="tostr.edge")
        edge_text.append(", ".join(result.outbound_edges))
        console.print(edge_text)

    # Fields
    if result.fields:
        console.print(Text("fields:", style="tostr.section"))
        for f in result.fields:
            console.print(Text(f"    {f.id} | {f.signature}"))

    # Methods
    if result.methods:
        console.print(Text("methods:", style="tostr.section"))
        for m in result.methods:
            console.print(Text(f"    {m.id} | {m.signature}"))
            if m.description:
                console.print(Text(f"        // {m.description}", style="tostr.comment"))

    # Classes
    if result.classes:
        console.print(Text("classes:", style="tostr.section"))
        for c in result.classes:
            console.print(Text(f"    {c.id} | {c.uid}"))
            if c.description:
                console.print(Text(f"        // {c.description}", style="tostr.comment"))

    # Files / Directories recursion (limited to 1 level for inspect usually)
    if result.files:
        for f in result.files:
            console.print(Text(f"File: {f.id} | {f.filepath}", style="bold blue"))
    
    if result.directories:
        for d in result.directories:
            console.print(Text(f"Directory: {d.id} | {d.filepath}", style="bold yellow"))

    # Body
    if result.body:
        syntax = Syntax(
            result.body, 
            language, 
            theme="monokai", 
            line_numbers=True, 
            start_line=result.start_line
        )
        console.print(syntax)

def _short_uid(uid: str) -> str:
    """Trim the redundant filepath prefix from a code struct's UID.

    Files/directories (no '#') are returned unchanged; classes/methods keep the
    '#member' portion since their parent file already shows the path."""
    return re.sub(r"^[^#]*#", "#", uid)

def _render_skeleton(result: SkeletonResult, tree: Tree = None) -> Tree:
    """Recursively builds a Rich Tree for the skeleton."""
    label = Text()
    label.append(result.id, style="tostr.uid")
    label.append(f" | {_short_uid(result.uid)}")
    
    if tree is None:
        tree = Tree(label)
        node = tree
    else:
        node = tree.add(label)
    
    for child in result.children:
        _render_skeleton(child, node)
    return tree

def _render_search(results: List[SearchResult]):
    if not results:
        console.print("No results found matching your query.")
        return
    
    for r in results:
        res_text = Text()
        res_text.append(r.id, style="tostr.uid")
        res_text.append(f"|{r.uid} ", style="bold cyan")
        res_text.append(f"({r.type}) ", style="dim")
        res_text.append(f"[dist: {r.distance:.4f}]", style="tostr.line_num")
        console.print(res_text)
    console.print('\n')

@app.command()
def inspect(
    ids: Annotated[
        List[str], 
        typer.Argument(help="List of struct IDs or UIDs to inspect")
    ],
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory to scan",
        exists=True,       # Typer automatically checks if the path exists
        file_okay=False,   # Typer blocks files, only allowing directories
        dir_okay=True,
        resolve_path=True  # Converts relative paths to absolute paths automatically
    ),
    include_body: Annotated[
        bool, 
        typer.Option(
            "--body/--no-body", 
            help="Include code body in output"
            )
        ] = False,
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty/--raw",
            help="Pretty format output with line wrapping and indentation (disable for raw output)"
        )
    ] = True,
    debug: Annotated[
        bool, 
        typer.Option(
            "--debug/--no-debug", 
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False,
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            "-m",
            help="Maximum number of lines to include in the output (default: 500)"
        )
    ] = 500
):
    """Output the AST details for specific struct IDs or UIDs."""
    configure_cli_logging(debug)
    
    start_time = time.perf_counter()
    try:
        results = asyncio.run(inspect_async(ids, path, include_body=include_body))
        for res in results:
            _render_inspect(res, pretty=pretty)
            console.print("") # spacing
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.debug(f"Finished in {elapsed_time:.4f} seconds.")


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(help="The search query to embed and find similar structs for")
    ],
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    filter: Annotated[
        str,
        typer.Option(
            "--filter",
            "-f",
            help="Filter by struct type (e.g., 'class', 'method')"
        )
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of results to return"
        )
    ] = 5,
    debug: Annotated[
        bool, 
        typer.Option(
            "--debug/--no-debug", 
            "-d/-nd",
            help="Enable debug logging"
            )
    ] = False
):
    """Search for structs by embedding a search term and finding the top K matches."""
    configure_cli_logging(debug)
    try:
        results = asyncio.run(search_async(query, path, filter_type=filter, top_k=top_k))
        _render_search(results)
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)

@app.command()
def skeleton(
    subpath: Annotated[
        str, 
        typer.Argument(help="File or directory path relative to the project root to generate a skeleton for")
    ] = ".",
    path: Path = typer.Argument(
        ".", 
        help="Path to the project directory to scan",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    ),
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty/--raw",
            help="Pretty format output with line wrapping and indentation (disable for raw output)"
        )
    ] = True,
    debug: Annotated[
        bool, 
        typer.Option(
            "--debug/--no-debug",
            help="Enable debug logging"
            )
    ] = False,
    depth: Annotated[
        int,
        typer.Option(
            "--depth",
            "-d",
            help="Depth of directory/file/class nesting to traverse (default: 4). Class members are never expanded; only top-level functions on a file are shown. Use inspect for a class's or file's members."
        )
    ] = 4,
    files_only: Annotated[
        bool,
        typer.Option(
            "--files-only",
            "-f",
            help="Only generate skeleton for files (default: False)"
        )
    ] = False,
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            "-m",
            help="Maximum number of lines to include in the output (default: 500)"
        )
    ] = 500
    
):
    """Output the .tost skeleton format for all files matching a specific subpath."""
    configure_cli_logging(debug)
    
    start_time = time.perf_counter()
    try:
        result = asyncio.run(skeleton_async(subpath, path, depth=depth, files_only=files_only))
        tree = _render_skeleton(result)
        console.print(tree)
    except TostrError as e:
        typer.secho(f"❌ Error: {e}", fg="red", err=True)
        raise typer.Exit(code=1)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.debug(f"Finished in {elapsed_time:.4f} seconds.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()

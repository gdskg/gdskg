import os
import sys
import subprocess
import collections
import time
import json
import sqlite3
import uvicorn
import typer
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from git import Repo
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.routing import Route, Mount

from core.vector_store import VectorStore
from analysis.embedder import ONNXEmbedder
from core.graph_store import GraphStore
from core.extractor import GraphExtractor
from core.plugin_manager import PluginManager, run_runtime_plugins
from core.schema import NodeType
from analysis.search_engine import SearchEngine
from mcp_server.server import mcp, DEFAULT_GRAPH_PATH, get_function_history

app = typer.Typer(help="Git-Derived Software Knowledge Graph CLI")
console = Console()

@app.command()
def build(
    repository: str = typer.Option(..., "--repository", "-R", help="Path or URL to the git repository to analyze"),
    graph: Path = typer.Option(..., "--graph", "-G", help="Directory where the knowledge graph will be stored", file_okay=False, dir_okay=True, resolve_path=True),
    overwrite: bool = typer.Option(False, "--overwrite", "-O", help="Overwrite existing graph database if it exists"),
    plugins: Optional[List[str]] = typer.Option(None, "--build-plugin", help="List of plugins to run at build time"),
    parameters: Optional[List[str]] = typer.Option(None, "--parameter", "-X", help="Plugin parameters in format 'PluginName:Key=Value'"),
    include_bots: bool = typer.Option(False, "--include-bots", help="Include commits authored by bots")
):
    """
    Build the Git-Derived Software Knowledge Graph.

    Args:
        repository (str): Path or URL to the git repository to analyze.
        graph (Path): Directory where the knowledge graph will be stored.
        overwrite (bool): Overwrite existing graph database if it exists.
        plugins (List[str], optional): List of plugins to run at build time.
        parameters (List[str], optional): Plugin parameters in 'PluginName:Key=Value' format.
        include_bots (bool): Include commits authored by bots.

    Returns:
        None
    """
    console.print(f"[bold green]Starting GDSKG Build[/bold green]")
    
    repo_path = _prepare_repository(repository)
    console.print(f"Repository Path: [blue]{repo_path}[/blue]")
    console.print(f"Graph Output: [blue]{graph}[/blue]")
    
    _validate_git_repo(repo_path)
    db_path = _prepare_database_path(graph, overwrite)
    
    store = GraphStore(db_path)
    loaded_plugins = _load_plugins(plugins)
    plugin_config = _parse_plugin_parameters(parameters)

    _run_build_process(repo_path, store, loaded_plugins, plugin_config, include_bots, db_path)
    console.print("[yellow]Detailed analysis complete.[/yellow]")

@app.command()
def query(
    query_str: str = typer.Argument(..., help="Search query (natural language or keywords)"),
    graph: Path = typer.Option(..., "--graph", "-G", help="Directory where the knowledge graph is stored", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    repository: Optional[str] = typer.Option(None, "--repository", "-R", help="Optionally filter results by repository name"),
    depth: int = typer.Option(0, "--depth", "-D", help="Depth of connections to retrieve"),
    traverse: Optional[List[str]] = typer.Option(None, "--dfs-attribute", "-T", help="Specific node types to allowed traversal through"),
    semantic_only: bool = typer.Option(False, "--semantic-only", "-S", help="Disable keyword search and only use semantic embeddings"),
    min_score: float = typer.Option(0.0, "--min-score", "-M", help="Minimum relevance threshold"),
    top_n: int = typer.Option(5, "--top-n", "-N", help="Maximum number of base commits to return"),
    plugins: Optional[List[str]] = typer.Option(None, "--runtime-plugin", help="List of plugins to run at runtime"),
    parameters: Optional[List[str]] = typer.Option(None, "--parameter", "-X", help="Plugin parameters in format 'PluginName:Key=Value'"),
    filters: Optional[List[str]] = typer.Option(None, "--filter", "-F", help="Filter results by node type and value"),
    show_matches: bool = typer.Option(False, "--show-matches", help="Show detailed explanations of matches"),
    all_matches: bool = typer.Option(False, "--all-matches", help="Do not filter deep traversal nodes"),
    all_files: bool = typer.Option(False, "--all-files", help="Show all files edited in matching commits")
):
    """
    Query the Git-Derived Software Knowledge Graph.

    Args:
        query_str (str): Search query (natural language or keywords).
        graph (Path): Directory where the knowledge graph is stored.
        repository (str, optional): Optionally filter results by repository name.
        depth (int): Depth of connections to retrieve.
        traverse (List[str], optional): Specific node types to allowed traversal through.
        semantic_only (bool): Disable keyword search and only use semantic embeddings.
        min_score (float): Minimum relevance threshold.
        top_n (int): Maximum number of base commits to return.
        plugins (List[str], optional): List of plugins to run at runtime.
        parameters (List[str], optional): Plugin parameters in 'PluginName:Key=Value' format.
        filters (List[str], optional): Filter results by node type and value.
        show_matches (bool): Show detailed explanations of matches.
        all_matches (bool): Do not filter deep traversal nodes.
        all_files (bool): Show all files edited in matching commits.

    Returns:
        None
    """
    db_path = graph / "gdskg.db"
    if not db_path.exists():
        console.print(f"[bold red]Error[/bold red]: Database not found at {db_path}")
        raise typer.Exit(code=1)

    if plugins:
        depth = max(depth, 1)

    parsed_filters = _parse_metadata_filters(filters)
    searcher = SearchEngine(str(db_path))
    results = searcher.search(query_str, repo_name=repository, depth=depth, traverse_types=traverse, semantic_only=semantic_only, min_score=min_score, top_n=top_n, filters=parsed_filters, all_matches=all_matches, all_files=all_files)

    if plugins and results:
        commit_ids = [res['id'] for res in results]
        run_runtime_plugins(str(db_path), commit_ids, plugins, parameters)
        results = searcher.search(query_str, repo_name=repository, depth=depth, traverse_types=traverse, semantic_only=semantic_only, min_score=min_score, top_n=top_n, all_matches=all_matches, all_files=all_files)

    if not results:
        console.print("[yellow]No relevant matches found.[/yellow]")
        return

    _display_results_table(results, query_str, depth, show_matches, top_n)

@app.command()
def history(
    function_name: str = typer.Argument(..., help="Name of the function to get history for"),
    graph: Path = typer.Option(..., "--graph", "-G", help="Directory where the knowledge graph is stored", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    plugins: Optional[List[str]] = typer.Option(None, "--runtime-plugin", help="List of plugins to run at runtime"),
    parameters: Optional[List[str]] = typer.Option(None, "--parameter", "-X", help="Plugin parameters in format 'PluginName:Key=Value'")
):
    """
    Get the version history of a specific function.

    Args:
        function_name (str): Name of the function to get history for.
        graph (Path): Directory where the knowledge graph is stored.
        plugins (List[str], optional): List of plugins to run at runtime.
        parameters (List[str], optional): Plugin parameters in 'PluginName:Key=Value' format.

    Returns:
        None
    """
    result = get_function_history(function_name, graph_path=str(graph), plugins=plugins, parameters=parameters)
    console.print(result)

@app.command()
def serve(
    transport: str = typer.Option("sse", "--transport", "-t", help="Transport protocol: 'sse' or 'stdio'"),
    port: int = typer.Option(8015, "--port", "-p", help="Port to listen on (SSE only)"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to listen on (SSE only)"),
    graph: Path = typer.Option(None, "--graph", "-G", help="Directory where the knowledge graph is stored", file_okay=False, dir_okay=True, resolve_path=True)
):
    """
    Start the MCP server.

    Args:
        transport (str): Transport protocol: 'sse' or 'stdio'.
        port (int): Port to listen on (SSE only).
        host (str): Host to listen on (SSE only).
        graph (Path, optional): Directory where the knowledge graph is stored.

    Returns:
        None
    """
    global console
    g_path = graph if graph else DEFAULT_GRAPH_PATH
    db_path = g_path / "gdskg.db"
    
    if transport.lower() == "stdio":
        console = Console(stderr=True)
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
    
    if not db_path.exists():
        _attempt_auto_build(g_path)

    if transport.lower() == "stdio":
        sys.stdout = _real_stdout
        mcp.run(transport="stdio")
    else:
        _run_sse_server(host, port)

def _prepare_repository(repository: str) -> Path:
    """
    Ensure the repository is available as a local directory.
    
    If given a URL, it clones or pulls the repository to a local cache.
    Otherwise, it resolves the provided path.

    Args:
        repository (str): The repository name, local path, or URL.

    Returns:
        Path: The local path to the repository.
    """
    if repository.startswith(("http://", "https://")):
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        repo_url = repository.replace("https://", f"https://{token}@") if token and "@" not in repository else repository
        cache_dir = Path.home() / ".gdskg" / "cache"
        repo_name = repository.split("/")[-1].replace(".git", "")
        checkout_path = cache_dir / repo_name
        
        if checkout_path.exists():
            console.print("Repository already cached. Updating...")
            try:
                repo = Repo(checkout_path)
                repo.remotes.origin.pull()
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to update cached repo: {e}[/yellow]")
        else:
            console.print("Cloning repository...")
            try:
                Repo.clone_from(repo_url, checkout_path)
            except Exception as e:
                console.print(f"[bold red]Error[/bold red]: Failed to clone repository: {e}")
                raise typer.Exit(code=1)
        return checkout_path
    
    path = Path(repository).resolve()
    if not path.exists():
        console.print(f"[bold red]Error[/bold red]: Path {path} does not exist.")
        raise typer.Exit(code=1)
    return path

def _validate_git_repo(repo_path: Path):
    """
    Verify that the given path is a valid Git repository.

    Args:
        repo_path (Path): Path to the repository directory.

    Returns:
        None
    """
    if not (repo_path / ".git").exists():
        console.print("[bold red]Error[/bold red]: The specified directory is not a git repository.")
        raise typer.Exit(code=1)

def _prepare_database_path(graph: Path, overwrite: bool) -> Path:
    """
    Initialize the graph storage directory and determine the database path.

    Args:
        graph (Path): The graph directory path.
        overwrite (bool): Whether to overwrite the existing database.

    Returns:
        Path: The path to the SQLite database file.
    """
    graph.mkdir(parents=True, exist_ok=True)
    db_path = graph / "gdskg.db"
    
    if db_path.exists():
        if overwrite:
            console.print(f"Removing existing database at {db_path}...")
            db_path.unlink()
        else:
            console.print(f"Appending to existing database at {db_path}...")
    return db_path

def _load_plugins(plugins: Optional[List[str]]) -> List[Any]:
    """
    Load and return a list of plugin instances based on provided names.

    Args:
        plugins (List[str], optional): A list of plugin names.

    Returns:
        List[Any]: A list of loaded plugin instances.
    """
    manager = PluginManager()
    if plugins:
        console.print(f"Loading plugins: {plugins}...")
        manager.load_plugins(plugins)
    loaded = manager.get_plugins()
    if loaded:
        console.print(f"[green]Enabled {len(loaded)} plugins.[/green]")
    return loaded

def _parse_plugin_parameters(parameters: Optional[List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Convert a list of parameter strings into a nested configuration dictionary.

    Args:
        parameters (List[str], optional): List of parameter strings ('Plugin:Key=Val').

    Returns:
        Dict[str, Dict[str, str]]: Nested configuration dictionary.
    """
    config = {}
    if not parameters:
        return config
    for param in parameters:
        try:
            if ":" in param and "=" in param:
                plugin_part, rest = param.split(":", 1)
                key, value = rest.split("=", 1)
                if plugin_part not in config:
                    config[plugin_part] = {}
                config[plugin_part][key] = value
        except Exception as e:
            console.print(f"[yellow]Warning: Error parsing parameter '{param}': {e}[/yellow]")
    return config

def _run_build_process(repo_path: Path, store: GraphStore, plugins: List[Any], plugin_config: Dict, include_bots: bool, db_path: Path):
    """
    Execute the core graph extraction and semantic indexing pipeline.

    Args:
        repo_path (Path): Path to the git repository.
        store (GraphStore): Graph storage instance.
        plugins (List[Any]): Loaded plugin instances.
        plugin_config (Dict): Plugin configurations.
        include_bots (bool): Whether to include bot commits.
        db_path (Path): Path to the database file.

    Returns:
        None
    """
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), "•", TimeElapsedColumn(), "•", TimeRemainingColumn(), console=console, transient=True) as progress:
        extractor = GraphExtractor(repo_path, store, plugins=plugins, plugin_config=plugin_config, progress=progress, skip_bots=not include_bots)
        try:
            extractor.process_repo()
            console.print(f"[bold green]✓[/bold green] Graph built with [blue]{store.count_nodes()}[/blue] nodes.")
            
            _run_semantic_indexing(db_path, progress)
        except Exception as e:
            console.print(f"[bold red]✗ Build failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        finally:
            store.close()

def _run_semantic_indexing(db_path: Path, progress: Progress):
    """Generate and store semantic vector embeddings for the graph nodes."""
    t0 = time.time()
    task = progress.add_task("[cyan]Phase 4: Semantic indexing...[/cyan]", total=None)
    vstore = VectorStore()
    embedder = ONNXEmbedder()
    
    def callback(count, total=None):
        if total is not None:
            progress.update(task, total=total)
        if count > 0:
            progress.advance(task, count)
            
    count = vstore.build_from_graph(str(db_path), embedder, progress_callback=callback)
    duration = time.time() - t0
    console.print(f"[bold green]✓[/bold green] Phase 4 complete! [blue]{count}[/blue] embeddings generated in [blue]{duration:.2f}s[/blue].")

def _parse_metadata_filters(filters: Optional[List[str]]) -> Dict[str, str]:
    """
    Convert a list of filter strings like 'Type:Value' into a dictionary.

    Args:
        filters (List[str], optional): List of filter strings.

    Returns:
        Dict[str, str]: Parsed filters dictionary.
    """
    parsed = {}
    if filters:
        for f in filters:
            if ":" in f:
                ftype, fval = f.split(":", 1)
                parsed[ftype.upper()] = fval
    return parsed

def _display_results_table(results: List[Dict], query: str, depth: int, show_matches: bool, top_n: int):
    """
    Render a Rich table displaying the search results.

    Args:
        results (List[Dict]): List of search results.
        query (str): The search query.
        depth (int): The search depth.
        show_matches (bool): Whether to show match reasons.
        top_n (int): The top_n parameter.

    Returns:
        None
    """
    table = Table(title=f"Search Results for: '{query}' (Depth: {depth})", box=None, show_lines=True)
    table.add_column("Rel.", justify="right", style="cyan")
    table.add_column("Commit", style="magenta")
    table.add_column("Details", style="white")

    for res in results:
        detail = _build_result_detail_text(res, depth, show_matches)
        table.add_row(f"{res['relevance']:.2f}", res["id"][:8], detail)

    console.print(table)
    msg = f"Top {len(results)} relevant commits" if len(results) >= top_n else f"Found {len(results)} relevant {'commit' if len(results) == 1 else 'commits'}"
    console.print(f"\n[bold green]{msg}.[/bold green]")

def _build_result_detail_text(res: Dict, depth: int, show_matches: bool) -> Text:
    """
    Format the detailed information for a single search result.

    Args:
        res (Dict): The search result data.
        depth (int): The search depth.
        show_matches (bool): Whether to show match reasons.

    Returns:
        Text: A Rich Text object containing the formatted detail.
    """
    detail = Text()
    detail.append(f"{res['message'].split('\n')[0]}\n", style="bold white")
    detail.append(f"Author: ", style="dim")
    detail.append(f"{res['author']}  ", style="green")
    detail.append(f"Date: ", style="dim")
    detail.append(f"{res['date'][:10]}\n", style="green")

    if show_matches and res.get('reasons'):
        detail.append("Matched because:\n", style="bold cyan")
        for reason in res['reasons']:
            detail.append(f" • {reason}\n", style="dim cyan")
        detail.append("\n")

    if depth > 0:
        _append_connections_to_detail(detail, res.get('connections', {}))
    return detail

def _append_connections_to_detail(detail: Text, connections: Dict):
    """
    Append traversed graph connections to the result detail text.

    Args:
        detail (Text): The detail text to append to.
        connections (Dict): The connected nodes data.

    Returns:
        None
    """
    grouped = collections.defaultdict(list)
    for node_id, info in connections.items():
        grouped[info['type']].append((node_id, info.get('attributes', {})))

    type_colors = {
        NodeType.FILE.value: "yellow", NodeType.SYMBOL.value: "blue",
        NodeType.AUTHOR.value: "green", NodeType.COMMIT.value: "magenta",
        NodeType.REPOSITORY.value: "cyan", NodeType.TIME_BUCKET.value: "white",
        NodeType.COMMIT_MESSAGE.value: "italic white", NodeType.COMMENT.value: "italic dim white",
        "PULL_REQUEST": "bold cyan"
    }

    for ntype, nodes in grouped.items():
        color = type_colors.get(ntype, "white")
        label = ntype.replace('_', ' ').title()
        detail.append(f"{label}: ", style=f"dim {color}")
        
        if ntype == "PULL_REQUEST":
            for node_id, attrs in nodes:
                num = attrs.get('number', '?')
                title = attrs.get('title', 'PR')
                detail.append(f"#{num} {title}", style=color).append("\n")
                if attrs.get('body'):
                    for line in attrs['body'].split('\n'):
                        detail.append(f"      {line}\n", style=color)
        else:
            names = []
            for node_id, attrs in nodes:
                if ntype == NodeType.FILE.value:
                    names.append(Path(node_id).name)
                elif ntype in (NodeType.COMMIT_MESSAGE.value, NodeType.COMMENT.value):
                    content = attrs.get('content', node_id).strip().split('\n')[0]
                    names.append(content[:50] + ("..." if len(content) > 50 else ""))
                else:
                    names.append(node_id)
            
            if ntype == NodeType.KEYWORD.value:
                detail.append(", ".join(names[:5]), style=color)
                if len(names) > 5:
                    detail.append(f" (+{len(names)-5} more)", style=f"dim {color}")
            else:
                detail.append(", ".join(names), style=color)
            detail.append("\n")

def _attempt_auto_build(g_path: Path):
    """
    Automatically trigger a graph build if the database is missing and GDSKG_REPO is set.

    Args:
        g_path (Path): The graph directory path.

    Returns:
        None
    """
    repo_env = os.environ.get("GDSKG_REPO")
    if not repo_env:
        console.print("[yellow]Warning: Starting server without database. Index a repository first.[/yellow]")
        return

    console.print(f"Auto-building graph for [blue]{repo_env}[/blue]...")
    cmd = [sys.executable, __file__, "build", "--repository", repo_env, "--graph", str(g_path)]
    if os.environ.get("GDSKG_OVERWRITE", "false").lower() == "true":
        cmd.append("--overwrite")
    if os.environ.get("GDSKG_INCLUDE_BOTS", "false").lower() == "true":
        cmd.append("--include-bots")
    for p in os.environ.get("GDSKG_PLUGINS", "").split(","):
        if p.strip(): cmd.extend(["--build-plugin", p.strip()])
    
    try:
        subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)
        console.print("[bold green]✓ Auto-build complete.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Auto-build failed:[/bold red] {e}")

def _run_sse_server(host: str, port: int):
    """
    Start the MCP server using the SSE transport.

    Args:
        host (str): Host to listen on.
        port (int): Port to listen on.

    Returns:
        None
    """
    mcp.settings.host, mcp.settings.port = host, port
    mcp.settings.sse_path, mcp.settings.message_path = "/mcp", "/mcp/messages"
    
    mcp_sse_app = mcp.sse_app()
    app = Starlette(routes=[
        Route("/", lambda r: RedirectResponse(url="/mcp"), methods=["GET"]),
        Route("/", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
        Route("/mcp", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
        Mount("/", mcp_sse_app)
    ])
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    
    console.print(f"[bold green]Starting GDSKG MCP Server[/bold green]")
    console.print(f"URL: [blue]http://{host}:{port}/mcp[/blue]")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app()


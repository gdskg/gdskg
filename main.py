import typer
import sys
import os
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from core.vector_store import VectorStore
from analysis.embedder import ONNXEmbedder

app = typer.Typer(help="Git-Derived Software Knowledge Graph CLI")
console = Console()

@app.command()
def build(
    repository: str = typer.Option(
        ..., "--repository", "-R", 
        help="Path or URL to the git repository to analyze"
    ),
    graph: Path = typer.Option(
        ..., "--graph", "-G",
        help="Directory where the knowledge graph (SQLite db) will be stored",
        file_okay=False, dir_okay=True, resolve_path=True
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-O",
        help="Overwrite existing graph database if it exists"
    ),
    plugins: Optional[List[str]] = typer.Option(
        None, "--plugin", "-P",
        help="List of plugins to enable (e.g., 'python-complexity'). Plugins must exist under https://github.com/gdskg/"
    ),
    parameters: Optional[List[str]] = typer.Option(
        None, "--parameter", "-X",
        help="Plugin parameters in format 'PluginName:Key=Value' (e.g., 'ClickUpTask:regex=^CU-\\d+$')"
    )
):
    """
    Build the Git-Derived Software Knowledge Graph.
    """
    console.print(f"[bold green]Starting GDSKG Build[/bold green]")
    
    # Check if repository is a URL
    repo_path = None
    if repository.startswith("http://") or repository.startswith("https://"):
        import os
        from git import Repo
        
        console.print(f"Detected remote repository URL: [blue]{repository}[/blue]")
        
        # Inject token if available
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if token and "@" not in repository:
            # Simple injection for HTTPS
            repo_url = repository.replace("https://", f"https://{token}@")
        else:
            repo_url = repository
            
        # Determine local checkout path
        # Use a consistent cache location
        cache_dir = Path.home() / ".gdskg" / "cache"
        repo_name = repository.split("/")[-1].replace(".git", "")
        checkout_path = cache_dir / repo_name
        
        console.print(f"Checkout path: [blue]{checkout_path}[/blue]")
        
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
        
        repo_path = checkout_path
    else:
        # Local path
        repo_path = Path(repository).resolve()
        if not repo_path.exists():
             console.print(f"[bold red]Error[/bold red]: Path {repo_path} does not exist.")
             raise typer.Exit(code=1)

    console.print(f"Repository Path: [blue]{repo_path}[/blue]")
    console.print(f"Graph Output: [blue]{graph}[/blue]")
    
    if not (repo_path / ".git").exists():
        console.print("[bold red]Error[/bold red]: The specified directory is not a git repository.")
        raise typer.Exit(code=1)

    graph.mkdir(parents=True, exist_ok=True)
    db_path = graph / "gdskg.db"
    
    if db_path.exists():
        if overwrite:
            console.print(f"Removing existing database at {db_path}...")
            db_path.unlink()
        else:
            console.print(f"Appending to existing database at {db_path}...")
    
    # Initialize DB
    from core.graph_store import GraphStore
    from core.extractor import GraphExtractor
    from core.plugin_manager import PluginManager
    
    store = GraphStore(db_path)
    
    # Load Plugins
    plugin_manager = PluginManager()
    if plugins:
        console.print(f"Loading plugins: {plugins}...")
        plugin_manager.load_plugins(plugins)
    
    loaded_plugins = plugin_manager.get_plugins()
    if loaded_plugins:
        console.print(f"[green]Enabled {len(loaded_plugins)} plugins.[/green]")

    # Parse Parameters
    plugin_config = {}
    if parameters:
        for param in parameters:
            try:
                # Expect format Plugin:Key=Value
                if ":" in param and "=" in param:
                    plugin_part, rest = param.split(":", 1)
                    key, value = rest.split("=", 1)
                    if plugin_part not in plugin_config:
                        plugin_config[plugin_part] = {}
                    plugin_config[plugin_part][key] = value
                else:
                    console.print(f"[yellow]Warning: Ignoring malformed parameter '{param}'. Expected format 'Plugin:Key=Value'[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Error parsing parameter '{param}': {e}[/yellow]")
    
    if plugin_config:
        console.print(f"Plugin configuration: {plugin_config}")

    # Start Processing
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        extractor = GraphExtractor(repo_path, store, plugins=loaded_plugins, plugin_config=plugin_config, 
                                   progress=progress, task_id=None)
        
        try:
            extractor.process_repo()
            
            node_count = store.count_nodes()
            console.print(f"[bold green]✓[/bold green] Graph built with [blue]{node_count}[/blue] nodes.")
            
            # Semantic Indexing
            embed_task = progress.add_task("[cyan]Semantic indexing...", total=None)
            
            vstore = VectorStore(db_path)
            embedder = ONNXEmbedder()
            
            def update_embed_progress(count, total=None):
                if total is not None:
                    progress.update(embed_task, total=total)
                if count > 0:
                    progress.advance(embed_task, count)
            
            embedded_count = vstore.build_from_graph(str(db_path), embedder, progress_callback=update_embed_progress)
            console.print(f"[bold green]✓[/bold green] Semantic indexing complete! [blue]{embedded_count}[/blue] embeddings generated.")
            
        except Exception as e:
            console.print(f"[bold red]✗ Build failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        finally:
            store.close()
        
    console.print("[yellow]Detailed analysis complete.[/yellow]")

@app.command()
def query(
    query_str: str = typer.Argument(..., help="Search query (natural language or keywords)"),
    graph: Path = typer.Option(
        ..., "--graph", "-G",
        help="Directory where the knowledge graph (SQLite db) is stored",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    repository: Optional[str] = typer.Option(
        None, "--repository", "-R",
        help="Optionally filter results by repository name"
    ),
    depth: int = typer.Option(
        0, "--depth", "-D",
        help="Depth of connections to retrieve (0 = just the node, 1 = immediate neighbors, etc.)"
    ),
    traverse: Optional[List[str]] = typer.Option(
        None, "--dfs-attribute", "-T",
        help="Specific node types to allowed traversal through (e.g., AUTHOR, TIME_BUCKET). Default: blocks AUTHOR, TIME_BUCKET, REPOSITORY."
    ),
    semantic_only: bool = typer.Option(
        False, "--semantic-only", "-S",
        help="Disable keyword search and only use semantic embeddings"
    ),
    min_score: float = typer.Option(
        0.0, "--min-score", "-M",
        help="Minimum relevance score to include in results"
    ),
    top_n: int = typer.Option(
        5, "--top-n", "-N",
        help="Maximum number of base commits to return"
    ),
    plugins: Optional[List[str]] = typer.Option(
        None, "--plugin", "-P",
        help="List of plugins to enable (e.g., 'GitHubPR')."
    ),
    parameters: Optional[List[str]] = typer.Option(
        None, "--parameter", "-X",
        help="Plugin parameters in format 'PluginName:Key=Value'."
    )
):
    """
    Query the Git-Derived Software Knowledge Graph.
    """
    db_path = graph / "gdskg.db"
    if not db_path.exists():
        console.print(f"[bold red]Error[/bold red]: Database not found at {db_path}")
        raise typer.Exit(code=1)

    from analysis.search_engine import SearchEngine
    from rich.table import Table
    from rich.text import Text

    if plugins:
        depth = max(depth, 1)

    searcher = SearchEngine(str(db_path)) # Uses shared DB for vectors by default
    results = searcher.search(query_str, repo_name=repository, depth=depth, traverse_types=traverse, semantic_only=semantic_only, min_score=min_score, top_n=top_n)

    if plugins and results:
        from core.plugin_manager import run_runtime_plugins
        commit_ids = [res['id'] for res in results]
        run_runtime_plugins(str(db_path), commit_ids, plugins, parameters)
        # Re-run search to include newly added plugin nodes
        results = searcher.search(query_str, repo_name=repository, depth=depth, traverse_types=traverse, semantic_only=semantic_only, min_score=min_score, top_n=top_n)

    if not results:
        console.print("[yellow]No relevant matches found.[/yellow]")
        return

    table = Table(title=f"Search Results for: '{query_str}' (Depth: {depth})", box=None, show_lines=True)
    table.add_column("Rel.", justify="right", style="cyan")
    table.add_column("Commit", style="magenta")
    table.add_column("Details", style="white")

    for res in results:
        # Create a detail block
        detail = Text()
        detail.append(f"{res['message'].split('\n')[0]}\n", style="bold white")
        detail.append(f"Author: ", style="dim")
        detail.append(f"{res['author']}  ", style="green")
        detail.append(f"Date: ", style="dim")
        detail.append(f"{res['date'][:10]}\n", style="green")

        if res.get('reasons'):
            detail.append("Matched because:\n", style="bold cyan")
            for reason in res['reasons']:
                detail.append(f" • {reason}\n", style="dim cyan")
            detail.append("\n")

        if depth > 0:
            # Group connections by type
            from core.schema import NodeType
            grouped = {} # NodeType -> List[str]
            for node_id, info in res['connections'].items():
                ntype = info['type']
                attrs = info.get('attributes', {})
                
                # Determine display name
                if ntype == NodeType.REPOSITORY.value:
                    display_name = attrs.get('name', Path(node_id).name)
                elif ntype == NodeType.COMMIT_MESSAGE.value:
                    # Show a snippet of the message
                    content = attrs.get('content', node_id).strip()
                    display_name = content.split('\n')[0][:50]
                    if len(content.split('\n')[0]) > 50:
                        display_name += "..."
                elif ntype == NodeType.FILE.value:
                    display_name = Path(node_id).name
                elif ntype == NodeType.COMMENT.value:
                    content = attrs.get('content', node_id).strip()
                    display_name = content.split('\n')[0][:50]
                    if len(content.split('\n')[0]) > 50:
                        display_name += "..."
                elif ntype == "PULL_REQUEST":
                    number = attrs.get('number', '?')
                    title = attrs.get('title', 'Extracted Pull Request')
                    body = attrs.get('body')
                    
                    display_name = f"#{number} {title}"
                    if body:
                        # Append the full description, indented slightly
                        indented_body = "\n".join([f"      {line}" for line in body.split('\n')])
                        display_name += f"\n{indented_body}"
                else:
                    display_name = node_id

                if ntype not in grouped:
                    grouped[ntype] = []
                grouped[ntype].append(display_name)
            
            type_colors = {
                NodeType.FILE.value: "yellow",
                NodeType.SYMBOL.value: "blue",
                NodeType.AUTHOR.value: "green",
                NodeType.COMMIT.value: "magenta",
                NodeType.REPOSITORY.value: "cyan",
                NodeType.TIME_BUCKET.value: "white",
                NodeType.COMMIT_MESSAGE.value: "italic white",
                NodeType.COMMENT.value: "italic dim white",
                "PULL_REQUEST": "bold cyan"
            }

            for ntype, nodes in grouped.items():
                color = type_colors.get(ntype, "white")
                label = ntype.replace('_', ' ').title()
                detail.append(f"{label}: ", style=f"dim {color}")
                
                # For PRs, we want to skip the "comma separated" logic and just print them on their own lines
                if ntype == "PULL_REQUEST":
                    for pr in nodes:
                        detail.append(pr, style=color)
                        detail.append("\n")
                else:
                    node_names = [Path(n).name if ntype == NodeType.FILE.value else n for n in nodes]
                    
                    if ntype == NodeType.KEYWORD.value:
                        detail.append(", ".join(node_names[:5]), style=color)
                        if len(node_names) > 5:
                            detail.append(f" (+{len(node_names)-5} more)", style=f"dim {color}")
                    else:
                        detail.append(", ".join(node_names), style=color)
                        
                    detail.append("\n")

        table.add_row(
            f"{res['relevance']:.2f}",
            res["id"][:8],
            detail
        )

    console.print(table)
    console.print(f"\n[bold green]Found {len(results)} relevant commits.[/bold green]")

@app.command()
def history(
    function_name: str = typer.Argument(..., help="Name of the function to get history for"),
    graph: Path = typer.Option(
        ..., "--graph", "-G",
        help="Directory where the knowledge graph (SQLite db) is stored",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    plugins: Optional[List[str]] = typer.Option(
        None, "--plugin", "-P",
        help="List of plugins to enable (e.g., 'GitHubPR')."
    ),
    parameters: Optional[List[str]] = typer.Option(
        None, "--parameter", "-X",
        help="Plugin parameters in format 'PluginName:Key=Value'."
    )
):
    """
    Get the version history of a specific function.
    """
    from mcp_server.server import get_function_history
    
    result = get_function_history(function_name, graph_path=str(graph), plugins=plugins, parameters=parameters)
    console.print(result)

@app.command()
def serve(
    transport: str = typer.Option("sse", "--transport", "-t", help="Transport protocol: 'sse' (default) or 'stdio'"),
    port: int = typer.Option(8015, "--port", "-p", help="Port to listen on (SSE only)"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to listen on (SSE only)")
):
    """
    Start the MCP server.
    """
    from mcp_server.server import mcp
    
    if transport.lower() == "stdio":
        mcp.run(transport="stdio")
    else:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import RedirectResponse
        from starlette.routing import Route, Mount
        
        # Configure FastMCP settings
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.settings.sse_path = "/mcp" 
        mcp.settings.message_path = "/mcp/messages" # Distinct but nested path
        
        # Get the SSE app
        mcp_sse_app = mcp.sse_app()
        
        # Robust wrapper with 307 Redirects for POST body preservation
        app = Starlette(
            routes=[
                # GET / -> /mcp
                Route("/", lambda r: RedirectResponse(url="/mcp"), methods=["GET"]),
                # POST / -> /mcp/messages
                Route("/", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
                # POST /mcp -> /mcp/messages (Fixes 405 Method Not Allowed)
                Route("/mcp", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
                # Mount the actual app
                Mount("/", mcp_sse_app)
            ]
        )
        
        # Add CORS support
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        console.print(f"[bold green]Starting GDSKG MCP Server (Perfected Standard SSE)[/bold green]")
        console.print(f"Standard URL: [blue]http://{host}:{port}/mcp[/blue]")
        
        uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    app()

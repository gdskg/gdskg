import typer
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console

app = typer.Typer(help="Git-Derived Software Knowledge Graph CLI")
console = Console()

@app.command()
def build(
    repository: Path = typer.Option(
        ..., "--repository", "-R", 
        help="Path to the local git repository to analyze",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
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
    console.print(f"Repository: [blue]{repository}[/blue]")
    console.print(f"Graph Output: [blue]{graph}[/blue]")
    
    if not (repository / ".git").exists():
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
    console.print(f"[bold green]Starting extraction...[/bold green]")
    extractor = GraphExtractor(repository, store, plugins=loaded_plugins, plugin_config=plugin_config)
    
    try:
        extractor.process_repo()
        node_count = store.count_nodes()
        console.print(f"[bold green]Success![/bold green] Graph built with {node_count} nodes.")
    except Exception as e:
        console.print(f"[bold red]Failed:[/bold red] {e}")
        # raise e # Debugging
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

    searcher = SearchEngine(str(db_path))
    results = searcher.search(query_str, repo_name=repository, depth=depth, traverse_types=traverse)

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
                else:
                    display_name = node_id

                if ntype not in grouped:
                    grouped[ntype] = []
                grouped[ntype].append(display_name)
            
            # Simple color mapping for types
            type_colors = {
                NodeType.FILE.value: "yellow",
                NodeType.SYMBOL.value: "blue",
                NodeType.AUTHOR.value: "green",
                NodeType.COMMIT.value: "magenta",
                NodeType.REPOSITORY.value: "cyan",
                NodeType.TIME_BUCKET.value: "white",
                NodeType.COMMIT_MESSAGE.value: "italic white"
            }

            for ntype, nodes in grouped.items():
                color = type_colors.get(ntype, "white")
                label = ntype.replace('_', ' ').title()
                detail.append(f"{label}: ", style=f"dim {color}")
                
                # Show up to 5 nodes per type
                node_names = [Path(n).name if ntype == NodeType.FILE.value else n for n in nodes]
                detail.append(", ".join(node_names[:5]), style=color)
                if len(node_names) > 5:
                    detail.append(f" (+{len(node_names)-5} more)", style=f"dim {color}")
                detail.append("\n")

        table.add_row(
            str(res["relevance"]),
            res["id"][:8],
            detail
        )

    console.print(table)
    console.print(f"\n[bold green]Found {len(results)} relevant commits.[/bold green]")

if __name__ == "__main__":
    app()

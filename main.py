import typer
import sys
from pathlib import Path
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
    
    store = GraphStore(db_path)
    
    # Start Processing
    console.print(f"[bold green]Starting extraction...[/bold green]")
    extractor = GraphExtractor(repository, store)
    
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

if __name__ == "__main__":
    app()

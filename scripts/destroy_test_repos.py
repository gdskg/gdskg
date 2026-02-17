import shutil
from pathlib import Path
from rich.console import Console

console = Console()

def main():
    base_dir = Path("/tmp/gdskg_test_repos")
    if base_dir.exists():
        console.print(f"[yellow]Removing {base_dir}...[/yellow]")
        shutil.rmtree(base_dir)
        console.print("[green]Cleanup complete.[/green]")
    else:
        console.print("[blue]No test repositories found to clean.[/blue]")

if __name__ == "__main__":
    main()

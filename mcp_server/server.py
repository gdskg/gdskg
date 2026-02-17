from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import os

# Create the MCP server
mcp = FastMCP("gdskg")

# --- Server Status (file-based, shared between tray app and server process) ---
# The status file lives at ~/.gdskg/server_status and contains either "started" or "stopped".
# This allows the tray app to toggle the server state from a separate process.
_STATUS_DIR = Path.home() / ".gdskg"
_STATUS_FILE = _STATUS_DIR / "server_status"


def _ensure_status_dir():
    """Create the status directory if it doesn't exist."""
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)


def _read_server_status() -> bool:
    """Read the server status from the file. Returns True if started, False if stopped."""
    try:
        if _STATUS_FILE.exists():
            return _STATUS_FILE.read_text().strip() == "started"
    except Exception:
        pass
    return True  # Default to started if file missing or unreadable


def _write_server_status(started: bool):
    """Write the server status to the file."""
    _ensure_status_dir()
    _STATUS_FILE.write_text("started" if started else "stopped")


def start_server() -> str:
    """Start the GDSKG server. When started, the server will process incoming tool requests."""
    _write_server_status(True)
    return "Server started. Ready to handle requests."


def stop_server() -> str:
    """Stop the GDSKG server. When stopped, the server will reject incoming tool requests."""
    _write_server_status(False)
    return "Server stopped. Requests will be rejected until the server is started again."


def get_server_status() -> str:
    """Get the current status of the GDSKG server."""
    started = _read_server_status()
    return f"Server is {'started' if started else 'stopped'}."


# Auto-set server to started on module load
_write_server_status(True)

# Helper to get the GDSKG root directory
def get_project_root() -> Path:
    # Assuming this file is at gdskg/mcp_server/server.py
    return Path(__file__).resolve().parent.parent.parent

# Set up paths
PROJECT_ROOT = get_project_root()
# Standard GDSKG graph location
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "gdskg_graph"

# Ensure potential import paths are available
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "gdskg"))


# --- Knowledge Graph Tools ---

@mcp.tool()
def query_knowledge_graph(query: str, graph_path: str = str(DEFAULT_GRAPH_PATH), limit: int = 10) -> str:
    """
    Query the Git-Derived Software Knowledge Graph using natural language or keywords.
    
    Args:
        query: The search query string.
        graph_path: Path to the knowledge graph SQLite database directory. Defaults to './gdskg_graph'.
        limit: Maximum number of results to return.
        
    Returns:
        A formatted string containing the search results.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."

    graph_dir = Path(graph_path)
    db_path = graph_dir / "gdskg.db"
    
    if not db_path.exists():
        return f"Error: Database not found at {db_path}. Please index a repository first."

    try:
        # Import internally (with 'core' and 'analysis' in path)
        from analysis.search_engine import SearchEngine
        from core.schema import NodeType

        # SearchEngine.search signature: search(query, repo_name=None, depth=0, traverse_types=None)
        # It doesn't seem to have a limit parameter in the search method, but returns a list.
        # We can slice the result.
        
        searcher = SearchEngine(str(db_path))
        results = searcher.search(query)
        
        if not results:
            return "No relevant matches found."

        # Format results
        output = f"Search Results for: '{query}'\n\n"
        
        for i, res in enumerate(results[:limit]):
            output += f"{i+1}. [Relevance: {res['relevance']:.2f}] Commit: {res['id'][:8]}\n"
            output += f"   Message: {res['message'].split('\n')[0]}\n"
            output += f"   Author: {res['author']} | Date: {res['date'][:10]}\n"
            
            # Basic connection info if available
            connections = res.get('connections', {})
            if connections:
                output += "   Related:\n"
                grouped = {}
                for node_id, info in connections.items():
                    ntype = info['type']
                    if ntype not in grouped:
                        grouped[ntype] = []
                    grouped[ntype].append(node_id)
                
                for ntype, nodes in grouped.items():
                    # clean up node names
                    display_nodes = [Path(n).name if ntype == NodeType.FILE.value else n for n in nodes]
                    output += f"     - {ntype}: {', '.join(display_nodes[:3])}"
                    if len(display_nodes) > 3:
                        output += f" (+{len(display_nodes)-3} more)"
                    output += "\n"
            
            output += "\n"
            
        return output

    except Exception as e:
        return f"Error querying graph: {str(e)}"

@mcp.tool()
def index_folder(
    path: str, 
    graph_path: str = str(DEFAULT_GRAPH_PATH), 
    overwrite: bool = False,
    plugins: list[str] = None,
    parameters: list[str] = None
) -> str:
    """
    Index a local git repository folder into the knowledge graph.
    
    Args:
        path: Absolute path to the local git repository folder.
        graph_path: Directory where the knowledge graph (SQLite db) will be stored. Defaults to './gdskg_graph'.
        overwrite: If True, overwrites the existing graph database.
        plugins: Optional list of plugins to enable (e.g., ['GitHubPR']).
        parameters: Optional list of plugin parameters in format 'PluginName:Key=Value'.
        
    Returns:
        Success message with node count or error message.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."
    repo_path = Path(path).resolve()
    graph_dir = Path(graph_path).resolve()
    
    if not repo_path.exists():
        return f"Error: Repository path '{repo_path}' does not exist."
    
    if not (repo_path / ".git").exists():
         return f"Error: The specified directory '{repo_path}' is not a git repository."

    try:
        # Create graph directory
        graph_dir.mkdir(parents=True, exist_ok=True)
        db_path = graph_dir / "gdskg.db"
        
        if db_path.exists():
            if overwrite:
                db_path.unlink()
            else:
                # Appending is supported
                pass

        # Initialize DB and Core modules
        from core.graph_store import GraphStore
        from core.extractor import GraphExtractor
        from core.plugin_manager import PluginManager
        
        store = GraphStore(db_path)
        
        # Load Plugins
        plugin_manager = PluginManager()
        if plugins:
            try:
                plugin_manager.load_plugins(plugins)
            except Exception as e:
                 return f"Error loading plugins: {e}"
        
        loaded_plugins = plugin_manager.get_plugins()
        if plugins and len(loaded_plugins) < len(plugins):
             loaded_names = [p.__class__.__name__ for p in loaded_plugins]
             # This is a bit tricky because PluginManager logs and continues.
             # We can't easily know WHICH ones failed without checking PluginManager state.
             # But we can at least return a message.
             return f"Error: Only {len(loaded_plugins)}/{len(plugins)} plugins loaded successfully. Please check your plugin names and connectivity."
        
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
                        pass # Ignore or log warning
                except Exception as e:
                    pass # Ignore error
        
        # Initialize Extractor
        extractor = GraphExtractor(repo_path, store, plugins=loaded_plugins, plugin_config=plugin_config)
        
        # Run extraction
        extractor.process_repo()
        node_count = store.count_nodes()
        store.close()
        
        msg = f"Successfully indexed '{repo_path}'. Graph built at '{db_path}' with {node_count} nodes."
        if loaded_plugins:
             msg += f" Plugins enabled: {len(loaded_plugins)}."
        return msg

    except Exception as e:
        return f"Error indexing folder: {str(e)}"

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    mcp.run()

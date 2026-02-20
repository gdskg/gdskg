import os
import sys
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
from core.vector_store import VectorStore
from analysis.embedder import ONNXEmbedder

mcp = FastMCP("gdskg")


# --- Server Status (file-based, shared between tray app and server process) ---
# The status file lives at ~/.gdskg/server_status and contains either "started" or "stopped".
# This allows the tray app to toggle the server state from a separate process.
_STATUS_DIR = Path.home() / ".gdskg"
_STATUS_FILE = _STATUS_DIR / "server_status"


def _ensure_status_dir() -> None:
    """
    Ensure the directory for storing server status files exists.
    """
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)



def _read_server_status() -> bool:
    """Read the server status from the file. Returns True if started, False if stopped."""
    try:
        if _STATUS_FILE.exists():
            return _STATUS_FILE.read_text().strip() == "started"
    except Exception:
        pass
    return True  # Default to started if file missing or unreadable


def _write_server_status(started: bool) -> None:
    """
    Persist the server's lifecycle state to a status file.

    Args:
        started (bool): True if the server should be marked as active, False otherwise.
    """
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
    """
    Dynamically determine the root directory of the GDSKG project.

    Returns:
        Path: The absolute path to the project root.
    """
    # In Docker: /app/mcp_server/server.py -> /app
    # Local: gdskg/mcp_server/server.py -> gdskg
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "gdskg_graph"


# Ensure potential import paths are available
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "gdskg"))


# --- Knowledge Graph Tools ---

@mcp.tool()
def query_knowledge_graph(
    query: str,
    graph_path: str = str(DEFAULT_GRAPH_PATH),
    limit: int = 10,
    depth: int = 1,
    traverse_types: Optional[List[str]] = None,
    repo_name: Optional[str] = None,
) -> str:
    """
    Query the Git-Derived Software Knowledge Graph using natural language or keywords.
    
    Args:
        query: The search query string.
        graph_path: Path to the knowledge graph SQLite database directory. Defaults to './gdskg_graph'.
        limit: Maximum number of results to return.
        depth: Traversal depth for finding related nodes (0=direct matches only, 1=immediate neighbors, 2+=deeper). Default 1.
        traverse_types: Optional list of node types to traverse through (e.g., ['File', 'Symbol']). Restricts which relationship types are followed during depth search.
        repo_name: Optional repository name to scope the search to.
        
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
        from analysis.search_engine import SearchEngine
        from core.schema import NodeType

        searcher = SearchEngine(str(db_path))
        results = searcher.search(query, repo_name=repo_name, depth=depth, traverse_types=traverse_types)
        
        if not results:
            return "No relevant matches found."

        output = f"Search Results for: '{query}'\n\n"

        
        for i, res in enumerate(results[:limit]):
            output += f"{i+1}. [Relevance: {res['relevance']:.2f}] Commit: {res['id'][:8]}\n"
            output += f"   Message: {res['message'].split('\n')[0]}\n"
            author_info = f"   Author: {res['author']} | Date: {res['date'][:10]}\n"
            output += author_info
            
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
def index_repository(
    repository: str, 
    graph_path: str = str(DEFAULT_GRAPH_PATH), 
    overwrite: bool = False,
    plugins: list[str] = None,
    parameters: list[str] = None
) -> str:
    """
    Index a git repository (local folder or remote URL) into the knowledge graph.
    
    Args:
        repository: Path or URL to the git repository to analyze.
        graph_path: Directory where the knowledge graph (SQLite db) will be stored. Defaults to './gdskg_graph'.
        overwrite: If True, overwrites the existing graph database.
        plugins: Optional list of plugins to enable (e.g., ['GitHubPR']).
        parameters: Optional list of plugin parameters in format 'PluginName:Key=Value'.
        
    Returns:
        Success message with node count or error message.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."

    repo_path = None
    
    # Check if repository is a URL
    if repository.startswith("http://") or repository.startswith("https://"):
        import subprocess
        
        # Determine local checkout path
        cache_dir = Path.home() / ".gdskg" / "cache"
        repo_name = repository.split("/")[-1].replace(".git", "")
        checkout_path = cache_dir / repo_name

        
        # Build clone URL with token authentication
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        if token and "@" not in repository:
            repo_url = repository.replace("https://", f"https://x-access-token:{token}@")
        else:
            repo_url = repository
        
        if checkout_path.exists():
            try:
                subprocess.run(["git", "-C", str(checkout_path), "pull"], 
                             check=True, capture_output=True, text=True)
            except Exception:
                pass  # Continue with existing checkout
        else:
            checkout_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try with token first
            result = subprocess.run(
                ["git", "clone", repo_url, str(checkout_path)],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                # If token-auth failed, retry without token (public repos)
                import shutil
                if checkout_path.exists():
                    shutil.rmtree(checkout_path)
                result2 = subprocess.run(
                    ["git", "clone", repository, str(checkout_path)],
                    capture_output=True, text=True
                )
                if result2.returncode != 0:
                    return f"Error: Failed to clone repository. stderr: {result2.stderr}"
        
        repo_path = checkout_path
    else:
        # Local path
        repo_path = Path(repository).resolve()
        if not repo_path.exists():
             return f"Error: Repository path '{repo_path}' does not exist."

    if not (repo_path / ".git").exists():
         return f"Error: The specified directory '{repo_path}' is not a git repository."

    graph_dir = Path(graph_path).resolve()

    try:
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

        
        plugin_manager = PluginManager()

        if plugins:
            try:
                plugin_manager.load_plugins(plugins)
            except Exception as e:
                 return f"Error loading plugins: {e}"
        
        loaded_plugins = plugin_manager.get_plugins()
        if plugins and len(loaded_plugins) < len(plugins):
             return f"Error: Only {len(loaded_plugins)}/{len(plugins)} plugins loaded successfully. Please check your plugin names and connectivity."
        
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
                except Exception:
                    pass 
        
        extractor = GraphExtractor(repo_path, store, plugins=loaded_plugins, plugin_config=plugin_config)

        
        extractor.process_repo()
        node_count = store.count_nodes()
        store.close()
        
        vector_store = VectorStore(db_path)
        embedder = ONNXEmbedder()
        embedded_count = vector_store.build_from_graph(str(db_path), embedder)
        vector_store.close()
        
        msg = f"Successfully indexed '{repository}'. Graph built at '{db_path}' with {node_count} nodes. Vector DB stored {embedded_count} semantic embeddings."
        if loaded_plugins:
             msg += f" Plugins enabled: {len(loaded_plugins)}."
        return msg

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error indexing repository: {str(e)}\n\nTraceback:\n{tb}"

@mcp.tool()
def get_function_history(
    function_name: str,
    graph_path: str = str(DEFAULT_GRAPH_PATH),
) -> str:
    """
    Finds the highest relevance function node for a given function name parameter and returns its history.
    This includes all versions of the function over time, showing the text and the commit that created each version.
    
    Args:
        function_name: The name of the function to get history for.
        graph_path: Path to the knowledge graph SQLite database directory. Defaults to './gdskg_graph'.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."
    
    graph_dir = Path(graph_path)
    db_path = graph_dir / "gdskg.db"
    
    if not db_path.exists():
        return f"Error: Database not found at {db_path}. Please index a repository first."

    try:
        import sqlite3
        import json
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, attributes FROM nodes WHERE type IN ('FUNCTION', 'SYMBOL') AND id LIKE ?", (f"%{function_name}%",))
            matches = cursor.fetchall()
            if not matches:
                return f"No function found matching '{function_name}'."
                
            best_match = None
            for match in matches:
                if match[0].endswith(function_name):
                    if not best_match or len(match[0]) < len(best_match[0]):
                        best_match = match
                        
            if not best_match:
                best_match = min(matches, key=lambda x: len(x[0]))
                
            func_id = best_match[0]
            
            cursor.execute("SELECT target_id FROM edges WHERE source_id=? AND type='HAS_HISTORY'", (func_id,))
            history_rows = cursor.fetchall()
            if not history_rows:
                return f"Function '{func_id}' has no recorded history."
                
            output = f"Histories found for function: {func_id}\n\n"
            
            for h_idx, (history_id,) in enumerate(history_rows):
                cursor.execute("""
                    SELECT n.id, n.attributes 
                    FROM edges e 
                    JOIN nodes n ON e.target_id = n.id 
                    WHERE e.source_id=? AND e.type='HAS_VERSION'
                """, (history_id,))
                version_rows = cursor.fetchall()
                
                if not version_rows:
                    continue
                
                version_map = {row[0]: json.loads(row[1]) for row in version_rows}
                
                placeholders = ",".join(["?"]*len(version_map))
                cursor.execute(f"""
                    SELECT source_id, target_id
                    FROM edges
                    WHERE type='PREVIOUS_VERSION' AND source_id IN ({placeholders}) AND target_id IN ({placeholders})
                """, list(version_map.keys()) * 2)
                
                prev_edges = cursor.fetchall()
                prev_to_next = {src: tgt for src, tgt in prev_edges}
                next_to_prev = {tgt: src for src, tgt in prev_edges}
                
                roots = [vid for vid in version_map.keys() if vid not in next_to_prev]
                
                ordered_versions = []
                visited = set()
                for root in roots:
                    curr = root
                    while curr and curr not in visited:
                        ordered_versions.append(curr)
                        visited.add(curr)
                        curr = prev_to_next.get(curr)
                        
                if not ordered_versions:
                    continue
                    
                last_v_attr = version_map[ordered_versions[-1]]
                latest_path = last_v_attr.get('file_path', 'Unknown file')
                
                output += f"--- History Instance {h_idx+1} (Latest Path: {latest_path}) ---\n"
                
                for i, vid in enumerate(ordered_versions):
                    v_attr = version_map[vid]
                    commit_id = v_attr.get('commit_id', 'Unknown')
                    
                    cursor.execute("SELECT attributes FROM nodes WHERE id=?", (commit_id,))
                    commit_row = cursor.fetchone()
                    commit_msg = "Unknown"
                    if commit_row:
                        commit_attr = json.loads(commit_row[0])
                        commit_msg = commit_attr.get('message', '').split('\n')[0]
                    
                    v_file = v_attr.get('file_path', latest_path)
                    
                    output += f"Version {i+1} (Commit: {commit_id[:8]} - File: {v_file})\n"
                    output += f"Message: {commit_msg}\n"
                    output += "-" * 40 + "\n"
                    output += f"{v_attr.get('content', '')}\n\n"
            
            return output
    except Exception as e:
         return f"Error retrieving function history: {str(e)}"

if __name__ == "__main__":
    import multiprocessing
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import RedirectResponse
    from starlette.routing import Route, Mount
    
    multiprocessing.freeze_support()
    
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
    
    print("Starting GDSKG MCP Server (Robust Direct)")
    uvicorn.run(app, host="0.0.0.0", port=8015, log_level="info")

import os
import collections
import sys
import sqlite3
import json
import subprocess
import shutil
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP
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

mcp = FastMCP("gdskg")

_STATUS_DIR = Path.home() / ".gdskg"
_STATUS_FILE = _STATUS_DIR / "server_status"

def get_project_root() -> Path:
    """
    Dynamically determine the root directory of the GDSKG project.

    Returns:
        Path: The absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "gdskg_graph"

# Ensure potential import paths are available
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "gdskg"))

def _ensure_status_dir() -> None:
    """
    Ensure the ~/.gdskg directory exists for status tracking.

    Returns:
        None
    """
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)

def _read_server_status() -> bool:
    """
    Read the server status from the file.

    Returns:
        bool: True if the status is 'started', False otherwise.
    """
    try:
        if _STATUS_FILE.exists():
            return _STATUS_FILE.read_text().strip() == "started"
    except Exception:
        pass
    return True

def _write_server_status(started: bool) -> None:
    """
    Write the current server status to the filesystem.

    Args:
        started (bool): Whether the server status should be 'started'.

    Returns:
        None
    """
    _ensure_status_dir()
    _STATUS_FILE.write_text("started" if started else "stopped")

@mcp.tool()
def start_server() -> str:
    """
    Start the GDSKG server.

    Returns:
        str: Confirmation message.
    """
    _write_server_status(True)
    return "Server started. Ready to handle requests."

@mcp.tool()
def stop_server() -> str:
    """
    Stop the GDSKG server.

    Returns:
        str: Confirmation message.
    """
    _write_server_status(False)
    return "Server stopped."

@mcp.tool()
def get_server_status() -> str:
    """
    Get the current status of the GDSKG server.

    Returns:
        str: A message describing the current status.
    """
    started = _read_server_status()
    return f"Server is {'started' if started else 'stopped'}."

_write_server_status(True)

@mcp.tool()
def query_knowledge_graph(
    query: str,
    graph_path: str = str(DEFAULT_GRAPH_PATH),
    limit: int = 10,
    depth: int = 2,
    traverse_types: Optional[List[str]] = None,
    repo_name: Optional[str] = None,
    semantic_only: bool = False,
    min_score: float = 0.0,
    top_n: int = 5,
    plugins: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
    filters: Optional[List[str]] = None,
) -> str:
    """
    Query the Git-Derived Software Knowledge Graph.

    Args:
        query (str): The search query.
        graph_path (str, optional): The path to the graph directory. Defaults to DEFAULT_GRAPH_PATH.
        limit (int, optional): The maximum number of results to display. Defaults to 10.
        depth (int, optional): The traversal depth for connected nodes. Defaults to 2.
        traverse_types (List[str], optional): Node types to follow during traversal. Defaults to None.
        repo_name (str, optional): Repository name to filter by. Defaults to None.
        semantic_only (bool, optional): If True, only use vector search. Defaults to False.
        min_score (float, optional): The minimum relevance score. Defaults to 0.0.
        top_n (int, optional): The number of top commits for traversal. Defaults to 5.
        plugins (List[str], optional): Runtime plugins to execute. Defaults to None.
        parameters (List[str], optional): Plugin-specific parameters. Defaults to None.
        filters (List[str], optional): Metadata filters in 'Type:Value' format. Defaults to None.

    Returns:
        str: A formatted string containing the search results or an error message.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."

    db_path = Path(graph_path) / "gdskg.db"
    if not db_path.exists():
        return f"Warning: Knowledge Graph Database not found at '{db_path}'. Use `index_repository` first."

    try:
        from analysis.search_engine import SearchEngine
        parsed_filters = _parse_filters(filters)
        searcher = SearchEngine(str(db_path))
        
        results = searcher.search(query, repo_name=repo_name, depth=depth, traverse_types=traverse_types, semantic_only=semantic_only, min_score=min_score, top_n=top_n, filters=parsed_filters)
        
        if plugins and results:
            commit_ids = [res['id'] for res in results[:limit]]
            run_runtime_plugins(str(db_path), commit_ids, plugins, parameters)
            results = searcher.search(query, repo_name=repo_name, depth=depth, traverse_types=traverse_types, semantic_only=semantic_only, min_score=min_score, top_n=top_n)

        if not results:
            return "No relevant matches found."

        return _format_query_results(query, results, limit)
    except Exception as e:
        return f"Error querying graph: {str(e)}"

def _parse_filters(filters: Optional[List[str]]) -> Dict[str, str]:
    """
    Parse metadata filters from strings like 'Type:Value'.

    Args:
        filters (List[str], optional): A list of filter strings.

    Returns:
        Dict[str, str]: A dictionary of parsed filters.
    """
    parsed = {}
    if filters:
        for f in filters:
            if ":" in f:
                ftype, fval = f.split(":", 1)
                parsed[ftype.upper()] = fval
    return parsed

def _format_query_results(query: str, results: List[Dict], limit: int) -> str:
    """
    Format search results into a human-readable string for the MCP client.

    Args:
        query (str): The original search query.
        results (List[Dict]): The list of search results.
        limit (int): The maximum number of results to format.

    Returns:
        str: The formatted results string.
    """
    output = f"Search Results for: '{query}'\n\n"
    for i, res in enumerate(results[:limit]):
        output += f"{i+1}. [Relevance: {res['relevance']:.2f}] Commit: {res['id'][:8]}\n"
        output += f"   Message: {res['message'].split('\n')[0]}\n"
        output += f"   Author: {res['author']} | Date: {res['date'][:10]}\n"
        
        connections = res.get('connections', {})
        if connections:
            output += "   Related:\n"
            grouped = collections.defaultdict(list)
            for node_id, info in connections.items():
                grouped[info['type']].append(node_id)
            
            for ntype, nodes in grouped.items():
                display = [Path(n).name if ntype == NodeType.FILE.value else n for n in nodes]
                if ntype == NodeType.KEYWORD.value:
                    output += f"     - {ntype}: {', '.join(display[:3])}"
                    if len(display) > 3: output += f" (+{len(display)-3} more)"
                else:
                    output += f"     - {ntype}: {', '.join(display)}"
                output += "\n"
        output += "\n"
    return output

@mcp.tool()
def index_repository(
    repository: str, 
    graph_path: str = str(DEFAULT_GRAPH_PATH), 
    overwrite: bool = False,
    plugins: list[str] = None,
    parameters: list[str] = None,
    include_bots: bool = False
) -> str:
    """
    Index a git repository into the knowledge graph.

    Args:
        repository (str): The URL or local path to the repository.
        graph_path (str, optional): The path to the graph directory. Defaults to DEFAULT_GRAPH_PATH.
        overwrite (bool, optional): If True, overwrite the existing database. Defaults to False.
        plugins (List[str], optional): Plugins to run during indexing. Defaults to None.
        parameters (List[str], optional): Plugin-specific parameters. Defaults to None.
        include_bots (bool, optional): If True, include bot-authored commits. Defaults to False.

    Returns:
        str: A summary message of the indexing results.
    """
    if not _read_server_status():
        return "Server is stopped."

    try:
        repo_path = _prepare_repo(repository)
        db_path = _prepare_db(graph_path, overwrite)
        
        store = GraphStore(db_path)
        loaded_plugins = _load_and_validate_plugins(plugins)
        plugin_config = _parse_plugin_params(parameters)
        
        extractor = GraphExtractor(repo_path, store, plugins=loaded_plugins, plugin_config=plugin_config, skip_bots=not include_bots)
        extractor.process_repo()
        node_count = store.count_nodes()
        store.close()
        
        vstore = VectorStore()
        embedder = ONNXEmbedder()
        embedded_count = vstore.build_from_graph(str(db_path), embedder)
        vstore.close()
        
        msg = f"Successfully indexed '{repository}'. Graph built at {db_path}. {node_count} nodes, {embedded_count} embeddings."
        if loaded_plugins: msg += f" Plugins enabled: {len(loaded_plugins)}."
        return msg
    except Exception as e:
        return str(e) or f"Error indexing repository: {traceback.format_exc()}"

def _prepare_repo(repository: str) -> Path:
    """
    Clones or pulls a repository if it's a URL, otherwise validates the local path.

    Args:
        repository (str): The repository name, local path, or URL.

    Returns:
        Path: The local path to the repository.
    """
    if not repository.startswith(("http://", "https://")):
        path = Path(repository).resolve()
        if not path.exists(): raise ValueError(f"Repo path '{path}' does not exist.")
        if not (path / ".git").exists(): raise ValueError(f"'{path}' is not a git repo.")
        return path

    cache_dir = Path.home() / ".gdskg" / "cache"
    repo_name = repository.split("/")[-1].replace(".git", "")
    checkout_path = cache_dir / repo_name
    
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    repo_url = repository.replace("https://", f"https://x-access-token:{token}@") if token and "@" not in repository else repository
    
    if checkout_path.exists():
        subprocess.run(["git", "-C", str(checkout_path), "pull"], capture_output=True)
    else:
        checkout_path.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["git", "clone", repo_url, str(checkout_path)], capture_output=True, text=True)
        if res.returncode != 0:
            res2 = subprocess.run(["git", "clone", repository, str(checkout_path)], capture_output=True, text=True)
            if res2.returncode != 0: raise RuntimeError(f"Clone failed: {res2.stderr}")
    return checkout_path

def _prepare_db(graph_path: str, overwrite: bool) -> Path:
    """
    Ensures the graph directory exists and handles database overwriting.

    Args:
        graph_path (str): The path to the graph directory.
        overwrite (bool): Whether to overwrite the existing database.

    Returns:
        Path: The path to the SQLite database file.
    """
    graph_dir = Path(graph_path).resolve()
    graph_dir.mkdir(parents=True, exist_ok=True)
    db_path = graph_dir / "gdskg.db"
    if db_path.exists() and overwrite: db_path.unlink()
    return db_path

def _load_and_validate_plugins(plugins: Optional[List[str]]) -> List[Any]:
    """
    Load requested plugins and ensure they are valid.

    Args:
        plugins (List[str], optional): A list of plugin names.

    Returns:
        List[Any]: A list of loaded plugin instances.
    """
    if not plugins: return []
    manager = PluginManager()
    manager.load_plugins(plugins)
    loaded_plugins = manager.get_plugins()
    if plugins and len(loaded_plugins) < len(plugins):
        raise ValueError(f"Error: Only {len(loaded_plugins)}/{len(plugins)} plugins loaded successfully.")
    return loaded_plugins

def _parse_plugin_params(params: Optional[List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Parse plugin parameters from a list of strings.

    Args:
        params (List[str], optional): A list of parameter strings ('Plugin:Key=Val').

    Returns:
        Dict[str, Dict[str, str]]: A nested dictionary of configurations.
    """
    config = {}
    if params:
        for p in params:
            if ":" in p and "=" in p:
                name, rest = p.split(":", 1)
                key, val = rest.split("=", 1)
                if name not in config: config[name] = {}
                config[name][key] = val
    return config

@mcp.tool()
def get_function_history(
    function_name: str,
    graph_path: str = str(DEFAULT_GRAPH_PATH),
    plugins: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
) -> str:
    """
    Finds a function node's history across the codebase.

    Args:
        function_name (str): The name of the function.
        graph_path (str, optional): Path to the graph directory. Defaults to DEFAULT_GRAPH_PATH.
        plugins (List[str], optional): Runtime plugins to execute. Defaults to None.
        parameters (List[str], optional): Plugin-specific parameters. Defaults to None.

    Returns:
        str: A formatted history of the function versions.
    """
    if not _read_server_status(): return "Server stopped."
    
    db_path = Path(graph_path) / "gdskg.db"
    if not db_path.exists(): return "Index repo first."

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            func_id = _find_best_function_match(cursor, function_name)
            if not func_id: return f"No function matching '{function_name}'."
            
            history_rows = cursor.execute("SELECT target_id FROM edges WHERE source_id=? AND type='HAS_HISTORY'", (func_id,)).fetchall()
            if not history_rows: return f"'{func_id}' has no history."
            
            output = f"History for: {func_id}\n\n"
            for h_idx, (hid,) in enumerate(history_rows):
                output += _format_history_instance(cursor, hid, str(db_path), plugins, parameters)
            return output
    except Exception as e:
        return f"Error: {str(e)}"

def _find_best_function_match(cursor, name: str) -> Optional[str]:
    """
    Find the best matching function node ID for a given name.

    Args:
        cursor (sqlite3.Cursor): The graph database cursor.
        name (str): The function name or partial ID.

    Returns:
        Optional[str]: The best matching node ID, or None if no match found.
    """
    cursor.execute("SELECT id FROM nodes WHERE type IN ('FUNCTION', 'SYMBOL') AND id LIKE ?", (f"%{name}%",))
    matches = [m[0] for m in cursor.fetchall()]
    if not matches: return None
    
    exact = [m for m in matches if m.endswith(name)]
    return min(exact if exact else matches, key=len)

def _format_history_instance(cursor: sqlite3.Cursor, history_id: str, db_path: str, plugins: Optional[List[str]], params: Optional[List[str]]) -> str:
    """
    Format a specific function history instance into a readable string.

    Args:
        cursor (sqlite3.Cursor): The graph database cursor.
        history_id (str): The identifier of the HISTORY node.
        db_path (str): The path to the database file.
        plugins (List[str], optional): Runtime plugins to run.
        params (List[str], optional): Plugin parameters.

    Returns:
        str: The formatted history instance output.
    """
    versions = cursor.execute("SELECT n.id, n.attributes FROM edges e JOIN nodes n ON e.target_id = n.id WHERE e.source_id=? AND e.type='HAS_VERSION'", (history_id,)).fetchall()
    if not versions: return ""
    
    v_map = {v[0]: json.loads(v[1]) for v in versions}
    ordered_vids = _order_function_versions(cursor, list(v_map.keys()))
            
    if plugins:
        cids = [v_map[v].get('commit_id') for v in ordered_vids if v_map[v].get('commit_id') not in (None, 'Unknown')]
        if cids: run_runtime_plugins(db_path, cids, plugins, params)

    return _generate_history_output(cursor, ordered_vids, v_map)

def _order_function_versions(cursor: sqlite3.Cursor, v_ids: List[str]) -> List[str]:
    """
    Order function versions based on PREVIOUS_VERSION edges.

    Args:
        cursor (sqlite3.Cursor): The graph database cursor.
        v_ids (List[str]): List of version node IDs.

    Returns:
        List[str]: The ordered list of version node IDs.
    """
    if not v_ids: return []
    placeholders = ",".join(["?"] * len(v_ids))
    prev_edges = cursor.execute(f"SELECT source_id, target_id FROM edges WHERE type='PREVIOUS_VERSION' AND source_id IN ({placeholders}) AND target_id IN ({placeholders})", v_ids + v_ids).fetchall()
    
    p2n = {s: t for s, t in prev_edges}
    n2p = {t: s for s, t in prev_edges}
    roots = [vid for vid in v_ids if vid not in n2p]
    
    ordered = []
    for r in roots:
        curr = r
        while curr and curr not in ordered:
            ordered.append(curr)
            curr = p2n.get(curr)
    return ordered

def _generate_history_output(cursor: sqlite3.Cursor, ordered_vids: List[str], v_map: Dict[str, Any]) -> str:
    """
    Generate the final readable string for function history.

    Args:
        cursor (sqlite3.Cursor): The graph database cursor.
        ordered_vids (List[str]): The ordered version node IDs.
        v_map (Dict[str, Any]): A mapping of version IDs to their metadata.

    Returns:
        str: The generated output string.
    """
    if not ordered_vids: return ""
    output = f"--- Instance (File: {v_map[ordered_vids[-1]].get('file_path', 'Unknown')}) ---\n"
    for i, vid in enumerate(ordered_vids):
        v = v_map[vid]
        cid = v.get('commit_id', 'Unknown')
        c_row = cursor.execute("SELECT attributes FROM nodes WHERE id=?", (cid,)).fetchone()
        msg = json.loads(c_row[0]).get('message', '').split('\n')[0] if c_row else "Unknown"
        output += f"V{i+1} ({cid[:8]}): {msg}\n" + "-"*40 + f"\n{v.get('content', '')}\n\n"
    return output

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    mcp_sse_app = mcp.sse_app()
    app = Starlette(routes=[
        Route("/", lambda r: RedirectResponse(url="/mcp"), methods=["GET"]),
        Route("/", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
        Route("/mcp", lambda r: RedirectResponse(url="/mcp/messages", status_code=307), methods=["POST"]),
        Mount("/", mcp_sse_app)
    ])
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    print("Starting GDSKG MCP Server", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=8015, log_level="info")


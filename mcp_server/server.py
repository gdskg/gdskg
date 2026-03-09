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
DEFAULT_GRAPH_PATH = Path(os.environ.get("GDSKG_GRAPH_DB_DIR", Path.home() / ".gdskg" / "graph_db"))

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
def query_knowledge_graph(
    query: str,
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
    excluded_commits: Optional[List[str]] = None,
    offset: int = 0,
    negative_query: str = "",
    exclude_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Query the Git-Derived Software Knowledge Graph.

    Args:
        query (str): The search query.
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
        excluded_commits (List[str], optional): List of commit IDs to ignore. Defaults to None.
        offset (int, optional): The number of results to skip. Defaults to 0.
        negative_query (str, optional): Terms or meanings to avoid in results. Defaults to "".

    Returns:
        Dict[str, Any]: A dictionary containing a human-readable summary and hydrated search results.
    """
    if not _read_server_status():
        return {"error": "Server is stopped. Please start the server first."}

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists():
        return {"warning": f"Knowledge Graph Database not found at '{db_path}'. Use `index_repository` first."}

    try:
        from analysis.search_engine import SearchEngine
        parsed_filters = _parse_filters(filters)
        searcher = SearchEngine(str(db_path))
        
        results = searcher.search(
            query, 
            repo_name=repo_name, 
            depth=depth, 
            traverse_types=traverse_types, 
            semantic_only=semantic_only, 
            min_score=min_score, 
            top_n=top_n, 
            filters=parsed_filters,
            excluded_commits=excluded_commits,
            offset=offset,
            negative_query=negative_query,
            exclude_types=exclude_types
        )
        
        if plugins and results:
            commit_ids = [res['id'] for res in results[:limit]]
            run_runtime_plugins(str(db_path), commit_ids, plugins, parameters)
            results = searcher.search(
                query, 
                repo_name=repo_name, 
                depth=depth, 
                traverse_types=traverse_types, 
                semantic_only=semantic_only, 
                min_score=min_score, 
                top_n=top_n,
                excluded_commits=excluded_commits,
                offset=offset,
                negative_query=negative_query,
                exclude_types=exclude_types
            )

        if not results:
            return {"message": "No relevant matches found.", "results": []}

        formatted_text = _format_query_results(query, results, limit)
        
        # Hydrate results with raw metadata for the AI
        hydrated_results = []
        for res in results[:limit]:
            hydrated_res = {
                "id": res["id"],
                "type": NodeType.COMMIT.value,
                "relevance": res["relevance"],
                "message": res["message"],
                "author": res["author"],
                "date": res["date"],
                "raw_metadata": {
                    "commit_id": res["id"],
                    "related_nodes": []
                }
            }
            
            for node_id, info in res.get('connections', {}).items():
                hydrated_res["raw_metadata"]["related_nodes"].append({
                    "id": node_id,
                    "type": info["type"],
                    "attributes": info.get("attributes", {})
                })
            hydrated_results.append(hydrated_res)

        return {
            "description": formatted_text,
            "results": hydrated_results
        }
    except Exception as e:
        return {"error": f"Error querying graph: {str(e)}"}

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
        output += f"   Message: {res['message']}\n"
        output += f"   Author: {res['author']} | Date: {res['date'][:10]}\n"
        
        connections = res.get('connections', {})
        if connections:
            output += "   Related:\n"
            grouped = collections.defaultdict(list)
            for node_id, info in connections.items():
                grouped[info['type']].append((node_id, info.get('attributes', {})))
            
            for ntype, nodes in grouped.items():
                display = []
                for node_id, attrs in nodes:
                    if ntype == NodeType.FILE.value:
                        display.append(f"{Path(node_id).name} [AST Node ID: {node_id}]")
                    elif ntype in (NodeType.FUNCTION.value, NodeType.FUNCTION_VERSION.value):
                        display.append(f"{node_id} [AST Node ID: {node_id}]")
                    elif ntype in (NodeType.COMMENT.value, NodeType.COMMIT_MESSAGE.value):
                        display.append(attrs.get('content', node_id).strip())
                    elif ntype == NodeType.PULL_REQUEST.value:
                        num = attrs.get('number', '?')
                        title = attrs.get('title', 'PR')
                        display.append(f"PR #{num}: {title}")
                    elif ntype == NodeType.CLICKUP_TASK.value:
                        name = attrs.get('name', node_id)
                        display.append(f"Task: {name}")
                    else:
                        display.append(node_id)
                
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
    overwrite: bool = False,
    plugins: list[str] = None,
    parameters: list[str] = None,
    include_bots: bool = False
) -> str:
    """
    Index a git repository into the knowledge graph.

    Args:
        repository (str): The URL or local path to the repository.
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
        repos = [r.strip() for r in repository.split(",")]
        db_path = _prepare_db(str(DEFAULT_GRAPH_PATH), overwrite)
        store = GraphStore(db_path)
        loaded_plugins = _load_and_validate_plugins(plugins)
        plugin_config = _parse_plugin_params(parameters)
        
        for repo_url in repos:
            repo_path = _prepare_repo(repo_url)
            extractor = GraphExtractor(repo_path, store, plugins=loaded_plugins, plugin_config=plugin_config, skip_bots=not include_bots)
            extractor.process_repo()
            
        node_count = store.count_nodes()
        store.close()
        
        vstore = VectorStore()
        embedder = ONNXEmbedder()
        embedded_count = vstore.build_from_graph(str(db_path), embedder)
        vstore.close()
        
        msg = f"Successfully indexed {len(repos)} repositories. Graph built at {db_path}. {node_count} nodes, {embedded_count} embeddings."
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
    plugins: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
) -> str:
    """
    Finds a function node's history across the codebase.

    Args:
        function_name (str): The name of the function.
        plugins (List[str], optional): Runtime plugins to execute. Defaults to None.
        parameters (List[str], optional): Plugin-specific parameters. Defaults to None.

    Returns:
        str: A formatted history of the function versions.
    """
    if not _read_server_status(): return "Server stopped."
    
    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists(): return "Index repo first."

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            func_id, error_msg = _find_best_function_match(cursor, function_name)
            if error_msg: return error_msg
            if not func_id: return f"No function matching '{function_name}'."
            
            history_rows = cursor.execute("SELECT target_id FROM edges WHERE source_id=? AND type='HAS_HISTORY'", (func_id,)).fetchall()
            if not history_rows: return f"'{func_id}' has no history."
            
            output = f"History for: {func_id}\n\n"
            for h_idx, (hid,) in enumerate(history_rows):
                output += _format_history_instance(cursor, hid, str(db_path), plugins, parameters)
            return output
    except Exception as e:
        return f"Error: {str(e)}"

def _find_best_function_match(cursor, name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the best matching function node ID for a given name. Returns a tuple of (node_id, error_message).
    If multiple matches are found, returns an error message with options.

    Args:
        cursor (sqlite3.Cursor): The graph database cursor.
        name (str): The function name or partial ID.

    Returns:
        Tuple[Optional[str], Optional[str]]: The matched node ID or an error message.
    """
    cursor.execute("SELECT id FROM nodes WHERE type IN ('FUNCTION', 'SYMBOL') AND id LIKE ?", (f"%{name}%",))
    matches = [m[0] for m in cursor.fetchall()]
    if not matches: return None, f"No function matching '{name}'. Please check spelling or query for AST node IDs in broader searches."
    
    exact = [m for m in matches if m.endswith(name)]
    if exact and len(exact) == 1:
        return exact[0], None

    if len(matches) == 1:
        return matches[0], None

    options = exact if exact else matches
    msg = f"Multiple matches found for '{name}'. Did you mean:\n"
    for i, opt in enumerate(options[:10]):
        msg += f"[{i+1}] {opt}\n"
    if len(options) > 10:
        msg += f"... and {len(options) - 10} more matches.\n"
    msg += "Please query again using one of the exact function IDs above."
    
    return None, msg

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
        msg = json.loads(c_row[0]).get('message', 'Unknown') if c_row else "Unknown"
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

@mcp.tool()
def get_ast_nodes(
    node_id: str,
    max_depth: int = 0,
    query: Optional[str] = None
) -> str:
    """
    Retrieve the AST nodes linked to a given FILE or FUNCTION_VERSION node.
    
    Args:
        node_id (str): The ID of the file or function version.
        max_depth (int, optional): Maximum depth to traverse the AST (0 for infinite).
        query (str, optional): Optional semantic query to filter AST nodes by similarity.
        
    Returns:
        str: A formatted string of the AST nodes found.
    """
    if not _read_server_status():
        return "Server is stopped. Please start the server first."

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists():
        return f"Warning: Knowledge Graph Database not found at '{db_path}'. Use `index_repository` first."

    try:
        from analysis.search_engine import SearchEngine
        searcher = SearchEngine(str(db_path))
        
        try:
            nodes = searcher.get_ast_nodes(node_id, max_depth, query=query)
        except ValueError as ve:
            return f"Error: {ve}"
        except Exception as e:
            return f"Error querying graph: {str(e)}"
            
        if not nodes:
            return f"No AST nodes found linked from {node_id}."

        output = f"Found {len(nodes)} AST nodes linked from {node_id}"
        if query:
            output += f" matching '{query}'"
        output += ":\n\n"

        for node in nodes:
            safe_text = node['attributes'].get('text', '').replace('\n', ' ')
            if safe_text:
                safe_text = f" -> '{safe_text}'"
            sim_str = f" [score: {node.get('similarity', 0):.2f}]" if query else ""
            output += f" - {node['id']} ({node['attributes'].get('ast_type', 'unknown')}){safe_text}{sim_str}\n"

        return output
    except Exception as e:
        return f"Error querying graph: {str(e)}"

@mcp.tool()
def get_dependencies(
    node_id: str
) -> Dict[str, Any]:
    """
    Retrieve what a node uses (dependencies) and what uses it (dependents).
    
    Args:
        node_id (str): The ID of the node (repo, file, function, symbol). 
                      Supports fuzzy path resolution (e.g., 'AccountCard.tsx').
                      
    Returns:
        Dict[str, Any]: A map of dependencies and dependents.
    """
    if not _read_server_status():
        return {"error": "Server is stopped."}

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists():
        return {"error": "Index repo first."}

    try:
        from analysis.search_engine import SearchEngine
        searcher = SearchEngine(str(db_path))
        return searcher.get_dependencies(node_id)
    except Exception as e:
        return {"error": f"Error retrieving dependencies: {str(e)}"}

@mcp.tool()
def list_node_types() -> Dict[str, Any]:
    """
    Retrieve all unique node types currently present in the knowledge graph.
    
    Returns:
        Dict[str, Any]: A list of node types.
    """
    if not _read_server_status():
        return {"error": "Server is stopped."}

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists():
        return {"error": "Index repo first."}

    try:
        from analysis.search_engine import SearchEngine
        searcher = SearchEngine(str(db_path))
        types = searcher.get_node_types()
        return {"node_types": types}
    except Exception as e:
        return {"error": f"Error retrieving node types: {str(e)}"}

@mcp.tool()
def list_edge_types() -> Dict[str, Any]:
    """
    Retrieve all unique edge types currently present in the knowledge graph.
    
    Returns:
        Dict[str, Any]: A list of edge types.
    """
    if not _read_server_status():
        return {"error": "Server is stopped."}

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    if not db_path.exists():
        return {"error": "Index repo first."}

    try:
        from analysis.search_engine import SearchEngine
        searcher = SearchEngine(str(db_path))
        types = searcher.get_edge_types()
        return {"edge_types": types}
    except Exception as e:
        return {"error": f"Error retrieving edge types: {str(e)}"}

@mcp.tool()
def amend_graph(
    action: str,
    entity_type: str,
    data: str
) -> Dict[str, Any]:
    """
    Apply an amendment (correction or addition) to the knowledge graph.
    
    Args:
        action (str): The action to perform (e.g., 'add_node', 'update_node', 'add_edge', 'update_edge').
        entity_type (str): The type of entity being amended (e.g., node type or edge type).
        data (str): A JSON string containing the data for the amendment.
                   For nodes: {"id": "node_id", "attributes": {"key": "value"}}
                   For edges: {"source_id": "src", "target_id": "dst", "attributes": {"key": "value"}}
                   
    Returns:
        Dict[str, Any]: A success message or an error.
    """
    if not _read_server_status():
        return {"error": "Server is stopped."}

    db_path = DEFAULT_GRAPH_PATH / "gdskg.db"
    
    try:
        from core.graph_store import GraphStore
        from core.schema import Node, Edge, NodeType, EdgeType
        import json
        
        parsed_data = json.loads(data)
        store = GraphStore(db_path)
        
        # Ensure type is safely assigned even if not in Enum
        try:
            enum_type = NodeType(entity_type)
            stored_type = enum_type.value
        except ValueError:
            try:
                enum_type = EdgeType(entity_type)
                stored_type = enum_type.value
            except ValueError:
                stored_type = entity_type

        if action in ("add_node", "update_node"):
            node_id = parsed_data.get("id")
            if not node_id:
                return {"error": "Missing 'id' in data for node."}
            attributes = parsed_data.get("attributes", {})
            
            node = Node(id=node_id, type=stored_type, attributes=attributes)
            store.upsert_node(node)
            store.flush()
            store.log_amendment(action, stored_type, parsed_data)
            
        elif action in ("add_edge", "update_edge"):
            source_id = parsed_data.get("source_id")
            target_id = parsed_data.get("target_id")
            if not source_id or not target_id:
                return {"error": "Missing 'source_id' or 'target_id' in data for edge."}
            attributes = parsed_data.get("attributes", {})
            
            edge = Edge(source_id=source_id, target_id=target_id, type=stored_type, attributes=attributes)
            store.upsert_edge(edge)
            store.flush()
            store.log_amendment(action, stored_type, parsed_data)
        else:
            return {"error": f"Unsupported amendment action: {action}"}
            
        store.close()
        return {"message": f"Successfully applied {action} for {entity_type}."}
        
    except Exception as e:
        return {"error": f"Error applying amendment: {str(e)}"}

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import os

# Create the MCP server
mcp = FastMCP("gdskg")

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
    mcp.run()

import os
import sys
import importlib.util
import shutil
import subprocess
import logging
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Type, Any, Optional

from core.plugin_interfaces import PluginInterface
from core.graph_store import GraphStore
from core.extractor import GraphAPIWrapper
from core.schema import Node, Edge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PLUGIN_DIR = Path.home() / ".gdskg" / "plugins"

class PluginManager:
    """Manager for loading, installing, and life-cycle of GDSKG plugins."""

    def __init__(self):
        self.plugins: List[PluginInterface] = []
        self._ensure_plugin_dir()

    def _ensure_plugin_dir(self) -> None:
        if not PLUGIN_DIR.exists():
            PLUGIN_DIR.mkdir(parents=True)

    def load_plugins(self, plugin_names: List[str]) -> None:
        """
        Load multiple plugins by their names.

        Args:
            plugin_names (List[str]): A list of plugin names to load.

        Returns:
            None
        """
        for name in plugin_names:
            try:
                plugin_path = self._get_or_install_plugin(name)
                plugin_instance = self._load_plugin_module(name, plugin_path)
                if plugin_instance:
                    self.plugins.append(plugin_instance)
                    logger.info(f"Loaded plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")

    def _get_or_install_plugin(self, name: str) -> Path:
        """
        Locate a plugin on disk or clone it from GitHub.

        Args:
            name (str): The name of the plugin.

        Returns:
            Path: The local path to the plugin directory.
        """
        project_root = Path(__file__).parent.parent
        local_plugin_dir = project_root / "plugins" / name
        
        if local_plugin_dir.exists():
            logger.info(f"Found local plugin: {name} at {local_plugin_dir}")
            return local_plugin_dir

        target_dir = PLUGIN_DIR / name
        if target_dir.exists():
            return target_dir

        repo_url = f"https://github.com/gdskg/{name}.git"
        logger.info(f"Installing plugin {name} from {repo_url}...")
        
        try:
            subprocess.check_call(["git", "clone", repo_url, str(target_dir)], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone plugin {name}: {e}")
            
        return target_dir

    def _load_plugin_module(self, name: str, path: Path) -> PluginInterface:
        """
        Dynamically load a plugin module from a specific file path.

        Args:
            name (str): The plugin name.
            path (Path): Path to the plugin directory.

        Returns:
            PluginInterface: An instance of the loaded plugin.
        """
        entry_point = path / "plugin.py"
        if not entry_point.exists():
             raise FileNotFoundError(f"Could not find plugin.py in {path}")
             
        spec = importlib.util.spec_from_file_location(f"gdskg_plugin_{name}", entry_point)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec for {name}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"gdskg_plugin_{name}"] = module
        spec.loader.exec_module(module)
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, PluginInterface) and attr is not PluginInterface:
                return attr()
                
        raise ValueError(f"No PluginInterface implementation found in {name}")

    def get_plugins(self) -> List[PluginInterface]:
        """
        Return the list of loaded plugin instances.

        Returns:
            List[PluginInterface]: Current loaded plugins.
        """
        return self.plugins

def run_runtime_plugins(db_path: str, commit_ids: List[str], plugins: List[str], parameters: List[str] = None):
    """
    Load and execute run-time plugins for a specific set of commits.
    
    This function initializes the PluginManager, loads the requested plugins,
    and processes each commit ID by fetching its context and invoking the plugins.

    Args:
        db_path (str): Path to the graph database.
        commit_ids (List[str]): List of SHAs to process.
        plugins (List[str]): List of plugin names to run.
        parameters (List[str], optional): Configuration parameters. Defaults to None.

    Returns:
        None
    """
    plugin_manager = PluginManager()
    try:
        plugin_manager.load_plugins(plugins)
    except Exception as e:
        logger.error(f"Error loading plugins: {e}")
        return

    loaded_plugins = plugin_manager.get_plugins()
    if not loaded_plugins:
        return

    store = GraphStore(Path(db_path))
    api = GraphAPIWrapper(store)
    plugin_config = _parse_plugin_parameters(parameters)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for commit_id in commit_ids:
            _process_single_commit_runtime(cursor, commit_id, loaded_plugins, api, plugin_config)
    
    store.flush()
    store.close()

def _parse_plugin_parameters(parameters: Optional[List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Parse plugin-specific parameters from strings like 'PluginName:Key=Value'.

    Args:
        parameters (List[str], optional): Raw parameter strings.

    Returns:
        Dict[str, Dict[str, str]]: Parsed configuration dictionary.
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
        except Exception:
            pass
    return config

def _process_single_commit_runtime(cursor, commit_id, loaded_plugins, api, plugin_config):
    """
    Fetch context for a commit and execute all loaded plugins on it.

    Args:
        cursor (sqlite3.Cursor): Database cursor.
        commit_id (str): SHA of the commit.
        loaded_plugins (List[PluginInterface]): Plugins to run.
        api (GraphAPIWrapper): Graph API for plugins.
        plugin_config (Dict): Configuration for plugins.
    """
    cursor.execute("SELECT type, attributes FROM nodes WHERE id=?", (commit_id,))
    row = cursor.fetchone()
    if not row:
        return
    commit_node = Node(id=commit_id, type=row[0], attributes=json.loads(row[1]) if row[1] else {})

    related_nodes, related_edges = _fetch_commit_context(cursor, commit_id)

    for plugin in loaded_plugins:
        try:
            name = _get_plugin_config_name(plugin)
            config = plugin_config.get(name, {})
            plugin.process(commit_node, related_nodes, related_edges, api, config)
        except Exception as e:
            logger.error(f"Error executing plugin {plugin}: {e}")

def _fetch_commit_context(cursor, commit_id):
    """
    Retrieve all nodes and edges directly connected to a specific commit.

    Args:
        cursor (sqlite3.Cursor): Database cursor.
        commit_id (str): SHA of the commit.

    Returns:
        Tuple[List[Node], List[Edge]]: Connected nodes and edges.
    """
    related_nodes = []
    related_edges = []
    
    cursor.execute("SELECT target_id, type FROM edges WHERE source_id=?", (commit_id,))
    for tid, etype in cursor.fetchall():
        related_edges.append(Edge(source_id=commit_id, target_id=tid, type=etype))
        _append_node_if_exists(cursor, tid, related_nodes)

    cursor.execute("SELECT source_id, type FROM edges WHERE target_id=?", (commit_id,))
    for sid, etype in cursor.fetchall():
        related_edges.append(Edge(source_id=sid, target_id=commit_id, type=etype))
        _append_node_if_exists(cursor, sid, related_nodes)
        
    return related_nodes, related_edges

def _append_node_if_exists(cursor, node_id, nodes_list):
    """
    Fetch a node's metadata and append it to the list if found in the database.

    Args:
        cursor (sqlite3.Cursor): Database cursor.
        node_id (str): Identifier of the node.
        nodes_list (List[Node]): List to append to.
    """
    cursor.execute("SELECT type, attributes FROM nodes WHERE id=?", (node_id,))
    row = cursor.fetchone()
    if row:
        nodes_list.append(Node(id=node_id, type=row[0], attributes=json.loads(row[1]) if row[1] else {}))

def _get_plugin_config_name(plugin):
    """
    Determine the configuration key name for a given plugin instance.

    Args:
        plugin (PluginInterface): The plugin instance.

    Returns:
        str: The name to use for configuration lookup.
    """
    module_name = plugin.__module__
    if module_name.startswith("gdskg_plugin_"):
        return module_name.replace("gdskg_plugin_", "")
    if "plugins." in module_name:
        parts = module_name.split('.')
        if "plugins" in parts:
            idx = parts.index("plugins")
            if idx + 1 < len(parts):
                return parts[idx+1]
    return module_name


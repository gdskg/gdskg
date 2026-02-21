import os
import sys
import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Type
import logging

from core.plugin_interfaces import PluginInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PLUGIN_DIR = Path.home() / ".gdskg" / "plugins"

class PluginManager:
    """
    Manager for loading, installing, and life-cycle of GDSKG plugins.
    
    Supports loading plugins from a local directory or automatically
    cloning them from a remote GitHub repository.
    """

    def __init__(self):
        """
        Initialize the PluginManager.
        """

        self.plugins: List[PluginInterface] = []
        self._ensure_plugin_dir()

    def _ensure_plugin_dir(self) -> None:
        """
        Ensure the global plugin directory exists in the user's home directory.
        """

        if not PLUGIN_DIR.exists():
            PLUGIN_DIR.mkdir(parents=True)

    def load_plugins(self, plugin_names: List[str]) -> None:
        """
        Load multiple plugins by their names.

        This method attempts to locate the plugin locally, install it if missing,
        and then dynamically load the module into memory.

        Args:
            plugin_names (List[str]): A list of plugin names to load.
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
        Locate a plugin on disk, cloning it from GitHub if not found.

        Checks the project's local `plugins/` folder first, then falls back
        to the global `~/.gdskg/plugins/` directory.

        Args:
            name (str): The name of the plugin (corresponds to the GitHub repo name).

        Returns:
            Path: The file system path to the plugin directory.

        Raises:
            RuntimeError: If cloning from GitHub fails.
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

        Expects a `plugin.py` file within the plugin directory that
        contains a class implementing `PluginInterface`.

        Args:
            name (str): The name used for registration in `sys.modules`.
            path (Path): The directory path containing the plugin.

        Returns:
            PluginInterface: An instance of the loaded plugin class.

        Raises:
            FileNotFoundError: If `plugin.py` is missing.
            ImportError: If the module cannot be loaded.
            ValueError: If no `PluginInterface` implementation is found.
        """

        # Expecting a file named 'plugin.py' or 'main.py' or just definitions in __init__.py
        # Current logic: look for 'plugin.py'
        
        # entry_point = path / "plugin.py"
        # if not entry_point.exists():
        #    entry_point = path / "main.py"
        
        # Better approach: load the directory as a package? 
        # Or Just append to path and import? 
        
        # Let's try adding to sys.path and importing the module name
        # But module name might crash if multiple plugins use same internal struct.
        # So we use importlib to load from path.
        
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
        Get the list of successfully loaded plugin instances.

        Returns:
            List[PluginInterface]: The list of loaded plugins.
        """

        return self.plugins

def run_runtime_plugins(db_path: str, commit_ids: List[str], plugins: List[str], parameters: List[str] = None):
    from core.graph_store import GraphStore
    from core.extractor import GraphAPIWrapper
    from core.schema import Node, Edge
    import sqlite3
    import json

    plugin_manager = PluginManager()
    try:
        plugin_manager.load_plugins(plugins)
    except Exception as e:
        print(f"Error loading plugins: {e}")
        return

    loaded_plugins = plugin_manager.get_plugins()

    if loaded_plugins:
        # store = GraphStore(db_path) expects Path object
        from pathlib import Path
        store = GraphStore(Path(db_path))
        api = GraphAPIWrapper(store)
        plugin_config = {}

        if parameters:
            for param in parameters:
                try:
                    if ":" in param and "=" in param:
                        plugin_part, rest = param.split(":", 1)
                        key, value = rest.split("=", 1)
                        if plugin_part not in plugin_config:
                            plugin_config[plugin_part] = {}
                        plugin_config[plugin_part][key] = value
                except Exception:
                    pass 

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for commit_id in commit_ids:
                cursor.execute("SELECT type, attributes FROM nodes WHERE id=?", (commit_id,))
                row = cursor.fetchone()
                if not row: continue
                commit_node = Node(id=commit_id, type=row[0], attributes=json.loads(row[1]) if row[1] else {})

                cursor.execute("SELECT target_id, type FROM edges WHERE source_id=?", (commit_id,))
                edges_out = cursor.fetchall()
                cursor.execute("SELECT source_id, type FROM edges WHERE target_id=?", (commit_id,))
                edges_in = cursor.fetchall()

                related_nodes = []
                related_edges = []
                for target_id, etype in edges_out:
                    related_edges.append(Edge(source_id=commit_id, target_id=target_id, type=etype))
                    cursor.execute("SELECT type, attributes FROM nodes WHERE id=?", (target_id,))
                    nrow = cursor.fetchone()
                    if nrow:
                        related_nodes.append(Node(id=target_id, type=nrow[0], attributes=json.loads(nrow[1]) if nrow[1] else {}))

                for source_id, etype in edges_in:
                    related_edges.append(Edge(source_id=source_id, target_id=commit_id, type=etype))
                    cursor.execute("SELECT type, attributes FROM nodes WHERE id=?", (source_id,))
                    nrow = cursor.fetchone()
                    if nrow:
                        related_nodes.append(Node(id=source_id, type=nrow[0], attributes=json.loads(nrow[1]) if nrow[1] else {}))

                for plugin in loaded_plugins:
                    try:
                        module_name = plugin.__module__
                        potential_name = module_name
                        if module_name.startswith("gdskg_plugin_"):
                            potential_name = module_name.replace("gdskg_plugin_", "")
                        elif "plugins." in module_name:
                            parts = module_name.split('.')
                            if "plugins" in parts:
                                idx = parts.index("plugins")
                                if idx + 1 < len(parts):
                                    potential_name = parts[idx+1]
                        config = plugin_config.get(potential_name, {})
                        plugin.process(commit_node, related_nodes, related_edges, api, config)
                    except Exception as e:
                        print(f"Error executing plugin {plugin}: {e}")
        
        store.flush()
        store.close()

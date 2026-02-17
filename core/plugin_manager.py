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
    def __init__(self):
        self.plugins: List[PluginInterface] = []
        self._ensure_plugin_dir()

    def _ensure_plugin_dir(self):
        if not PLUGIN_DIR.exists():
            PLUGIN_DIR.mkdir(parents=True)

    def load_plugins(self, plugin_names: List[str]):
        """
        Load plugins by name. 
        1. Checks if plugin exists in local cache.
        2. If not, clones from github.
        3. Loads the module and instantiates the plugin.
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
        # Check local project plugins folder first (Portable)
        project_root = Path(__file__).parent.parent
        local_plugin_dir = project_root / "plugins" / name
        
        if local_plugin_dir.exists():
            logger.info(f"Found local plugin: {name} at {local_plugin_dir}")
            return local_plugin_dir

        target_dir = PLUGIN_DIR / name
        
        # Simple check: if dir exists, assume it's installed. 
        # In a real system, we might want to git pull to update.
        if target_dir.exists():
            # optional: git pull
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
        
        # Find class implementing PluginInterface
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, PluginInterface) and attr is not PluginInterface:
                return attr()
                
        raise ValueError(f"No PluginInterface implementation found in {name}")

    def get_plugins(self) -> List[PluginInterface]:
        return self.plugins

import os
import sys
os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", "/usr/bin/git")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
import git

import re
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.progress import Progress

from core.graph_store import GraphStore
from core.schema import Node, Edge, NodeType, EdgeType
from analysis.symbol_extractor import SymbolExtractor
from analysis.secret_scanner import SecretScanner
from analysis.tree_sitter_utils import TreeSitterUtils
from core.plugin_interfaces import PluginInterface, GraphInterface

class GraphExtractor:
    def __init__(self, repo_path: Path, graph_store: GraphStore, plugins: List[PluginInterface] = None, plugin_config: Dict[str, Dict[str, Any]] = None):
        self.repo_path = repo_path
        self.store = graph_store
        self.repo = git.Repo(repo_path)
        self.symbol_extractor = SymbolExtractor()
        self.secret_scanner = SecretScanner()
        self.plugins = plugins or []
        self.plugin_config = plugin_config or {}
    
    def process_repo(self):
        """
        Main entry point: Iterate through commits and build the graph.
        """
        # Create Repository Node
        repo_node = Node(
            id=str(self.repo_path), # Using path as ID for now, could be remote URL
            type=NodeType.REPOSITORY,
            attributes={
                "name": self.repo_path.name,
                "remotes": [r.url for r in self.repo.remotes]
            }
        )
        self.store.upsert_node(repo_node)

        # Iterate commits in topological order
        commits = list(self.repo.iter_commits(topo_order=True, reverse=True)) # Process oldest to newest
        
        from rich.console import Console
        stderr_console = Console(file=sys.stderr)
        
        with Progress(console=stderr_console) as progress:
            task = progress.add_task("[green]Processing commits...", total=len(commits))
            
            for commit in commits:
                self._process_commit(commit, repo_node)
                progress.advance(task)

    def _process_commit(self, commit: git.Commit, repo_node: Node):
        """
        Process a single commit.
        """
        # Commit Node
        commit_node = Node(
            id=commit.hexsha,
            type=NodeType.COMMIT,
            attributes={
                "message": commit.message,
                "timestamp": commit.authored_datetime.isoformat(),
                "author_name": commit.author.name,
                "author_email": commit.author.email
            }
        )
        self.store.upsert_node(commit_node)
        
        # Commit Message Node
        # ID is a hash of the message to link identical messages
        msg_id = hashlib.sha256(commit.message.encode()).hexdigest()
        msg_node = Node(
            id=f"MSG:{msg_id}",
            type=NodeType.COMMIT_MESSAGE,
            attributes={"content": commit.message}
        )
        self.store.upsert_node(msg_node)
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=msg_node.id,
            type=EdgeType.HAS_MESSAGE
        ))
        
        # Part of Repo Edge
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=repo_node.id,
            type=EdgeType.PART_OF_REPO
        ))

        # Author Node & Edge
        author_id = commit.author.email
        author_node = Node(
            id=author_id,
            type=NodeType.AUTHOR,
            attributes={"name": commit.author.name}
        )
        self.store.upsert_node(author_node)
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=author_node.id,
            type=EdgeType.AUTHORED_BY
        ))

        # Parent Edges (History)
        for parent in commit.parents:
            # We assume parents are already processed due to topological sort, 
            # but we still create edges. If parent doesn't exist in Nodes, it's fine for now (or we create placeholder).
            # Actually, the prompt doesn't explicitly ask for Commit->Commit edges in "Relationship Definitions",
            # but standard git graphs usually have them. The prompt lists:
            # AUTHORED_BY, PART_OF_REPO, OCCURRED_IN, MODIFIED_FILE, MODIFIED_SYMBOL...
            # It DOES NOT list Commit->Commit. I will skip for now to strictly follow spec, 
            # or maybe add it if implied. "Parent SHA(s)" is an attribute of Commit Node.
            pass

        # Time Bucket
        # Extract YYYY or YYYY-MM
        dt = commit.authored_datetime
        time_bucket_id = f"{dt.year}-{dt.month:02d}"
        time_node = Node(
            id=time_bucket_id,
            type=NodeType.TIME_BUCKET,
            attributes={"year": dt.year, "month": dt.month}
        )
        self.store.upsert_node(time_node)
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=time_node.id,
            type=EdgeType.OCCURRED_IN
        ))

        # Process Files (Diff)
        # We need to compare with parent[0] (or empty tree if no parent)
        parent = commit.parents[0] if commit.parents else None
        
        # Get diffs
        # If no parent, we diff against empty tree
        if parent:
            diffs = parent.diff(commit, create_patch=True)
        else:
            # Initial commit: diff against empty tree
            # gitpython generic_diff might be tricky here, 
            # alternative: iterate commit.tree?
            # For simplicity in this "Two-Pass" which relies on diff hunks,
            # we can likely treat initial commit files as all "Added".
            # gitpython `commit.diff(git.NULL_TREE)`
            # We use Reverse=True to diff NULL -> Commit (Additions)
            diffs = commit.diff(git.NULL_TREE, create_patch=True)
        
        # Note: diffs might be a generator, converting to list consumes it! 
        # But gitpython diff returns DiffIndex which is a list-like object usually.
        # But to be safe, let's re-generate or just iterate.
        # DiffIndex is a list subclass.
        
        for diff in diffs:
            self._process_diff_item(diff, commit_node)

        # --- Plugin processing ---
        # "process baseline nodes... then process in a consistent order all applied plugins"
        # Gather related nodes/edges for the context
        # We constructed: commit_node, msg_node, author_node, time_node
        # And edges connecting them.
        # Also file nodes/edges from _process_diff_item (which are committed to store but we might not have list here easily without tracking)
        # The prompt says: "sends the current commit node and all its directly related edges + nodes"
        
        # For simplicity, we can pass what we have locally. 
        # To pass ALL related nodes (including files), we'd need to return them from _process_diff_item or query the store.
        # Let's assume for now passing the core metadata nodes (Author, Msg, Time) is sufficient, 
        # or we fetch from store if needed. 
        # BUT, _process_diff_item inserts into store.
        
        # Let's collect the local nodes we created in this scope:
        related_nodes = [msg_node, author_node, time_node, repo_node] # + file nodes?
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author_node.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]
        
        
        # Instantiate API wrapper
        api = GraphAPIWrapper(self.store)
        
        for plugin in self.plugins:
            try:
                # specific config for this plugin
                # We use the class name as the key for now
                p_name = plugin.__class__.__name__
                # Also try matching by module name if needed, but class name is safer for now? 
                # Or main.py should pass a dict keyed by...? 
                # Let's use class name for simplicity as we don't know the "CLI name" here easily 
                # unless we store it on the plugin instance.
                # Actually, main.py loads plugins. 
                # But here we just have instances. 
                # Let's assume keys in plugin_config match class names OR we rely on a 'name' attribute if we added one. 
                # Since we didn't add 'name' to interface, let's try class name.
                # However, the user passes "-P ClickUpTask:regex...", so the key is "ClickUpTask".
                # If the class is "ClickUpPlugin", that matches? No.
                # "ClickUpTask" maps to "ClickUpTask" folder. 
                # The class name inside usually matches but not guaranteed. 
                # Let's try to match loosely or assume main.py handles mapping?
                # BETTER: plugin_config is passed as is. 
                # BUT we need to know WHICH config to pass to THIS plugin.
                # Let's assume plugin has a name attribute? No.
                # Let's try to check based on the plugin module name or directory?
                
                # Hack: Try to find a config key that matches the plugin's class name or module name.
                # If plugin is `sc.plugins.ClickUpTask.plugin.ClickUpPlugin`, maybe key is `ClickUpTask`?
                
                # Let's pass the whole config? No, that exposes too much.
                # Let's just pass `plugin_config.get(plugin.__class__.__name__, {})` AND `plugin_config.get(folder_name)`?
                
                # Simpler: update PluginInterface to have a `name` property?
                # Or just pass the whole dict and let plugin pick? 
                
                # The user said: "-P ClickUpTask:regex".
                # ClickUpTask is the directory name. 
                # So we want to pass config["ClickUpTask"].
                # How do we know this plugin instance corresponds to "ClickUpTask"?
                # In PluginManager, we load by name. We could attach the name to the instance.
                
                # Modification: I will pass specific config if found, otherwise empty.
                # I'll try to guess the name from the module.
                
                config = {}
                # plugin.__module__ might be 'gdskg_plugin_ClickUpTask' (from PluginManager loader)
                # or 'plugins.ClickUpTask.plugin'
                
                parts = plugin.__module__.split('.')
                # If using our loader: `gdskg_plugin_{name}`
                module_name = plugin.__module__
                
                if module_name.startswith("gdskg_plugin_"):
                    potential_name = module_name.replace("gdskg_plugin_", "")
                    config = self.plugin_config.get(potential_name, {})
                elif "plugins." in module_name:
                    # local load: plugins.ClickUpTask.plugin
                    # extract 'ClickUpTask'
                    try:
                        idx = parts.index("plugins")
                        if idx + 1 < len(parts):
                            potential_name = parts[idx+1]
                            config = self.plugin_config.get(potential_name, {})
                    except:
                        pass

                plugin.process(commit_node, related_nodes, related_edges, api, config)
            except Exception as e:
                # "If a plugin fails, the system should proceed. If a plugin errors, it should be caught."
                print(f"Error executing plugin {plugin}: {e}")

    def _process_diff_item(self, diff: git.Diff, commit_node: Node):
        # File path
        # a_path is source, b_path is target. 
        file_path = diff.b_path if diff.b_path else diff.a_path
        if not file_path:
            return

        file_node = Node(
            id=file_path,
            type=NodeType.FILE,
            attributes={"name": Path(file_path).name, "extension": Path(file_path).suffix}
        )
        self.store.upsert_node(file_node)
        
        # Modified File Edge
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=file_node.id,
            type=EdgeType.MODIFIED_FILE,
            attributes={"change_type": diff.change_type}
        ))

        # Content Analysis
        # We need the content of the file at this commit (for Pass 1 & Secrets)
        if diff.change_type == 'D':
            return # Skip analysis for deleted files (metadata remains in graph)

        try:
            # Blob content
            blob = diff.b_blob
            if not blob:
                return
                
            content = blob.data_stream.read().decode('utf-8', errors='replace')
            language = TreeSitterUtils.map_extension_to_language(file_node.attributes["extension"])
            
            if language:
                # --- Pass 1a: Extract Definitions (for Node Creation) ---
                # We only want to create SYMBOL nodes for things defined in this file (Functions, Classes)
                definitions = self.symbol_extractor.extract_symbols(content, language)
                for name, canonical in definitions.items():
                    symbol_node = Node(
                        id=canonical,
                        type=NodeType.SYMBOL,
                        attributes={"name": name, "file": file_path}
                    )
                    self.store.upsert_node(symbol_node)
                    # Optional: Edge File -> Symbol (DEFINES)
                
                # --- Pass 1b: Build Full Symbol Table (for Resolution) ---
                # This includes Imports + Definitions to resolve usages in the diff
                full_symbol_table = self.symbol_extractor.build_symbol_table(content, language)
                
                # --- Pass 2: Diff Filtering ---
                affected_lines = set()
                # Use diff object for patch
                patch = diff.diff.decode('utf-8', errors='replace') if isinstance(diff.diff, bytes) else diff.diff
                
                # Parse patch for line numbers
                current_line = 0
                for line in patch.splitlines():
                    if line.startswith('@@'):
                        m = re.search(r'\+(\d+)(?:,(\d+))?', line)
                        if m:
                            current_line = int(m.group(1))
                    elif line.startswith('+') and not line.startswith('+++'):
                        affected_lines.add(current_line)
                        current_line += 1
                    elif not line.startswith('-') and not line.startswith('---'):
                         current_line += 1
                
                if affected_lines:
                    # Pass full_symbol_table to resolve identifiers in the hunks
                    modified_symbols = self.symbol_extractor.filter_diff_symbols(content, language, affected_lines, full_symbol_table)
                    for sym_name in modified_symbols:
                        self.store.upsert_symbol(sym_name, file_path) # Ensure it exists (if from import? maybe skipping if external?)
                        # If sym_name comes from 'import os', upsert_symbol creates a node 'os'. 
                        # This satisfies "connect symbols".
                        self.store.upsert_edge(Edge(
                            source_id=sym_name,
                            target_id=commit_node.id,
                            type=EdgeType.MODIFIED_SYMBOL 
                        ))

            # --- Secret Scanning ---
            secrets = self.secret_scanner.scan(content)
            for secret in secrets:
                # Secret/Env Node
                secret_node = Node(
                    id=f"ENV:{secret}", 
                    type=NodeType.SECRET,
                    attributes={"variable": secret}
                )
                self.store.upsert_node(secret_node)
                
                # References Edge
                self.store.upsert_edge(Edge(
                    source_id=secret_node.id,
                    target_id=commit_node.id,
                    type=EdgeType.REFERENCES_ENV
                ))

        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            pass

class GraphAPIWrapper(GraphInterface):
    def __init__(self, store: GraphStore):
        self.store = store
    
    def add_node(self, id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        self.store.upsert_node(Node(id=id, type=type, attributes=attributes or {}))
        
    def add_edge(self, source_id: str, target_id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        self.store.upsert_edge(Edge(source_id=source_id, target_id=target_id, type=type, attributes=attributes or {}))




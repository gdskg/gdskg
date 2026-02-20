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
from analysis.keyword_extractor import KeywordExtractor
from analysis.tree_sitter_utils import TreeSitterUtils
from core.plugin_interfaces import PluginInterface, GraphInterface

class GraphExtractor:
    """
    Core engine for extracting a knowledge graph from a Git repository.
    
    This class orchestrates the extraction process, iterating through commits,
    analyzing diffs, extracting symbols, and invoking plugins to build a
    comprehensive representation of the software evolution.
    """

    def __init__(self, repo_path: Path, graph_store: GraphStore, plugins: List[PluginInterface] = None, plugin_config: Dict[str, Dict[str, Any]] = None):
        """
        Initialize the GraphExtractor.

        Args:
            repo_path (Path): The file system path to the Git repository.
            graph_store (GraphStore): The storage backend where nodes and edges will be persisted.
            plugins (List[PluginInterface], optional): A list of plugins to execute during extraction.
            plugin_config (Dict[str, Dict[str, Any]], optional): Configuration settings for the plugins, 
                keyed by plugin name.
        """

        self.repo_path = repo_path
        self.store = graph_store
        self.repo = git.Repo(repo_path)
        self.symbol_extractor = SymbolExtractor()
        self.secret_scanner = SecretScanner()
        self.keyword_extractor = KeywordExtractor()
        self.plugins = plugins or []
        self.plugin_config = plugin_config or {}
        
        # Two-pass collection
        import collections
        self.term_counts = collections.Counter()
        self.term_to_nodes = collections.defaultdict(set)

    def _process_keywords(self) -> None:
        """
        Finalize keyword analysis by identifying major terms and linking them to nodes.
        """
        major_terms = self.keyword_extractor.get_major_terms(self.term_counts, top_n=100, min_freq=2)
        
        for term in major_terms:
            keyword_node_id = f"KEYWORD:{term}"
            keyword_node = Node(
                id=keyword_node_id,
                type=NodeType.KEYWORD,
                attributes={"term": term, "frequency": self.term_counts[term]}
            )
            self.store.upsert_node(keyword_node)
            
            for node_id in self.term_to_nodes[term]:
                self.store.upsert_edge(Edge(
                    source_id=node_id,
                    target_id=keyword_node_id,
                    type=EdgeType.HAS_KEYWORD,
                    weight=1.0
                ))
    
    def process_repo(self) -> None:
        """
        Main entry point for graph extraction.
        
        Iterates through all commits in the repository in topological order (oldest to newest),
        performing full analysis on each commit and its associated changes.
        """

        repo_node = Node(
            id=str(self.repo_path),
            type=NodeType.REPOSITORY,
            attributes={
                "name": self.repo_path.name,
                "remotes": [r.url for r in self.repo.remotes]
            }
        )
        self.store.upsert_node(repo_node)

        commits = list(self.repo.iter_commits(topo_order=True, reverse=True))

        
        from rich.console import Console
        stderr_console = Console(file=sys.stderr)
        
        with Progress(console=stderr_console) as progress:
            task = progress.add_task("[green]Processing commits...", total=len(commits))
            
            for commit in commits:
                self._process_commit(commit, repo_node)
                progress.advance(task)
            
            # Second pass: Process major keywords
            self._process_keywords()

    def _process_commit(self, commit: git.Commit, repo_node: Node) -> None:
        """
        Process a specific Git commit and metadata.

        Args:
            commit (git.Commit): The GitPython Commit object to analyze.
            repo_node (Node): The parent repository node for context.
        """

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

        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=repo_node.id,
            type=EdgeType.PART_OF_REPO
        ))

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

        # First pass keyword collection
        keywords = self.keyword_extractor.extract_keywords(commit.message)
        for kw in keywords:
            self.term_counts[kw] += 1
            self.term_to_nodes[kw].add(commit_node.id)

        parent = commit.parents[0] if commit.parents else None
        
        if parent:
            diffs = parent.diff(commit, create_patch=True)
        else:
            diffs = commit.diff(git.NULL_TREE, create_patch=True)
        
        for diff in diffs:
            self._process_diff_item(diff, commit_node)

        related_nodes = [msg_node, author_node, time_node, repo_node]
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author_node.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]
        
        api = GraphAPIWrapper(self.store)
        
        for plugin in self.plugins:
            try:
                config = {}
                module_name = plugin.__module__
                
                if module_name.startswith("gdskg_plugin_"):
                    potential_name = module_name.replace("gdskg_plugin_", "")
                    config = self.plugin_config.get(potential_name, {})
                elif "plugins." in module_name:
                    try:
                        parts = module_name.split('.')
                        idx = parts.index("plugins")
                        if idx + 1 < len(parts):
                            potential_name = parts[idx+1]
                            config = self.plugin_config.get(potential_name, {})
                    except:
                        pass

                plugin.process(commit_node, related_nodes, related_edges, api, config)
            except Exception as e:
                print(f"Error executing plugin {plugin}: {e}")

    def _process_diff_item(self, diff: git.Diff, commit_node: Node) -> None:
        """
        Analyze a specific file change (diff) within a commit.

        This involves identifying file metadata, performing symbol extraction,
        diff filtering, and secret scanning on the file content.

        Args:
            diff (git.Diff): The GitPython Diff object representing the change.
            commit_node (Node): The node representing the current commit.
        """

        file_path = diff.b_path if diff.b_path else diff.a_path
        if not file_path:
            return

        file_node = Node(
            id=file_path,
            type=NodeType.FILE,
            attributes={"name": Path(file_path).name, "extension": Path(file_path).suffix}
        )
        self.store.upsert_node(file_node)

        # Extract keywords from file names
        file_keywords = self.keyword_extractor.extract_keywords(file_node.attributes["name"], is_code=True)
        for fkw in file_keywords:
            self.term_counts[fkw] += 1
            self.term_to_nodes[fkw].add(file_node.id)

        
        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=file_node.id,
            type=EdgeType.MODIFIED_FILE,
            attributes={"change_type": diff.change_type}
        ))

        if diff.change_type == 'D':
            return
            # Skip analysis for deleted files -- no more insights to glean
            # that have not already been gleaned

        try:
            blob = diff.b_blob
            if not blob:
                return
                
            content = blob.data_stream.read().decode('utf-8', errors='replace')
            language = TreeSitterUtils.map_extension_to_language(file_node.attributes["extension"])
            
            if language:
                definitions = self.symbol_extractor.extract_symbols(content, language)

                for name, canonical in definitions.items():
                    symbol_node = Node(
                        id=canonical,
                        type=NodeType.SYMBOL,
                        attributes={"name": name, "file": file_path}
                    )
                    self.store.upsert_node(symbol_node)
                    self.store.upsert_edge(Edge(
                        source_id=file_node.id,
                        target_id=symbol_node.id,
                        type=EdgeType.HAS_SYMBOL,
                        attributes={}
                    ))

                    # Extract keywords from symbol names
                    symbol_keywords = self.keyword_extractor.extract_keywords(name, is_code=True)
                    for skw in symbol_keywords:
                        self.term_counts[skw] += 1
                        self.term_to_nodes[skw].add(symbol_node.id)
                
                full_symbol_table = self.symbol_extractor.build_symbol_table(content, language)
                
                affected_lines = set()

                patch = diff.diff.decode('utf-8', errors='replace') if isinstance(diff.diff, bytes) else diff.diff
                
                # Parse patch for line numbers and hunk keywords
                current_line = 0
                added_lines_content = []
                for line in patch.splitlines():
                    if line.startswith('@@'):
                        m = re.search(r'\+(\d+)(?:,(\d+))?', line)
                        if m:
                            current_line = int(m.group(1))
                    elif line.startswith('+') and not line.startswith('+++'):
                        affected_lines.add(current_line)
                        added_lines_content.append(line[1:]) # Strip '+'
                        current_line += 1
                    elif not line.startswith('-') and not line.startswith('---'):
                         current_line += 1
                
                # Extract keywords from the hunk content
                hunk_text = " ".join(added_lines_content)
                hunk_keywords = self.keyword_extractor.extract_keywords(hunk_text, is_code=True)
                for hkw in hunk_keywords:
                    self.term_counts[hkw] += 1
                    self.term_to_nodes[hkw].add(commit_node.id)
                
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
                
                # Extract and attach comments/docstrings
                comments = self.symbol_extractor.extract_comments(content, language, affected_lines)
                for comment_text, comment_type, line_num in comments:
                    # Deterministic ID for comments within a file/commit
                    comment_hash = hashlib.sha256(f"{file_path}:{line_num}:{comment_text}".encode()).hexdigest()[:12]
                    comment_node = Node(
                        id=f"COMMENT:{comment_hash}",
                        type=NodeType.COMMENT,
                        attributes={
                            "content": comment_text,
                            "type": comment_type,
                            "file": file_path,
                            "line": line_num
                        }
                    )
                    self.store.upsert_node(comment_node)
                    self.store.upsert_edge(Edge(
                        source_id=comment_node.id,
                        target_id=commit_node.id,
                        type=EdgeType.HAS_COMMENT
                    ))
                    self.store.upsert_edge(Edge(
                        source_id=file_node.id,
                        target_id=comment_node.id,
                        type=EdgeType.CONTAINS_COMMENT
                    ))

            secrets = self.secret_scanner.scan(content)
            for secret in secrets:
                secret_node = Node(
                    id=f"ENV:{secret}", 
                    type=NodeType.SECRET,
                    attributes={"variable": secret}
                )
                self.store.upsert_node(secret_node)
                
                self.store.upsert_edge(Edge(
                    source_id=secret_node.id,
                    target_id=commit_node.id,
                    type=EdgeType.REFERENCES_ENV
                ))

        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            pass

class GraphAPIWrapper(GraphInterface):
    """
    A thin wrapper around GraphStore that implements GraphInterface.
    
    This is passed to plugins to provide a controlled environment for 
    modifying the knowledge graph.
    """

    def __init__(self, store: GraphStore):
        """
        Initialize the API wrapper.

        Args:
            store (GraphStore): The underlying graph storage to wrap.
        """

        self.store = store
    
    def add_node(self, id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Add or update a node in the graph.

        Args:
            id (str): Unique identifier for the node.
            type (str): The type of the node (from NodeType).
            attributes (Dict[str, Any], optional): Key-value pairs of node properties.
        """

        self.store.upsert_node(Node(id=id, type=type, attributes=attributes or {}))
        
    def add_edge(self, source_id: str, target_id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Add or update an edge between two nodes.

        Args:
            source_id (str): The ID of the source node.
            target_id (str): The ID of the target node.
            type (str): The type of the edge (from EdgeType).
            attributes (Dict[str, Any], optional): Key-value pairs of edge properties.
        """

        self.store.upsert_edge(Edge(source_id=source_id, target_id=target_id, type=type, attributes=attributes or {}))


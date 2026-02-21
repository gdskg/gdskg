import os
import sys
os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", "/usr/bin/git")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
import git

import re
import hashlib
import difflib
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import collections
from rich.progress import Progress
import subprocess
import binascii

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

    def __init__(self, repo_path: Path, graph_store: GraphStore, plugins: List[PluginInterface] = None, 
                 plugin_config: Dict[str, Dict[str, Any]] = None, progress=None, task_id=None,
                 skip_bots: bool = True):
        """
        Initialize the GraphExtractor.

        Args:
            repo_path (Path): The file system path to the Git repository.
            graph_store (GraphStore): The storage backend where nodes and edges will be persisted.
            plugins (List[PluginInterface], optional): A list of plugins to execute during extraction.
            plugin_config (Dict[str, Dict[str, Any]], optional): Configuration settings for the plugins, 
                keyed by plugin name.
            progress (rich.progress.Progress, optional): Shared progress instance.
            task_id (TaskID, optional): Existing task ID within the shared progress.
        """

        self.repo_path = repo_path
        self.store = graph_store
        self.repo = git.Repo(repo_path)
        self.symbol_extractor = SymbolExtractor()
        self.secret_scanner = SecretScanner()
        self.keyword_extractor = KeywordExtractor()
        self.plugins = plugins or []
        self.plugin_config = plugin_config or {}
        self.progress = progress
        self.task_id = task_id
        self.skip_bots = skip_bots
        
        import collections
        self.term_counts = collections.Counter()
        self.term_to_nodes = collections.defaultdict(set)
        
        self.active_functions: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

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
        
        Three-phase process:
        1. Parse git log for commit metadata + diff info (single subprocess)
        2. Parallel pre-computation: blob reads, tree parsing, symbol extraction (ThreadPoolExecutor)
        3. Sequential graph building using cached data (deterministic ordering)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        repo_node = Node(
            id=str(self.repo_path),
            type=NodeType.REPOSITORY,
            attributes={
                "name": self.repo_path.name,
                "remotes": [r.url for r in self.repo.remotes]
            }
        )
        self.store.upsert_node(repo_node)

        # ── Phase 1: Unified Git Log Stream ────────────────────────────────
        commit_order = []
        commit_meta = {}
        self.diff_cache = collections.defaultdict(list)
        
        log_process = subprocess.Popen(
            ['git', '-C', str(self.repo_path), 'log', '--no-abbrev', '--raw', '-p', '-U0',
             '--topo-order', '--reverse',
             '--format=format:^C^%H|~|%P|~|%aI|~|%an|~|%ae|~|%cI|~|%cn|~|%ce|~|%B^E^'],
            stdout=subprocess.PIPE, text=True, errors='replace', bufsize=1024*1024
        )
        
        current_commit = None
        current_diff = None
        current_diffs = None
        in_message = False
        message_buffer = []
        skip_current_commit = False

        for line in log_process.stdout:
            if line.startswith('^C^'):
                # Handle previous diff
                if not skip_current_commit and current_diff is not None:
                    current_diff['diff_lines'] = "".join(current_diff['diff_lines'])
                    current_diffs.append(current_diff)
                
                skip_current_commit = False
                
                # Parse header
                raw = line[3:]
                parts = raw.split('|~|', 8)
                if len(parts) >= 9:
                    hexsha = parts[0]
                    author_name = parts[3]
                    
                    bot_indicator = "[bot]" in author_name.lower() or "dependabot" in author_name.lower() or "greenkeeper" in author_name.lower()
                    if self.skip_bots and bot_indicator:
                        skip_current_commit = True
                        if '^E^' not in parts[8]:
                            in_message = True
                        else:
                            in_message = False
                        current_commit = None
                        current_diff = None
                        current_diffs = None
                        continue

                    commit_order.append(hexsha)
                    current_commit = hexsha
                    current_diff = None
                    current_diffs = self.diff_cache[current_commit]
                    
                    commit_meta[hexsha] = {
                        'hexsha': hexsha,
                        'parents': parts[1].split() if parts[1] else [],
                        'author_date': parts[2],
                        'author_name': parts[3],
                        'author_email': parts[4],
                        'committer_date': parts[5],
                        'committer_name': parts[6],
                        'committer_email': parts[7],
                        'message': parts[8]
                    }
                    if '^E^' not in parts[8]:
                        in_message = True
                        message_buffer = [parts[8]]
                    else:
                        commit_meta[hexsha]['message'] = parts[8].split('^E^')[0].strip()
                        in_message = False
                continue

            if in_message:
                if '^E^' in line:
                    if not skip_current_commit:
                        msg_part = line.split('^E^')[0]
                        message_buffer.append(msg_part)
                        commit_meta[current_commit]['message'] = "".join(message_buffer).strip()
                    in_message = False
                else:
                    if not skip_current_commit:
                        message_buffer.append(line)
                continue

            if skip_current_commit:
                continue

            if line.startswith(':'):
                if current_diff is not None:
                    current_diff['diff_lines'] = "".join(current_diff['diff_lines'])
                    current_diffs.append(current_diff)
                
                parts = line.strip('\n').split('\t')
                meta = parts[0].split()
                if len(meta) >= 5:
                    status = meta[4]
                    b_hexsha = meta[3]
                    if len(parts) == 2:
                        a_path = parts[1]
                        b_path = parts[1] if status[0] != 'D' else None
                    elif len(parts) == 3:
                        a_path = parts[1]
                        b_path = parts[2]
                    else:
                        a_path = b_path = None
                        
                    current_diff = {
                        'a_path': a_path,
                        'b_path': b_path,
                        'change_type': status, 
                        'b_blob_hexsha': b_hexsha if b_hexsha != '0' * 40 else None,
                        'diff_lines': []
                    }
                else:
                    current_diff = None
            elif current_diff is not None:
                current_diff['diff_lines'].append(line)
        
        if not skip_current_commit and current_commit and current_diff is not None:
            current_diff['diff_lines'] = "".join(current_diff['diff_lines'])
            current_diffs.append(current_diff)
            
        log_process.wait()

        # ── Phase 2: Parallel Pre-computation ──────────────────────────────
        blob_tasks = []
        seen_blobs = set()
        for hexsha in commit_order:
            for d in self.diff_cache.get(hexsha, []):
                blob_sha = d.get('b_blob_hexsha')
                file_path = d.get('b_path') or d.get('a_path')
                if blob_sha and blob_sha not in seen_blobs and d['change_type'][0] != 'D':
                    ext = Path(file_path).suffix if file_path else ""
                    blob_tasks.append((blob_sha, ext))
                    seen_blobs.add(blob_sha)
        
        self.blob_cache = {}
        self.tree_cache = {}
        self.symbol_cache = {}
        
        import threading
        odb_lock = threading.Lock()
        results_lock = threading.Lock()
        
        def preprocess_blob(blob_sha, ext):
            try:
                with odb_lock:
                    binsha = binascii.unhexlify(blob_sha)
                    stream = self.repo.odb.stream(binsha)
                    content = stream.read().decode('utf-8', errors='replace')
                
                language = TreeSitterUtils.map_extension_to_language(ext)
                tree = None
                definitions, symbol_table = {}, {}
                
                if language:
                    parser = TreeSitterUtils.get_parser(language)
                    if parser:
                        tree = parser.parse(bytes(content, "utf8"))
                        definitions, symbol_table = self.symbol_extractor.extract_all_symbols(content, language, tree=tree)
                
                with results_lock:
                    self.blob_cache[blob_sha] = content
                    if tree is not None:
                        self.tree_cache[blob_sha] = tree
                    self.symbol_cache[blob_sha] = (definitions, symbol_table)
            except Exception:
                pass

        if self.progress is not None:
            cache_task = self.progress.add_task("[cyan]Pre-caching file data...", total=len(blob_tasks))
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(preprocess_blob, sha, ext) for sha, ext in blob_tasks]
            for future in as_completed(futures):
                future.result()
                if self.progress is not None:
                    self.progress.advance(cache_task)

        # ── Phase 3: Sequential graph building ──────────────────────────────
        if self.progress is not None:
            graph_task = self.progress.add_task("[green]Building graph...", total=len(commit_order))
            for hexsha in commit_order:
                self._process_commit_fast(commit_meta[hexsha], repo_node)
                self.progress.advance(graph_task)
        else:
            from rich.progress import Progress
            from rich.console import Console
            stderr_console = Console(file=sys.stderr)
            
            with Progress(console=stderr_console) as progress:
                task = progress.add_task("[green]Building graph...", total=len(commit_order))
                
                for hexsha in commit_order:
                    self._process_commit_fast(commit_meta[hexsha], repo_node)
                    progress.advance(task)
            
        # Second pass: Process major keywords
        self._process_keywords()
        self.store.finalize()

    def _process_commit_fast(self, meta: dict, repo_node: Node) -> None:
        """
        Process a commit using pre-parsed metadata dict (no GitPython overhead).
        """
        hexsha = meta['hexsha']
        message = meta['message']
        author_name = meta['author_name']
        author_email = meta['author_email']
        author_date = meta['author_date']
        committer_name = meta['committer_name']
        committer_email = meta['committer_email']
        committer_date = meta['committer_date']
        parents = meta['parents']

        commit_node = Node(
            id=hexsha,
            type=NodeType.COMMIT,
            attributes={
                "message": message,
                "timestamp": author_date,
                "author_name": author_name,
                "author_email": author_email,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "committer_date": committer_date,
                "parents": parents
            }
        )
        self.store.upsert_node(commit_node)

        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=repo_node.id,
            type=EdgeType.PART_OF_REPO
        ))

        # Handle parents
        for p_sha in parents:
            self.store.upsert_edge(Edge(
                source_id=commit_node.id,
                target_id=p_sha,
                type=EdgeType.PARENT_OF, # Note: id -> parent means id is child
                attributes={}
            ))

        msg_id = hashlib.sha256(message.encode()).hexdigest()
        msg_node = Node(
            id=f"MSG:{msg_id}",
            type=NodeType.COMMIT_MESSAGE,
            attributes={"content": message}
        )
        self.store.upsert_node(msg_node)

        author_node = Node(
            id=author_email,
            type=NodeType.AUTHOR,
            attributes={"name": author_name}
        )
        self.store.upsert_node(author_node)

        committer_node = Node(
            id=committer_email,
            type=NodeType.AUTHOR, # Reuse AUTHOR type for person
            attributes={"name": committer_name}
        )
        self.store.upsert_node(committer_node)

        # Parse date for time bucket (ISO format: 2024-01-15T10:30:00+00:00)
        try:
            date_part = author_date[:10]  # "2024-01-15"
            year = int(date_part[:4])
            month = int(date_part[5:7])
        except (ValueError, IndexError):
            year, month = 2000, 1
            
        time_bucket_id = f"{year}-{month:02d}"
        time_node = Node(
            id=time_bucket_id,
            type=NodeType.TIME_BUCKET,
            attributes={"year": year, "month": month}
        )
        self.store.upsert_node(time_node)

        # Build list of related things for plugins
        related_nodes = [msg_node, author_node, committer_node, time_node, repo_node]
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author_node.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, committer_node.id, EdgeType.COMMITTED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]

        for re in related_edges:
            self.store.upsert_edge(re)

        # First pass keyword collection
        keywords = self.keyword_extractor.extract_keywords(message)
        for kw in keywords:
            self.term_counts[kw] += 1
            self.term_to_nodes[kw].add(commit_node.id)

        raw_diffs = self.diff_cache.get(hexsha, [])
        for diff_data in raw_diffs:
            self._process_diff_item(diff_data, commit_node)

        related_nodes = [msg_node, author_node, time_node, repo_node]
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author_node.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]
        
        api = GraphAPIWrapper(self.store)
        
        for plugin in self.plugins:
            if getattr(plugin, 'plugin_type', 'build') != 'build':
                continue
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

    def _process_commit(self, commit: git.Commit, repo_node: Node) -> None:
        """
        Process a specific Git commit and metadata (legacy method).

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

        raw_diffs = self.diff_cache.get(commit.hexsha, [])
        for diff_data in raw_diffs:
            self._process_diff_item(diff_data, commit_node)

        related_nodes = [msg_node, author_node, time_node, repo_node]
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author_node.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]
        
        api = GraphAPIWrapper(self.store)
        
        for plugin in self.plugins:
            if getattr(plugin, 'plugin_type', 'build') != 'build':
                continue
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

    def _process_diff_item(self, diff_data: dict, commit_node: Node) -> None:
        """
        Analyze a specific file change (diff) within a commit.

        Uses pre-cached blob content, parsed trees, and extracted symbols
        from the parallel pre-computation phase.

        Args:
            diff_data (dict): Raw diff dict with keys: a_path, b_path, change_type, b_blob_hexsha, diff_lines
            commit_node (Node): The node representing the current commit.
        """

        file_path = diff_data.get('b_path') or diff_data.get('a_path')
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

        edge_attributes = {"change_type": diff_data['change_type']}
        if diff_data.get('a_path') and diff_data.get('a_path') != file_path:
            edge_attributes["source_path"] = diff_data['a_path']

        self.store.upsert_edge(Edge(
            source_id=commit_node.id,
            target_id=file_node.id,
            type=EdgeType.MODIFIED_FILE,
            attributes=edge_attributes
        ))

        if diff_data['change_type'][0] == 'D':
            return

        try:
            blob_sha = diff_data.get('b_blob_hexsha')
            if not blob_sha:
                return
                
            # Use pre-cached blob content
            content = self.blob_cache.get(blob_sha)
            if content is None:
                return
            
            extension = file_node.attributes["extension"]
            language = TreeSitterUtils.map_extension_to_language(extension)
            
            if language:
                # Use pre-cached tree and symbols
                tree = self.tree_cache.get(blob_sha)
                cached_symbols = self.symbol_cache.get(blob_sha, ({}, {}))
                definitions, full_symbol_table = cached_symbols

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
                
                affected_lines = set()

                patch = diff_data['diff_lines']
                
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
                    # Combined traversal 2: diff symbols + modified functions + comments in one pass
                    modified_symbols, modified_funcs, comments = self.symbol_extractor.analyze_diff(
                        content, language, affected_lines, full_symbol_table, tree=tree
                    )
                    
                    for sym_name in modified_symbols:
                        self.store.upsert_symbol(sym_name, file_path)
                        self.store.upsert_edge(Edge(
                            source_id=sym_name,
                            target_id=commit_node.id,
                            type=EdgeType.MODIFIED_SYMBOL 
                        ))
                    
                    # Extract function versions
                    for func_name, func_content in modified_funcs.items():
                        func_node = Node(id=func_name, type=NodeType.FUNCTION, attributes={"name": func_name})
                        self.store.upsert_node(func_node)
                        self.store.upsert_edge(Edge(
                            source_id=commit_node.id,
                            target_id=func_node.id,
                            type=EdgeType.MODIFIED_FUNCTION
                        ))
                        
                        candidates = self.active_functions[func_name]
                        best_candidate = None
                        best_score = -1
                        
                        len_a = len(func_content)
                        for candidate in candidates:
                            is_same_file = candidate['last_file'] == file_path
                            len_b = len(candidate['last_content'])
                            max_similarity = 2.0 * min(len_a, len_b) / (len_a + len_b) if (len_a + len_b) > 0 else 1.0
                            
                            max_possible_score = max_similarity + (1.0 if is_same_file else 0.0)
                            if max_possible_score <= best_score:
                                continue
                                
                            similarity = difflib.SequenceMatcher(None, func_content, candidate['last_content']).quick_ratio()
                            score = similarity + (1.0 if is_same_file else 0.0)
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = candidate
                                
                        history_id = None
                        prev_version_id = None
                        
                        if best_candidate and ((best_candidate['last_file'] == file_path) or (best_score > 0.6)):
                            history_id = best_candidate['history_id']
                            prev_version_id = best_candidate['last_version_id']
                            best_candidate['last_content'] = func_content
                            best_candidate['last_file'] = file_path
                        else:
                            history_id = f"HISTORY:{func_name}:{uuid.uuid4().hex[:8]}"
                            history_node = Node(id=history_id, type=NodeType.FUNCTION_HISTORY, attributes={"function_name": func_name})
                            self.store.upsert_node(history_node)
                            self.store.upsert_edge(Edge(
                                source_id=func_node.id,
                                target_id=history_node.id,
                                type=EdgeType.HAS_HISTORY
                            ))
                            best_candidate = {
                                'history_id': history_id,
                                'last_version_id': None,
                                'last_content': func_content,
                                'last_file': file_path
                            }
                            self.active_functions[func_name].append(best_candidate)
                            
                        hash_content = hashlib.sha256(func_content.encode('utf-8')).hexdigest()
                        version_id = f"VERSION:{history_id}:{commit_node.id}:{hash_content[:8]}"
                        version_node = Node(
                            id=version_id,
                            type=NodeType.FUNCTION_VERSION,
                            attributes={"content": func_content, "commit_id": commit_node.id, "file_path": file_path}
                        )
                        self.store.upsert_node(version_node)
                        self.store.upsert_edge(Edge(
                            source_id=history_id,
                            target_id=version_node.id,
                            type=EdgeType.HAS_VERSION
                        ))
                        self.store.upsert_edge(Edge(
                            source_id=commit_node.id,
                            target_id=version_node.id,
                            type=EdgeType.CREATED_VERSION
                        ))
                        
                        if prev_version_id:
                            self.store.upsert_edge(Edge(
                                source_id=prev_version_id,
                                target_id=version_node.id,
                                type=EdgeType.PREVIOUS_VERSION
                            ))
                        best_candidate['last_version_id'] = version_node.id
                
                    # Process comments from combined traversal
                    for comment_text, comment_type, line_num in comments:
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


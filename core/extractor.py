import os
import sys
import re
import hashlib
import difflib
import uuid
import binascii
import collections
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import git
from rich.progress import Progress

from core.graph_store import GraphStore
from core.schema import Node, Edge, NodeType, EdgeType
from analysis.symbol_extractor import SymbolExtractor
from analysis.secret_scanner import SecretScanner
from analysis.keyword_extractor import KeywordExtractor
from analysis.tree_sitter_utils import TreeSitterUtils
from core.plugin_interfaces import PluginInterface, GraphInterface

os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", "/usr/bin/git")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

class GraphExtractor:
    """Core engine for extracting a knowledge graph from a Git repository."""

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
        
        self.term_counts = collections.Counter()
        self.term_to_nodes = collections.defaultdict(set)
        self.active_functions: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        
        self.blob_cache = {}
        self.tree_cache = {}
        self.symbol_cache = {}
        self.diff_cache = collections.defaultdict(list)

    def process_repo(self) -> None:
        """
        Main entry point for graph extraction.
        
        Executes the three-phase extraction process:
        1. Parse the git log for commits and diffs.
        2. Pre-cache blob content and extract symbols in parallel.
        3. Iterate through commits and build the graph structure.

        Returns:
            None
        """
        repo_node = self._upsert_repo_node()
        
        commit_order, commit_meta = self._phase1_parse_git_log()
        self._phase2_precache_blobs(commit_order)
        self._phase3_build_graph(commit_order, commit_meta, repo_node)
        
        self._process_keywords()
        self.store.finalize()

    def _upsert_repo_node(self) -> Node:
        """
        Create or update the root repository node in the graph.

        Returns:
            Node: The upserted repository node object.
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
        return repo_node

    def _phase1_parse_git_log(self) -> tuple:
        """
        Execute git log and parse the output into commit order and metadata dictionaries.

        Returns:
            tuple: (commit_order, commit_meta) where commit_order is a List[str] of SHAs 
                   and commit_meta is a Dict mapping SHAs to their metadata.
        """
        commit_order = []
        commit_meta = {}
        
        log_process = subprocess.Popen(
            ['git', '-C', str(self.repo_path), 'log', '--no-abbrev', '--raw', '-p', '-U0',
             '--topo-order', '--reverse',
             '--format=format:^C^%H|~|%P|~|%aI|~|%an|~|%ae|~|%cI|~|%cn|~|%ce|~|%B^E^'],
            stdout=subprocess.PIPE, text=True, errors='replace', bufsize=1024*1024
        )
        
        ctx = {'current_commit': None, 'current_diff': None, 'in_message': False, 
               'message_buffer': [], 'skip_current': False, 'current_diffs': None}

        for line in log_process.stdout:
            if line.startswith('^C^'):
                self._handle_new_commit_header(line, commit_order, commit_meta, ctx)
            elif ctx['in_message']:
                self._handle_message_line(line, commit_meta, ctx)
            elif ctx['skip_current']:
                continue
            elif line.startswith(':'):
                self._handle_diff_header(line, ctx)
            elif ctx['current_diff'] is not None:
                ctx['current_diff']['diff_lines'].append(line)
        
        if not ctx['skip_current'] and ctx['current_commit'] and ctx['current_diff'] is not None:
            ctx['current_diff']['diff_lines'] = "".join(ctx['current_diff']['diff_lines'])
            ctx['current_diffs'].append(ctx['current_diff'])
            
        log_process.wait()
        return commit_order, commit_meta

    def _handle_new_commit_header(self, line: str, commit_order: list, commit_meta: dict, ctx: dict):
        """
        Process a new commit header line from the git log output.

        Args:
            line (str): The current line being parsed.
            commit_order (list): The list to append the commit SHA to.
            commit_meta (dict): The dictionary to store commit metadata in.
            ctx (dict): The parsing context state.
        """
        if not ctx['skip_current'] and ctx['current_diff'] is not None:
            ctx['current_diff']['diff_lines'] = "".join(ctx['current_diff']['diff_lines'])
            ctx['current_diffs'].append(ctx['current_diff'])
        
        ctx['skip_current'] = False
        parts = line[3:].split('|~|', 8)
        if len(parts) >= 9:
            hexsha, author_name = parts[0], parts[3]
            
            is_bot = any(x in author_name.lower() for x in ["[bot]", "dependabot", "greenkeeper"])
            if self.skip_bots and is_bot:
                ctx['skip_current'] = True
                ctx['in_message'] = '^E^' not in parts[8]
                ctx['current_commit'] = ctx['current_diff'] = ctx['current_diffs'] = None
                return

            commit_order.append(hexsha)
            ctx['current_commit'] = hexsha
            ctx['current_diff'] = None
            ctx['current_diffs'] = self.diff_cache[hexsha]
            
            commit_meta[hexsha] = {
                'hexsha': hexsha,
                'parents': parts[1].split() if parts[1] else [],
                'author_date': parts[2], 'author_name': parts[3], 'author_email': parts[4],
                'committer_date': parts[5], 'committer_name': parts[6], 'committer_email': parts[7],
                'message': parts[8]
            }
            if '^E^' not in parts[8]:
                ctx['in_message'] = True
                ctx['message_buffer'] = [parts[8]]
            else:
                commit_meta[hexsha]['message'] = parts[8].split('^E^')[0].strip()
                ctx['in_message'] = False

    def _handle_message_line(self, line: str, commit_meta: dict, ctx: dict):
        """
        Process lines belonging to the commit message body.

        Args:
            line (str): The current line being parsed.
            commit_meta (dict): The dictionary containing commit metadata.
            ctx (dict): The parsing context state.
        """
        if '^E^' in line:
            if not ctx['skip_current']:
                msg_part = line.split('^E^')[0]
                ctx['message_buffer'].append(msg_part)
                commit_meta[ctx['current_commit']]['message'] = "".join(ctx['message_buffer']).strip()
            ctx['in_message'] = False
        elif not ctx['skip_current']:
            ctx['message_buffer'].append(line)

    def _handle_diff_header(self, line: str, ctx: dict):
        """
        Process a raw diff header line from the git log output.

        Args:
            line (str): The current line being parsed.
            ctx (dict): The parsing context state.
        """
        if ctx['current_diff'] is not None:
            ctx['current_diff']['diff_lines'] = "".join(ctx['current_diff']['diff_lines'])
            ctx['current_diffs'].append(ctx['current_diff'])
        
        parts = line.strip('\n').split('\t')
        meta = parts[0].split()
        if len(meta) >= 5:
            status, b_hexsha = meta[4], meta[3]
            a_path = parts[1]
            b_path = parts[2] if len(parts) == 3 else (parts[1] if status[0] != 'D' else None)
                
            ctx['current_diff'] = {
                'a_path': a_path, 'b_path': b_path, 'change_type': status, 
                'b_blob_hexsha': b_hexsha if b_hexsha != '0' * 40 else None, 'diff_lines': []
            }
        else:
            ctx['current_diff'] = None

    def _phase2_precache_blobs(self, commit_order: list):
        """
        Pre-fetch and analyze file contents in parallel to improve performance.

        Args:
            commit_order (list): The list of commit SHAs to process in chronological order.
        """
        blob_tasks = []
        seen_blobs = set()
        for hexsha in commit_order:
            for d in self.diff_cache.get(hexsha, []):
                sha = d.get('b_blob_hexsha')
                path = d.get('b_path') or d.get('a_path')
                if sha and sha not in seen_blobs and d['change_type'][0] != 'D':
                    blob_tasks.append((sha, Path(path).suffix if path else ""))
                    seen_blobs.add(sha)
        
        odb_lock = threading.Lock()
        results_lock = threading.Lock()
        
        def preprocess(sha, ext):
            try:
                with odb_lock:
                    content = self.repo.odb.stream(binascii.unhexlify(sha)).read().decode('utf-8', errors='replace')
                
                lang = TreeSitterUtils.map_extension_to_language(ext)
                tree, defs, table = None, {}, {}
                if lang:
                    parser = TreeSitterUtils.get_parser(lang)
                    if parser:
                        tree = parser.parse(bytes(content, "utf8"))
                        defs, table = self.symbol_extractor.extract_all_symbols(content, lang, tree=tree)
                
                with results_lock:
                    self.blob_cache[sha] = content
                    if tree: self.tree_cache[sha] = tree
                    self.symbol_cache[sha] = (defs, table)
            except Exception: pass

        if self.progress:
            t = self.progress.add_task("[cyan]Pre-caching file data...", total=len(blob_tasks))
        
        with ThreadPoolExecutor(max_workers=16) as exc:
            futures = [exc.submit(preprocess, s, e) for s, e in blob_tasks]
            for f in as_completed(futures):
                f.result()
                if self.progress: self.progress.advance(t)

    def _phase3_build_graph(self, commit_order, commit_meta, repo_node):
        """
        Build the actual graph nodes and edges from the collected commit data.

        Args:
            commit_order (List[str]): SHAs in processing order.
            commit_meta (Dict): Metadata for each commit SHA.
            repo_node (Node): The root repository node.
        """
        if self.progress:
            t = self.progress.add_task("[green]Building graph...", total=len(commit_order))
            for hexsha in commit_order:
                self._process_commit_fast(commit_meta[hexsha], repo_node)
                self.progress.advance(t)
        else:
            from rich.console import Console
            with Progress(console=Console(file=sys.stderr)) as p:
                t = p.add_task("[green]Building graph...", total=len(commit_order))
                for hexsha in commit_order:
                    self._process_commit_fast(commit_meta[hexsha], repo_node)
                    p.advance(t)

    def _process_commit_fast(self, meta: dict, repo_node: Node) -> None:
        """
        Process a single commit, creating its node and processing its diffs.

        Args:
            meta (dict): Metadata for the commit.
            repo_node (Node): The repository node the commit belongs to.
        """
        commit_node = Node(
            id=meta['hexsha'], type=NodeType.COMMIT,
            attributes={
                "message": meta['message'], "timestamp": meta['author_date'],
                "author_name": meta['author_name'], "author_email": meta['author_email'],
                "committer_name": meta['committer_name'], "committer_email": meta['committer_email'],
                "committer_date": meta['committer_date'], "parents": meta['parents']
            }
        )
        self.store.upsert_node(commit_node)
        self._link_commit_basics(commit_node, repo_node, meta)
        
        for diff_data in self.diff_cache.get(meta['hexsha'], []):
            self._process_diff_item(diff_data, commit_node)

        self._run_plugins(commit_node, repo_node, meta)

    def _link_commit_basics(self, commit_node, repo_node, meta):
        """
        Link a commit node to its repository, parents, author, and time nodes.

        Args:
            commit_node (Node): The node representing the commit.
            repo_node (Node): The repository node.
            meta (dict): Commit metadata containing timestamps and parent SHAs.
        """
        self._link_commit_repo_and_parents(commit_node, repo_node, meta)
        self._link_commit_message_and_authors(commit_node, meta)
        self._link_commit_time_and_keywords(commit_node, meta)

    def _link_commit_repo_and_parents(self, commit_node, repo_node, meta):
        """
        Link commit to its repository and parent commits.

        Args:
            commit_node (Node): The current commit node.
            repo_node (Node): The root repository node.
            meta (dict): Metadata containing parent SHAs.
        """
        self.store.upsert_edge(Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO))
        for p_sha in meta['parents']:
            self.store.upsert_edge(Edge(commit_node.id, p_sha, EdgeType.PARENT_OF))

    def _link_commit_message_and_authors(self, commit_node, meta):
        """
        Link commit to its message, author, and committer nodes.

        Args:
            commit_node (Node): The current commit node.
            meta (dict): Metadata containing message and author details.
        """
        msg_id = hashlib.sha256(meta['message'].encode()).hexdigest()
        msg_node = Node(id=f"MSG:{msg_id}", type=NodeType.COMMIT_MESSAGE, attributes={"content": meta['message']})
        self.store.upsert_node(msg_node)
        self.store.upsert_edge(Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE))

        author = Node(id=meta['author_email'], type=NodeType.AUTHOR, attributes={"name": meta['author_name']})
        committer = Node(id=meta['committer_email'], type=NodeType.AUTHOR, attributes={"name": meta['committer_name']})
        self.store.upsert_node(author)
        self.store.upsert_node(committer)
        self.store.upsert_edge(Edge(commit_node.id, author.id, EdgeType.AUTHORED_BY))
        self.store.upsert_edge(Edge(commit_node.id, committer.id, EdgeType.COMMITTED_BY))

    def _link_commit_time_and_keywords(self, commit_node, meta):
        """
        Link commit to its time bucket and extracted message keywords.

        Args:
            commit_node (Node): The current commit node.
            meta (dict): Metadata containing author date and message content.
        """
        try:
            date_part = meta['author_date'][:10]
            year, month = int(date_part[:4]), int(date_part[5:7])
        except Exception: year, month = 2000, 1
        time_node = Node(id=f"{year}-{month:02d}", type=NodeType.TIME_BUCKET, attributes={"year": year, "month": month})
        self.store.upsert_node(time_node)
        self.store.upsert_edge(Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN))

        for kw in self.keyword_extractor.extract_keywords(meta['message']):
            self.term_counts[kw] += 1
            self.term_to_nodes[kw].add(commit_node.id)

    def _run_plugins(self, commit_node, repo_node, meta):
        """
        Execute build-time plugins for the current commit context.

        Args:
            commit_node (Node): The current commit node.
            repo_node (Node): The repository node.
            meta (dict): Commit metadata for additional context.
        """
        msg_id = hashlib.sha256(meta['message'].encode()).hexdigest()
        msg_node = Node(id=f"MSG:{msg_id}", type=NodeType.COMMIT_MESSAGE, attributes={"content": meta['message']})
        author = Node(id=meta['author_email'], type=NodeType.AUTHOR, attributes={"name": meta['author_name']})
        date_part = meta['author_date'][:10]
        time_node = Node(id=date_part[:7], type=NodeType.TIME_BUCKET, attributes={})
        
        related_nodes = [msg_node, author, time_node, repo_node]
        related_edges = [
            Edge(commit_node.id, msg_node.id, EdgeType.HAS_MESSAGE),
            Edge(commit_node.id, repo_node.id, EdgeType.PART_OF_REPO),
            Edge(commit_node.id, author.id, EdgeType.AUTHORED_BY),
            Edge(commit_node.id, time_node.id, EdgeType.OCCURRED_IN)
        ]
        
        api = GraphAPIWrapper(self.store)
        for plugin in self.plugins:
            if getattr(plugin, 'plugin_type', 'build') != 'build':
                continue
            try:
                name = plugin.__module__.split('.')[-2] if 'plugins.' in plugin.__module__ else plugin.__module__.replace("gdskg_plugin_", "")
                config = self.plugin_config.get(name, {})
                plugin.process(commit_node, related_nodes, related_edges, api, config)
            except Exception as e:
                print(f"Error executing plugin {plugin}: {e}")

    def _process_diff_item(self, diff_data: dict, commit_node: Node) -> None:
        """
        Process a single file diff within a commit.

        Args:
            diff_data (dict): Data about the file change (paths, change type, blob SHA).
            commit_node (Node): The commit node that introduced the change.
        """
        file_path = diff_data.get('b_path') or diff_data.get('a_path')
        if not file_path: return

        file_node = Node(id=file_path, type=NodeType.FILE, attributes={"name": Path(file_path).name, "extension": Path(file_path).suffix})
        self.store.upsert_node(file_node)
        
        for fkw in self.keyword_extractor.extract_keywords(file_node.attributes["name"], is_code=True):
            self.term_counts[fkw] += 1
            self.term_to_nodes[fkw].add(file_node.id)

        attrs = {"change_type": diff_data['change_type']}
        if diff_data.get('a_path') and diff_data.get('a_path') != file_path:
            attrs["source_path"] = diff_data['a_path']

        self.store.upsert_edge(Edge(commit_node.id, file_node.id, EdgeType.MODIFIED_FILE, attributes=attrs))

        if diff_data['change_type'][0] == 'D': return

        try:
            sha = diff_data.get('b_blob_hexsha')
            content = self.blob_cache.get(sha)
            if not content: return
            
            lang = TreeSitterUtils.map_extension_to_language(file_node.attributes["extension"])
            if lang:
                self._process_language_diff(diff_data, file_node, commit_node, content, lang, sha)

            self._scan_for_secrets(content, commit_node)
        except Exception: pass

    def _scan_for_secrets(self, content: str, commit_node: Node):
        """
        Scan file content for potential secrets and link them to the commit.

        Args:
            content (str): The text content of the file.
            commit_node (Node): The commit node to link potential secrets to.
        """
        secrets = self.secret_scanner.scan(content)
        for secret in secrets:
            s_node = Node(id=f"ENV:{secret}", type=NodeType.SECRET, attributes={"variable": secret})
            self.store.upsert_node(s_node)
            self.store.upsert_edge(Edge(s_node.id, commit_node.id, EdgeType.REFERENCES_ENV))

    def _process_language_diff(self, diff_data, file_node, commit_node, content, lang, sha):
        """
        Perform language-aware diff analysis including symbols, functions, and comments.

        Args:
            diff_data (dict): The diff metadata.
            file_node (Node): The node representing the file.
            commit_node (Node): The node representing the commit.
            content (str): The full content of the file at this commit.
            lang (str): The programming language name.
            sha (str): The blob SHA of the file content.
        """
        tree = self.tree_cache.get(sha)
        defs, table = self.symbol_cache.get(sha, ({}, {}))

        for name, canonical in defs.items():
            sym = Node(id=canonical, type=NodeType.SYMBOL, attributes={"name": name, "file": file_node.id})
            self.store.upsert_node(sym)
            self.store.upsert_edge(Edge(file_node.id, sym.id, EdgeType.HAS_SYMBOL))
            for skw in self.keyword_extractor.extract_keywords(name, is_code=True):
                self.term_counts[skw] += 1
                self.term_to_nodes[skw].add(sym.id)
        
        patch = diff_data['diff_lines']
        affected, added_text = self._parse_patch(patch)
        for hkw in self.keyword_extractor.extract_keywords(added_text, is_code=True):
            self.term_counts[hkw] += 1
            self.term_to_nodes[hkw].add(commit_node.id)
        
        if affected:
            mod_syms, mod_funcs, comments = self.symbol_extractor.analyze_diff(content, lang, affected, table, tree=tree)
            
            for sym_name in mod_syms:
                self.store.upsert_symbol(sym_name, file_node.id)
                self.store.upsert_edge(Edge(sym_name, commit_node.id, EdgeType.MODIFIED_SYMBOL))
            
            for f_name, f_content in mod_funcs.items():
                self._process_function_version(f_name, f_content, file_node.id, commit_node)
        
            for text, c_type, line in comments:
                self._process_comment(text, c_type, line, file_node, commit_node)

    def _parse_patch(self, patch: str) -> tuple:
        """
        Parse a unified diff patch to identify affected line numbers and added text.

        Args:
            patch (str): The raw unified diff patch string.

        Returns:
            tuple: (affected_lines, added_text) where affected_lines is a Set[int] 
                   of line numbers and added_text is a single string of all added lines.
        """
        affected = set()
        added_lines = []
        curr = 0
        for line in patch.splitlines():
            if line.startswith('@@'):
                m = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if m: curr = int(m.group(1))
            elif line.startswith('+') and not line.startswith('+++'):
                affected.add(curr)
                added_lines.append(line[1:])
                curr += 1
            elif not line.startswith('-') and not line.startswith('---'):
                curr += 1
        return affected, " ".join(added_lines)

    def _process_function_version(self, f_name, f_content, file_path, commit_node):
        """
        Track function versions over time and link them to their history.

        Args:
            f_name (str): The name of the function.
            f_content (str): The full content of the function at this version.
            file_path (str): The path to the file containing the function.
            commit_node (Node): The node representing the commit.
        """
        f_node = Node(id=f_name, type=NodeType.FUNCTION, attributes={"name": f_name})
        self.store.upsert_node(f_node)
        self.store.upsert_edge(Edge(commit_node.id, f_node.id, EdgeType.MODIFIED_FUNCTION))
        
        best, best_score = self._find_best_history_match(f_name, f_content, file_path)
                
        if best and (best['last_file'] == file_path or best_score > 0.6):
            h_id, prev_v_id = best['history_id'], best['last_version_id']
            best.update({'last_content': f_content, 'last_file': file_path})
        else:
            h_id = f"HISTORY:{f_name}:{uuid.uuid4().hex[:8]}"
            self.store.upsert_node(Node(id=h_id, type=NodeType.FUNCTION_HISTORY, attributes={"function_name": f_name}))
            self.store.upsert_edge(Edge(f_node.id, h_id, EdgeType.HAS_HISTORY))
            best = {'history_id': h_id, 'last_version_id': None, 'last_content': f_content, 'last_file': file_path}
            self.active_functions[f_name].append(best)
            prev_v_id = None
            
        v_id = f"VERSION:{h_id}:{commit_node.id}:{hashlib.sha256(f_content.encode('utf-8')).hexdigest()[:8]}"
        self.store.upsert_node(Node(id=v_id, type=NodeType.FUNCTION_VERSION, attributes={"content": f_content, "commit_id": commit_node.id, "file_path": file_path}))
        self.store.upsert_edge(Edge(h_id, v_id, EdgeType.HAS_VERSION))
        self.store.upsert_edge(Edge(commit_node.id, v_id, EdgeType.CREATED_VERSION))
        if prev_v_id:
            self.store.upsert_edge(Edge(prev_v_id, v_id, EdgeType.PREVIOUS_VERSION))
        best['last_version_id'] = v_id

    def _find_best_history_match(self, f_name: str, content: str, file_path: str) -> tuple:
        """
        Identify the most likely previous version of a function for history tracking.

        Args:
            f_name (str): The name of the function.
            content (str): The current content of the function.
            file_path (str): The current file path.

        Returns:
            tuple: (best_match_dict, similarity_score)
        """
        best, best_score = None, -1
        len_a = len(content)
        for cand in self.active_functions[f_name]:
            is_same = cand['last_file'] == file_path
            len_b = len(cand['last_content'])
            max_sim = 2.0 * min(len_a, len_b) / (len_a + len_b) if (len_a + len_b) > 0 else 1.0
            if max_sim + (1.0 if is_same else 0.0) <= best_score: continue
                
            sim = difflib.SequenceMatcher(None, content, cand['last_content']).quick_ratio()
            score = sim + (1.0 if is_same else 0.0)
            if score > best_score:
                best_score, best = score, cand
        return best, best_score

    def _process_comment(self, text, c_type, line, file_node, commit_node):
        """
        Create a comment node and link it to its file and commit.

        Args:
            text (str): The content of the comment.
            c_type (str): The category of comment (e.g., 'inline', 'docstring').
            line (int): The line number where the comment starts.
            file_node (Node): The file where the comment is located.
            commit_node (Node): The commit that introduced/modified the comment.
        """
        h = hashlib.sha256(f"{file_node.id}:{line}:{text}".encode()).hexdigest()[:12]
        c_node = Node(id=f"COMMENT:{h}", type=NodeType.COMMENT, attributes={"content": text, "type": c_type, "file": file_node.id, "line": line})
        self.store.upsert_node(c_node)
        self.store.upsert_edge(Edge(c_node.id, commit_node.id, EdgeType.HAS_COMMENT))
        self.store.upsert_edge(Edge(file_node.id, c_node.id, EdgeType.CONTAINS_COMMENT))

    def _process_keywords(self) -> None:
        """
        Finalize keyword nodes and link them to all relevant nodes found during extraction.

        Returns:
            None
        """
        major = self.keyword_extractor.get_major_terms(self.term_counts, top_n=100, min_freq=2)
        for term in major:
            k_id = f"KEYWORD:{term}"
            self.store.upsert_node(Node(id=k_id, type=NodeType.KEYWORD, attributes={"term": term, "frequency": self.term_counts[term]}))
            for node_id in self.term_to_nodes[term]:
                self.store.upsert_edge(Edge(source_id=node_id, target_id=k_id, type=EdgeType.HAS_KEYWORD))

class GraphAPIWrapper(GraphInterface):
    """Thin wrapper around GraphStore for plugins."""
    def __init__(self, store: GraphStore):
        self.store = store
    
    def add_node(self, id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """Add a node to the graph storage."""
        self.store.upsert_node(Node(id=id, type=type, attributes=attributes or {}))
        
    def add_edge(self, source_id: str, target_id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """Add an edge to the graph storage."""
        self.store.upsert_edge(Edge(source_id=source_id, target_id=target_id, type=type, attributes=attributes or {}))



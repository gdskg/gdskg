import git
import re
from pathlib import Path
from typing import Optional
from rich.progress import Progress

from core.graph_store import GraphStore
from core.schema import Node, Edge, NodeType, EdgeType
from analysis.symbol_extractor import SymbolExtractor
from analysis.secret_scanner import SecretScanner
from analysis.tree_sitter_utils import TreeSitterUtils

class GraphExtractor:
    def __init__(self, repo_path: Path, graph_store: GraphStore):
        self.repo_path = repo_path
        self.store = graph_store
        self.repo = git.Repo(repo_path)
        self.symbol_extractor = SymbolExtractor()
        self.secret_scanner = SecretScanner()
    
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
        
        with Progress() as progress:
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
            diffs = commit.diff(git.NULL_TREE, create_patch=True)

        for diff in diffs:
            self._process_diff_item(diff, commit_node)

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
            
            # --- Pass 1: Global Symbol Table for this file state ---
            # Ideally we cache this or use it for resolution.
            # In this streamlined flow, we extract and ensuring they exist as nodes.
            # Note: The Spec says "Bind... to create a temporary Symbol Table".
            # We will extract *definitions* here.
            
            if language:
                symbols_pass1 = self.symbol_extractor.extract_symbols(content, language)
                for name, canonical in symbols_pass1.items():
                    # Upsert Symbol Node
                    symbol_node = Node(
                        id=canonical,
                        type=NodeType.SYMBOL,
                        attributes={"name": name, "file": file_path}
                    )
                    self.store.upsert_node(symbol_node)
                    # We could link Symbol -> File here, but spec only asks for Symbol -> Commit
            
            # --- Pass 2: Diff Filtering ---
            # Identify modified lines
            affected_lines = set()
            # gitpython diff parsing to find hunk lines
            # diff.diff is the patch string. We need to parse it.
            # Simple header parsing: @@ -old,count +new,count @@
            patch = diff.diff.decode('utf-8', errors='replace') if isinstance(diff.diff, bytes) else diff.diff
            
            # We want "new" lines for Added/Modified symbols.
            # This is a complex parser. Simplification:
            # Use `diff` object properties if available. GitPython doesn't give line numbers easily.
            # We must parse the patch header.
            
            # Example: @@ -1,5 +1,6 @@
            current_line = 0
            for line in patch.splitlines():
                if line.startswith('@@'):
                    # Parse header
                    m = re.search(r'\+(\d+)(?:,(\d+))?', line)
                    if m:
                        current_line = int(m.group(1))
                elif line.startswith('+') and not line.startswith('+++'):
                    affected_lines.add(current_line)
                    current_line += 1
                elif not line.startswith('-'):
                    current_line += 1
            
            if language and affected_lines:
                modified_symbols = self.symbol_extractor.filter_diff_symbols(content, language, affected_lines)
                for sym_name in modified_symbols:
                    # In Pass 2 we get raw identifiers. We should try to resolve to canonical if possible.
                    # For now, use the identifier as ID (or lookup in symbols_pass1 if exact match)
                    
                    canonical = symbols_pass1.get(sym_name, sym_name)
                    
                    # Ensure node exists (might have been missed in Pass 1 if it's not a definition but a usage? 
                    # Spec: "Modified Symbol" - usually implies definition change.
                    sym_node = Node(id=canonical, type=NodeType.SYMBOL, attributes={"name": sym_name})
                    self.store.upsert_node(sym_node)
                    
                    self.store.upsert_edge(Edge(
                        source_id=sym_node.id, 
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



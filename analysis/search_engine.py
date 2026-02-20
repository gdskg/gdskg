import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from core.schema import NodeType, EdgeType

class SearchEngine:
    """
    Search engine for querying the knowledge graph.
    
    Provides capabilities to search for commits based on keywords, 
    filter by repository, and traverse the graph to find related nodes.
    """

    def __init__(self, db_path: str, vector_db_path: str = None):
        """
        Initialize the SearchEngine.

        Args:
            db_path (str): The file path to the SQLite database.
            vector_db_path (str, optional): Path to the vector database. Defaults to the same as db_path.
        """

        self.db_path = db_path
        self.vector_db_path = vector_db_path or db_path

    def search(self, query: str, repo_name: str = None, depth: int = 1, traverse_types: List[str] = None, semantic_only: bool = False) -> List[Dict[str, Any]]:
        """
        Search for relevant commits based on keywords and traverse the graph.

        Args:
            query (str): The search query string containing keywords.
            repo_name (str, optional): The name of the repository to scope the search to.
            depth (int, optional): The traversal depth for finding related nodes. Defaults to 1.
            traverse_types (List[str], optional): A list of node types allowed for traversal. 
                If None, uses a default blacklist (Author, TimeBucket, Repository).

        Returns:
            List[Dict[str, Any]]: A list of matching commits with their metadata and connections, 
                sorted by relevance.
        """

        if semantic_only:
            keywords = []
        else:
            stop_words = {'the', 'and', 'for', 'how', 'to', 'of', 'in', 'is', 'a', 'with', 'that', 'this', 'or'}
            raw_keywords = query.lower().split()
            keywords = [kw for kw in raw_keywords if kw not in stop_words]
            if not keywords: # Fallback to all if everything is a stop word
                keywords = raw_keywords

        commits = {}
        
        # Determine allowed types early
        allowed_types = set()
        if traverse_types:
            allowed_types = {t.upper() for t in traverse_types}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            repo_ids = set()

            if repo_name:
                cursor.execute("""
                    SELECT id FROM nodes 
                    WHERE type = ? AND json_extract(attributes, '$.name') = ?
                """, (NodeType.REPOSITORY.value, repo_name))
                repo_ids = {r[0] for r in cursor.fetchall()}
                if not repo_ids:
                    return []

            for kw in keywords:

                # 1. Search COMMIT nodes directly (message/id)
                cursor.execute("""
                    SELECT id, attributes FROM nodes 
                    WHERE type = ? AND (id LIKE ? OR json_extract(attributes, '$.message') LIKE ?)
                """, (NodeType.COMMIT.value, f"%{kw}%", f"%{kw}%"))
                for row in cursor.fetchall():
                    commit_id, attr_json = row
                    self._ensure_commit_in_results(commits, commit_id, attr_json, repo_ids, cursor)
                    if commit_id in commits:
                        commits[commit_id]["relevance"] += 10
                        commits[commit_id]["reasons"].append(f"Keyword '{kw}' found in commit metadata")

                # 2. Search via KEYWORD nodes
                cursor.execute("""
                    SELECT n.id, n.attributes FROM nodes n
                    JOIN edges e ON n.id = e.source_id
                    JOIN nodes k ON e.target_id = k.id
                    WHERE n.type = ? AND k.type = ? AND k.id = ?
                """, (NodeType.COMMIT.value, NodeType.KEYWORD.value, f"KEYWORD:{kw}"))
                for row in cursor.fetchall():
                    commit_id, attr_json = row
                    self._ensure_commit_in_results(commits, commit_id, attr_json, repo_ids, cursor)
                    if commit_id in commits:
                        commits[commit_id]["relevance"] += 8
                        reasons = commits[commit_id]["reasons"]
                        if f"Significant keyword '{kw}' found in change hunks" not in reasons:
                            reasons.append(f"Significant keyword '{kw}' found in change hunks")

                # 3. Search other node types
                cursor.execute("""
                    SELECT id, type, attributes FROM nodes 
                    WHERE 
                      (type = ? AND (id LIKE ? OR json_extract(attributes, '$.name') LIKE ?)) OR
                      (type = ? AND id LIKE ?) OR
                      (type = ? AND json_extract(attributes, '$.content') LIKE ?) OR
                      (type = ? AND json_extract(attributes, '$.content') LIKE ?)
                """, (
                    NodeType.SYMBOL.value, f"%{kw}%", f"%{kw}%",
                    NodeType.FILE.value, f"%{kw}%",
                    NodeType.COMMIT_MESSAGE.value, f"%{kw}%",
                    NodeType.COMMENT.value, f"%{kw}%"
                ))
                
                matched_rows = cursor.fetchall()
                for node_id, n_type, n_attr_json in matched_rows:
                    if n_type == NodeType.COMMIT_MESSAGE.value:
                        cursor.execute("""
                            SELECT source_id FROM edges 
                            WHERE target_id = ? AND type = ?
                        """, (node_id, EdgeType.HAS_MESSAGE.value))
                        reason = f"Keyword '{kw}' found in commit message"
                    elif n_type == NodeType.COMMENT.value:
                        cursor.execute("""
                            SELECT target_id FROM edges 
                            WHERE source_id = ? AND type = ?
                        """, (node_id, EdgeType.HAS_COMMENT.value))
                        reason = f"Keyword '{kw}' found in code comment"
                    else:
                        cursor.execute("""
                            SELECT target_id FROM edges 
                            WHERE source_id = ? AND (type = ? OR type = ?)
                        """, (node_id, EdgeType.MODIFIED_SYMBOL.value, EdgeType.MODIFIED_FILE.value))
                        label = "symbol" if n_type == NodeType.SYMBOL.value else "file"
                        reason = f"Keyword '{kw}' found in {label} '{node_id}'"
                    
                    linked_commits = [r[0] for r in cursor.fetchall()]
                    for cid in linked_commits:
                        self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                        if cid in commits:
                            commits[cid]["relevance"] += 5
                            if reason not in commits[cid]["reasons"]:
                                commits[cid]["reasons"].append(reason)

            # 4. Semantic Search via Vector DB
            if Path(self.vector_db_path).exists():
                try:
                    from core.vector_store import VectorStore
                    from analysis.embedder import ONNXEmbedder
                    
                    vector_store = VectorStore(Path(self.vector_db_path))
                    embedder = ONNXEmbedder()
                    
                    # Embed full query (not just keywords)
                    query_embedding = embedder.embed([query])
                    if len(query_embedding) > 0:
                        semantic_results = vector_store.search(query_embedding[0], top_k=20)
                        for node_id, node_type, similarity in semantic_results:
                            if similarity < 0.20: # Threshold
                                continue
                                
                            # Find commits connected to this node
                            if node_type == NodeType.SYMBOL.value:
                                cursor.execute("""
                                    SELECT target_id FROM edges 
                                    WHERE source_id = ? AND type = ?
                                """, (node_id, EdgeType.MODIFIED_SYMBOL.value))
                                reason = f"Similar meaning identified (matched {node_type} '{node_id}' score={similarity:.2f})"
                            elif node_type == NodeType.FUNCTION.value:
                                # Function node connected to commit? Let's check MODIFIED_FUNCTION
                                cursor.execute("""
                                    SELECT source_id FROM edges 
                                    WHERE target_id = ? AND type = ?
                                """, (node_id, EdgeType.MODIFIED_FUNCTION.value))
                                reason = f"Similar meaning identified (matched {node_type} '{node_id}' score={similarity:.2f})"
                            else:
                                continue
                                
                            linked_commits = [r[0] for r in cursor.fetchall()]
                            for cid in linked_commits:
                                self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                                if cid in commits:
                                    # Base relevance score boost from similarity
                                    commits[cid]["relevance"] += (similarity * 20)
                                    if reason not in commits[cid]["reasons"]:
                                        commits[cid]["reasons"].append(reason)
                                        
                except Exception as e:
                    print(f"Warning: Vector search failed: {e}")

            if depth > 0:

                for cid in commits:
                    queue = [(cid, 0)]
                    visited = {cid}

                    
                    while queue:
                        curr_id, dist = queue.pop(0)
                        if dist >= depth:
                            continue
                            
                        cursor.execute("""

                            SELECT target_id, type FROM edges WHERE source_id = ?
                            UNION
                            SELECT source_id, type FROM edges WHERE target_id = ?
                        """, (curr_id, curr_id))
                        
                        for neighbor_id, _ in cursor.fetchall():
                            if neighbor_id not in visited:
                                visited.add(neighbor_id)
                                
                                cursor.execute("SELECT type, attributes FROM nodes WHERE id = ?", (neighbor_id,))

                                n_row = cursor.fetchone()
                                if n_row:
                                    n_type, n_attrs_json = n_row
                                    n_attrs = json.loads(n_attrs_json) if n_attrs_json else {}
                                    commits[cid]["connections"][neighbor_id] = {
                                        "type": n_type,
                                        "distance": dist + 1,
                                        "attributes": n_attrs
                                    }
                                    
                                    if traverse_types is not None:

                                        # Always allow reaching COMMIT nodes (the goal), otherwise check allowlist
                                        if n_type != NodeType.COMMIT.value and n_type not in allowed_types:
                                            continue
                                    else:
                                        # Default Blacklist Mode (if no types specified)
                                        # Block noisy types by default
                                        if n_type in [NodeType.AUTHOR.value, NodeType.TIME_BUCKET.value, NodeType.REPOSITORY.value]:
                                            continue
                                        
                                    queue.append((neighbor_id, dist + 1))

        # Sort by relevance
        result = list(commits.values())
        result.sort(key=lambda x: x["relevance"], reverse=True)
        return result
    def _ensure_commit_in_results(self, commits: Dict, commit_id: str, attr_json: Optional[str], repo_ids: Set[str], cursor: sqlite3.Cursor):
        """Helper to check repo visibility and initialize commit in results if missing."""
        if commit_id in commits:
            return

        if repo_ids:
            cursor.execute("""
                SELECT 1 FROM edges 
                WHERE source_id = ? AND target_id IN (%s) AND type = ?
            """ % ",".join("?" * len(repo_ids)), (commit_id, *repo_ids, EdgeType.PART_OF_REPO.value))
            if not cursor.fetchone():
                return

        if not attr_json:
            cursor.execute("SELECT attributes FROM nodes WHERE id = ?", (commit_id,))
            row = cursor.fetchone()
            if row:
                attr_json = row[0]
            else:
                return

        attrs = json.loads(attr_json)
        commits[commit_id] = {
            "id": commit_id,
            "message": attrs.get("message", "No message"),
            "author": attrs.get("author_name", attrs.get("author", "Unknown")),
            "date": attrs.get("timestamp", attrs.get("date", "Unknown")),
            "relevance": 0,
            "connections": {},
            "reasons": []
        }

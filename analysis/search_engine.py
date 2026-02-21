import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from core.schema import NodeType, EdgeType
import math

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

    def search(self, query: str = "", repo_name: str = None, depth: int = 1, traverse_types: List[str] = None, semantic_only: bool = False, min_score: float = 10.0, top_n: int = 5, filters: Dict[str, str] = None, all_matches: bool = False, all_files: bool = False) -> List[Dict[str, Any]]:
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
        
        query = query or ""


        from analysis.keyword_extractor import KeywordExtractor
        extractor = KeywordExtractor()
        stop_words = extractor.stop_words
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

            allowed_commits = None
            if filters:
                for f_type_raw, f_val in filters.items():
                    f_type = f_type_raw.upper()
                    
                    cursor.execute("""
                        SELECT id FROM nodes 
                        WHERE type = ? AND (id LIKE ? OR json_extract(attributes, '$.name') LIKE ? OR json_extract(attributes, '$.content') LIKE ?)
                    """, (f_type, f"%{f_val}%", f"%{f_val}%", f"%{f_val}%"))
                    
                    matched_filter_node_ids = [r[0] for r in cursor.fetchall()]
                    if not matched_filter_node_ids:
                        return []
                        
                    placeholders = ",".join(["?"] * len(matched_filter_node_ids))
                    
                    cursor.execute(f"""
                        SELECT source_id FROM edges WHERE target_id IN ({placeholders}) AND source_id IN (SELECT id FROM nodes WHERE type=?)
                        UNION
                        SELECT target_id FROM edges WHERE source_id IN ({placeholders}) AND target_id IN (SELECT id FROM nodes WHERE type=?)
                    """, matched_filter_node_ids + [NodeType.COMMIT.value] + matched_filter_node_ids + [NodeType.COMMIT.value])
                    
                    commits_for_filter = {r[0] for r in cursor.fetchall()}
                    if not commits_for_filter:
                        return []
                        
                    if allowed_commits is None:
                        allowed_commits = commits_for_filter
                    else:
                        allowed_commits.intersection_update(commits_for_filter)
                        if not allowed_commits:
                            return []

            if allowed_commits is not None and not keywords and not query:
                for cid in allowed_commits:
                    self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                    if cid in commits:
                        commits[cid]["relevance"] += 5
                        commits[cid]["reasons"].append(f"Matched strict filters: {filters}")


            # 1. Search ALL nodes for keywords
            search_keywords = [] if semantic_only else keywords
            for kw in search_keywords:
                # Search across various node types and common attributes (name, message, content, id)
                cursor.execute("""
                    SELECT id, type, attributes FROM nodes 
                    WHERE 
                      id LIKE ? OR 
                      json_extract(attributes, '$.name') LIKE ? OR 
                      json_extract(attributes, '$.message') LIKE ? OR 
                      json_extract(attributes, '$.content') LIKE ? OR
                      json_extract(attributes, '$.author_name') LIKE ? OR
                      json_extract(attributes, '$.author_email') LIKE ?
                """, (f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"))
                
                matched_nodes = cursor.fetchall()
                for node_id, n_type, n_attr_json in matched_nodes:
                    # If it's a COMMIT, score it directly
                    if n_type == NodeType.COMMIT.value:
                        if allowed_commits is not None and node_id not in allowed_commits:
                            continue
                        self._ensure_commit_in_results(commits, node_id, n_attr_json, repo_ids, cursor)
                        if node_id in commits:
                            kw_count = commits[node_id]["keyword_matches"].get(kw, 0)
                            commits[node_id]["relevance"] += 10.0 / (kw_count + 1)
                            commits[node_id]["keyword_matches"][kw] = kw_count + 1
                            if kw_count == 0:
                                commits[node_id]["reasons"].append(f"Keyword '{kw}' found in commit metadata")
                    
                    # If it's a KEYWORD node, find linked commits
                    elif n_type == NodeType.KEYWORD.value:
                        cursor.execute("""
                            SELECT source_id FROM edges 
                            WHERE target_id = ? AND type = ?
                        """, (node_id, EdgeType.HAS_KEYWORD.value))
                        for (cid,) in cursor.fetchall():
                            if allowed_commits is not None and cid not in allowed_commits:
                                continue
                            self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                            if cid in commits:
                                kw_count = commits[cid]["keyword_matches"].get(kw, 0)
                                commits[cid]["relevance"] += 8.0 / (kw_count + 1)
                                commits[cid]["keyword_matches"][kw] = kw_count + 1
                                rsn = f"Significant keyword '{kw}' found in change hunks"
                                if rsn not in commits[cid]["reasons"]:
                                    commits[cid]["reasons"].append(rsn)
                                # Ensure trigger node is in connections
                                commits[cid]["connections"][node_id] = {
                                    "type": n_type,
                                    "distance": 0, # Trigger node
                                    "attributes": json.loads(n_attr_json) if n_attr_json else {}
                                }
                    
                    # For other types, find linked commits via various edge types
                    else:
                        # Find commits that are "related" to this node within 1 step
                        # This covers: MODIFIED_FILE, MODIFIED_SYMBOL, MODIFIED_FUNCTION, HAS_MESSAGE, AUTHORED_BY, etc.
                        cursor.execute("""
                            SELECT source_id FROM edges WHERE target_id = ?
                            UNION
                            SELECT target_id FROM edges WHERE source_id = ?
                        """, (node_id, node_id))
                        
                        potential_commit_ids = [r[0] for r in cursor.fetchall()]
                        for potential_cid in potential_commit_ids:
                            # Verify it's actually a commit (or at least linked to one)
                            # Actually, we'll just check if it exists as a COMMIT node
                            cursor.execute("SELECT type FROM nodes WHERE id = ?", (potential_cid,))
                            row = cursor.fetchone()
                            if row and row[0] == NodeType.COMMIT.value:
                                cid = potential_cid
                                if allowed_commits is not None and cid not in allowed_commits:
                                    continue
                                self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                                if cid in commits:
                                    kw_count = commits[cid]["keyword_matches"].get(kw, 0)
                                    commits[cid]["relevance"] += 5.0 / (kw_count + 1)
                                    commits[cid]["keyword_matches"][kw] = kw_count + 1
                                    reason = f"Keyword '{kw}' matches {n_type} '{node_id}'"
                                    if reason not in commits[cid]["reasons"]:
                                        commits[cid]["reasons"].append(reason)
                                    # Ensure trigger node is in connections
                                    commits[cid]["connections"][node_id] = {
                                        "type": n_type,
                                        "distance": 0, # Trigger node
                                        "attributes": json.loads(n_attr_json) if n_attr_json else {}
                                    }


            # Pre-compute semantic relevance set for connection filtering
            semantic_relevance_set = set()

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
                        all_semantic_results = vector_store.search(query_embedding[0], top_k=200)
                        for nid, ntype, sim in all_semantic_results:
                            if sim >= 0.20:
                                semantic_relevance_set.add(nid)

                        semantic_results = all_semantic_results[:20]
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
                                if allowed_commits is not None and cid not in allowed_commits:
                                    continue
                                self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                                if cid in commits:
                                    # Base relevance score boost from similarity
                                    sem_count = commits[cid]["semantic_matches"]
                                    commits[cid]["relevance"] += (similarity * 20.0) / (sem_count + 1)
                                    commits[cid]["semantic_matches"] = sem_count + 1
                                    if reason not in commits[cid]["reasons"]:
                                        commits[cid]["reasons"].append(reason)
                                        
                except Exception as e:
                    print(f"Warning: Vector search failed: {e}")

            # Commit High File Count Penalty
            for cdict in commits.values():
                cid = cdict["id"]
                cursor.execute("""
                    SELECT count(*) FROM edges
                    WHERE source_id = ? AND type = ?
                """, (cid, EdgeType.MODIFIED_FILE.value))
                num_files = cursor.fetchone()[0]
                if num_files > 0:
                    penalty = max(1.0, math.log10(num_files))
                    if penalty > 1.0:
                        cdict["relevance"] /= (penalty * 3)
                        cdict["reasons"].append(f"Penalized relevance (divided by {penalty:.2f}) for modifying {num_files} files")

            # Filter and sort base commits here so depth traversal only processes top records
            sorted_commits = [r for r in commits.values() if r["relevance"] >= min_score]
            sorted_commits.sort(key=lambda x: x["relevance"], reverse=True)
            top_commits = sorted_commits[:top_n]

            if depth > 0:
                for cdict in top_commits:
                    cid = cdict["id"]
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
                                    is_trigger_node = neighbor_id in cdict["connections"] and cdict["connections"][neighbor_id].get("distance", -1) == 0

                                    include_connection = True
                                    if not all_matches and not is_trigger_node:
                                        if n_type == NodeType.KEYWORD.value:
                                            text_match = any(kw in neighbor_id.lower() or kw in str(n_attrs.get("name", "")).lower() for kw in keywords)
                                            if not text_match and neighbor_id not in semantic_relevance_set:
                                                include_connection = False
                                        elif n_type == NodeType.SYMBOL.value:
                                            if neighbor_id not in semantic_relevance_set:
                                                include_connection = False
                                        elif n_type == NodeType.FILE.value and not all_files:
                                            if neighbor_id not in semantic_relevance_set:
                                                cursor.execute("""
                                                    SELECT target_id, type FROM edges WHERE source_id = ? AND type IN (?, ?, ?)
                                                    UNION
                                                    SELECT source_id, type FROM edges WHERE target_id = ? AND type IN (?, ?, ?)
                                                """, (neighbor_id, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value, 
                                                      neighbor_id, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value))
                                                linked_nodes = [r[0] for r in cursor.fetchall()]
                                                
                                                has_relevant_child = False
                                                for ln in linked_nodes:
                                                    if ln in semantic_relevance_set or any(kw in ln.lower() for kw in keywords):
                                                        has_relevant_child = True
                                                        break
                                                        
                                                if not has_relevant_child:
                                                    include_connection = False
                                        elif n_type in [NodeType.COMMIT.value, NodeType.COMMIT_MESSAGE.value, NodeType.COMMENT.value, NodeType.SECRET.value, NodeType.FUNCTION.value]:
                                            if neighbor_id not in semantic_relevance_set:
                                                text_match = False
                                                n_text = str(n_attrs.get("message", "")) + " " + str(n_attrs.get("text", "")) + " " + str(n_attrs.get("title", "")) + " " + str(n_attrs.get("name", ""))
                                                n_text = n_text.lower()
                                                for kw in keywords:
                                                    if kw in n_text:
                                                        text_match = True
                                                        break
                                                if not text_match:
                                                    include_connection = False
                                                    
                                    if include_connection:
                                        # Only add if we aren't overwriting a distance 0 trigger node with a higher distance
                                        if not is_trigger_node:
                                            cdict["connections"][neighbor_id] = {
                                                "type": n_type,
                                                "distance": dist + 1,
                                                "attributes": n_attrs
                                            }
                                    else:
                                        # If the node is not deemed relevant, do not use it to traverse further!
                                        continue
                                    
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

        return top_commits
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
            "relevance": 0.0,
            "keyword_matches": {},
            "semantic_matches": 0,
            "connections": {},
            "reasons": []
        }

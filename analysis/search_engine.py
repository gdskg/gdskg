import sqlite3
import json
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from core.schema import NodeType, EdgeType
from analysis.keyword_extractor import KeywordExtractor
from core.vector_store import VectorStore
from analysis.embedder import ONNXEmbedder

class SearchEngine:
    """Search engine for querying the knowledge graph."""

    def __init__(self, db_path: str, vector_db_path: str = None):
        """
        Initialize the SearchEngine with the specified knowledge graph and vector database.

        Args:
            db_path (str): The local filesystem path to the GDSKG SQLite database.
            vector_db_path (str, optional): The path to the vector database. 
                Defaults to ~/.gdskg/vector_db/gdskg_vectors.db if not provided.
        """
        self.db_path = db_path
        if vector_db_path:
            self.vector_db_path = vector_db_path
        else:
            vector_dir = Path(os.environ.get("GDSKG_VECTOR_DB_DIR", Path.home() / ".gdskg" / "vector_db"))
            vector_dir.mkdir(parents=True, exist_ok=True)
            self.vector_db_path = str(vector_dir / "gdskg_vectors.db")

    def search(self, query: str = "", repo_name: str = None, depth: int = 1, traverse_types: List[str] = None, exclude_types: List[str] = None, semantic_only: bool = False, min_score: float = 10.0, top_n: int = 5, filters: Dict[str, str] = None, all_matches: bool = False, all_files: bool = False, excluded_commits: List[str] = None, offset: int = 0, negative_query: str = "") -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using keyword and/or semantic search, applying filters and depth-based exploration.

        Args:
            query (str, optional): The user's search query string. Defaults to "".
            repo_name (str, optional): Filter results to a specific repository by name. Defaults to None.
            depth (int, optional): The graph traversal depth for finding connected context. Defaults to 1.
            traverse_types (List[str], optional): List of node types to follow during traversal. Defaults to None.
            semantic_only (bool, optional): If True, bypass keyword search and use semantic search exclusively. Defaults to False.
            min_score (float, optional): The minimum relevance score for a commit to be included in results. Defaults to 10.0.
            top_n (int, optional): The maximum number of base commit results to return. Defaults to 5.
            filters (Dict[str, str], optional): Metadata filters in 'Type:Value' format. Defaults to None.
            all_matches (bool, optional): If True, include all traversal matches regardless of relevance. Defaults to False.
            all_files (bool, optional): If True, include all modified files in results. Defaults to False.
            excluded_commits (List[str], optional): List of commit IDs to exclude from results. Defaults to None.
            offset (int, optional): The number of results to skip for pagination. Defaults to 0.
            negative_query (str, optional): Terms to avoid in keyword and semantic search. Defaults to "".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the most relevant commits and their context.
        """
        query = query.strip() if query else ""
        keywords = self._extract_keywords(query)
        commits = {}
        
        negative_keywords = []
        if negative_query:
            negative_keywords = self._extract_keywords(negative_query)
        
        # AST_NODE should not be traversed in a general semantic/keyword search
        allowed_types = {t.upper() for t in traverse_types} if traverse_types else set()
        if allowed_types:
            allowed_types.discard(NodeType.AST_NODE.value)

        exclude_types = {t.upper().rstrip('S') for t in exclude_types} if exclude_types else set()
        # Ensure we can still see commits if not explicitly excluding them, 
        # but usually exclude_types is for traversal noise.

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            repo_ids = self._get_repo_ids(cursor, repo_name) if repo_name else set()
            if repo_name and not repo_ids:
                return []

            allowed_commits = self._apply_strict_filters(cursor, filters)
            if filters and allowed_commits is None:
                return []

            if allowed_commits is not None and not query:
                self._add_filter_only_matches(commits, allowed_commits, repo_ids, cursor, filters)

            search_keywords = [] if semantic_only else keywords
            if search_keywords:
                self._search_keywords(cursor, search_keywords, commits, repo_ids, allowed_commits)
            
            if negative_keywords and not semantic_only:
                self._search_keywords(cursor, negative_keywords, commits, repo_ids, allowed_commits, is_negative=True)

            semantic_relevance_set = set()
            if query:
                semantic_relevance_set = self._run_semantic_search(query, commits, cursor, repo_ids, allowed_commits)
            
            negative_semantic_set = set()
            if negative_query:
                negative_semantic_set = self._run_semantic_search(negative_query, commits, cursor, repo_ids, allowed_commits, is_negative=True)

            self._apply_file_count_penalty(cursor, commits)

            if excluded_commits:
                # Support both full SHAs and short prefixes
                to_delete = []
                for cid in excluded_commits:
                    if cid in commits:
                        to_delete.append(cid)
                    else:
                        # Try prefix match
                        for full_sha in commits.keys():
                            if full_sha.startswith(cid):
                                to_delete.append(full_sha)
                
                for key in set(to_delete):
                    del commits[key]

            sorted_commits = [r for r in commits.values() if r["relevance"] >= min_score]
            sorted_commits.sort(key=lambda x: x["relevance"], reverse=True)
            
            top_commits = sorted_commits[offset : offset + top_n]

            if depth > 0:
                self._run_graph_traversal(cursor, top_commits, depth, keywords, semantic_relevance_set, allowed_types, exclude_types, all_matches, all_files, negative_keywords, negative_semantic_set)

        return top_commits

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract searchable keywords from the query string, removing stop words.

        Args:
            query (str): The raw query string from the user.

        Returns:
            List[str]: A list of cleaned, searchable keyword strings.
        """
        extractor = KeywordExtractor()
        raw = query.lower().split()
        keywords = [kw for kw in raw if kw not in extractor.stop_words]
        return keywords if keywords else raw

    def _get_repo_ids(self, cursor: sqlite3.Cursor, repo_name: str) -> Set[str]:
        """
        Retrieve node IDs for all repositories matching the given name.

        Args:
            cursor (sqlite3.Cursor): The database cursor for current connection.
            repo_name (str): The name of the repository to look up.

        Returns:
            Set[str]: A set of unique node IDs belonging to the matched repositories.
        """
        cursor.execute("SELECT id FROM nodes WHERE type = ? AND json_extract(attributes, '$.name') = ?", (NodeType.REPOSITORY.value, repo_name))
        return {r[0] for r in cursor.fetchall()}

    def _apply_strict_filters(self, cursor: sqlite3.Cursor, filters: Optional[Dict[str, str]]) -> Optional[Set[str]]:
        """
        Return a set of commit IDs that strictly satisfy the provided metadata filters.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            filters (Dict[str, str], optional): Metadata filters to apply.

        Returns:
            Optional[Set[str]]: A set of allowed commit IDs, or None if no filters were provided.
        """
        if not filters:
            return None
        allowed = None
        for ftype, fval in filters.items():
            cursor.execute("""
                SELECT id FROM nodes WHERE type = ? AND (id LIKE ? OR json_extract(attributes, '$.name') LIKE ? OR json_extract(attributes, '$.content') LIKE ?)
            """, (ftype.upper(), f"%{fval}%", f"%{fval}%", f"%{fval}%"))
            matched_nodes = [r[0] for r in cursor.fetchall()]
            if not matched_nodes:
                return set()
            
            p = ",".join(["?"] * len(matched_nodes))
            cursor.execute(f"""
                SELECT source_id FROM edges WHERE target_id IN ({p}) AND source_id IN (SELECT id FROM nodes WHERE type=?)
                UNION
                SELECT target_id FROM edges WHERE source_id IN ({p}) AND target_id IN (SELECT id FROM nodes WHERE type=?)
            """, matched_nodes + [NodeType.COMMIT.value] + matched_nodes + [NodeType.COMMIT.value])
            matches = {r[0] for r in cursor.fetchall()}
            if not matches:
                return set()
            allowed = matches if allowed is None else allowed.intersection(matches)
            if not allowed:
                return set()
        return allowed

    def _add_filter_only_matches(self, commits: Dict, allowed_commits: Set[str], repo_ids: Set[Set[str]], cursor: sqlite3.Cursor, filters: Dict):
        """
        Add commits to results that match strict filters, even if no keywords match.

        Args:
            commits (Dict): The results dictionary to update.
            allowed_commits (Set[str]): The set of commit IDs allowed by filters.
            repo_ids (Set[str]): The set of repository IDs allowed by repository filter.
            cursor (sqlite3.Cursor): The database cursor.
            filters (Dict): The original filters applied.
        """
        for cid in allowed_commits:
            self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
            if cid in commits:
                commits[cid]["relevance"] += 5
                commits[cid]["reasons"].append(f"Matched strict filters: {filters}")

    def _search_keywords(self, cursor: sqlite3.Cursor, keywords: List[str], commits: Dict, repo_ids: Set[str], allowed_commits: Optional[Set[str]], is_negative: bool = False):
        """
        Execute keyword-based search against the graph and update relevance scores.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            keywords (List[str]): The keywords to search for.
            commits (Dict): The results dictionary to populate.
            repo_ids (Set[str]): The set of allowed repository IDs.
            allowed_commits (Set[str], optional): The set of allowed commit IDs from strict filters.
            is_negative (bool): If True, matching nodes will penalize relevance instead of increasing it.
        """
        for kw in keywords:
            cursor.execute("""
                SELECT id, type, attributes FROM nodes 
                WHERE id LIKE ? OR json_extract(attributes, '$.name') LIKE ? OR json_extract(attributes, '$.message') LIKE ? OR 
                      json_extract(attributes, '$.content') LIKE ? OR json_extract(attributes, '$.author_name') LIKE ? OR json_extract(attributes, '$.author_email') LIKE ? OR
                      json_extract(attributes, '$.body') LIKE ? OR json_extract(attributes, '$.description') LIKE ?
            """, (f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"))
            
            for nid, ntype, n_attr_json in cursor.fetchall():
                linked_commits = []
                if ntype == NodeType.COMMIT.value:
                    linked_commits = [nid]
                elif ntype == NodeType.KEYWORD.value:
                    cursor.execute("SELECT source_id FROM edges WHERE target_id = ? AND type = ?", (nid, EdgeType.HAS_KEYWORD.value))
                    linked_commits = [r[0] for r in cursor.fetchall()]
                else:
                    cursor.execute("SELECT source_id FROM edges WHERE target_id = ? UNION SELECT target_id FROM edges WHERE source_id = ?", (nid, nid))
                    for pid in [r[0] for r in cursor.fetchall()]:
                        cursor.execute("SELECT type FROM nodes WHERE id = ?", (pid,))
                        row = cursor.fetchone()
                        if row and row[0] == NodeType.COMMIT.value:
                            linked_commits.append(pid)

                for cid in linked_commits:
                    if allowed_commits is not None and cid not in allowed_commits:
                        continue
                    
                    if is_negative:
                        if cid in commits:
                            del commits[cid]
                        continue

                    self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                    if cid in commits:
                        kw_count = commits[cid]["keyword_matches"].get(kw, 0)
                        score = 10.0 if ntype == NodeType.COMMIT.value else (8.0 if ntype == NodeType.KEYWORD.value else 5.0)
                        commits[cid]["relevance"] += score / (kw_count + 1)
                        commits[cid]["keyword_matches"][kw] = kw_count + 1
                        reason = f"Keyword '{kw}' matches {ntype} '{nid}'"
                        if reason not in commits[cid]["reasons"]:
                            commits[cid]["reasons"].append(reason)
                        if ntype != NodeType.COMMIT.value:
                            commits[cid]["connections"][nid] = {"type": ntype, "distance": 0, "attributes": json.loads(n_attr_json) if n_attr_json else {}}

    def _run_semantic_search(self, query: str, commits: Dict, cursor: sqlite3.Cursor, repo_ids: Set[str], allowed_commits: Optional[Set[str]], is_negative: bool = False) -> Set[str]:
        """
        Execute semantic search using vector embeddings and update relevance scores.

        Args:
            query (str): The user's query string for similarity matching.
            commits (Dict): The results dictionary to update.
            cursor (sqlite3.Cursor): The database cursor.
            repo_ids (Set[str]): The set of allowed repository IDs.
            allowed_commits (Set[str], optional): The set of allowed commit IDs from filters.
            is_negative (bool): If True, similarity will be subtracted from relevance.

        Returns:
            Set[str]: A set of node IDs identified as semantically relevant.
        """
        semantic_relevance_set = set()
        if not Path(self.vector_db_path).exists():
            return semantic_relevance_set
            
        try:
            vector_store = VectorStore(Path(self.vector_db_path))
            embedder = ONNXEmbedder()
            query_embedding = embedder.embed([query])
            if not query_embedding:
                return semantic_relevance_set
                
            all_results = vector_store.search(query_embedding[0], top_k=200)
            for nid, ntype, sim in all_results:
                if sim >= 0.20:
                    semantic_relevance_set.add(nid)

            candidates = all_results if is_negative else all_results[:20]
            for nid, ntype, sim in candidates:
                if sim < 0.20:
                    continue
                
                linked_commits = []
                reason = f"Similar meaning identified (matched {ntype} '{nid}' score={sim:.2f})"
                if ntype == NodeType.SYMBOL.value:
                    cursor.execute("SELECT target_id FROM edges WHERE source_id = ? AND type = ?", (nid, EdgeType.MODIFIED_SYMBOL.value))
                    linked_commits = [r[0] for r in cursor.fetchall()]
                elif ntype == NodeType.FUNCTION.value:
                    cursor.execute("SELECT source_id FROM edges WHERE target_id = ? AND type = ?", (nid, EdgeType.MODIFIED_FUNCTION.value))
                    linked_commits = [r[0] for r in cursor.fetchall()]
                elif ntype == NodeType.PULL_REQUEST.value:
                    cursor.execute("SELECT source_id FROM edges WHERE target_id = ? AND type = ?", (nid, EdgeType.RELATED_TO_PR.value))
                    linked_commits = [r[0] for r in cursor.fetchall()]
                elif ntype == NodeType.CLICKUP_TASK.value:
                    cursor.execute("SELECT source_id FROM edges WHERE target_id = ? AND type = ?", (nid, EdgeType.RELATED_TO_TASK.value))
                    linked_commits = [r[0] for r in cursor.fetchall()]
                elif ntype in [NodeType.COMMIT.value, NodeType.COMMIT_MESSAGE.value, "COMMIT_SYMBOLS"]:
                    linked_commits = [nid]
                    reason = f"Similar meaning identified in commit context (matched {ntype} score={sim:.2f})"
                
                for cid in linked_commits:
                    if allowed_commits is not None and cid not in allowed_commits:
                        continue
                    
                    if is_negative:
                        if cid in commits:
                            penalty = 200.0 * sim
                            commits[cid]["relevance"] -= penalty
                            commits[cid]["reasons"].append(f"Avoided concept identified (similarity={sim:.2f}, penalty={penalty:.2f})")
                        continue

                    self._ensure_commit_in_results(commits, cid, None, repo_ids, cursor)
                    if cid in commits:
                        sem_count = commits[cid]["semantic_matches"]
                        commits[cid]["relevance"] += (sim * 20.0) / (sem_count + 1)
                        commits[cid]["semantic_matches"] = sem_count + 1
                        if reason not in commits[cid]["reasons"]:
                            commits[cid]["reasons"].append(reason)
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
        return semantic_relevance_set

    def _apply_file_count_penalty(self, cursor: sqlite3.Cursor, commits: Dict):
        """
        Apply a relevance penalty to commits that modify an unusually high number of files.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            commits (Dict): The results dictionary containing current scores.
        """
        for cdict in commits.values():
            cursor.execute("SELECT count(*) FROM edges WHERE source_id = ? AND type = ?", (cdict["id"], EdgeType.MODIFIED_FILE.value))
            num_files = cursor.fetchone()[0]
            if num_files > 0:
                penalty = max(1.0, math.log10(num_files))
                if penalty > 1.0:
                    cdict["relevance"] /= (penalty)
                    cdict["reasons"].append(f"Penalized relevance (divided by {penalty:.2f}) for modifying {num_files} files")

    def _run_graph_traversal(self, cursor: sqlite3.Cursor, top_commits: List[Dict], depth: int, keywords: List[str], semantic_set: Set[str], allowed_types: Set[str], exclude_types: Set[str], all_matches: bool, all_files: bool, negative_keywords: List[str] = None, negative_semantic_set: Set[str] = None):
        """
        Perform breadth-first traversal from top results to gather connected context nodes.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            top_commits (List[Dict]): The list of top relevant commits to start traversal from.
            depth (int): The maximum traversal distance.
            keywords (List[str]): Keywords to use for relevance checks during traversal.
            semantic_set (Set[str]): The set of semantically relevant node IDs.
            allowed_types (Set[str]): Node types allowed to be added as context.
            all_matches (bool): If True, include all nodes encountered during traversal.
            all_files (bool): If True, include all modified files as context.
            negative_keywords (List[str]): Keywords to avoid in connections.
            negative_semantic_set (Set[str]): Semantic meanings to avoid in connections.
        """
        negative_keywords = negative_keywords or []
        negative_semantic_set = negative_semantic_set or set()

        for cdict in top_commits:
            cid = cdict["id"]
            queue, visited = [(cid, 0)], {cid}
            while queue:
                curr_id, dist = queue.pop(0)
                if dist >= depth: continue
                
                cursor.execute("SELECT target_id, type FROM edges WHERE source_id = ? UNION SELECT source_id, type FROM edges WHERE target_id = ?", (curr_id, curr_id))
                for nid, _ in cursor.fetchall():
                    if nid in visited: continue
                    visited.add(nid)
                    
                    cursor.execute("SELECT type, attributes FROM nodes WHERE id = ?", (nid,))
                    row = cursor.fetchone()
                    if not row: continue
                    ntype, n_attrs_json = row
                    n_attrs = json.loads(n_attrs_json) if n_attrs_json else {}

                    is_trigger = nid in cdict["connections"] and cdict["connections"][nid].get("distance", -1) == 0
                    if not all_matches and not is_trigger:
                        if not self._is_node_relevant(cursor, nid, ntype, n_attrs, keywords, semantic_set, all_files, negative_keywords, negative_semantic_set):
                            continue
                    
                    if negative_keywords or negative_semantic_set:
                        if self._is_node_avoided(cursor, nid, ntype, n_attrs, negative_keywords, negative_semantic_set):
                            continue

                    if exclude_types and (ntype in exclude_types or ntype.replace('_', ' ') in exclude_types or ntype.replace('_', '') in exclude_types):
                        continue

                    if not is_trigger:
                        cdict["connections"][nid] = {"type": ntype, "distance": dist + 1, "attributes": n_attrs}
                    
                    if ntype != NodeType.COMMIT.value and allowed_types and ntype not in allowed_types:
                        continue
                    if not allowed_types and ntype in [NodeType.AUTHOR.value, NodeType.TIME_BUCKET.value, NodeType.REPOSITORY.value]:
                        continue
                    queue.append((nid, dist + 1))

    def _is_node_relevant(self, cursor: sqlite3.Cursor, nid: str, ntype: str, attrs: Dict, keywords: List[str], semantic_set: Set[str], all_files: bool, negative_keywords: List[str] = None, negative_semantic_set: Set[str] = None) -> bool:
        """
        Determine if a connected node is relevant enough to be included in the search results.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            nid (str): The ID of the node being checked.
            ntype (str): The type of the node.
            attrs (Dict): The node's metadata attributes.
            keywords (List[str]): The search keywords.
            semantic_set (Set[str]): The set of semantically relevant node IDs.
            all_files (bool): If True, any modified file node is considered relevant.
            negative_keywords (List[str]): Excluded keywords.
            negative_semantic_set (Set[str]): Excluded meanings.

        Returns:
            bool: True if the node should be included as context.
        """
        if negative_keywords or negative_semantic_set:
            if self._is_node_avoided(cursor, nid, ntype, attrs, negative_keywords, negative_semantic_set):
                return False

        if nid in semantic_set: return True
        if ntype == NodeType.KEYWORD.value:
            return any(kw in nid.lower() or kw in str(attrs.get("name", "")).lower() for kw in keywords)
        if ntype == NodeType.SYMBOL.value: return False
        if ntype == NodeType.FILE.value and not all_files:
            cursor.execute("SELECT target_id FROM edges WHERE source_id = ? AND type IN (?, ?, ?) UNION SELECT source_id FROM edges WHERE target_id = ? AND type IN (?, ?, ?)", 
                           (nid, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value, nid, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value))
            return any(ln in semantic_set or any(kw in ln.lower() for kw in keywords) for ln, in cursor.fetchall())
        if ntype in [NodeType.COMMIT.value, NodeType.COMMIT_MESSAGE.value, NodeType.COMMENT.value, NodeType.SECRET.value, NodeType.FUNCTION.value, NodeType.PULL_REQUEST.value, NodeType.CLICKUP_TASK.value]:
            text = " ".join(str(v).lower() for v in attrs.values())
            return any(kw in text for kw in keywords)
        return True

    def _is_node_avoided(self, cursor: sqlite3.Cursor, nid: str, ntype: str, attrs: Dict, negative_keywords: List[str], negative_semantic_set: Set[str]) -> bool:
        """
        Check if a node matches negative criteria and should be avoided.
        """
        if negative_semantic_set and nid in negative_semantic_set:
            return True
        if negative_keywords:
            text = (str(nid) + " " + " ".join(str(v) for v in attrs.values())).lower()
            if any(kw in text for kw in negative_keywords):
                return True
        return False

    def _ensure_commit_in_results(self, commits: Dict, commit_id: str, attr_json: Optional[str], repo_ids: Set[str], cursor: sqlite3.Cursor):
        """
        Ensure a commit is present in the results dictionary, initializing it if necessary.

        Args:
            commits (Dict): The results dictionary.
            commit_id (str): The ID of the commit to ensure.
            attr_json (str, optional): Pre-fetched attributes JSON for the commit.
            repo_ids (Set[str]): The set of allowed repository IDs.
            cursor (sqlite3.Cursor): The database cursor.
        """
        if commit_id in commits: return
        if repo_ids:
            p = ",".join("?" * len(repo_ids))
            cursor.execute(f"SELECT 1 FROM edges WHERE source_id = ? AND target_id IN ({p}) AND type = ?", (commit_id, *repo_ids, EdgeType.PART_OF_REPO.value))
            if not cursor.fetchone(): return

        if not attr_json:
            cursor.execute("SELECT attributes FROM nodes WHERE id = ?", (commit_id,))
            row = cursor.fetchone()
            if not row: return
            attr_json = row[0]

        attrs = json.loads(attr_json)
        commits[commit_id] = {
            "id": commit_id,
            "message": attrs.get("message", "No message"),
            "author": attrs.get("author_name", attrs.get("author", "Unknown")),
            "date": attrs.get("timestamp", attrs.get("date", "Unknown")),
            "relevance": 0.0, "keyword_matches": {}, "semantic_matches": 0, "connections": {}, "reasons": []
        }

    def get_ast_nodes(self, node_id: str, max_depth: int = 0, query: str = None) -> List[Dict]:
        """
        Retrieve the AST nodes linked to a given FILE or FUNCTION_VERSION node.
        
        Args:
            node_id: The ID of the file or function version.
            max_depth: Maximum depth to traverse the AST (0 for infinite).
            query: Optional semantic query to filter AST nodes by similarity.
            
        Returns:
            A list of dictionary node representations.
        """
        import json
        from pathlib import Path
        from analysis.ast_extractor import AstExtractor
        from analysis.tree_sitter_utils import TreeSitterUtils
        from core.schema import NodeType, EdgeType
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Determine the node type (with fuzzy resolution)
            supported_types = (NodeType.FILE.value, NodeType.FUNCTION.value, NodeType.FUNCTION_VERSION.value)
            cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            
            # If exact match exists but is NOT a supported type, ignore it and try fuzzy
            if row and row[1] not in supported_types:
                row = None

            if not row:
                # Try fuzzy matching
                placeholders = ",".join("?" * len(supported_types))
                cursor.execute(f"SELECT id, type, attributes FROM nodes WHERE type IN ({placeholders}) AND (id LIKE ? OR id LIKE ?)", supported_types + (f"%{node_id}%", f"FILE:%{node_id}%"))
                fuzzy_matches = cursor.fetchall()
                
                if len(fuzzy_matches) == 1:
                    row = fuzzy_matches[0]
                    node_id = row[0]
                elif len(fuzzy_matches) > 1:
                    # Prioritize FILE nodes if multiple matches found
                    file_matches = [m for m in fuzzy_matches if m[1] == 'FILE']
                    if len(file_matches) == 1:
                        row = file_matches[0]
                        node_id = row[0]
                    else:
                        # Prefer exact basename match if possible
                        basename = node_id.split('/')[-1]
                        exact_basename_matches = [m for m in fuzzy_matches if m[0].endswith(basename)]
                        if len(exact_basename_matches) == 1:
                            row = exact_basename_matches[0]
                            node_id = row[0]
                        else:
                            # Return ambiguity error
                            suggestions = "\n".join(f"- {m[0]}" for m in fuzzy_matches[:5])
                            raise ValueError(f"Ambiguous node identifier '{node_id}'. Multiple matches found:\n{suggestions}")
            
            if not row:
                base_msg = f"Node '{node_id}' does not exist in the graph. Check the format (e.g., 'FILE:path/to/file.py' or 'FUNCTION:pkg.module.function')."
                
                # Try finding a similarly named node to suggest
                cursor.execute("SELECT id FROM nodes WHERE type IN ('FILE', 'FUNCTION', 'FUNCTION_VERSION') AND id LIKE ? LIMIT 5", (f"%{node_id}%",))
                similar_nodes = [r[0] for r in cursor.fetchall()]
                if not similar_nodes:
                    # Try even looser search (basename equivalent)
                    base_id = node_id.split('/')[-1].split(':')[-1]
                    cursor.execute("SELECT id FROM nodes WHERE type IN ('FILE', 'FUNCTION', 'FUNCTION_VERSION') AND id LIKE ? LIMIT 5", (f"%{base_id}%",))
                    similar_nodes = [r[0] for r in cursor.fetchall()]
                
                if similar_nodes:
                    suggestions = "\n".join(f"- {s}" for s in similar_nodes)
                    raise ValueError(f"{base_msg}\nDid you mean:\n{suggestions}")
                raise ValueError(base_msg)
                
            actual_id, ntype, attrs_json = row
            node_id = actual_id
            attrs = json.loads(attrs_json) if attrs_json else {}
            
            file_content = ""
            language = ""
            
            if ntype == NodeType.FILE.value:
                rel_path = node_id.replace("FILE:", "") if node_id.startswith("FILE:") else node_id
                
                # Dynamic fallback for finding the relative file path from repo root
                try_paths = [
                    Path.cwd() / rel_path,
                    Path(self.db_path).parent.parent / rel_path
                ]
                
                # Check for cached clone or root_path metadata in the REPOSITORY node
                cursor.execute("SELECT id, attributes FROM nodes WHERE type = ?", (NodeType.REPOSITORY.value,))
                repo_row = cursor.fetchone()
                if repo_row:
                    repo_id, repo_attrs_json = repo_row
                    repo_attrs = json.loads(repo_attrs_json) if repo_attrs_json else {}
                    
                    # Try root_path if available
                    root_path = repo_attrs.get("root_path")
                    if root_path:
                        try_paths.append(Path(root_path) / rel_path)
                    
                    # Try relative to the repo_id (legacy fallback)
                    try_paths.append(Path(repo_id) / rel_path)
                    
                    cache_dir = Path.home() / ".gdskg" / "cache" / repo_id
                    if cache_dir.exists():
                        try_paths.append(cache_dir / rel_path)
                
                file_path = None
                for p in try_paths:
                    if p.exists() and p.is_file():
                        file_path = p
                        break
                        
                if not file_path:
                    # Provide an informative error identifying the paths checked
                    checked = [str(p) for p in try_paths]
                    raise ValueError(f"File '{rel_path}' could not be located on the filesystem. Checked: {checked}")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    raise ValueError(f"Could not read file content for '{node_id}': {e}")
                    
                extension = attrs.get('extension', Path(file_path).suffix)
                language = TreeSitterUtils.map_extension_to_language(extension)
                
            elif ntype == NodeType.FUNCTION_VERSION.value:
                file_content = attrs.get('content', '')
                if not file_content:
                    raise ValueError(f"Function version '{node_id}' has no content stored in its attributes.")
                
                file_path = attrs.get('file_path', '')
                extension = Path(file_path).suffix
                language = TreeSitterUtils.map_extension_to_language(extension)
                
            elif ntype == NodeType.FUNCTION.value:
                cursor.execute("""
                    SELECT target_id FROM edges WHERE source_id = ? AND type = ?
                """, (node_id, EdgeType.HAS_HISTORY.value))
                history_row = cursor.fetchone()
                if not history_row:
                    raise ValueError(f"Function '{node_id}' has no history edge in the graph.")
                
                cursor.execute("""
                    SELECT n.id, n.attributes FROM nodes n
                    JOIN edges e ON n.id = e.target_id
                    WHERE e.source_id = ? AND e.type = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM edges e2 WHERE e2.source_id = n.id AND e2.type = ?
                    )
                """, (history_row[0], EdgeType.HAS_VERSION.value, EdgeType.PREVIOUS_VERSION.value))
                
                v_row = cursor.fetchone()
                if not v_row:
                    raise ValueError(f"Function '{node_id}' has a history group but no latest version node.")
                
                v_attrs = json.loads(v_row[1])
                file_content = v_attrs.get('content', '')
                extension = Path(v_attrs.get('file_path', '')).suffix
                language = TreeSitterUtils.map_extension_to_language(extension)
                if not file_content:
                    raise ValueError(f"Latest version for function '{node_id}' has no stored content.")
                
            else:
                raise ValueError(f"Node '{node_id}' is of type '{ntype}'. AST extraction is only supported for FILE, FUNCTION, and FUNCTION_VERSION types.")
                
        if not language:
            raise ValueError(f"Could determine correct tree-sitter language for '{node_id}'.")
            
        # 2. Extract and filter AST nodes in memory
        nodes = AstExtractor.get_ast_nodes(file_content, language, max_depth, query)
        if not nodes:
            raise ValueError(f"AST extraction returned no nodes. The language '{language}' might not be installed or parseable, or the content could be empty.")
        return nodes

    def get_dependencies(self, node_id: str) -> Dict[str, Any]:
        """
        Retrieve incoming and outgoing dependencies for a given node.
        
        Args:
            node_id: The ID of the node to analyze.
            
        Returns:
            A dictionary containing the node ID and its dependencies/dependents.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Resolve node ID (fuzzy)
            cursor.execute("SELECT id, type FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            
            if not row:
                cursor.execute("SELECT id, type FROM nodes WHERE type IN ('FILE', 'FUNCTION', 'SYMBOL', 'COMMIT') AND (id LIKE ? OR id LIKE ?)", (f"%{node_id}%", f"FILE:%{node_id}%"))
                matches = cursor.fetchall()
                if len(matches) == 1:
                    row = matches[0]
                elif len(matches) > 1:
                    basename = node_id.split('/')[-1]
                    exact_basename_matches = [m for m in matches if m[0].endswith(basename)]
                    if len(exact_basename_matches) == 1:
                        row = exact_basename_matches[0]
                    else:
                        suggestions = "\n".join(f"- {m[0]}" for m in matches[:5])
                        raise ValueError(f"Ambiguous node identifier '{node_id}'. Multiple matches found:\n{suggestions}")
            
            if not row:
                 raise ValueError(f"Node '{node_id}' not found in graph.")
                 
            resolved_id, ntype = row
            
            # 2. Get outgoing edges (dependencies)
            # We join with 'nodes' to get the type and attributes of the target
            cursor.execute("""
                SELECT e.target_id, e.type, n.type, n.attributes
                FROM edges e 
                JOIN nodes n ON e.target_id = n.id 
                WHERE e.source_id = ? 
                AND e.type NOT IN ('CONTAINS_COMMENT', 'HAS_KEYWORD')
            """, (resolved_id,))
            
            dependencies = []
            for tid, etype, target_ntype, attrs_json in cursor.fetchall():
                attrs = json.loads(attrs_json) if attrs_json else {}
                # Extract a readable name
                name = attrs.get('name')
                if not name:
                    if target_ntype == NodeType.FILE.value:
                        name = tid.split('/')[-1].split(':')[-1]
                    elif target_ntype == NodeType.COMMIT.value:
                        name = attrs.get('message', '').split('\n')[0] or tid[:8]
                    else:
                        name = tid
                
                dependencies.append({
                    "id": tid, 
                    "name": name,
                    "type": etype, 
                    "node_type": target_ntype
                })
            
            # 3. Get incoming edges (dependents)
            cursor.execute("""
                SELECT e.source_id, e.type, n.type, n.attributes
                FROM edges e 
                JOIN nodes n ON e.source_id = n.id 
                WHERE e.target_id = ? 
                AND e.type NOT IN ('CONTAINS_COMMENT', 'HAS_KEYWORD')
            """, (resolved_id,))
            
            dependents = []
            for sid, etype, source_ntype, attrs_json in cursor.fetchall():
                attrs = json.loads(attrs_json) if attrs_json else {}
                # Extract a readable name
                name = attrs.get('name')
                if not name:
                    if source_ntype == NodeType.FILE.value:
                        name = sid.split('/')[-1].split(':')[-1]
                    elif source_ntype == NodeType.COMMIT.value:
                        name = attrs.get('message', '').split('\n')[0] or sid[:8]
                    else:
                        name = sid
                
                dependents.append({
                    "id": sid, 
                    "name": name,
                    "type": etype, 
                    "node_type": source_ntype
                })
            
            return {
                "node": resolved_id,
                "type": ntype,
                "dependencies": dependencies,
                "dependents": dependents
            }

    def get_node_types(self) -> List[str]:
        """
        Retrieve all unique node types currently present in the knowledge graph.
        
        Returns:
            List[str]: A list of alphabetical node types.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT type FROM nodes ORDER BY type")
            return [row[0] for row in cursor.fetchall()]

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

    def search(self, query: str = "", repo_name: str = None, depth: int = 1, traverse_types: List[str] = None, semantic_only: bool = False, min_score: float = 10.0, top_n: int = 5, filters: Dict[str, str] = None, all_matches: bool = False, all_files: bool = False) -> List[Dict[str, Any]]:
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

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the most relevant commits and their context.
        """
        query = query or ""
        keywords = self._extract_keywords(query)
        commits = {}
        allowed_types = {t.upper() for t in traverse_types} if traverse_types else set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            repo_ids = self._get_repo_ids(cursor, repo_name) if repo_name else set()
            if repo_name and not repo_ids:
                return []

            allowed_commits = self._apply_strict_filters(cursor, filters)
            if filters and allowed_commits is None:
                return []

            if allowed_commits is not None and not keywords and not query:
                self._add_filter_only_matches(commits, allowed_commits, repo_ids, cursor, filters)

            search_keywords = [] if semantic_only else keywords
            self._search_keywords(cursor, search_keywords, commits, repo_ids, allowed_commits)
            
            semantic_relevance_set = self._run_semantic_search(query, commits, cursor, repo_ids, allowed_commits)
            self._apply_file_count_penalty(cursor, commits)

            sorted_commits = [r for r in commits.values() if r["relevance"] >= min_score]
            sorted_commits.sort(key=lambda x: x["relevance"], reverse=True)
            top_commits = sorted_commits[:top_n]

            if depth > 0:
                self._run_graph_traversal(cursor, top_commits, depth, keywords, semantic_relevance_set, allowed_types, all_matches, all_files)

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

    def _search_keywords(self, cursor: sqlite3.Cursor, keywords: List[str], commits: Dict, repo_ids: Set[str], allowed_commits: Optional[Set[str]]):
        """
        Execute keyword-based search against the graph and update relevance scores.

        Args:
            cursor (sqlite3.Cursor): The database cursor.
            keywords (List[str]): The keywords to search for.
            commits (Dict): The results dictionary to populate.
            repo_ids (Set[str]): The set of allowed repository IDs.
            allowed_commits (Set[str], optional): The set of allowed commit IDs from strict filters.
        """
        for kw in keywords:
            cursor.execute("""
                SELECT id, type, attributes FROM nodes 
                WHERE id LIKE ? OR json_extract(attributes, '$.name') LIKE ? OR json_extract(attributes, '$.message') LIKE ? OR 
                      json_extract(attributes, '$.content') LIKE ? OR json_extract(attributes, '$.author_name') LIKE ? OR json_extract(attributes, '$.author_email') LIKE ?
            """, (f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"))
            
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

    def _run_semantic_search(self, query: str, commits: Dict, cursor: sqlite3.Cursor, repo_ids: Set[str], allowed_commits: Optional[Set[str]]) -> Set[str]:
        """
        Execute semantic search using vector embeddings and update relevance scores.

        Args:
            query (str): The user's query string for similarity matching.
            commits (Dict): The results dictionary to update.
            cursor (sqlite3.Cursor): The database cursor.
            repo_ids (Set[str]): The set of allowed repository IDs.
            allowed_commits (Set[str], optional): The set of allowed commit IDs from filters.

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

            for nid, ntype, sim in all_results[:20]:
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
                elif ntype in [NodeType.COMMIT.value, NodeType.COMMIT_MESSAGE.value, "COMMIT_SYMBOLS"]:
                    linked_commits = [nid]
                    reason = f"Similar meaning identified in commit context (matched {ntype} score={sim:.2f})"
                
                for cid in linked_commits:
                    if allowed_commits is not None and cid not in allowed_commits:
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

    def _run_graph_traversal(self, cursor: sqlite3.Cursor, top_commits: List[Dict], depth: int, keywords: List[str], semantic_set: Set[str], allowed_types: Set[str], all_matches: bool, all_files: bool):
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
        """
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
                        if not self._is_node_relevant(cursor, nid, ntype, n_attrs, keywords, semantic_set, all_files):
                            continue
                    
                    if not is_trigger:
                        cdict["connections"][nid] = {"type": ntype, "distance": dist + 1, "attributes": n_attrs}
                    
                    if ntype != NodeType.COMMIT.value and allowed_types and ntype not in allowed_types:
                        continue
                    if not allowed_types and ntype in [NodeType.AUTHOR.value, NodeType.TIME_BUCKET.value, NodeType.REPOSITORY.value]:
                        continue
                    queue.append((nid, dist + 1))

    def _is_node_relevant(self, cursor: sqlite3.Cursor, nid: str, ntype: str, attrs: Dict, keywords: List[str], semantic_set: Set[str], all_files: bool) -> bool:
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

        Returns:
            bool: True if the node should be included as context.
        """
        if nid in semantic_set: return True
        if ntype == NodeType.KEYWORD.value:
            return any(kw in nid.lower() or kw in str(attrs.get("name", "")).lower() for kw in keywords)
        if ntype == NodeType.SYMBOL.value: return False
        if ntype == NodeType.FILE.value and not all_files:
            cursor.execute("SELECT target_id FROM edges WHERE source_id = ? AND type IN (?, ?, ?) UNION SELECT source_id FROM edges WHERE target_id = ? AND type IN (?, ?, ?)", 
                           (nid, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value, nid, EdgeType.HAS_SYMBOL.value, EdgeType.HAS_KEYWORD.value, EdgeType.MODIFIED_SYMBOL.value))
            return any(ln in semantic_set or any(kw in ln.lower() for kw in keywords) for ln, in cursor.fetchall())
        if ntype in [NodeType.COMMIT.value, NodeType.COMMIT_MESSAGE.value, NodeType.COMMENT.value, NodeType.SECRET.value, NodeType.FUNCTION.value]:
            text = " ".join(str(v).lower() for v in attrs.values())
            return any(kw in text for kw in keywords)
        return True

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

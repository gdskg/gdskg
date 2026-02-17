import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from core.schema import NodeType, EdgeType

class SearchEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def search(self, query: str, repo_name: str = None, depth: int = 1, traverse_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant commits based on keywords.
        1. Find nodes (COMMIT, SYMBOL, FILE) matching query.
        2. Traverse to find related COMMIT nodes.
        3. Filter by repository if repo_name is provided.
        4. Traverse neighbors up to the specified depth.
        5. Return unique commit details with connections.
        """
        keywords = query.lower().split()
        if not keywords:
            return []
        
        allowed_types = set()
        if traverse_types:
            allowed_types = {t.upper() for t in traverse_types}

        commits = {} # hash -> metadata
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # --- Get valid repository IDs if repo_name is provided ---
            repo_ids = set()
            if repo_name:
                cursor.execute("""
                    SELECT id FROM nodes 
                    WHERE type = ? AND json_extract(attributes, '$.name') = ?
                """, (NodeType.REPOSITORY.value, repo_name))
                repo_ids = {r[0] for r in cursor.fetchall()}
                if not repo_ids:
                    return [] # No such repository

            # --- Search for nodes directly matching keywords ---
            for kw in keywords:
                # 1. Search COMMIT nodes
                cursor.execute("""
                    SELECT id, attributes FROM nodes 
                    WHERE type = ? AND (id LIKE ? OR attributes LIKE ?)
                """, (NodeType.COMMIT.value, f"%{kw}%", f"%{kw}%"))
                for row in cursor.fetchall():
                    commit_id, attr_json = row
                    
                    # Filter by repo if needed
                    if repo_ids:
                        cursor.execute("""
                            SELECT 1 FROM edges 
                            WHERE source_id = ? AND target_id IN (%s) AND type = ?
                        """ % ",".join("?" * len(repo_ids)), (commit_id, *repo_ids, EdgeType.PART_OF_REPO.value))
                        if not cursor.fetchone():
                            continue

                    if commit_id not in commits:
                        attrs = json.loads(attr_json)
                        commits[commit_id] = {
                            "id": commit_id,
                            "message": attrs.get("message", "No message"),
                            "author": attrs.get("author_name", attrs.get("author", "Unknown")),
                            "date": attrs.get("timestamp", attrs.get("date", "Unknown")),
                            "relevance": 0,
                            "connections": {} # node_id -> {"type": NodeType, "distance": int}
                        }
                    commits[commit_id]["relevance"] += 10

                # 2. Search SYMBOL, FILE, and COMMIT_MESSAGE nodes
                cursor.execute("""
                    SELECT id, type FROM nodes 
                    WHERE (type = ? OR type = ? OR type = ?) AND (id LIKE ? OR attributes LIKE ?)
                """, (NodeType.SYMBOL.value, NodeType.FILE.value, NodeType.COMMIT_MESSAGE.value, f"%{kw}%", f"%{kw}%"))
                
                matched_rows = cursor.fetchall()
                for node_id, n_type in matched_rows:
                    if n_type == NodeType.COMMIT_MESSAGE.value:
                        # Follow HAS_MESSAGE back to COMMIT (target is message, source is commit)
                        cursor.execute("""
                            SELECT source_id FROM edges 
                            WHERE target_id = ? AND type = ?
                        """, (node_id, EdgeType.HAS_MESSAGE.value))
                    else:
                        cursor.execute("""
                            SELECT target_id FROM edges 
                            WHERE source_id = ? AND (type = ? OR type = ?)
                        """, (node_id, EdgeType.MODIFIED_SYMBOL.value, EdgeType.MODIFIED_FILE.value))
                    
                    linked_commits = [r[0] for r in cursor.fetchall()]
                    for cid in linked_commits:
                        # Filter by repo if needed
                        if repo_ids:
                            cursor.execute("""
                                SELECT 1 FROM edges 
                                WHERE source_id = ? AND target_id IN (%s) AND type = ?
                            """ % ",".join("?" * len(repo_ids)), (cid, *repo_ids, EdgeType.PART_OF_REPO.value))
                            if not cursor.fetchone():
                                continue

                        if cid not in commits:
                            cursor.execute("SELECT attributes FROM nodes WHERE id = ?", (cid,))
                            c_row = cursor.fetchone()
                            if c_row:
                                c_attrs = json.loads(c_row[0])
                                commits[cid] = {
                                    "id": cid,
                                    "message": c_attrs.get("message", "No message"),
                                    "author": c_attrs.get("author_name", c_attrs.get("author", "Unknown")),
                                    "date": c_attrs.get("timestamp", c_attrs.get("date", "Unknown")),
                                    "relevance": 0,
                                    "connections": {}
                                }
                        commits[cid]["relevance"] += 5

            # --- Fetch Connections using BFS per commit ---
            if depth > 0:
                for cid in commits:
                    queue = [(cid, 0)] # (node_id, current_distance)
                    visited = {cid}
                    
                    while queue:
                        curr_id, dist = queue.pop(0)
                        if dist >= depth:
                            continue
                            
                        # Find all neighbors (incoming and outgoing)
                        cursor.execute("""
                            SELECT target_id, type FROM edges WHERE source_id = ?
                            UNION
                            SELECT source_id, type FROM edges WHERE target_id = ?
                        """, (curr_id, curr_id))
                        
                        for neighbor_id, _ in cursor.fetchall():
                            if neighbor_id not in visited:
                                visited.add(neighbor_id)
                                
                                # Fetch node type and attributes
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
                                    
                                    # Logic to stop traversing through nodes
                                    # If allowed types are specified (strict whitelist mode)
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

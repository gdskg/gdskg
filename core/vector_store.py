import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os

from core.schema import NodeType, EdgeType

class VectorStore:
    """Persistent storage for node embeddings using SQLite and numpy."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the VectorStore with the specified SQLite database path.

        Args:
            db_path (Path, optional): Path to the SQLite database file. 
                Defaults to ~/.gdskg/vector_db/gdskg_vectors.db.
        """
        if db_path is None:
            vector_dir = Path(os.environ.get("GDSKG_VECTOR_DB_DIR", Path.home() / ".gdskg" / "vector_db"))
            vector_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = vector_dir / "gdskg_vectors.db"
        elif db_path.is_dir():
            self.db_path = db_path / "gdskg_vectors.db"
        else:
            self.db_path = db_path
        self._init_db()
        
    def _init_db(self) -> None:
        """
        Initialize the SQLite database schema for embeddings.

        Returns:
            None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    node_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    node_type TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (node_id, chunk_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_type ON embeddings(node_type)")
            conn.commit()
            
    def upsert_embedding(self, node_id: str, node_type: str, embeddings: List[np.ndarray]) -> None:
        """
        Store or update the embeddings (chunks) for a node.

        Args:
            node_id (str): The unique identifier of the node.
            node_type (str): The type of the node.
            embeddings (List[np.ndarray]): A list of vector embeddings for the node's content.

        Returns:
            None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embeddings WHERE node_id = ?", (node_id,))
            
            for chunk_id, emb in enumerate(embeddings):
                emb_blob = emb.astype(np.float32).tobytes()
                cursor.execute("""
                    INSERT INTO embeddings (node_id, chunk_id, node_type, embedding)
                    VALUES (?, ?, ?, ?)
                """, (node_id, chunk_id, node_type, emb_blob))
            conn.commit()

    def upsert_embeddings(self, items: List[Tuple[str, str, List[np.ndarray]]]) -> None:
        """
        Batch store embeddings (chunks) for multiple nodes.

        Args:
            items (List[Tuple[str, str, List[np.ndarray]]]): A list of (node_id, node_type, embeddings) tuples.

        Returns:
            None
        """
        if not items:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            node_ids = list({item[0] for item in items})
            for i in range(0, len(node_ids), 900):
                batch_ids = node_ids[i:i+900]
                placeholders = ",".join(["?"] * len(batch_ids))
                cursor.execute(f"DELETE FROM embeddings WHERE node_id IN ({placeholders})", tuple(batch_ids))
            
            data = []
            for node_id, node_type, embeddings in items:
                for chunk_id, emb in enumerate(embeddings):
                    data.append((node_id, chunk_id, node_type, emb.astype(np.float32).tobytes()))
            
            cursor.executemany("""
                INSERT INTO embeddings (node_id, chunk_id, node_type, embedding)
                VALUES (?, ?, ?, ?)
            """, data)
            conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int = 5, node_types: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Compute cosine similarity between the query embedding and stored embeddings.

        Args:
            query_embedding (np.ndarray): The vector embedding of the search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            node_types (List[str], optional): Filter results to specific node types. Defaults to None.

        Returns:
            List[Tuple[str, str, float]]: A list of (node_id, node_type, similarity) tuples.
        """
        query_emb = query_embedding.astype(np.float32)
        if query_emb.ndim > 1:
            query_emb = np.mean(query_emb, axis=0)
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0:
            return []
            
        results = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if node_types:
                placeholders = ",".join("?" * len(node_types))
                cursor.execute(f"SELECT node_id, node_type, embedding FROM embeddings WHERE node_type IN ({placeholders})", tuple(node_types))
            else:
                cursor.execute("SELECT node_id, node_type, embedding FROM embeddings")
                
            for row in cursor:
                node_id, node_type, emb_blob = row
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                norm_emb = np.linalg.norm(emb)
                
                if norm_emb == 0:
                    continue
                    
                sim = float(np.dot(query_emb, emb) / (norm_q * norm_emb))
                
                if node_id not in results or sim > results[node_id][1]:
                    results[node_id] = (node_type, sim)
                
        sorted_results = [(nid, ntype, sim) for nid, (ntype, sim) in results.items()]
        sorted_results.sort(key=lambda x: x[2], reverse=True)
        return sorted_results[:top_k]

    def build_from_graph(self, graph_path: str, embedder, progress_callback=None) -> int:
        """
        Extract nodes from graph, embed them, and store them.

        Args:
            graph_path (str): Path to the GraphStore SQLite database.
            embedder (ONNXEmbedder): The embedder instance to use.
            progress_callback (callable, optional): Callback for progress updates.

        Returns:
            int: The total number of embedding chunks stored.
        """
        nodes_to_embed = []
        
        with sqlite3.connect(graph_path) as conn:
            cursor = conn.cursor()
            self._extract_symbols(cursor, nodes_to_embed)
            self._extract_latest_function_versions(cursor, nodes_to_embed)
            self._extract_commits_and_symbols(cursor, nodes_to_embed)

        return self._batch_embed_and_store(nodes_to_embed, embedder, progress_callback)

    def _extract_symbols(self, cursor: sqlite3.Cursor, nodes_to_embed: list) -> None:
        """
        Extract all symbol nodes from the graph for embedding.

        Args:
            cursor (sqlite3.Cursor): The graph database cursor.
            nodes_to_embed (list): The list to append extraction results to.
        """
        cursor.execute("SELECT id, attributes FROM nodes WHERE type = ?", (NodeType.SYMBOL.value,))
        for node_id, attr_json in cursor.fetchall():
            try:
                attrs = json.loads(attr_json)
                content = attrs.get('name', '')
                if content:
                    nodes_to_embed.append((node_id, NodeType.SYMBOL.value, content))
            except Exception:
                pass

    def _extract_latest_function_versions(self, cursor: sqlite3.Cursor, nodes_to_embed: list) -> None:
        """
        Identify and extract the most recent version of each function for semantic indexing.

        Args:
            cursor (sqlite3.Cursor): The graph database cursor.
            nodes_to_embed (list): The list to append extraction results to.
        """
        cursor.execute("SELECT id FROM nodes WHERE type = ?", (NodeType.FUNCTION.value,))
        function_ids = [r[0] for r in cursor.fetchall()]
        
        for f_id in function_ids:
            cursor.execute("""
                SELECT v.id, v.attributes
                FROM edges h_edge
                JOIN edges v_edge ON h_edge.target_id = v_edge.source_id
                JOIN nodes v ON v_edge.target_id = v.id
                WHERE h_edge.source_id = ? 
                  AND h_edge.type = 'HAS_HISTORY' 
                  AND v_edge.type = 'HAS_VERSION'
            """, (f_id,))
            
            versions = cursor.fetchall()
            if not versions:
                continue
                
            v_ids = [v[0] for v in versions]
            v_map = {v[0]: v[1] for v in versions}
            
            placeholders = ",".join("?" * len(v_ids))
            cursor.execute(f"SELECT source_id FROM edges WHERE type='PREVIOUS_VERSION' AND source_id IN ({placeholders})", tuple(v_ids))
            has_next = {r[0] for r in cursor.fetchall()}
            
            latest_v_ids = [vid for vid in v_ids if vid not in has_next]
            
            for latest_vid in latest_v_ids:
                attr_json = v_map[latest_vid]
                try:
                    attrs = json.loads(attr_json)
                    content = attrs.get('content', '')
                    if content:
                        nodes_to_embed.append((f_id, NodeType.FUNCTION.value, content))
                except Exception:
                    pass

    def _extract_commits_and_symbols(self, cursor: sqlite3.Cursor, nodes_to_embed: list) -> None:
        """
        Extract commit messages and the names of modified symbols for semantic indexing.

        Args:
            cursor (sqlite3.Cursor): The graph database cursor.
            nodes_to_embed (list): The list to append extraction results to.
        """
        cursor.execute("SELECT id, attributes FROM nodes WHERE type = ?", (NodeType.COMMIT.value,))
        for c_id, attr_json in cursor.fetchall():
            try:
                attrs = json.loads(attr_json)
                message = attrs.get('message', '')
                if message:
                    nodes_to_embed.append((c_id, NodeType.COMMIT_MESSAGE.value, message))
            except Exception:
                pass
            
            cursor.execute("""
                SELECT n.attributes FROM edges e
                JOIN nodes n ON e.target_id = n.id
                WHERE e.source_id = ? AND e.type = ? AND n.type = ?
            """, (c_id, EdgeType.MODIFIED_SYMBOL.value, NodeType.SYMBOL.value))
            
            commit_symbols = []
            for (s_attr_json,) in cursor.fetchall():
                try:
                    s_attrs = json.loads(s_attr_json)
                    name = s_attrs.get('name', '')
                    if name:
                        commit_symbols.append(name)
                except Exception:
                    pass
            
            if commit_symbols:
                all_symbols_text = " ".join(commit_symbols)
                nodes_to_embed.append((c_id, "COMMIT_SYMBOLS", all_symbols_text))

    def _batch_embed_and_store(self, nodes_to_embed: list, embedder, progress_callback) -> int:
        """
        Process a list of nodes through the embedder and store the results in batches.

        Args:
            nodes_to_embed (list): List of (id, type, content) tuples.
            embedder: The embedder instance.
            progress_callback: Callback for progress updates.

        Returns:
            int: Total number of chunks embedded.
        """
        embed_count = 0
        batch_size = 32
        
        if progress_callback:
            progress_callback(0, len(nodes_to_embed))
            
        for i in range(0, len(nodes_to_embed), batch_size):
            batch = nodes_to_embed[i:i+batch_size]
            texts = [item[2] for item in batch]
            
            embeddings_list = embedder.embed(texts)
            
            store_items = []
            for j, embeddings in enumerate(embeddings_list):
                if len(embeddings) > 0:
                    store_items.append((batch[j][0], batch[j][1], embeddings))
                
            self.upsert_embeddings(store_items)
            
            chunks_embedded = sum(len(embs) for _, _, embs in store_items)
            embed_count += chunks_embedded
            
            if progress_callback:
                progress_callback(len(store_items))
            
        return embed_count

    def close(self) -> None:
        """
        Close any open connections or resources (placeholder for future use).

        Returns:
            None
        """
        pass


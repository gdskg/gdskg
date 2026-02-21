import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os

from core.schema import NodeType

class VectorStore:
    """
    Persistent storage for node embeddings using SQLite and numpy for similarity matching.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_type ON embeddings(node_type)")
            conn.commit()
            
    def upsert_embedding(self, node_id: str, node_type: str, embedding: np.ndarray) -> None:
        """Store or update the embedding for a node."""
        emb_blob = embedding.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO embeddings (node_id, node_type, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_type = excluded.node_type,
                    embedding = excluded.embedding
            """, (node_id, node_type, emb_blob))
            conn.commit()

    def upsert_embeddings(self, items: List[Tuple[str, str, np.ndarray]]) -> None:
        """Batch store embeddings."""
        if not items:
            return
            
        data = []
        for node_id, node_type, emb in items:
            data.append((node_id, node_type, emb.astype(np.float32).tobytes()))
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO embeddings (node_id, node_type, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    node_type = excluded.node_type,
                    embedding = excluded.embedding
            """, data)
            conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int = 5, node_types: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Compute cosine similarity between the query embedding and all stored embeddings.
        Returns top_k results.
        """
        query_emb = query_embedding.astype(np.float32)
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0:
            return []
            
        results = []
        
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
                    
                # Cosine similarity
                sim = np.dot(query_emb, emb) / (norm_q * norm_emb)
                results.append((node_id, node_type, float(sim)))
                
        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def build_from_graph(self, graph_path: str, embedder, progress_callback=None) -> int:
        """
        Extract the latest version of FUNCTION nodes, and all SYMBOL nodes,
        embed them, and store them.
        """
        import json
        
        nodes_to_embed = []
        
        # We need to find the latest version of functions, and all symbols.
        with sqlite3.connect(graph_path) as conn:
            cursor = conn.cursor()
            
            # Get Symbols
            cursor.execute("SELECT id, attributes FROM nodes WHERE type = ?", (NodeType.SYMBOL.value,))
            for node_id, attr_json in cursor.fetchall():
                try:
                    attrs = json.loads(attr_json)
                    content = attrs.get('name', '')
                    if content:
                        nodes_to_embed.append((node_id, NodeType.SYMBOL.value, content))
                except Exception:
                    pass
            
            # Get latest version of functions
            # A function node has a HAS_HISTORY edge to a FUNCTION_HISTORY node.
            # Then the history has HAS_VERSION edges to FUNCTION_VERSION nodes.
            # The latest version is the target of a HAS_VERSION edge that is NOT the source of a PREVIOUS_VERSION edge, or simply by looking at the content.
            # For simplicity, we can fetch all FUNCTION_VERSION nodes, group by history, and select latest if order is maintained,
            # or simply fetch all functions, and their history, and traverse to the latest.
            
            cursor.execute("SELECT id FROM nodes WHERE type = ?", (NodeType.FUNCTION.value,))
            function_ids = [r[0] for r in cursor.fetchall()]
            
            for f_id in function_ids:
                # Get the latest version content for this function
                # Note: this is a heuristic approach to find the latest version text quickly
                # The server.py history logic does a more complex graph traversal.
                
                # Fetch all versions for this function's histories
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
                    
                # Let's find the latest (the one not pointing to next)
                # Or simply the one with no source_id in PREVIOUS_VERSION
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

        # Batch embed and store
        embed_count = 0
        batch_size = 32
        
        if progress_callback:
            progress_callback(0, len(nodes_to_embed))
            
        for i in range(0, len(nodes_to_embed), batch_size):
            batch = nodes_to_embed[i:i+batch_size]
            texts = [item[2] for item in batch]
            
            # Embed texts
            embeddings = embedder.embed(texts)
            
            store_items = []
            for j, emb in enumerate(embeddings):
                store_items.append((batch[j][0], batch[j][1], emb))
                
            self.upsert_embeddings(store_items)
            embed_count += len(store_items)
            if progress_callback:
                progress_callback(len(store_items))
            
        return embed_count

    def close(self) -> None:
        pass

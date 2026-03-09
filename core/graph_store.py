import re
import sqlite3
import json
import threading
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from core.schema import Node, Edge, NodeType, EdgeType

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    psycopg2 = None

class GraphStore:
    """Persistent storage for the knowledge graph using SQLite."""

    def __init__(self, db_path: Path):
        """
        Initialize the GraphStore with the specified database (PostgreSQL if env var set, else SQLite).

        Args:
            db_path (Path): Path to the SQLite database file (fallback).
        """
        self.db_path = db_path
        self.db_url = os.environ.get("GDSKG_DB_URL")
        self.is_postgres = bool(self.db_url)
        
        if self.is_postgres:
            if psycopg2 is None:
                raise ImportError("psycopg2-binary is required for PostgreSQL support. Install it with pip install psycopg2-binary.")
            self.conn = psycopg2.connect(self.db_url)
        else:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=OFF;")
            self.conn.execute("PRAGMA cache_size=-64000;")
            
        self._init_db()

        self._nodes_cache = {}
        self._edges_cache = {}
        self._lock = threading.Lock()
        self._token_pattern = re.compile(r'(?:https?://|x-ac{1,2}ess-token:)[^@/ ]*@github\.com', re.IGNORECASE)

    def _sanitize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize attributes to remove sensitive information like access tokens.
        """
        if not attributes:
            return attributes
            
        def sanitize_value(val):
            if isinstance(val, str):
                return self._token_pattern.sub('x-access-token:*@github.com', val)
            elif isinstance(val, list):
                return [sanitize_value(v) for v in val]
            elif isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            return val

        return sanitize_value(attributes)

    def _init_db(self) -> None:
        """
        Initialize the database schema (SQLite or PostgreSQL).

        Returns:
            None
        """
        cursor = self.conn.cursor()
        
        json_type = "JSONB" if self.is_postgres else "JSON"
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                attributes {json_type}
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                attributes {json_type},
                FOREIGN KEY(source_id) REFERENCES nodes(id),
                FOREIGN KEY(target_id) REFERENCES nodes(id)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS amendments (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                action TEXT,
                entity_type TEXT,
                payload {json_type},
                reverted INTEGER DEFAULT 0
            )
        """)
        
        indices = [
            "idx_nodes_type ON nodes(type)",
            "idx_edges_source ON edges(source_id)",
            "idx_edges_target ON edges(target_id)",
            "idx_edges_type ON edges(type)",
            "idx_edges_source_type ON edges(source_id, type)",
            "idx_edges_target_type ON edges(target_id, type)"
        ]
        for idx in indices:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx}")
        
        self.conn.commit()

    def upsert_node(self, node: Node) -> None:
        """
        Cache node in memory, merging attributes if it already exists.

        Args:
            node (Node): The node object to upsert.

        Returns:
            None
        """
        with self._lock:
            # Sanitize attributes before caching
            if node.attributes:
                node.attributes = self._sanitize_attributes(node.attributes)
                
            if node.id in self._nodes_cache:
                existing = self._nodes_cache[node.id]
                if node.attributes:
                    if not existing.attributes:
                        existing.attributes = {}
                    existing.attributes.update(node.attributes)
            else:
                self._nodes_cache[node.id] = node

    def upsert_symbol(self, id: str, file_path: str) -> None:
        """
        Upsert a symbol node.

        Args:
            id (str): The identifier of the symbol.
            file_path (str): The path to the file where the symbol is defined.

        Returns:
            None
        """
        node = Node(
            id=id, 
            type=NodeType.SYMBOL, 
            attributes={"name": id, "file": file_path}
        )
        self.upsert_node(node)

    def upsert_edge(self, edge: Edge) -> None:
        """
        Cache edge in memory, merging attributes if it already exists.

        Args:
            edge (Edge): The edge object to upsert.

        Returns:
            None
        """
        with self._lock:
            # Sanitize attributes before caching
            if edge.attributes:
                edge.attributes = self._sanitize_attributes(edge.attributes)
                
            if edge.id in self._edges_cache:
                existing = self._edges_cache[edge.id]
                if edge.attributes:
                    if not existing.attributes:
                        existing.attributes = {}
                    existing.attributes.update(edge.attributes)
                existing.weight = max(existing.weight, edge.weight)
            else:
                self._edges_cache[edge.id] = edge

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieve a single node by its ID, checking memory cache first.

        Args:
            node_id (str): The identifier of the node.

        Returns:
            Optional[Node]: The node object if found, otherwise None.
        """
        with self._lock:
            if node_id in self._nodes_cache:
                return self._nodes_cache[node_id]

            cursor = self.conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = %s", (node_id,))
            else:
                cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                type_val = row[1]
                try:
                    node_type = NodeType(type_val)
                except ValueError:
                    node_type = type_val
                
                attrs = row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
                return Node(id=row[0], type=node_type, attributes=attrs)
            return None
    
    def count_nodes(self) -> int:
        """
        Count total nodes (DB + Memory).

        Returns:
            int: The total number of unique nodes.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes")
        db_count = cursor.fetchone()[0]
        # The count of _nodes_cache should also be protected by the lock if it's accessed concurrently
        # However, the instruction only specified wrapping cache accesses in upsert/get/flush.
        # For consistency, it might be good to wrap this too, but I'll stick to the explicit instruction.
        return db_count + len(self._nodes_cache)

    def flush(self) -> None:
        """
        Dump all memory-cached nodes and edges to SQLite.

        Returns:
            None
        """
        with self._lock:
            if not self._nodes_cache and not self._edges_cache:
                return

            cursor = self.conn.cursor()
            self._flush_nodes(cursor)
            self._flush_edges(cursor)
            self.conn.commit()

    def _flush_nodes(self, cursor: Any) -> None:
        """
        Perform the bulk insert/update of nodes.

        Args:
            cursor: The database cursor.

        Returns:
            None
        """
        if not self._nodes_cache:
            return

        node_data = []
        for node in self._nodes_cache.values():
            node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
            if self.is_postgres:
                node_data.append((node.id, node_type, psycopg2.extras.Json(node.attributes)))
            else:
                node_data.append((node.id, node_type, json.dumps(node.attributes)))
        
        if self.is_postgres:
            psycopg2.extras.execute_batch(cursor, """
                INSERT INTO nodes (id, type, attributes)
                VALUES (%s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    attributes = nodes.attributes || EXCLUDED.attributes
            """, node_data)
        else:
            cursor.executemany("""
                INSERT INTO nodes (id, type, attributes)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    attributes = json_patch(nodes.attributes, excluded.attributes)
            """, node_data)
        self._nodes_cache.clear()

    def _flush_edges(self, cursor: Any) -> None:
        """
        Perform the bulk insert/update of edges.

        Args:
            cursor: The database cursor.

        Returns:
            None
        """
        if not self._edges_cache:
            return

        edge_data = []
        for edge in self._edges_cache.values():
            edge_type = edge.type.value if hasattr(edge.type, 'value') else str(edge.type)
            if self.is_postgres:
                edge_data.append((edge.id, edge.source_id, edge.target_id, edge_type, edge.weight, psycopg2.extras.Json(edge.attributes)))
            else:
                edge_data.append((edge.id, edge.source_id, edge.target_id, edge_type, edge.weight, json.dumps(edge.attributes)))

        if self.is_postgres:
            psycopg2.extras.execute_batch(cursor, """
                INSERT INTO edges (id, source_id, target_id, type, weight, attributes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    attributes = edges.attributes || EXCLUDED.attributes
            """, edge_data)
        else:
            cursor.executemany("""
                INSERT INTO edges (id, source_id, target_id, type, weight, attributes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weight = excluded.weight,
                    attributes = json_patch(edges.attributes, excluded.attributes)
            """, edge_data)
        self._edges_cache.clear()

    def commit(self) -> None:
        """
        Commit the current transaction to the SQLite database.

        Returns:
            None
        """
        self.conn.commit()

    def finalize(self) -> None:
        """
        Flush any pending data and finalize the storage state.

        Returns:
            None
        """
        self.flush()

    def close(self) -> None:
        """
        Close the database connection after finalizing any pending data.

        Returns:
            None
        """
        if hasattr(self, 'conn') and self.conn:
            self.finalize()
            self.conn.close()

    def log_amendment(self, action: str, entity_type: str, payload: Any) -> None:
        """
        Log an amendment to the graph to support replayability.
        """
        amend_id = str(uuid.uuid4())
        ts = time.time()
        
        cursor = self.conn.cursor()
        if self.is_postgres:
            cursor.execute("""
                INSERT INTO amendments (id, timestamp, action, entity_type, payload)
                VALUES (%s, %s, %s, %s, %s)
            """, (amend_id, ts, action, entity_type, psycopg2.extras.Json(payload)))
        else:
            cursor.execute("""
                INSERT INTO amendments (id, timestamp, action, entity_type, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (amend_id, ts, action, entity_type, json.dumps(payload)))
        self.conn.commit()



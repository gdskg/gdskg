import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Tuple
from core.schema import Node, Edge, NodeType, EdgeType

class GraphStore:
    """Persistent storage for the knowledge graph using SQLite."""

    def __init__(self, db_path: Path):
        """
        Initialize the GraphStore with the specified SQLite database path.

        Args:
            db_path (Path): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=OFF;")
        self.conn.execute("PRAGMA cache_size=-64000;")
        self._init_db()

        self._nodes_cache = {}
        self._edges_cache = {}

    def _init_db(self) -> None:
        """
        Initialize the SQLite database schema.

        Returns:
            None
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                attributes JSON
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                attributes JSON,
                FOREIGN KEY(source_id) REFERENCES nodes(id),
                FOREIGN KEY(target_id) REFERENCES nodes(id)
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
        if node_id in self._nodes_cache:
            return self._nodes_cache[node_id]

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if row:
            return Node(id=row[0], type=NodeType(row[1]), attributes=json.loads(row[2]))
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
        return db_count + len(self._nodes_cache)

    def flush(self) -> None:
        """
        Dump all memory-cached nodes and edges to SQLite.

        Returns:
            None
        """
        if not self._nodes_cache and not self._edges_cache:
            return

        cursor = self.conn.cursor()
        self._flush_nodes(cursor)
        self._flush_edges(cursor)
        self.conn.commit()

    def _flush_nodes(self, cursor: sqlite3.Cursor) -> None:
        """
        Perform the bulk insert/update of nodes.

        Args:
            cursor (sqlite3.Cursor): The database cursor.

        Returns:
            None
        """
        if not self._nodes_cache:
            return

        node_data = []
        for node in self._nodes_cache.values():
            node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
            node_data.append((node.id, node_type, json.dumps(node.attributes)))
        
        cursor.executemany("""
            INSERT INTO nodes (id, type, attributes)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                attributes = json_patch(nodes.attributes, excluded.attributes)
        """, node_data)
        self._nodes_cache.clear()

    def _flush_edges(self, cursor: sqlite3.Cursor) -> None:
        """
        Perform the bulk insert/update of edges.

        Args:
            cursor (sqlite3.Cursor): The database cursor.

        Returns:
            None
        """
        if not self._edges_cache:
            return

        edge_data = []
        for edge in self._edges_cache.values():
            edge_type = edge.type.value if hasattr(edge.type, 'value') else str(edge.type)
            edge_data.append((edge.id, edge.source_id, edge.target_id, edge_type, edge.weight, json.dumps(edge.attributes)))

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
        if hasattr(self, 'conn'):
            self.finalize()
            self.conn.close()



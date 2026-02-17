import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Tuple
from core.schema import Node, Edge, NodeType, EdgeType

class GraphStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Nodes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    attributes JSON
                )
            """)
            
            # Edges Table
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
            
            # Indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")
            
            conn.commit()

    def upsert_node(self, node: Node):
        """Insert or update a node."""
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO nodes (id, type, attributes)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    attributes = json_patch(nodes.attributes, excluded.attributes)
            """, (node.id, node_type, json.dumps(node.attributes)))
            conn.commit()

    def upsert_symbol(self, id: str, file_path: str):
        """Helper to upsert a symbol node."""
        node = Node(
            id=id, 
            type=NodeType.SYMBOL, 
            attributes={"name": id, "file": file_path}
        )
        self.upsert_node(node)

    def upsert_edge(self, edge: Edge):
        """Insert or update an edge."""
        edge_type = edge.type.value if hasattr(edge.type, 'value') else str(edge.type)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO edges (id, source_id, target_id, type, weight, attributes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weight = excluded.weight,
                    attributes = json_patch(edges.attributes, excluded.attributes)
            """, (edge.id, edge.source_id, edge.target_id, edge_type, edge.weight, json.dumps(edge.attributes)))
            conn.commit()

    def upsert_nodes(self, nodes: List[Node]):
        """Batch upsert nodes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            data = []
            for node in nodes:
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                data.append((node.id, node_type, json.dumps(node.attributes)))
            
            cursor.executemany("""
                INSERT INTO nodes (id, type, attributes)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    attributes = json_patch(nodes.attributes, excluded.attributes)
            """, data)
            conn.commit()

    def upsert_edges(self, edges: List[Edge]):
        """Batch upsert edges."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            data = []
            for edge in edges:
                edge_type = edge.type.value if hasattr(edge.type, 'value') else str(edge.type)
                data.append((edge.id, edge.source_id, edge.target_id, edge_type, edge.weight, json.dumps(edge.attributes)))

            cursor.executemany("""
                INSERT INTO edges (id, source_id, target_id, type, weight, attributes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weight = excluded.weight,
                    attributes = json_patch(edges.attributes, excluded.attributes)
            """, data)
            conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                return Node(id=row[0], type=NodeType(row[1]), attributes=json.loads(row[2]))
        return None
    
    def count_nodes(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            return cursor.fetchone()[0]

    def close(self):
        pass

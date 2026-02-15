import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Tuple
from core.schema import Node, Edge, NodeType, EdgeType

class GraphStore:
    """
    Persistent storage for the knowledge graph using SQLite.
    
    Handles the creation of the schema and provides methods for upserting
    nodes and edges, both individually and in batches.
    """

    def __init__(self, db_path: Path):
        """
        Initialize the GraphStore.

        Args:
            db_path (Path): The file path to the SQLite database.
        """

        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """
        Initialize the SQLite database schema by creating necessary tables and indices.
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
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
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")
            
            conn.commit()

    def upsert_node(self, node: Node) -> None:
        """
        Insert a new node or update an existing one if the ID already exists.

        Args:
            node (Node): The node object to store.
        """

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

    def upsert_symbol(self, id: str, file_path: str) -> None:
        """
        Helper method to specifically upsert a symbol node.

        Args:
            id (str): The unique identifier of the symbol.
            file_path (str): The path to the file where the symbol is defined.
        """

        node = Node(
            id=id, 
            type=NodeType.SYMBOL, 
            attributes={"name": id, "file": file_path}
        )
        self.upsert_node(node)

    def upsert_edge(self, edge: Edge) -> None:
        """
        Insert a new edge or update an existing one if the ID already exists.

        Args:
            edge (Edge): The edge object to store.
        """

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

    def upsert_nodes(self, nodes: List[Node]) -> None:
        """
        Perform a batch upsert of multiple nodes.

        Args:
            nodes (List[Node]): A list of node objects to store.
        """

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

    def upsert_edges(self, edges: List[Edge]) -> None:
        """
        Perform a batch upsert of multiple edges.

        Args:
            edges (List[Edge]): A list of edge objects to store.
        """

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
        """
        Retrieve a single node by its ID.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            Optional[Node]: The node object if found, otherwise None.
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, type, attributes FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                return Node(id=row[0], type=NodeType(row[1]), attributes=json.loads(row[2]))
        return None
    
    def count_nodes(self) -> int:
        """
        Count the total number of nodes in the graph.

        Returns:
            int: The total count of nodes.
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            return cursor.fetchone()[0]

    def close(self) -> None:
        """
        Close the database connection (no-op as currently using context managers).
        """

        pass

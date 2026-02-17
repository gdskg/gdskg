import pytest
from pathlib import Path
import sqlite3
from typing import List, Dict, Any

from core.graph_store import GraphStore
from core.extractor import GraphExtractor
from core.plugin_interfaces import PluginInterface, GraphInterface
from core.schema import Node, Edge

# Reusing the path from generated repos
PYTHON_REPO = Path("/tmp/gdskg_test_repos/python_test_repo")

class TestMockPlugin(PluginInterface):
    """
    A mock plugin that adds a specific node to every commit.
    """
    def process(self, commit_node: Node, related_nodes: List[Node], related_edges: List[Edge], graph_api: GraphInterface) -> None:
        # Add a node connected to the commit
        plugin_node_id = f"PLUGIN_DATA:{commit_node.id}"
        graph_api.add_node(
            id=plugin_node_id,
            type="PLUGIN_NODE",
            attributes={"info": "injected"}
        )
        graph_api.add_edge(
            source_id=commit_node.id,
            target_id=plugin_node_id,
            type="HAS_PLUGIN_DATA"
        )

@pytest.fixture(scope="module")
def plugin_graph_db(tmp_path_factory):
    if not PYTHON_REPO.exists():
        pytest.fail("Python test repo not found. Ensure conftest.py generates it.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "plugin_test.db"
    store = GraphStore(db_path)
    
    # Initialize with our mock plugin
    mock_plugin = TestMockPlugin()
    extractor = GraphExtractor(PYTHON_REPO, store, plugins=[mock_plugin])
    extractor.process_repo()
    
    return db_path

def query_one(db_path, query, args=()):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(query, args)
        return c.fetchone()

def query_all(db_path, query, args=()):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(query, args)
        return c.fetchall()

def test_plugin_nodes_created(plugin_graph_db):
    """
    Verify that the plugin created nodes for the commits.
    """
    # Count commits first to know expected number
    res = query_one(plugin_graph_db, "SELECT count(*) FROM nodes WHERE type='COMMIT'")
    commit_count = res[0]
    
    # Count plugin nodes
    res = query_one(plugin_graph_db, "SELECT count(*) FROM nodes WHERE type='PLUGIN_NODE'")
    plugin_node_count = res[0]
    
    assert plugin_node_count > 0
    assert plugin_node_count == commit_count, "One plugin node per commit expected"

def test_plugin_edges_created(plugin_graph_db):
    """
    Verify edges connect commits to plugin nodes.
    """
    edges = query_all(plugin_graph_db, "SELECT source_id, target_id FROM edges WHERE type='HAS_PLUGIN_DATA'")
    assert len(edges) > 0
    
    # Verify strict connectivity
    for source, target in edges:
        # Source should be a commit (hexsha)
        # Target should be PLUGIN_DATA:<hexsha>
        assert target == f"PLUGIN_DATA:{source}"

def test_plugin_attributes(plugin_graph_db):
    """
    Verify attributes were saved correctly.
    """
    nodes = query_all(plugin_graph_db, "SELECT attributes FROM nodes WHERE type='PLUGIN_NODE' LIMIT 1")
    import json
    attrs = json.loads(nodes[0][0])
    assert attrs["info"] == "injected"

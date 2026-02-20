import os
import shutil
from pathlib import Path
import pytest
from core.graph_store import GraphStore
from core.schema import Node, Edge, NodeType, EdgeType
from analysis.search_engine import SearchEngine
from core.vector_store import VectorStore
from analysis.embedder import ONNXEmbedder
import json

@pytest.fixture
def test_dir():
    d = Path("test_vector_search_dir")
    d.mkdir(exist_ok=True)
    yield d
    shutil.rmtree(d)

def test_semantic_search(test_dir):
    db_path = test_dir / "gdskg.db"
    vector_db_path = db_path
    
    store = GraphStore(db_path)
    
    commit_id = "commit_1234"
    store.upsert_node(Node(id=commit_id, type=NodeType.COMMIT, attributes={"message": "Added DB setup", "author": "Alice", "date": "2023-01-01"}))
    
    func_id = "func_setup_db"
    store.upsert_node(Node(id=func_id, type=NodeType.FUNCTION, attributes={"name": "setup_db", "file": "db.py"}))
    
    store.upsert_edge(Edge(source_id=commit_id, target_id=func_id, type=EdgeType.MODIFIED_FUNCTION))
    
    hist_id = "hist_setup_db"
    store.upsert_node(Node(id=hist_id, type=NodeType.FUNCTION_HISTORY, attributes={}))
    store.upsert_edge(Edge(source_id=func_id, target_id=hist_id, type=EdgeType.HAS_HISTORY))
    
    v1_id = "v1_setup_db"
    store.upsert_node(Node(id=v1_id, type=NodeType.FUNCTION_VERSION, attributes={"content": "def setup_db():\\n    # initialize database connection\\n    pass"}))
    store.upsert_edge(Edge(source_id=hist_id, target_id=v1_id, type=EdgeType.HAS_VERSION))
    
    store.close()
    
    vstore = VectorStore(vector_db_path)
    embedder = ONNXEmbedder()
    count = vstore.build_from_graph(str(db_path), embedder)
    
    assert count > 0, "Should have embedded at least 1 node"
    
    searcher = SearchEngine(str(db_path), str(vector_db_path))
    
    results = searcher.search("start the persistence layer")
    print("SEARCH ENGINE RESULTS:", results)

    
    assert len(results) > 0, "Should find the commit through semantic search"
    commit_res = [res for res in results if res['id'] == commit_id]
    assert len(commit_res) == 1
    
    reasons = commit_res[0]['reasons']
    found_semantic_reason = any("Similar meaning identified" in r for r in reasons)
    assert found_semantic_reason, "Should include exact reason indicating similar meaning"

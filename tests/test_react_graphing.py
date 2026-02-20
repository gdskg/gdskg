import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

REACT_REPO = Path("/tmp/gdskg_test_repos/react_test_repo")

@pytest.fixture(scope="module")
def react_graph(tmp_path_factory):
    if not REACT_REPO.exists():
        pytest.fail("React test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "react_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(REACT_REPO, store)
    extractor.process_repo()
    return db_path

def query_one(db_path, query, args=()):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(query, args)
    return c.fetchone()

def query_all(db_path, query, args=()):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(query, args)
    return c.fetchall()

class TestReactGraphing:
    def test_file_nodes(self, react_graph):
        files = query_all(react_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "package.json" in file_ids
        assert "src/App.jsx" in file_ids

    def test_component_symbol(self, react_graph):
        res = query_all(react_graph, "SELECT id FROM nodes WHERE type='SYMBOL' AND id LIKE '%App%'")
        assert len(res) > 0, "App component symbol not extracted"

    def test_component_evolution(self, react_graph):
        files = query_all(react_graph, "SELECT id FROM nodes WHERE type='FILE' AND id='src/App.jsx'")
        file_id = files[0][0]
        
        connections = query_all(react_graph, 
                                "SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_FILE'",
                                (file_id,))
        assert len(connections) == 2, "App.jsx should be modified in 2 commits"
    
    def test_component_function_tracking(self, react_graph):
        history_query = query_all(react_graph, "SELECT id FROM nodes WHERE type='FUNCTION_HISTORY' AND id LIKE '%App%'")
        assert len(history_query) > 0, "No function history logic found for App function component"

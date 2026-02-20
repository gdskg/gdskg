import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

C_REPO = Path("/tmp/gdskg_test_repos/c_test_repo")

@pytest.fixture(scope="module")
def c_graph(tmp_path_factory):
    if not C_REPO.exists():
        pytest.fail("C test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "c_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(C_REPO, store)
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

class TestCGraphing:
    def test_file_nodes(self, c_graph):
        files = query_all(c_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "src/main.c" in file_ids
        assert "src/utils.h" in file_ids
        assert "src/utils.c" in file_ids

    def test_component_symbol(self, c_graph):
        res = query_all(c_graph, "SELECT id FROM nodes WHERE type='SYMBOL'")
        sym_map = {s[0] for s in res}
        assert any("main" in s for s in sym_map), "main function symbol not extracted"
        assert any("add" in s for s in sym_map), "add function symbol not extracted"

    def test_component_evolution(self, c_graph):
        files = query_all(c_graph, "SELECT id FROM nodes WHERE type='FILE' AND id='src/main.c'")
        file_id = files[0][0]
        
        connections = query_all(c_graph, 
                                "SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_FILE'",
                                (file_id,))
        assert len(connections) == 2, "main.c should be modified in 2 commits"
    
    def test_component_function_tracking(self, c_graph):
        history_query = query_all(c_graph, "SELECT id FROM nodes WHERE type='FUNCTION_HISTORY' AND id LIKE '%main%'")
        assert len(history_query) > 0, "No function history logic found for main function"

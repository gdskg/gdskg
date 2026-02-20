import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

PYTHON_REPO = Path("/tmp/gdskg_test_repos/python_test_repo")

@pytest.fixture(scope="module")
def python_graph(tmp_path_factory):
    if not PYTHON_REPO.exists():
        pytest.fail("Python test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "python_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(PYTHON_REPO, store)
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

class TestPythonGraphing:
    def test_repo_node_exists(self, python_graph):
        res = query_one(python_graph, "SELECT count(*) FROM nodes WHERE type='REPOSITORY'")
        assert res[0] == 1

    def test_file_nodes_exist(self, python_graph):
        files = query_all(python_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "main.py" in file_ids
        assert "utils.py" in file_ids
        assert "core/logic.py" in file_ids
        assert "config.py" in file_ids

    def test_symbols_extraction(self, python_graph):
        symbols = query_all(python_graph, "SELECT id, attributes FROM nodes WHERE type='SYMBOL'")
        sym_map = {s[0]: s[1] for s in symbols}
        
        assert any("main" in s for s in sym_map), "main function not found"
        assert any("helper" in s for s in sym_map), "helper function not found"
        assert any("Processor" in s for s in sym_map), "Processor class not found"
        assert any("new_helper" in s for s in sym_map), "new_helper function not found"

    def test_secret_detection(self, python_graph):
        secrets = query_all(python_graph, "SELECT id FROM nodes WHERE type='SECRET'")
        secret_ids = [s[0] for s in secrets]
        assert "ENV:DB_PASS" in secret_ids
        
    def test_symbol_evolution(self, python_graph):
        res = query_one(python_graph, "SELECT id FROM nodes WHERE type='SYMBOL' AND id LIKE '%helper%' AND id NOT LIKE '%new_helper%'")
        assert res, "helper symbol node missing"
        helper_id = res[0]
        
        edges = query_all(python_graph, 
                          "SELECT target_id FROM edges WHERE source_id=? AND type='MODIFIED_SYMBOL'", 
                          (helper_id,))
        
        assert len(edges) >= 2, f"Expected evolution for helper, found {len(edges)} modifications"

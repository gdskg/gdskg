import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

TYPESCRIPT_REPO = Path("/tmp/gdskg_test_repos/typescript_test_repo")

@pytest.fixture(scope="module")
def ts_graph(tmp_path_factory):
    if not TYPESCRIPT_REPO.exists():
        pytest.fail("Typescript test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "ts_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(TYPESCRIPT_REPO, store)
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

class TestTypeScriptGraphing:
    def test_file_nodes(self, ts_graph):
        files = query_all(ts_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "src/index.ts" in file_ids
        assert "src/Component.tsx" in file_ids
        assert "package.json" in file_ids

    def test_component_symbol(self, ts_graph):
        res = query_all(ts_graph, "SELECT id FROM nodes WHERE type='SYMBOL' AND id LIKE '%Button%'")
        assert len(res) > 0, "Button component symbol not extracted"

    def test_component_evolution(self, ts_graph):
        files = query_all(ts_graph, "SELECT id FROM nodes WHERE type='FILE' AND id='src/Component.tsx'")
        file_id = files[0][0]
        
        connections = query_all(ts_graph, 
                                "SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_FILE'",
                                (file_id,))
        assert len(connections) == 2, "Component.tsx should be modified in 2 commits"

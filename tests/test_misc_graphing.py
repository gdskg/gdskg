import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

MISC_REPO = Path("/tmp/gdskg_test_repos/misc_test_repo")

@pytest.fixture(scope="module")
def misc_graph(tmp_path_factory):
    if not MISC_REPO.exists():
        pytest.fail("Misc test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "misc_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(MISC_REPO, store)
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

class TestMiscGraphing:
    def test_file_nodes(self, misc_graph):
        files = query_all(misc_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "index.html" in file_ids
        assert "style.css" in file_ids
        assert "README.md" in file_ids
        assert "Dockerfile" in file_ids
        assert "config.yaml" in file_ids
        assert "data.json" in file_ids

    def test_file_evolution(self, misc_graph):
        """Verify that all files are tracked as modified across commits."""
        for filename in ["index.html", "style.css", "README.md", "Dockerfile", "config.yaml", "data.json"]:
            files = query_all(misc_graph, "SELECT id FROM nodes WHERE type='FILE' AND id=?", (filename,))
            assert len(files) == 1, f"Missing or duplicate FILE node: {filename}"
            file_id = files[0][0]
            
            connections = query_all(misc_graph, 
                                    "SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_FILE'",
                                    (file_id,))
            assert len(connections) == 2, f"{filename} should be modified in 2 commits, found {len(connections)}"

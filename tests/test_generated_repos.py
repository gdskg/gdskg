import pytest
from pathlib import Path
import sqlite3
from core.graph_store import GraphStore
from core.extractor import GraphExtractor

# These paths matches what provided in generate_test_repo.py
PYTHON_REPO = Path("/tmp/gdskg_test_repos/python_test_repo")
TYPESCRIPT_REPO = Path("/tmp/gdskg_test_repos/typescript_test_repo")

@pytest.fixture(scope="module")
def python_graph(tmp_path_factory):
    if not PYTHON_REPO.exists():
        pytest.fail("Python test repo not found.")
    
    db_path = tmp_path_factory.mktemp("graphs") / "python_extensive.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(PYTHON_REPO, store)
    extractor.process_repo()
    return db_path

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

class TestPythonRepo:
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
        
        # Check specific symbols
        # main.py -> main
        assert any("main" in s for s in sym_map), "main function not found"
        # utils.py -> helper
        assert any("helper" in s for s in sym_map), "helper function not found"
        # core/logic.py -> Processor (Class)
        assert any("Processor" in s for s in sym_map), "Processor class not found"
        # utils.py -> new_helper (Added in refactor)
        assert any("new_helper" in s for s in sym_map), "new_helper function not found"

    def test_secret_detection(self, python_graph):
        secrets = query_all(python_graph, "SELECT id FROM nodes WHERE type='SECRET'")
        secret_ids = [s[0] for s in secrets]
        assert "ENV:DB_PASS" in secret_ids
        # AWS_SECRET is a hardcoded string, not an env var access like os.getenv
        # The SecretScanner currently looks for ENV patterns.
        # If the scanner is good, it might skip strings unless configured otherwise.
        # Based on current regex, it matches `os.getenv` etc.
        # So AWS_SECRET might NOT be found if it's just 'AKIA...' string literal 
        # unless we added entropy scanning. The current prompt says regex based on accessors.
        
    def test_symbol_evolution(self, python_graph):
        """
        Verify that 'helper' was modified in a later commit.
        Structure: (Symbol: helper) -[MODIFIED_SYMBOL]-> (Commit)
        """
        # Find ID for helper
        res = query_one(python_graph, "SELECT id FROM nodes WHERE type='SYMBOL' AND id LIKE '%helper%' AND id NOT LIKE '%new_helper%'")
        assert res, "helper symbol node missing"
        helper_id = res[0]
        
        # Check edges
        edges = query_all(python_graph, 
                          "SELECT target_id FROM edges WHERE source_id=? AND type='MODIFIED_SYMBOL'", 
                          (helper_id,))
        
        # Commit 1: Created (Modified/Added)
        # Commit 3: Refactored (Modified)
        # Should have at least 2 edges if we track creation and modification
        assert len(edges) >= 2, f"Expected evolution for helper, found {len(edges)} modifications"

class TestTypeScriptRepo:
    def test_file_nodes(self, ts_graph):
        files = query_all(ts_graph, "SELECT id FROM nodes WHERE type='FILE'")
        file_ids = [f[0] for f in files]
        assert "src/index.ts" in file_ids
        assert "src/Component.tsx" in file_ids
        assert "package.json" in file_ids # JSON might be skipped by extractor language filter? 
        # Extractor language filter relies on tree-sitter mappings. 
        # If json is supported it should be there.

    def test_component_symbol(self, ts_graph):
        # We want to see if 'Button' is extracted.
        # As noted, 'export const Button = ...' is tricky for simple logic.
        res = query_all(ts_graph, "SELECT id FROM nodes WHERE type='SYMBOL' AND id LIKE '%Button%'")
        
        # If we failed to extract it, this test might fail. 
        # But let's assert what we expect.
        assert len(res) > 0, "Button component symbol not extracted"

    def test_component_evolution(self, ts_graph):
        # src/Component.tsx modified in 2 commits (Create, Modify)
        files = query_all(ts_graph, "SELECT id FROM nodes WHERE type='FILE' AND id='src/Component.tsx'")
        file_id = files[0][0]
        
        connections = query_all(ts_graph, 
                                "SELECT source_id FROM edges WHERE target_id=? AND type='MODIFIED_FILE'",
                                (file_id,))
        assert len(connections) == 2, "Component.tsx should be modified in 2 commits"

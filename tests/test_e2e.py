import pytest
import git
from pathlib import Path
from core.graph_store import GraphStore
from core.extractor import GraphExtractor
from core.schema import NodeType, EdgeType

@pytest.fixture
def temp_git_repo(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)
    return repo, repo_dir

def test_basic_symbol_extraction(temp_git_repo):
    repo, repo_path = temp_git_repo
    
    # Commit 1: Add a python file with a function
    file_path = repo_path / "main.py"
    file_path.write_text("def hello():\n    print('world')\n")
    repo.index.add([str(file_path)])
    repo.index.commit("Initial commit")
    
    # Run Extractor
    db_path = repo_path / "graph.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(repo_path, store)
    extractor.process_repo()
    
    # Verify
    # Check Commit Node
    assert store.count_nodes() > 0
    
    # Check Symbol Node
    # Pass 1 should have found "hello"
    # Pass 2 should have linked it to commit because it's new
    
    # Inspect DB
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT id, attributes FROM nodes WHERE type='SYMBOL'")
    symbols = c.fetchall()
    print("Symbols found:", symbols)
    
    # We expect 'start_scope.hello' or just 'hello' depending on extractor logic
    # Our extractor output canonical name. In bare file, scope is "".
    # logic: canonical = f"{scope}.{name}" if scope else name -> "hello"
    
    found_hello = False
    for s_id, attrs in symbols:
        if s_id == "hello":
            found_hello = True
            break
    assert found_hello, "Symbol 'hello' not found in graph"
    
    # Check Edge MODIFIED_SYMBOL
    c.execute("SELECT * FROM edges WHERE type='MODIFIED_SYMBOL'")
    edges = c.fetchall()
    assert len(edges) > 0, "No MODIFIED_SYMBOL edges found"

def test_secret_scanning(temp_git_repo):
    repo, repo_path = temp_git_repo
    
    file_path = repo_path / "config.py"
    file_path.write_text("import os\nkey = os.getenv('STRIPE_API_KEY')\n")
    repo.index.add([str(file_path)])
    repo.index.commit("Add config with secret")
    
    db_path = repo_path / "graph.db"
    store = GraphStore(db_path)
    extractor = GraphExtractor(repo_path, store)
    extractor.process_repo()
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT id FROM nodes WHERE type='SECRET'")
    secrets = [row[0] for row in c.fetchall()]
    assert "ENV:STRIPE_API_KEY" in secrets

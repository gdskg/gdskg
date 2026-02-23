import sys
from pathlib import Path
import pytest
import os
import json
import sqlite3

# Ensure gdskg is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from mcp_server.server import (
    index_repository,
    get_dependencies,
    _write_server_status,
)
from analysis.search_engine import SearchEngine
from core.schema import EdgeType

@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """Fixture to provide temporary directories and mock graph path."""
    repo_dir = tmp_path / "repo"
    graph_dir = tmp_path / "graph"
    repo_dir.mkdir()
    graph_dir.mkdir(exist_ok=True)
    monkeypatch.setattr('mcp_server.server.DEFAULT_GRAPH_PATH', graph_dir)
    _write_server_status(True)
    
    # Init a git repo in repo_dir
    os.system(f"git init {repo_dir}")
    # Create a file with contents that will trigger keyword and comment extraction
    (repo_dir / "app.py").write_text("# This is a comment\ndef main():\n    print('hello world')\n")
    os.system(f"cd {repo_dir} && git add . && git commit -m 'Initial commit'")
    
    index_repository(str(repo_dir), overwrite=True)
    return repo_dir, graph_dir

def test_get_dependencies_clarity(test_env):
    repo_dir, graph_dir = test_env
    db_path = graph_dir / "gdskg.db"
    
    # Manually add a comment and keyword to ensure they exist for testing filtering
    # GDSKG extractor might already do this, but being explicit.
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO nodes (id, type) VALUES ('COMMENT:test', 'COMMENT')")
        cursor.execute("INSERT OR IGNORE INTO nodes (id, type) VALUES ('KEYWORD:main', 'KEYWORD')")
        cursor.execute("INSERT OR IGNORE INTO edges (source_id, target_id, type) VALUES ('FILE:app.py', 'COMMENT:test', 'CONTAINS_COMMENT')")
        cursor.execute("INSERT OR IGNORE INTO edges (source_id, target_id, type) VALUES ('FILE:app.py', 'KEYWORD:main', 'HAS_KEYWORD')")
        
        conn.commit()

    # Get dependencies for app.py
    result = get_dependencies("app.py")
    
    assert isinstance(result, dict)
    assert "node" in result
    assert "app.py" in result["node"]
    
    # Verify filtering: COMMENT and KEYWORD edges should NOT be in dependencies
    for dep in result["dependencies"]:
        assert dep["type"] not in ('CONTAINS_COMMENT', 'HAS_KEYWORD')
        assert "node_type" in dep # Verify node_type enrichment
        
    for dep in result["dependents"]:
        assert dep["type"] not in ('CONTAINS_COMMENT', 'HAS_KEYWORD')
        assert "node_type" in dep # Verify node_type enrichment

    # Verify we still get the repository or commit as dependency/dependent
    rel_ids = [d["id"] for d in result["dependencies"]] + [d["id"] for d in result["dependents"]]
    # At least one should be the repository or commit
    assert any("repo" in rid or len(rid) == 40 for rid in rel_ids)

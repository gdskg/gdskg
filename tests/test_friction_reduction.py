
import sys
from pathlib import Path
import pytest
import os
import shutil
import json

# Ensure gdskg is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from mcp_server.server import (
    index_repository,
    query_knowledge_graph,
    get_ast_nodes,
    get_dependencies,
    _write_server_status,
)
from analysis.search_engine import SearchEngine

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
    (repo_dir / "AccountCard.tsx").write_text("export const AccountCard = () => { const x = useStore(); return <div>Account</div>; }")
    (repo_dir / "utils.ts").write_text("export const addMonths = (d: Date, m: int) => d;")
    os.system(f"cd {repo_dir} && git add . && git commit -m 'Initial commit'")
    
    index_repository(str(repo_dir), overwrite=True)
    return repo_dir, graph_dir

def test_fuzzy_ast_resolution(test_env):
    repo_dir, graph_dir = test_env
    # AccountCard.tsx should resolve even without FILE: prefix
    result = get_ast_nodes("AccountCard.tsx")
    assert "Found" in result
    assert "AccountCard" in result
    
    # Test partial match
    result_partial = get_ast_nodes("AccountCard")
    assert "Found" in result_partial
    assert "AccountCard" in result_partial

def test_hydrated_query_results(test_env):
    repo_dir, graph_dir = test_env
    result = query_knowledge_graph("AccountCard")
    
    assert isinstance(result, dict)
    assert "description" in result
    assert "results" in result
    assert len(result["results"]) > 0
    
    first_match = result["results"][0]
    assert "raw_metadata" in first_match
    assert "id" in first_match
    assert "commit_id" in first_match["raw_metadata"]
    
    # Check if related nodes are present
    assert len(first_match["raw_metadata"]["related_nodes"]) > 0
    # At least one related node should be AccountCard.tsx
    related_ids = [n["id"] for n in first_match["raw_metadata"]["related_nodes"]]
    assert any("AccountCard.tsx" in rid for rid in related_ids)

def test_get_dependencies(test_env):
    repo_dir, graph_dir = test_env
    # Get dependencies for AccountCard.tsx
    result = get_dependencies("AccountCard.tsx")
    
    assert isinstance(result, dict)
    assert "node" in result
    assert "dependencies" in result
    assert "dependents" in result
    # In GDSKG, node ID for a file is often just the relative path
    assert "AccountCard.tsx" in result["node"]
    
    # It should have a dependency on the repository or commit
    assert len(result["dependencies"]) > 0 or len(result["dependents"]) > 0


import sys
from pathlib import Path
import pytest
import os
import shutil

# Ensure gdskg/mcp_server is importable
# Assuming this test file is in gdskg/tests/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from mcp_server.server import (
    index_repository,
    query_knowledge_graph,
    start_server,
    stop_server,
    get_server_status,
    _write_server_status,
    _read_server_status,
)

@pytest.fixture
def test_dirs(tmp_path):
    """Fixture to provide temporary directories for repo and graph."""
    repo_dir = tmp_path / "repo"
    graph_dir = tmp_path / "graph"
    repo_dir.mkdir()
    
    # Init a git repo in repo_dir
    os.system(f"git init {repo_dir}")
    (repo_dir / "test_file.txt").write_text("def hello(): pass")
    os.system(f"cd {repo_dir} && git add . && git commit -m 'Initial commit'")
    
    return repo_dir, graph_dir

def test_index_folder(test_dirs):
    repo_path, graph_path = test_dirs
    
    # Test indexing
    result = index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True)
    assert "Successfully indexed" in result
    assert "Graph built at" in result
    assert graph_path.exists()
    assert (graph_path / "gdskg.db").exists()

def test_query_knowledge_graph(test_dirs):
    repo_path, graph_path = test_dirs
    
    # Index first
    index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True)
    
    # Test valid query
    # We added 'def hello(): pass', so 'hello' should be a symbol if extraction works on txt/py
    # Wait, extension .txt might not trigger python extraction. Let's rename to .py
    (repo_path / "test_file.py").write_text("def world(): pass")
    os.system(f"cd {repo_path} && git add test_file.py && git commit -m 'Add python file'")
    
    # Re-index to catch the new commit
    index_repository(str(repo_path), graph_path=str(graph_path), overwrite=False) # Append
    
    result = query_knowledge_graph("world", graph_path=str(graph_path))
    # It should find something related to "world" or at least the commit message "Add python file" if query matches
    # or just "Add python file" -> commit message node.
    
    # Let's query for "python"
    result_python = query_knowledge_graph("python", graph_path=str(graph_path))
    assert "Search Results for: 'python'" in result_python
    # Assert we found something
    assert "Commit:" in result_python

def test_query_no_results(test_dirs):
    repo_path, graph_path = test_dirs
    # Index empty repo or just use new graph path
    graph_path.mkdir(exist_ok=True)
    # If no db, query returns error
    result = query_knowledge_graph("missing", graph_path=str(graph_path))
    assert "Warning: Knowledge Graph Database not found" in result

    # Create empty db? No, index_repository handles creation. 
    # Let's index then query garbage.
    index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True)
    result = query_knowledge_graph("supercalifragilisticexpialidocious_xyz", graph_path=str(graph_path))
    assert "No relevant matches found" in result

def test_index_with_plugins(test_dirs):
    repo_path, graph_path = test_dirs
    # We can't easily mock the external API calls of GitHubPR/ClickUpTask here without more setup,
    # but we can verify that index_repository accepts the arguments and tries to load them.
    # We'll use a bogus plugin name to see if it reports an error loading it, which proves the path works.
    result = index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True, plugins=["NonExistentPlugin"])
    assert "Error: Only 0/1 plugins loaded successfully" in result

    # Now try with a real plugin name but bogus parameters (which should be ignored or handled gracefully)
    # GitHubPR exists in the plugins folder.
    result = index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True, plugins=["GitHubPR"], parameters=["GitHubPR:key=value"])
    assert "Successfully indexed" in result
    assert "Plugins enabled: 1" in result


# --- Server Status Tests ---

@pytest.fixture(autouse=True)
def reset_server_state():
    """Ensure the server is in the started state before each test."""
    _write_server_status(True)
    yield
    _write_server_status(True)


def test_server_starts_started():
    """Server should be started by default after import."""
    assert _read_server_status() is True
    result = get_server_status()
    assert "started" in result


def test_stop_server_gates_requests(test_dirs):
    """When server is stopped, tool requests should be rejected."""
    repo_path, graph_path = test_dirs

    # First index while server is running
    index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True)

    # Stop the server
    result = stop_server()
    assert "stopped" in result.lower()

    # Requests should be rejected
    q_result = query_knowledge_graph("test", graph_path=str(graph_path))
    assert "Server is stopped" in q_result

    i_result = index_repository(str(repo_path), graph_path=str(graph_path))
    assert "Server is stopped" in i_result


def test_start_server_resumes_requests(test_dirs):
    """After stopping and restarting, tool requests should work again."""
    repo_path, graph_path = test_dirs

    stop_server()
    start_server()

    result = index_repository(str(repo_path), graph_path=str(graph_path), overwrite=True)
    assert "Successfully indexed" in result


def test_get_server_status():
    """get_server_status should reflect the current state."""
    assert "started" in get_server_status()

    stop_server()
    assert "stopped" in get_server_status()

    start_server()
    assert "started" in get_server_status()

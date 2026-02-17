
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

from mcp_server.server import index_folder, query_knowledge_graph

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
    result = index_folder(str(repo_path), graph_path=str(graph_path), overwrite=True)
    assert "Successfully indexed" in result
    assert "Graph built at" in result
    assert graph_path.exists()
    assert (graph_path / "gdskg.db").exists()

def test_query_knowledge_graph(test_dirs):
    repo_path, graph_path = test_dirs
    
    # Index first
    index_folder(str(repo_path), graph_path=str(graph_path), overwrite=True)
    
    # Test valid query
    # We added 'def hello(): pass', so 'hello' should be a symbol if extraction works on txt/py
    # Wait, extension .txt might not trigger python extraction. Let's rename to .py
    (repo_path / "test_file.py").write_text("def world(): pass")
    os.system(f"cd {repo_path} && git add test_file.py && git commit -m 'Add python file'")
    
    # Re-index to catch the new commit
    index_folder(str(repo_path), graph_path=str(graph_path), overwrite=False) # Append
    
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
    assert "Error: Database not found" in result

    # Create empty db? No, index_folder handles creation. 
    # Let's index then query garbage.
    index_folder(str(repo_path), graph_path=str(graph_path), overwrite=True)
    result = query_knowledge_graph("supercalifragilisticexpialidocious_xyz", graph_path=str(graph_path))
    assert "No relevant matches found" in result

def test_index_with_plugins(test_dirs):
    repo_path, graph_path = test_dirs
    # We can't easily mock the external API calls of GitHubPR/ClickUpTask here without more setup,
    # but we can verify that index_folder accepts the arguments and tries to load them.
    # We'll use a bogus plugin name to see if it reports an error loading it, which proves the path works.
    result = index_folder(str(repo_path), graph_path=str(graph_path), overwrite=True, plugins=["NonExistentPlugin"])
    assert "Error: Only 0/1 plugins loaded successfully" in result

    # Now try with a real plugin name but bogus parameters (which should be ignored or handled gracefully)
    # GitHubPR exists in the plugins folder.
    result = index_folder(str(repo_path), graph_path=str(graph_path), overwrite=True, plugins=["GitHubPR"], parameters=["GitHubPR:key=value"])
    assert "Successfully indexed" in result
    assert "Plugins enabled: 1" in result

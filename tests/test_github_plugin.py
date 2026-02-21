
import pytest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from core.plugin_interfaces import GraphInterface
from core.schema import Node, NodeType

# We need to import the plugin. 
# Since it's outside the main repo, we can append to sys.path or load it dynamically.
import sys
import importlib.util
import sys

PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "GitHubPR" / "plugin.py"

spec = importlib.util.spec_from_file_location("github_plugin_module", PLUGIN_PATH)
plugin_module = importlib.util.module_from_spec(spec)
sys.modules["github_plugin_module"] = plugin_module
spec.loader.exec_module(plugin_module)

from github_plugin_module import GitHubPlugin

class MockGraphAPI(GraphInterface):
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, id, type, attributes=None):
        self.nodes[id] = {"type": type, "attributes": attributes}

    def add_edge(self, source_id, target_id, type, attributes=None):
        self.edges.append({"source": source_id, "target": target_id, "type": type})

@pytest.fixture
def mock_graph_api():
    return MockGraphAPI()

@pytest.fixture
def context_nodes():
    commit_node = Node(id="commit123", type=NodeType.COMMIT, attributes={})
    
    msg_node = Node(id="msg1", type=NodeType.COMMIT_MESSAGE, attributes={"content": "Fixes bug #42 in UI"})
    
    repo_node = Node(id="repo1", type=NodeType.REPOSITORY, attributes={
        "remotes": ["https://github.com/gdskg/core.git"]
    })
    
    return commit_node, [msg_node, repo_node]

def test_github_plugin_initialization():
    # Test without PAT
    if "GITHUB_PAT" in os.environ:
        del os.environ["GITHUB_PAT"]
    
    plugin = GitHubPlugin()
    assert plugin.enabled == True

    # Test with PAT
    os.environ["GITHUB_PAT"] = "dummy_token"
    plugin = GitHubPlugin()
    assert plugin.enabled == True

@patch("requests.get")
def test_github_plugin_process(mock_get, mock_graph_api, context_nodes):
    os.environ["GITHUB_PAT"] = "dummy_token"
    plugin = GitHubPlugin()
    
    commit_node, related_nodes = context_nodes
    
    # Mock GitHub API Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "number": 42,
        "title": "Fix UI Bug",
        "state": "open",
        "html_url": "https://github.com/gdskg/core/pull/42",
        "user": {"login": "testuser"},
        "created_at": "2023-01-01T00:00:00Z"
    }
    mock_get.return_value = mock_response
    
    plugin.process(commit_node, related_nodes, [], mock_graph_api)
    
    # Verify Node Creation
    pr_id = "PR:gdskg/core#42"
    assert pr_id in mock_graph_api.nodes
    assert mock_graph_api.nodes[pr_id]["type"] == "PULL_REQUEST"
    assert mock_graph_api.nodes[pr_id]["attributes"]["title"] == "Fix UI Bug"
    
    # Verify Edge Creation
    edge = next((e for e in mock_graph_api.edges if e["target"] == pr_id), None)
    assert edge is not None
    assert edge["source"] == "commit123"
    
    # Verify API Call
    mock_get.assert_called_with(
        "https://api.github.com/repos/gdskg/core/pulls/42",
        headers={"Authorization": "token dummy_token", "Accept": "application/vnd.github.v3+json"}
    )

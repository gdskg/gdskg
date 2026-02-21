
import pytest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from core.schema import Node, NodeType
from core.plugin_interfaces import GraphInterface

# Import plugin dynamically
import importlib.util
import sys

PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "ClickUpTask" / "plugin.py"

spec = importlib.util.spec_from_file_location("clickup_plugin_module", PLUGIN_PATH)
plugin_module = importlib.util.module_from_spec(spec)
sys.modules["clickup_plugin_module"] = plugin_module
spec.loader.exec_module(plugin_module)

from clickup_plugin_module import ClickUpPlugin

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
    # Message with a custom task ID
    msg_node = Node(id="msg1", type=NodeType.COMMIT_MESSAGE, attributes={"content": "Completed features for CU-12345 adn CU-67890"})
    return commit_node, [msg_node]

def test_clickup_plugin_initialization():
    if "CLICKUP_API_KEY" in os.environ:
        del os.environ["CLICKUP_API_KEY"]
    
    plugin = ClickUpPlugin()
    assert plugin.enabled == True

    os.environ["CLICKUP_API_KEY"] = "dummy_key"
    plugin = ClickUpPlugin()
    assert plugin.enabled == True

@patch("requests.get")
def test_clickup_plugin_process(mock_get, mock_graph_api, context_nodes):
    os.environ["CLICKUP_API_KEY"] = "dummy_key"
    plugin = ClickUpPlugin()
    
    commit_node, related_nodes = context_nodes
    
    # Configuration passing regex
    config = {
        "regex": r"CU-(\d+)",
        "organization": "team_1"
    }

    # Mock API Responses
    def side_effect(url, headers, params):
        # Check URL contains task ID
        if "12345" in url:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "12345",
                "name": "Feature A",
                "status": {"status": "Complete"},
                "url": "https://app.clickup.com/t/12345",
                "description": "Desc A",
                "team_id": "team_1"
            }
            return mock_resp
        elif "67890" in url:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "67890",
                "name": "Feature B",
                "status": {"status": "In Progress"},
                "url": "https://app.clickup.com/t/67890",
                "description": "Desc B",
                "team_id": "team_1"
            }
            return mock_resp
        return MagicMock(status_code=404)

    mock_get.side_effect = side_effect
    
    plugin.process(commit_node, related_nodes, [], mock_graph_api, config)
    
    # Verify Nodes
    assert "CLICKUP_TASK:12345" in mock_graph_api.nodes
    assert mock_graph_api.nodes["CLICKUP_TASK:12345"]["attributes"]["title"] == "Feature A"
    
    assert "CLICKUP_TASK:67890" in mock_graph_api.nodes
    assert mock_graph_api.nodes["CLICKUP_TASK:67890"]["attributes"]["title"] == "Feature B"
    
    # Verify Edges
    edge = next((e for e in mock_graph_api.edges if e["target"] == "CLICKUP_TASK:12345"), None)
    assert edge["source"] == "commit123"
    assert edge["type"] == "RELATED_TO_TASK"

    # Verify Calls
    # Logic in plugin uses next() on regex matches if they are tuples?
    # re.findall(r"CU-(\d+)", "CU-12345") returns ['12345'] (list of strings) if one group.
    # So our plugin logic `if isinstance(task_id, tuple)` helps if multiple groups.
    # With one group, it is a string.
    assert mock_get.call_count == 2

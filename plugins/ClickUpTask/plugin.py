
import os
import re
import requests
import logging
from typing import List, Dict, Any
from core.plugin_interfaces import PluginInterface, GraphInterface
from core.schema import Node, Edge, NodeType

logger = logging.getLogger(__name__)

class ClickUpPlugin(PluginInterface):
    def __init__(self):
        self.api_key = os.environ.get("CLICKUP_API_KEY")
        self.enabled = True
        if not self.api_key:
            logger.warning("[ClickUpPlugin] CLICKUP_API_KEY environment variable not found. Plugin disabled.")
            self.enabled = False
        else:
            logger.info("[ClickUpPlugin] Initialized.")
        
        self.task_cache = {}

    def process(self, commit_node: Node, related_nodes: List[Node], related_edges: List[Edge], graph_api: GraphInterface, config: Dict[str, Any] = None) -> None:
        if not self.enabled:
            return
        
        config = config or {}
        
        # Get parameters
        # User defined regex to find task IDs, e.g. "CU-(\d+)" or "#([a-zA-Z0-9]+)"
        # Default to a generic pattern if not provided? Or silently return?
        # User said: "accept that regex parameter ... maybe we don't always have a full link"
        task_regex = config.get("regex")
        organization_id = config.get("organization") # Team ID
        
        if not task_regex:
            logger.debug("[ClickUpPlugin] No regex parameter provided. Skipping.")
            return

        # Extract Commit Message
        msg_node = next((n for n in related_nodes if n.type == NodeType.COMMIT_MESSAGE or str(n.type) == "COMMIT_MESSAGE"), None)
        if not msg_node:
            return
            
        message = msg_node.attributes.get("content", "")
        
        try:
            matches = re.findall(task_regex, message)
        except re.error as e:
            logger.error(f"[ClickUpPlugin] Invalid regex '{task_regex}': {e}")
            return
            
        for task_id in matches:
            # If regex has groups, findall returns tuples or strings. 
            # We assume the matching group IS the task ID.
            if isinstance(task_id, tuple):
                # Join them? Or just take the first non-empty? 
                # Let's assume the user provided a regex where the first group is the ID.
                task_id = next((g for g in task_id if g), None)
                
            if not task_id:
                continue
                
            self._process_task(task_id, organization_id, commit_node, graph_api)

    def _process_task(self, task_id: str, team_id: str, commit_node: Node, graph_api: GraphInterface):
        cache_key = task_id
        if cache_key in self.task_cache:
            task_data = self.task_cache[cache_key]
        else:
            # ClickUp API
            # Try to fetch task. 
            # If team_id is provided, assume it might be a custom ID and append query param
            url = f"https://api.clickup.com/api/v2/task/{task_id}"
            params = {}
            if team_id:
                params["custom_task_ids"] = "true"
                params["team_id"] = team_id
                
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    # data usually has keys like 'id', 'name', 'text_content', 'status', 'team_id'
                    task_data = {
                        "id": data.get("id"),
                        "title": data.get("name"),
                        "status": data.get("status", {}).get("status"),
                        "url": data.get("url"),
                        "description": data.get("description"), # potentially large
                        "team_id": data.get("team_id")
                    }
                    self.task_cache[cache_key] = task_data
                else:
                    logger.warning(f"[ClickUpPlugin] Failed to fetch Task {task_id}: {response.status_code} {response.text}")
                    return
            except Exception as e:
                logger.error(f"[ClickUpPlugin] Error fetching task: {e}")
                return

        # Add to Graph
        node_id = f"CLICKUP_TASK:{task_data['id']}"
        graph_api.add_node(
            id=node_id,
            type="CLICKUP_TASK",
            attributes=task_data
        )
        
        graph_api.add_edge(
            source_id=commit_node.id,
            target_id=node_id,
            type="RELATED_TO_TASK"
        )

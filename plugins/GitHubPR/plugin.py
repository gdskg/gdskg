
import os
import re
import requests
import logging
from typing import List, Optional, Dict, Any
from core.plugin_interfaces import PluginInterface, GraphInterface
from core.schema import Node, Edge, NodeType

logger = logging.getLogger(__name__)

class GitHubPlugin(PluginInterface):
    plugin_type = "runtime"
    
    def __init__(self):
        self.pat = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
        self.enabled = True
        if not self.pat:
            logger.info("[GitHubPlugin] GITHUB_PAT/GITHUB_TOKEN not found. Running in unauthenticated mode (rate limited).")
        else:
            logger.info("[GitHubPlugin] Initialized with PAT.")
        
        # Simple in-memory cache for PRs to avoid repeated API calls
        # Key: (owner, repo, pr_number) -> Value: PR Node Attributes
        self.pr_cache = {}

    def process(self, commit_node: Node, related_nodes: List[Node], related_edges: List[Edge], graph_api: GraphInterface, config: Dict[str, Any] = None) -> None:
        if not self.enabled:
            return
            
        # Optional manual config token
        if config and "token" in config and not self.pat:
            self.pat = config["token"]

        # 1. Identify Repository Remote URL
        repo_node = next((n for n in related_nodes if n.type == NodeType.REPOSITORY or str(n.type) == "REPOSITORY"), None)
        if not repo_node:
            # Fallback: check edges? In current extractor logic, related_nodes should contain it.
            return
            
        remotes = repo_node.attributes.get("remotes", [])
        if not remotes:
            return

        # Extract owner/repo from first valid remote (simplification)
        # Supports: https://github.com/owner/repo.git or git@github.com:owner/repo.git
        owner, repo_name = self._parse_remote(remotes[0])
        if not owner or not repo_name:
            return

        # 2. Extract PR Number from Commit Message
        msg_node = next((n for n in related_nodes if n.type == NodeType.COMMIT_MESSAGE or str(n.type) == "COMMIT_MESSAGE"), None)
        if not msg_node:
            return
            
        message = msg_node.attributes.get("content", "")
        # Look for PR patterns like #123, (#123), etc.
        # This matches # followed by digits
        matches = re.findall(r"#(\d+)", message)
        
        for pr_num in matches:
            pr_node_id = f"PR:{owner}/{repo}#{pr_num}"
            if any(n.id == pr_node_id for n in related_nodes):
                continue
            self._process_pr(owner, repo_name, pr_num, commit_node, graph_api)

    def _parse_remote(self, remote_url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extracts (owner, repo) from a remote URL.
        """
        # Remove .git suffix
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]
            
        pattern = r"github\.com[:/]([^/]+)/([^/]+)$"
        match = re.search(pattern, remote_url)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _process_pr(self, owner: str, repo: str, pr_number: str, commit_node: Node, graph_api: GraphInterface):
        cache_key = (owner, repo, pr_number)
        
        if cache_key in self.pr_cache:
            pr_data = self.pr_cache[cache_key]
        else:
            # Fetch from GitHub API
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
            headers = {
                "Accept": "application/vnd.github.v3+json"
            }
            if self.pat:
                headers["Authorization"] = f"token {self.pat}"
                
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    pr_data = {
                        "number": data.get("number"),
                        "title": data.get("title"),
                        "body": data.get("body"), # ADDED PR DESCRIPTION
                        "state": data.get("state"),
                        "url": data.get("html_url"),
                        "author": data.get("user", {}).get("login"),
                        "created_at": data.get("created_at")
                    }
                    self.pr_cache[cache_key] = pr_data
                else:
                    logger.warning(f"[GitHubPlugin] Failed to fetch PR #{pr_number} for {owner}/{repo}: {response.status_code}")
                    return
            except Exception as e:
                logger.error(f"[GitHubPlugin] Error fetching PR: {e}")
                return

        # 3. Add to Graph
        pr_node_id = f"PR:{owner}/{repo}#{pr_number}"
        graph_api.add_node(
            id=pr_node_id,
            type="PULL_REQUEST",
            attributes=pr_data
        )
        
        # Edge: Commit -> Pull Request (RELATED_TO or PART_OF_PR?)
        # User requested: "related to the commit node"
        graph_api.add_edge(
            source_id=commit_node.id,
            target_id=pr_node_id,
            type="RELATED_TO_PR"
        )

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.schema import Node, Edge

class GraphInterface(ABC):
    """
    Interface provided to plugins to interact with the graph.
    """
    @abstractmethod
    def add_node(self, id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Insert a new node into the graph.
        """
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Insert a new edge into the graph.
        """
        pass

class PluginInterface(ABC):
    """
    Abstract base class that all plugins must implement.
    """
    @abstractmethod
    def process(self, commit_node: Node, related_nodes: List[Node], related_edges: List[Edge], graph_api: GraphInterface, config: Dict[str, Any] = None) -> None:
        """
        Process a commit and its related data.
        
        Args:
            commit_node: The main commit node being processed.
            related_nodes: List of nodes directly connected to the commit (files, author, etc.).
            related_edges: List of edges connecting the commit to related nodes.
            graph_api: Interface to add new nodes/edges to the graph.
            config: Optional configuration dictionary for the plugin.
        """
        pass

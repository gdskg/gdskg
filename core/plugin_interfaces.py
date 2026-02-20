from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.schema import Node, Edge

class GraphInterface(ABC):
    """
    Abstract interface for graph manipulation, provided to plugins.
    
    This interface decouples plugins from the underlying storage implementation,
    allowing them to add nodes and edges safely.
    """

    @abstractmethod
    def add_node(self, id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Add or update a node in the graph.

        Args:
            id (str): Unique identifier for the node.
            type (str): The type of the node.
            attributes (Dict[str, Any], optional): Properties of the node.
        """

        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, type: str, attributes: Dict[str, Any] = None) -> None:
        """
        Add or update an edge between two nodes in the graph.

        Args:
            source_id (str): The identifier of the source node.
            target_id (str): The identifier of the target node.
            type (str): The type of the relationship.
            attributes (Dict[str, Any], optional): Properties of the edge.
        """

        pass

class PluginInterface(ABC):
    """
    Base class for all GDSKG plugins.
    
    Plugins are invoked during the graph extraction process for each commit,
    enabling enrichment of the graph with metadata from external sources or
    custom analysis.
    """
    
    plugin_type: str = "build"  # Either 'build' or 'runtime'

    @abstractmethod
    def process(self, commit_node: Node, related_nodes: List[Node], related_edges: List[Edge], graph_api: GraphInterface, config: Dict[str, Any] = None) -> None:
        """
        Main hook for plugin logic execution.

        Args:
            commit_node (Node): The node representing the current commit.
            related_nodes (List[Node]): A list of nodes directly connected to the commit.
            related_edges (List[Edge]): A list of edges connected to the commit.
            graph_api (GraphInterface): The API object for adding to the graph.
            config (Dict[str, Any], optional): Plugin-specific configuration values.
        """
        pass


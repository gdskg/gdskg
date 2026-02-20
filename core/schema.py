from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib

class NodeType(str, Enum):
    """
    Enumeration of all possible node types in the knowledge graph.
    """

    COMMIT = "COMMIT"
    REPOSITORY = "REPOSITORY"
    AUTHOR = "AUTHOR"
    FILE = "FILE"
    TIME_BUCKET = "TIME_BUCKET"
    SYMBOL = "SYMBOL"
    FUNCTION = "FUNCTION"
    FUNCTION_HISTORY = "FUNCTION_HISTORY"
    FUNCTION_VERSION = "FUNCTION_VERSION"
    SECRET = "SECRET"
    COMMIT_MESSAGE = "COMMIT_MESSAGE"
    KEYWORD = "KEYWORD"
    COMMENT = "COMMENT"

class EdgeType(str, Enum):
    """
    Enumeration of all possible relationship (edge) types in the knowledge graph.
    """

    AUTHORED_BY = "AUTHORED_BY"
    PART_OF_REPO = "PART_OF_REPO"
    OCCURRED_IN = "OCCURRED_IN"
    MODIFIED_FILE = "MODIFIED_FILE"
    MODIFIED_SYMBOL = "MODIFIED_SYMBOL"
    MODIFIED_FUNCTION = "MODIFIED_FUNCTION"
    CONTEXTUAL_SYMBOL = "CONTEXTUAL_SYMBOL"
    REFERENCES_ENV = "REFERENCES_ENV"
    HAS_MESSAGE = "HAS_MESSAGE"
    HAS_SYMBOL = "HAS_SYMBOL"
    HAS_KEYWORD = "HAS_KEYWORD"
    CONTAINS_COMMENT = "CONTAINS_COMMENT"
    HAS_COMMENT = "HAS_COMMENT"
    HAS_HISTORY = "HAS_HISTORY"
    HAS_VERSION = "HAS_VERSION"
    PREVIOUS_VERSION = "PREVIOUS_VERSION"
    CREATED_VERSION = "CREATED_VERSION"
    COMMITTED_BY = "COMMITTED_BY"
    PARENT_OF = "PARENT_OF"

@dataclass
class Node:
    """
    Represents a single entity (node) within the knowledge graph.
    
    Attributes:
        id (str): A unique identifier for the node.
        type (NodeType): The category of the node.
        attributes (Dict[str, Any]): A flexible dictionary of additional metadata.
    """

    id: str

    type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """
        Alias for the node's unique identifier.
        """
        return self.id


@dataclass
class Edge:
    """
    Represents a relationship (edge) between two nodes in the graph.
    
    Attributes:
        source_id (str): The identifier of the starting node.
        target_id (str): The identifier of the ending node.
        type (EdgeType): The type of relationship.
        weight (float): A numerical value representing the strength of the edge.
        attributes (Dict[str, Any]): A flexible dictionary of additional metadata.
    """

    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        Generate a deterministic unique identifier for the edge.
        
        The ID is derived from a SHA-256 hash of the source ID, target ID, and edge type.
        """

        raw = f"{self.source_id}|{self.target_id}|{self.type}"

        return hashlib.sha256(raw.encode()).hexdigest()

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib

class NodeType(str, Enum):
    COMMIT = "COMMIT"
    REPOSITORY = "REPOSITORY"
    AUTHOR = "AUTHOR"
    FILE = "FILE"
    TIME_BUCKET = "TIME_BUCKET"
    SYMBOL = "SYMBOL"
    SECRET = "SECRET"
    COMMIT_MESSAGE = "COMMIT_MESSAGE"

class EdgeType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"
    PART_OF_REPO = "PART_OF_REPO"
    OCCURRED_IN = "OCCURRED_IN"
    MODIFIED_FILE = "MODIFIED_FILE"
    MODIFIED_SYMBOL = "MODIFIED_SYMBOL"
    CONTEXTUAL_SYMBOL = "CONTEXTUAL_SYMBOL"
    REFERENCES_ENV = "REFERENCES_ENV"
    HAS_MESSAGE = "HAS_MESSAGE"

@dataclass
class Node:
    id: str  # Primary Key
    type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.id

@dataclass
class Edge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        # Deterministic ID for edges based on source, target, and type
        raw = f"{self.source_id}|{self.target_id}|{self.type}"
        return hashlib.sha256(raw.encode()).hexdigest()

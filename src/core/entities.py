import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum


class EntityType(Enum):
    """
    Enumeration of universal entity types for maximum flexibility in the system.

    This enum defines the various types of entities that can exist within the Unified AI Agent System.
    Each entity type represents a distinct category of object with specific roles and functionalities.
    """

    AGENT = "agent"
    TASK = "task"
    MESSAGE = "message"
    MEMORY = "memory"
    PLUGIN = "plugin"
    PROMPT = "prompt"
    STATE = "state"
    CONTEXT = "context"
    CAPABILITY = "capability"
    PATTERN = "pattern"
    WORKFLOW = "workflow"
    RESOURCE = "resource"


class Priority(Enum):
    """
    Enumeration of universal priority levels for tasks and other entities.

    This enum provides a standardized way to assign importance and urgency to entities,
    particularly tasks, ensuring that critical operations are handled first.
    """

    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


class MemoryType(Enum):
    """
    Enumeration of hierarchical memory types.

    This enum categorizes different types of memory that the system can utilize,
    ranging from short-term working memory to long-term persistent storage.
    Each type serves a different purpose in the agent's cognitive architecture.
    """

    WORKING = "working"  # Immediate context
    EPISODIC = "episodic"  # Specific experiences
    SEMANTIC = "semantic"  # General knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    META = "meta"  # Self-awareness
    COLLECTIVE = "collective"  # Shared knowledge
    PERSISTENT = "persistent"  # Long-term storage


@dataclass
class UniversalEntity:
    """
    A universal entity that can represent anything in the system.

    This is the core data structure for all objects in the Unified AI Agent System.
    It is designed to be highly flexible and can represent agents, tasks, memories,
    and any other concept required by the system.

    Attributes:
        id: A unique identifier for the entity.
        type: The type of the entity, as defined by the EntityType enum.
        name: A human-readable name for the entity.
        description: A brief description of the entity's purpose or content.
        content: The main content or data of the entity.
        metadata: A dictionary for storing arbitrary metadata.
        capabilities: A set of capabilities associated with the entity.
        relationships: A dictionary representing relationships to other entities.
        state: A dictionary for storing the entity's current state.
        priority: The priority level of the entity.
        created_at: The timestamp when the entity was created.
        updated_at: The timestamp when the entity was last updated.
        accessed_at: The timestamp when the entity was last accessed.
        access_count: The number of times the entity has been accessed.
        importance: A score representing the importance of the entity.
        expires_at: An optional timestamp when the entity should expire.
        tags: A set of tags for categorizing and searching for the entity.
    """

    id: str = field(default_factory=lambda: f"entity_{uuid.uuid4().hex[:8]}")
    type: EntityType = EntityType.AGENT
    name: str = ""
    description: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    relationships: Dict[str, Set[str]] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Initializes the UniversalEntity after its creation.

        If the `name` attribute is not provided, it is automatically generated
        based on the entity's type and ID. This ensures that every entity has a
        default name.
        """
        if not self.name:
            self.name = f"{self.type.value}_{self.id}"

    def update(self, **kwargs):
        """
        Updates the entity with new data from keyword arguments.

        This method iterates through the provided keyword arguments and updates
        the corresponding attributes of the entity. It also updates the
        `updated_at`, `access_count`, and `accessed_at` timestamps.

        Args:
            **kwargs: A dictionary of attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.access_count += 1
        self.accessed_at = datetime.now()

    def add_capability(self, capability: str):
        """
        Adds a capability to the entity.

        Args:
            capability: The capability to add.
        """
        self.capabilities.add(capability)
        self.updated_at = datetime.now()

    def add_relationship(self, relation_type: str, entity_id: str):
        """
        Adds a relationship to another entity.

        Args:
            relation_type: The type of the relationship (e.g., "parent", "child").
            entity_id: The ID of the entity to which the relationship is being added.
        """
        if relation_type not in self.relationships:
            self.relationships[relation_type] = set()
        self.relationships[relation_type].add(entity_id)
        self.updated_at = datetime.now()

    def add_tag(self, tag: str):
        """
        Adds a tag to the entity.

        Args:
            tag: The tag to add.
        """
        self.tags.add(tag)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the entity to a dictionary.

        This method serializes the entity's attributes into a dictionary,
        making it suitable for storage or transmission. It handles the
        conversion of enums, sets, and datetime objects to JSON-serializable
        formats.

        Returns:
            A dictionary representation of the entity.
        """
        return {
            **asdict(self),
            "type": self.type.value,
            "priority": self.priority.name,
            "capabilities": list(self.capabilities),
            "relationships": {k: list(v) for k, v in self.relationships.items()},
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalEntity":
        """
        Creates an entity from a dictionary.

        This class method deserializes a dictionary into a UniversalEntity
        object. It handles the conversion of string representations of enums,
        lists, and ISO-formatted timestamps back to their respective Python
        types.

        Args:
            data: A dictionary containing the entity's attributes.

        Returns:
            A UniversalEntity object.
        """
        data["type"] = EntityType(data["type"])
        data["priority"] = Priority[data["priority"]]
        data["capabilities"] = set(data.get("capabilities", []))
        data["relationships"] = {
            k: set(v) for k, v in data.get("relationships", {}).items()
        }
        data["tags"] = set(data.get("tags", []))
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["accessed_at"] = datetime.fromisoformat(data["accessed_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)

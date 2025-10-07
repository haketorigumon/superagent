import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum


class EntityType(Enum):
    """Universal entity types for maximum flexibility"""
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
    """Universal priority system"""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


class MemoryType(Enum):
    """Hierarchical memory types"""
    WORKING = "working"      # Immediate context
    EPISODIC = "episodic"    # Specific experiences
    SEMANTIC = "semantic"    # General knowledge
    PROCEDURAL = "procedural" # How-to knowledge
    META = "meta"           # Self-awareness
    COLLECTIVE = "collective" # Shared knowledge
    PERSISTENT = "persistent" # Long-term storage


@dataclass
class UniversalEntity:
    """Universal entity that can represent anything in the system"""
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
        if not self.name:
            self.name = f"{self.type.value}_{self.id}"

    def update(self, **kwargs):
        """Update entity with new data"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.access_count += 1
        self.accessed_at = datetime.now()

    def add_capability(self, capability: str):
        """Add a capability"""
        self.capabilities.add(capability)
        self.updated_at = datetime.now()

    def add_relationship(self, relation_type: str, entity_id: str):
        """Add a relationship to another entity"""
        if relation_type not in self.relationships:
            self.relationships[relation_type] = set()
        self.relationships[relation_type].add(entity_id)
        self.updated_at = datetime.now()

    def add_tag(self, tag: str):
        """Add a tag"""
        self.tags.add(tag)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'type': self.type.value,
            'priority': self.priority.name,
            'capabilities': list(self.capabilities),
            'relationships': {k: list(v) for k, v in self.relationships.items()},
            'tags': list(self.tags),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UniversalEntity':
        """Create from dictionary"""
        data['type'] = EntityType(data['type'])
        data['priority'] = Priority[data['priority']]
        data['capabilities'] = set(data.get('capabilities', []))
        data['relationships'] = {k: set(v) for k, v in data.get('relationships', {}).items()}
        data['tags'] = set(data.get('tags', []))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['accessed_at'] = datetime.fromisoformat(data['accessed_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
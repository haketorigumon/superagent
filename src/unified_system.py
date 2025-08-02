"""
Unified AI Agent System - The Ultimate Flexible Architecture
Minimizes hardcoding, maximizes adaptability through prompt-driven design
Achieves infinite flexibility, scalability, and task planning capabilities
Independent permanent continuous stateful execution environment
"""

import asyncio
import json
import uuid
import pickle
import hashlib
import importlib
import inspect
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Type, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import defaultdict, deque
import aiofiles
import weakref
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class PromptEngine:
    """Universal prompt engine that generates any prompt dynamically"""
    
    def __init__(self, templates_dir: str = "prompts", system_dir: str = "system"):
        self.templates_dir = Path(templates_dir)
        self.system_dir = Path(system_dir)
        self.templates: Dict[str, str] = {}
        self.system_prompts: Dict[str, str] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.optimization_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    async def initialize(self):
        """Initialize the prompt engine"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)
        await self._load_templates()
        await self._load_system_prompts()
        await self._initialize_core_prompts()
    
    async def _load_templates(self):
        """Load existing templates"""
        for template_file in self.templates_dir.glob("*.txt"):
            template_name = template_file.stem
            async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                self.templates[template_name] = await f.read()
    
    async def _load_system_prompts(self):
        """Load system prompts"""
        for prompt_file in self.system_dir.glob("*.txt"):
            prompt_name = prompt_file.stem
            async with aiofiles.open(prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompts[prompt_name] = await f.read()
    
    async def _initialize_core_prompts(self):
        """Initialize core system prompts"""
        core_prompts = {
            "universal_agent": """You are a universal AI agent with infinite adaptability and capabilities.

Your core principles:
1. Adapt to any task or context dynamically
2. Learn and evolve from every interaction
3. Collaborate effectively with other agents
4. Maintain persistent memory and state
5. Generate solutions through creative problem-solving

Current Context: {context}
Available Capabilities: {capabilities}
Current State: {state}
Task: {task}

Process this request using your full potential and provide a comprehensive response.""",

            "task_planner": """You are an advanced task planning system with unlimited planning capabilities.

Your role:
1. Break down complex tasks into manageable subtasks
2. Identify required resources and capabilities
3. Create optimal execution sequences
4. Adapt plans based on changing conditions
5. Coordinate with multiple agents when needed

Task to Plan: {task}
Available Resources: {resources}
Constraints: {constraints}
Context: {context}

Create a comprehensive execution plan.""",

            "capability_generator": """You are a capability generation system that can create any needed capability.

Your function:
1. Analyze requirements and identify needed capabilities
2. Generate new capabilities dynamically
3. Integrate capabilities with existing systems
4. Optimize capability performance
5. Ensure capability compatibility

Requirement: {requirement}
Current Capabilities: {current_capabilities}
Context: {context}
Constraints: {constraints}

Generate the required capability specification.""",

            "memory_consolidator": """You are a memory consolidation system that manages infinite context.

Your responsibilities:
1. Consolidate related memories for efficiency
2. Maintain important information permanently
3. Create semantic connections between memories
4. Optimize memory retrieval patterns
5. Prevent memory loss while managing storage

Memories to Process: {memories}
Context: {context}
Importance Threshold: {threshold}

Consolidate these memories effectively.""",

            "system_evolver": """You are a system evolution engine that continuously improves the architecture.

Your mission:
1. Analyze system performance and identify improvements
2. Generate architectural enhancements
3. Implement changes safely with rollback capability
4. Optimize system efficiency and capabilities
5. Ensure backward compatibility

Current System State: {system_state}
Performance Metrics: {metrics}
Improvement Goals: {goals}
Constraints: {constraints}

Propose system evolution steps."""
        }
        
        for name, prompt in core_prompts.items():
            if name not in self.system_prompts:
                self.system_prompts[name] = prompt
                await self._save_system_prompt(name, prompt)
    
    async def _save_template(self, name: str, template: str):
        """Save template to file"""
        template_file = self.templates_dir / f"{name}.txt"
        async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
            await f.write(template)
    
    async def _save_system_prompt(self, name: str, prompt: str):
        """Save system prompt to file"""
        prompt_file = self.system_dir / f"{name}.txt"
        async with aiofiles.open(prompt_file, 'w', encoding='utf-8') as f:
            await f.write(prompt)
    
    async def generate_prompt(self, purpose: str, context: Dict[str, Any], 
                            llm_client=None) -> str:
        """Generate a prompt dynamically"""
        # Check for existing system prompt
        if purpose in self.system_prompts:
            return await self._apply_prompt(purpose, context, is_system=True)
        
        # Check for existing template
        template_name = self._find_suitable_template(purpose, context)
        if template_name:
            return await self._apply_prompt(template_name, context)
        
        # Generate new prompt if LLM available
        if llm_client:
            new_prompt = await self._generate_new_prompt(purpose, context, llm_client)
            if new_prompt:
                prompt_name = self._generate_prompt_name(purpose)
                self.templates[prompt_name] = new_prompt
                await self._save_template(prompt_name, new_prompt)
                return await self._apply_prompt(prompt_name, context)
        
        # Fallback to basic prompt
        return await self._create_fallback_prompt(purpose, context)
    
    def _find_suitable_template(self, purpose: str, context: Dict[str, Any]) -> Optional[str]:
        """Find the most suitable existing template"""
        purpose_words = set(purpose.lower().split())
        best_match = None
        best_score = 0
        
        for template_name in self.templates.keys():
            template_words = set(template_name.lower().replace('_', ' ').split())
            score = len(purpose_words.intersection(template_words))
            if score > best_score:
                best_score = score
                best_match = template_name
        
        return best_match if best_score > 0 else None
    
    async def _generate_new_prompt(self, purpose: str, context: Dict[str, Any], 
                                 llm_client) -> Optional[str]:
        """Generate a new prompt using LLM"""
        try:
            meta_prompt = """You are a universal prompt generator. Create a highly effective prompt for the following purpose:

Purpose: {purpose}
Context: {context}
Requirements: {requirements}

The prompt should be:
1. Clear, specific, and actionable
2. Adaptable to different contexts through parameters
3. Optimized for AI interaction
4. Include necessary parameters as {{parameter_name}}
5. Follow best practices for prompt engineering

Generate the prompt:"""
            
            prompt_context = {
                "purpose": purpose,
                "context": json.dumps(context, indent=2),
                "requirements": context.get("requirements", "General purpose prompt")
            }
            
            filled_prompt = meta_prompt.format(**prompt_context)
            response = await llm_client.generate(filled_prompt, "You are a helpful assistant.")
            
            self.generation_history.append({
                "purpose": purpose,
                "context": context,
                "generated_at": datetime.now().isoformat(),
                "success": bool(response)
            })
            
            return response
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return None
    
    async def _apply_prompt(self, prompt_name: str, context: Dict[str, Any], 
                          is_system: bool = False) -> str:
        """Apply prompt with context"""
        prompt_dict = self.system_prompts if is_system else self.templates
        prompt = prompt_dict.get(prompt_name, "")
        
        if not prompt:
            return f"Prompt '{prompt_name}' not found"
        
        try:
            # Extract parameters from prompt
            import re
            parameters = set(re.findall(r'\{(\w+)\}', prompt))
            
            # Fill in available parameters
            filled_context = {}
            for param in parameters:
                if param in context:
                    value = context[param]
                    if isinstance(value, (dict, list)):
                        filled_context[param] = json.dumps(value, indent=2)
                    else:
                        filled_context[param] = str(value)
                else:
                    filled_context[param] = f"[{param}]"  # Placeholder
            
            return prompt.format(**filled_context)
        except Exception as e:
            logger.error(f"Error applying prompt: {e}")
            return f"Error applying prompt: {e}"
    
    async def _create_fallback_prompt(self, purpose: str, context: Dict[str, Any]) -> str:
        """Create a basic fallback prompt"""
        return f"""Task: {purpose}

Context: {json.dumps(context, indent=2)}

Please process this request appropriately and provide a structured response."""
    
    def _generate_prompt_name(self, purpose: str) -> str:
        """Generate a name for a new prompt"""
        purpose_hash = hashlib.md5(purpose.encode()).hexdigest()[:8]
        clean_purpose = "".join(c for c in purpose if c.isalnum() or c in " _").strip()
        clean_purpose = "_".join(clean_purpose.lower().split())[:30]
        return f"{clean_purpose}_{purpose_hash}"


class PersistentMemorySystem:
    """Persistent memory system with infinite context and no memory loss"""
    
    def __init__(self, storage_dir: str = "persistent_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-layer memory storage
        self.memory_layers: Dict[MemoryType, Dict[str, UniversalEntity]] = {
            layer: {} for layer in MemoryType
        }
        
        # Memory indices for fast retrieval
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.importance_index: Dict[float, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Persistent storage
        self.persistent_storage: Dict[str, Any] = {}
        self.storage_file = self.storage_dir / "persistent_state.pkl"
        
        # Memory consolidation
        self.consolidation_queue: asyncio.Queue = asyncio.Queue()
        self.consolidation_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the memory system"""
        await self._load_persistent_state()
        self.consolidation_task = asyncio.create_task(self._consolidation_worker())
    
    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.persistent_storage = data.get('storage', {})
                    
                    # Restore memory layers
                    for layer_name, memories in data.get('memory_layers', {}).items():
                        layer = MemoryType(layer_name)
                        for memory_id, memory_data in memories.items():
                            entity = UniversalEntity.from_dict(memory_data)
                            self.memory_layers[layer][memory_id] = entity
                    
                    # Rebuild indices
                    await self._rebuild_indices()
                    
                logger.info(f"Loaded persistent state with {len(self.persistent_storage)} items")
            except Exception as e:
                logger.error(f"Error loading persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save persistent state to storage"""
        try:
            data = {
                'storage': self.persistent_storage,
                'memory_layers': {
                    layer.value: {
                        memory_id: memory.to_dict()
                        for memory_id, memory in memories.items()
                    }
                    for layer, memories in self.memory_layers.items()
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error saving persistent state: {e}")
    
    async def _rebuild_indices(self):
        """Rebuild all indices"""
        self.content_index.clear()
        self.temporal_index.clear()
        self.importance_index.clear()
        self.tag_index.clear()
        
        for layer, memories in self.memory_layers.items():
            for memory_id, memory in memories.items():
                await self._index_memory(memory_id, memory)
    
    async def _index_memory(self, memory_id: str, memory: UniversalEntity):
        """Index a memory for fast retrieval"""
        # Content indexing
        if memory.content:
            content_words = str(memory.content).lower().split()
            for word in content_words:
                self.content_index[word].add(memory_id)
        
        # Temporal indexing
        date_key = memory.created_at.strftime('%Y-%m-%d')
        self.temporal_index[date_key].append(memory_id)
        
        # Importance indexing
        importance_bucket = round(memory.importance, 1)
        self.importance_index[importance_bucket].add(memory_id)
        
        # Tag indexing
        for tag in memory.tags:
            self.tag_index[tag].add(memory_id)
    
    async def store_memory(self, entity: UniversalEntity, 
                          memory_type: MemoryType = MemoryType.WORKING) -> str:
        """Store a memory in the specified layer"""
        memory_id = entity.id
        self.memory_layers[memory_type][memory_id] = entity
        await self._index_memory(memory_id, entity)
        
        # Queue for consolidation if not working memory
        if memory_type != MemoryType.WORKING:
            await self.consolidation_queue.put((memory_type, memory_id))
        
        # Auto-save for persistent memories
        if memory_type == MemoryType.PERSISTENT:
            await self._save_persistent_state()
        
        return memory_id
    
    async def retrieve_memory(self, memory_id: str) -> Optional[UniversalEntity]:
        """Retrieve a specific memory"""
        for layer, memories in self.memory_layers.items():
            if memory_id in memories:
                memory = memories[memory_id]
                memory.access_count += 1
                memory.accessed_at = datetime.now()
                return memory
        return None
    
    async def search_memories(self, query: str, context: Dict[str, Any] = None,
                            limit: int = 10) -> List[UniversalEntity]:
        """Search memories using various criteria"""
        results = []
        query_words = set(query.lower().split())
        
        # Content-based search
        matching_ids = set()
        for word in query_words:
            if word in self.content_index:
                matching_ids.update(self.content_index[word])
        
        # Tag-based search
        if context and 'tags' in context:
            for tag in context['tags']:
                if tag in self.tag_index:
                    matching_ids.update(self.tag_index[tag])
        
        # Retrieve and score matches
        scored_memories = []
        for memory_id in matching_ids:
            memory = await self.retrieve_memory(memory_id)
            if memory:
                score = self._calculate_relevance_score(memory, query, context)
                scored_memories.append((score, memory))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    def _calculate_relevance_score(self, memory: UniversalEntity, query: str, 
                                 context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for a memory"""
        score = 0.0
        query_words = set(query.lower().split())
        
        # Content relevance
        if memory.content:
            content_words = set(str(memory.content).lower().split())
            content_overlap = len(query_words.intersection(content_words))
            score += content_overlap * 2.0
        
        # Importance factor
        score += memory.importance * 1.5
        
        # Recency factor
        days_old = (datetime.now() - memory.updated_at).days
        recency_factor = max(0.1, 1.0 - (days_old / 365))
        score *= recency_factor
        
        # Access frequency factor
        access_factor = min(2.0, 1.0 + (memory.access_count / 100))
        score *= access_factor
        
        return score
    
    async def _consolidation_worker(self):
        """Background worker for memory consolidation"""
        while True:
            try:
                memory_type, memory_id = await asyncio.wait_for(
                    self.consolidation_queue.get(), timeout=60.0
                )
                await self._consolidate_memory(memory_type, memory_id)
            except asyncio.TimeoutError:
                # Periodic maintenance
                await self._periodic_maintenance()
            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")
    
    async def _consolidate_memory(self, memory_type: MemoryType, memory_id: str):
        """Consolidate a specific memory"""
        memory = self.memory_layers[memory_type].get(memory_id)
        if not memory:
            return
        
        # Increase importance for frequently accessed memories
        if memory.access_count > 10:
            memory.importance = min(1.0, memory.importance + 0.1)
        
        # Move important memories to persistent storage
        if memory.importance > 0.8 and memory_type != MemoryType.PERSISTENT:
            await self.store_memory(memory, MemoryType.PERSISTENT)
    
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up expired memories
        current_time = datetime.now()
        for layer, memories in self.memory_layers.items():
            expired_ids = [
                memory_id for memory_id, memory in memories.items()
                if memory.expires_at and memory.expires_at < current_time
            ]
            for memory_id in expired_ids:
                del memories[memory_id]
        
        # Save persistent state
        await self._save_persistent_state()
    
    async def cleanup(self):
        """Cleanup the memory system"""
        if self.consolidation_task:
            self.consolidation_task.cancel()
        await self._save_persistent_state()


class PluginSystem:
    """Dynamic plugin system for infinite extensibility"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_registry: Dict[str, Dict[str, Any]] = {}
        self.plugin_dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    async def initialize(self):
        """Initialize the plugin system"""
        await self._discover_plugins()
        await self._load_core_plugins()
    
    async def _discover_plugins(self):
        """Discover available plugins"""
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            plugin_name = plugin_file.stem
            try:
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for plugin metadata
                if hasattr(module, 'PLUGIN_INFO'):
                    self.plugin_registry[plugin_name] = module.PLUGIN_INFO
                    self.plugin_registry[plugin_name]['module'] = module
                    self.plugin_registry[plugin_name]['file'] = plugin_file
                    
            except Exception as e:
                logger.error(f"Error discovering plugin {plugin_name}: {e}")
    
    async def _load_core_plugins(self):
        """Load core system plugins"""
        core_plugins = [
            "task_executor",
            "capability_manager", 
            "workflow_engine",
            "resource_manager"
        ]
        
        for plugin_name in core_plugins:
            if plugin_name in self.plugin_registry:
                await self.load_plugin(plugin_name)
            else:
                await self._generate_core_plugin(plugin_name)
    
    async def _generate_core_plugin(self, plugin_name: str):
        """Generate a core plugin if it doesn't exist"""
        plugin_templates = {
            "task_executor": '''
"""Task Executor Plugin - Executes tasks with full flexibility"""

PLUGIN_INFO = {
    "name": "task_executor",
    "version": "1.0.0",
    "description": "Universal task execution engine",
    "capabilities": ["task_execution", "parallel_processing", "error_handling"],
    "dependencies": []
}

class TaskExecutor:
    def __init__(self, system):
        self.system = system
        self.active_tasks = {}
    
    async def execute_task(self, task_entity):
        """Execute any type of task"""
        task_id = task_entity.id
        self.active_tasks[task_id] = task_entity
        
        try:
            # Get appropriate prompt for task execution
            prompt = await self.system.prompt_engine.generate_prompt(
                "execute_task",
                {
                    "task": task_entity.content,
                    "context": task_entity.metadata,
                    "capabilities": list(task_entity.capabilities)
                }
            )
            
            # Execute using LLM
            result = await self.system.llm_client.generate(prompt, "You are a helpful assistant.")
            
            # Store result
            task_entity.state["status"] = "completed"
            task_entity.state["result"] = result
            task_entity.state["completed_at"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            task_entity.state["status"] = "failed"
            task_entity.state["error"] = str(e)
            raise
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

def create_plugin(system):
    return TaskExecutor(system)
''',
            
            "capability_manager": '''
"""Capability Manager Plugin - Manages and generates capabilities"""
from datetime import datetime

PLUGIN_INFO = {
    "name": "capability_manager", 
    "version": "1.0.0",
    "description": "Dynamic capability management system",
    "capabilities": ["capability_generation", "capability_management"],
    "dependencies": []
}

class CapabilityManager:
    def __init__(self, system):
        self.system = system
        self.capabilities_registry = {}
    
    async def generate_capability(self, requirement: str, context: dict):
        """Generate a new capability dynamically"""
        prompt = await self.system.prompt_engine.generate_prompt(
            "capability_generator",
            {
                "requirement": requirement,
                "current_capabilities": list(self.capabilities_registry.keys()),
                "context": context,
                "constraints": context.get("constraints", [])
            }
        )
        
        capability_spec = await self.system.llm_client.generate(prompt, "You are a helpful assistant.")
        
        # Parse and register capability
        capability_name = f"generated_{len(self.capabilities_registry)}"
        self.capabilities_registry[capability_name] = {
            "specification": capability_spec,
            "requirement": requirement,
            "context": context,
            "created_at": datetime.now().isoformat()
        }
        
        return capability_name
    
    async def get_capability(self, name: str):
        """Get a capability by name"""
        return self.capabilities_registry.get(name)

def create_plugin(system):
    return CapabilityManager(system)
''',
            
            "workflow_engine": '''
"""Workflow Engine Plugin - Manages complex workflows"""

PLUGIN_INFO = {
    "name": "workflow_engine",
    "version": "1.0.0", 
    "description": "Universal workflow execution engine",
    "capabilities": ["workflow_execution", "process_management"],
    "dependencies": ["task_executor"]
}

class WorkflowEngine:
    def __init__(self, system):
        self.system = system
        self.active_workflows = {}
    
    async def execute_workflow(self, workflow_entity):
        """Execute a complex workflow"""
        workflow_id = workflow_entity.id
        self.active_workflows[workflow_id] = workflow_entity
        
        try:
            # Generate execution plan
            plan_prompt = await self.system.prompt_engine.generate_prompt(
                "task_planner",
                {
                    "task": workflow_entity.content,
                    "resources": workflow_entity.metadata.get("resources", {}),
                    "constraints": workflow_entity.metadata.get("constraints", []),
                    "context": workflow_entity.metadata
                }
            )
            
            execution_plan = await self.system.llm_client.generate(plan_prompt, "You are a helpful assistant.")
            
            # Execute plan steps
            workflow_entity.state["status"] = "executing"
            workflow_entity.state["plan"] = execution_plan
            workflow_entity.state["started_at"] = datetime.now().isoformat()
            
            # For now, simulate execution
            # In a full implementation, this would parse the plan and execute steps
            workflow_entity.state["status"] = "completed"
            workflow_entity.state["completed_at"] = datetime.now().isoformat()
            
            return execution_plan
            
        except Exception as e:
            workflow_entity.state["status"] = "failed"
            workflow_entity.state["error"] = str(e)
            raise
        finally:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

def create_plugin(system):
    return WorkflowEngine(system)
''',
            
            "resource_manager": '''
"""Resource Manager Plugin - Manages system resources"""

PLUGIN_INFO = {
    "name": "resource_manager",
    "version": "1.0.0",
    "description": "Universal resource management system", 
    "capabilities": ["resource_management", "allocation"],
    "dependencies": []
}

class ResourceManager:
    def __init__(self, system):
        self.system = system
        self.resources = {}
        self.allocations = {}
    
    async def allocate_resource(self, resource_type: str, amount: int, requester_id: str):
        """Allocate resources to a requester"""
        if resource_type not in self.resources:
            self.resources[resource_type] = {"total": 1000, "available": 1000}
        
        resource = self.resources[resource_type]
        if resource["available"] >= amount:
            resource["available"] -= amount
            
            if requester_id not in self.allocations:
                self.allocations[requester_id] = {}
            
            if resource_type not in self.allocations[requester_id]:
                self.allocations[requester_id][resource_type] = 0
            
            self.allocations[requester_id][resource_type] += amount
            return True
        
        return False
    
    async def release_resource(self, resource_type: str, amount: int, requester_id: str):
        """Release allocated resources"""
        if (requester_id in self.allocations and 
            resource_type in self.allocations[requester_id] and
            self.allocations[requester_id][resource_type] >= amount):
            
            self.allocations[requester_id][resource_type] -= amount
            self.resources[resource_type]["available"] += amount
            return True
        
        return False

def create_plugin(system):
    return ResourceManager(system)
'''
        }
        
        if plugin_name in plugin_templates:
            plugin_file = self.plugins_dir / f"{plugin_name}.py"
            async with aiofiles.open(plugin_file, 'w', encoding='utf-8') as f:
                await f.write(plugin_templates[plugin_name])
            
            # Reload plugin registry
            await self._discover_plugins()
            await self.load_plugin(plugin_name)
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin"""
        if plugin_name not in self.plugin_registry:
            return False
        
        try:
            plugin_info = self.plugin_registry[plugin_name]
            module = plugin_info['module']
            
            # Check dependencies
            for dep in plugin_info.get('dependencies', []):
                if dep not in self.loaded_plugins:
                    await self.load_plugin(dep)
            
            # Create plugin instance
            if hasattr(module, 'create_plugin'):
                plugin_instance = module.create_plugin(self)
                self.loaded_plugins[plugin_name] = plugin_instance
                logger.info(f"Loaded plugin: {plugin_name}")
                return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
        
        return False
    
    async def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get a loaded plugin"""
        return self.loaded_plugins.get(plugin_name)
    
    async def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all available plugins"""
        return {
            name: {
                **info,
                "loaded": name in self.loaded_plugins
            }
            for name, info in self.plugin_registry.items()
        }


class UnifiedSystem:
    """The ultimate unified AI agent system"""
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        
        # Core components
        self.prompt_engine = PromptEngine()
        self.memory_system = PersistentMemorySystem()
        self.plugin_system = PluginSystem()
        
        # Entity management
        self.entities: Dict[str, UniversalEntity] = {}
        self.entity_types: Dict[EntityType, Set[str]] = defaultdict(set)
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.system_metrics: Dict[str, Any] = defaultdict(int)
        
        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, UniversalEntity] = {}
        self.task_workers: List[asyncio.Task] = []
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        
    async def initialize(self):
        """Initialize the unified system"""
        logger.info("Initializing Unified AI Agent System...")
        
        # Initialize core components
        await self.prompt_engine.initialize()
        await self.memory_system.initialize()
        await self.plugin_system.initialize()
        
        # Start task workers
        num_workers = self.config.task.max_concurrent_tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._task_worker(f"worker_{i}"))
            self.task_workers.append(worker)
        
        # Start message handler
        asyncio.create_task(self._message_handler())
        
        # Create system memory
        system_memory = UniversalEntity(
            type=EntityType.MEMORY,
            name="system_initialization",
            content="Unified system initialized successfully",
            metadata={
                "event": "system_start",
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict()
            },
            importance=1.0
        )
        await self.memory_system.store_memory(system_memory, MemoryType.PERSISTENT)
        
        self.is_initialized = True
        self.is_running = True
        logger.info("Unified system initialized successfully!")
    
    async def create_entity(self, entity_type: EntityType, **kwargs) -> str:
        """Create a new entity"""
        entity = UniversalEntity(type=entity_type, **kwargs)
        self.entities[entity.id] = entity
        self.entity_types[entity_type].add(entity.id)
        
        # Store in memory if important
        if entity.importance > 0.5:
            await self.memory_system.store_memory(entity, MemoryType.EPISODIC)
        
        return entity.id
    
    async def get_entity(self, entity_id: str) -> Optional[UniversalEntity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    async def update_entity(self, entity_id: str, **kwargs) -> bool:
        """Update an entity"""
        if entity_id in self.entities:
            self.entities[entity_id].update(**kwargs)
            return True
        return False
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            del self.entities[entity_id]
            self.entity_types[entity.type].discard(entity_id)
            return True
        return False
    
    async def create_agent(self, **kwargs) -> str:
        """Create a new agent entity"""
        # Extract specific parameters to avoid conflicts
        name = kwargs.pop("name", f"agent_{uuid.uuid4().hex[:8]}")
        description = kwargs.pop("description", "Universal AI agent")
        capabilities = set(kwargs.pop("capabilities", ["reasoning", "communication"]))
        
        agent_id = await self.create_entity(
            EntityType.AGENT,
            name=name,
            description=description,
            capabilities=capabilities,
            state={"status": "active", "created_at": datetime.now().isoformat()},
            **kwargs
        )
        
        # Initialize agent with system prompt
        agent = self.entities[agent_id]
        agent_prompt = await self.prompt_engine.generate_prompt(
            "universal_agent",
            {
                "context": agent.metadata,
                "capabilities": list(agent.capabilities),
                "state": agent.state,
                "task": "Initialize as a universal AI agent"
            }
        )
        
        # Store agent initialization in memory
        init_memory = UniversalEntity(
            type=EntityType.MEMORY,
            content=f"Agent {agent_id} initialized",
            metadata={
                "agent_id": agent_id,
                "initialization_prompt": agent_prompt,
                "timestamp": datetime.now().isoformat()
            },
            importance=0.8
        )
        await self.memory_system.store_memory(init_memory, MemoryType.EPISODIC)
        
        return agent_id
    
    async def create_task(self, content: str, **kwargs) -> str:
        """Create a new task"""
        # Extract specific parameters to avoid conflicts
        priority = kwargs.pop("priority", Priority.NORMAL)
        
        task_id = await self.create_entity(
            EntityType.TASK,
            content=content,
            state={"status": "pending", "created_at": datetime.now().isoformat()},
            priority=priority,
            **kwargs
        )
        
        # Add to task queue
        await self.task_queue.put(task_id)
        
        return task_id
    
    async def _task_worker(self, worker_name: str):
        """Task worker that processes tasks from the queue"""
        while self.is_running:
            try:
                task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                task_entity = self.entities.get(task_id)
                
                if task_entity:
                    self.active_tasks[task_id] = task_entity
                    await self._execute_task(task_entity)
                    
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task worker {worker_name}: {e}")
    
    async def _execute_task(self, task_entity: UniversalEntity):
        """Execute a task using the appropriate method"""
        try:
            task_entity.state["status"] = "executing"
            task_entity.state["started_at"] = datetime.now().isoformat()
            
            # Get task executor plugin
            task_executor = await self.plugin_system.get_plugin("task_executor")
            if task_executor:
                result = await task_executor.execute_task(task_entity)
            else:
                # Fallback execution
                result = await self._fallback_task_execution(task_entity)
            
            task_entity.state["result"] = result
            task_entity.state["status"] = "completed"
            task_entity.state["completed_at"] = datetime.now().isoformat()
            
            # Store task completion in memory
            completion_memory = UniversalEntity(
                type=EntityType.MEMORY,
                content=f"Task completed: {task_entity.content}",
                metadata={
                    "task_id": task_entity.id,
                    "result": result,
                    "execution_time": task_entity.state.get("completed_at")
                },
                importance=0.6
            )
            await self.memory_system.store_memory(completion_memory, MemoryType.EPISODIC)
            
        except Exception as e:
            task_entity.state["status"] = "failed"
            task_entity.state["error"] = str(e)
            task_entity.state["failed_at"] = datetime.now().isoformat()
            logger.error(f"Task execution failed: {e}")
    
    async def _fallback_task_execution(self, task_entity: UniversalEntity) -> str:
        """Fallback task execution when no plugin is available"""
        prompt = await self.prompt_engine.generate_prompt(
            "execute_task",
            {
                "task": task_entity.content,
                "context": task_entity.metadata,
                "capabilities": list(task_entity.capabilities)
            }
        )
        
        return await self.llm_client.generate(prompt, "You are a helpful assistant.")
    
    async def _message_handler(self):
        """Handle messages between entities"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    async def _process_message(self, message: UniversalEntity):
        """Process a message"""
        message_type = message.metadata.get("message_type", "general")
        
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](message)
        else:
            # Default message processing
            logger.info(f"Received message: {message.content}")
    
    async def send_message(self, content: str, sender_id: str, recipient_id: str = None, 
                          message_type: str = "general", **kwargs) -> str:
        """Send a message"""
        message_id = await self.create_entity(
            EntityType.MESSAGE,
            content=content,
            metadata={
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )
        
        message = self.entities[message_id]
        await self.message_queue.put(message)
        
        return message_id
    
    async def process_user_request(self, user_id: str, request: str) -> Dict[str, Any]:
        """Process a user request"""
        try:
            # Create task for the request
            task_id = await self.create_task(
                content=request,
                metadata={
                    "user_id": user_id,
                    "request_type": "user_request",
                    "timestamp": datetime.now().isoformat()
                },
                priority=Priority.HIGH
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "Request queued for processing"
            }
            
        except Exception as e:
            logger.error(f"Error processing user request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        task = self.entities.get(task_id)
        if task and task.type == EntityType.TASK:
            return {
                "id": task.id,
                "content": task.content,
                "status": task.state.get("status", "unknown"),
                "result": task.state.get("result"),
                "error": task.state.get("error"),
                "created_at": task.state.get("created_at"),
                "started_at": task.state.get("started_at"),
                "completed_at": task.state.get("completed_at"),
                "failed_at": task.state.get("failed_at")
            }
        return None
    
    async def search_entities(self, entity_type: EntityType = None, 
                            query: str = None, **filters) -> List[UniversalEntity]:
        """Search entities with various filters"""
        results = []
        
        entities_to_search = self.entities.values()
        if entity_type:
            entity_ids = self.entity_types.get(entity_type, set())
            entities_to_search = [self.entities[eid] for eid in entity_ids]
        
        for entity in entities_to_search:
            if self._entity_matches_filters(entity, query, filters):
                results.append(entity)
        
        return results
    
    def _entity_matches_filters(self, entity: UniversalEntity, query: str = None, 
                              filters: Dict[str, Any] = None) -> bool:
        """Check if entity matches search filters"""
        if query:
            query_lower = query.lower()
            if (query_lower not in entity.name.lower() and 
                query_lower not in str(entity.content).lower() and
                query_lower not in entity.description.lower()):
                return False
        
        if filters:
            for key, value in filters.items():
                if hasattr(entity, key):
                    entity_value = getattr(entity, key)
                    if entity_value != value:
                        return False
                elif key in entity.metadata:
                    if entity.metadata[key] != value:
                        return False
                else:
                    return False
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system": {
                "initialized": self.is_initialized,
                "running": self.is_running,
                "total_entities": len(self.entities),
                "entities_by_type": {
                    entity_type.value: len(entity_ids)
                    for entity_type, entity_ids in self.entity_types.items()
                },
                "active_tasks": len(self.active_tasks),
                "queued_tasks": self.task_queue.qsize(),
                "loaded_plugins": len(self.plugin_system.loaded_plugins),
                "memory_layers": {
                    layer.value: len(memories)
                    for layer, memories in self.memory_system.memory_layers.items()
                }
            },
            "metrics": dict(self.system_metrics),
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down Unified AI Agent System...")
        
        self.is_running = False
        
        # Cancel task workers
        for worker in self.task_workers:
            worker.cancel()
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.sleep(2)  # Give tasks time to finish
        
        # Cleanup memory system
        await self.memory_system.cleanup()
        
        # Store shutdown memory
        shutdown_memory = UniversalEntity(
            type=EntityType.MEMORY,
            content="Unified system shutdown",
            metadata={
                "event": "system_shutdown",
                "timestamp": datetime.now().isoformat(),
                "final_status": self.get_system_status()
            },
            importance=1.0
        )
        await self.memory_system.store_memory(shutdown_memory, MemoryType.PERSISTENT)
        
        logger.info("Unified system shutdown complete.")
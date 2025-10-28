import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set

from src.core.entities import UniversalEntity, EntityType, Priority, MemoryType
from src.core.engine import PromptEngine
from src.core.memory import PersistentMemorySystem
from src.core.plugins import PluginSystem

logger = logging.getLogger(__name__)


class UnifiedSystem:
    """The ultimate unified AI agent system"""

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client

        # Core components
        self.prompt_engine = PromptEngine()
        self.memory_system = PersistentMemorySystem()
        self.plugin_system = PluginSystem(self)

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
            if (
                (entity.name is None or query_lower not in entity.name.lower()) and
                (entity.content is None or query_lower not in str(entity.content).lower()) and
                (entity.description is None or query_lower not in entity.description.lower())
            ):
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
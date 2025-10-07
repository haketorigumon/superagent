import importlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Set

import aiofiles

logger = logging.getLogger(__name__)


class PluginSystem:
    """Dynamic plugin system for infinite extensibility"""

    def __init__(self, system, plugins_dir: str = "plugins"):
        self.system = system
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
from datetime import datetime

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
from datetime import datetime

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
                plugin_instance = module.create_plugin(self.system)
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
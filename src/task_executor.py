
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

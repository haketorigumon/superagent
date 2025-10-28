"""Task Executor Plugin - Executes tasks with full flexibility"""

from datetime import datetime

PLUGIN_INFO = {
    "name": "task_executor",
    "version": "1.0.0",
    "description": "Universal task execution engine",
    "capabilities": ["task_execution", "parallel_processing", "error_handling"],
    "dependencies": [],
}


class TaskExecutor:
    def __init__(self, system):
        self.system = system
        self.active_tasks = {}

    def _is_complex_task(self, task_content: str) -> bool:
        """
        Determines if a task is complex based on keywords.
        """
        complex_keywords = [" and ", " then ", " first ", " second "]
        return any(keyword in task_content.lower() for keyword in complex_keywords)

    async def execute_task(self, task_entity):
        """Execute any type of task"""
        task_id = task_entity.id
        self.active_tasks[task_id] = task_entity

        try:
            if self._is_complex_task(task_entity.content):
                return await self._execute_complex_task(task_entity)
            else:
                return await self._execute_simple_task(task_entity)
        except Exception as e:
            task_entity.state["status"] = "failed"
            task_entity.state["error"] = str(e)
            raise
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    async def _execute_simple_task(self, task_entity):
        """Executes a simple, single-step task."""
        prompt = await self.system.prompt_engine.generate_prompt(
            "execute_task",
            {
                "task": task_entity.content,
                "context": task_entity.metadata,
                "capabilities": list(task_entity.capabilities),
            },
        )

        result = await self.system.llm_client.generate(
            prompt, "You are a helpful assistant."
        )

        task_entity.state["status"] = "completed"
        task_entity.state["result"] = result
        task_entity.state["completed_at"] = datetime.now().isoformat()

        return result

    async def _execute_complex_task(self, task_entity):
        """Executes a complex task by generating and following a plan."""
        plan_prompt = await self.system.prompt_engine.generate_prompt(
            "task_planner",
            {
                "task": task_entity.content,
                "resources": task_entity.metadata.get("resources", {}),
                "constraints": task_entity.metadata.get("constraints", []),
                "context": task_entity.metadata,
            },
        )

        plan = await self.system.llm_client.generate(
            plan_prompt, "You are a helpful assistant."
        )

        task_entity.state["plan"] = plan

        # Simple plan execution: assume plan is a numbered list of steps
        steps = [
            line.strip()
            for line in plan.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

        results = []
        for step in steps:
            step_result = await self._execute_simple_task_step(step)
            results.append(step_result)

        final_result = "\n".join(results)
        task_entity.state["status"] = "completed"
        task_entity.state["result"] = final_result
        task_entity.state["completed_at"] = datetime.now().isoformat()

        return final_result

    async def _execute_simple_task_step(self, step: str) -> str:
        """Executes a single step of a plan."""
        prompt = await self.system.prompt_engine.generate_prompt(
            "execute_task", {"task": step}
        )
        return await self.system.llm_client.generate(
            prompt, "You are a helpful assistant."
        )


def create_plugin(system):
    return TaskExecutor(system)

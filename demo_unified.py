import asyncio
from src.core.unified_system import UnifiedSystem, EntityType, Priority
from src.utils.unified_config import load_config
from src.utils.llm.llm_client import LLMClient
from rich.console import Console
from rich.panel import Panel

console = Console()

async def main():
    """
    A comprehensive demonstration of the Unified AI Agent System.
    """
    console.print(Panel.fit("[bold green]🚀 Unified AI Agent System - Demonstration[/bold green]"))

    # Load configuration
    console.print("[cyan]Loading configuration...[/cyan]")
    config = await load_config("unified_config.yaml")

    # Initialize LLM client
    console.print("[cyan]Initializing LLM client...[/cyan]")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.get_api_key_for_provider(config.llm.provider),
    )

    # Initialize the unified system
    console.print("[cyan]Initializing the unified system...[/cyan]")
    system = UnifiedSystem(config, llm_client)
    await system.initialize()

    console.print("[green]✅ System initialized successfully.[/green]\n")

    # 1. Create a specialized agent
    console.print(Panel("[bold yellow]Step 1: Create a specialized agent[/bold yellow]"))
    agent_id = await system.create_agent(
        name="CodeGenerationExpert",
        description="An expert in generating Python code.",
        capabilities={"code_generation", "python_expert"},
    )
    console.print(f"Agent created: [bold cyan]CodeGenerationExpert[/bold cyan] (ID: {agent_id})\n")

    # 2. Create a task for the agent
    console.print(Panel("[bold yellow]Step 2: Create a task for the agent[/bold yellow]"))
    task_content = "Write a simple Python function to calculate the factorial of a number."
    task_id = await system.create_task(
        content=task_content,
        priority=Priority.HIGH,
        metadata={"target_agent": agent_id},
    )
    console.print(f"Task created: [bold cyan]{task_content}[/bold cyan] (ID: {task_id})\n")

    # 3. Wait for the task to complete
    console.print(Panel("[bold yellow]Step 3: Wait for task completion[/bold yellow]"))
    console.print("Waiting for the task to be processed...")
    result = None
    for _ in range(10):  # Wait for up to 10 seconds
        task_status = await system.get_task_status(task_id)
        if task_status and task_status["status"] == "completed":
            console.print("[green]✅ Task completed![/green]")
            result = task_status.get("result")
            break
        await asyncio.sleep(1)

    # 4. Display the result
    console.print(Panel("[bold yellow]Step 4: Display the result[/bold yellow]"))
    if result:
        console.print(f"Result for task '{task_content}':")
        console.print(Panel(result, style="green", border_style="green"))
    else:
        console.print("[red]Task did not complete in time.[/red]")

    # Shutdown the system
    console.print("\n[cyan]Shutting down the system...[/cyan]")
    await system.shutdown()
    console.print("[green]✅ System shutdown complete.[/green]")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Unified AI Agent System Demo
Demonstrates the capabilities of the new unified architecture
"""

import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.unified_system import UnifiedSystem, EntityType, Priority
from src.utils.unified_config import load_config
from src.utils.llm_client import LLMClient

console = Console()


async def main():
    """Main demo function"""
    console.print(Panel.fit(
        "[bold blue]🚀 Unified AI Agent System Demo[/bold blue]\n"
        "[dim]Showcasing infinite flexibility and capabilities[/dim]",
        style="blue"
    ))
    
    # Load configuration
    console.print("[cyan]Loading configuration...[/cyan]")
    config = await load_config("unified_config.yaml")
    
    # Initialize LLM client
    console.print("[cyan]Initializing LLM client...[/cyan]")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key=config.get_api_key_for_provider(config.llm.provider),
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    # Test LLM connection
    console.print("[cyan]Testing LLM connection...[/cyan]")
    async with llm_client as client:
        if not await client.check_connection():
            console.print("[red]❌ Cannot connect to LLM service[/red]")
            console.print("[yellow]Please ensure your LLM service is running[/yellow]")
            return
    
    console.print("[green]✅ LLM connection successful[/green]")
    
    # Initialize unified system
    console.print("[cyan]Initializing unified system...[/cyan]")
    system = UnifiedSystem(config, llm_client)
    await system.initialize()
    
    console.print("[green]✅ Unified system initialized![/green]")
    
    # Run demonstrations
    await demo_basic_functionality(system)
    await demo_agent_creation(system)
    await demo_task_processing(system)
    await demo_memory_system(system)
    await demo_plugin_system(system)
    await demo_prompt_generation(system)
    
    # Shutdown
    console.print("[yellow]Shutting down system...[/yellow]")
    await system.shutdown()
    console.print("[green]✅ Demo completed successfully![/green]")


async def demo_basic_functionality(system: UnifiedSystem):
    """Demonstrate basic system functionality"""
    console.print(Panel.fit("[bold green]Demo 1: Basic Functionality[/bold green]", style="green"))
    
    # Show system status
    status = system.get_system_status()
    console.print(f"[blue]System Status:[/blue]")
    console.print(f"  - Initialized: {status['system']['initialized']}")
    console.print(f"  - Running: {status['system']['running']}")
    console.print(f"  - Total Entities: {status['system']['total_entities']}")
    console.print(f"  - Loaded Plugins: {status['system']['loaded_plugins']}")
    
    # Create some basic entities
    console.print(f"[blue]Creating basic entities...[/blue]")
    
    message_id = await system.create_entity(
        EntityType.MESSAGE,
        content="Hello from the unified system!",
        metadata={"demo": "basic_functionality"}
    )
    
    context_id = await system.create_entity(
        EntityType.CONTEXT,
        content={"demo_phase": "basic", "timestamp": "2025-08-02"},
        metadata={"purpose": "demonstration"}
    )
    
    console.print(f"[green]✅ Created message entity: {message_id}[/green]")
    console.print(f"[green]✅ Created context entity: {context_id}[/green]")


async def demo_agent_creation(system: UnifiedSystem):
    """Demonstrate agent creation and management"""
    console.print(Panel.fit("[bold green]Demo 2: Agent Creation[/bold green]", style="green"))
    
    # Create different types of agents
    agents = []
    
    # General purpose agent
    agent1_id = await system.create_agent(
        name="GeneralAssistant",
        description="A general-purpose AI assistant",
        capabilities={"reasoning", "communication", "problem_solving"},
        metadata={"role": "assistant", "specialization": "general"}
    )
    agents.append(agent1_id)
    console.print(f"[green]✅ Created general assistant: {agent1_id}[/green]")
    
    # Specialized expert agent
    agent2_id = await system.create_agent(
        name="TechnicalExpert",
        description="A technical expert specializing in AI and programming",
        capabilities={"technical_analysis", "code_generation", "system_design"},
        metadata={"role": "expert", "specialization": "technical"}
    )
    agents.append(agent2_id)
    console.print(f"[green]✅ Created technical expert: {agent2_id}[/green]")
    
    # Creative agent
    agent3_id = await system.create_agent(
        name="CreativeWriter",
        description="A creative agent specializing in writing and storytelling",
        capabilities={"creative_writing", "storytelling", "content_generation"},
        metadata={"role": "creative", "specialization": "writing"}
    )
    agents.append(agent3_id)
    console.print(f"[green]✅ Created creative writer: {agent3_id}[/green]")
    
    # Show agent information
    console.print(f"[blue]Agent Summary:[/blue]")
    for agent_id in agents:
        agent = await system.get_entity(agent_id)
        if agent:
            console.print(f"  - {agent.name}: {len(agent.capabilities)} capabilities")


async def demo_task_processing(system: UnifiedSystem):
    """Demonstrate task creation and processing"""
    console.print(Panel.fit("[bold green]Demo 3: Task Processing[/bold green]", style="green"))
    
    # Create various types of tasks
    tasks = []
    
    # Simple reasoning task
    task1_id = await system.create_task(
        content="Explain the concept of artificial intelligence in simple terms",
        metadata={"type": "explanation", "complexity": "simple"},
        priority=Priority.NORMAL
    )
    tasks.append(task1_id)
    console.print(f"[green]✅ Created reasoning task: {task1_id}[/green]")
    
    # Creative task
    task2_id = await system.create_task(
        content="Write a short story about a robot learning to feel emotions",
        metadata={"type": "creative", "genre": "science_fiction"},
        priority=Priority.HIGH
    )
    tasks.append(task2_id)
    console.print(f"[green]✅ Created creative task: {task2_id}[/green]")
    
    # Technical task
    task3_id = await system.create_task(
        content="Design a simple algorithm for sorting a list of numbers",
        metadata={"type": "technical", "domain": "algorithms"},
        priority=Priority.NORMAL
    )
    tasks.append(task3_id)
    console.print(f"[green]✅ Created technical task: {task3_id}[/green]")
    
    # Wait for tasks to complete
    console.print(f"[blue]Waiting for tasks to complete...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_progress = progress.add_task("Processing tasks...", total=None)
        
        completed_tasks = 0
        max_wait = 60  # seconds
        
        for _ in range(max_wait):
            completed_count = 0
            for task_id in tasks:
                task_status = await system.get_task_status(task_id)
                if task_status and task_status["status"] in ["completed", "failed"]:
                    completed_count += 1
            
            if completed_count == len(tasks):
                completed_tasks = completed_count
                break
            
            await asyncio.sleep(1)
        
        progress.update(task_progress, description=f"Completed {completed_count}/{len(tasks)} tasks")
    
    # Show task results
    console.print(f"[blue]Task Results:[/blue]")
    for i, task_id in enumerate(tasks, 1):
        task_status = await system.get_task_status(task_id)
        if task_status:
            status = task_status["status"]
            if status == "completed":
                result = task_status.get("result", "No result")
                result_preview = result[:100] + "..." if len(result) > 100 else result
                console.print(f"  Task {i}: [green]✅ {status}[/green] - {result_preview}")
            else:
                console.print(f"  Task {i}: [yellow]⏳ {status}[/yellow]")


async def demo_memory_system(system: UnifiedSystem):
    """Demonstrate memory system capabilities"""
    console.print(Panel.fit("[bold green]Demo 4: Memory System[/bold green]", style="green"))
    
    # Store various types of memories
    console.print(f"[blue]Storing memories...[/blue]")
    
    # Store episodic memory
    from src.core.unified_system import UniversalEntity, MemoryType
    
    episodic_memory = UniversalEntity(
        type=EntityType.MEMORY,
        content="The user asked about artificial intelligence and received a comprehensive explanation",
        metadata={
            "memory_type": "episodic",
            "participants": ["user", "system"],
            "topic": "artificial_intelligence"
        },
        importance=0.8,
        tags={"ai", "explanation", "user_interaction"}
    )
    
    memory_id1 = await system.memory_system.store_memory(episodic_memory, MemoryType.EPISODIC)
    console.print(f"[green]✅ Stored episodic memory: {memory_id1}[/green]")
    
    # Store semantic memory
    semantic_memory = UniversalEntity(
        type=EntityType.MEMORY,
        content="Artificial Intelligence is the simulation of human intelligence in machines",
        metadata={
            "memory_type": "semantic",
            "domain": "computer_science",
            "concept": "artificial_intelligence"
        },
        importance=0.9,
        tags={"ai", "definition", "knowledge"}
    )
    
    memory_id2 = await system.memory_system.store_memory(semantic_memory, MemoryType.SEMANTIC)
    console.print(f"[green]✅ Stored semantic memory: {memory_id2}[/green]")
    
    # Store procedural memory
    procedural_memory = UniversalEntity(
        type=EntityType.MEMORY,
        content="To create an AI agent: 1) Define capabilities, 2) Initialize with prompts, 3) Connect to LLM",
        metadata={
            "memory_type": "procedural",
            "process": "agent_creation",
            "steps": 3
        },
        importance=0.7,
        tags={"procedure", "agent_creation", "how_to"}
    )
    
    memory_id3 = await system.memory_system.store_memory(procedural_memory, MemoryType.PROCEDURAL)
    console.print(f"[green]✅ Stored procedural memory: {memory_id3}[/green]")
    
    # Search memories
    console.print(f"[blue]Searching memories...[/blue]")
    
    search_results = await system.memory_system.search_memories(
        "artificial intelligence",
        context={"tags": ["ai"]},
        limit=5
    )
    
    console.print(f"[green]Found {len(search_results)} relevant memories[/green]")
    for i, memory in enumerate(search_results, 1):
        content_preview = str(memory.content)[:80] + "..." if len(str(memory.content)) > 80 else str(memory.content)
        console.print(f"  {i}. {content_preview} (importance: {memory.importance})")


async def demo_plugin_system(system: UnifiedSystem):
    """Demonstrate plugin system capabilities"""
    console.print(Panel.fit("[bold green]Demo 5: Plugin System[/bold green]", style="green"))
    
    # List available plugins
    plugins = await system.plugin_system.list_plugins()
    console.print(f"[blue]Available plugins:[/blue]")
    
    for plugin_name, plugin_info in plugins.items():
        status = "✅ Loaded" if plugin_info["loaded"] else "⏳ Available"
        capabilities = ", ".join(plugin_info.get("capabilities", []))
        console.print(f"  - {plugin_name}: {status} - {capabilities}")
    
    # Test plugin functionality
    console.print(f"[blue]Testing plugin functionality...[/blue]")
    
    # Test task executor
    task_executor = await system.plugin_system.get_plugin("task_executor")
    if task_executor:
        console.print(f"[green]✅ Task executor plugin is available[/green]")
    
    # Test capability manager
    capability_manager = await system.plugin_system.get_plugin("capability_manager")
    if capability_manager:
        console.print(f"[green]✅ Capability manager plugin is available[/green]")
        
        # Generate a new capability
        new_capability = await capability_manager.generate_capability(
            "natural language understanding",
            {"domain": "conversational_ai", "complexity": "advanced"}
        )
        console.print(f"[green]✅ Generated new capability: {new_capability}[/green]")
    
    # Test workflow engine
    workflow_engine = await system.plugin_system.get_plugin("workflow_engine")
    if workflow_engine:
        console.print(f"[green]✅ Workflow engine plugin is available[/green]")
    
    # Test resource manager
    resource_manager = await system.plugin_system.get_plugin("resource_manager")
    if resource_manager:
        console.print(f"[green]✅ Resource manager plugin is available[/green]")
        
        # Test resource allocation
        allocated = await resource_manager.allocate_resource("cpu", 100, "demo_user")
        console.print(f"[green]✅ Resource allocation successful: {allocated}[/green]")


async def demo_prompt_generation(system: UnifiedSystem):
    """Demonstrate dynamic prompt generation"""
    console.print(Panel.fit("[bold green]Demo 6: Prompt Generation[/bold green]", style="green"))
    
    # Generate prompts for different purposes
    console.print(f"[blue]Generating dynamic prompts...[/blue]")
    
    # Generate a prompt for creative writing
    creative_prompt = await system.prompt_engine.generate_prompt(
        "creative_writing",
        {
            "genre": "science_fiction",
            "theme": "artificial_consciousness",
            "length": "short_story",
            "tone": "thoughtful"
        },
        system.llm_client
    )
    
    console.print(f"[green]✅ Generated creative writing prompt[/green]")
    console.print(f"[dim]Preview: {creative_prompt[:100]}...[/dim]")
    
    # Generate a prompt for technical analysis
    technical_prompt = await system.prompt_engine.generate_prompt(
        "technical_analysis",
        {
            "domain": "machine_learning",
            "task": "algorithm_comparison",
            "complexity": "intermediate",
            "output_format": "structured_report"
        },
        system.llm_client
    )
    
    console.print(f"[green]✅ Generated technical analysis prompt[/green]")
    console.print(f"[dim]Preview: {technical_prompt[:100]}...[/dim]")
    
    # Generate a prompt for problem solving
    problem_solving_prompt = await system.prompt_engine.generate_prompt(
        "problem_solving",
        {
            "problem_type": "optimization",
            "constraints": ["time", "resources", "accuracy"],
            "approach": "systematic",
            "stakeholders": ["users", "developers", "managers"]
        },
        system.llm_client
    )
    
    console.print(f"[green]✅ Generated problem solving prompt[/green]")
    console.print(f"[dim]Preview: {problem_solving_prompt[:100]}...[/dim]")
    
    # Show prompt templates
    console.print(f"[blue]Available prompt templates:[/blue]")
    for template_name in system.prompt_engine.templates.keys():
        console.print(f"  - {template_name}")
    
    console.print(f"[blue]Available system prompts:[/blue]")
    for prompt_name in system.prompt_engine.system_prompts.keys():
        console.print(f"  - {prompt_name}")


if __name__ == "__main__":
    asyncio.run(main())
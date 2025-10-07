#!/usr/bin/env python3
"""
AI Agent System - Main Entry Point
The ultimate flexible AI agent architecture with infinite capabilities
Completely prompt-driven, plugin-based, and self-evolving system
"""

import asyncio
import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional, List

from src.core.unified_system import UnifiedSystem, EntityType, Priority
from src.utils.unified_config import load_config
from src.utils.llm.llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(
    help="Unified AI Agent System - The ultimate flexible AI architecture",
    rich_markup_mode="rich"
)


@app.command()
def start(
    config_file: str = typer.Option("unified_config.yaml", help="Configuration file path"),
    provider: Optional[str] = typer.Option(None, help="LLM provider override"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    api_key: Optional[str] = typer.Option(None, help="API key override"),
    web_interface: bool = typer.Option(True, help="Enable web interface"),
    host: str = typer.Option("0.0.0.0", help="Web interface host"),
    port: int = typer.Option(12000, help="Web interface port"),
    debug: bool = typer.Option(False, help="Enable debug mode")
):
    """Start the unified AI agent system"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]🚀 Unified AI Agent System[/bold blue]\n"
        "[dim]The ultimate flexible AI architecture[/dim]",
        style="blue"
    ))
    
    asyncio.run(_start_system(
        config_file, provider, model, api_key, 
        web_interface, host, port
    ))


async def _start_system(config_file: str, provider: Optional[str], model: Optional[str], 
                       api_key: Optional[str], web_interface: bool, host: str, port: int):
    """Start the unified system"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load configuration
        task = progress.add_task("Loading configuration...", total=None)
        config = await load_config(config_file)
        
        # Apply overrides
        if provider:
            config.set_runtime_override("llm", "provider", provider)
        if model:
            config.set_runtime_override("llm", "model", model)
        if api_key:
            config.set_runtime_override("llm", "api_key", api_key)
        if host != "0.0.0.0":
            config.set_runtime_override("web", "host", host)
        if port != 12000:
            config.set_runtime_override("web", "port", port)
        
        progress.update(task, description="Initializing LLM client...")
        
        # Initialize LLM client
        llm_client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.get_api_key_for_provider(config.llm.provider),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Test LLM connection
        progress.update(task, description="Testing LLM connection...")
        async with llm_client as client:
            if not await client.check_connection():
                console.print("[red]❌ Cannot connect to LLM service[/red]")
                console.print(f"[yellow]Provider: {config.llm.provider}[/yellow]")
                console.print(f"[yellow]Model: {config.llm.model}[/yellow]")
                if config.llm.base_url:
                    console.print(f"[yellow]URL: {config.llm.base_url}[/yellow]")
                console.print("[dim]Please check your configuration and ensure the LLM service is running.[/dim]")
                return
        
        progress.update(task, description="Initializing unified system...")
        
        # Initialize unified system
        system = UnifiedSystem(config, llm_client)
        await system.initialize()
        
        progress.update(task, description="System ready!", completed=True)
    
    console.print("[green]✅ Unified AI Agent System initialized successfully![/green]")
    
    # Display system status
    status = system.get_system_status()
    _display_system_status(status)
    
    if web_interface:
        console.print(f"[cyan]🌐 Web interface starting on http://{host}:{port}[/cyan]")
        console.print("[yellow]Note: Web interface implementation is in progress[/yellow]")
    
    # Start interactive mode
    await _interactive_mode(system)


def _display_system_status(status: dict):
    """Display system status in a nice table"""
    table = Table(title="System Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    system_info = status["system"]
    table.add_row(
        "System", 
        "🟢 Running" if system_info["running"] else "🔴 Stopped",
        f"Initialized: {system_info['initialized']}"
    )
    
    table.add_row(
        "Entities",
        f"{system_info['total_entities']} total",
        ", ".join([f"{k}: {v}" for k, v in system_info["entities_by_type"].items()])
    )
    
    table.add_row(
        "Tasks",
        f"{system_info['active_tasks']} active",
        f"{system_info['queued_tasks']} queued"
    )
    
    table.add_row(
        "Plugins",
        f"{system_info['loaded_plugins']} loaded",
        "Dynamic loading enabled"
    )
    
    table.add_row(
        "Memory",
        f"{sum(system_info['memory_layers'].values())} memories",
        ", ".join([f"{k}: {v}" for k, v in system_info["memory_layers"].items()])
    )
    
    console.print(table)


async def _interactive_mode(system: UnifiedSystem):
    """Interactive mode for the unified system"""
    console.print("\n[bold green]🎯 Interactive Mode[/bold green]")
    console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]\n")
    
    user_id = "interactive_user"
    
    while system.is_running:
        try:
            user_input = console.input("[bold blue]>>> [/bold blue]")
            
            if not user_input.strip():
                continue
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
            
            if user_input.lower() == "help":
                _show_help()
                continue
            
            if user_input.lower() == "status":
                status = system.get_system_status()
                _display_system_status(status)
                continue
            
            if user_input.lower().startswith("create agent"):
                await _handle_create_agent(system, user_input)
                continue
            
            if user_input.lower().startswith("list"):
                await _handle_list_command(system, user_input)
                continue
            
            # Process as general request
            console.print("[dim]Processing request...[/dim]")
            
            response = await system.process_user_request(user_id, user_input)
            
            if response.get("success"):
                task_id = response["task_id"]
                console.print(f"[green]✅ Task created: {task_id}[/green]")
                
                # Wait for task completion
                max_wait = 30
                for _ in range(max_wait):
                    task_status = await system.get_task_status(task_id)
                    if task_status and task_status["status"] in ["completed", "failed"]:
                        break
                    await asyncio.sleep(1)
                
                # Display result
                task_status = await system.get_task_status(task_id)
                if task_status:
                    if task_status["status"] == "completed":
                        result = task_status.get("result", "No result available")
                        console.print(f"[bold green]Result:[/bold green] {result}")
                    elif task_status["status"] == "failed":
                        error = task_status.get("error", "Unknown error")
                        console.print(f"[bold red]Error:[/bold red] {error}")
                    else:
                        console.print(f"[yellow]Task status: {task_status['status']}[/yellow]")
            else:
                error = response.get("error", "Unknown error")
                console.print(f"[bold red]Error:[/bold red] {error}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[yellow]Shutting down system...[/yellow]")
    await system.shutdown()
    console.print("[green]✅ System shutdown complete[/green]")


def _show_help():
    """Show help information"""
    help_table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    help_table.add_column("Command", style="green")
    help_table.add_column("Description", style="dim")
    
    commands = [
        ("help", "Show this help message"),
        ("status", "Display system status"),
        ("create agent [name]", "Create a new agent"),
        ("list agents", "List all agents"),
        ("list tasks", "List all tasks"),
        ("list entities", "List all entities"),
        ("exit/quit/bye", "Exit the system"),
        ("[any text]", "Process as a general request")
    ]
    
    for command, description in commands:
        help_table.add_row(command, description)
    
    console.print(help_table)


async def _handle_create_agent(system: UnifiedSystem, command: str):
    """Handle create agent command"""
    parts = command.split()
    agent_name = parts[2] if len(parts) > 2 else f"agent_{len(system.entities) + 1}"
    
    try:
        agent_id = await system.create_agent(
            name=agent_name,
            description=f"Agent created via interactive command",
            capabilities={"reasoning", "communication", "task_execution"}
        )
        console.print(f"[green]✅ Created agent: {agent_name} (ID: {agent_id})[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to create agent: {e}[/red]")


async def _handle_list_command(system: UnifiedSystem, command: str):
    """Handle list commands"""
    parts = command.split()
    if len(parts) < 2:
        console.print("[red]Usage: list [agents|tasks|entities][/red]")
        return
    
    list_type = parts[1].lower()
    
    if list_type == "agents":
        agents = await system.search_entities(EntityType.AGENT)
        if agents:
            table = Table(title="Agents", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="dim")
            table.add_column("Name", style="green")
            table.add_column("Capabilities", style="blue")
            table.add_column("Status", style="yellow")
            
            for agent in agents:
                status = agent.state.get("status", "unknown")
                capabilities = ", ".join(list(agent.capabilities)[:3])
                if len(agent.capabilities) > 3:
                    capabilities += "..."
                table.add_row(agent.id, agent.name, capabilities, status)
            
            console.print(table)
        else:
            console.print("[yellow]No agents found[/yellow]")
    
    elif list_type == "tasks":
        tasks = await system.search_entities(EntityType.TASK)
        if tasks:
            table = Table(title="Tasks", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="dim")
            table.add_column("Content", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Priority", style="blue")
            
            for task in tasks:
                status = task.state.get("status", "unknown")
                content = str(task.content)[:50] + "..." if len(str(task.content)) > 50 else str(task.content)
                table.add_row(task.id, content, status, task.priority.name)
            
            console.print(table)
        else:
            console.print("[yellow]No tasks found[/yellow]")
    
    elif list_type == "entities":
        total_entities = len(system.entities)
        if total_entities > 0:
            table = Table(title="All Entities", show_header=True, header_style="bold cyan")
            table.add_column("Type", style="blue")
            table.add_column("Count", style="green")
            
            for entity_type, entity_ids in system.entity_types.items():
                table.add_row(entity_type.value, str(len(entity_ids)))
            
            console.print(table)
            console.print(f"[dim]Total entities: {total_entities}[/dim]")
        else:
            console.print("[yellow]No entities found[/yellow]")
    
    else:
        console.print(f"[red]Unknown list type: {list_type}[/red]")
        console.print("[dim]Available types: agents, tasks, entities[/dim]")


@app.command()
def config(
    config_file: str = typer.Option("unified_config.yaml", help="Configuration file path"),
    show: bool = typer.Option(False, help="Show current configuration"),
    reset: bool = typer.Option(False, help="Reset to default configuration"),
    backup: bool = typer.Option(False, help="Create configuration backup"),
    restore: Optional[str] = typer.Option(None, help="Restore from backup file")
):
    """Manage system configuration"""
    asyncio.run(_manage_config(config_file, show, reset, backup, restore))


async def _manage_config(config_file: str, show: bool, reset: bool, 
                        backup: bool, restore: Optional[str]):
    """Manage configuration"""
    config = await load_config(config_file)
    
    if show:
        console.print(Panel.fit("[bold blue]Current Configuration[/bold blue]", style="blue"))
        
        config_dict = config.to_dict()
        for section, values in config_dict.items():
            if section in ["environment_overrides", "runtime_overrides"] and not values:
                continue
            
            table = Table(title=section.title(), show_header=True, header_style="bold cyan")
            table.add_column("Setting", style="green")
            table.add_column("Value", style="yellow")
            
            if isinstance(values, dict):
                for key, value in values.items():
                    table.add_row(key, str(value))
            else:
                table.add_row(section, str(values))
            
            console.print(table)
    
    if reset:
        console.print("[yellow]Resetting configuration to defaults...[/yellow]")
        await config.reset_to_defaults()
        console.print("[green]✅ Configuration reset complete[/green]")
    
    if backup:
        backup_file = await config.backup_config()
        if backup_file:
            console.print(f"[green]✅ Configuration backed up to: {backup_file}[/green]")
        else:
            console.print("[red]❌ Failed to create backup[/red]")
    
    if restore:
        success = await config.restore_config(restore)
        if success:
            console.print(f"[green]✅ Configuration restored from: {restore}[/green]")
        else:
            console.print(f"[red]❌ Failed to restore from: {restore}[/red]")


@app.command()
def providers():
    """List supported LLM providers"""
    console.print(Panel.fit("[bold blue]Supported LLM Providers[/bold blue]", style="blue"))
    
    providers_info = [
        ("ollama", "Local LLM runtime", "❌ No API key required"),
        ("openai", "OpenAI GPT models", "✅ API key required"),
        ("anthropic", "Claude models", "✅ API key required"),
        ("google", "Gemini models", "✅ API key required"),
        ("azure", "Azure OpenAI", "✅ API key required"),
        ("cohere", "Cohere models", "✅ API key required"),
        ("huggingface", "Hugging Face models", "✅ API key required"),
        ("together", "Together AI", "✅ API key required"),
        ("groq", "Groq models", "✅ API key required"),
        ("deepseek", "DeepSeek models", "✅ API key required"),
        ("moonshot", "Moonshot AI", "✅ API key required"),
        ("zhipu", "Zhipu AI", "✅ API key required"),
        ("baidu", "Baidu AI", "✅ API key required"),
        ("alibaba", "Alibaba Cloud", "✅ API key required")
    ]
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="green")
    table.add_column("Description", style="dim")
    table.add_column("API Key", style="yellow")
    
    for provider, description, api_key in providers_info:
        table.add_row(provider, description, api_key)
    
    console.print(table)
    console.print("\n[dim]💡 Example: unified_main.py start --provider openai --model gpt-4[/dim]")


@app.command()
def test_connection(
    provider: str = typer.Option("ollama", help="Provider to test"),
    model: str = typer.Option("llama3", help="Model to test"),
    api_key: Optional[str] = typer.Option(None, help="API key for testing"),
    config_file: str = typer.Option("unified_config.yaml", help="Configuration file path")
):
    """Test connection to an LLM provider"""
    asyncio.run(_test_connection(provider, model, api_key, config_file))


async def _test_connection(provider: str, model: str, api_key: Optional[str], config_file: str):
    """Test LLM connection"""
    console.print(f"[blue]🔍 Testing connection to {provider} with model {model}...[/blue]")
    
    config = await load_config(config_file)
    
    if not api_key:
        api_key = config.get_api_key_for_provider(provider)
    
    llm_client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        async with llm_client as client:
            if await client.check_connection():
                console.print(f"[green]✅ Successfully connected to {provider}[/green]")
                
                # Test generation
                response = await client.generate(
                    "Hello! Please respond with a brief greeting.",
                    "You are a helpful assistant."
                )
                
                if response:
                    console.print(f"[green]📝 Test response:[/green] {response}")
                    console.print("[green]🎉 Provider is working correctly![/green]")
                else:
                    console.print("[yellow]⚠️ Connection successful but no response generated[/yellow]")
            else:
                console.print(f"[red]❌ Failed to connect to {provider}[/red]")
                console.print("[dim]💡 Check your API key and network connection[/dim]")
    
    except Exception as e:
        console.print(f"[red]❌ Connection test failed: {e}[/red]")


@app.command()
def version():
    """Show version information"""
    console.print(Panel.fit(
        "[bold blue]Unified AI Agent System[/bold blue]\n"
        "[dim]Version: 1.0.0[/dim]\n"
        "[dim]The ultimate flexible AI architecture[/dim]",
        style="blue"
    ))


if __name__ == "__main__":
    app()

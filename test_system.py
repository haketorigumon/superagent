#!/usr/bin/env python3
"""
Test script for the Unified AI Agent System
Validates core functionality without requiring LLM connection
"""

import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.core.unified_system import (
    UnifiedSystem, UniversalEntity, EntityType, Priority, MemoryType
)
from src.utils.unified_config import load_config

console = Console()


class MockLLMClient:
    """Mock LLM client for testing without actual LLM connection"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def check_connection(self):
        return True
    
    async def generate(self, prompt, system_prompt=""):
        # Mock response based on prompt content
        if "explain" in prompt.lower():
            return "This is a mock explanation of the requested topic."
        elif "create" in prompt.lower():
            return "This is a mock creative response."
        elif "analyze" in prompt.lower():
            return "This is a mock analysis of the provided data."
        else:
            return "This is a mock response to your request."


async def test_basic_functionality():
    """Test basic system functionality"""
    console.print(Panel.fit("[bold green]Test 1: Basic Functionality[/bold green]", style="green"))
    
    # Load configuration
    config = await load_config("unified_config.yaml")
    console.print("✅ Configuration loaded")
    
    # Create mock LLM client
    llm_client = MockLLMClient()
    console.print("✅ Mock LLM client created")
    
    # Initialize unified system
    system = UnifiedSystem(config, llm_client)
    await system.initialize()
    console.print("✅ Unified system initialized")
    
    # Test system status
    status = system.get_system_status()
    assert status["system"]["initialized"] == True
    assert status["system"]["running"] == True
    console.print("✅ System status verified")
    
    return system


async def test_entity_management(system):
    """Test entity creation and management"""
    console.print(Panel.fit("[bold green]Test 2: Entity Management[/bold green]", style="green"))
    
    # Create different types of entities
    agent_id = await system.create_entity(
        EntityType.AGENT,
        name="TestAgent",
        description="A test agent",
        capabilities={"reasoning", "communication"}
    )
    console.print(f"✅ Created agent: {agent_id}")
    
    task_id = await system.create_entity(
        EntityType.TASK,
        content="Test task content",
        priority=Priority.HIGH
    )
    console.print(f"✅ Created task: {task_id}")
    
    message_id = await system.create_entity(
        EntityType.MESSAGE,
        content="Test message",
        metadata={"sender": "test_user"}
    )
    console.print(f"✅ Created message: {message_id}")
    
    # Test entity retrieval
    agent = await system.get_entity(agent_id)
    assert agent is not None
    assert agent.name == "TestAgent"
    console.print("✅ Entity retrieval verified")
    
    # Test entity update
    await system.update_entity(agent_id, description="Updated test agent")
    updated_agent = await system.get_entity(agent_id)
    assert updated_agent.description == "Updated test agent"
    console.print("✅ Entity update verified")
    
    # Test entity search
    agents = await system.search_entities(EntityType.AGENT)
    assert len(agents) >= 1
    console.print(f"✅ Found {len(agents)} agents")
    
    return {"agent_id": agent_id, "task_id": task_id, "message_id": message_id}


async def test_agent_creation(system):
    """Test agent creation with capabilities"""
    console.print(Panel.fit("[bold green]Test 3: Agent Creation[/bold green]", style="green"))
    
    # Create specialized agents
    assistant_id = await system.create_agent(
        name="GeneralAssistant",
        description="A general-purpose assistant",
        capabilities={"reasoning", "communication", "problem_solving"}
    )
    console.print(f"✅ Created general assistant: {assistant_id}")
    
    expert_id = await system.create_agent(
        name="TechnicalExpert",
        description="A technical expert",
        capabilities={"technical_analysis", "code_generation", "system_design"}
    )
    console.print(f"✅ Created technical expert: {expert_id}")
    
    # Verify agents were created with proper capabilities
    assistant = await system.get_entity(assistant_id)
    assert "reasoning" in assistant.capabilities
    assert "communication" in assistant.capabilities
    console.print("✅ Agent capabilities verified")
    
    return {"assistant_id": assistant_id, "expert_id": expert_id}


async def test_task_processing(system):
    """Test task creation and processing"""
    console.print(Panel.fit("[bold green]Test 4: Task Processing[/bold green]", style="green"))
    
    # Create various tasks
    simple_task = await system.create_task(
        content="Explain the concept of artificial intelligence",
        metadata={"type": "explanation"},
        priority=Priority.NORMAL
    )
    console.print(f"✅ Created simple task: {simple_task}")
    
    complex_task = await system.create_task(
        content="Design a machine learning algorithm",
        metadata={"type": "technical", "complexity": "high"},
        priority=Priority.HIGH
    )
    console.print(f"✅ Created complex task: {complex_task}")
    
    # Wait for tasks to be processed
    console.print("⏳ Waiting for task processing...")
    await asyncio.sleep(2)  # Give tasks time to process
    
    # Check task status
    simple_status = await system.get_task_status(simple_task)
    if simple_status:
        console.print(f"✅ Simple task status: {simple_status['status']}")
        if simple_status.get("result"):
            result_preview = simple_status["result"][:50] + "..."
            console.print(f"   Result: {result_preview}")
    
    complex_status = await system.get_task_status(complex_task)
    if complex_status:
        console.print(f"✅ Complex task status: {complex_status['status']}")
    
    return {"simple_task": simple_task, "complex_task": complex_task}


async def test_memory_system(system):
    """Test memory storage and retrieval"""
    console.print(Panel.fit("[bold green]Test 5: Memory System[/bold green]", style="green"))
    
    # Store different types of memories
    episodic_memory = UniversalEntity(
        type=EntityType.MEMORY,
        content="User asked about AI and received explanation",
        metadata={"memory_type": "episodic", "topic": "artificial_intelligence"},
        importance=0.8,
        tags={"ai", "explanation", "user_interaction"}
    )
    
    memory_id = await system.memory_system.store_memory(episodic_memory, MemoryType.EPISODIC)
    console.print(f"✅ Stored episodic memory: {memory_id}")
    
    # Store semantic memory
    semantic_memory = UniversalEntity(
        type=EntityType.MEMORY,
        content="AI is the simulation of human intelligence in machines",
        metadata={"memory_type": "semantic", "domain": "computer_science"},
        importance=0.9,
        tags={"ai", "definition", "knowledge"}
    )
    
    semantic_id = await system.memory_system.store_memory(semantic_memory, MemoryType.SEMANTIC)
    console.print(f"✅ Stored semantic memory: {semantic_id}")
    
    # Test memory retrieval
    retrieved_memory = await system.memory_system.retrieve_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == episodic_memory.content
    console.print("✅ Memory retrieval verified")
    
    # Test memory search
    search_results = await system.memory_system.search_memories(
        "artificial intelligence",
        context={"tags": ["ai"]},
        limit=5
    )
    console.print(f"✅ Found {len(search_results)} relevant memories")
    
    return {"episodic_id": memory_id, "semantic_id": semantic_id}


async def test_plugin_system(system):
    """Test plugin system functionality"""
    console.print(Panel.fit("[bold green]Test 6: Plugin System[/bold green]", style="green"))
    
    # List available plugins
    plugins = await system.plugin_system.list_plugins()
    console.print(f"✅ Found {len(plugins)} plugins")
    
    for plugin_name, plugin_info in plugins.items():
        status = "loaded" if plugin_info["loaded"] else "available"
        console.print(f"   - {plugin_name}: {status}")
    
    # Test plugin functionality
    task_executor = await system.plugin_system.get_plugin("task_executor")
    if task_executor:
        console.print("✅ Task executor plugin available")
    
    capability_manager = await system.plugin_system.get_plugin("capability_manager")
    if capability_manager:
        console.print("✅ Capability manager plugin available")
    
    workflow_engine = await system.plugin_system.get_plugin("workflow_engine")
    if workflow_engine:
        console.print("✅ Workflow engine plugin available")
    
    resource_manager = await system.plugin_system.get_plugin("resource_manager")
    if resource_manager:
        console.print("✅ Resource manager plugin available")
        
        # Test resource allocation
        allocated = await resource_manager.allocate_resource("cpu", 100, "test_user")
        console.print(f"✅ Resource allocation: {allocated}")
    
    return plugins


async def test_prompt_system(system):
    """Test prompt generation system"""
    console.print(Panel.fit("[bold green]Test 7: Prompt System[/bold green]", style="green"))
    
    # Test system prompt generation
    context = {
        "context": "Testing prompt system",
        "capabilities": ["reasoning", "communication"],
        "state": {"status": "active"},
        "task": "Generate a test response"
    }
    
    prompt = await system.prompt_engine.generate_prompt("universal_agent", context)
    assert len(prompt) > 0
    console.print("✅ System prompt generated successfully")
    console.print(f"   Preview: {prompt[:100]}...")
    
    # Test template availability
    console.print(f"✅ Available templates: {len(system.prompt_engine.templates)}")
    console.print(f"✅ Available system prompts: {len(system.prompt_engine.system_prompts)}")
    
    return {"prompt": prompt}


async def test_user_request_processing(system):
    """Test user request processing"""
    console.print(Panel.fit("[bold green]Test 8: User Request Processing[/bold green]", style="green"))
    
    # Process a user request
    response = await system.process_user_request(
        "test_user",
        "What is artificial intelligence?"
    )
    
    assert response.get("success") == True
    task_id = response.get("task_id")
    console.print(f"✅ User request processed, task ID: {task_id}")
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check result
    task_status = await system.get_task_status(task_id)
    if task_status:
        console.print(f"✅ Task status: {task_status['status']}")
        if task_status.get("result"):
            result_preview = task_status["result"][:50] + "..."
            console.print(f"   Result: {result_preview}")
    
    return {"task_id": task_id, "response": response}


async def run_all_tests():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold blue]🧪 Unified AI Agent System Tests[/bold blue]\n"
        "[dim]Testing core functionality without LLM dependency[/dim]",
        style="blue"
    ))
    
    try:
        # Test 1: Basic functionality
        system = await test_basic_functionality()
        
        # Test 2: Entity management
        entities = await test_entity_management(system)
        
        # Test 3: Agent creation
        agents = await test_agent_creation(system)
        
        # Test 4: Task processing
        tasks = await test_task_processing(system)
        
        # Test 5: Memory system
        memories = await test_memory_system(system)
        
        # Test 6: Plugin system
        plugins = await test_plugin_system(system)
        
        # Test 7: Prompt system
        prompts = await test_prompt_system(system)
        
        # Test 8: User request processing
        requests = await test_user_request_processing(system)
        
        # Final system status
        console.print(Panel.fit("[bold green]Test Summary[/bold green]", style="green"))
        final_status = system.get_system_status()
        console.print(f"✅ Total entities: {final_status['system']['total_entities']}")
        console.print(f"✅ Active tasks: {final_status['system']['active_tasks']}")
        console.print(f"✅ Loaded plugins: {final_status['system']['loaded_plugins']}")
        console.print(f"✅ Memory layers: {sum(final_status['system']['memory_layers'].values())}")
        
        # Shutdown system
        console.print("\n[yellow]Shutting down system...[/yellow]")
        await system.shutdown()
        
        console.print(Panel.fit(
            "[bold green]🎉 All Tests Passed![/bold green]\n"
            "[dim]Unified AI Agent System is working correctly[/dim]",
            style="green"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Test Failed![/bold red]\n"
            f"[dim]Error: {str(e)}[/dim]",
            style="red"
        ))
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

import asyncio
import asyncio
import pytest
import pytest_asyncio
from src.core.unified_system import UnifiedSystem, EntityType, Priority, MemoryType
from src.utils.unified_config import load_config
from src.utils.llm.llm_client import LLMClient

@pytest_asyncio.fixture
async def unified_system():
    """Provides a fully initialized UnifiedSystem instance for testing."""
    config = await load_config("unified_config.yaml")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.get_api_key_for_provider(config.llm.provider),
    )
    system = UnifiedSystem(config, llm_client)
    await system.initialize()
    yield system
    await system.shutdown()

@pytest.mark.asyncio
async def test_system_initialization(unified_system: UnifiedSystem):
    """Tests if the UnifiedSystem initializes correctly."""
    assert unified_system.is_initialized
    assert unified_system.is_running
    # Check if core components are created
    assert unified_system.prompt_engine is not None
    assert unified_system.memory_system is not None
    assert unified_system.plugin_system is not None

@pytest.mark.asyncio
async def test_create_agent(unified_system: UnifiedSystem):
    """Tests the creation of a new agent."""
    agent_id = await unified_system.create_agent(
        name="TestAgent",
        description="An agent for testing purposes.",
        capabilities={"testing", "assertion"},
    )
    agent = await unified_system.get_entity(agent_id)
    assert agent is not None
    assert agent.type == EntityType.AGENT
    assert agent.name == "TestAgent"
    assert "testing" in agent.capabilities

@pytest.mark.asyncio
async def test_create_and_process_task(unified_system: UnifiedSystem):
    """Tests creating a task and verifying its completion."""
    task_content = "This is a test task."
    task_id = await unified_system.create_task(
        content=task_content,
        priority=Priority.HIGH,
    )

    # Give the system time to process the task
    await asyncio.sleep(1)  # Adjust if tasks take longer

    task_status = await unified_system.get_task_status(task_id)
    assert task_status is not None
    assert task_status["status"] == "completed"
    assert task_status["content"] == task_content
    assert "simulated response" in task_status["result"]

@pytest.mark.asyncio
async def test_memory_storage_and_retrieval(unified_system: UnifiedSystem):
    """Tests the persistent memory system."""
    memory_content = "This is an important memory."
    memory_id = await unified_system.create_entity(
        EntityType.MEMORY,
        content=memory_content,
        importance=0.9,
        tags={"test", "important"},
    )

    # Store it in persistent memory
    memory_entity = await unified_system.get_entity(memory_id)
    await unified_system.memory_system.store_memory(memory_entity, MemoryType.PERSISTENT)

    # Retrieve the memory
    retrieved_memory = await unified_system.memory_system.retrieve_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == memory_content
    assert "important" in retrieved_memory.tags

@pytest.mark.asyncio
async def test_search_memories(unified_system: UnifiedSystem):
    """Tests searching for memories."""
    # First, store a memory to search for
    await unified_system.create_entity(
        EntityType.MEMORY,
        content="Searchable test memory content.",
        tags={"search_test"},
        importance=0.8,
    )

    # Give it a moment to be indexed
    await asyncio.sleep(0.1)

    search_results = await unified_system.memory_system.search_memories("searchable")
    assert len(search_results) > 0
    assert "searchable" in search_results[0].content.lower()

    # Test searching by tag
    search_results_by_tag = await unified_system.memory_system.search_memories(
        "any", context={"tags": ["search_test"]}
    )
    assert len(search_results_by_tag) > 0
    assert "search_test" in search_results_by_tag[0].tags